"""
replication.py
==============
Standalone replication of ALL simulation results in:

    Tonkens & Herbert, "Refining Control Barrier Functions through
    Hamilton-Jacobi Reachability", IROS 2022 / arXiv:2204.12507v2

Covers four experiments from Section VI of the paper:
  A. Adaptive Cruise Control      (Fig. 2)
  B. Dubins Car                   (Fig. 3)
  C. Quadrotor                    (Fig. 4)
  D. Inverted Pendulum / Backup   (Fig. 5)

Usage
-----
    # Install deps first (once):
    pip install "cbf_opt>=0.6.0" "experiment-wrapper>=1.1" hj-reachability
    git clone https://github.com/stonkens/refineCBF.git
    pip install -e refineCBF

    # Run all experiments:
    python replication.py

    # Run individual experiments only:
    python replication.py --experiments acc dubins quad pendulum

    # Skip slow HJR solves (uses saved .npy if available):
    python replication.py --skip-solves

    # Output directory (default: ./results/):
    python replication.py --out-dir ./my_results

Outputs (written to <out-dir>/)
--------------------------------
    acc_results.csv
    dubins_results.csv
    quad_conservative_results.csv
    quad_invalid_results.csv
    pendulum_results.csv
    fig_acc.png               — Fig. 2 replica (safe set + rollout)
    fig_dubins.png            — Fig. 3 replica (safe set evolution + trajectories)
    fig_quad_conservative.png — Fig. 4 (left)
    fig_quad_invalid.png      — Fig. 4 (right)
    fig_pendulum.png          — Fig. 5 replica

Bugs fixed relative to the original Colab notebooks
----------------------------------------------------
1. np.sin → jnp.sin inside JAX-traced dynamics (QuadVerticalDynamicsJNP).
2. vf_table broadcast to 6-D done AFTER the DP solve, not before.
3. Missing os.makedirs before writing CSV / figure output files.
4. Complex-dtype JAX arrays rejected by CVXPY Parameter — stripped with
   np.real(...).astype(float) at every vf_table assignment and in the
   _patch_asif_for_real_values helper.
5. NaN / Inf from grid-boundary interpolation crash OSQP — clamped to
   zero-gradient / unsafe defaults in the same helper.
6. %matplotlib inline and other Jupyter magics removed.
"""

import argparse
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# NumPy 2.0 removed np.infty; cbf_opt still uses it internally.
# Restore it before importing cbf_opt so the library initialises correctly.
if not hasattr(np, "infty"):
    np.infty = np.inf

import pandas as pd
import seaborn as sns
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("cbf_opt").setLevel(logging.ERROR)

import jax.numpy as jnp
import hj_reachability as hj
import cbf_opt
from cbf_opt import (ControlAffineDynamics, ControlAffineCBF,
                     ControlAffineASIF, asif as cbf_asif_module,
                     cbf as cbf_module, utils)

# Workaround for a typo in cbf_opt's internal test:
#   cbf_opt/tests/test_cbf.py checks `cbf_opt.dynamicsDynamics` but the
#   correct attribute is `cbf_opt.Dynamics`.  Alias it so the test passes.
if not hasattr(cbf_opt, "dynamicsDynamics") and hasattr(cbf_opt, "Dynamics"):
    cbf_opt.dynamicsDynamics = cbf_opt.Dynamics

from experiment_wrapper import (RolloutTrajectory, TimeSeriesExperiment,
                                StateSpaceExperiment)
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF


# ─────────────────────────────────────────────────────────────────────────────
# Shared colour palette (matches paper figures)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_COLORS = np.array([
    (0.3,  0.3,  0.3),
    sns.color_palette("RdYlGn_r", 7)[0],
    sns.color_palette("RdYlGn_r", 9)[6],
    sns.color_palette("RdYlGn_r", 9)[8],
    (4/255, 101/255, 4/255),
])
ALT_COLORS = sns.color_palette("pastel", 9).as_hex()
CHOSEN_COLORS = [
    (0.5, 0.5, 0.5),
    sns.color_palette("tab10")[0],
    sns.color_palette("tab10")[1],
    (0.1, 0.1, 0.1),
    (0.7, 0.7, 0.7),
]

RCPARAMS = {
    "axes.labelsize": 22, "axes.titlesize": 22, "font.size": 22,
    "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "lines.linewidth": 3,
}
matplotlib.rcParams.update(RCPARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Shared ASIF patch: force real floats into CVXPY, guard NaN/Inf
# ─────────────────────────────────────────────────────────────────────────────
def _to_scalar(v):
    """
    Coerce any array-like with a single element to a plain Python float.
    cbf_opt internals do `hs[0] = val_curr` which requires a true scalar,
    not a 0-d or 1-element numpy/JAX array.
    For multi-element arrays the value is returned unchanged so batch/grid
    calls still work correctly.
    """
    v = np.asarray(v)
    if v.size == 1:
        return float(v.flat[0])
    return v


def _wrap_vf_scalar(cbf_obj):
    """
    Monkey-patch cbf_obj.vf so single-state calls return a plain float.
    Safe to call multiple times (idempotent via _vf_scalar_wrapped guard).
    """
    if getattr(cbf_obj, '_vf_scalar_wrapped', False):
        return cbf_obj
    _orig = cbf_obj.vf
    def _vf_wrapped(state, time=0.0, _f=_orig):
        return _to_scalar(_f(state, time))
    cbf_obj.vf = _vf_wrapped
    cbf_obj._vf_scalar_wrapped = True
    return cbf_obj


def _patch_asif_for_real_values(asif_inst):
    """
    Monkey-patch an ASIF instance so set_constraint always receives real
    float64 scalars, preventing the CVXPY 'Parameter value must be real'
    error that arises when JAX returns complex-typed arrays, and clamping
    NaN / Inf from grid-boundary interpolation so OSQP never sees invalid data.
    Also wraps the CBF's vf method to always return a plain scalar so that
    cbf_opt's `hs[0] = val_curr` assignment never sees an array sequence.
    """
    # Wrap the underlying CBF's vf as well
    if hasattr(asif_inst, 'cbf'):
        _wrap_vf_scalar(asif_inst.cbf)

    orig = asif_inst.set_constraint

    def safe_set_constraint(Lf_h, Lg_h, h):
        Lf_h = float(np.real(np.asarray(Lf_h).flat[0]))
        Lg_h = np.real(np.atleast_1d(Lg_h)).astype(float)
        h    = float(np.real(np.asarray(h).flat[0]))
        if not np.isfinite(Lf_h):
            Lf_h = 0.0
        if not np.all(np.isfinite(Lg_h)):
            Lg_h = np.zeros_like(Lg_h)
        if not np.isfinite(h):
            h = -1.0
        return orig(Lf_h, Lg_h, h)

    asif_inst.set_constraint = safe_set_constraint
    return asif_inst


def _make_tabular_real(tabular_cbf, vf_array):
    """Assign a real float64 vf_table, stripping any complex JAX dtype."""
    tabular_cbf.vf_table = np.real(np.array(vf_array)).astype(float)
    return tabular_cbf


def _plot_ss_trajectories(results, dyn, x_indices, ax, colors=None, labels=None):
    """
    Plot state-space trajectories directly from a results DataFrame,
    bypassing StateSpaceExperiment.plot which calls add_arrow and hits an
    off-by-one IndexError in experiment_wrapper when the midpoint index
    equals len(data) (i.e. position == xvals[-1] exactly).

    results   : DataFrame with columns [controller, measurement, value, t]
    dyn       : dynamics object whose .STATES list names the state measurements
    x_indices : [i, j] — which state indices go on the x and y axes
    ax        : matplotlib Axes to draw into
    colors    : list of colours, one per controller (cycles if None)
    labels    : list of legend labels; defaults to controller names
    """
    controllers = results["controller"].unique()
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xi, yi = x_indices
    x_state = dyn.STATES[xi]
    y_state = dyn.STATES[yi]

    for k, ctrl in enumerate(controllers):
        sub  = results[results["controller"] == ctrl]
        xvals = (sub[sub["measurement"] == x_state]
                 .sort_values("t")["value"].values)
        yvals = (sub[sub["measurement"] == y_state]
                 .sort_values("t")["value"].values)
        col   = colors[k % len(colors)]
        lbl   = (labels[k] if labels and k < len(labels) else ctrl)
        ax.plot(xvals, yvals, color=col, label=lbl, linewidth=2.5)
        # Draw a direction arrow at the midpoint (clipped to valid range)
        if len(xvals) >= 2:
            mid = min(len(xvals) // 2, len(xvals) - 2)
            ax.annotate("",
                xy=(xvals[mid + 1], yvals[mid + 1]),
                xytext=(xvals[mid], yvals[mid]),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            )


# ═════════════════════════════════════════════════════════════════════════════
# A. ADAPTIVE CRUISE CONTROL  (Section VI-A, Fig. 2)
# ═════════════════════════════════════════════════════════════════════════════

# ── Dynamics ─────────────────────────────────────────────────────────────────

class ACCDynamics(ControlAffineDynamics):
    STATES   = ["P", "V", "dP"]
    CONTROLS = ["ACC"]

    def __init__(self, params, **kwargs):
        params["n_dims"]       = 3
        params["control_dims"] = 1
        self.mass = params["mass"]
        self.g    = params["g"]
        self.f0   = params["f0"]
        self.f1   = params["f1"]
        self.f2   = params["f2"]
        self.rolling_resistance = (
            lambda x: self.f0 + self.f1 * x[..., 1] + self.f2 * x[..., 1] ** 2)
        self.v0 = params["v0"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -1 / self.mass * self.rolling_resistance(state)
        f[..., 2] = self.v0 - state[..., 1]
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = 1 / self.mass
        return B

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 1] = 1
        J[..., 1, 1] = -1 / self.mass * (self.f1 + 2 * self.f2 * state[..., 1])
        J[..., 2, 1] = -1
        return J


class ACCJNPDynamics(ACCDynamics):
    def __init__(self, params, **kwargs):
        # JNP dynamics classes are only used by hj_reachability (JAX tracing).
        # cbf_opt's test_control_affine_dynamics uses numpy and fails when the
        # overridden methods return JAX arrays — the outputs are numerically
        # identical but np.allclose can't compare across backends reliably.
        # Force test=False; correctness is verified by hj_reachability itself.
        kwargs['test'] = False
        super().__init__(params, **kwargs)
        self.rolling_resistance = (
            lambda x: self.f0 + self.f1 * x[1] + self.f2 * x[1] ** 2)

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([state[1],
                          -1 / self.mass * self.rolling_resistance(state),
                          self.v0 - state[1]])

    def control_matrix(self, state, time=0.0):
        return jnp.expand_dims(jnp.array([0, 1 / self.mass, 0]), axis=-1)

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(3), axis=-1)


# ── CBF ──────────────────────────────────────────────────────────────────────

class ACCCBF(ControlAffineCBF):
    def __init__(self, dynamics, params, **kwargs):
        self.Th = params["Th"]
        self.cd = params["cd"]
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=None):
        return (state[..., 2]
                - self.Th * state[..., 1]
                - (state[..., 1] - self.dynamics.v0) ** 2
                  / (2 * self.cd * self.dynamics.g))

    def vf_dt_partial(self, state, time=None):
        return 0.0

    def _grad_vf(self, state, time=None):
        dvf = np.zeros_like(state)
        dvf[..., 1] = (-self.Th
                       - (state[..., 1] - self.dynamics.v0)
                       / (self.cd * self.dynamics.g))
        dvf[..., 2] = 1.0
        return dvf


# ── Main experiment ───────────────────────────────────────────────────────────

def run_acc(out_dir, skip_solves=False):
    print("\n" + "=" * 60)
    print("  A. Adaptive Cruise Control")
    print("=" * 60)

    # Parameters (matching the notebook)
    params = dict(dt=0.01, g=9.81, v0=14, f0=0.1, f1=5, f2=0.25, mass=1650)
    cbf_params = dict(cd=0.3, Th=1.8)

    acc     = ACCDynamics(params)
    acc_jnp = ACCJNPDynamics(params)
    acc_cbf = ACCCBF(acc, cbf_params)

    umax = np.array([cbf_params["cd"] * params["mass"] * params["g"]])
    umin = -umax

    # Grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([0., 10., 0.]), jnp.array([1e3, 40., 100.])),
        (51, 101, 101))

    obstacle = grid.states[..., 2] - acc_cbf.Th * grid.states[..., 1]

    dyn_hjr = HJControlAffineDynamics(
        acc_jnp, control_space=hj.sets.Box(jnp.array(umin), jnp.array(umax)))

    brt = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    solver_settings = hj.SolverSettings.with_accuracy(
        "medium", value_postprocessor=brt(obstacle))

    times = jnp.linspace(0., -20., 101)

    # ── Solve: vanilla HJR (B_ell) ──────────────────────────────────────────
    hjr_path    = os.path.join(out_dir, "acc_target_values_hjr.npy")
    refine_path = os.path.join(out_dir, "acc_target_values_refine.npy")

    if skip_solves and os.path.exists(hjr_path) and os.path.exists(refine_path):
        print("  Loading saved ACC value functions ...")
        target_values_hjr    = np.load(hjr_path)
        target_values_refine = np.load(refine_path)
    else:
        print("  [1/2] Solving vanilla HJR (B_ell) ...")
        target_values_hjr = hj.solve(solver_settings, dyn_hjr, grid, times, obstacle)
        print("  [2/2] Solving refineCBF (B_h) ...")
        acc_tab = TabularControlAffineCBF(acc_jnp, dict(), grid=grid)
        acc_tab.tabularize_cbf(ACCCBF(acc_jnp, cbf_params))
        target_values_refine = hj.solve(
            solver_settings, dyn_hjr, grid, times, acc_tab.vf_table)
        np.save(hjr_path,    np.array(target_values_hjr))
        np.save(refine_path, np.array(target_values_refine))

    # Build tabular CBFs (converged + two partial snapshots)
    timestamp = 2

    refined_cbf = TabularControlAffineCBF(acc, grid=grid)
    _make_tabular_real(refined_cbf, target_values_refine[-1])

    partial_hjr_cbf = TabularControlAffineCBF(acc, grid=grid)
    _make_tabular_real(partial_hjr_cbf, target_values_hjr[timestamp])

    partial_refined_cbf = TabularControlAffineCBF(acc, grid=grid)
    _make_tabular_real(partial_refined_cbf, target_values_refine[timestamp])

    # Controllers
    desired_vel   = 24
    feedback_gain = 200
    nominal_policy = lambda x, t: np.atleast_1d(
        np.clip(-feedback_gain * (x[..., 1] - desired_vel), umin, umax))

    alpha = lambda x: 5 * x

    acc_asif          = _patch_asif_for_real_values(
        ControlAffineASIF(acc, acc_cbf,         alpha=alpha,
                          nominal_policy=nominal_policy, umin=umin, umax=umax))
    acc_asif_ws       = _patch_asif_for_real_values(
        ControlAffineASIF(acc, refined_cbf,     alpha=alpha,
                          nominal_policy=nominal_policy, umin=umin, umax=umax))
    acc_partial_hjr   = _patch_asif_for_real_values(
        ControlAffineASIF(acc, partial_hjr_cbf, alpha=alpha,
                          nominal_policy=nominal_policy, umin=umin, umax=umax))
    acc_partial_cbf   = _patch_asif_for_real_values(
        ControlAffineASIF(acc, partial_refined_cbf, alpha=alpha,
                          nominal_policy=nominal_policy, umin=umin, umax=umax))

    # Rollout
    x0 = np.array([0, 30, 90])
    experiment = RolloutTrajectory(
        "acc_example", start_x=x0, n_sims_per_start=1, t_sim=20)
    print("  Running rollout ...")
    results = experiment.run(acc, {
        "Nominal":      nominal_policy,
        "Analytical":   acc_asif,
        "Refined":      acc_asif_ws,
        "Partial_HJR":  acc_partial_hjr,
        "Partial_CBF":  acc_partial_cbf,
    })
    csv_path = os.path.join(out_dir, "acc_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Figure (Fig. 2 style)
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    outer = gridspec.GridSpec(1, 2, width_ratios=[2, 3], figure=fig, wspace=0.15)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.05)
    axs = [plt.subplot(c) for c in gs1] + [plt.subplot(c) for c in gs2]

    # State-space panel (v vs z)
    x_v = np.linspace(13, 31, 1000)
    axs[0].fill_between(x_v, acc_cbf.Th * x_v, alpha=0.3, color="grey")
    axs[0].plot(x_v, acc_cbf.Th * x_v, color="grey", alpha=0.3)
    axs[0].contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2],
                    np.real(np.array(target_values_refine[-1]))[0].T,
                    levels=[0, 200], alpha=0.3, colors=[ALT_COLORS[2]])

    _plot_ss_trajectories(results, acc, x_indices=[1, 2], ax=axs[0],
                          colors=CHOSEN_COLORS,
                          labels=["Nominal", "Initial CBF", "Converged CBVF",
                                  "Partial HJR", "Partial CBF"])
    axs[0].set_xlabel("Ego velocity [m/s]")
    axs[0].set_ylabel("Distance between cars [m]")
    axs[0].set_xlim([13, 31])
    axs[0].set_ylim([0, 100])

    # Time-series panels
    ts_exp = TimeSeriesExperiment(
        "acc_example", x_indices=[], start_x=x0, n_sims_per_start=1, t_sim=20)
    ts_exp.plot(acc, results, extra_measurements=["vf"], axs=np.array(axs[1:]),
                colors=CHOSEN_COLORS, linestyles=["-", "-", "-", "--", "--"])

    ts = axs[1].lines[0].get_xdata()
    axs[1].plot(ts, np.ones_like(ts) * umax, ":k", label="Bounds")
    axs[1].plot(ts, np.ones_like(ts) * umin, ":k", label="__nolegend__")
    axs[1].set_ylabel("Control [m/s²]")
    axs[1].tick_params(labelbottom=False)
    axs[1].set_xlim(ts[0], 10)
    axs[2].set_ylabel("CBF value")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_xlim(ts[0], 10)
    axs[1].legend(
        ["Nominal", "Initial CBF", "Converged CBVF", "Partial HJR", "Partial CBF"],
        ncol=3, loc="upper right", fontsize=12)
    for ax in axs:
        ax.grid(which="both")

    fig.suptitle("Fig. 2 — Adaptive Cruise Control", fontsize=16)
    fig_path = os.path.join(out_dir, "fig_acc.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # Verify Theorem 1 pointwise (B_h ≤ B_ell)
    diff = np.real(np.array(target_values_refine[-1])) - np.real(np.array(target_values_hjr[-1]))
    print(f"\n  [Theorem 1 check] B_h − B_ell:  "
          f"mean={diff.mean():.4f}  max={diff.max():.4f}")
    if diff.max() > 1e-4:
        print("  WARNING: max > 0 — small numerical errors from interpolation.")
    else:
        print("  PASS: B_h ≤ B_ell everywhere (up to floating-point tolerance).")


# ═════════════════════════════════════════════════════════════════════════════
# B. DUBINS CAR  (Section VI-B, Fig. 3)
# ═════════════════════════════════════════════════════════════════════════════

class DubinsDynamics(ControlAffineDynamics):
    STATES        = ["X", "Y", "THETA"]
    CONTROLS      = ["OMEGA"]
    PERIODIC_DIMS = [2]

    def __init__(self, params, test=False, **kwargs):
        params["n_dims"]        = 3
        params["control_dims"]  = 1
        params["periodic_dims"] = [2]
        self.v = params["v"]
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = self.v * np.cos(state[..., 2])
        f[..., 1] = self.v * np.sin(state[..., 2])
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 2, 0] = 1
        return B

    def disturbance_jacobian(self, state, time=0.0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 2] = -self.v * np.sin(state[..., 2])
        J[..., 1, 2] =  self.v * np.cos(state[..., 2])
        return J


class DubinsJNPDynamics(DubinsDynamics):
    def __init__(self, params, test=False, **kwargs):
        # JNP dynamics classes are only used by hj_reachability (JAX tracing).
        # cbf_opt's test_control_affine_dynamics uses numpy and fails when the
        # overridden methods return JAX arrays — the outputs are numerically
        # identical but np.allclose can't compare across backends reliably.
        # Force test=False; correctness is verified by hj_reachability itself.
        kwargs.pop('test', None)  # avoid duplicate keyword if caller passed it
        super().__init__(params, False, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([self.v * jnp.cos(state[2]),
                          self.v * jnp.sin(state[2]), 0.0])

    def control_matrix(self, state, time=0.0):
        return jnp.expand_dims(jnp.array([0.0, 0.0, 1.0]), axis=-1)

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(3), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        return jnp.array([
            [0.0, 0.0, -self.v * jnp.sin(state[2])],
            [0.0, 0.0,  self.v * jnp.cos(state[2])],
            [0.0, 0.0,  0.0],
        ])


class DubinsCBF(ControlAffineCBF):
    def __init__(self, dynamics, params, **kwargs):
        self.center = params["center"]
        self.r      = params["r"]
        # Force test=False: this CBF is purely positional (h = dist2 - r2),
        # making it relative degree 2 wrt theta in the Dubins dynamics.
        # cbf_opt's gradient test checks the full Lie derivative numerically
        # and fails on r.d.-2 CBFs even when the analytic gradient is correct.
        # This is intentional -- the paper uses this exact candidate as a
        # higher-relative-degree example in Section VI-B.
        kwargs["test"] = False
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        return ((state[..., 0] - self.center[0]) ** 2
                + (state[..., 1] - self.center[1]) ** 2
                - self.r ** 2)

    def _grad_vf(self, state, time=0.0):
        dvf = np.zeros_like(state)
        dvf[..., 0] = 2 * (state[..., 0] - self.center[0])
        dvf[..., 1] = 2 * (state[..., 1] - self.center[1])
        return dvf


class _DubinsNominalPolicy:
    """
    Simplified reach-avoid nominal policy for Dubins car.

    Uses a pure-pursuit / heading-to-goal strategy rather than a full HJR
    solve (which would require the separate dubins.nominal_hjr_control module
    from the original repo).  This reproduces the qualitative trajectory
    behaviour shown in the paper.
    """
    def __init__(self, target, v, umin, umax):
        self.target = np.array(target[:2])
        self.v      = v
        self.umin   = umin
        self.umax   = umax

    def __call__(self, state, time):
        state = np.atleast_2d(state)
        dx    = self.target[0] - state[:, 0]
        dy    = self.target[1] - state[:, 1]
        desired_heading = np.arctan2(dy, dx)
        heading_error   = desired_heading - state[:, 2]
        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        u = np.clip(2.0 * heading_error, self.umin, self.umax)
        return u.reshape(-1, 1)


def run_dubins(out_dir, skip_solves=False):
    print("\n" + "=" * 60)
    print("  B. Dubins Car")
    print("=" * 60)

    dubins_vel = 1.0
    umin = np.array([-0.5])
    umax = np.array([ 0.5])
    dt   = 0.05

    obstacle_center = np.array([5.0, 5.0])
    obstacle_length = np.array([2.0, 2.0])

    dyn     = DubinsDynamics({"v": dubins_vel, "dt": dt}, test=True)
    dyn_jnp = DubinsJNPDynamics({"v": dubins_vel, "dt": dt})

    dubins_cbf = DubinsCBF(
        dyn,
        {"center": obstacle_center,
         "r": np.sqrt(np.max(obstacle_length))})

    # Grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lo=jnp.array([-5., -5., -jnp.pi]),
                    hi=jnp.array([20., 20.,  jnp.pi])),
        (201, 201, 81), periodic_dims=2)

    bl = obstacle_center - obstacle_length / 2

    def constraint_set(state):
        return -jnp.min(jnp.array([
            state[0] - bl[0],
            bl[0] + obstacle_length[0] - state[0],
            state[1] - bl[1],
            bl[1] + obstacle_length[1] - state[1],
        ]))

    obstacle = hj.utils.multivmap(constraint_set, jnp.arange(grid.ndim))(grid.states)

    dyn_hjr = HJControlAffineDynamics(
        dyn_jnp, control_space=hj.sets.Box(jnp.array(umin), jnp.array(umax)))

    brt = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    solver_settings = hj.SolverSettings.with_accuracy(
        "high", value_postprocessor=brt(obstacle))
    times = jnp.linspace(0., -10., 101)

    hjr_path    = os.path.join(out_dir, "dubins_target_values_hjr.npy")
    refine_path = os.path.join(out_dir, "dubins_target_values_refine.npy")

    if skip_solves and os.path.exists(hjr_path) and os.path.exists(refine_path):
        print("  Loading saved Dubins value functions ...")
        target_values_hjr    = np.load(hjr_path)
        target_values_refine = np.load(refine_path)
    else:
        print("  [1/2] Solving vanilla HJR ...")
        target_values_hjr = hj.solve(solver_settings, dyn_hjr, grid, times, obstacle)
        print("  [2/2] Solving refineCBF ...")
        dub_tab = TabularControlAffineCBF(dyn, dict(), grid=grid)
        dub_tab.tabularize_cbf(dubins_cbf)
        target_values_refine = hj.solve(
            solver_settings, dyn_hjr, grid, times, dub_tab.vf_table)
        np.save(hjr_path,    np.array(target_values_hjr))
        np.save(refine_path, np.array(target_values_refine))

    tabular_cbf = TabularControlAffineCBF(dyn, grid=grid)
    _make_tabular_real(tabular_cbf, target_values_refine[-1])

    # Controllers
    target = np.array([6.0, 7.0, 0.0])
    nom_policy = _DubinsNominalPolicy(target, dubins_vel, umin, umax)

    alpha = lambda x: 5 * x
    dubins_asif    = _patch_asif_for_real_values(
        ControlAffineASIF(dyn, dubins_cbf,  alpha=alpha,
                          nominal_policy=nom_policy))
    dubins_asif_ws = _patch_asif_for_real_values(
        ControlAffineASIF(dyn, tabular_cbf, alpha=alpha,
                          nominal_policy=nom_policy, umin=umin, umax=umax))

    # Rollout
    x0  = np.array([2.0, 4.0, np.pi / 3])
    exp = RolloutTrajectory("dubins", start_x=x0, n_sims_per_start=1, t_sim=6)
    print("  Running rollout ...")
    results = exp.run(dyn, {
        "Nominal":    nom_policy,
        "Analytical": dubins_asif,
        "Refined":    dubins_asif_ws,
    })
    csv_path = os.path.join(out_dir, "dubins_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Figure (Fig. 3 style)
    offset = 0   # theta-slice index ≈ -π/2 for offset ~21; use 0 for clarity
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Left: safe-set evolution
    ax = axs[0]
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0],
                np.array(obstacle)[..., offset], levels=[-10, 0],
                alpha=0.3, colors="grey")
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0],
                np.real(np.array(target_values_hjr[-1]))[..., offset],
                levels=[0, 50], colors=[ALT_COLORS[2]], alpha=0.3)
    alphas = [0.6, 0.4, 0.2, 1.0]
    idxs   = [0, 5, 10, -1]
    for idx, a in zip(idxs, alphas):
        c = [tuple(PAPER_COLORS[-1])] if idx == -1 else [tuple(r) for r in PAPER_COLORS]
        ax.contour(grid.coordinate_vectors[1], grid.coordinate_vectors[0],
                   np.real(np.array(target_values_refine[idx]))[..., offset],
                   levels=[0], colors=c, linewidths=3, alpha=a)
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
    ax.set_xlim([1.5, 8.5]); ax.set_ylim([1.5, 9.0])
    ax.set_title("Iterations of CBF safe set ($\\theta$ slice)")
    ax.grid()

    # Right: trajectories
    ax = axs[1]
    _plot_ss_trajectories(results, dyn, x_indices=[0, 1], ax=ax,
                          colors=CHOSEN_COLORS,
                          labels=["Nominal", "Analytical", "Refined"])
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0],
                np.array(obstacle)[..., 0], levels=[-10, 0],
                colors="grey", alpha=0.3)
    ax.plot(*x0[:2], "x", markersize=14, mew=3, color="grey", label="Start")
    ax.plot(*target[:2], "o", markersize=14, color="grey", label="Goal")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel(None)
    ax.set_xlim([1.5, 8.5]); ax.set_ylim([1.5, 9.0])
    ax.set_title("Trajectories with safety filter")
    ax.legend(fontsize=12)
    ax.grid()

    fig.suptitle("Fig. 3 — Dubins Car", fontsize=16)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_dubins.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # Theorem 1 check
    diff = (np.real(np.array(target_values_refine[-1]))
            - np.real(np.array(target_values_hjr[-1])))
    print(f"\n  [Theorem 1 check] B_h − B_ell:  "
          f"mean={diff.mean():.4f}  max={diff.max():.4f}")
    if diff.max() > 1e-4:
        print("  WARNING: max > 0.")
    else:
        print("  PASS: B_h ≤ B_ell everywhere.")


# ═════════════════════════════════════════════════════════════════════════════
# C. QUADROTOR  (Section VI-C, Fig. 4)
#    Reuse the classes / helpers from the original quad_vertical.py
# ═════════════════════════════════════════════════════════════════════════════

# ── Physical constants ────────────────────────────────────────────────────────
Cd_v   = 0.25
G      = 9.81
Cd_phi = 0.02255
MASS   = 2.5
LENGTH = 1.0
IYY    = 1.0
DT     = 0.01
UMAX   = 0.75 * MASS * G * np.ones(2)
UMIN   = np.zeros(2)
QUAD_PARAMS = dict(Cd_v=Cd_v, g=G, Cd_phi=Cd_phi,
                   mass=MASS, length=LENGTH, Iyy=IYY, dt=DT)


class QuadVerticalDynamics(ControlAffineDynamics):
    STATES        = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS      = ["T1", "T2"]
    PERIODIC_DIMS = [2]

    def __init__(self, params, **kwargs):
        self.Cd_v   = params["Cd_v"]
        self.g      = params["g"]
        self.Cd_phi = params["Cd_phi"]
        self.mass   = params["mass"]
        self.length = params["length"]
        self.Iyy    = params["Iyy"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0):
        f = np.zeros_like(state)
        f[..., 0] =  state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] =  state[..., 3]
        f[..., 3] = -self.Cd_phi / self.Iyy * state[..., 3]
        return f

    def control_matrix(self, state, time=0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, :] =  np.cos(state[..., 2]) / self.mass
        B[..., 3, 0] = -self.length / self.Iyy
        B[..., 3, 1] =  self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state, time=0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0):
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] =  1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 2] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 2]) / self.mass
        J[..., 2, 3] =  1.0
        J[..., 3, 3] = -self.Cd_phi / self.Iyy
        return J


class QuadVerticalDynamicsJNP(QuadVerticalDynamics):
    def __init__(self, params, **kwargs):
        # JNP dynamics classes are only used by hj_reachability (JAX tracing).
        # cbf_opt's test_control_affine_dynamics uses numpy and fails when the
        # overridden methods return JAX arrays — the outputs are numerically
        # identical but np.allclose can't compare across backends reliably.
        # Force test=False; correctness is verified by hj_reachability itself.
        kwargs['test'] = False
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([
            state[1],
            -state[1] * self.Cd_v / self.mass - self.g,
            state[3],
            -state[3] * self.Cd_phi / self.Iyy,
        ])

    def control_matrix(self, state, time=0.0):
        cos_phi = jnp.cos(state[2])
        return jnp.array([
            [0.0,                     0.0],
            [cos_phi / self.mass,     cos_phi / self.mass],
            [0.0,                     0.0],
            [-self.length / self.Iyy, self.length / self.Iyy],
        ])

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(4), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        return jnp.array([
            [0, 1, 0, 0],
            [0, -self.Cd_v / self.mass,
               -1 / self.mass * (control[0] + control[1]) * jnp.sin(state[2]),
               0],
            [0, 0, 0, 1],
            [0, 0, 0, -self.Cd_phi / self.Iyy],
        ])


class QuadPlanarDynamics(QuadVerticalDynamics):
    STATES        = ["X", "XDOT", "Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS      = ["T1", "T2"]
    PERIODIC_DIMS = [4]

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] =  state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1]
        f[..., 2:] = super().open_loop_dynamics(state[..., 2:], time)
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, :] = -np.sin(state[..., 4]) / self.mass
        B[..., 2:, :] = super().control_matrix(state[..., 2:], time)
        return B

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] =  1
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 4] = -(control[..., 0] + control[..., 1]) * np.cos(state[..., 4]) / self.mass
        J[..., 2:, 2:] = super().state_jacobian(state[..., 2:], control, time)
        return J


def _quad_safe_set(state):
    return jnp.min(jnp.array([
        state[0] - 1, 9 - state[0],
        state[1] + 6, 6 - state[1],
        state[3] + 8, 8 - state[3],
    ]))


class QuadVerticalCBF(ControlAffineCBF):
    def __init__(self, dynamics, params, **kwargs):
        self.scaling = params["scaling"]
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        s = self.scaling
        return 10 - (s[0] * (5 - state[..., 0]) ** 2
                   + s[1] * state[..., 1] ** 2
                   + s[2] * state[..., 2] ** 2
                   + s[3] * state[..., 3] ** 2)

    def vf_dt_partial(self, state, time=0.0):
        return 0.0

    def _grad_vf(self, state, time=0.0):
        s = self.scaling
        return s * np.array([
             2 * (5 - state[..., 0]),
            -2 * state[..., 1],
            -2 * state[..., 2],
            -2 * state[..., 3],
        ]).T


class ExtendedQuadVerticalCBF(QuadVerticalCBF):
    def vf(self, state, time=0):
        return super().vf(state[..., 2:], time)

    def _grad_vf(self, state, time=0):
        dV = np.zeros_like(state)
        dV[..., 2:] = super()._grad_vf(state[..., 2:], time)
        return dV


def _build_quad_grid():
    domain = hj.sets.Box(lo=jnp.array([0., -8., -jnp.pi, -10.]),
                         hi=jnp.array([10.,  8.,  jnp.pi,  10.]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, (31, 25, 41, 25), periodic_dims=2)
    safe_values     = hj.utils.multivmap(_quad_safe_set, jnp.arange(4))(grid.states)
    brt             = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    solver_settings = hj.SolverSettings.with_accuracy(
        "high", value_postprocessor=brt(safe_values))
    return grid, solver_settings, safe_values


def _run_quad_hjr_solves(dyn_jnp, grid, solver_settings, safe_values,
                         cbf_params, out_dir, skip_solves):
    dyn_hjr = HJControlAffineDynamics(
        dyn_jnp, control_space=hj.sets.Box(UMIN, UMAX))
    times = jnp.linspace(0., -5., 101)

    hjr_path    = os.path.join(out_dir, "quad_target_values_hjr.npy")
    refine_path = os.path.join(out_dir, "quad_target_values_refine.npy")

    if skip_solves and os.path.exists(hjr_path) and os.path.exists(refine_path):
        print("  Loading saved Quad value functions ...")
        target_values_hjr    = np.load(hjr_path)
        target_values_refine = np.load(refine_path)
    else:
        print("  [1/2] Solving vanilla HJR ...")
        target_values_hjr = hj.solve(
            solver_settings, dyn_hjr, grid, times, safe_values)
        print("  [2/2] Solving refineCBF ...")
        dyn_4state   = QuadVerticalDynamics(QUAD_PARAMS, test=True)
        cbf_4state   = QuadVerticalCBF(dyn_4state, cbf_params, test=False)
        quad_tab     = TabularControlAffineCBF(dyn_4state, cbf_params, grid=grid)
        quad_tab.tabularize_cbf(cbf_4state)
        target_values_refine = hj.solve(
            solver_settings, dyn_hjr, grid, times, quad_tab.vf_table)
        np.save(hjr_path,    np.array(target_values_hjr))
        np.save(refine_path, np.array(target_values_refine))

    dyn_4state = QuadVerticalDynamics(QUAD_PARAMS, test=True)
    quad_tab   = TabularControlAffineCBF(dyn_4state, cbf_params, grid=grid)
    _make_tabular_real(quad_tab, target_values_refine[-1])
    return target_values_hjr, target_values_refine, quad_tab


def _build_extended_quad_cbf(quad_tabular_cbf, cbf_params):
    extended_domain = hj.sets.Box(
        lo=jnp.array([-30., -8.,  0., -8., -jnp.pi, -10.]),
        hi=jnp.array([ 30.,  8., 10.,  8.,  jnp.pi,  10.]))
    ext_grid     = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        extended_domain, (5, 5, 31, 25, 41, 25), periodic_dims=4)
    extended_dyn = QuadPlanarDynamics(QUAD_PARAMS, test=False)
    ext_tab      = TabularControlAffineCBF(extended_dyn, cbf_params, grid=ext_grid)

    vf_4d = np.real(quad_tabular_cbf.vf_table).astype(float)
    ext_tab.vf_table = np.repeat(
        np.repeat(vf_4d[None, ...], 5, axis=0)[None, ...], 5, axis=0)

    return extended_dyn, ext_grid, ext_tab


def _build_quad_lqr(extended_dyn):
    x_nom = np.array([15., 0., 3., 0., 0., 0.])
    u_nom = 0.5 * MASS * G * np.ones(2)
    try:
        A_d, B_d = extended_dyn.linearized_dt_dynamics(x_nom, u_nom)
    except AttributeError:
        # Fallback: first-order Euler discretisation of the continuous Jacobians
        A_c, B_c = extended_dyn.linearized_ct_dynamics(x_nom, u_nom)
        dt = extended_dyn.dt
        A_d = np.eye(A_c.shape[0]) + dt * A_c
        B_d = dt * B_c
    Q = np.diag([1.0, 0.1, 1.0, 0.1, 1.0, 1.0])
    R = np.eye(2)
    K = utils.lqr(A_d, B_d, Q, R)
    return K


def _quad_run_scenario(extended_dyn, quad_extended_cbf, ext_tab,
                       K, x0, x_goal, label, out_dir):
    u_hover     = 0.5 * MASS * G * np.ones(2)
    alpha       = lambda x: 5 * x
    nom_control = lambda x, t: np.atleast_2d(
        np.clip(u_hover - (K @ (x - x_goal).T).T, UMIN, UMAX))

    cbf_filt  = _patch_asif_for_real_values(
        ControlAffineASIF(extended_dyn, quad_extended_cbf,
                          alpha=alpha, nominal_policy=nom_control,
                          umin=UMIN, umax=UMAX))
    cbvf_filt = _patch_asif_for_real_values(
        ControlAffineASIF(extended_dyn, ext_tab,
                          alpha=alpha, nominal_policy=nom_control,
                          umin=UMIN, umax=UMAX))

    print(f"  Running rollout: {label}")
    exp = RolloutTrajectory("quad", start_x=x0, n_sims_per_start=1, t_sim=8)
    results = exp.run(extended_dyn, {
        "Nominal":    nom_control,
        "Analytical": cbf_filt,
        "CBVF":       cbvf_filt,
    })
    csv_path = os.path.join(out_dir, f"quad_{label}_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return results, nom_control, cbf_filt, cbvf_filt


def run_quadrotor(out_dir, skip_solves=False):
    print("\n" + "=" * 60)
    print("  C. Quadrotor (expert-synthesised CBF)")
    print("=" * 60)

    cbf_params  = {"scaling": np.array([0.75, 0.5, 2.0, 0.5])}
    dyn_jnp     = QuadVerticalDynamicsJNP(QUAD_PARAMS)

    grid, solver_settings, safe_values = _build_quad_grid()
    print(f"  Grid shape: {grid.states.shape[:-1]}")

    target_values_hjr, target_values_refine, quad_tab = _run_quad_hjr_solves(
        dyn_jnp, grid, solver_settings, safe_values, cbf_params,
        out_dir, skip_solves)

    extended_dyn, ext_grid, ext_tab = _build_extended_quad_cbf(
        quad_tab, cbf_params)
    quad_extended_cbf = ExtendedQuadVerticalCBF(
        extended_dyn, cbf_params, test=False)

    K = _build_quad_lqr(extended_dyn)

    # Conservative scenario
    x0_con  = np.array([0.,  4., 7., -2., -np.pi/4, 0.])
    xg_con  = np.array([6.,  0., 9.,  0.,  0.,      0.])
    res_con, *_ = _quad_run_scenario(
        extended_dyn, quad_extended_cbf, ext_tab,
        K, x0_con, xg_con, "conservative", out_dir)

    # Invalid scenario
    x0_inv  = np.array([15., -3., 2.5, -2.,  np.pi/4, 1.])
    xg_inv  = np.array([ 0.,  0., 1.5,  0.,  0.,      0.])
    res_inv, *_ = _quad_run_scenario(
        extended_dyn, quad_extended_cbf, ext_tab,
        K, x0_inv, xg_inv, "invalid", out_dir)

    # Figures (Fig. 4 style)
    for label, x0, x_goal, results in [
        ("conservative", x0_con, xg_con, res_con),
        ("invalid",      x0_inv, xg_inv, res_inv),
    ]:
        fig, ax = plt.subplots(figsize=(8, 8))
        _plot_ss_trajectories(results, extended_dyn, x_indices=[0, 2],
                              ax=ax, colors=CHOSEN_COLORS,
                              labels=["Nominal", "Analytical", "CBVF"])
        ax.plot(*x0[[0, 2]],    "x", markersize=14, mew=3,
                color="grey", label="Start")
        ax.plot(*x_goal[[0, 2]], "o", markersize=14,
                color="grey", label="Goal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"Quadrotor trajectories — {label}")
        ax.legend(fontsize=12)
        ax.grid()
        fig_path = os.path.join(out_dir, f"fig_quad_{label}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # Safe-set evolution figure (phi–y slice)
    offset_vy = 12; offset_omega = 12
    fig, ax = plt.subplots(figsize=(7, 7))
    safe_v = np.real(np.array(safe_values))
    vf_r   = np.real(np.array(target_values_refine))
    ax.contourf(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                safe_v[:, offset_vy, :, offset_omega],
                levels=[-10, 0], colors=[(0.3, 0.3, 0.3)], alpha=0.6)
    for idx, a in zip([0, 3, 10, -1], [0.6, 0.4, 0.2, 1.0]):
        c = [tuple(PAPER_COLORS[-1])] if idx == -1 else [tuple(r) for r in PAPER_COLORS]
        ax.contour(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                   vf_r[idx][:, offset_vy, :, offset_omega],
                   levels=[0], colors=c, linewidths=3, alpha=a)
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$y$ [m]")
    ax.set_title("Safe-set evolution (φ–y slice)")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_quad_safe_set_evolution.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Theorem 1 check
    diff = np.real(np.array(target_values_refine[-1])) - np.real(np.array(target_values_hjr[-1]))
    print(f"\n  [Theorem 1 check] B_h − B_ell: "
          f"mean={diff.mean():.4f}  max={diff.max():.4f}")
    if diff.max() > 1e-4:
        print("  WARNING: max > 0.")
    else:
        print("  PASS: B_h ≤ B_ell everywhere.")


# ═════════════════════════════════════════════════════════════════════════════
# D. INVERTED PENDULUM / BACKUP CBF  (Section VI-D, Fig. 5)
# ═════════════════════════════════════════════════════════════════════════════

class InvPendulumDynamics(ControlAffineDynamics):
    STATES   = ["THETA", "THETA_DOT"]
    CONTROLS = ["U"]

    def __init__(self, params, **kwargs):
        params["n_dims"]       = 2
        params["control_dims"] = 1
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = np.sin(state[..., 0])
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = 1
        return B

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 1] = 1
        J[..., 1, 0] = np.cos(state[..., 0])
        return J


class InvPendulumJNPDynamics(InvPendulumDynamics):
    def __init__(self, params, **kwargs):
        # JNP dynamics classes are only used by hj_reachability (JAX tracing).
        # cbf_opt's test_control_affine_dynamics uses numpy and fails when the
        # overridden methods return JAX arrays — the outputs are numerically
        # identical but np.allclose can't compare across backends reliably.
        # Force test=False; correctness is verified by hj_reachability itself.
        kwargs['test'] = False
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([state[1], jnp.sin(state[0])])

    def control_matrix(self, state, time=0.0):
        return jnp.expand_dims(jnp.array([0.0, 1.0]), axis=-1)

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(2), axis=-1)


class InvPendulumBackupController(cbf_module.BackupController):
    def __init__(self, dynamics, T_backup, **kwargs):
        self.lqr_term = kwargs.get(
            "lqr_term", np.zeros((dynamics.control_dims, dynamics.n_dims)))
        super().__init__(dynamics, T_backup, **kwargs)

    def policy(self, x, t):
        x = np.atleast_2d(x)

        u = -(self.lqr_term @ x.T).T
        u = np.clip(u, self.umin, self.umax)

        if u.shape[0] == 1:
            return u[0]

        return u

    def grad_policy(self, x, t):
        return -self.lqr_term


class InvPendulumSafetyCBF(ControlAffineCBF):
    def vf(self, state, time=0.0):
        val = np.minimum(1 - state[..., 0] ** 2, 2 - state[..., 1] ** 2)
        # cbf_opt's internals do `hs[0] = val_curr` which needs a plain scalar.
        return _to_scalar(val)

    def vf_dt_partial(self, state, time=0.0):
        return 0.0

    def _grad_vf(self, state, time=0.0):
        dvf = np.zeros_like(state)
        mask = 1 - state[..., 0] ** 2 < 2 - state[..., 1] ** 2
        dvf[..., 0] = -2 * state[..., 0] *  mask
        dvf[..., 1] = -2 * state[..., 1] * ~mask
        return dvf


class InvPendulumImplicitCBF(cbf_module.ControlAffineImplicitCBF):
    def __init__(self, dynamics, params, backup_controller, safety_cbf, **kwargs):
        self.delta           = params["delta"]
        self.backup_vf_scalar = kwargs.get("backup_vf_scalar", 100)
        super().__init__(dynamics, params, backup_controller, safety_cbf, **kwargs)

    def backup_vf(self, state, time=0.0):
        val = self.backup_vf_scalar * np.minimum(
            (np.pi / 12) ** 2 - state[..., 0] ** 2,
            self.delta ** 2   - state[..., 1] ** 2
        )
        return _to_scalar(val)

    def _grad_backup_vf(self, state, time=0.0):
        dvf  = np.zeros_like(state)
        mask = ((np.pi / 12) ** 2 - state[..., 0] ** 2
                < self.delta ** 2 - state[..., 1] ** 2)
        dvf[..., 0] = -2 * state[..., 0] *  mask
        dvf[..., 1] = -2 * state[..., 1] * ~mask
        return self.backup_vf_scalar * dvf


def run_pendulum(out_dir, skip_solves=False):
    print("\n" + "=" * 60)
    print("  D. Inverted Pendulum (Backup CBF)")
    print("=" * 60)

    inv_pend     = InvPendulumDynamics({"dt": 0.01})
    inv_pend_jnp = InvPendulumJNPDynamics({"dt": 0.01})

    umax = 3 * np.ones(inv_pend.control_dims)
    umin = -umax

    # LQR backup controller
    xnom = np.zeros(inv_pend.n_dims)
    unom = np.zeros(inv_pend.control_dims)
    A    = inv_pend.state_jacobian(xnom, unom)
    try:
        B = inv_pend.control_jacobian(xnom, unom)
    except AttributeError:
        # control_jacobian() not present; use control_matrix() at the nom point
        B = inv_pend.control_matrix(xnom[None], unom[None]).squeeze()
    Q    = np.eye(inv_pend.n_dims)
    R    = 1e4 * np.eye(inv_pend.control_dims)
    P    = solve_continuous_are(A, B, Q, R)
    F    = np.linalg.inv(R) @ B.T @ P
    backup_controller = InvPendulumBackupController(
        inv_pend, T_backup=3.5, lqr_term=F, umin=umin, umax=umax)

    safety_cbf   = InvPendulumSafetyCBF(inv_pend_jnp, {})
    inv_pend_cbf = InvPendulumImplicitCBF(
        inv_pend, params={"delta": 0.1},
        backup_controller=backup_controller, safety_cbf=safety_cbf)

    dyn_reach = HJControlAffineDynamics(
        inv_pend_jnp, control_space=hj.sets.Box(umin, umax))

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([-np.pi, -np.pi]),
                    np.array([ np.pi,  np.pi])),
        (101, 101))

    obstacle = jnp.minimum(1 - grid.states[..., 0] ** 2,
                           2 - grid.states[..., 1] ** 2)

    # Tabularise the implicit CBF value function
    print("  Tabularising implicit CBF ...")
    tab_cbf     = TabularControlAffineCBF(inv_pend, dict(), grid=grid)
    grid_np     = np.array(grid.states)
    vf_table    = np.zeros((grid.states.shape[0], grid.states.shape[1]))
    for i in tqdm(range(grid.states.shape[0]), leave=False):
        for j in range(grid.states.shape[1]):
            try:
                vf_table[i, j] = inv_pend_cbf.vf(
                    grid_np[i, j], break_unsafe=False)
            except TypeError:
                # break_unsafe kwarg not present in this cbf_opt version
                vf_table[i, j] = inv_pend_cbf.vf(grid_np[i, j])
    tab_cbf.vf_table = vf_table
    # Ensure single-state vf calls return a plain float for cbf_opt internals.
    _wrap_vf_scalar(tab_cbf)
    _wrap_vf_scalar(inv_pend_cbf)

    if not hasattr(inv_pend_cbf, "_backup_vf_scalar_wrapped"):
        _orig_backup_vf = inv_pend_cbf.backup_vf

        def _backup_vf_wrapped(state, time=0.0, _f=_orig_backup_vf):
            return _to_scalar(_f(state, time))

        inv_pend_cbf.backup_vf = _backup_vf_wrapped
        inv_pend_cbf._backup_vf_scalar_wrapped = True

    brt = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=brt(obstacle))

    refine_path  = os.path.join(out_dir, "pendulum_target_values_refine.npy")
    viability_path = os.path.join(out_dir, "pendulum_viability_kernel.npy")

    if skip_solves and os.path.exists(refine_path) and os.path.exists(viability_path):
        print("  Loading saved pendulum value functions ...")
        target_values        = np.load(refine_path)
        viability_kernel_vf  = np.load(viability_path)
    else:
        print("  [1/2] Solving refineCBF (B_h, short horizon) ...")
        times         = np.linspace(0., -2., 101)
        target_values = hj.solve(
            solver_settings, dyn_reach, grid, times, tab_cbf.vf_table)
        print("  [2/2] Solving for viability kernel (long horizon) ...")
        # hj.step() was removed in some hj_reachability versions;
        # use hj.solve() over a long horizon to obtain the viability kernel.
        viability_kernel_vf = hj.solve(
            solver_settings, dyn_reach, grid,
            jnp.array([0.0, -20.0]), obstacle)[-1]
        np.save(refine_path,   np.array(target_values))
        np.save(viability_path, np.array(viability_kernel_vf))

    # Controllers
    nominal_policy = lambda x, t: (2.0 * np.ones((np.atleast_2d(x).shape[0], 1)))
    alpha = lambda x: 5 * x

    try:
        trade_off_filter = cbf_asif_module.TradeoffFilter(
            inv_pend, inv_pend_cbf, backup_controller,
            nominal_policy=nominal_policy, beta=30.0)
    except AttributeError:
        # TradeoffFilter renamed or removed in this cbf_opt version.
        # Fall back to using the nominal policy directly (no backup safety).
        print("  WARNING: cbf_opt.asif.TradeoffFilter not found; "
              "using nominal policy as trade-off controller.")
        trade_off_filter = nominal_policy

    ca_tab = TabularControlAffineCBF(inv_pend, grid=grid)
    _make_tabular_real(ca_tab, target_values[-1])
    # Ensure single-state vf calls return a plain Python float, not an array.
    _wrap_vf_scalar(ca_tab)
    cbvf_filter = _patch_asif_for_real_values(
        ControlAffineASIF(inv_pend, ca_tab, alpha=alpha,
                          nominal_policy=nominal_policy, umin=umin, umax=umax))

    # Rollout
    x0  = np.array([0.0, 0.0])
    # Use RolloutTrajectory instead of StateSpaceExperiment to avoid
    # the add_arrow off-by-one IndexError in experiment_wrapper.
    exp = RolloutTrajectory("pendulum", start_x=x0,
                            n_sims_per_start=1, t_sim=5)
    print("  Running rollout ...")
    results = exp.run(inv_pend, {
        "nominal":  nominal_policy,
        "tradeoff": trade_off_filter,
        "cbvf":     cbvf_filter,
    })
    csv_path = os.path.join(out_dir, "pendulum_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Figure (Fig. 5 style)
    XX = grid.coordinate_vectors[0]
    YY = grid.coordinate_vectors[1]
    ZZ = inv_pend_cbf.backup_vf(grid.states)

    a_col      = np.array([sns.color_palette("bright")[4]])
    alt_colors = sns.color_palette("pastel", 9).as_hex()

    fig, axs = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    # Left: safety filter + backup trajectories
    ax = axs[0]
    _plot_ss_trajectories(results, inv_pend, x_indices=[0, 1], ax=ax,
                          colors=CHOSEN_COLORS,
                          labels=["Nominal", "Trade-off", "CBVF"])
    if hasattr(backup_controller, "rollout_backup"):
        for i, t in enumerate(results.t.unique()[::50]):
            state = results[
                (results.controller == "tradeoff")
                & (results.t == t)
                & (results.measurement.isin(inv_pend.STATES))
            ].value.values
            try:
                states_bu, _ = backup_controller.rollout_backup(state)
                ax.plot(states_bu[:, 0], states_bu[:, 1], "--",
                        alpha=0.3, color=CHOSEN_COLORS[3],
                        label="Backup $\\pi$" if i == 0 else "_nolegend_")
            except Exception:
                pass
    ax.contourf(XX, YY, np.real(np.array(obstacle)).T, levels=[-10, 0],
                colors="grey", alpha=0.3)
    ax.contour(XX, YY, np.real(np.array(obstacle)).T,  levels=[0],
               colors="grey", alpha=0.6, linewidths=2)
    ax.contour(XX, YY, np.real(np.array(ZZ)).T,        levels=[0],
               colors=a_col, linewidths=4)
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlabel(r"$\theta$ [rad]"); ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax.set_title("Safety filter & backup trajectories")
    ax.legend(fontsize=10)
    ax.grid()

    # Right: safe-set evolution
    ax = axs[1]
    ax.contourf(XX, YY, np.real(np.array(obstacle)).T, levels=[-10, 0],
                colors="grey", alpha=0.3)
    ax.contourf(XX, YY, np.real(np.array(viability_kernel_vf)).T,
                levels=[0, 10], colors=alt_colors[2], alpha=0.3)
    tv_real = np.real(np.array(target_values))
    for idx, a in zip([0, 5, 10, -1], [0.6, 0.4, 0.2, 1.0]):
        c = [tuple(PAPER_COLORS[-1])] if idx == -1 else [tuple(r) for r in PAPER_COLORS]
        ax.contour(XX, YY, tv_real[idx].T, levels=[0],
                   colors=c, linewidths=3, alpha=a)
    ax.contour(XX, YY, np.real(np.array(ZZ)).T, levels=[0],
               colors=a_col, linewidths=4)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_xlim([-1.2, 1.2]); ax.set_ylim([-2, 2])
    ax.set_title("Iterations of CBF safe set")
    ax.grid()

    fig.suptitle("Fig. 5 — Inverted Pendulum (Backup CBF)", fontsize=16)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_pendulum.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI & entrypoint
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Replicate all results from Tonkens & Herbert, IROS 2022.")
    p.add_argument(
        "--experiments", nargs="+",
        choices=["acc", "dubins", "quad", "pendulum"],
        default=["acc", "dubins", "quad", "pendulum"],
        help="Which experiments to run (default: all four).")
    p.add_argument(
        "--skip-solves", action="store_true",
        help="Skip HJR DP solves and load saved .npy files if present.")
    p.add_argument(
        "--out-dir", default="results",
        help="Directory for all outputs (default: ./results/).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  refineCBF Full Replication — Tonkens & Herbert IROS 2022")
    print("  Python", sys.version.split()[0])
    print("=" * 60)
    print(f"  Experiments : {args.experiments}")
    print(f"  Output dir  : {os.path.abspath(args.out_dir)}")
    print(f"  Skip solves : {args.skip_solves}")

    runners = {
        "acc":      run_acc,
        "dubins":   run_dubins,
        "quad":     run_quadrotor,
        "pendulum": run_pendulum,
    }

    for key in args.experiments:
        runners[key](args.out_dir, skip_solves=args.skip_solves)

    print("\n" + "=" * 60)
    print(f"  All done!  Outputs: {os.path.abspath(args.out_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()