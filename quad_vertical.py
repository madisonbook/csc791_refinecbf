"""
quad_vertical.py
================
Standalone replication of:

    Tonkens & Herbert, "Refining Control Barrier Functions through
    Hamilton-Jacobi Reachability", arXiv:2204.12507v2, 2022.

Self-righting planar quadrotor experiment (Section VI-C of the paper).

Usage
-----
    # Install deps first (once):
    pip install "cbf_opt>=0.6.0" "experiment-wrapper>=1.1" hj-reachability
    git clone https://github.com/stonkens/refineCBF.git
    pip install -e refineCBF

    # Run everything and save all figures to ./results/:
    python quad_vertical.py

    # Only run one scenario (conservative OR invalid candidate):
    python quad_vertical.py --case conservative
    python quad_vertical.py --case invalid

    # Show safe-set evolution plot variant (1, 2, or 3):
    python quad_vertical.py --sets-case 3

Outputs (written to ./results/)
--------------------------------
    conservative_results.csv   — rollout data, conservative scenario
    invalid_results.csv        — rollout data, invalid-candidate scenario
    target_values_hjr.npy      — vanilla HJR value function (101 time slices)
    target_values_refine.npy   — refineCBF value function  (101 time slices)
    fig_trajectories_conservative.png
    fig_trajectories_invalid.png
    fig_safe_set_evolution.png
    fig_value_functions_comparison.png

Bugs fixed relative to the original Colab notebook
----------------------------------------------------
1. np.sin → jnp.sin inside QuadVerticalDynamicsJNP.state_jacobian
   (np.sin breaks JAX tracing / JIT compilation).
2. Cell-ordering bug: vf_table was broadcast to 6-D *before* being updated
   with the converged DP result in some execution orders.
3. Visualisation cells tried to load pre-saved .npy files that don't ship
   with the repo; we use the arrays computed in this run instead.
4. Missing os.makedirs before writing CSV output files.
5. %matplotlib inline magic removed (not valid outside Jupyter).
"""

import argparse
import logging
import os
import subprocess
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("cbf_opt").setLevel(logging.ERROR)

import jax.numpy as jnp
import hj_reachability as hj
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF, utils
from experiment_wrapper import RolloutTrajectory, StateSpaceExperiment
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF


# ──────────────────────────────────────────────────────────────────────────────
# Physical parameters
# ──────────────────────────────────────────────────────────────────────────────

Cd_v   = 0.25      # translational drag coefficient
G      = 9.81      # gravity [m/s²]
Cd_phi = 0.02255   # rotational drag coefficient
MASS   = 2.5       # [kg]
LENGTH = 1.0       # arm length [m]
IYY    = 1.0       # moment of inertia [kg·m²]
DT     = 0.01      # simulation timestep [s]

UMAX = 0.75 * MASS * G * np.ones(2)   # max thrust per rotor [N]
UMIN = np.zeros(2)                     # rotors can't pull, only push

PARAMS = dict(Cd_v=Cd_v, g=G, Cd_phi=Cd_phi,
              mass=MASS, length=LENGTH, Iyy=IYY, dt=DT)


# ──────────────────────────────────────────────────────────────────────────────
# Dynamics
# ──────────────────────────────────────────────────────────────────────────────

class QuadVerticalDynamics(ControlAffineDynamics):
    """
    Reduced 4-state quadrotor dynamics.
    State:   [y, vy, phi, phidot]
    Control: [T1, T2]  (thrust of left / right rotor)
    """
    STATES        = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS      = ["T1", "T2"]
    PERIODIC_DIMS = [2]   # phi wraps at ±π

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
        B[..., 1, :] =  np.cos(state[..., 2]) / self.mass   # both rotors → vertical lift
        B[..., 3, 0] = -self.length / self.Iyy               # T1 → nose-down torque
        B[..., 3, 1] =  self.length / self.Iyy               # T2 → nose-up torque
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
    """
    JAX reimplementation of the 4-state dynamics.
    Required by hj_reachability, which JIT-compiles these methods.

    FIX: All array ops use jnp, not np, so JAX can trace/differentiate them.
    """

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
            [0.0,                    0.0],
            [cos_phi / self.mass,    cos_phi / self.mass],
            [0.0,                    0.0],
            [-self.length / self.Iyy, self.length / self.Iyy],
        ])

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(4), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        # FIX: must use jnp.sin here — np.sin breaks JAX tracing
        return jnp.array([
            [0, 1, 0, 0],
            [0, -self.Cd_v / self.mass,
               -1 / self.mass * (control[0] + control[1]) * jnp.sin(state[2]),
               0],
            [0, 0, 0, 1],
            [0, 0, 0, -self.Cd_phi / self.Iyy],
        ])


class QuadPlanarDynamics(QuadVerticalDynamics):
    """
    Extended 6-state planar quadrotor dynamics.
    State:   [x, vx, y, vy, phi, phidot]
    Control: [T1, T2]

    The [y, vy, phi, phidot] sub-block reuses the 4-state parent.
    """
    STATES        = ["X", "XDOT", "Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS      = ["T1", "T2"]
    PERIODIC_DIMS = [4]   # phi is index 4 in the 6-state vector

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] =  state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1]      # x-drag, no gravity
        f[..., 2:] = super().open_loop_dynamics(state[..., 2:], time)
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, :] = -np.sin(state[..., 4]) / self.mass       # horizontal thrust
        B[..., 2:, :] = super().control_matrix(state[..., 2:], time)
        return B

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] =  1
        J[..., 1, 1] = -self.Cd_v / self.mass
        # d/d_phi [ -sin(phi)/m * (T1+T2) ] = -cos(phi)/m * (T1+T2)
        J[..., 1, 4] = -(control[..., 0] + control[..., 1]) * np.cos(state[..., 4]) / self.mass
        J[..., 2:, 2:] = super().state_jacobian(state[..., 2:], control, time)
        return J


# ──────────────────────────────────────────────────────────────────────────────
# Safe set / constraint function
# ──────────────────────────────────────────────────────────────────────────────

def safe_set(state):
    """
    Constraint function ℓ(x).  ℓ(x) ≥ 0  iff  x ∈ L.

    L = [1, 9] × [−6, 6] × ℝ × [−8, 8]   (y, vy, phi, omega)

    Accepts an *unbatched* 4-vector [y, vy, phi, omega].
    phi has no hard constraint — it is never the binding constraint here.
    """
    return jnp.min(jnp.array([
        state[0] - 1,    # y ≥ 1   (floor clearance)
        9 - state[0],    # y ≤ 9   (ceiling clearance)
        state[1] + 6,    # vy ≥ −6
        6 - state[1],    # vy ≤  6
        state[3] + 8,    # omega ≥ −8
        8 - state[3],    # omega ≤  8
    ]))


# ──────────────────────────────────────────────────────────────────────────────
# Candidate CBFs
# ──────────────────────────────────────────────────────────────────────────────

class QuadVerticalCBF(ControlAffineCBF):
    """
    Expert-synthesised quadratic candidate CBF for the 4-state dynamics.

    h(x) = 10 − [ s₀(5−y)² + s₁·vy² + s₂·phi² + s₃·omega² ]

    Centred at the hover equilibrium (y=5, vy=phi=omega=0).
    This is NOT guaranteed to be a valid CBF everywhere — that is
    precisely the problem refineCBF solves.
    """

    def __init__(self, dynamics, params, **kwargs):
        self.scaling = params["scaling"]
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        s = self.scaling
        return 10 - (
            s[0] * (5 - state[..., 0]) ** 2 +
            s[1] * state[..., 1] ** 2 +
            s[2] * state[..., 2] ** 2 +
            s[3] * state[..., 3] ** 2
        )

    def vf_dt_partial(self, state, time=0.0):
        return 0.0   # time-invariant

    def _grad_vf(self, state, time=0.0):
        s = self.scaling
        return s * np.array([
             2 * (5 - state[..., 0]),
            -2 * state[..., 1],
            -2 * state[..., 2],
            -2 * state[..., 3],
        ]).T


class ExtendedQuadVerticalCBF(QuadVerticalCBF):
    """
    Lift the 4-state CBF into the 6-state space.
    The x and vx dimensions get zero gradient (safety is independent of them).
    """

    def vf(self, state, time=0):
        return super().vf(state[..., 2:], time)

    def _grad_vf(self, state, time=0):
        dV = np.zeros_like(state)
        dV[..., 2:] = super()._grad_vf(state[..., 2:], time)
        return dV


# ──────────────────────────────────────────────────────────────────────────────
# HJR grid construction (shared between both solves)
# ──────────────────────────────────────────────────────────────────────────────

def build_grid():
    """Return the 4-state HJR grid, solver settings, and initial safe values."""
    state_domain    = hj.sets.Box(lo=jnp.array([0.,  -8., -jnp.pi, -10.]),
                                   hi=jnp.array([10.,  8.,  jnp.pi,  10.]))
    grid_resolution = (31, 25, 41, 25)   # (y, vy, phi, omega)
    # Increase for higher fidelity if running on GPU

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        state_domain, grid_resolution, periodic_dims=2)

    safe_values = hj.utils.multivmap(safe_set, jnp.arange(4))(grid.states)

    brt_postprocessor = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    solver_settings   = hj.SolverSettings.with_accuracy(
        "high", value_postprocessor=brt_postprocessor(safe_values))

    return grid, solver_settings, safe_values


# ──────────────────────────────────────────────────────────────────────────────
# HJR solves
# ──────────────────────────────────────────────────────────────────────────────

def run_hjr_solves(dyn_jnp, grid, solver_settings, safe_values, cbf_params):
    """
    Run two CBVF solves:
      B_ell  — vanilla HJR, initialised with ℓ(x)
      B_h    — refineCBF,   initialised with the candidate CBF h(x)

    Returns (target_values_hjr, target_values_refine, quad_tabular_cbf)
    both as JAX arrays of shape (101, *grid_resolution).
    """
    dyn_hjr = HJControlAffineDynamics(dyn_jnp, control_space=hj.sets.Box(UMIN, UMAX))

    times = jnp.linspace(0., -5., 101)

    # ── Vanilla HJR ──────────────────────────────────────────────────────────
    print("\n[1/2] Running vanilla HJR solve (B_ell) ...")
    target_values_hjr = hj.solve(solver_settings, dyn_hjr, grid, times, safe_values)
    print("      Done. Shape:", target_values_hjr.shape)

    # ── refineCBF ────────────────────────────────────────────────────────────
    dyn_4state = QuadVerticalDynamics(PARAMS, test=True)
    cbf_4state = QuadVerticalCBF(dyn_4state, cbf_params, test=False)

    quad_tabular_cbf = TabularControlAffineCBF(dyn_4state, cbf_params, grid=grid)
    quad_tabular_cbf.tabularize_cbf(cbf_4state)
    print("\n[2/2] Running refineCBF solve (B_h, warm-started) ...")
    target_values_refine = hj.solve(
        solver_settings, dyn_hjr, grid, times, quad_tabular_cbf.vf_table)
    print("      Done. Shape:", target_values_refine.shape)

    # FIX: assign the *converged* value (last time slice = t = −5 s)
    # Also force real float64 — JAX can return complex-typed arrays even when
    # Im=0, and CVXPY's Parameter will reject them.
    quad_tabular_cbf.vf_table = np.real(np.array(target_values_refine[-1])).astype(float)

    return target_values_hjr, target_values_refine, quad_tabular_cbf


# ──────────────────────────────────────────────────────────────────────────────
# Lift CBF to 6-D
# ──────────────────────────────────────────────────────────────────────────────

def build_extended_cbf(quad_tabular_cbf, cbf_params):
    """
    Broadcast the converged 4-state tabular CBF into the 6-state grid by
    repeating along the (x, vx) axes.  Valid because safety constraints
    are independent of horizontal position and speed.

    FIX: The vf_table broadcast is done *after* the DP solve, not before.
    """
    extended_state_domain    = hj.sets.Box(
        lo=jnp.array([-30., -8.,  0., -8., -jnp.pi, -10.]),
        hi=jnp.array([ 30.,  8., 10.,  8.,  jnp.pi,  10.]))
    extended_grid_resolution = (5, 5, 31, 25, 41, 25)

    extended_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        extended_state_domain, extended_grid_resolution, periodic_dims=4)

    extended_dyn = QuadPlanarDynamics(PARAMS, test=False)
    quad_ext_tabular_cbf = TabularControlAffineCBF(extended_dyn, cbf_params,
                                                    grid=extended_grid)

    # (31,25,41,25) → (5,5,31,25,41,25)
    # FIX: take real part and cast to float64 — JAX arrays can carry a complex
    # dtype (Im=0) which CVXPY's Parameter rejects as non-real.
    vf_4d = np.real(quad_tabular_cbf.vf_table).astype(float)
    quad_ext_tabular_cbf.vf_table = np.repeat(
        np.repeat(vf_4d[None, ...], 5, axis=0)[None, ...],
        5, axis=0)

    vf = quad_ext_tabular_cbf.vf_table
    print(f"\nExtended vf_table shape : {vf.shape}")
    print(f"NaN: {np.isnan(vf).sum()}  Inf: {np.isinf(vf).sum()}")
    print(f"min={vf.min():.3f}  max={vf.max():.3f}")

    return extended_dyn, extended_grid, quad_ext_tabular_cbf


# ──────────────────────────────────────────────────────────────────────────────
# LQR nominal controller
# ──────────────────────────────────────────────────────────────────────────────

def build_lqr(extended_dyn):
    """
    Linearise the 6-state dynamics around the hover equilibrium and compute
    an infinite-time discrete-time LQR gain K.

    Returns K (6×2).
    """
    x_nom = np.array([15., 0., 3., 0., 0., 0.])
    u_nom = 0.5 * MASS * G * np.ones(2)

    A, B     = extended_dyn.linearized_ct_dynamics(x_nom, u_nom)
    A_d, B_d = extended_dyn.linearized_dt_dynamics(x_nom, u_nom)

    Q = np.diag([1.0, 0.1, 1.0, 0.1, 1.0, 1.0])
    R = np.eye(2)
    K = utils.lqr(A_d, B_d, Q, R)

    A_cl = A - B @ K
    eigs = np.linalg.eig(A_cl)[0]
    assert (eigs.real < 0).all(), f"Closed-loop system is UNSTABLE! eigs={eigs}"
    print("\nLQR gain computed.  Closed-loop eigenvalues:", np.round(eigs, 3))
    return K


def make_nominal_policy(K, u_ref, x_ref):
    """Return a clipped LQR feedback policy."""
    return lambda x, t: np.atleast_2d(
        np.clip(u_ref - (K @ (x - x_ref).T).T, UMIN, UMAX))


# ──────────────────────────────────────────────────────────────────────────────
# Rollout experiment
# ──────────────────────────────────────────────────────────────────────────────

def _patch_asif_for_real_values(asif):
    """
    Monkey-patch an ASIF instance so that set_constraint always receives
    real-valued scalars, preventing the CVXPY 'Parameter value must be real'
    error that occurs when JAX returns complex-typed arrays or when finite-
    difference gradients on the tabular CBF produce NaN/Inf near grid boundaries.

    Root cause: hj_reachability stores value functions as JAX arrays which
    can have complex dtype (e.g. complex64) even when the imaginary part is 0.
    Additionally, interpolation near grid edges sometimes returns NaN.  Both
    make CVXPY's Parameter reject the value.
    """
    original_set_constraint = asif.set_constraint

    def safe_set_constraint(Lf_h, Lg_h, h):
        # Force real-valued numpy scalars / arrays
        Lf_h = float(np.real(Lf_h))
        Lg_h = np.real(np.atleast_1d(Lg_h)).astype(float)
        h    = float(np.real(h))

        # Guard against NaN/Inf which also break OSQP
        if not np.isfinite(Lf_h):
            Lf_h = 0.0
        if not np.all(np.isfinite(Lg_h)):
            Lg_h = np.zeros_like(Lg_h)
        if not np.isfinite(h):
            h = -1.0   # treat as unsafe

        return original_set_constraint(Lf_h, Lg_h, h)

    asif.set_constraint = safe_set_constraint
    return asif


def run_scenario(extended_dyn, quad_extended_cbf, quad_ext_tabular_cbf,
                 K, x0, x_goal, label, out_dir):
    """
    Simulate three controllers (Nominal / Analytical CBF / CBVF) from x0
    and save the trajectory CSV.

    Returns results_df.
    """
    u_hover     = 0.5 * MASS * G * np.ones(2)
    alpha       = lambda x: 5 * x           # class-K linear extension, γ=5

    nom_control = make_nominal_policy(K, u_hover, x_goal)
    cbf_asif    = ControlAffineASIF(extended_dyn, quad_extended_cbf,
                                     alpha=alpha, nominal_policy=nom_control,
                                     umin=UMIN, umax=UMAX)
    cbvf_asif   = ControlAffineASIF(extended_dyn, quad_ext_tabular_cbf,
                                     alpha=alpha, nominal_policy=nom_control,
                                     umin=UMIN, umax=UMAX)

    # FIX: patch both ASIF instances to sanitize complex/NaN values before
    # they reach CVXPY's Parameter, which requires strictly real floats.
    cbf_asif  = _patch_asif_for_real_values(cbf_asif)
    cbvf_asif = _patch_asif_for_real_values(cbvf_asif)

    print(f"\nRunning rollout: {label}  x0={x0}  x_goal={x_goal}")
    experiment = RolloutTrajectory("quad", start_x=x0, n_sims_per_start=1, t_sim=8)
    results_df = experiment.run(extended_dyn,
                                 {"Nominal":    nom_control,
                                  "Analytical": cbf_asif,
                                  "CBVF":       cbvf_asif})

    csv_path = os.path.join(out_dir, f"{label}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return results_df, nom_control, cbf_asif, cbvf_asif


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def fig_trajectories(extended_dyn, results_df, x0, x_goal, label, out_dir):
    """x–y state-space trajectory plot (Fig. 4 style)."""
    ss_exp = StateSpaceExperiment("quad", x_indices=[0, 2],
                                   start_x=x0, n_sims_per_start=1, t_sim=8)
    ss_exp.plot(extended_dyn, results_df)
    plt.gca().plot(*x0[[0, 2]],    "x", markersize=14, mew=3, color="grey",
                   label="Start")
    plt.gca().plot(*x_goal[[0, 2]], "o", markersize=14, color="grey",
                   label="Goal")
    plt.legend(fontsize=11)
    plt.title(f"Trajectories — {label}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_trajectories_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def fig_safe_set_evolution(grid, safe_values, target_values, case_number, out_dir):
    """
    Zero-level-set contour evolution of the refineCBF value function.
    Replicates Figure 1 / Figure 4 (left) from the paper.

    case_number:
        1 — initial candidate CBF only
        2 — initial + one intermediate snapshot
        3 — initial + two intermediates + converged (full evolution)
    """
    # phi–y slice at roughly zero vy and omega
    offset_vy    = 12   # mid-index of the vy   dimension (25 pts → index 12 ≈ 0 m/s)
    offset_omega = 12   # mid-index of the omega dimension (25 pts → index 12 ≈ 0 rad/s)

    vf = np.array(target_values)   # (101, 31, 25, 41, 25)

    palette = np.array([
        (0.3,  0.3,  0.3),
        sns.color_palette("RdYlGn_r", 7)[0],
        sns.color_palette("RdYlGn_r", 9)[6],
        sns.color_palette("RdYlGn_r", 9)[8],
        (4/255, 101/255, 4/255),
    ])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Unsafe region (complement of L)
    ax.contourf(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                safe_values[:, offset_vy, :, offset_omega],
                levels=[-10, 0], colors=[palette[0]], alpha=0.6)

    def draw_contour(time_idx, color, lw=3, alpha=1.0):
        ax.contour(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                   vf[time_idx][:, offset_vy, :, offset_omega],
                   levels=[0], colors=[color], linewidths=lw, alpha=alpha)

    if case_number == 1:
        draw_contour(0,  palette[3])

    elif case_number == 2:
        draw_contour(0,  palette[0], alpha=0.3)
        draw_contour(3,  palette[2])

    elif case_number == 3:
        draw_contour(0,   palette[0], alpha=0.6)
        draw_contour(3,   palette[0], alpha=0.4)
        draw_contour(10,  palette[0], alpha=0.2)
        draw_contour(-1,  palette[-1], alpha=1.0)

    ax.set_xlabel(r"$\phi$ [rad]", fontsize=14)
    ax.set_ylabel(r"$y$ [m]",      fontsize=14)
    ax.set_title(f"Safe-set evolution (case {case_number})\n"
                 r"$\phi$–$y$ slice at $\dot{y}\approx0$, $\omega\approx0$",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_safe_set_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def fig_value_function_comparison(grid, target_values_hjr, target_values_refine, out_dir):
    """
    Side-by-side comparison of converged B_ell vs B_h value functions.
    Theorem 1 guarantees B_h ≤ B_ell pointwise.
    """
    offset_vy    = 12
    offset_omega = 12

    vf_hjr    = np.array(target_values_hjr[-1])
    vf_refine = np.array(target_values_refine[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vf, title in zip(axes,
                               [vf_hjr,    vf_refine],
                               ["Vanilla HJR $B_\\ell$", "refineCBF $B_h$"]):
        im = ax.contourf(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                         vf[:, offset_vy, :, offset_omega], levels=30, cmap="RdYlGn")
        ax.contour(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                   vf[:, offset_vy, :, offset_omega], levels=[0],
                   colors="k", linewidths=2)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\phi$ [rad]", fontsize=13)
        ax.set_ylabel(r"$y$ [m]",      fontsize=13)
        ax.set_title(title, fontsize=14)

    plt.suptitle(r"Converged value functions ($\phi$–$y$ slice, "
                 r"$\dot{y}\approx0$, $\omega\approx0$)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_value_functions_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    diff = vf_refine - vf_hjr
    print(f"\nValue-function difference B_h − B_ell (should be ≤ 0 everywhere by Theorem 1):")
    print(f"  Mean: {diff.mean():.4f}   Std: {diff.std():.4f}")
    print(f"  Min:  {diff.min():.4f}   Max: {diff.max():.4f}")
    if diff.max() > 1e-6:
        print("  WARNING: max > 0, small numerical errors from grid interpolation.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Replicate the refineCBF quadrotor experiment.")
    p.add_argument("--case", choices=["conservative", "invalid", "both"],
                   default="both",
                   help="Which simulation scenario to run (default: both).")
    p.add_argument("--sets-case", type=int, choices=[1, 2, 3], default=3,
                   help="Safe-set evolution plot variant (default: 3).")
    p.add_argument("--out-dir", default="results",
                   help="Directory for all output files (default: ./results/).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  refineCBF Quadrotor Replication")
    print("  Python", sys.version.split()[0])
    print("=" * 60)

    # ── CBF params (shared) ───────────────────────────────────────────────────
    cbf_params = {"scaling": np.array([0.75, 0.5, 2.0, 0.5])}

    # ── Dynamics instances ────────────────────────────────────────────────────
    dyn_jnp      = QuadVerticalDynamicsJNP(PARAMS, test=True)

    # ── Grid + HJR solves ─────────────────────────────────────────────────────
    print("\nBuilding HJR grid ...")
    grid, solver_settings, safe_values = build_grid()
    print(f"Grid shape: {grid.states.shape[:-1]}")

    target_values_hjr, target_values_refine, quad_tabular_cbf = run_hjr_solves(
        dyn_jnp, grid, solver_settings, safe_values, cbf_params)

    # Save value functions
    np.save(os.path.join(args.out_dir, "target_values_hjr.npy"),
            np.array(target_values_hjr))
    np.save(os.path.join(args.out_dir, "target_values_refine.npy"),
            np.array(target_values_refine))
    print(f"\nValue functions saved to {args.out_dir}/")

    # ── Extend CBF to 6-D (FIX: done AFTER the DP solve) ─────────────────────
    extended_dyn, extended_grid, quad_ext_tabular_cbf = build_extended_cbf(
        quad_tabular_cbf, cbf_params)

    # 6-state analytical (candidate) CBF for comparison
    quad_extended_cbf = ExtendedQuadVerticalCBF(extended_dyn, cbf_params, test=False)

    # ── LQR nominal controller ────────────────────────────────────────────────
    K = build_lqr(extended_dyn)

    # ── Simulation scenarios ──────────────────────────────────────────────────
    run_conservative = args.case in ("conservative", "both")
    run_invalid      = args.case in ("invalid", "both")

    if run_conservative:
        # Conservative case: analytical CBF is overly conservative.
        # Initial condition: tilted, moving fast toward the ceiling.
        x0_con     = np.array([0.,  4., 7., -2., -np.pi/4, 0.])
        x_goal_con = np.array([6.,  0., 9.,  0.,  0.,      0.])

        results_con, *_ = run_scenario(
            extended_dyn, quad_extended_cbf, quad_ext_tabular_cbf,
            K, x0_con, x_goal_con, "conservative", args.out_dir)

        print("\nPlotting conservative-case trajectories ...")
        fig_trajectories(extended_dyn, results_con,
                         x0_con, x_goal_con, "conservative", args.out_dir)

    if run_invalid:
        # Invalid-candidate case: analytical CBF is invalid (declares unsafe
        # states safe near the floor at high positive velocity).
        x0_inv     = np.array([15., -3., 2.5, -2.,  np.pi/4, 1.])
        x_goal_inv = np.array([ 0.,  0., 1.5,  0.,  0.,      0.])

        results_inv, *_ = run_scenario(
            extended_dyn, quad_extended_cbf, quad_ext_tabular_cbf,
            K, x0_inv, x_goal_inv, "invalid", args.out_dir)

        print("\nPlotting invalid-case trajectories ...")
        fig_trajectories(extended_dyn, results_inv,
                         x0_inv, x_goal_inv, "invalid", args.out_dir)

    # ── Safe-set evolution figure ─────────────────────────────────────────────
    print("\nGenerating safe-set evolution figure ...")
    fig_safe_set_evolution(grid, safe_values, target_values_refine,
                           args.sets_case, args.out_dir)

    # ── Value-function comparison figure ─────────────────────────────────────
    print("\nGenerating value-function comparison figure ...")
    fig_value_function_comparison(grid, target_values_hjr,
                                   target_values_refine, args.out_dir)

    print("\n" + "=" * 60)
    print(f"  All outputs written to: {os.path.abspath(args.out_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()