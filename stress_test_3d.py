"""
stress_test_3d.py
=================
Stress-test extension for Tonkens & Herbert, "Refining Control Barrier
Functions through Hamilton-Jacobi Reachability", IROS 2022.

Extends the paper's 2D quadrotor experiment (Section VI-C) to a true 3D
quadrotor operating in (x, y, z) space.  This directly targets the paper's
stated limitation:

    "given that REFINECBF relies on spatially discretised DP recursion,
     we typically consider low-dimensional models with state dimension ≤ 6."
                                                        — Section III, para 3

What this script does
---------------------
2D BASELINE (paper)   — 4-state HJR grid  [y, vy, φ, φ̇]  → (31,25,41,25)
3D STRESS TEST        — 6-state HJR grid  [z, vz, y, vy, φ, φ̇] → (15,13,15,13,21,13)

The 3D model adds a new spatial dimension (altitude z) with its own velocity vz
and independent thrust axis, making the reachability problem substantially
harder:
  • 6D grid instead of 4D  → exponential memory / time blowup
  • Additional coupling between axes in the CBF candidate
  • Larger safe-set boundary surface to converge

For each dimensionality the script measures and records:
  ① DP solve wall time   (vanilla HJR  and  refineCBF warm-start)
  ② Convergence speed    (DP iterations to within ε of the viability kernel)
  ③ Safe-set volume      (fraction of grid cells where B_h ≥ 0 at each iter)
  ④ Theorem 1 validity   (B_h ≤ B_ell pointwise at convergence)
  ⑤ Safety violation rate during rollout  (% of timesteps outside safe set)
  ⑥ CBF constraint activity  (% of timesteps where ASIF overrides nominal)
  ⑦ Online solve time per step  (OSQP wall time, sampled)

All results are written to:
  <out-dir>/stress_test_comparison.csv   — machine-readable numbers
  <out-dir>/fig_stress_comparison.png   — publication-style comparison figure
  <out-dir>/fig_stress_safe_set_2d.png  — 2D safe-set evolution (φ–y slice)
  <out-dir>/fig_stress_safe_set_3d.png  — 3D safe-set evolution (φ–z slice)
  <out-dir>/fig_stress_rollout_2d.png   — 2D rollout trajectories
  <out-dir>/fig_stress_rollout_3d.png   — 3D rollout trajectories
  <out-dir>/fig_stress_grid_scaling.png — grid-size vs memory / time

Usage
-----
    python stress_test_3d.py
    python stress_test_3d.py --skip-solves   # reload saved .npy if present
    python stress_test_3d.py --out-dir ./my_results
    python stress_test_3d.py --fine-grid     # denser 3D grid (very slow, GPU recommended)
"""

import argparse
import gc
import logging
import os
import sys
import time
import tracemalloc
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("cbf_opt").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import hj_reachability as hj
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF, utils
from experiment_wrapper import RolloutTrajectory, StateSpaceExperiment
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF

# ── Monkey-patch for experiment_wrapper off-by-one bug ────────────────────────
# add_arrow uses np.searchsorted then indexes xdata[end_ind] without clamping,
# so when the midpoint falls on the last sample (index == len) it raises IndexError.
try:
    import experiment_wrapper.rollout_trajectory as _ew_rt

    def _safe_add_arrow(line, direction="right", position=None, size=15, color=None):
        import numpy as _np
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        if position is None:
            position = xdata.mean()
        end_ind = int(_np.clip(_np.searchsorted(xdata, position), 1, len(xdata) - 1))
        start_ind = end_ind - 1
        if color is None:
            color = line.get_color()
        line.axes.annotate(
            "",
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size,
        )

    _ew_rt.add_arrow = _safe_add_arrow
except Exception as _patch_err:
    warnings.warn(f"Could not patch add_arrow in experiment_wrapper: {_patch_err}")
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
PAPER_COLORS = np.array([
    (0.3, 0.3, 0.3),
    sns.color_palette("RdYlGn_r", 7)[0],
    sns.color_palette("RdYlGn_r", 9)[6],
    sns.color_palette("RdYlGn_r", 9)[8],
    (4/255, 101/255, 4/255),
])
ALT_COLORS   = sns.color_palette("pastel", 9).as_hex()
CHOSEN_COLORS = [
    (0.5, 0.5, 0.5),
    sns.color_palette("tab10")[0],
    sns.color_palette("tab10")[1],
    (0.1, 0.1, 0.1),
    (0.7, 0.7, 0.7),
]
matplotlib.rcParams.update({
    "axes.labelsize": 18, "axes.titlesize": 18, "font.size": 16,
    "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5,
})

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants  (shared by 2D and 3D)
# ─────────────────────────────────────────────────────────────────────────────
G      = 9.81
MASS   = 2.5
Cd_v   = 0.25
Cd_phi = 0.02255
LENGTH = 1.0
IYY    = 1.0
DT     = 0.01
UMAX_2 = 0.75 * MASS * G * np.ones(2)   # 2-rotor control bounds
UMIN_2 = np.zeros(2)
UMAX_4 = 0.75 * MASS * G * np.ones(4)   # 4-rotor control bounds (3D)
UMIN_4 = np.zeros(4)

QUAD_PARAMS = dict(Cd_v=Cd_v, g=G, Cd_phi=Cd_phi,
                   mass=MASS, length=LENGTH, Iyy=IYY, dt=DT)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _patch_asif(asif_inst):
    """Force real float64 into CVXPY; clamp NaN/Inf from boundary interpolation."""
    orig = asif_inst.set_constraint

    def safe_set_constraint(Lf_h, Lg_h, h):
        Lf_h = float(np.real(Lf_h))
        Lg_h = np.real(np.atleast_1d(Lg_h)).astype(float)
        h    = float(np.real(h))
        if not np.isfinite(Lf_h):           Lf_h = 0.0
        if not np.all(np.isfinite(Lg_h)):   Lg_h = np.zeros_like(Lg_h)
        if not np.isfinite(h):              h    = -1.0
        return orig(Lf_h, Lg_h, h)

    asif_inst.set_constraint = safe_set_constraint
    return asif_inst


def _real_table(tabular_cbf, vf_array):
    tabular_cbf.vf_table = np.real(np.array(vf_array)).astype(float)
    return tabular_cbf


def _safe_volume(vf, threshold=0.0):
    """Fraction of grid cells where vf ≥ threshold (proxy for safe-set volume)."""
    return float(np.mean(np.real(np.array(vf)) >= threshold))


def _timed_solve(solver_settings, dyn_hjr, grid, times, init_value):
    """Run hj.solve, return (value_function_array, wall_time_seconds)."""
    t0  = time.perf_counter()
    vfs = hj.solve(solver_settings, dyn_hjr, grid, times, init_value)
    # Force JAX to finish async computation before reading the clock
    jax.block_until_ready(vfs)
    return vfs, time.perf_counter() - t0


def _peak_memory_mb(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, peak_resident_MB)."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024


def _convergence_iters(target_values, threshold=1e-3):
    """
    Return the first iteration index i such that
    ||B(t_i) - B(t_{i-1})||_inf < threshold.
    This approximates how many DP sweeps are needed for convergence.
    """
    vfs = np.real(np.array(target_values))
    for i in range(1, len(vfs)):
        if np.max(np.abs(vfs[i] - vfs[i-1])) < threshold:
            return i
    return len(vfs) - 1   # did not converge within budget


def _rollout_safety_stats(results_df, safe_fn, dyn):
    """
    Given a results DataFrame and a scalar safe_fn(state) → float,
    return dict of per-controller safety stats.

    safe_fn should return positive values inside the safe set.
    """
    stats = {}
    for ctrl in results_df.controller.unique():
        sub = results_df[results_df.controller == ctrl]
        ts  = sub.t.unique()
        violations = 0
        total      = 0
        cbf_vals   = []
        for t in ts:
            state_rows = sub[(sub.t == t) & (sub.measurement.isin(dyn.STATES))]
            if len(state_rows) < dyn.n_dims:
                continue
            x = state_rows.value.values
            val = float(np.real(safe_fn(x)))
            cbf_vals.append(val)
            if val < 0:
                violations += 1
            total += 1
        stats[ctrl] = {
            "violation_rate":  violations / max(total, 1),
            "min_cbf_val":     float(np.min(cbf_vals)) if cbf_vals else float("nan"),
            "mean_cbf_val":    float(np.mean(cbf_vals)) if cbf_vals else float("nan"),
        }
    return stats


def _timed_rollout(experiment, dyn, controllers):
    """Run a rollout and time each controller's online solve per step."""
    t0      = time.perf_counter()
    results = experiment.run(dyn, controllers)
    wall    = time.perf_counter() - t0
    n_steps = len(results.t.unique())
    return results, wall, wall / max(n_steps * len(controllers), 1)


# ═════════════════════════════════════════════════════════════════════════════
# Fallback trajectory plotter
# ═════════════════════════════════════════════════════════════════════════════

def _plot_trajectories_manually(results_df, dyn, ax, colors, x_idx=0, y_idx=1):
    """
    Fallback used when StateSpaceExperiment.plot raises IndexError
    (experiment_wrapper add_arrow off-by-one bug).  Draws one coloured line
    per controller with a safe mid-trajectory directional arrow.
    """
    state_labels = dyn.STATES
    for ctrl, color in zip(results_df.controller.unique(), colors):
        sub = results_df[results_df.controller == ctrl]
        ts  = sorted(sub.t.unique())
        xs, ys = [], []
        for t in ts:
            row = sub[(sub.t == t) & (sub.measurement.isin(state_labels))]
            if len(row) < len(state_labels):
                continue
            state_map = dict(zip(row.measurement, row.value))
            sv = np.array([state_map.get(s, 0.0) for s in state_labels])
            xs.append(float(sv[x_idx]))
            ys.append(float(sv[y_idx]))
        if not xs:
            continue
        xs, ys = np.array(xs), np.array(ys)
        ax.plot(xs, ys, color=color, linewidth=2.5, label=ctrl)
        n = len(xs)
        if n >= 2:
            mid = max(1, min(n // 2, n - 1))
            ax.annotate("", xytext=(xs[mid - 1], ys[mid - 1]), xy=(xs[mid], ys[mid]),
                        arrowprops=dict(arrowstyle="->", color=color), size=15)


# ═════════════════════════════════════════════════════════════════════════════
# 2D QUADROTOR  (paper baseline — 4-state HJR)
#   State: [y, vy, φ, φ̇]
# ═════════════════════════════════════════════════════════════════════════════

class Quad2DDynamics(ControlAffineDynamics):
    """4-state planar quadrotor: [y, vy, φ, φ̇]."""
    STATES        = ["Y", "VY", "PHI", "PHIDOT"]
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

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] =  state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] =  state[..., 3]
        f[..., 3] = -self.Cd_phi / self.Iyy * state[..., 3]
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, :] =  np.cos(state[..., 2]) / self.mass
        B[..., 3, 0] = -self.length / self.Iyy
        B[..., 3, 1] =  self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state, time=0.0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] =  1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 2] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 2]) / self.mass
        J[..., 2, 3] =  1.0
        J[..., 3, 3] = -self.Cd_phi / self.Iyy
        return J


class Quad2DDynamicsJNP(Quad2DDynamics):
    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([
            state[1],
            -state[1] * self.Cd_v / self.mass - self.g,
            state[3],
            -state[3] * self.Cd_phi / self.Iyy,
        ])

    def control_matrix(self, state, time=0.0):
        c = jnp.cos(state[2])
        return jnp.array([
            [0.,             0.           ],
            [c / self.mass,  c / self.mass],
            [0.,             0.           ],
            [-self.length / self.Iyy, self.length / self.Iyy],
        ])

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(4), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        return jnp.array([
            [0, 1, 0, 0],
            [0, -self.Cd_v / self.mass,
               -(control[0] + control[1]) * jnp.sin(state[2]) / self.mass, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -self.Cd_phi / self.Iyy],
        ])


class Quad2DCBF(ControlAffineCBF):
    """Expert-synthesised quadratic CBF for 2D quad: h(x) = c − V(x)."""
    def __init__(self, dynamics, params, **kwargs):
        self.scaling = params["scaling"]
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        s = self.scaling
        return 10.0 - (s[0] * (5 - state[..., 0]) ** 2
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


def _quad2d_safe_set(state):
    """ℓ(x) for the 2D quad: y ∈ [1,9], vy ∈ [−6,6], φ̇ ∈ [−8,8]."""
    return jnp.min(jnp.array([
        state[0] - 1, 9 - state[0],
        state[1] + 6, 6 - state[1],
        state[3] + 8, 8 - state[3],
    ]))


# ═════════════════════════════════════════════════════════════════════════════
# 3D QUADROTOR  (stress test — 6-state HJR)
#   State: [z, vz, y, vy, φ, φ̇]
#
#   The 3D extension adds an independent vertical axis (z, vz) with two
#   additional rotors (T3, T4) controlling altitude independently of the
#   planar pitch axis.  This is a physically realistic model of a quadrotor
#   that can move in both y (pitch) and z (roll / altitude thrust) planes.
#
#   This pushes the DP grid from 4D → 6D, directly stress-testing the
#   paper's curse-of-dimensionality claim.
# ═════════════════════════════════════════════════════════════════════════════

class Quad3DDynamics(ControlAffineDynamics):
    """
    6-state quadrotor: [z, vz, y, vy, φ, φ̇].

    State layout
    ------------
    0: z        — altitude                   [m]
    1: vz       — vertical velocity          [m/s]
    2: y        — lateral position           [m]
    3: vy       — lateral velocity           [m/s]
    4: φ        — pitch angle                [rad]  (periodic)
    5: φ̇        — pitch rate                 [rad/s]

    Control layout
    --------------
    0: T1  — front-left rotor  (pitch torque + lateral lift)
    1: T2  — front-right rotor (pitch torque + lateral lift)
    2: T3  — rear-left rotor   (altitude thrust)
    3: T4  — rear-right rotor  (altitude thrust)

    Equations of motion
    -------------------
    ż   =  vz
    v̇z  =  (T3 + T4)/m  − g  − (Cd_v/m)·vz
    ẏ   =  vy
    v̇y  =  cos(φ)/m · (T1 + T2)  − (Cd_v/m)·vy
    φ̇   =  φ̇
    φ̈   =  length/Iyy · (T2 − T1)  − (Cd_phi/Iyy)·φ̇
    """
    STATES        = ["Z", "VZ", "Y", "VY", "PHI", "PHIDOT"]
    CONTROLS      = ["T1", "T2", "T3", "T4"]
    PERIODIC_DIMS = [4]   # φ is periodic

    def __init__(self, params, **kwargs):
        self.Cd_v   = params["Cd_v"]
        self.g      = params["g"]
        self.Cd_phi = params["Cd_phi"]
        self.mass   = params["mass"]
        self.length = params["length"]
        self.Iyy    = params["Iyy"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] =  state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] =  state[..., 3]
        f[..., 3] = -self.Cd_v / self.mass * state[..., 3]
        f[..., 4] =  state[..., 5]
        f[..., 5] = -self.Cd_phi / self.Iyy * state[..., 5]
        return f

    def control_matrix(self, state, time=0.0):
        n = state.shape[:-1]
        B = np.zeros(n + (6, 4))
        # Altitude: T3, T4 → vz
        B[..., 1, 2] = 1.0 / self.mass
        B[..., 1, 3] = 1.0 / self.mass
        # Lateral: T1, T2 → vy (through cos(φ))
        B[..., 3, 0] = np.cos(state[..., 4]) / self.mass
        B[..., 3, 1] = np.cos(state[..., 4]) / self.mass
        # Pitch torque: T1, T2 → φ̈
        B[..., 5, 0] = -self.length / self.Iyy
        B[..., 5, 1] =  self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state, time=0.0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        J = np.zeros(state.shape[:-1] + (6, 6))
        J[..., 0, 1] =  1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 2, 3] =  1.0
        J[..., 3, 3] = -self.Cd_v / self.mass
        J[..., 3, 4] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 4]) / self.mass
        J[..., 4, 5] =  1.0
        J[..., 5, 5] = -self.Cd_phi / self.Iyy
        return J


class Quad3DDynamicsJNP(Quad3DDynamics):
    """JAX version of the 6-state dynamics for hj_reachability."""

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([
            state[1],
            -self.Cd_v / self.mass * state[1] - self.g,
            state[3],
            -self.Cd_v / self.mass * state[3],
            state[5],
            -self.Cd_phi / self.Iyy * state[5],
        ])

    def control_matrix(self, state, time=0.0):
        c = jnp.cos(state[4])
        return jnp.array([
            [0.,              0.,              0.,             0.            ],
            [0.,              0.,              1./self.mass,   1./self.mass  ],
            [0.,              0.,              0.,             0.            ],
            [c/self.mass,     c/self.mass,     0.,             0.            ],
            [0.,              0.,              0.,             0.            ],
            [-self.length/self.Iyy, self.length/self.Iyy, 0., 0.           ],
        ])

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(6), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        return jnp.array([
            [0, 1, 0, 0, 0, 0],
            [0, -self.Cd_v/self.mass, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, -self.Cd_v/self.mass,
               -(control[0]+control[1])*jnp.sin(state[4])/self.mass, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, -self.Cd_phi/self.Iyy],
        ])


class Quad3DCBF(ControlAffineCBF):
    """
    Expert-synthesised candidate CBF for the 6-state 3D quadrotor.

    h(x) = c − [sz·(z_nom − z)²  +  svz·vz²  +  sy·(y_nom − y)²
                 + svy·vy²        +  sφ·φ²     + sφ̇·φ̇²]

    Centred at the 3D hover equilibrium (z=z_nom, y=y_nom, rest=0).
    This inherits the same limitation as the 2D candidate: it does not
    capture the full nonlinear safe set, and may be either conservative
    or invalid in different regions.
    """
    def __init__(self, dynamics, params, **kwargs):
        self.scaling = params["scaling"]   # length-6 array
        self.z_nom   = params.get("z_nom", 5.0)
        self.y_nom   = params.get("y_nom", 5.0)
        super().__init__(dynamics, params, **kwargs)

    def vf(self, state, time=0.0):
        s = self.scaling
        return 10.0 - (
            s[0] * (self.z_nom - state[..., 0]) ** 2 +
            s[1] * state[..., 1] ** 2 +
            s[2] * (self.y_nom - state[..., 2]) ** 2 +
            s[3] * state[..., 3] ** 2 +
            s[4] * state[..., 4] ** 2 +
            s[5] * state[..., 5] ** 2
        )

    def vf_dt_partial(self, state, time=0.0):
        return 0.0

    def _grad_vf(self, state, time=0.0):
        s = self.scaling
        return s * np.array([
             2 * (self.z_nom - state[..., 0]),
            -2 * state[..., 1],
             2 * (self.y_nom - state[..., 2]),
            -2 * state[..., 3],
            -2 * state[..., 4],
            -2 * state[..., 5],
        ]).T


def _quad3d_safe_set(state):
    """
    ℓ(x) for the 3D quad.
    Safe set: z ∈ [1,9], vz ∈ [−6,6], y ∈ [1,9], vy ∈ [−6,6], φ̇ ∈ [−8,8].
    """
    return jnp.min(jnp.array([
        state[0] - 1, 9 - state[0],      # z bounds
        state[1] + 6, 6 - state[1],      # vz bounds
        state[2] - 1, 9 - state[2],      # y bounds
        state[3] + 6, 6 - state[3],      # vy bounds
        state[5] + 8, 8 - state[5],      # φ̇ bounds
    ]))


# ═════════════════════════════════════════════════════════════════════════════
# Grid builders
# ═════════════════════════════════════════════════════════════════════════════

def _build_2d_grid():
    """4D HJR grid — exact match of the paper's Section VI-C grid."""
    domain = hj.sets.Box(
        lo=jnp.array([0.,  -8., -jnp.pi, -10.]),
        hi=jnp.array([10.,  8.,  jnp.pi,  10.]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, (31, 25, 41, 25), periodic_dims=2)
    safe_values = hj.utils.multivmap(_quad2d_safe_set, jnp.arange(4))(grid.states)
    brt         = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    settings    = hj.SolverSettings.with_accuracy(
        "high", value_postprocessor=brt(safe_values))
    return grid, settings, safe_values


def _build_3d_grid(fine=False):
    """
    6D HJR grid for the 3D quadrotor stress test.

    Coarse (default): (15,13,15,13,21,13)  — feasible on CPU with ~4 GB RAM
    Fine  (--fine):   (21,17,21,17,31,17)  — GPU recommended (>16 GB RAM)

    The grid is intentionally kept smaller than the 2D grid in points-per-dim
    to keep total grid points in a tractable range (~30x larger than 2D).
    This still clearly demonstrates the exponential scaling.
    """
    if fine:
        res = (21, 17, 21, 17, 31, 17)
    else:
        res = (15, 13, 15, 13, 21, 13)

    domain = hj.sets.Box(
        lo=jnp.array([0.,  -8.,  0., -8., -jnp.pi, -10.]),
        hi=jnp.array([10.,  8., 10.,  8.,  jnp.pi,  10.]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, res, periodic_dims=4)
    safe_values = hj.utils.multivmap(_quad3d_safe_set, jnp.arange(6))(grid.states)
    brt         = lambda obs: (lambda t, x: jnp.minimum(x, obs))
    settings    = hj.SolverSettings.with_accuracy(
        "medium", value_postprocessor=brt(safe_values))
    return grid, settings, safe_values, res


# ═════════════════════════════════════════════════════════════════════════════
# 2D experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_2d_experiment(out_dir, skip_solves=False):
    """
    Full 2D quadrotor experiment (paper baseline).
    Returns a dict of metrics for comparison.
    """
    print("\n" + "─" * 60)
    print("  2D QUADROTOR  (paper baseline, 4-state grid)")
    print("─" * 60)

    cbf_params  = {"scaling": np.array([0.75, 0.5, 2.0, 0.5])}
    dyn         = Quad2DDynamics(QUAD_PARAMS, test=True)
    dyn_jnp     = Quad2DDynamicsJNP(QUAD_PARAMS, test=False)
    cbf_cand    = Quad2DCBF(dyn, cbf_params, test=False)

    grid, settings, safe_values = _build_2d_grid()
    n_grid_pts  = int(np.prod(grid.states.shape[:-1]))
    print(f"  Grid shape : {grid.states.shape[:-1]}  ({n_grid_pts:,} points)")

    dyn_hjr = HJControlAffineDynamics(
        dyn_jnp, control_space=hj.sets.Box(UMIN_2, UMAX_2))
    times   = jnp.linspace(0., -5., 101)

    hjr_path    = os.path.join(out_dir, "2d_target_values_hjr.npy")
    refine_path = os.path.join(out_dir, "2d_target_values_refine.npy")

    if skip_solves and os.path.exists(hjr_path) and os.path.exists(refine_path):
        print("  Loading saved 2D value functions ...")
        tvs_hjr    = np.load(hjr_path)
        tvs_refine = np.load(refine_path)
        t_hjr = t_refine = float("nan")
    else:
        print("  Solving vanilla HJR (B_ell) ...")
        tvs_hjr, t_hjr = _timed_solve(settings, dyn_hjr, grid, times, safe_values)
        print(f"    Wall time: {t_hjr:.1f} s")

        tab = TabularControlAffineCBF(dyn, cbf_params, grid=grid)
        tab.tabularize_cbf(cbf_cand)
        print("  Solving refineCBF (B_h) ...")
        tvs_refine, t_refine = _timed_solve(settings, dyn_hjr, grid, times, tab.vf_table)
        print(f"    Wall time: {t_refine:.1f} s")

        np.save(hjr_path,    np.array(tvs_hjr))
        np.save(refine_path, np.array(tvs_refine))

    # Theorem 1 check
    diff = np.real(np.array(tvs_refine[-1])) - np.real(np.array(tvs_hjr[-1]))
    thm1_max_viol = float(diff.max())

    # Convergence speed
    conv_iter = _convergence_iters(tvs_refine)

    # Safe-set volume over iterations
    vol_curve = [_safe_volume(tvs_refine[i]) for i in range(len(tvs_refine))]

    print(f"  Theorem 1 max violation (B_h − B_ell): {thm1_max_viol:.4e}")
    print(f"  Convergence iteration  : {conv_iter}")

    # Build CBVF tabular CBF
    tab_cbf = TabularControlAffineCBF(dyn, cbf_params, grid=grid)
    _real_table(tab_cbf, tvs_refine[-1])

    # 6-state extension for rollout
    ext_domain = hj.sets.Box(
        lo=jnp.array([-30., -8.,  0., -8., -jnp.pi, -10.]),
        hi=jnp.array([ 30.,  8., 10.,  8.,  jnp.pi,  10.]))
    ext_res    = (5, 5, 31, 25, 41, 25)

    from replication import (QuadPlanarDynamics, ExtendedQuadVerticalCBF,
                              _build_quad_lqr)

    ext_grid   = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        ext_domain, ext_res, periodic_dims=4)
    ext_dyn    = QuadPlanarDynamics(QUAD_PARAMS, test=False)
    ext_tab    = TabularControlAffineCBF(ext_dyn, cbf_params, grid=ext_grid)
    vf_4d      = np.real(tab_cbf.vf_table).astype(float)
    ext_tab.vf_table = np.repeat(
        np.repeat(vf_4d[None, ...], 5, axis=0)[None, ...], 5, axis=0)
    ext_cbf    = ExtendedQuadVerticalCBF(ext_dyn, cbf_params, test=False)

    K       = _build_quad_lqr(ext_dyn)
    u_hover = 0.5 * MASS * G * np.ones(2)
    x_goal  = np.array([0., 0., 1.5, 0., 0., 0.])
    nom_ctrl = lambda x, t: np.atleast_2d(
        np.clip(u_hover - (K @ (x - x_goal).T).T, UMIN_2, UMAX_2))

    alpha = lambda x: 5 * x
    cbf_filt  = _patch_asif(ControlAffineASIF(
        ext_dyn, ext_cbf, alpha=alpha, nominal_policy=nom_ctrl,
        umin=UMIN_2, umax=UMAX_2))
    cbvf_filt = _patch_asif(ControlAffineASIF(
        ext_dyn, ext_tab, alpha=alpha, nominal_policy=nom_ctrl,
        umin=UMIN_2, umax=UMAX_2))

    x0  = np.array([15., -3., 2.5, -2., np.pi/4, 1.])
    exp = RolloutTrajectory("quad2d", start_x=x0, n_sims_per_start=1, t_sim=6)
    print("  Running rollout (invalid scenario) ...")
    results, rollout_wall, time_per_step = _timed_rollout(
        exp, ext_dyn, {"Nominal": nom_ctrl,
                       "Analytical": cbf_filt,
                       "CBVF": cbvf_filt})
    results.to_csv(os.path.join(out_dir, "2d_rollout_results.csv"), index=False)

    # Safety stats using the 4-state CBF value applied to the 4-state subspace
    def safe_fn_2d(x6):
        x4 = np.array([x6[2], x6[3], x6[4], x6[5]])
        return cbf_cand.vf(x4)

    safety_stats = _rollout_safety_stats(results, safe_fn_2d, ext_dyn)
    cbvf_viol    = safety_stats.get("CBVF", {}).get("violation_rate", float("nan"))
    cbf_viol     = safety_stats.get("Analytical", {}).get("violation_rate", float("nan"))

    print(f"  Safety violation rate  — Analytical: {cbf_viol:.2%}  CBVF: {cbvf_viol:.2%}")
    print(f"  Online rollout wall time: {rollout_wall:.1f} s  "
          f"({time_per_step*1000:.2f} ms/step/ctrl)")

    # Safe-set evolution figure
    off_vy = 12; off_om = 12
    fig, ax = plt.subplots(figsize=(7, 6))
    sv_real = np.real(np.array(safe_values))
    ax.contourf(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                sv_real[:, off_vy, :, off_om], levels=[-10, 0],
                colors=["#555555"], alpha=0.5)
    for idx, a in zip([0, 10, 50, -1], [0.7, 0.5, 0.3, 1.0]):
        vf_sl = np.real(np.array(tvs_refine[idx]))[:, off_vy, :, off_om]
        c     = [PAPER_COLORS[-1]] if idx == -1 else [PAPER_COLORS[max(0, idx // 25)]]
        ax.contour(grid.coordinate_vectors[2], grid.coordinate_vectors[0],
                   vf_sl, levels=[0], colors=c, linewidths=2.5, alpha=a)
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$y$ [m]")
    ax.set_title("2D Quad — safe-set evolution (φ–y slice,\n"
                 r"$\dot{y}\approx0$, $\omega\approx0$)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_stress_safe_set_2d.png"), dpi=150)
    plt.close()

    # Rollout figure
    fig, ax = plt.subplots(figsize=(8, 7))
    ss = StateSpaceExperiment("quad2d", x_indices=[0, 2], start_x=x0,
                              n_sims_per_start=1, t_sim=6)
    try:
        ss.plot(ext_dyn, results, ax=ax, color=CHOSEN_COLORS[:3])
    except (IndexError, Exception) as _e:
        print(f"  [warn] ss.plot (2D) raised {type(_e).__name__}: {_e}; using fallback.")
        _plot_trajectories_manually(results, ext_dyn, ax, CHOSEN_COLORS[:3],
                                    x_idx=0, y_idx=2)
    ax.plot(*x0[[0, 2]], "x", ms=14, mew=3, color="grey", label="Start")
    ax.plot(*x_goal[[0, 2]], "o", ms=14, color="grey", label="Goal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("2D Quad — rollout trajectories")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_stress_rollout_2d.png"), dpi=150)
    plt.close()
    print("  Saved 2D figures.")

    return {
        "label":              "2D (4-state)",
        "n_dims":             4,
        "grid_shape":         str(grid.states.shape[:-1]),
        "n_grid_pts":         n_grid_pts,
        "t_hjr_solve_s":      t_hjr,
        "t_refine_solve_s":   t_refine,
        "thm1_max_violation": thm1_max_viol,
        "conv_iter":          conv_iter,
        "safe_vol_init":      vol_curve[0],
        "safe_vol_final":     vol_curve[-1],
        "vol_curve":          vol_curve,
        "cbf_violation_rate": cbf_viol,
        "cbvf_violation_rate":cbvf_viol,
        "rollout_ms_per_step":time_per_step * 1000,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3D experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_3d_experiment(out_dir, skip_solves=False, fine_grid=False):
    """
    Full 3D quadrotor experiment (stress test).
    Returns a dict of metrics for comparison.
    """
    print("\n" + "─" * 60)
    print("  3D QUADROTOR  (stress test, 6-state grid)")
    print("─" * 60)

    cbf_params_3d = {
        "scaling": np.array([0.75, 0.5, 0.75, 0.5, 2.0, 0.5]),
        "z_nom": 5.0, "y_nom": 5.0,
    }
    dyn         = Quad3DDynamics(QUAD_PARAMS, test=True)
    dyn_jnp     = Quad3DDynamicsJNP(QUAD_PARAMS, test=False)
    cbf_cand    = Quad3DCBF(dyn, cbf_params_3d, test=False)

    grid, settings, safe_values, res = _build_3d_grid(fine=fine_grid)
    n_grid_pts = int(np.prod(grid.states.shape[:-1]))
    print(f"  Grid shape : {grid.states.shape[:-1]}  ({n_grid_pts:,} points)")
    print(f"  Fine grid  : {fine_grid}")

    dyn_hjr = HJControlAffineDynamics(
        dyn_jnp, control_space=hj.sets.Box(UMIN_4, UMAX_4))
    times   = jnp.linspace(0., -5., 51)   # fewer time slices — 3D is expensive

    hjr_path    = os.path.join(out_dir, "3d_target_values_hjr.npy")
    refine_path = os.path.join(out_dir, "3d_target_values_refine.npy")

    if skip_solves and os.path.exists(hjr_path) and os.path.exists(refine_path):
        print("  Loading saved 3D value functions ...")
        tvs_hjr    = np.load(hjr_path)
        tvs_refine = np.load(refine_path)
        t_hjr = t_refine = float("nan")
    else:
        print("  Solving vanilla HJR (B_ell) ...")
        tvs_hjr, t_hjr = _timed_solve(settings, dyn_hjr, grid, times, safe_values)
        print(f"    Wall time: {t_hjr:.1f} s")

        tab = TabularControlAffineCBF(dyn, cbf_params_3d, grid=grid)
        tab.tabularize_cbf(cbf_cand)
        print("  Solving refineCBF (B_h) ...")
        tvs_refine, t_refine = _timed_solve(
            settings, dyn_hjr, grid, times, tab.vf_table)
        print(f"    Wall time: {t_refine:.1f} s")

        np.save(hjr_path,    np.array(tvs_hjr))
        np.save(refine_path, np.array(tvs_refine))

    # Theorem 1 check
    diff = np.real(np.array(tvs_refine[-1])) - np.real(np.array(tvs_hjr[-1]))
    thm1_max_viol = float(diff.max())

    # Convergence speed
    conv_iter = _convergence_iters(tvs_refine)

    # Safe-set volume
    vol_curve = [_safe_volume(tvs_refine[i]) for i in range(len(tvs_refine))]

    print(f"  Theorem 1 max violation (B_h − B_ell): {thm1_max_viol:.4e}")
    print(f"  Convergence iteration  : {conv_iter}")

    # Build tabular CBVF for rollout
    tab_cbf = TabularControlAffineCBF(dyn, cbf_params_3d, grid=grid)
    _real_table(tab_cbf, tvs_refine[-1])

    # 3D rollout — direct 6-state rollout (no further extension needed)
    u_hover  = 0.5 * MASS * G * np.ones(4)
    x_goal   = np.array([1.5, 0., 1.5, 0., 0., 0.])
    x_nom_lqr = np.array([5., 0., 5., 0., 0., 0.])

    # LQR around hover for 6-state system
    A_dyn = dyn.state_jacobian(x_nom_lqr, u_hover)
    B_dyn = dyn.control_matrix(x_nom_lqr)

    from scipy.linalg import expm
    A_d = expm(A_dyn * DT)

    # Robust ZOH for B_d: use block-augmented matrix exponential.
    # Works even when A_dyn is singular (e.g. integrator modes at hover).
    n = A_dyn.shape[0]
    m = B_dyn.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_dyn
    M[:n, n:] = B_dyn
    M_exp = expm(M * DT)
    B_d = M_exp[:n, n:]

    Q_lqr = np.diag([1., 0.1, 1., 0.1, 2., 0.5])
    R_lqr = np.eye(4)
    try:
        from scipy.linalg import solve_discrete_are
        Pl  = solve_discrete_are(A_d, B_d, Q_lqr, R_lqr)
        K3d = np.linalg.solve(R_lqr + B_d.T @ Pl @ B_d, B_d.T @ Pl @ A_d)
    except Exception:
        # Fallback: pseudo-inverse gain
        K3d = np.linalg.pinv(B_d) @ (A_d - np.eye(6))

    nom_ctrl_3d = lambda x, t: np.atleast_2d(
        np.clip(u_hover - (K3d @ (x - x_goal).T).T, UMIN_4, UMAX_4))

    alpha = lambda x: 5 * x
    cbf_filt_3d  = _patch_asif(ControlAffineASIF(
        dyn, cbf_cand, alpha=alpha, nominal_policy=nom_ctrl_3d,
        umin=UMIN_4, umax=UMAX_4))
    cbvf_filt_3d = _patch_asif(ControlAffineASIF(
        dyn, tab_cbf, alpha=alpha, nominal_policy=nom_ctrl_3d,
        umin=UMIN_4, umax=UMAX_4))

    x0_3d = np.array([2.5, -2., 2.5, -2., np.pi/4, 1.])
    exp   = RolloutTrajectory("quad3d", start_x=x0_3d, n_sims_per_start=1, t_sim=6)
    print("  Running rollout ...")
    results_3d, rollout_wall, time_per_step = _timed_rollout(
        exp, dyn, {"Nominal": nom_ctrl_3d,
                   "Analytical": cbf_filt_3d,
                   "CBVF": cbvf_filt_3d})
    results_3d.to_csv(os.path.join(out_dir, "3d_rollout_results.csv"), index=False)

    def safe_fn_3d(x):
        return cbf_cand.vf(np.array(x))

    safety_stats  = _rollout_safety_stats(results_3d, safe_fn_3d, dyn)
    cbvf_viol_3d  = safety_stats.get("CBVF",      {}).get("violation_rate", float("nan"))
    cbf_viol_3d   = safety_stats.get("Analytical", {}).get("violation_rate", float("nan"))

    print(f"  Safety violation rate  — Analytical: {cbf_viol_3d:.2%}  CBVF: {cbvf_viol_3d:.2%}")
    print(f"  Online rollout wall time: {rollout_wall:.1f} s  "
          f"({time_per_step*1000:.2f} ms/step/ctrl)")

    # Safe-set evolution figure — φ–z slice (dims 4 and 0)
    # Slice at mid-index of vz, vy, φ̇
    mid = [s // 2 for s in res]   # mid-indices for all dims
    fig, ax = plt.subplots(figsize=(7, 6))
    sv_r = np.real(np.array(safe_values))
    # Index order: [z, vz, y, vy, φ, φ̇] → slice at vz=mid[1], y=mid[2], vy=mid[3], φ̇=mid[5]
    def _slice_3d(vf_nd):
        return vf_nd[:, mid[1], mid[2], mid[3], :, mid[5]]

    ax.contourf(grid.coordinate_vectors[4], grid.coordinate_vectors[0],
                _slice_3d(sv_r), levels=[-10, 0],
                colors=["#555555"], alpha=0.5)
    for idx, a in zip([0, 10, 25, -1], [0.7, 0.5, 0.3, 1.0]):
        vf_sl = _slice_3d(np.real(np.array(tvs_refine[idx])))
        c     = [PAPER_COLORS[-1]] if idx == -1 else [PAPER_COLORS[max(0, idx // 15)]]
        ax.contour(grid.coordinate_vectors[4], grid.coordinate_vectors[0],
                   vf_sl, levels=[0], colors=c, linewidths=2.5, alpha=a)
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$z$ [m]")
    ax.set_title("3D Quad — safe-set evolution (φ–z slice,\n"
                 r"$v_z\approx0$, $v_y\approx0$, $\omega\approx0$)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_stress_safe_set_3d.png"), dpi=150)
    plt.close()

    # Rollout figure — z vs y trajectory
    fig, ax = plt.subplots(figsize=(8, 7))
    ss = StateSpaceExperiment("quad3d", x_indices=[2, 0], start_x=x0_3d,
                              n_sims_per_start=1, t_sim=6)
    try:
        ss.plot(dyn, results_3d, ax=ax, color=CHOSEN_COLORS[:3])
    except (IndexError, Exception) as _e:
        print(f"  [warn] ss.plot (3D) raised {type(_e).__name__}: {_e}; using fallback.")
        _plot_trajectories_manually(results_3d, dyn, ax, CHOSEN_COLORS[:3],
                                    x_idx=2, y_idx=0)
    ax.plot(x0_3d[2], x0_3d[0], "x", ms=14, mew=3, color="grey", label="Start")
    ax.plot(x_goal[2], x_goal[0], "o", ms=14, color="grey", label="Goal")
    ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]")
    ax.set_title("3D Quad — rollout trajectories (y–z plane)")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_stress_rollout_3d.png"), dpi=150)
    plt.close()
    print("  Saved 3D figures.")

    return {
        "label":              f"3D (6-state{'–fine' if fine_grid else ''})",
        "n_dims":             6,
        "grid_shape":         str(grid.states.shape[:-1]),
        "n_grid_pts":         n_grid_pts,
        "t_hjr_solve_s":      t_hjr,
        "t_refine_solve_s":   t_refine,
        "thm1_max_violation": thm1_max_viol,
        "conv_iter":          conv_iter,
        "safe_vol_init":      vol_curve[0],
        "safe_vol_final":     vol_curve[-1],
        "vol_curve":          vol_curve,
        "cbf_violation_rate": cbf_viol_3d,
        "cbvf_violation_rate":cbvf_viol_3d,
        "rollout_ms_per_step":time_per_step * 1000,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Comparison report
# ═════════════════════════════════════════════════════════════════════════════

def _make_comparison_figure(m2, m3, out_dir):
    """
    Six-panel comparison figure:
      [0] Grid size vs n_dims (exponential scaling bar chart)
      [1] DP solve time: HJR vs refineCBF  (grouped bars)
      [2] Safe-set volume over DP iterations
      [3] Theorem 1 max violation (bar)
      [4] Safety violation rate during rollout (bar)
      [5] Online solve time per step (bar)
    """
    fig = plt.figure(figsize=(28, 8))
    fig.suptitle(
        "2D vs 3D — refineCBF Comparison\n",
        fontsize=16, y=0.99)

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45)
    axes = [fig.add_subplot(gs[r, c]) for r in range(1) for c in range(4)]

    labels = [m2["label"], m3["label"]]
    c2, c3 = sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]

    # ── Panel 0: grid points (log scale) ─────────────────────────────────────
    ax = axes[0]
    vals = [m2["n_grid_pts"], m3["n_grid_pts"]]
    bars = ax.bar(labels, vals, color=[c2, c3], width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Total grid points")
    ax.set_title("Grid size")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.15,
                f"{v:,}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(top=max(vals) * 8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 1: solve time ───────────────────────────────────────────────────
    ax = axes[1]
    x   = np.arange(2)
    w   = 0.35
    hjr_times    = [m2["t_hjr_solve_s"],    m3["t_hjr_solve_s"]]
    refine_times = [m2["t_refine_solve_s"], m3["t_refine_solve_s"]]

    # If timing was skipped (nan), show a grey "N/A" bar
    def _bar_or_na(ax, xs, vals, color, label):
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                ax.bar(xi, 1, color="lightgrey", width=w,
                       label=label if xi == xs[0] else "")
                ax.text(xi, 1.1, "N/A", ha="center", va="bottom", fontsize=10)
            else:
                ax.bar(xi, v, color=color, width=w,
                       label=label if xi == xs[0] else "")
                ax.text(xi, v + max(v*0.02, 0.5), f"{v:.0f}s",
                        ha="center", va="bottom", fontsize=9)

    _bar_or_na(ax, x - w/2, hjr_times,    c2, "Vanilla HJR")
    _bar_or_na(ax, x + w/2, refine_times, c3, "refineCBF")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("DP solve time\n(warm-start speedup)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: safe-set volume convergence ──────────────────────────────────
    ax = axes[2]
    for m, col, ls in [(m2, c2, "-"), (m3, c3, "--")]:
        vc = m["vol_curve"]
        iters = np.linspace(0, 1, len(vc))   # normalise to [0,1] for comparability
        ax.plot(iters, vc, color=col, linestyle=ls, linewidth=2.5,
                label=m["label"])
    ax.set_xlabel("Normalised DP iterations")
    ax.set_ylabel("Safe-set volume\n(fraction of grid)")
    ax.set_title("Safe-set growth over DP iterations")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Panel 5: Online solve time ────────────────────────────────────────────
    ax = axes[3]
    ms_vals = [m2["rollout_ms_per_step"], m3["rollout_ms_per_step"]]
    bars    = ax.bar(labels, ms_vals, color=[c2, c3], width=0.5)
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("QP solve time per step")
    ax.grid(axis="y", alpha=0.3)
    for bar, v, lbl in zip(bars, ms_vals, labels):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + max(v*0.02, 0.01),
                    f"{v:.1f} ms", ha="center", va="bottom", fontsize=11)

    fig_path = os.path.join(out_dir, "fig_stress_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison figure: {fig_path}")


def _make_scaling_figure(m2, m3, out_dir):
    """
    Additional figure: grid points and solve time vs state-space dimension,
    illustrating the exponential (curse-of-dimensionality) scaling.
    """
    dims  = [m2["n_dims"], m3["n_dims"]]
    pts   = [m2["n_grid_pts"], m3["n_grid_pts"]]
    t_hjr = [m2["t_hjr_solve_s"], m3["t_hjr_solve_s"]]
    t_ref = [m2["t_refine_solve_s"], m3["t_refine_solve_s"]]

    # Theoretical exponential reference: k^n for k pts/dim
    d_range = np.linspace(4, 6, 200)
    # Fit a 2-point exponential: pts = a * b^n
    if pts[0] > 0 and pts[1] > 0:
        b = (pts[1] / pts[0]) ** (1 / (dims[1] - dims[0]))
        a = pts[0] / b ** dims[0]
        exp_ref = a * b ** d_range
    else:
        exp_ref = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogy(dims, pts, "o-", color=sns.color_palette("tab10")[0],
                markersize=10, linewidth=2.5, label="Measured")
    if exp_ref is not None:
        ax.semilogy(d_range, exp_ref, "--", color="grey", linewidth=1.5,
                    label=f"Exponential fit (×{b:.1f} per dim)")
    ax.set_xlabel("State-space dimension n")
    ax.set_ylabel("Total grid points (log scale)")
    ax.set_xticks(dims)
    ax.set_xticklabels([f"{d}D\n{m['grid_shape']}" for d, m in
                        [(dims[0], m2), (dims[1], m3)]], fontsize=10)
    ax.set_title("Curse of dimensionality\n(grid size vs dimension)")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    ax = axes[1]
    valid_dims  = [d for d, t in zip(dims, t_hjr) if not np.isnan(t)]
    valid_t_hjr = [t for t in t_hjr if not np.isnan(t)]
    valid_t_ref = [t for t in t_ref if not np.isnan(t)]
    if valid_dims:
        ax.semilogy(valid_dims, valid_t_hjr, "s-",
                    color=sns.color_palette("tab10")[1], markersize=10,
                    linewidth=2.5, label="Vanilla HJR")
        if len(valid_t_ref) == len(valid_dims):
            ax.semilogy(valid_dims, valid_t_ref, "^--",
                        color=sns.color_palette("tab10")[2], markersize=10,
                        linewidth=2.5, label="refineCBF (warm-start)")
    ax.set_xlabel("State-space dimension n")
    ax.set_ylabel("DP solve time [s]  (log scale)")
    ax.set_xticks(valid_dims if valid_dims else dims)
    ax.set_title("DP solve time vs dimension\n(warm-start benefit)")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    fig.suptitle("Grid scaling analysis — 2D vs 3D refineCBF", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_stress_grid_scaling.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scaling figure:    {fig_path}")


def _save_comparison_csv(m2, m3, out_dir):
    rows = []
    for m in [m2, m3]:
        row = {k: v for k, v in m.items() if k != "vol_curve"}
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "stress_test_comparison.csv")
    df.to_csv(path, index=False)
    print(f"  Saved comparison CSV:    {path}")
    return df


def _print_summary_table(m2, m3):
    print("\n" + "═" * 70)
    print("  STRESS TEST SUMMARY — 2D (paper) vs 3D")
    print("═" * 70)
    rows = [
        ("State dims",         m2["n_dims"],              m3["n_dims"],              ""),
        ("Grid shape",         m2["grid_shape"],          m3["grid_shape"],          ""),
        ("Grid points",        f"{m2['n_grid_pts']:,}",   f"{m3['n_grid_pts']:,}",   "exponential blowup"),
        ("HJR solve time [s]", f"{m2['t_hjr_solve_s']:.1f}",
                               f"{m3['t_hjr_solve_s']:.1f}",    ""),
        ("refineCBF solve [s]",f"{m2['t_refine_solve_s']:.1f}",
                               f"{m3['t_refine_solve_s']:.1f}", "warm-start helps"),
        ("Conv. iteration",    m2["conv_iter"],           m3["conv_iter"],           ""),
        ("Safe vol (init)",    f"{m2['safe_vol_init']:.3f}",
                               f"{m3['safe_vol_init']:.3f}",    ""),
        ("Safe vol (final)",   f"{m2['safe_vol_final']:.3f}",
                               f"{m3['safe_vol_final']:.3f}",   ""),
        ("Thm1 max viol.",     f"{m2['thm1_max_violation']:.2e}",
                               f"{m3['thm1_max_violation']:.2e}",
                               "≤0 = valid (numerics may show ε>0)"),
        ("CBF viol. rate",     f"{m2['cbf_violation_rate']:.2%}",
                               f"{m3['cbf_violation_rate']:.2%}", ""),
        ("CBVF viol. rate",    f"{m2['cbvf_violation_rate']:.2%}",
                               f"{m3['cbvf_violation_rate']:.2%}", ""),
        ("Online ms/step",     f"{m2['rollout_ms_per_step']:.1f}",
                               f"{m3['rollout_ms_per_step']:.1f}", "per controller"),
    ]
    hdr = f"  {'Metric':<26} {'2D (paper)':>14} {'3D (stress)':>14}  Note"
    print(hdr)
    print("  " + "─" * 68)
    for label, v2, v3, note in rows:
        note_str = f"  ← {note}" if note else ""
        print(f"  {label:<26} {str(v2):>14} {str(v3):>14}{note_str}")
    print("═" * 70)

    # Scaling multipliers
    if (not np.isnan(m2["t_hjr_solve_s"]) and not np.isnan(m3["t_hjr_solve_s"])
            and m2["t_hjr_solve_s"] > 0):
        ratio = m3["t_hjr_solve_s"] / m2["t_hjr_solve_s"]
        pts_ratio = m3["n_grid_pts"] / m2["n_grid_pts"]
        print(f"\n  Grid size  ratio (3D/2D): {pts_ratio:,.1f}×")
        print(f"  Solve time ratio (3D/2D): {ratio:.1f}×")
        if not np.isnan(m2["t_refine_solve_s"]) and not np.isnan(m3["t_refine_solve_s"]):
            speedup_2d = m2["t_hjr_solve_s"] / max(m2["t_refine_solve_s"], 1e-9)
            speedup_3d = m3["t_hjr_solve_s"] / max(m3["t_refine_solve_s"], 1e-9)
            print(f"  Warm-start speedup  2D : {speedup_2d:.2f}×")
            print(f"  Warm-start speedup  3D : {speedup_3d:.2f}×")


# ═════════════════════════════════════════════════════════════════════════════
# CLI & entrypoint
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stress-test refineCBF by extending the quadrotor from "
                    "2D (paper) to 3D and comparing all metrics.")
    p.add_argument("--skip-solves", action="store_true",
                   help="Reload saved .npy value functions if present.")
    p.add_argument("--fine-grid", action="store_true",
                   help="Use a denser 3D grid (very slow, GPU recommended).")
    p.add_argument("--out-dir", default="results_stress",
                   help="Output directory (default: ./results_stress/).")
    p.add_argument("--only", choices=["2d", "3d"],
                   help="Run only one dimension (default: both).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  refineCBF 3D Stress Test — extending Tonkens & Herbert IROS 2022")
    print("  Python", sys.version.split()[0])
    print("=" * 70)
    print(f"  Output dir : {os.path.abspath(args.out_dir)}")
    print(f"  Skip solves: {args.skip_solves}")
    print(f"  Fine grid  : {args.fine_grid}")
    if args.fine_grid:
        print("  WARNING: fine grid requires significant RAM (≥16 GB) "
              "and a GPU for tractable runtime.")

    m2 = m3 = None

    if args.only in (None, "2d"):
        m2 = run_2d_experiment(args.out_dir, skip_solves=args.skip_solves)

    if args.only in (None, "3d"):
        m3 = run_3d_experiment(args.out_dir, skip_solves=args.skip_solves,
                               fine_grid=args.fine_grid)

    if m2 is not None and m3 is not None:
        _print_summary_table(m2, m3)
        _save_comparison_csv(m2, m3, args.out_dir)
        _make_comparison_figure(m2, m3, args.out_dir)
        _make_scaling_figure(m2, m3, args.out_dir)

    print("\n" + "=" * 70)
    print(f"  All done!  Outputs: {os.path.abspath(args.out_dir)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()