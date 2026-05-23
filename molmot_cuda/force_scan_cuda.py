"""
CUDA-accelerated force scanning and MOT characterisation utilities.

Higher-level convenience functions wrapping the GPU rate equation solver.
Drop-in replacements for ``molmot.mot.force_scan`` functions with
GPU acceleration.

Usage
-----
>>> from molmot_cuda.force_scan_cuda import (
...     force_vs_z_cuda, force_vs_v_cuda,
...     optimize_parameters_cuda, capture_velocity_cuda,
... )
>>> F, pop, R = force_vs_z_cuda(sim, z_arr, v=0.0)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy is required for CUDA-accelerated computation. "
        "Install with: pip install cupy-cuda12x"
    )

from .rate_equations_cuda import (
    CUDAMolData,
    solve_rate_equations_batch,
    force_vs_z_cuda as _force_vs_z_raw,
    force_vs_v_cuda as _force_vs_v_raw,
    force_map_2d_cuda as _force_map_2d_raw,
    spring_constant_cuda as _spring_constant_raw,
    damping_coefficient_cuda as _damping_coefficient_raw,
    optimize_parameters_cuda as _optimize_params_raw,
)


# =====================================================================
# Simulator-level wrappers (matching CPU force_scan.py API)
# =====================================================================

def force_vs_z_cuda(
    simulator,
    z_arr: np.ndarray,
    v: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force as a function of position (GPU-accelerated).

    Drop-in replacement for ``molmot.mot.force_scan.force_vs_z``.
    Accepts a simulator object (DCMOTSimulator or RFMOTSimulator)
    and returns the same output format.

    For RF MOT, the two half-cycles are computed separately on GPU
    and time-averaged, matching the CPU implementation.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    z_arr : np.ndarray
        Array of positions (m).
    v : float
        Fixed velocity (m/s).

    Returns
    -------
    F_arr : np.ndarray, shape (N_z,)
        Force at each position (N).
    pop_arr : np.ndarray, shape (N_z, n_ground)
        Ground-state populations at each position.
    R_arr : np.ndarray, shape (N_z,)
        Scattering rate at each position (rad/s).
    """
    from molmot.mot.simulator import RFMOTSimulator, DCMOTSimulator

    if isinstance(simulator, DCMOTSimulator):
        mol = simulator.mol_data
        beams = simulator._beams
        B_grad = simulator.B_gradient
        Gamma_eff = simulator.Gamma_eff_factor

        return _force_vs_z_raw(
            mol, beams, z_arr, v, B_grad, Gamma_eff)

    elif isinstance(simulator, RFMOTSimulator):
        mol = simulator.mol_data
        B_grad = simulator.B_gradient
        Gamma_eff = simulator.Gamma_eff_factor

        from molmot.obe.fields import make_rf_mot_beams

        # Phase 0: sigma+ with +B
        beams_0 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=0)
        F0, pop0, R0 = _force_vs_z_raw(
            mol, beams_0, z_arr, v, B_grad, Gamma_eff)

        # Phase 1: sigma- with -B
        beams_1 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=1)
        F1, pop1, R1 = _force_vs_z_raw(
            mol, beams_1, z_arr, v, -B_grad, Gamma_eff)

        # Time average
        return 0.5 * (F0 + F1), 0.5 * (pop0 + pop1), 0.5 * (R0 + R1)

    else:
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}. "
            f"Expected DCMOTSimulator or RFMOTSimulator.")


def force_vs_v_cuda(
    simulator,
    v_arr: np.ndarray,
    z: float = 0.0,
) -> np.ndarray:
    """
    Compute force as a function of velocity (GPU-accelerated).

    Drop-in replacement for ``molmot.mot.force_scan.force_vs_v``.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    v_arr : np.ndarray
        Array of velocities (m/s).
    z : float
        Fixed position (m).

    Returns
    -------
    F_arr : np.ndarray, shape (N_v,)
        Force at each velocity (N).
    """
    from molmot.mot.simulator import RFMOTSimulator, DCMOTSimulator

    if isinstance(simulator, DCMOTSimulator):
        mol = simulator.mol_data
        beams = simulator._beams
        B_grad = simulator.B_gradient
        Gamma_eff = simulator.Gamma_eff_factor

        return _force_vs_v_raw(
            mol, beams, v_arr, z, B_grad, Gamma_eff)

    elif isinstance(simulator, RFMOTSimulator):
        mol = simulator.mol_data
        B_grad = simulator.B_gradient
        Gamma_eff = simulator.Gamma_eff_factor

        from molmot.obe.fields import make_rf_mot_beams

        beams_0 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=0)
        F0 = _force_vs_v_raw(
            mol, beams_0, v_arr, z, B_grad, Gamma_eff)

        beams_1 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=1)
        F1 = _force_vs_v_raw(
            mol, beams_1, v_arr, z, -B_grad, Gamma_eff)

        return 0.5 * (F0 + F1)

    else:
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}")


def spring_constant_cuda(
    simulator,
    z0: float = 0.0,
    dz: float = 0.3e-3,
) -> float:
    """
    Compute the spring constant k = -dF/dz at position z0 (GPU).

    Drop-in replacement for ``molmot.mot.force_scan.spring_constant``.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    z0 : float
        Equilibrium position (m).
    dz : float
        Finite difference step (m).

    Returns
    -------
    k : float
        Spring constant (N/m).  Positive = restoring.
    """
    from molmot.mot.simulator import RFMOTSimulator, DCMOTSimulator

    if isinstance(simulator, DCMOTSimulator):
        return _spring_constant_raw(
            simulator.mol_data, simulator._beams,
            simulator.B_gradient, z0, dz,
            simulator.Gamma_eff_factor)

    elif isinstance(simulator, RFMOTSimulator):
        # For RF MOT, compute both phases and average
        from molmot.obe.fields import make_rf_mot_beams
        mol = simulator.mol_data

        beams_0 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=0)
        k0 = _spring_constant_raw(
            mol, beams_0, simulator.B_gradient, z0, dz,
            simulator.Gamma_eff_factor)

        beams_1 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=1)
        k1 = _spring_constant_raw(
            mol, beams_1, -simulator.B_gradient, z0, dz,
            simulator.Gamma_eff_factor)

        return 0.5 * (k0 + k1)
    else:
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}")


def damping_coefficient_cuda(
    simulator,
    z0: float = 0.0,
    dv: float = 0.1,
) -> float:
    """
    Compute the damping coefficient alpha = -dF/dv at z0 (GPU).

    Drop-in replacement for ``molmot.mot.force_scan.damping_coefficient``.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    z0 : float
        Position (m).
    dv : float
        Finite difference step (m/s).

    Returns
    -------
    alpha : float
        Friction coefficient dF/dv (N*s/m).
    """
    from molmot.mot.simulator import RFMOTSimulator, DCMOTSimulator

    if isinstance(simulator, DCMOTSimulator):
        return _damping_coefficient_raw(
            simulator.mol_data, simulator._beams,
            simulator.B_gradient, z0, dv,
            simulator.Gamma_eff_factor)

    elif isinstance(simulator, RFMOTSimulator):
        from molmot.obe.fields import make_rf_mot_beams
        mol = simulator.mol_data

        beams_0 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=0)
        a0 = _damping_coefficient_raw(
            mol, beams_0, simulator.B_gradient, z0, dv,
            simulator.Gamma_eff_factor)

        beams_1 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=1)
        a1 = _damping_coefficient_raw(
            mol, beams_1, -simulator.B_gradient, z0, dv,
            simulator.Gamma_eff_factor)

        return 0.5 * (a0 + a1)
    else:
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}")


# =====================================================================
# CUDA kernel for trajectory integration (capture velocity)
# =====================================================================

_TRAJECTORY_KERNEL_SOURCE = r"""
extern "C" __global__
void trajectory_kernel(
    // --- system sizes ---
    const int n_ground,
    const int n_excited,
    const int n_states,
    const int n_beams,
    const double Gamma,
    const double k_wave,
    const double Gamma_eff,
    const double hbar_val,
    // --- molecular data ---
    const double* __restrict__ energies,
    const double* __restrict__ zeeman_z_diag,
    const double* __restrict__ d_squared,
    const double* __restrict__ BR,
    // --- beam data ---
    const double* __restrict__ beam_dirs,
    const double* __restrict__ beam_freqs,
    const double* __restrict__ beam_pols,
    const double* __restrict__ beam_s0s,
    // --- trajectory parameters ---
    const int N_traj,          // number of trajectories
    const double* __restrict__ v0_arr,  // [N_traj] initial velocities
    const double z_start,
    const double B_gradient,
    const double mass,
    const double dt,
    const int n_steps,
    const double z_escape,
    // --- output ---
    int* __restrict__ trapped_out  // [N_traj] 1=trapped, 0=escaped
)
{
    int itraj = blockIdx.x * blockDim.x + threadIdx.x;
    if (itraj >= N_traj) return;

    int n_g = n_ground;
    int n_e = n_excited;
    double inv_2pi = 1.0 / (2.0 * 3.14159265358979323846);
    double gamma_half = Gamma / 2.0;
    double gamma_half_sq = gamma_half * gamma_half;

    double z = z_start;
    double v = v0_arr[itraj];
    int is_trapped = 1;

    for (int step = 0; step < n_steps; step++) {
        // --- Compute force at current (v, z) ---
        double B_gauss = B_gradient * z * 1.0e4;

        double Es[MAX_STATES];
        for (int i = 0; i < n_g + n_e; i++) {
            Es[i] = energies[i] + B_gauss * zeeman_z_diag[i] * Gamma * inv_2pi;
        }

        // Saturation totals
        double s_total[MAX_GROUND];
        for (int ig = 0; ig < n_g; ig++) s_total[ig] = 0.0;

        for (int ib = 0; ib < n_beams; ib++) {
            double kdir = beam_dirs[ib];
            double freq = beam_freqs[ib];
            int pol = (int)beam_pols[ib];
            double s0 = beam_s0s[ib];
            double doppler = -k_wave * kdir * v * inv_2pi;

            for (int ig = 0; ig < n_g; ig++) {
                for (int ie = 0; ie < n_e; ie++) {
                    double d2 = d_squared[ig * n_e * 3 + ie * 3 + pol];
                    if (d2 < 1.0e-15) continue;
                    double omega_trans = Es[n_g + ie] - Es[ig];
                    double delta_eff = (freq + doppler - omega_trans)
                                       * 2.0 * 3.14159265358979323846;
                    double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);
                    s_total[ig] += s0 * d2 * L;
                }
            }
        }

        // Excitation rates with saturation
        double R_sum[MAX_GROUND * MAX_EXCITED];
        double F_total_arr[MAX_GROUND * MAX_EXCITED];
        for (int i = 0; i < n_g * n_e; i++) {
            R_sum[i] = 0.0;
            F_total_arr[i] = 0.0;
        }

        for (int ib = 0; ib < n_beams; ib++) {
            double kdir = beam_dirs[ib];
            double freq = beam_freqs[ib];
            int pol = (int)beam_pols[ib];
            double s0 = beam_s0s[ib];
            double doppler = -k_wave * kdir * v * inv_2pi;

            for (int ig = 0; ig < n_g; ig++) {
                double sat_corr = 1.0 / (1.0 + s_total[ig]);
                for (int ie = 0; ie < n_e; ie++) {
                    double d2 = d_squared[ig * n_e * 3 + ie * 3 + pol];
                    if (d2 < 1.0e-15) continue;
                    double omega_trans = Es[n_g + ie] - Es[ig];
                    double delta_eff = (freq + doppler - omega_trans)
                                       * 2.0 * 3.14159265358979323846;
                    double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);
                    double rate = gamma_half * s0 * d2 * L * Gamma_eff * sat_corr;
                    int ige = ig * n_e + ie;
                    R_sum[ige] += rate;
                    F_total_arr[ige] += hbar_val * k_wave * kdir * rate;
                }
            }
        }

        // Population matrix
        double M[MAX_GROUND * MAX_GROUND];
        for (int i = 0; i < n_g * n_g; i++) M[i] = 0.0;

        for (int ig = 0; ig < n_g; ig++) {
            for (int ie = 0; ie < n_e; ie++) {
                M[ig * n_g + ig] -= R_sum[ig * n_e + ie];
            }
            for (int ik = 0; ik < n_g; ik++) {
                for (int ie = 0; ie < n_e; ie++) {
                    M[ig * n_g + ik] += R_sum[ik * n_e + ie] * BR[ie * n_g + ig];
                }
            }
        }

        for (int j = 0; j < n_g; j++) {
            M[(n_g - 1) * n_g + j] = 1.0;
        }
        double b[MAX_GROUND];
        for (int i = 0; i < n_g; i++) b[i] = 0.0;
        b[n_g - 1] = 1.0;

        // Gaussian elimination with partial pivoting
        double A[MAX_GROUND * MAX_GROUND];
        for (int i = 0; i < n_g * n_g; i++) A[i] = M[i];

        for (int col = 0; col < n_g; col++) {
            double max_val = fabs(A[col * n_g + col]);
            int max_row = col;
            for (int row = col + 1; row < n_g; row++) {
                double val = fabs(A[row * n_g + col]);
                if (val > max_val) {
                    max_val = val;
                    max_row = row;
                }
            }
            if (max_row != col) {
                for (int j = 0; j < n_g; j++) {
                    double tmp = A[col * n_g + j];
                    A[col * n_g + j] = A[max_row * n_g + j];
                    A[max_row * n_g + j] = tmp;
                }
                double tmp_b = b[col];
                b[col] = b[max_row];
                b[max_row] = tmp_b;
            }
            for (int row = col + 1; row < n_g; row++) {
                double pivot = A[col * n_g + col];
                if (fabs(pivot) < 1.0e-30) continue;
                double factor = A[row * n_g + col] / pivot;
                for (int j = col; j < n_g; j++) {
                    A[row * n_g + j] -= factor * A[col * n_g + j];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        double p[MAX_GROUND];
        for (int i = 0; i < n_g; i++) p[i] = 0.0;
        for (int i = n_g - 1; i >= 0; i--) {
            double s = b[i];
            for (int j = i + 1; j < n_g; j++) {
                s -= A[i * n_g + j] * p[j];
            }
            if (fabs(A[i * n_g + i]) > 1.0e-30) {
                p[i] = s / A[i * n_g + i];
            }
        }

        double total_p = 0.0;
        for (int i = 0; i < n_g; i++) {
            if (p[i] < 0.0) p[i] = 0.0;
            total_p += p[i];
        }
        if (total_p > 0.0) {
            for (int i = 0; i < n_g; i++) p[i] /= total_p;
        } else {
            double inv_n = 1.0 / (double)n_g;
            for (int i = 0; i < n_g; i++) p[i] = inv_n;
        }

        // Force
        double force = 0.0;
        for (int ig = 0; ig < n_g; ig++) {
            for (int ie = 0; ie < n_e; ie++) {
                force += p[ig] * F_total_arr[ig * n_e + ie];
            }
        }

        // Update trajectory
        v += (force / mass) * dt;
        z += v * dt;

        if (fabs(z) > z_escape) {
            is_trapped = 0;
            break;
        }
    }

    trapped_out[itraj] = is_trapped;
}
"""

_traj_kernel_cache = {}


def _get_traj_kernel(n_ground: int, n_excited: int, n_beams: int):
    """Compile or retrieve cached trajectory integration kernel."""
    key = (n_ground, n_excited, n_beams)
    if key not in _traj_kernel_cache:
        n_states = n_ground + n_excited
        defines = (
            f"#define MAX_GROUND {n_ground}\n"
            f"#define MAX_EXCITED {n_excited}\n"
            f"#define MAX_BEAMS {n_beams}\n"
            f"#define MAX_STATES {n_states}\n"
        )
        source = defines + _TRAJECTORY_KERNEL_SOURCE
        kernel = cp.RawKernel(source, "trajectory_kernel")
        _traj_kernel_cache[key] = kernel
    return _traj_kernel_cache[key]


def capture_velocity_cuda(
    simulator,
    z_start: float = 3e-3,
    v_max: float = 15.0,
    dv: float = 0.5,
    dt: float = 1e-6,
    n_steps: int = 5000,
    z_escape: float = 0.015,
) -> float:
    """
    Estimate the capture velocity by trajectory simulation on GPU.

    All test velocities are integrated in parallel on the GPU.
    For each initial velocity v0, a molecule starts at ``z_start``
    with velocity ``-v0`` and is considered captured if |z| < z_escape
    after ``n_steps`` time steps.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    z_start : float
        Initial position (m).
    v_max : float
        Maximum test velocity (m/s).
    dv : float
        Velocity step (m/s).
    dt : float
        Time step for trajectory (s).
    n_steps : int
        Number of time steps.
    z_escape : float
        Escape boundary (m).

    Returns
    -------
    v_capture : float
        Maximum captured velocity (m/s).

    Notes
    -----
    For the RF MOT, this function currently only supports the
    DC MOT simulator.  For the RF MOT, the time-varying fields
    would require a more complex kernel.  Use the CPU version
    for RF MOT capture velocity estimation.
    """
    from molmot.mot.simulator import DCMOTSimulator, RFMOTSimulator
    from .rate_equations_cuda import _prepare_mol_data, _prepare_beam_data

    if isinstance(simulator, RFMOTSimulator):
        # Fall back to CPU for RF MOT (time-varying fields)
        from molmot.mot.force_scan import capture_velocity
        return capture_velocity(
            simulator, z_start=z_start, v_max=v_max, dv=dv,
            dt=dt, n_steps=n_steps, z_escape=z_escape)

    if not isinstance(simulator, DCMOTSimulator):
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}")

    mol_data = simulator.mol_data
    beams = simulator._beams
    B_gradient = simulator.B_gradient
    Gamma_eff = simulator.Gamma_eff_factor
    mass = mol_data.mass

    # Prepare GPU data
    mol = _prepare_mol_data(mol_data)
    bm = _prepare_beam_data(beams)

    d_energies = cp.asarray(mol["energies"])
    d_zeeman = cp.asarray(mol["zeeman_z_diag"])
    d_dsq = cp.asarray(mol["d_squared"])
    d_BR = cp.asarray(mol["BR"])
    d_dirs = cp.asarray(bm["beam_dirs"])
    d_freqs = cp.asarray(bm["beam_freqs"])
    d_pols = cp.asarray(bm["beam_pols"])
    d_s0s = cp.asarray(bm["beam_s0s"])

    # Build velocity test array: negative velocities (moving towards trap)
    v_test_arr = np.arange(dv, v_max + dv, dv, dtype=np.float64)
    v0_arr = -v_test_arr  # negative = moving towards origin

    N_traj = len(v0_arr)
    d_v0 = cp.asarray(v0_arr)
    d_trapped = cp.zeros(N_traj, dtype=cp.int32)

    n_g = mol["n_ground"]
    n_e = mol["n_excited"]
    n_b = bm["n_beams"]

    kernel = _get_traj_kernel(n_g, n_e, n_b)

    block = (256,)
    grid = ((N_traj + block[0] - 1) // block[0],)

    kernel(
        grid, block,
        (
            np.int32(n_g),
            np.int32(n_e),
            np.int32(n_g + n_e),
            np.int32(n_b),
            np.float64(mol["Gamma"]),
            np.float64(mol["k"]),
            np.float64(Gamma_eff),
            np.float64(1.0545718e-34),
            d_energies,
            d_zeeman,
            d_dsq,
            d_BR,
            d_dirs,
            d_freqs,
            d_pols,
            d_s0s,
            np.int32(N_traj),
            d_v0,
            np.float64(z_start),
            np.float64(B_gradient),
            np.float64(mass),
            np.float64(dt),
            np.int32(n_steps),
            np.float64(z_escape),
            d_trapped,
        )
    )

    trapped = cp.asnumpy(d_trapped)

    # Find maximum captured velocity (last contiguous trapped from low v)
    v_capture = 0.0
    for i in range(N_traj):
        if trapped[i]:
            v_capture = v_test_arr[i]
        else:
            break

    return v_capture


def optimize_parameters_cuda(
    mol_data,
    param_ranges: Dict[str, np.ndarray],
    B_gradient: float,
    mot_type: str = "dc",
    s0: float = 1.0,
    dz: float = 0.3e-3,
    dv: float = 0.1,
    Gamma_eff_factor: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """
    Optimise MOT parameters by scanning detuning and split (GPU).

    Drop-in replacement for ``molmot.mot.force_scan.optimize_parameters``
    with GPU acceleration.

    Parameters
    ----------
    mol_data : MolecularData
    param_ranges : dict
        Must contain:
        - "delta": np.ndarray of detunings (in Gamma)
        - "split": np.ndarray of splits (in Gamma), only for DC MOT
    B_gradient : float
        Magnetic field gradient (T/m).
    mot_type : str
        "dc" or "rf".
    s0 : float
        Saturation parameter.
    dz, dv : float
        Finite difference steps.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Scan results with optimal parameters.
    """
    delta_scan = param_ranges["delta"]

    if mot_type == "dc":
        split_scan = param_ranges.get("split", np.array([0.0]))
        return _optimize_params_raw(
            mol_data, delta_scan, split_scan, s0, B_gradient,
            dz=dz, dv=dv, Gamma_eff_factor=Gamma_eff_factor,
            verbose=verbose)

    elif mot_type == "rf":
        # For RF MOT, scan detuning only.
        # Each evaluation requires two phases (both computed on GPU).
        from molmot.obe.fields import make_rf_mot_beams

        n_d = len(delta_scan)
        mass = float(mol_data.mass)
        K_arr = np.zeros(n_d, dtype=np.float64)
        beta_arr = np.zeros(n_d, dtype=np.float64)

        v_arr = np.array([-dv, 0.0, dv], dtype=np.float64)
        z_arr = np.array([-dz, 0.0, dz], dtype=np.float64)

        t0 = time.time()
        for i_d, det in enumerate(delta_scan):
            # Phase 0
            beams_0 = make_rf_mot_beams(
                mol_data, delta_Gamma=det, s0=s0, phase=0)
            F0, _, _ = solve_rate_equations_batch(
                mol_data, beams_0, v_arr, z_arr, B_gradient,
                Gamma_eff_factor)

            # Phase 1
            beams_1 = make_rf_mot_beams(
                mol_data, delta_Gamma=det, s0=s0, phase=1)
            F1, _, _ = solve_rate_equations_batch(
                mol_data, beams_1, v_arr, z_arr, -B_gradient,
                Gamma_eff_factor)

            F_avg = 0.5 * (F0 + F1)

            # Spring constant at v=0 (index 1)
            k_spring = -(F_avg[1, 2] - F_avg[1, 0]) / (2.0 * dz)
            # Damping at z=0 (index 1)
            alpha = -(F_avg[2, 1] - F_avg[0, 1]) / (2.0 * dv)

            K_arr[i_d] = k_spring
            beta_arr[i_d] = alpha / mass

            if verbose and (i_d + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i_d + 1) * (n_d - i_d - 1)
                print(f"  {i_d + 1}/{n_d} ({elapsed:.1f}s, "
                      f"~{eta:.0f}s remaining)")

        idx_best = np.argmax(K_arr * (K_arr > 0) * (beta_arr > 0))

        if verbose:
            elapsed = time.time() - t0
            print(f"  Scan complete: {n_d} points in {elapsed:.1f}s")

        return {
            "delta_scan": delta_scan,
            "K_arr": K_arr,
            "beta_arr": beta_arr,
            "delta_best": float(delta_scan[idx_best]),
            "k_best": float(K_arr[idx_best]),
            "beta_best": float(beta_arr[idx_best]),
        }

    else:
        raise ValueError(f"Unknown mot_type: {mot_type}")


def force_map_2d_cuda(
    simulator,
    v_arr: np.ndarray,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full 2D force map F(v, z) on GPU.

    Drop-in replacement for manual double-loop over force evaluations.

    Parameters
    ----------
    simulator : DCMOTSimulator or RFMOTSimulator
    v_arr : np.ndarray, shape (N_v,)
        Velocity grid (m/s).
    z_arr : np.ndarray, shape (N_z,)
        Position grid (m).

    Returns
    -------
    force_map : np.ndarray, shape (N_v, N_z)
    pop_map : np.ndarray, shape (N_v, N_z, n_ground)
    scatter_map : np.ndarray, shape (N_v, N_z)
    """
    from molmot.mot.simulator import RFMOTSimulator, DCMOTSimulator

    if isinstance(simulator, DCMOTSimulator):
        return _force_map_2d_raw(
            simulator.mol_data, simulator._beams,
            v_arr, z_arr, simulator.B_gradient,
            simulator.Gamma_eff_factor)

    elif isinstance(simulator, RFMOTSimulator):
        from molmot.obe.fields import make_rf_mot_beams
        mol = simulator.mol_data

        beams_0 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=0)
        F0, pop0, R0 = _force_map_2d_raw(
            mol, beams_0, v_arr, z_arr,
            simulator.B_gradient, simulator.Gamma_eff_factor)

        beams_1 = make_rf_mot_beams(
            mol, delta_Gamma=simulator.delta_Gamma,
            s0=simulator.s0, phase=1)
        F1, pop1, R1 = _force_map_2d_raw(
            mol, beams_1, v_arr, z_arr,
            -simulator.B_gradient, simulator.Gamma_eff_factor)

        return (0.5 * (F0 + F1),
                0.5 * (pop0 + pop1),
                0.5 * (R0 + R1))

    else:
        raise TypeError(
            f"Unknown simulator type: {type(simulator).__name__}")
