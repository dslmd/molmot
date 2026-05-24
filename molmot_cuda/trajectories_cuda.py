"""
CUDA-accelerated 1D trajectory integration.

One CUDA thread per particle.  Each thread runs a full Euler loop
that evaluates the rate-equation force in-place at every step --
exactly the algorithm in ``molmot.propagation.trajectories_jit
._trajectory_1d_jit`` but parallelised across particles on the GPU.

Two entry points:
  * ``simulate_trajectory_cuda(simulator, z0, v0, t_max, dt, ...)``
    -- single particle, optionally returns the full trajectory.
  * ``simulate_trajectories_cuda(simulator, z0_arr, v0_arr, t_max,
    dt, ...)`` -- batch of N particles, returns final (z, v) plus
    optional full trajectories.

Both reuse molecular/beam data uploads via ``CUDAMolData``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
    _prepare_mol_data,
    _prepare_beam_data,
)


# =====================================================================
# CUDA kernel: Euler integration of N independent 1D trajectories
# =====================================================================

_TRAJECTORY_KERNEL_SOURCE = r"""
extern "C" __global__
void traj_kernel(
    // --- system sizes ---
    const int n_ground,
    const int n_excited,
    const int n_beams,
    const double Gamma,
    const double k_wave,
    const double Gamma_eff,
    const double hbar_val,
    // --- molecular data (read-only) ---
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
    const int N_traj,
    const double* __restrict__ z0_arr,
    const double* __restrict__ v0_arr,
    const double B_gradient,
    const double mass,
    const double dt,
    const int n_steps,
    const double z_escape,
    // --- save-trajectory toggle ---
    const int save_full,                // 0 or 1
    double* __restrict__ z_traj_out,    // [N_traj * n_steps]  (or unused)
    double* __restrict__ v_traj_out,    // [N_traj * n_steps]  (or unused)
    // --- always-written outputs ---
    double* __restrict__ z_final_out,   // [N_traj]
    double* __restrict__ v_final_out,   // [N_traj]
    int*    __restrict__ n_actual_out   // [N_traj]   (step count run, ==n_steps if not escaped)
)
{
    int itraj = blockIdx.x * blockDim.x + threadIdx.x;
    if (itraj >= N_traj) return;

    int n_g = n_ground;
    int n_e = n_excited;
    double inv_2pi = 1.0 / (2.0 * 3.14159265358979323846);
    double gamma_half = Gamma / 2.0;
    double gamma_half_sq = gamma_half * gamma_half;

    double z = z0_arr[itraj];
    double v = v0_arr[itraj];

    if (save_full) {
        z_traj_out[itraj * n_steps + 0] = z;
        v_traj_out[itraj * n_steps + 0] = v;
    }

    int n_actual = n_steps;
    int escaped = 0;

    for (int step = 1; step < n_steps; step++) {
        // ---- Zeeman-shifted energies (use previous-step z) ----
        double B_gauss = B_gradient * z * 1.0e4;
        double Es[MAX_STATES];
        for (int i = 0; i < n_g + n_e; i++) {
            Es[i] = energies[i] + B_gauss * zeeman_z_diag[i] * Gamma * inv_2pi;
        }

        // ---- Saturation totals ----
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
                    double L = gamma_half_sq /
                               (delta_eff * delta_eff + gamma_half_sq);
                    s_total[ig] += s0 * d2 * L;
                }
            }
        }

        // ---- Excitation rates + per-beam force contributions ----
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
                    double L = gamma_half_sq /
                               (delta_eff * delta_eff + gamma_half_sq);
                    double rate = gamma_half * s0 * d2 * L
                                  * Gamma_eff * sat_corr;
                    int ige = ig * n_e + ie;
                    R_sum[ige] += rate;
                    F_total_arr[ige] += hbar_val * k_wave * kdir * rate;
                }
            }
        }

        // ---- Population matrix M[n_g x n_g] ----
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

        // ---- Gaussian elimination with partial pivoting ----
        double A[MAX_GROUND * MAX_GROUND];
        for (int i = 0; i < n_g * n_g; i++) A[i] = M[i];

        for (int col = 0; col < n_g; col++) {
            double max_val = fabs(A[col * n_g + col]);
            int max_row = col;
            for (int row = col + 1; row < n_g; row++) {
                double val = fabs(A[row * n_g + col]);
                if (val > max_val) { max_val = val; max_row = row; }
            }
            if (max_row != col) {
                for (int j = 0; j < n_g; j++) {
                    double tmp = A[col * n_g + j];
                    A[col * n_g + j] = A[max_row * n_g + j];
                    A[max_row * n_g + j] = tmp;
                }
                double tmp_b = b[col]; b[col] = b[max_row]; b[max_row] = tmp_b;
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

        // ---- Back substitution ----
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

        // ---- Net force ----
        double force = 0.0;
        for (int ig = 0; ig < n_g; ig++) {
            for (int ie = 0; ie < n_e; ie++) {
                force += p[ig] * F_total_arr[ig * n_e + ie];
            }
        }

        // ---- Euler step (matches CPU: v(j) = v(j-1) + a*dt; z(j) = z(j-1) + v(j)*dt) ----
        v += (force / mass) * dt;
        z += v * dt;

        if (save_full) {
            z_traj_out[itraj * n_steps + step] = z;
            v_traj_out[itraj * n_steps + step] = v;
        }

        if (fabs(z) > z_escape) {
            n_actual = step;
            // Fill remaining slots with the escape value (matches CPU
            // behaviour: z_arr[j:] = z_arr[j], v_arr[j:] = v_arr[j])
            if (save_full) {
                for (int s = step + 1; s < n_steps; s++) {
                    z_traj_out[itraj * n_steps + s] = z;
                    v_traj_out[itraj * n_steps + s] = v;
                }
            }
            escaped = 1;
            break;
        }
    }

    z_final_out[itraj] = z;
    v_final_out[itraj] = v;
    n_actual_out[itraj] = n_actual;
    (void)escaped;
}
"""


_traj_kernel_cache: dict = {}


def _get_traj_kernel(n_ground: int, n_excited: int, n_beams: int):
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
        _traj_kernel_cache[key] = cp.RawKernel(source, "traj_kernel")
    return _traj_kernel_cache[key]


# =====================================================================
# Public batch entry point
# =====================================================================

def simulate_trajectories_cuda(
    simulator,
    z0_arr: np.ndarray,
    v0_arr: np.ndarray,
    t_max: float,
    dt: float = 1e-6,
    z_escape: float = 0.015,
    save_full: bool = False,
    cuda_data: Optional[CUDAMolData] = None,
    block_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Integrate N independent 1D trajectories in parallel on the GPU.

    Parameters
    ----------
    simulator : DCMOTSimulator
        Must expose ``mol_data``, ``_beams``, ``B_gradient`` and
        ``Gamma_eff_factor``.
    z0_arr, v0_arr : np.ndarray of shape (N,)
        Per-particle initial positions (m) and velocities (m/s).
    t_max : float
        Total simulation time (s).
    dt : float
        Time step (s).  Default 1e-6.
    z_escape : float
        Escape distance (m); trajectory truncates when ``|z| > z_escape``.
    save_full : bool
        If True, return per-step (z, v) trajectories of shape
        (N, n_steps).  Memory: 16 bytes per particle per step.
    cuda_data : CUDAMolData, optional
        Pre-uploaded mol/beam data.  Pass to avoid repeated transfers.
    block_size : int
        CUDA threads per block (1 thread = 1 trajectory).

    Returns
    -------
    z_final : np.ndarray, shape (N,)
    v_final : np.ndarray, shape (N,)
    n_actual : np.ndarray, shape (N,)
        Step count actually run before escape (== n_steps if trapped).
    z_traj  : np.ndarray, shape (N, n_steps)  or None
    v_traj  : np.ndarray, shape (N, n_steps)  or None
    """
    z0_arr = np.ascontiguousarray(z0_arr, dtype=np.float64)
    v0_arr = np.ascontiguousarray(v0_arr, dtype=np.float64)
    assert z0_arr.shape == v0_arr.shape and z0_arr.ndim == 1
    N = z0_arr.size
    n_steps = int(t_max / dt)

    mol_data = simulator.mol_data
    beams = simulator._beams
    Gamma_eff = getattr(simulator, "Gamma_eff_factor", 1.0)

    if cuda_data is None:
        cuda_data = CUDAMolData(mol_data, beams, Gamma_eff)
    kernel = _get_traj_kernel(cuda_data.n_ground, cuda_data.n_excited,
                              cuda_data.n_beams)

    d_z0 = cp.asarray(z0_arr)
    d_v0 = cp.asarray(v0_arr)
    d_zf = cp.zeros(N, dtype=cp.float64)
    d_vf = cp.zeros(N, dtype=cp.float64)
    d_nactual = cp.zeros(N, dtype=cp.int32)

    if save_full:
        d_ztraj = cp.empty(N * n_steps, dtype=cp.float64)
        d_vtraj = cp.empty(N * n_steps, dtype=cp.float64)
    else:
        # Pass a single-element placeholder; the kernel won't read it
        d_ztraj = cp.empty(1, dtype=cp.float64)
        d_vtraj = cp.empty(1, dtype=cp.float64)

    block = (block_size,)
    grid = ((N + block_size - 1) // block_size,)

    kernel(
        grid, block,
        (
            np.int32(cuda_data.n_ground),
            np.int32(cuda_data.n_excited),
            np.int32(cuda_data.n_beams),
            np.float64(cuda_data.Gamma),
            np.float64(cuda_data.k),
            np.float64(cuda_data.Gamma_eff),
            np.float64(cuda_data.hbar_val),
            cuda_data.d_energies,
            cuda_data.d_zeeman,
            cuda_data.d_dsq,
            cuda_data.d_BR,
            cuda_data.d_dirs,
            cuda_data.d_freqs,
            cuda_data.d_pols,
            cuda_data.d_s0s,
            np.int32(N),
            d_z0,
            d_v0,
            np.float64(simulator.B_gradient),
            np.float64(cuda_data.mass),
            np.float64(dt),
            np.int32(n_steps),
            np.float64(z_escape),
            np.int32(1 if save_full else 0),
            d_ztraj,
            d_vtraj,
            d_zf,
            d_vf,
            d_nactual,
        )
    )

    z_final = cp.asnumpy(d_zf)
    v_final = cp.asnumpy(d_vf)
    n_actual = cp.asnumpy(d_nactual)

    if save_full:
        z_traj = cp.asnumpy(d_ztraj).reshape(N, n_steps)
        v_traj = cp.asnumpy(d_vtraj).reshape(N, n_steps)
        return z_final, v_final, n_actual, z_traj, v_traj

    return z_final, v_final, n_actual, None, None


def simulate_trajectory_cuda(
    simulator,
    z0: float,
    v0: float,
    t_max: float,
    dt: float = 1e-6,
    z_escape: float = 0.015,
    cuda_data: Optional[CUDAMolData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-particle 1D trajectory on the GPU.

    Same return shape as ``molmot.propagation.trajectories.simulate_trajectory``
    -- ``(t_arr, z_arr, v_arr)``, all of length ``int(t_max / dt)``.
    """
    z_final, v_final, n_actual, z_traj, v_traj = simulate_trajectories_cuda(
        simulator,
        np.array([z0]), np.array([v0]),
        t_max=t_max, dt=dt, z_escape=z_escape,
        save_full=True, cuda_data=cuda_data,
    )
    n_steps = int(t_max / dt)
    t_arr = np.arange(n_steps) * dt
    return t_arr, z_traj[0], v_traj[0]
