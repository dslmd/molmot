"""
CUDA-accelerated multi-level rate-equation solver for molecular MOT force.

Port of molmot.obe.rate_equations / rate_equations_jit to CuPy RawKernel.

The key insight: the rate equation solver is called thousands of times at
different (v, z) grid points.  On GPU, we batch ALL (v, z) points into a
single kernel launch — one CUDA thread per (v, z) pair.

Each thread independently:
  1. Computes Zeeman-shifted energies (n_states values)
  2. Accumulates excitation rates for all beams (n_beams x n_g x n_e)
  3. Builds the n_g x n_g population matrix
  4. Solves by Gaussian elimination with partial pivoting
  5. Sums force and scattering rate weighted by populations

The n_g x n_g system (12x12 for SrOH) is small enough to fit entirely
in registers / local memory.

Usage
-----
>>> from molmot_cuda.rate_equations_cuda import force_map_2d_cuda
>>> F, pop, R = force_map_2d_cuda(mol_data, beams, v_arr, z_arr, B_grad)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy is required for CUDA-accelerated computation. "
        "Install with: pip install cupy-cuda12x  (or the variant "
        "matching your CUDA toolkit version)."
    )


# =====================================================================
# CUDA kernel source
# =====================================================================

# The kernel is parameterised by MAX_GROUND, MAX_EXCITED, MAX_BEAMS,
# MAX_STATES which are #defined before compilation.  This allows the
# arrays to live in registers (stack) rather than dynamic allocation.

_KERNEL_SOURCE = r"""
extern "C" __global__
void rate_eq_kernel(
    // --- grid dimensions ---
    const int N_v,
    const int N_z,
    // --- molecular data (constant across grid) ---
    const int n_ground,
    const int n_excited,
    const int n_states,
    const int n_beams,
    const double Gamma,          // natural linewidth (rad/s)
    const double k_wave,         // wavenumber (1/m)
    const double Gamma_eff,      // effective Gamma factor
    const double hbar_val,       // hbar (J*s)
    // --- per-state arrays ---
    const double* __restrict__ energies,      // [n_states]
    const double* __restrict__ zeeman_z_diag, // [n_states]
    const double* __restrict__ d_squared,     // [n_ground * n_excited * 3]
    const double* __restrict__ BR,            // [n_excited * n_ground]
    // --- beam arrays ---
    const double* __restrict__ beam_dirs,     // [n_beams]  (kdir_z)
    const double* __restrict__ beam_freqs,    // [n_beams]  (freq_offset, Hz)
    const double* __restrict__ beam_pols,     // [n_beams]  (polarization index)
    const double* __restrict__ beam_s0s,      // [n_beams]  (saturation param)
    // --- velocity and position grids ---
    const double* __restrict__ v_arr,         // [N_v]
    const double* __restrict__ z_arr,         // [N_z]
    const double B_gradient,                  // T/m
    // --- outputs ---
    double* __restrict__ force_out,           // [N_v * N_z]
    double* __restrict__ pop_out,             // [N_v * N_z * n_ground]
    double* __restrict__ scatter_out          // [N_v * N_z]
)
{
    // 2D grid: blockIdx.x * blockDim.x + threadIdx.x -> v index
    //          blockIdx.y * blockDim.y + threadIdx.y -> z index
    int iv = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (iv >= N_v || iz >= N_z) return;

    double v = v_arr[iv];
    double z = z_arr[iz];
    int idx = iv * N_z + iz;   // linear output index

    int n_g = n_ground;
    int n_e = n_excited;

    // --- Zeeman-shifted energies ---
    double B_gauss = B_gradient * z * 1.0e4;
    double gamma_half = Gamma / 2.0;
    double gamma_half_sq = gamma_half * gamma_half;
    double inv_2pi = 1.0 / (2.0 * 3.14159265358979323846);

    double Es[MAX_STATES];
    for (int i = 0; i < n_g + n_e; i++) {
        Es[i] = energies[i] + B_gauss * zeeman_z_diag[i] * Gamma * inv_2pi;
    }

    // --- First pass: compute saturation totals per ground state ---
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
                double delta_eff = (freq + doppler - omega_trans) * 2.0 * 3.14159265358979323846;
                double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);

                s_total[ig] += s0 * d2 * L;
            }
        }
    }

    // --- Second pass: excitation rates with saturation correction ---
    // Accumulate R_sum[ig][ie] and F_total[ig][ie]
    double R_sum[MAX_GROUND * MAX_EXCITED];
    double F_total[MAX_GROUND * MAX_EXCITED];
    for (int i = 0; i < n_g * n_e; i++) {
        R_sum[i] = 0.0;
        F_total[i] = 0.0;
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
                double delta_eff = (freq + doppler - omega_trans) * 2.0 * 3.14159265358979323846;
                double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);

                double rate = gamma_half * s0 * d2 * L * Gamma_eff * sat_corr;
                int ige = ig * n_e + ie;
                R_sum[ige] += rate;
                F_total[ige] += hbar_val * k_wave * kdir * rate;
            }
        }
    }

    // --- Population matrix M[n_g x n_g] ---
    double M[MAX_GROUND * MAX_GROUND];
    for (int i = 0; i < n_g * n_g; i++) M[i] = 0.0;

    for (int ig = 0; ig < n_g; ig++) {
        // Loss from excitation
        for (int ie = 0; ie < n_e; ie++) {
            M[ig * n_g + ig] -= R_sum[ig * n_e + ie];
        }
        // Gain from decay of other ground states
        for (int ik = 0; ik < n_g; ik++) {
            for (int ie = 0; ie < n_e; ie++) {
                M[ig * n_g + ik] += R_sum[ik * n_e + ie] * BR[ie * n_g + ig];
            }
        }
    }

    // Replace last row with normalisation constraint
    for (int j = 0; j < n_g; j++) {
        M[(n_g - 1) * n_g + j] = 1.0;
    }
    double b[MAX_GROUND];
    for (int i = 0; i < n_g; i++) b[i] = 0.0;
    b[n_g - 1] = 1.0;

    // --- Gaussian elimination with partial pivoting ---
    double A[MAX_GROUND * MAX_GROUND];
    for (int i = 0; i < n_g * n_g; i++) A[i] = M[i];

    for (int col = 0; col < n_g; col++) {
        // Find pivot
        double max_val = fabs(A[col * n_g + col]);
        int max_row = col;
        for (int row = col + 1; row < n_g; row++) {
            double val = fabs(A[row * n_g + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        // Swap rows
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
        // Eliminate below
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

    // Clamp and normalise
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

    // --- Force and scattering rate ---
    double force = 0.0;
    double R_scatter = 0.0;
    for (int ig = 0; ig < n_g; ig++) {
        for (int ie = 0; ie < n_e; ie++) {
            int ige = ig * n_e + ie;
            force += p[ig] * F_total[ige];
            R_scatter += p[ig] * R_sum[ige];
        }
    }

    force_out[idx] = force;
    scatter_out[idx] = R_scatter;
    for (int ig = 0; ig < n_g; ig++) {
        pop_out[idx * n_g + ig] = p[ig];
    }
}
"""


# =====================================================================
# Kernel compilation cache
# =====================================================================

_kernel_cache = {}


def _get_kernel(n_ground: int, n_excited: int, n_beams: int) -> cp.RawKernel:
    """
    Get (or compile) the CUDA kernel for the given system dimensions.

    The kernel is compiled once per unique (n_ground, n_excited, n_beams)
    combination and cached for reuse.

    Parameters
    ----------
    n_ground : int
        Number of ground states.
    n_excited : int
        Number of excited states.
    n_beams : int
        Number of laser beams.

    Returns
    -------
    cp.RawKernel
        Compiled CUDA kernel.
    """
    key = (n_ground, n_excited, n_beams)
    if key not in _kernel_cache:
        n_states = n_ground + n_excited
        defines = (
            f"#define MAX_GROUND {n_ground}\n"
            f"#define MAX_EXCITED {n_excited}\n"
            f"#define MAX_BEAMS {n_beams}\n"
            f"#define MAX_STATES {n_states}\n"
        )
        source = defines + _KERNEL_SOURCE
        kernel = cp.RawKernel(source, "rate_eq_kernel")
        _kernel_cache[key] = kernel
    return _kernel_cache[key]


# =====================================================================
# Data preparation helpers
# =====================================================================

def _prepare_mol_data(mol_data):
    """
    Extract and flatten molecular data arrays for GPU transfer.

    Returns all arrays as contiguous float64 numpy arrays ready
    for cp.asarray().

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data container.

    Returns
    -------
    dict
        Dictionary of GPU-ready arrays.
    """
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited

    # Branching ratios: BR[ie, ig]
    d_sq = mol_data.d_squared  # (n_g, n_e, 3)
    BR = np.zeros((n_e, n_g), dtype=np.float64)
    for ie in range(n_e):
        total = 0.0
        for ig in range(n_g):
            total += np.sum(d_sq[ig, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq[ig, ie, :]) / total

    return {
        "energies": np.ascontiguousarray(mol_data.energies, dtype=np.float64),
        "zeeman_z_diag": np.ascontiguousarray(mol_data.zeeman_z_diag, dtype=np.float64),
        "d_squared": np.ascontiguousarray(d_sq, dtype=np.float64),
        "BR": np.ascontiguousarray(BR, dtype=np.float64),
        "Gamma": float(mol_data.Gamma),
        "k": float(mol_data.k),
        "n_ground": int(n_g),
        "n_excited": int(n_e),
        "n_states": int(mol_data.n_states),
    }


def _prepare_beam_data(beams):
    """
    Extract beam parameters into flat arrays for GPU transfer.

    Parameters
    ----------
    beams : list of LaserBeam

    Returns
    -------
    dict
        Dictionary with beam_dirs, beam_freqs, beam_pols, beam_s0s arrays.
    """
    n_beams = len(beams)
    beam_dirs = np.array([b.kdir_z for b in beams], dtype=np.float64)
    beam_freqs = np.array([b.freq_offset for b in beams], dtype=np.float64)
    beam_pols = np.array([b.polarization for b in beams], dtype=np.float64)
    beam_s0s = np.array([b.s0 for b in beams], dtype=np.float64)

    return {
        "n_beams": n_beams,
        "beam_dirs": beam_dirs,
        "beam_freqs": beam_freqs,
        "beam_pols": beam_pols,
        "beam_s0s": beam_s0s,
    }


class CUDAMolData:
    """
    GPU-resident molecular and beam data for repeated kernel launches.

    Transfers molecular constants and beam parameters to GPU memory
    once; reuse across multiple solve calls.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data container.
    beams : list of LaserBeam
        Laser beam configuration.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor (default 1.0).
    """

    def __init__(self, mol_data, beams, Gamma_eff_factor: float = 1.0):
        mol = _prepare_mol_data(mol_data)
        bm = _prepare_beam_data(beams)

        self.n_ground = mol["n_ground"]
        self.n_excited = mol["n_excited"]
        self.n_states = mol["n_states"]
        self.n_beams = bm["n_beams"]
        self.Gamma = mol["Gamma"]
        self.k = mol["k"]
        self.Gamma_eff = float(Gamma_eff_factor)
        self.hbar_val = 1.0545718e-34
        self.mass = float(mol_data.mass)

        # Transfer to GPU
        self.d_energies = cp.asarray(mol["energies"])
        self.d_zeeman = cp.asarray(mol["zeeman_z_diag"])
        self.d_dsq = cp.asarray(mol["d_squared"])
        self.d_BR = cp.asarray(mol["BR"])
        self.d_dirs = cp.asarray(bm["beam_dirs"])
        self.d_freqs = cp.asarray(bm["beam_freqs"])
        self.d_pols = cp.asarray(bm["beam_pols"])
        self.d_s0s = cp.asarray(bm["beam_s0s"])

        # Compile kernel for this system size
        self.kernel = _get_kernel(self.n_ground, self.n_excited, self.n_beams)


# =====================================================================
# Core batch solver
# =====================================================================

def solve_rate_equations_batch(
    mol_data,
    beams,
    v_arr: np.ndarray,
    z_arr: np.ndarray,
    B_gradient: float,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve steady-state rate equations for all (v, z) grid points on GPU.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data container (same interface as CPU version).
    beams : list of LaserBeam
        Laser beam configuration.
    v_arr : np.ndarray, shape (N_v,)
        Array of velocities (m/s).
    z_arr : np.ndarray, shape (N_z,)
        Array of positions (m).
    B_gradient : float
        Magnetic field gradient dBz/dz in T/m.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data.  If None, data is transferred each call.
        Pass this for repeated calls with same mol_data/beams to
        avoid redundant CPU->GPU transfers.

    Returns
    -------
    force_map : np.ndarray, shape (N_v, N_z)
        Radiation pressure force in Newtons.
    pop_map : np.ndarray, shape (N_v, N_z, n_ground)
        Steady-state ground-state populations.
    scatter_map : np.ndarray, shape (N_v, N_z)
        Total scattering rate in rad/s.
    """
    if cuda_data is None:
        cuda_data = CUDAMolData(mol_data, beams, Gamma_eff_factor)

    N_v = len(v_arr)
    N_z = len(z_arr)
    n_g = cuda_data.n_ground

    # Transfer grid arrays to GPU
    d_v = cp.asarray(np.ascontiguousarray(v_arr, dtype=np.float64))
    d_z = cp.asarray(np.ascontiguousarray(z_arr, dtype=np.float64))

    # Allocate output arrays on GPU
    d_force = cp.zeros(N_v * N_z, dtype=cp.float64)
    d_pop = cp.zeros(N_v * N_z * n_g, dtype=cp.float64)
    d_scatter = cp.zeros(N_v * N_z, dtype=cp.float64)

    # Launch configuration: 2D grid
    block = (16, 16)
    grid = ((N_v + block[0] - 1) // block[0],
            (N_z + block[1] - 1) // block[1])

    cuda_data.kernel(
        grid, block,
        (
            np.int32(N_v),
            np.int32(N_z),
            np.int32(cuda_data.n_ground),
            np.int32(cuda_data.n_excited),
            np.int32(cuda_data.n_states),
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
            d_v,
            d_z,
            np.float64(B_gradient),
            d_force,
            d_pop,
            d_scatter,
        )
    )

    # Transfer results back to CPU
    force_map = cp.asnumpy(d_force).reshape(N_v, N_z)
    pop_map = cp.asnumpy(d_pop).reshape(N_v, N_z, n_g)
    scatter_map = cp.asnumpy(d_scatter).reshape(N_v, N_z)

    return force_map, pop_map, scatter_map


# =====================================================================
# Convenience wrappers matching CPU API
# =====================================================================

def force_vs_z_cuda(
    mol_data,
    beams,
    z_arr: np.ndarray,
    v: float,
    B_gradient: float,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force as a function of position at fixed velocity (GPU).

    Equivalent to ``molmot.mot.force_scan.force_vs_z`` but computed
    entirely on GPU in a single batched kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    z_arr : np.ndarray
        Array of positions (m).
    v : float
        Fixed velocity (m/s).
    B_gradient : float
        Magnetic field gradient (T/m).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data for reuse.

    Returns
    -------
    F_arr : np.ndarray, shape (N_z,)
        Force at each position (N).
    pop_arr : np.ndarray, shape (N_z, n_ground)
        Ground-state populations at each position.
    R_arr : np.ndarray, shape (N_z,)
        Scattering rate at each position (rad/s).
    """
    v_arr = np.array([v], dtype=np.float64)
    F_map, pop_map, R_map = solve_rate_equations_batch(
        mol_data, beams, v_arr, z_arr, B_gradient,
        Gamma_eff_factor, cuda_data)
    return F_map[0, :], pop_map[0, :, :], R_map[0, :]


def force_vs_v_cuda(
    mol_data,
    beams,
    v_arr: np.ndarray,
    z: float,
    B_gradient: float,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> np.ndarray:
    """
    Compute force as a function of velocity at fixed position (GPU).

    Equivalent to ``molmot.mot.force_scan.force_vs_v`` but computed
    entirely on GPU in a single batched kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    v_arr : np.ndarray
        Array of velocities (m/s).
    z : float
        Fixed position (m).
    B_gradient : float
        Magnetic field gradient (T/m).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data for reuse.

    Returns
    -------
    F_arr : np.ndarray, shape (N_v,)
        Force at each velocity (N).
    """
    z_arr = np.array([z], dtype=np.float64)
    F_map, _, _ = solve_rate_equations_batch(
        mol_data, beams, v_arr, z_arr, B_gradient,
        Gamma_eff_factor, cuda_data)
    return F_map[:, 0]


def force_map_2d_cuda(
    mol_data,
    beams,
    v_arr: np.ndarray,
    z_arr: np.ndarray,
    B_gradient: float,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full 2D force map F(v, z) on GPU.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    v_arr : np.ndarray, shape (N_v,)
        Velocity grid (m/s).
    z_arr : np.ndarray, shape (N_z,)
        Position grid (m).
    B_gradient : float
        Magnetic field gradient (T/m).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data for reuse.

    Returns
    -------
    force_map : np.ndarray, shape (N_v, N_z)
        Force in Newtons.
    pop_map : np.ndarray, shape (N_v, N_z, n_ground)
        Ground-state populations.
    scatter_map : np.ndarray, shape (N_v, N_z)
        Scattering rate in rad/s.
    """
    return solve_rate_equations_batch(
        mol_data, beams, v_arr, z_arr, B_gradient,
        Gamma_eff_factor, cuda_data)


def spring_constant_cuda(
    mol_data,
    beams,
    B_gradient: float,
    z0: float = 0.0,
    dz: float = 0.3e-3,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> float:
    """
    Compute the spring constant k = -dF/dz at position z0 (GPU).

    Uses a two-point finite difference computed in a single kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    B_gradient : float
        Magnetic field gradient (T/m).
    z0 : float
        Equilibrium position (m).
    dz : float
        Finite difference step (m).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data for reuse.

    Returns
    -------
    k : float
        Spring constant (N/m).  Positive = restoring (trapping).
    """
    v_arr = np.array([0.0], dtype=np.float64)
    z_arr = np.array([z0 - dz, z0 + dz], dtype=np.float64)
    F_map, _, _ = solve_rate_equations_batch(
        mol_data, beams, v_arr, z_arr, B_gradient,
        Gamma_eff_factor, cuda_data)
    Fm = F_map[0, 0]
    Fp = F_map[0, 1]
    return -(Fp - Fm) / (2.0 * dz)


def damping_coefficient_cuda(
    mol_data,
    beams,
    B_gradient: float,
    z0: float = 0.0,
    dv: float = 0.1,
    Gamma_eff_factor: float = 1.0,
    cuda_data: Optional[CUDAMolData] = None,
) -> float:
    """
    Compute the damping coefficient alpha = -dF/dv at position z0 (GPU).

    Uses a two-point finite difference computed in a single kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    B_gradient : float
        Magnetic field gradient (T/m).
    z0 : float
        Position (m).
    dv : float
        Finite difference step (m/s).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    cuda_data : CUDAMolData, optional
        Pre-built GPU data for reuse.

    Returns
    -------
    alpha : float
        Friction coefficient dF/dv (N*s/m).
    """
    v_arr = np.array([-dv, dv], dtype=np.float64)
    z_arr = np.array([z0], dtype=np.float64)
    F_map, _, _ = solve_rate_equations_batch(
        mol_data, beams, v_arr, z_arr, B_gradient,
        Gamma_eff_factor, cuda_data)
    Fvm = F_map[0, 0]
    Fvp = F_map[1, 0]
    return -(Fvp - Fvm) / (2.0 * dv)


def optimize_parameters_cuda(
    mol_data,
    delta_arr: np.ndarray,
    split_arr: np.ndarray,
    s0: float,
    B_gradient: float,
    dz: float = 0.3e-3,
    dv: float = 0.1,
    Gamma_eff_factor: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    2D parameter scan over (detuning, split) on GPU.

    For each (delta, split) combination, builds the DC MOT beam
    configuration, computes the spring constant and damping coefficient,
    and finds the optimal parameters.

    This is massively faster than the CPU version because the inner
    finite-difference evaluations (4 force evaluations per parameter
    point) are batched into a single GPU kernel launch per parameter
    combination.  Additionally, multiple parameter points can be
    evaluated in rapid succession with minimal kernel launch overhead.

    Parameters
    ----------
    mol_data : MolecularData
    delta_arr : np.ndarray
        Array of detunings in units of Gamma.
    split_arr : np.ndarray
        Array of polarisation splits in units of Gamma.
    s0 : float
        Saturation parameter per frequency component per beam.
    B_gradient : float
        Magnetic field gradient (T/m).
    dz : float
        Finite difference step for spring constant (m).
    dv : float
        Finite difference step for damping coefficient (m/s).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Contains: delta_scan, split_scan, K_map, beta_map,
        delta_best, split_best, k_best, beta_best.
    """
    import time
    from molmot.obe.fields import make_dc_mot_beams

    n_d = len(delta_arr)
    n_s = len(split_arr)
    mass = float(mol_data.mass)

    K_map = np.zeros((n_d, n_s), dtype=np.float64)
    beta_map = np.zeros((n_d, n_s), dtype=np.float64)

    # Grid for finite differences: 2 v-points and 2 z-points
    # This lets us compute both k and alpha in ONE kernel launch
    v_arr = np.array([-dv, 0.0, dv], dtype=np.float64)
    z_arr = np.array([-dz, 0.0, dz], dtype=np.float64)

    t0 = time.time()
    total = n_d * n_s
    count = 0

    for i_d, det in enumerate(delta_arr):
        for i_s, sp in enumerate(split_arr):
            beams = make_dc_mot_beams(
                mol_data, delta_Gamma=det, split_Gamma=sp, s0=s0)

            F_map, _, _ = solve_rate_equations_batch(
                mol_data, beams, v_arr, z_arr, B_gradient,
                Gamma_eff_factor)

            # Spring constant: k = -(F(z0+dz) - F(z0-dz)) / (2*dz) at v=0
            # v=0 is index 1, z0-dz is index 0, z0+dz is index 2
            Fm_z = F_map[1, 0]  # v=0, z=-dz
            Fp_z = F_map[1, 2]  # v=0, z=+dz
            k_spring = -(Fp_z - Fm_z) / (2.0 * dz)

            # Damping: alpha = -(F(+dv) - F(-dv)) / (2*dv) at z=0
            # z=0 is index 1, v=-dv is index 0, v=+dv is index 2
            Fvm = F_map[0, 1]  # v=-dv, z=0
            Fvp = F_map[2, 1]  # v=+dv, z=0
            alpha = -(Fvp - Fvm) / (2.0 * dv)

            K_map[i_d, i_s] = k_spring
            beta_map[i_d, i_s] = alpha / mass

            count += 1
            if verbose and count % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  {count}/{total} ({elapsed:.1f}s, ~{eta:.0f}s remaining)")

    # Find optimum: maximum spring constant where both k > 0 and beta > 0
    merit = K_map.copy()
    merit[K_map <= 0] = 0
    merit[beta_map <= 0] = 0

    idx_best = np.unravel_index(np.argmax(merit), merit.shape)
    delta_best = delta_arr[idx_best[0]]
    split_best = split_arr[idx_best[1]]

    if verbose:
        elapsed = time.time() - t0
        print(f"  Scan complete: {total} points in {elapsed:.1f}s "
              f"({total / elapsed:.0f} pts/s)")
        print(f"  Best: delta={delta_best:.3f} Gamma, "
              f"split={split_best:.3f} Gamma")
        print(f"    k = {K_map[idx_best]:.4e} N/m, "
              f"beta = {beta_map[idx_best]:.1f} /s")

    return {
        "delta_scan": delta_arr,
        "split_scan": split_arr,
        "K_map": K_map,
        "beta_map": beta_map,
        "delta_best": float(delta_best),
        "split_best": float(split_best),
        "k_best": float(K_map[idx_best]),
        "beta_best": float(beta_map[idx_best]),
    }
