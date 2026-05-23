"""
CUDA-accelerated Floquet OBE sub-Doppler force calculator.

GPU-batched version of ``molmot.obe.floquet.FloquetOBE``.  All velocity
points are solved in parallel using CuPy batched linear algebra.

The Floquet expansion of the Lindblad master equation in a standing wave
gives a block-tridiagonal linear system for each velocity point.  The
system size is (2*n_max+1)*N^2, where N = n_ground + n_excited = 16
and n_max is the Floquet truncation order (typically 2--5).

For n_max=3, N=16: system size = 7*256 = 1792.  Solving 200 such
systems in parallel on the GPU (one per velocity point) gives a large
speedup over serial CPU solves.

References
----------
* Dalibard, Cohen-Tannoudji, JOSA B 6, 2023 (1989)
* Javanainen, PRE 47, 3547 (1993)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import _require_cupy, to_gpu, to_cpu

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HBAR = 1.054571817e-34


# ---------------------------------------------------------------------------
# Floquet block construction (CPU, done once)
# ---------------------------------------------------------------------------

def _build_floquet_blocks(mol_data, beam_pairs, Omega_Gamma,
                          Gamma_eff_factor=1.0):
    """
    Build the Liouvillian Fourier blocks L0, L_plus, L_minus on the CPU.

    These blocks are velocity-independent and are built once, then
    reused for all velocity points.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
    Omega_Gamma : float
    Gamma_eff_factor : float

    Returns
    -------
    L0 : np.ndarray, shape (N^2, N^2), complex128
    L_plus : np.ndarray, shape (N^2, N^2), complex128
    L_minus : np.ndarray, shape (N^2, N^2), complex128
    """
    N = mol_data.n_states
    N2 = N * N
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    Gamma = mol_data.Gamma
    Gamma_eff = Gamma * Gamma_eff_factor
    d = mol_data.tdm
    Omega0 = Omega_Gamma * Gamma
    Es_Hz = mol_data.energies.copy()
    E_e_mean = np.mean(Es_Hz[n_g:])

    # --- H0: rotating-frame detunings ---
    H0 = np.zeros((N, N), dtype=complex)
    laser_freqs = [bp['freq'] for bp in beam_pairs]

    for ig in range(n_g):
        omega_trans = E_e_mean - Es_Hz[ig]
        deltas = [lf - omega_trans for lf in laser_freqs]
        nearest_delta = min(deltas, key=abs)
        H0[ig, ig] = -2 * np.pi * nearest_delta

    for ie in range(n_e):
        ie_abs = ie + n_g
        H0[ie_abs, ie_abs] = 2 * np.pi * (Es_Hz[ie_abs] - E_e_mean)

    # --- H_plus, H_minus: standing-wave couplings ---
    H_plus = np.zeros((N, N), dtype=complex)
    H_minus = np.zeros((N, N), dtype=complex)

    for bp in beam_pairs:
        q_fwd = bp['q_fwd']
        q_bwd = bp['q_bwd']
        s0 = bp['s0']
        Om = Omega0 * np.sqrt(s0)

        for ig in range(n_g):
            for ie in range(n_e):
                ie_abs = ie + n_g
                d_fwd = abs(d[ig, ie_abs, q_fwd])
                if d_fwd > 1e-10:
                    coupling = Om * d_fwd / 2
                    H_plus[ie_abs, ig] += coupling
                    H_plus[ig, ie_abs] += np.conj(coupling)

                d_bwd = abs(d[ig, ie_abs, q_bwd])
                if d_bwd > 1e-10:
                    coupling = Om * d_bwd / 2
                    H_minus[ie_abs, ig] += coupling
                    H_minus[ig, ie_abs] += np.conj(coupling)

    # --- Liouvillian blocks ---
    I = np.eye(N, dtype=complex)

    # L0: coherent + dissipator
    L0 = -1j * (np.kron(I, H0) - np.kron(H0.T, I))

    d_sq = mol_data.d_squared
    for ie in range(n_e):
        ie_abs = ie + n_g
        total = sum(d_sq[ig, ie, q] for ig in range(n_g) for q in range(3))
        if total < 1e-30:
            continue
        for ig in range(n_g):
            for q in range(3):
                br = d_sq[ig, ie, q] / total
                if br < 1e-15:
                    continue
                Gc = Gamma_eff * br
                Lc = np.zeros((N, N), dtype=complex)
                Lc[ig, ie_abs] = np.sqrt(Gc)
                LdL = Lc.conj().T @ Lc
                L0 += np.kron(Lc.conj(), Lc)
                L0 -= 0.5 * np.kron(I, LdL)
                L0 -= 0.5 * np.kron(LdL.T, I)

    L_plus = -1j * (np.kron(I, H_plus) - np.kron(H_plus.T, I))
    L_minus = -1j * (np.kron(I, H_minus) - np.kron(H_minus.T, I))

    return L0, L_plus, L_minus, H_plus, H_minus


def _build_floquet_matrices_batch(L0, L_plus, L_minus, kv_arr, n_max, N):
    """
    Build batched Floquet matrices on the GPU for all velocity points.

    Parameters
    ----------
    L0, L_plus, L_minus : cp.ndarray, shape (N^2, N^2)
        Liouvillian blocks (already on GPU).
    kv_arr : cp.ndarray, shape (n_v,)
        k*v values for each velocity point.
    n_max : int
    N : int
        Number of states.

    Returns
    -------
    M_batch : cp.ndarray, shape (n_v, total_size, total_size)
    rhs_batch : cp.ndarray, shape (n_v, total_size)
    """
    N2 = N * N
    n_blocks = 2 * n_max + 1
    total_size = n_blocks * N2
    n_v = len(kv_arr)

    # Allocate batch
    M_batch = cp.zeros((n_v, total_size, total_size), dtype=cp.complex128)
    rhs_batch = cp.zeros((n_v, total_size), dtype=cp.complex128)

    # Build a template matrix (velocity-independent parts)
    M_template = cp.zeros((total_size, total_size), dtype=cp.complex128)

    for b in range(n_blocks):
        rs = b * N2
        re = (b + 1) * N2
        # Diagonal block: L0
        M_template[rs:re, rs:re] = L0
        # Sub-diagonal: L_plus
        if b > 0:
            cs = (b - 1) * N2
            ce = b * N2
            M_template[rs:re, cs:ce] = L_plus
        # Super-diagonal: L_minus
        if b < n_blocks - 1:
            cs = (b + 1) * N2
            ce = (b + 2) * N2
            M_template[rs:re, cs:ce] = L_minus

    # Broadcast template to all velocity points
    M_batch[:] = M_template[cp.newaxis, :, :]

    # Add velocity-dependent diagonal shift: -i*n*kv for each block
    I_N2 = cp.eye(N2, dtype=cp.complex128)
    for b in range(n_blocks):
        n_fourier = b - n_max
        if n_fourier == 0:
            continue
        rs = b * N2
        re = (b + 1) * N2
        # M_batch[:, rs:re, rs:re] -= 1j * n * kv * I_N2
        # Use broadcasting: kv_arr shape (n_v,) -> (n_v, 1, 1)
        shift = -1j * n_fourier * kv_arr[:, cp.newaxis, cp.newaxis] * I_N2[cp.newaxis, :, :]
        M_batch[:, rs:re, rs:re] += shift

    # Trace constraint for the n=0 block
    b0 = n_max
    row_trace = b0 * N2 + N2 - 1
    M_batch[:, row_trace, :] = 0.0
    for i in range(N):
        col_idx = b0 * N2 + i * N + i
        M_batch[:, row_trace, col_idx] = 1.0

    rhs_batch[:, row_trace] = 1.0

    return M_batch, rhs_batch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def force_vs_velocity_cuda(mol_data, beam_pairs, v_arr, B_gradient=0.0,
                           Omega_Gamma=1.0, n_max=3,
                           Gamma_eff_factor=1.0):
    """
    Compute the Floquet OBE force as a function of velocity on the GPU.

    All velocity points are solved in parallel using CuPy batched
    ``linalg.solve``.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data.
    beam_pairs : list of dict
        Beam pair configurations.  Each dict:
        ``{'freq': Hz, 'q_fwd': int, 'q_bwd': int, 's0': float}``
    v_arr : np.ndarray
        Velocities (m/s).
    B_gradient : float
        Magnetic field gradient (T/m).  Currently only z_macro=0 is
        supported (no Zeeman shift in the Floquet blocks).
    Omega_Gamma : float
        Peak Rabi frequency in units of Gamma.
    n_max : int
        Floquet truncation order.
    Gamma_eff_factor : float
        Scattering rate reduction factor.

    Returns
    -------
    F_arr : np.ndarray, shape (n_v,)
        Force at each velocity (N).
    rho0_arr : np.ndarray, shape (n_v, N, N)
        DC density matrix at each velocity.
    """
    _require_cupy()

    v_arr = np.asarray(v_arr, dtype=np.float64)
    n_v = len(v_arr)
    k = mol_data.k
    N = mol_data.n_states
    N2 = N * N
    n_blocks = 2 * n_max + 1

    # Build Floquet blocks on CPU, then transfer to GPU
    L0_cpu, Lp_cpu, Lm_cpu, H_plus_cpu, H_minus_cpu = _build_floquet_blocks(
        mol_data, beam_pairs, Omega_Gamma, Gamma_eff_factor)

    L0_d = cp.asarray(L0_cpu)
    Lp_d = cp.asarray(Lp_cpu)
    Lm_d = cp.asarray(Lm_cpu)
    H_plus_d = cp.asarray(H_plus_cpu)
    H_minus_d = cp.asarray(H_minus_cpu)

    # Replace exactly-zero velocities with a small offset (avoids singularity)
    v_eff = v_arr.copy()
    v_eff[np.abs(v_eff) < 1e-12] = 1e-6
    kv_arr_d = cp.asarray(k * v_eff)

    # Build batched Floquet matrices
    M_batch, rhs_batch = _build_floquet_matrices_batch(
        L0_d, Lp_d, Lm_d, kv_arr_d, n_max, N)

    # Batched solve: solve M_batch[i] @ x[i] = rhs_batch[i] for all i
    try:
        rho_vec_batch = cp.linalg.solve(M_batch, rhs_batch)
    except cp.linalg.LinAlgError:
        # Fallback: solve individually with lstsq
        rho_vec_batch = cp.zeros((n_v, n_blocks * N2), dtype=cp.complex128)
        for iv in range(n_v):
            rho_vec_batch[iv], _, _, _ = cp.linalg.lstsq(
                M_batch[iv], rhs_batch[iv], rcond=None)

    # Extract rho_0, rho_{+1}, rho_{-1} and compute forces
    F_arr = np.zeros(n_v, dtype=np.float64)
    rho0_arr = np.zeros((n_v, N, N), dtype=complex)

    b0 = n_max
    b_p1 = n_max + 1
    b_m1 = n_max - 1

    # Extract on GPU, then compute force
    for iv in range(n_v):
        rho_vec = rho_vec_batch[iv]
        rho_0 = rho_vec[b0 * N2:(b0 + 1) * N2].reshape((N, N), order='F')

        if b_p1 < n_blocks:
            rho_p1 = rho_vec[b_p1 * N2:(b_p1 + 1) * N2].reshape((N, N), order='F')
        else:
            rho_p1 = cp.zeros((N, N), dtype=cp.complex128)

        if b_m1 >= 0:
            rho_m1 = rho_vec[b_m1 * N2:(b_m1 + 1) * N2].reshape((N, N), order='F')
        else:
            rho_m1 = cp.zeros((N, N), dtype=cp.complex128)

        F_plus = cp.trace(rho_m1 @ H_plus_d)
        F_minus = cp.trace(rho_p1 @ H_minus_d)

        force_val = _HBAR * k * float(cp.imag(F_plus) - cp.imag(F_minus))
        F_arr[iv] = force_val
        rho0_arr[iv] = to_cpu(rho_0)

    return F_arr, rho0_arr


def force_vs_velocity_cuda_fast(mol_data, beam_pairs, v_arr,
                                B_gradient=0.0, Omega_Gamma=1.0,
                                n_max=3, Gamma_eff_factor=1.0):
    """
    Fully-batched Floquet force computation (no Python loop over velocities).

    Same as ``force_vs_velocity_cuda`` but extracts forces using batched
    matrix operations on the GPU.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
    v_arr : np.ndarray
    B_gradient : float
    Omega_Gamma : float
    n_max : int
    Gamma_eff_factor : float

    Returns
    -------
    F_arr : np.ndarray, shape (n_v,)
        Force at each velocity (N).
    """
    _require_cupy()

    v_arr = np.asarray(v_arr, dtype=np.float64)
    n_v = len(v_arr)
    k = mol_data.k
    N = mol_data.n_states
    N2 = N * N
    n_blocks = 2 * n_max + 1

    # Build Floquet blocks
    L0_cpu, Lp_cpu, Lm_cpu, H_plus_cpu, H_minus_cpu = _build_floquet_blocks(
        mol_data, beam_pairs, Omega_Gamma, Gamma_eff_factor)

    L0_d = cp.asarray(L0_cpu)
    Lp_d = cp.asarray(Lp_cpu)
    Lm_d = cp.asarray(Lm_cpu)
    H_plus_d = cp.asarray(H_plus_cpu)
    H_minus_d = cp.asarray(H_minus_cpu)

    v_eff = v_arr.copy()
    v_eff[np.abs(v_eff) < 1e-12] = 1e-6
    kv_arr_d = cp.asarray(k * v_eff)

    M_batch, rhs_batch = _build_floquet_matrices_batch(
        L0_d, Lp_d, Lm_d, kv_arr_d, n_max, N)

    # Batched solve
    rho_vec_batch = cp.linalg.solve(M_batch, rhs_batch)

    # Extract rho_{-1} and rho_{+1} blocks for all velocities
    b0 = n_max
    b_p1 = n_max + 1
    b_m1 = n_max - 1

    # rho_{-1}: block b_m1, reshape to (n_v, N, N) column-major
    rho_m1_flat = rho_vec_batch[:, b_m1 * N2:(b_m1 + 1) * N2]
    rho_m1_batch = rho_m1_flat.reshape((n_v, N, N), order='F')  # wrong!
    # Actually need to reshape each row individually for column-major.
    # CuPy reshape works element-by-element, so we need a transpose trick.
    # vec(rho) in column-major: vec[j*N + i] = rho[i, j]
    # So reshape(-1, N, N) gives row-major (rho[i,j] at [i,j]).
    # We need column-major: rho[i,j] = vec[j*N+i]
    # Reshape to (n_v, N, N) then transpose last two axes.
    rho_m1_batch = rho_m1_flat.reshape((n_v, N, N))  # row-major: [iv, j, i]
    rho_m1_batch = rho_m1_batch.transpose(0, 2, 1)    # now [iv, i, j] = rho_{-1}

    rho_p1_flat = rho_vec_batch[:, b_p1 * N2:(b_p1 + 1) * N2]
    rho_p1_batch = rho_p1_flat.reshape((n_v, N, N)).transpose(0, 2, 1)

    # Force: F = hbar*k * (Im(Tr(rho_{-1} @ H_+)) - Im(Tr(rho_{+1} @ H_-)))
    # Batched: Tr(A @ B) = sum_{i,j} A[i,j] * B[j,i] = sum(A * B.T)
    H_plus_T = H_plus_d.T
    H_minus_T = H_minus_d.T

    # Tr(rho_m1[iv] @ H_plus) = sum(rho_m1[iv] * H_plus_T)
    tr_m1 = cp.sum(rho_m1_batch * H_plus_T[cp.newaxis, :, :], axis=(1, 2))
    tr_p1 = cp.sum(rho_p1_batch * H_minus_T[cp.newaxis, :, :], axis=(1, 2))

    F_d = _HBAR * k * (cp.imag(tr_m1) - cp.imag(tr_p1))

    return to_cpu(F_d.real)


def scattering_rate_vs_velocity_cuda(mol_data, beam_pairs, v_arr,
                                     B_gradient=0.0, Omega_Gamma=1.0,
                                     n_max=3, Gamma_eff_factor=1.0):
    """
    Compute the scattering rate as a function of velocity on the GPU.

    Parameters
    ----------
    mol_data, beam_pairs, v_arr, B_gradient, Omega_Gamma, n_max,
    Gamma_eff_factor : same as force_vs_velocity_cuda

    Returns
    -------
    R_arr : np.ndarray, shape (n_v,)
        Scattering rate at each velocity (rad/s).
    """
    _require_cupy()

    v_arr = np.asarray(v_arr, dtype=np.float64)
    n_v = len(v_arr)
    k = mol_data.k
    N = mol_data.n_states
    N2 = N * N
    n_g = mol_data.n_ground

    # Build Floquet blocks
    L0_cpu, Lp_cpu, Lm_cpu, _, _ = _build_floquet_blocks(
        mol_data, beam_pairs, Omega_Gamma, Gamma_eff_factor)

    L0_d = cp.asarray(L0_cpu)
    Lp_d = cp.asarray(Lp_cpu)
    Lm_d = cp.asarray(Lm_cpu)

    v_eff = v_arr.copy()
    v_eff[np.abs(v_eff) < 1e-12] = 1e-6
    kv_arr_d = cp.asarray(k * v_eff)

    n_blocks = 2 * n_max + 1
    M_batch, rhs_batch = _build_floquet_matrices_batch(
        L0_d, Lp_d, Lm_d, kv_arr_d, n_max, N)

    rho_vec_batch = cp.linalg.solve(M_batch, rhs_batch)

    # Extract rho_0 diagonal (excited states)
    b0 = n_max
    Gamma_eff = mol_data.Gamma * Gamma_eff_factor

    rho0_flat = rho_vec_batch[:, b0 * N2:(b0 + 1) * N2]
    # rho_0[i,i] in column-major: index i*N + i
    excited_pops = cp.zeros(n_v, dtype=cp.float64)
    for i in range(n_g, N):
        excited_pops += cp.real(rho0_flat[:, i * N + i])

    R_d = Gamma_eff * cp.maximum(excited_pops, 0.0)

    return to_cpu(R_d)
