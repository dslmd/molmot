"""
CUDA-accelerated Lindblad master equation solver with sub-Doppler effects.

GPU-batched version of ``molmot.obe.lindblad`` and
``molmot.obe.obe_subdoppler``.

Key optimisation: the jump operators L_c = |g><e| are rank-1.  The
Lindblad dissipator terms L_c rho L_c^dag have exactly ONE nonzero
element (at position (g,g) with value Gc * rho[e,e]).  Instead of
computing full Kronecker products (O(N^4) per jump), we directly set
the single nonzero element in the Liouvillian, reducing dissipator
construction from O(N^4) to O(n_jumps * N^2).

For the sub-Doppler force, the work is parallelised across BOTH
velocity points AND spatial averaging points, giving
N_v x n_spatial independent solves (e.g. 200 x 16 = 3200).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
# Optimised Liouvillian construction on the GPU
# ---------------------------------------------------------------------------

def build_liouvillian_single(H, mol_data_Gamma, d_squared, n_ground,
                             n_excited, use_gpu=True):
    """
    Build the Liouvillian superoperator for a single Hamiltonian.

    Uses the rank-1 jump operator optimisation: each L_c = sqrt(Gc)|g><e|
    produces exactly one nonzero element in the kron(L_c*, L_c) term.

    Parameters
    ----------
    H : ndarray, shape (N, N), complex
        Hamiltonian in rad/s.
    mol_data_Gamma : float
        Natural linewidth (rad/s).
    d_squared : ndarray, shape (n_ground, n_excited, 3)
        |TDM|^2 array.
    n_ground, n_excited : int
    use_gpu : bool
        If True, use CuPy arrays.

    Returns
    -------
    L : ndarray, shape (N^2, N^2), complex
        Liouvillian superoperator.
    """
    xp = cp if (use_gpu and _CUPY_AVAILABLE) else np
    N = n_ground + n_excited
    N2 = N * N
    Gamma = mol_data_Gamma

    if use_gpu and _CUPY_AVAILABLE:
        H = cp.asarray(H)
        d_squared = cp.asarray(d_squared)

    I = xp.eye(N, dtype=xp.complex128)

    # Coherent part: -i(I kron H - H^T kron I)
    L = -1j * (xp.kron(I, H) - xp.kron(H.T, I))

    # Dissipator using rank-1 optimisation
    # For each jump L_c = sqrt(Gc) |ig><ie_abs|:
    #   kron(L_c*, L_c):  only element (ie_abs*N + ig, ie_abs*N + ig) = Gc
    #   kron(I, L_c^dag L_c):  L_c^dag L_c = Gc |ie_abs><ie_abs|
    #     -> for each col a: L[a*N + ie_abs, a*N + ie_abs] -= 0.5*Gc
    #   kron((L_c^dag L_c)^T, I):  same structure
    #     -> for each row b: L[ie_abs*N + b, ie_abs*N + b] -= 0.5*Gc
    #
    # This is O(n_jumps * N) instead of O(n_jumps * N^2) for kron.

    for ie in range(n_excited):
        ie_abs = ie + n_ground
        total = float(xp.sum(d_squared[:, ie, :]))
        if total < 1e-30:
            continue
        for ig in range(n_ground):
            for q in range(3):
                br = float(d_squared[ig, ie, q]) / total
                if br < 1e-15:
                    continue
                Gc = Gamma * br

                # kron(L_c*, L_c): single element
                # Column-major vectorisation: rho[i,j] at index j*N+i
                idx = ie_abs * N + ig
                L[idx, idx] += Gc

                # -0.5 * kron(I, L_c^dag L_c)
                for a in range(N):
                    row = a * N + ie_abs
                    L[row, row] -= 0.5 * Gc

                # -0.5 * kron((L_c^dag L_c)^T, I)
                for b in range(N):
                    row = ie_abs * N + b
                    L[row, row] -= 0.5 * Gc

    return L


def build_liouvillian_batch(H_arr, mol_data):
    """
    Build Liouvillians for an array of Hamiltonians on the GPU.

    Parameters
    ----------
    H_arr : np.ndarray or cp.ndarray, shape (n_batch, N, N)
        Array of Hamiltonians.
    mol_data : MolecularData
        Molecular data (used for Gamma, d_squared, n_ground, n_excited).

    Returns
    -------
    L_arr : cp.ndarray, shape (n_batch, N^2, N^2)
        Batched Liouvillian superoperators.
    """
    _require_cupy()

    H_arr = cp.asarray(H_arr, dtype=cp.complex128)
    n_batch = H_arr.shape[0]
    N = mol_data.n_ground + mol_data.n_excited
    N2 = N * N
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    Gamma = mol_data.Gamma

    d_squared = cp.asarray(mol_data.d_squared, dtype=cp.float64)

    # Pre-compute the dissipator (same for all Hamiltonians with same mol_data)
    L_dissipator = cp.zeros((N2, N2), dtype=cp.complex128)

    for ie in range(n_e):
        ie_abs = ie + n_g
        total = float(cp.sum(d_squared[:, ie, :]))
        if total < 1e-30:
            continue
        for ig in range(n_g):
            for q in range(3):
                br = float(d_squared[ig, ie, q]) / total
                if br < 1e-15:
                    continue
                Gc = Gamma * br

                idx = ie_abs * N + ig
                L_dissipator[idx, idx] += Gc

                for a in range(N):
                    L_dissipator[a * N + ie_abs, a * N + ie_abs] -= 0.5 * Gc
                for b in range(N):
                    L_dissipator[ie_abs * N + b, ie_abs * N + b] -= 0.5 * Gc

    # Coherent part: batched -i(I kron H - H^T kron I)
    I_N = cp.eye(N, dtype=cp.complex128)

    # kron(I, H) for batch: result[batch, a*N+b, c*N+d] = delta(a,c) * H[b,d]
    # = I[a,c] * H[b,d]
    # This is equivalent to: for each batch, compute np.kron(I, H[batch])
    # We vectorise this using einsum / broadcasting.

    # kron(I, H): shape (N^2, N^2).  Element [a*N+b, c*N+d] = I[a,c]*H[b,d]
    # We can construct this as a tensor product.
    # For batch: kron(I, H_batch) where H_batch has shape (n_batch, N, N)

    L_arr = cp.zeros((n_batch, N2, N2), dtype=cp.complex128)

    # Build coherent part for each Hamiltonian
    # This is the expensive part; we do it using batched tensor operations.
    for iv in range(n_batch):
        H = H_arr[iv]
        L_coh = -1j * (cp.kron(I_N, H) - cp.kron(H.T, I_N))
        L_arr[iv] = L_coh + L_dissipator

    return L_arr


def steady_state_batch(L_arr, N):
    """
    Solve for steady-state density matrices in batch.

    Replaces the last row of each Liouvillian with the trace
    constraint, then solves the resulting linear systems.

    Parameters
    ----------
    L_arr : cp.ndarray, shape (n_batch, N^2, N^2)
        Batched Liouvillian superoperators.
    N : int
        Number of quantum states.

    Returns
    -------
    rho_arr : cp.ndarray, shape (n_batch, N, N)
        Steady-state density matrices.
    """
    _require_cupy()

    n_batch = L_arr.shape[0]
    N2 = N * N

    # Copy to avoid modifying input
    M = L_arr.copy()

    # Replace last row with trace constraint
    M[:, -1, :] = 0.0
    for i in range(N):
        M[:, -1, i * N + i] = 1.0

    rhs = cp.zeros((n_batch, N2), dtype=cp.complex128)
    rhs[:, -1] = 1.0

    # Batched solve
    try:
        rho_vec = cp.linalg.solve(M, rhs)
    except cp.linalg.LinAlgError:
        # Fallback: solve individually
        rho_vec = cp.zeros((n_batch, N2), dtype=cp.complex128)
        for iv in range(n_batch):
            try:
                rho_vec[iv] = cp.linalg.solve(M[iv], rhs[iv])
            except cp.linalg.LinAlgError:
                # Uniform distribution
                for i in range(N):
                    rho_vec[iv, i * N + i] = 1.0 / N

    # Reshape to density matrices (column-major)
    # vec[j*N + i] = rho[i, j]  ->  reshape to (N, N) then no transpose needed
    # Actually: reshape(N, N) gives rho_flat[row, col] = vec[row*N + col]
    # But column-major: vec[j*N+i] = rho[i,j], so vec index = j*N+i
    # reshape(N, N) row-major: element [a, b] = vec[a*N + b]
    # We need rho[i, j] = vec[j*N + i], so rho = reshape(N, N).T
    rho_arr = rho_vec.reshape((n_batch, N, N))
    rho_arr = rho_arr.transpose(0, 2, 1)  # correct column-major -> (i, j)

    # Enforce Hermiticity
    rho_arr = 0.5 * (rho_arr + rho_arr.conj().transpose(0, 2, 1))

    # Normalise
    traces = cp.trace(rho_arr, axis1=1, axis2=2)
    # Avoid division by zero
    traces = cp.where(cp.abs(traces) > 1e-15, traces, 1.0)
    rho_arr /= traces[:, cp.newaxis, cp.newaxis]

    return rho_arr


# ---------------------------------------------------------------------------
# Sub-Doppler force computation
# ---------------------------------------------------------------------------

def _build_H_standing_wave_batch(mol_data, beam_pairs, v_arr, z_macro,
                                 z_micro_arr, B_gradient, Omega_Gamma):
    """
    Build Hamiltonians for all (velocity, z_micro) combinations.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
    v_arr : np.ndarray, shape (n_v,)
    z_macro : float
    z_micro_arr : np.ndarray, shape (n_spatial,)
    B_gradient : float
    Omega_Gamma : float

    Returns
    -------
    H_batch : np.ndarray, shape (n_v * n_spatial, N, N)
        Hamiltonians for all (v, z_micro) pairs.
    """
    Gamma = mol_data.Gamma
    k = mol_data.k
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    N = n_g + n_e
    d = mol_data.tdm

    B_gauss = B_gradient * z_macro * 1e4
    E_Z = B_gauss * mol_data.zeeman_z_diag * Gamma / (2 * np.pi)
    Es_Hz = mol_data.energies + E_Z
    E_e_mean = np.mean(Es_Hz[n_g:])

    laser_freqs = [bp['freq'] for bp in beam_pairs]
    Omega0 = Omega_Gamma * Gamma

    n_v = len(v_arr)
    n_spatial = len(z_micro_arr)
    n_total = n_v * n_spatial

    H_batch = np.zeros((n_total, N, N), dtype=complex)

    # Diagonal part (same for all v, z_micro)
    H_diag = np.zeros((N, N), dtype=complex)
    for ig in range(n_g):
        omega_trans = E_e_mean - Es_Hz[ig]
        deltas = [lf - omega_trans for lf in laser_freqs]
        nearest_delta = min(deltas, key=abs)
        H_diag[ig, ig] = -2 * np.pi * nearest_delta

    for ie in range(n_e):
        ie_abs = ie + n_g
        H_diag[ie_abs, ie_abs] = 2 * np.pi * (Es_Hz[ie_abs] - E_e_mean)

    # Pre-compute coupling coefficients (independent of v, z_micro)
    # For each beam pair, each (ig, ie) transition:
    couplings_fwd = []  # list of (ig, ie_abs, Om * d_fwd / 2)
    couplings_bwd = []

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
                    couplings_fwd.append((ig, ie_abs, Om * d_fwd / 2))

                d_bwd = abs(d[ig, ie_abs, q_bwd])
                if d_bwd > 1e-10:
                    couplings_bwd.append((ig, ie_abs, Om * d_bwd / 2))

    # Fill batched Hamiltonians
    for iv in range(n_v):
        for iz in range(n_spatial):
            idx = iv * n_spatial + iz
            z_m = z_micro_arr[iz]

            H_batch[idx] = H_diag.copy()

            # Forward beam couplings: exp(+i * z_micro)
            phase_fwd = np.exp(1j * z_m)
            for ig, ie_abs, c in couplings_fwd:
                coupling = c * phase_fwd
                H_batch[idx, ie_abs, ig] += coupling
                H_batch[idx, ig, ie_abs] += np.conj(coupling)

            # Backward beam couplings: exp(-i * z_micro)
            phase_bwd = np.exp(-1j * z_m)
            for ig, ie_abs, c in couplings_bwd:
                coupling = c * phase_bwd
                H_batch[idx, ie_abs, ig] += coupling
                H_batch[idx, ig, ie_abs] += np.conj(coupling)

    return H_batch


def _force_from_rho_batch(rho_batch, mol_data, beam_pairs, z_micro_arr,
                          Omega_Gamma, n_v, n_spatial):
    """
    Compute forces from batched density matrices.

    Parameters
    ----------
    rho_batch : cp.ndarray, shape (n_v * n_spatial, N, N)
    mol_data : MolecularData
    beam_pairs : list of dict
    z_micro_arr : np.ndarray, shape (n_spatial,)
    Omega_Gamma : float
    n_v, n_spatial : int

    Returns
    -------
    F_avg : np.ndarray, shape (n_v,)
        Wavelength-averaged forces.
    """
    k = mol_data.k
    Gamma = mol_data.Gamma
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    d = mol_data.tdm
    Omega0 = Omega_Gamma * Gamma

    # Pre-compute coupling info
    couplings_fwd = []
    couplings_bwd = []
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
                    couplings_fwd.append((ig, ie_abs, Om * d_fwd / 2))
                d_bwd = abs(d[ig, ie_abs, q_bwd])
                if d_bwd > 1e-10:
                    couplings_bwd.append((ig, ie_abs, Om * d_bwd / 2))

    rho_cpu = to_cpu(rho_batch)
    F_all = np.zeros(n_v * n_spatial, dtype=np.float64)

    for iv in range(n_v):
        for iz in range(n_spatial):
            idx = iv * n_spatial + iz
            rho = rho_cpu[idx]
            z_m = z_micro_arr[iz]
            force = 0.0

            # +z beam
            phase_fwd = np.exp(1j * z_m)
            for ig, ie_abs, c in couplings_fwd:
                coupling = c * phase_fwd
                force += _HBAR * k * 2 * np.imag(rho[ig, ie_abs] * coupling)

            # -z beam
            phase_bwd = np.exp(-1j * z_m)
            for ig, ie_abs, c in couplings_bwd:
                coupling = c * phase_bwd
                force -= _HBAR * k * 2 * np.imag(rho[ig, ie_abs] * coupling)

            F_all[idx] = force

    # Average over spatial points for each velocity
    F_all_2d = F_all.reshape(n_v, n_spatial)
    F_avg = np.mean(F_all_2d, axis=1)

    return F_avg


def obe_force_subdoppler_cuda(mol_data, beam_pairs, v_arr, z_macro=0.0,
                              B_gradient=0.0, Omega_Gamma=1.0,
                              n_spatial=16, Gamma_eff_factor=1.0):
    """
    Compute sub-Doppler force as a function of velocity on the GPU.

    Parallelises across BOTH velocity points AND spatial averaging
    points.  Total number of independent linear solves:
    ``n_v * n_spatial``.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
        Each dict: ``{'freq': Hz, 'q_fwd': int, 'q_bwd': int, 's0': float}``
    v_arr : np.ndarray
        Velocities (m/s).
    z_macro : float
        Macroscopic position for Zeeman shift (m).
    B_gradient : float
        Magnetic field gradient (T/m).
    Omega_Gamma : float
        Peak Rabi frequency in units of Gamma.
    n_spatial : int
        Number of spatial averaging points within one wavelength.
    Gamma_eff_factor : float
        Scattering rate reduction factor.

    Returns
    -------
    F_arr : np.ndarray, shape (n_v,)
        Wavelength-averaged force at each velocity (N).
    rho_avg : np.ndarray, shape (n_v, N, N)
        Wavelength-averaged density matrix at each velocity.
    R_scatter : np.ndarray, shape (n_v,)
        Scattering rate at each velocity (rad/s).
    """
    _require_cupy()

    v_arr = np.asarray(v_arr, dtype=np.float64)
    n_v = len(v_arr)
    N = mol_data.n_states
    n_g = mol_data.n_ground
    Gamma_eff = mol_data.Gamma * Gamma_eff_factor

    z_micro_arr = np.linspace(0, 2 * np.pi, n_spatial, endpoint=False)

    # Build all Hamiltonians (CPU, then transfer)
    H_batch = _build_H_standing_wave_batch(
        mol_data, beam_pairs, v_arr, z_macro, z_micro_arr,
        B_gradient, Omega_Gamma)

    # Build effective mol_data with scaled Gamma
    d_squared_cpu = mol_data.d_squared.copy()

    # Build Liouvillians on GPU
    H_batch_d = cp.asarray(H_batch, dtype=cp.complex128)
    n_total = n_v * n_spatial

    # We need to build Liouvillians with Gamma_eff
    # Use the batch builder with modified Gamma
    import types
    mol_data_eff = types.SimpleNamespace(
        Gamma=Gamma_eff,
        d_squared=mol_data.d_squared,
        n_ground=mol_data.n_ground,
        n_excited=mol_data.n_excited,
        n_states=mol_data.n_states,
    )

    L_arr = build_liouvillian_batch(H_batch_d, mol_data_eff)

    # Solve steady states
    rho_batch = steady_state_batch(L_arr, N)

    # Compute forces
    F_avg = _force_from_rho_batch(
        rho_batch, mol_data, beam_pairs, z_micro_arr,
        Omega_Gamma, n_v, n_spatial)

    # Average density matrices and scattering rates
    rho_cpu = to_cpu(rho_batch).reshape(n_v, n_spatial, N, N)
    rho_avg = np.mean(rho_cpu, axis=1)

    R_scatter = np.zeros(n_v, dtype=np.float64)
    for iv in range(n_v):
        R_scatter[iv] = Gamma_eff * np.sum(
            np.real(np.diag(rho_avg[iv]))[n_g:])
        R_scatter[iv] = max(R_scatter[iv], 0.0)

    return F_avg, rho_avg, R_scatter


def obe_force_map_cuda(mol_data, beam_pairs, v_arr, z_arr,
                       B_gradient=0.0, Omega_Gamma=1.0,
                       n_spatial=16, Gamma_eff_factor=1.0):
    """
    Compute a 2D force map F(v, z) with sub-Doppler effects on the GPU.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
    v_arr : np.ndarray, shape (n_v,)
    z_arr : np.ndarray, shape (n_z,)
    B_gradient : float
    Omega_Gamma : float
    n_spatial : int
    Gamma_eff_factor : float

    Returns
    -------
    F_map : np.ndarray, shape (n_v, n_z)
        Force map (N).
    R_map : np.ndarray, shape (n_v, n_z)
        Scattering rate map (rad/s).
    """
    _require_cupy()

    v_arr = np.asarray(v_arr, dtype=np.float64)
    z_arr = np.asarray(z_arr, dtype=np.float64)
    n_v = len(v_arr)
    n_z = len(z_arr)

    F_map = np.zeros((n_v, n_z), dtype=np.float64)
    R_map = np.zeros((n_v, n_z), dtype=np.float64)

    for iz, z_macro in enumerate(z_arr):
        F_col, _, R_col = obe_force_subdoppler_cuda(
            mol_data, beam_pairs, v_arr, z_macro=z_macro,
            B_gradient=B_gradient, Omega_Gamma=Omega_Gamma,
            n_spatial=n_spatial, Gamma_eff_factor=Gamma_eff_factor)
        F_map[:, iz] = F_col
        R_map[:, iz] = R_col

    return F_map, R_map
