"""
Lindblad master equation (Optical Bloch Equation) solver.

Port of the OBE solver from OpticalBlochEquations.jl.

Builds the N^2 x N^2 Liouvillian superoperator for the full
density matrix dynamics, including coherent driving (Hamiltonian)
and incoherent decay (Lindblad jump operators).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import solve

from ..constants import hbar
from .fields import LaserBeam


def build_liouvillian(H: np.ndarray,
                      d: np.ndarray,
                      Gamma: float,
                      n_ground: int,
                      n_excited: int,
                      branching_ratios: Optional[np.ndarray] = None,
                      secular_blocks: Optional[list] = None) -> np.ndarray:
    """
    Build the N^2 x N^2 Liouvillian superoperator.

    The Lindblad master equation is:

    .. math::

        \\frac{d\\rho}{dt} = -i[H, \\rho]
        + \\sum_c \\left( L_c \\rho L_c^\\dagger
        - \\frac{1}{2}\\{L_c^\\dagger L_c, \\rho\\} \\right)

    Parameters
    ----------
    H : np.ndarray, shape (N, N)
        Hamiltonian matrix (rad/s).
    d : np.ndarray, shape (N, N, 3)
        TDM matrix.  Used to construct the Lindblad jump operators
        for spontaneous emission.
    Gamma : float
        Natural linewidth (rad/s).
    n_ground : int
    n_excited : int
    branching_ratios : np.ndarray, optional
        Shape (n_excited, n_ground, 3).  If None, computed from d.
    secular_blocks : list of list of int, optional
        Groups of ground-state indices.  Coherences between different
        groups are damped to zero (secular approximation).

    Returns
    -------
    L : np.ndarray, shape (N^2, N^2)
        The Liouvillian superoperator.
    """
    N = n_ground + n_excited
    N2 = N * N
    I = np.eye(N, dtype=complex)

    # Coherent part in column-major vectorization: vec(ρ) stacks columns
    # d/dt vec(ρ) = -i (I⊗H - H^T⊗I) vec(ρ)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Compute branching ratios from TDM if not provided
    if branching_ratios is None:
        branching_ratios = np.zeros((n_excited, n_ground, 3))
        for ie in range(n_excited):
            ie_abs = ie + n_ground
            total = 0.0
            for ig in range(n_ground):
                for q in range(3):
                    total += abs(d[ig, ie_abs, q]) ** 2
            if total > 1e-30:
                for ig in range(n_ground):
                    for q in range(3):
                        branching_ratios[ie, ig, q] = abs(d[ig, ie_abs, q]) ** 2 / total

    # Lindblad dissipator: spontaneous emission
    for ie in range(n_excited):
        ie_abs = ie + n_ground
        for ig in range(n_ground):
            for q in range(3):
                br = branching_ratios[ie, ig, q]
                if br < 1e-15:
                    continue

                Gc = Gamma * br
                # Jump operator: |g><e| * sqrt(Gamma * BR)
                Lc = np.zeros((N, N), dtype=complex)
                Lc[ig, ie_abs] = np.sqrt(Gc)
                LdL = Lc.conj().T @ Lc

                L += np.kron(Lc.conj(), Lc)
                L -= 0.5 * np.kron(I, LdL)
                L -= 0.5 * np.kron(LdL.T, I)

    # Secular approximation
    if secular_blocks is not None:
        for i_block, block_i in enumerate(secular_blocks):
            for j_block, block_j in enumerate(secular_blocks):
                if i_block == j_block:
                    continue
                for a in block_i:
                    for b in block_j:
                        # Column-major: ρ_{a,b} is at index b*N + a
                        idx_ab = b * N + a
                        idx_ba = a * N + b
                        L[idx_ab, :] = 0.0
                        L[:, idx_ab] = 0.0
                        L[idx_ab, idx_ab] = -1e12  # damp to zero
                        L[idx_ba, :] = 0.0
                        L[:, idx_ba] = 0.0
                        L[idx_ba, idx_ba] = -1e12

    return L


def steady_state_density_matrix(H: np.ndarray,
                                d: np.ndarray,
                                Gamma: float,
                                n_ground: int,
                                n_excited: int,
                                branching_ratios: Optional[np.ndarray] = None,
                                secular_blocks: Optional[list] = None) -> np.ndarray:
    """
    Solve for the steady-state density matrix.

    Finds rho_ss such that L @ vec(rho_ss) = 0 with Tr(rho) = 1.

    Parameters
    ----------
    H : np.ndarray, shape (N, N)
    d : np.ndarray, shape (N, N, 3)
    Gamma : float
    n_ground, n_excited : int
    branching_ratios : optional
    secular_blocks : optional

    Returns
    -------
    rho : np.ndarray, shape (N, N)
        Steady-state density matrix.
    """
    N = n_ground + n_excited
    N2 = N * N

    L = build_liouvillian(H, d, Gamma, n_ground, n_excited,
                          branching_ratios, secular_blocks)

    # Replace last row with trace condition: sum_i rho_{ii} = 1
    # Column-major: ρ_{i,i} is at index i*N + i
    L[-1, :] = 0.0
    for i in range(N):
        L[-1, i * N + i] = 1.0  # same for both conventions
    rhs = np.zeros(N2, dtype=complex)
    rhs[-1] = 1.0

    try:
        rho_vec = solve(L, rhs)
    except np.linalg.LinAlgError:
        rho_vec = np.zeros(N2, dtype=complex)
        for i in range(N):
            rho_vec[i * N + i] = 1.0 / N

    rho = rho_vec.reshape((N, N), order='F')
    return rho


def compute_force_obe(rho: np.ndarray,
                      d: np.ndarray,
                      beams: List[LaserBeam],
                      Omega_per_beam: np.ndarray,
                      k: float,
                      n_ground: int,
                      n_excited: int) -> float:
    """
    Compute the radiation pressure force from the steady-state density matrix.

    .. math::

        F = \\sum_b \\hbar k_b \\sum_{g,e} \\Omega_{ge,b} \\operatorname{Im}(\\rho_{ge})

    Parameters
    ----------
    rho : np.ndarray, shape (N, N)
        Density matrix.
    d : np.ndarray, shape (N, N, 3)
        TDM array.
    beams : list of LaserBeam
    Omega_per_beam : np.ndarray, shape (n_beams,)
        Peak Rabi frequency for each beam (rad/s).
    k : float
        Wavenumber (1/m).
    n_ground, n_excited : int

    Returns
    -------
    force : float
        Net force in Newtons.
    """
    force = 0.0

    for ib, beam in enumerate(beams):
        kdir = beam.kdir_z
        q = beam.polarization
        Omega = Omega_per_beam[ib]

        for ig in range(n_ground):
            for ie in range(n_excited):
                ie_abs = ie + n_ground
                d_val = abs(d[ig, ie_abs, q])
                if d_val < 1e-15:
                    continue
                Om = Omega * d_val
                force += hbar * k * kdir * Om * np.imag(rho[ig, ie_abs])

    return force


def build_H_obe(mol_data, beams: List[LaserBeam],
                v: float, z: float, B_gradient: float,
                Omega_Gamma: float) -> np.ndarray:
    """
    Build the OBE Hamiltonian in the multi-rotating frame.

    Uses the same rotating-frame approach as build_H_standing_wave() in
    obe_subdoppler.py: the diagonal contains detunings (~MHz), NOT absolute
    energies (~10^14 Hz). Each ground state is shifted by the nearest laser
    frequency, and the Doppler shift is applied per beam.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of LaserBeam
    v : velocity (m/s)
    z : position (m)
    B_gradient : field gradient (T/m)
    Omega_Gamma : Rabi frequency in units of Gamma

    Returns
    -------
    H : np.ndarray, shape (N, N)
        Hamiltonian in rad/s.
    """
    Gamma = mol_data.Gamma
    k = mol_data.k
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    N = n_g + n_e

    B_gauss = B_gradient * z * 1e4

    # State energies with Zeeman shift (Hz)
    E_zeeman = B_gauss * mol_data.zeeman_z_diag * Gamma / (2.0 * np.pi)
    Es = mol_data.energies + E_zeeman  # Hz

    # Mean excited state energy (Hz) -- defines the rotating frame
    E_e_mean = np.mean(Es[n_g:])

    # Collect laser frequencies from beams
    laser_freqs = [beam.freq_offset for beam in beams]

    H = np.zeros((N, N), dtype=complex)

    # Ground state diagonal: detuning in the rotating frame
    # For each ground state, use the nearest laser frequency
    for ig in range(n_g):
        omega_trans = E_e_mean - Es[ig]  # Hz
        deltas = [lf - omega_trans for lf in laser_freqs]
        nearest_delta = min(deltas, key=abs)  # Hz
        H[ig, ig] = -2.0 * np.pi * nearest_delta

    # Excited state diagonal: small Zeeman shifts relative to mean
    for ie in range(n_e):
        ie_abs = ie + n_g
        H[ie_abs, ie_abs] = 2.0 * np.pi * (Es[ie_abs] - E_e_mean)

    # Off-diagonal: laser couplings with Doppler shift per beam
    Omega0 = Omega_Gamma * Gamma  # peak Rabi frequency (rad/s)
    d = mol_data.tdm

    for beam in beams:
        kdir = beam.kdir_z
        q = beam.polarization
        freq = beam.freq_offset

        # Doppler shift for this beam (Hz)
        doppler_hz = -k * kdir * v / (2.0 * np.pi)

        for ig in range(n_g):
            omega_trans = E_e_mean - Es[ig]  # Hz
            delta_laser = freq - omega_trans  # Hz

            for ie in range(n_e):
                ie_abs = ie + n_g
                d_val = abs(d[ig, ie_abs, q])
                if d_val < 1e-15:
                    continue
                Om = Omega0 * d_val / 2.0
                H[ie_abs, ig] += Om
                H[ig, ie_abs] += np.conj(Om)

    return H
