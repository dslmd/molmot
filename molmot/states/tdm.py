"""
Transition dipole moment (TDM) computation between eigenstates.

Port of the TDM handling in QuantumStates.jl / OpticalBlochEquations.jl.

The key operation is transforming the basis-level TDMs into the
eigenstate basis using the diagonalisation coefficients.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def compute_tdms(ground_states: list,
                 excited_states: list,
                 ground_basis: list,
                 excited_basis: list,
                 tdm_func: Callable,
                 ground_coeffs: Optional[np.ndarray] = None,
                 excited_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the TDM array between ground and excited eigenstates.

    Parameters
    ----------
    ground_states : list of State
        Ground eigenstates (from Hamiltonian.solve()).
    excited_states : list of State
        Excited eigenstates.
    ground_basis : list of BasisState
        The basis states spanning the ground manifold.
    excited_basis : list of BasisState
        The basis states spanning the excited manifold.
        These should be in the SAME coupling scheme as the ground basis
        (e.g., both case (b)) so that tdm_func can be applied.
    tdm_func : callable(s_ground, s_excited, p) -> float
        The basis-level TDM operator.  Takes two basis states and a
        polarisation index p in {-1, 0, +1} and returns the matrix element.
    ground_coeffs : np.ndarray, optional
        Shape (n_ground_eigenstates, n_ground_basis).
        If None, extracted from ground_states[k].coeffs.
    excited_coeffs : np.ndarray, optional
        Shape (n_excited_eigenstates, n_excited_basis).
        If None, extracted from excited_states[k].coeffs.

    Returns
    -------
    d : np.ndarray, shape (n_total, n_total, 3)
        TDM array where d[i, j, q] is the dipole matrix element between
        eigenstate i and eigenstate j for polarisation q = {sigma-, pi, sigma+}
        (indexed as q+1 = {0, 1, 2}).

        The indexing convention follows the Julia code:
        * i in 0..n_ground-1 are ground states
        * j in n_ground..n_total-1 are excited states
        * d[i, j, q] = <ground_i | T^1_q | excited_j>
    """
    n_g = len(ground_states)
    n_e = len(excited_states)
    n_gb = len(ground_basis)
    n_eb = len(excited_basis)
    n_total = n_g + n_e

    # Extract coefficients
    if ground_coeffs is None:
        ground_coeffs = np.array([s.coeffs for s in ground_states])
    if excited_coeffs is None:
        excited_coeffs = np.array([s.coeffs for s in excited_states])

    # Step 1: Compute basis-level TDM matrix d_basis[m, n, q]
    # where m indexes ground basis states and n indexes excited basis states
    d_basis = np.zeros((n_gb, n_eb, 3), dtype=complex)

    for m in range(n_gb):
        for n in range(n_eb):
            for qi, p in enumerate([-1, 0, 1]):
                val = tdm_func(ground_basis[m], excited_basis[n], p)
                if abs(val) > 1e-15:
                    d_basis[m, n, qi] = val

    # Step 2: Transform to eigenstate basis
    # d_eigen[i, j, q] = sum_m,n conj(c_g[i, m]) * c_e[j, n] * d_basis[m, n, q]
    d = np.zeros((n_total, n_total, 3), dtype=complex)

    for qi in range(3):
        # Matrix multiplication: C_g^dagger @ d_basis[:,:,qi] @ C_e^T
        # ground_coeffs is (n_g, n_gb), excited_coeffs is (n_e, n_eb)
        d_eigen_block = ground_coeffs.conj() @ d_basis[:, :, qi] @ excited_coeffs.T
        d[:n_g, n_g:n_total, qi] = d_eigen_block

    # Also fill the conjugate transpose (excited -> ground)
    for qi in range(3):
        # d[j, i, q] = conj(d[i, j, q]) * (-1)^q  (time-reversal)
        # Actually for TDM the convention varies; store both directions
        q_conj = 2 - qi  # maps q=-1->+1, 0->0, +1->-1
        d[n_g:n_total, :n_g, q_conj] = d[:n_g, n_g:n_total, qi].conj().T

    return d


def compute_tdms_direct(basis_all: list,
                        tdm_func: Callable,
                        n_ground: int) -> np.ndarray:
    """
    Compute basis-level TDM matrix directly (no eigenstate transformation).

    Parameters
    ----------
    basis_all : list of BasisState
        All basis states (ground + excited), ordered ground first.
    tdm_func : callable(s1, s2, p) -> float
    n_ground : int
        Number of ground states.

    Returns
    -------
    d : np.ndarray, shape (n_total, n_total, 3)
    """
    n_total = len(basis_all)
    d = np.zeros((n_total, n_total, 3), dtype=complex)

    for i in range(n_total):
        for j in range(n_total):
            for qi, p in enumerate([-1, 0, 1]):
                val = tdm_func(basis_all[i], basis_all[j], p)
                if abs(val) > 1e-15:
                    d[i, j, qi] = val

    return d
