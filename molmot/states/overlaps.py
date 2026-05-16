"""
Overlap integrals between Hund's case (a) and case (b) basis states,
and basis conversion utilities.

Port of StateOverlaps.jl from QuantumStates.jl.

References
----------
* Hirota, eq. 2.3.3 (case a ↔ case b transformation)
* Brown & Carrington, Section 5.4.2
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List

import numpy as np

from .basis import BasisState
from .case_b import HundsCaseB_LinearMolecule
from .case_a import HundsCaseA_LinearMolecule
from ..wigner import wigner3j


def _f(x):
    """Convert to Fraction."""
    return Fraction(x).limit_denominator(1000)


def overlap_caseb_casea(state_b: HundsCaseB_LinearMolecule,
                        state_a: HundsCaseA_LinearMolecule) -> complex:
    """
    Overlap integral <case_b | case_a> using Hirota eq. 2.3.3.

    .. math::

        \\langle N, K, S, J | J, P, S, \\Sigma \\rangle
        = (-1)^{-J+P+2S} \\sqrt{2N+1}
          \\begin{pmatrix} J & N & S \\\\ P & -K & -\\Sigma \\end{pmatrix}

    times Kronecker deltas in J, F, M, ell, and Lambda.

    Parameters
    ----------
    state_b : HundsCaseB_LinearMolecule
    state_a : HundsCaseA_LinearMolecule

    Returns
    -------
    complex
        The overlap (real for most cases).
    """
    J_b, J_a = _f(state_b.J), _f(state_a.J)
    F_b, F_a = _f(state_b.F), _f(state_a.F)
    M_b, M_a = _f(state_b.M), _f(state_a.M)
    ell_b, ell_a = _f(state_b.ell), _f(state_a.ell)
    Lambda_b, Lambda_a = _f(state_b.Lambda), _f(state_a.Lambda)

    # Kronecker deltas
    if J_b != J_a or F_b != F_a or M_b != M_a:
        return 0.0 + 0.0j
    if ell_b != ell_a or Lambda_b != Lambda_a:
        return 0.0 + 0.0j

    N = _f(state_b.N)
    K = _f(state_b.K)
    S = _f(state_b.S)
    J = J_b
    P = _f(state_a.P)
    Sigma = _f(state_a.Sigma)

    if _f(state_b.S) != _f(state_a.S):
        return 0.0 + 0.0j
    if _f(state_b.I) != _f(state_a.I):
        return 0.0 + 0.0j

    # Hirota eq. 2.3.3
    phase_exp = -J + P + 2 * S
    if not (phase_exp.denominator == 1):
        # Non-integer phase exponent — states are incompatible
        return 0.0 + 0.0j

    phase = (-1) ** int(phase_exp)
    prefactor = math.sqrt(float(2 * N + 1))
    w3j = wigner3j(J, N, S, P, -K, -Sigma)

    return complex(phase * prefactor * w3j)


def convert_basis(eigenstates: list,
                  eigencoeffs: np.ndarray,
                  old_basis: list,
                  new_basis: list,
                  overlap_func) -> np.ndarray:
    """
    Convert eigenstate coefficients from one basis to another using
    overlap integrals.

    Given eigenstates expressed as linear combinations of ``old_basis``
    states, re-express them in terms of ``new_basis`` states.

    Parameters
    ----------
    eigenstates : list
        Not used directly — included for interface compatibility.
    eigencoeffs : np.ndarray, shape (n_eigenstates, n_old_basis)
        Coefficients of each eigenstate in the old basis.
    old_basis : list of BasisState
        The old (original) basis states.
    new_basis : list of BasisState
        The new (target) basis states.
    overlap_func : callable(new_state, old_state) -> complex
        Overlap integral <new_basis_state | old_basis_state>.

    Returns
    -------
    new_coeffs : np.ndarray, shape (n_eigenstates, n_new_basis)
        Coefficients of each eigenstate in the new basis.
    """
    n_eig = eigencoeffs.shape[0]
    n_old = len(old_basis)
    n_new = len(new_basis)

    # Build the overlap matrix S[i_new, j_old] = <new_i | old_j>
    S = np.zeros((n_new, n_old), dtype=complex)
    for i, new_state in enumerate(new_basis):
        for j, old_state in enumerate(old_basis):
            S[i, j] = overlap_func(new_state, old_state)

    # New coefficients: c_new = c_old @ S^T
    # If eigencoeffs[k, j] = coefficient of eigenstate k in old basis state j,
    # then new_coeffs[k, i] = sum_j eigencoeffs[k, j] * S[i, j]^*
    new_coeffs = eigencoeffs @ S.conj().T

    return new_coeffs
