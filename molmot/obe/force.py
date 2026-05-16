"""
Radiation pressure and gradient force computation from wavefunction and density matrix.

Port of force.jl from OpticalBlochEquations.jl by Christian Hallas.

Provides two approaches:
  1. force_from_wavefunction: for use with SSE (stochastic Schrodinger) trajectories.
     Computes F = hbar * sum_q Im( conj(psi_g) * psi_e * d_{ge,q} * E_q ) * k_hat
  2. force_from_density_matrix: for use with OBE (Lindblad) steady-state solutions.
     Computes F = -Tr(rho * dH/dz).
  3. zeeman_gradient_force: conservative force from the spatially varying Zeeman shift.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..constants import hbar


def force_from_wavefunction(psi, d_ge, E_kq, n_ground, n_excited, k_dirs=None):
    """
    Compute radiation pressure force from a wavefunction in the interaction picture.

    This is the "fast" force calculation used in the SSE solver.
    It computes the expectation value of the dipole force operator:

        F_k = -2 * Re( -i * sum_q d_exp_q * (E_kq_plus - E_kq_minus) )

    where d_exp_q = sum_{g} conj(psi_g) * sum_{e} d_ge[g,e,q] * psi_{n_g+e}
    is the dipole expectation value (the d^+ part).

    This matches update_force! and update_d_exp! from the Julia code.

    Parameters
    ----------
    psi : np.ndarray, shape (n_states,), complex
        Wavefunction amplitudes (in the interaction picture / Heisenberg frame).
    d_ge : np.ndarray, shape (n_ground, n_excited, 3), real
        Transition dipole matrix elements d_ge[ig, ie, q] for q in {0,1,2}
        corresponding to sigma-, pi, sigma+ polarizations.
        Note: these are the NEGATIVE of the raw TDM (sign convention for -d.E).
    E_kq : np.ndarray, shape (6, 3), complex
        Electric field amplitudes for each of the 6 beam directions and 3
        polarization components. Rows 0-2 are +x, +y, +z beams; rows 3-5
        are -x, -y, -z beams.
    n_ground : int
        Number of ground states.
    n_excited : int
        Number of excited states.
    k_dirs : np.ndarray, optional, shape (3, 3)
        Unit vectors for the 3 beam axis directions. Default: x, y, z.

    Returns
    -------
    F : np.ndarray, shape (3,)
        Force vector [Fx, Fy, Fz] in natural units (hbar * k * Gamma).
        To convert to SI: F_SI = F * hbar * k * Gamma.
    """
    n_g = n_ground
    n_e = n_excited

    # Compute dipole expectation d_exp[q] = sum_g conj(psi_g) * (d_ge @ psi_e)[g,q]
    # This is the d^+ part of the dipole operator.
    psi_g = psi[:n_g]
    psi_e = psi[n_g:n_g + n_e]

    # psi_q[g, q] = sum_e d_ge[g, e, q] * psi_e[e]
    # d_ge: (n_g, n_e, 3), psi_e: (n_e,)
    psi_q_g = np.einsum('geq,e->gq', d_ge, psi_e)  # (n_g, 3)

    # d_exp[q] = sum_g conj(psi_g[g]) * psi_q_g[g, q]
    d_exp = np.conj(psi_g) @ psi_q_g  # (3,)

    # Force along each spatial direction k:
    # F[k] = -2 * Re( -i * sum_q d_exp[q] * (E_kq[k,q] - E_kq[k+3,q]) )
    # The factor of -i comes from the gradient of exp(ikr): d/dr exp(ikr) = ik exp(ikr)
    F = np.zeros(3)
    for k in range(3):
        F_k = 0.0
        for q in range(3):
            # E_diff = E_kq[k, q] - E_kq[k+3, q]  (forward minus backward beam)
            E_diff_re = np.real(E_kq[k, q]) - np.real(E_kq[k + 3, q])
            E_diff_im = np.imag(E_kq[k, q]) - np.imag(E_kq[k + 3, q])

            # Multiply by -i: (-i)(a + ib) = b - ia
            E_kq_re = E_diff_im
            E_kq_im = -E_diff_re

            # d_exp[q] * E_kq -> take real part
            d_re = np.real(d_exp[q])
            d_im = np.imag(d_exp[q])

            F_k_a_re = d_re * E_kq_re - d_im * E_kq_im
            F_k -= 2.0 * F_k_a_re

        F[k] = F_k

    return F


def force_from_density_matrix(rho, d_ge, E_kq, n_ground, n_excited):
    """
    Compute radiation pressure force from a density matrix.

    Uses the same dipole-force formula as the wavefunction version, but
    with rho_ge = rho[g, n_g+e] in place of conj(psi_g)*psi_e.

    Parameters
    ----------
    rho : np.ndarray, shape (N, N), complex
        Density matrix.
    d_ge : np.ndarray, shape (n_ground, n_excited, 3), real
        Transition dipole matrix (ground x excited x polarization).
    E_kq : np.ndarray, shape (6, 3), complex
        Electric field amplitudes.
    n_ground, n_excited : int

    Returns
    -------
    F : np.ndarray, shape (3,)
        Force in natural units.
    """
    n_g = n_ground
    n_e = n_excited

    # d_exp[q] = sum_{g,e} d_ge[g,e,q] * rho[n_g+e, g]  (the rho_eg element)
    # Note: rho_eg = conj(rho_ge), but we use rho[e_abs, g] directly.
    d_exp = np.zeros(3, dtype=complex)
    for q in range(3):
        for ig in range(n_g):
            for ie in range(n_e):
                ie_abs = ie + n_g
                d_exp[q] += d_ge[ig, ie, q] * rho[ie_abs, ig]

    F = np.zeros(3)
    for k in range(3):
        F_k = 0.0
        for q in range(3):
            E_diff_re = np.real(E_kq[k, q]) - np.real(E_kq[k + 3, q])
            E_diff_im = np.imag(E_kq[k, q]) - np.imag(E_kq[k + 3, q])

            E_kq_re = E_diff_im
            E_kq_im = -E_diff_re

            d_re = np.real(d_exp[q])
            d_im = np.imag(d_exp[q])

            F_k_a_re = d_re * E_kq_re - d_im * E_kq_im
            F_k -= 2.0 * F_k_a_re

        F[k] = F_k

    return F


def zeeman_gradient_force(psi, zeeman_matrices, B_gradient, mu_B_Gamma):
    """
    Compute the conservative Zeeman gradient force.

    For a quadrupole field B = B_gradient * (x, y, -2z) / 2 (anti-Helmholtz),
    the force is:

        F_i = -d/dr_i <psi| H_Zeeman(r) |psi>
            = -mu_B * dB_i/dr_i * <psi| Z_i |psi>

    where Z_i is the Zeeman matrix for the i-th field component.

    Parameters
    ----------
    psi : np.ndarray, shape (n_states,), complex
        Wavefunction.
    zeeman_matrices : dict
        Keys 'x', 'y', 'z', values are (n_states, n_states) complex arrays.
    B_gradient : float
        Magnetic field gradient in T/m.
    mu_B_Gamma : float
        Bohr magneton in units of hbar*Gamma (for natural-unit force output).

    Returns
    -------
    F : np.ndarray, shape (3,)
        Force from the Zeeman gradient.
    """
    # Quadrupole gradients: dBx/dx = B'/2, dBy/dy = B'/2, dBz/dz = -B'
    # (anti-Helmholtz coils)
    grad_B = np.array([B_gradient / 2, B_gradient / 2, -B_gradient])

    F = np.zeros(3)
    for i, axis in enumerate(['x', 'y', 'z']):
        Z = zeeman_matrices[axis]
        expectation = np.real(np.conj(psi) @ Z @ psi)
        F[i] = -mu_B_Gamma * grad_B[i] * expectation

    return F
