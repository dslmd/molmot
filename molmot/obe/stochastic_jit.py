"""
Numba JIT-compiled inner loop for the SSE (stochastic Schrodinger equation) solver.

The _rhs() method in stochastic.py is the most expensive function --
called 4x per RK4 step, millions of times per trajectory. This module
extracts the pure-numpy inner kernel and compiles it with @njit.

All arguments are plain numpy arrays (no Python objects), which is the
key requirement for Numba JIT compilation.

Usage:
    from molmot.obe.stochastic_jit import sse_rhs_jit
    du = sse_rhs_jit(u, t, ...)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _compute_fields_6beam_jit(r, t, omega_lasers, sats, eps, beam_radius_k):
    """
    JIT-compiled 6-beam field computation.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Position in natural units.
    t : float
        Time in natural units.
    omega_lasers : ndarray, shape (n_freqs,)
    sats : ndarray, shape (n_freqs,)
    eps : ndarray, shape (6, n_freqs, 3), complex
        Pre-computed rotated polarizations.
    beam_radius_k : float
        Beam radius in natural units.

    Returns
    -------
    E_kq : ndarray, shape (6, 3), complex
    """
    n_freqs = omega_lasers.shape[0]
    denom = beam_radius_k * beam_radius_k / 2.0

    # Gaussian beam profile (perpendicular distances squared)
    rp_sq_0 = r[1] * r[1] + r[2] * r[2]  # beam along x
    rp_sq_1 = r[0] * r[0] + r[2] * r[2]  # beam along y
    rp_sq_2 = r[0] * r[0] + r[1] * r[1]  # beam along z

    gauss_0 = np.sqrt(np.exp(-rp_sq_0 / denom))
    gauss_1 = np.sqrt(np.exp(-rp_sq_1 / denom))
    gauss_2 = np.sqrt(np.exp(-rp_sq_2 / denom))

    kr0 = r[0]
    kr1 = r[1]
    kr2 = r[2]

    E_kq = np.zeros((6, 3), dtype=np.complex128)

    for f in range(n_freqs):
        G = np.sqrt(sats[f]) / (2.0 * np.sqrt(2.0))
        wt = omega_lasers[f] * t

        # x-axis beams
        phase_fwd_x = -kr0 + wt
        phase_bwd_x = kr0 + wt
        a_fwd_x = G * gauss_0 * np.exp(1j * phase_fwd_x)
        a_bwd_x = G * gauss_0 * np.exp(1j * phase_bwd_x)

        # y-axis beams
        phase_fwd_y = -kr1 + wt
        phase_bwd_y = kr1 + wt
        a_fwd_y = G * gauss_1 * np.exp(1j * phase_fwd_y)
        a_bwd_y = G * gauss_1 * np.exp(1j * phase_bwd_y)

        # z-axis beams
        phase_fwd_z = -kr2 + wt
        phase_bwd_z = kr2 + wt
        a_fwd_z = G * gauss_2 * np.exp(1j * phase_fwd_z)
        a_bwd_z = G * gauss_2 * np.exp(1j * phase_bwd_z)

        for q in range(3):
            e0 = np.conj(eps[0, f, q])
            e1 = np.conj(eps[1, f, q])
            e2 = np.conj(eps[2, f, q])
            e3 = np.conj(eps[3, f, q])
            e4 = np.conj(eps[4, f, q])
            e5 = np.conj(eps[5, f, q])

            E_kq[0, q] += a_fwd_x * e0
            E_kq[1, q] += a_fwd_y * e1
            E_kq[2, q] += a_fwd_z * e2
            E_kq[3, q] += a_bwd_x * e3
            E_kq[4, q] += a_bwd_y * e4
            E_kq[5, q] += a_bwd_z * e5

    return E_kq


@njit(cache=True)
def sse_rhs_jit(
    u_re, u_im, r, v, t,
    omega0s, d_ge, d_eg,
    E_kq,
    n_ground, n_excited, mass_nat,
    H_Z_ground, zeeman_z_diag_excited,
    B_x_G, B_y_G, B_z_G,
    zeeman_x_ground, zeeman_y_ground, zeeman_z_ground,
):
    """
    JIT-compiled RHS of the SSE ODE: du/dt = f(t, u).

    All inputs are plain numpy arrays. Returns (du_re, du_im, dv) arrays.
    The position derivative is trivially dr/dt = v (done outside).

    Parameters
    ----------
    u_re, u_im : ndarray, shape (N,)
        Real and imaginary parts of the wavefunction.
    r : ndarray, shape (3,)
        Position in natural units.
    v : ndarray, shape (3,)
        Velocity in natural units.
    t : float
        Time in natural units.
    omega0s : ndarray, shape (N,)
        State energies (including diagonal Zeeman) in natural units.
    d_ge : ndarray, shape (n_g, n_e, 3)
        Transition dipole matrix elements (ground-excited).
    d_eg : ndarray, shape (n_e, n_g, 3)
        Transition dipole matrix elements (excited-ground).
    E_kq : ndarray, shape (6, 3), complex
        Electric field amplitudes.
    n_ground, n_excited : int
    mass_nat : float
        Mass in natural units.
    H_Z_ground : ndarray, shape (n_g, n_g), complex
        Full Zeeman Hamiltonian for ground states.
    zeeman_z_diag_excited : ndarray, shape (n_e,)
        Diagonal Zeeman for excited states.
    B_x_G, B_y_G, B_z_G : float
        B-field components in Gauss.
    zeeman_x_ground, zeeman_y_ground, zeeman_z_ground : ndarray, shape (n_g, n_g), complex
        Zeeman matrices for ground states.

    Returns
    -------
    du_re : ndarray, shape (N,)
    du_im : ndarray, shape (N,)
    F : ndarray, shape (3,)
        Force in natural units.
    """
    N = n_ground + n_excited
    n_g = n_ground
    n_e = n_excited

    # Reconstruct complex psi (DO NOT normalize)
    psi = np.empty(N, dtype=np.complex128)
    for i in range(N):
        psi[i] = u_re[i] + 1j * u_im[i]

    # Interaction picture transformation
    eiw0t = np.empty(N, dtype=np.complex128)
    for i in range(N):
        eiw0t[i] = np.exp(-1j * omega0s[i] * t)

    psi_int = np.empty(N, dtype=np.complex128)
    for i in range(N):
        psi_int[i] = psi[i] * eiw0t[i]

    # psi_q for ground states: psi_q_g[g, q] = sum_e d_ge[g, e, q] * psi_e_int[e]
    psi_q_g = np.zeros((n_g, 3), dtype=np.complex128)
    for ig in range(n_g):
        for q in range(3):
            val = 0.0 + 0.0j
            for ie in range(n_e):
                val += d_ge[ig, ie, q] * psi_int[n_g + ie]
            psi_q_g[ig, q] = val

    # psi_q for excited states: psi_q_e[e, q] = sum_g d_eg[e, g, q] * psi_g_int[g]
    psi_q_e = np.zeros((n_e, 3), dtype=np.complex128)
    for ie in range(n_e):
        for q in range(3):
            val = 0.0 + 0.0j
            for ig in range(n_g):
                val += d_eg[ie, ig, q] * psi_int[ig]
            psi_q_e[ie, q] = val

    # Total field: sum over all 6 beams
    E_total = np.zeros(3, dtype=np.complex128)
    for q in range(3):
        for b in range(6):
            E_total[q] += E_kq[b, q]

    # Dipole expectation: d_exp[q] = sum_g conj(psi_g_int[g]) * psi_q_g[g, q]
    d_exp = np.zeros(3, dtype=np.complex128)
    for q in range(3):
        val = 0.0 + 0.0j
        for ig in range(n_g):
            val += np.conj(psi_int[ig]) * psi_q_g[ig, q]
        d_exp[q] = val

    # Force calculation
    F = np.zeros(3)
    for k_dir in range(3):
        F_k = 0.0
        for q in range(3):
            E_diff_re = E_kq[k_dir, q].real - E_kq[k_dir + 3, q].real
            E_diff_im = E_kq[k_dir, q].imag - E_kq[k_dir + 3, q].imag
            # Multiply by -i
            Ekq_re = E_diff_im
            Ekq_im = -E_diff_re
            d_re = d_exp[q].real
            d_im = d_exp[q].imag
            F_k -= 2.0 * (d_re * Ekq_re - d_im * Ekq_im)
        F[k_dir] = F_k

    # dpsi/dt from laser coupling
    dpsi_int = np.zeros(N, dtype=np.complex128)

    # Ground states: dpsi_g = -i * sum_q E_total[q] * psi_q_g[g, q]
    for ig in range(n_g):
        val = 0.0 + 0.0j
        for q in range(3):
            val += E_total[q] * psi_q_g[ig, q]
        # Multiply by -i: (-i)(a+ib) = b - ia
        dpsi_int[ig] = val.imag - 1j * val.real

    # Excited states: dpsi_e = -i * sum_q conj(E_total[q]) * psi_q_e[e, q]
    for ie in range(n_e):
        val = 0.0 + 0.0j
        for q in range(3):
            val += np.conj(E_total[q]) * psi_q_e[ie, q]
        dpsi_int[n_g + ie] = val.imag - 1j * val.real

    # Non-Hermitian decay: -Gamma/2 * psi_e (Gamma=1 in natural units)
    for ie in range(n_e):
        dpsi_int[n_g + ie] -= 0.5 * psi_int[n_g + ie]

    # Off-diagonal Zeeman coupling for ground states
    # Build H_Z in interaction picture and apply off-diagonal part
    omega_g = omega0s[:n_g]
    phase_vec = np.empty(n_g, dtype=np.complex128)
    for i in range(n_g):
        phase_vec[i] = np.exp(-1j * omega_g[i] * t)

    # H_Z_int[i,j] = H_Z[i,j] * phase_vec[i] * conj(phase_vec[j])
    # then zero diagonal, then apply -i * H_Z_int @ psi_g_int
    H_Z = B_x_G * zeeman_x_ground + B_y_G * zeeman_y_ground + B_z_G * zeeman_z_ground
    for ig in range(n_g):
        val = 0.0 + 0.0j
        for jg in range(n_g):
            if ig == jg:
                continue  # skip diagonal
            H_ij = H_Z[ig, jg] * phase_vec[ig] * np.conj(phase_vec[jg])
            val += H_ij * psi_int[jg]
        # Multiply by -i
        dpsi_int[ig] += val.imag - 1j * val.real

    # Transform back from interaction picture
    du_re = np.empty(N)
    du_im = np.empty(N)
    for i in range(N):
        eiw0t_conj = np.exp(1j * omega0s[i] * t)
        dpsi_i = dpsi_int[i] * eiw0t_conj
        du_re[i] = dpsi_i.real
        du_im[i] = dpsi_i.imag

    return du_re, du_im, F
