"""
Numba JIT-compiled scattering rate inner loop for 3D force computation.

The 3D force computation in simulator_3d.py does a full Zeeman
diagonalization at each position (not JIT-able due to np.linalg.eigh),
but the scattering rate loop over beams/ground/excited states IS
JIT-compilable and is the dominant cost after diagonalization.

Usage:
    from molmot.mot.simulator_3d_jit import scattering_rate_loop_jit
"""

import numpy as np
from numba import njit


@njit(cache=True)
def scattering_rate_loop_jit(
    d_sq_eig, Es_g, Es_e, BR,
    beam_k_hats, beam_freqs, beam_q_labs, beam_s0s, beam_gauss_factors,
    pol_weights_all,
    v, k, Gamma, Gamma_eff_factor,
    n_ground, n_excited, n_beams,
):
    """
    JIT-compiled inner loop for 3D scattering rate and force computation.

    This replaces the two Python loops over beams/ground/excited states
    in rate_eq_force_3d().

    Parameters
    ----------
    d_sq_eig : ndarray, shape (n_g, n_e, 3)
        |TDM|^2 in the Zeeman eigenstate basis.
    Es_g : ndarray, shape (n_g,)
        Ground state energies (Hz).
    Es_e : ndarray, shape (n_e,)
        Excited state energies (Hz).
    BR : ndarray, shape (n_e, n_g)
        Branching ratios.
    beam_k_hats : ndarray, shape (n_beams, 3)
        Beam propagation unit vectors.
    beam_freqs : ndarray, shape (n_beams,)
        Beam frequencies (Hz).
    beam_q_labs : ndarray, shape (n_beams,), int
        Beam polarization indices.
    beam_s0s : ndarray, shape (n_beams,)
        Beam saturation parameters.
    beam_gauss_factors : ndarray, shape (n_beams,)
        Gaussian beam profile factors.
    pol_weights_all : ndarray, shape (n_beams, 3)
        Polarization decomposition weights in local B-frame for each beam.
    v : ndarray, shape (3,)
        Velocity (m/s).
    k : float
        Wavenumber (1/m).
    Gamma : float
        Natural linewidth (rad/s).
    Gamma_eff_factor : float
    n_ground, n_excited, n_beams : int

    Returns
    -------
    F_total : ndarray, shape (3,)
        Force (N).
    R_scatter : float
        Total scattering rate (rad/s).
    populations : ndarray, shape (n_g,)
        Steady-state populations.
    """
    hbar_val = 1.0545718e-34
    n_g = n_ground
    n_e = n_excited
    gamma_half = Gamma / 2.0
    gamma_half_sq = gamma_half * gamma_half

    # First pass: compute saturation totals and unsaturated rates
    s_total = np.zeros(n_g)
    # Store rates per beam for second pass
    beam_rate_sum = np.zeros((n_beams, n_g, n_e))

    for ib in range(n_beams):
        doppler_hz = 0.0
        for d in range(3):
            doppler_hz += -k * beam_k_hats[ib, d] * v[d]
        doppler_hz /= (2.0 * np.pi)

        s_eff = beam_s0s[ib] * beam_gauss_factors[ib]

        for ig in range(n_g):
            for ie in range(n_e):
                omega_trans = Es_e[ie] - Es_g[ig]
                rate_sum = 0.0
                for q_local in range(3):
                    d2 = d_sq_eig[ig, ie, q_local]
                    if d2 < 1e-15:
                        continue
                    delta_eff = (beam_freqs[ib] + doppler_hz - omega_trans) * 2.0 * np.pi
                    L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq)
                    rate_sum += pol_weights_all[ib, q_local] * d2 * L

                s_total[ig] += s_eff * rate_sum
                beam_rate_sum[ib, ig, ie] = rate_sum * s_eff

    # Build and solve population matrix
    R_exc_total = np.zeros((n_g, n_e))
    for ib in range(n_beams):
        for ig in range(n_g):
            sat_denom = 1.0 + s_total[ig]
            for ie in range(n_e):
                rate = gamma_half * beam_rate_sum[ib, ig, ie] * Gamma_eff_factor / sat_denom
                R_exc_total[ig, ie] += rate

    # Population matrix
    M = np.zeros((n_g, n_g))
    for ig in range(n_g):
        for ie in range(n_e):
            M[ig, ig] -= R_exc_total[ig, ie]
        for ik in range(n_g):
            for ie in range(n_e):
                M[ig, ik] += R_exc_total[ik, ie] * BR[ie, ig]

    # Replace last row with normalization
    M[n_g - 1, :] = 1.0
    b = np.zeros(n_g)
    b[n_g - 1] = 1.0

    # Gaussian elimination with partial pivoting
    A = M.copy()
    for col in range(n_g):
        max_val = abs(A[col, col])
        max_row = col
        for row in range(col + 1, n_g):
            if abs(A[row, col]) > max_val:
                max_val = abs(A[row, col])
                max_row = row
        if max_row != col:
            for j in range(n_g):
                A[col, j], A[max_row, j] = A[max_row, j], A[col, j]
            b[col], b[max_row] = b[max_row], b[col]
        for row in range(col + 1, n_g):
            if abs(A[col, col]) < 1e-30:
                continue
            factor = A[row, col] / A[col, col]
            for j in range(col, n_g):
                A[row, j] -= factor * A[col, j]
            b[row] -= factor * b[col]

    p = np.zeros(n_g)
    for i in range(n_g - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n_g):
            s -= A[i, j] * p[j]
        if abs(A[i, i]) > 1e-30:
            p[i] = s / A[i, i]

    # Clamp and normalize
    for i in range(n_g):
        if p[i] < 0.0:
            p[i] = 0.0
    total_p = 0.0
    for i in range(n_g):
        total_p += p[i]
    if total_p > 0:
        for i in range(n_g):
            p[i] /= total_p

    # Compute force
    F_total = np.zeros(3)
    R_scatter = 0.0

    for ib in range(n_beams):
        for ig in range(n_g):
            sat_denom = 1.0 + s_total[ig]
            for ie in range(n_e):
                rate = gamma_half * beam_rate_sum[ib, ig, ie] * Gamma_eff_factor / sat_denom
                for d in range(3):
                    F_total[d] += hbar_val * k * beam_k_hats[ib, d] * rate * p[ig]
                R_scatter += rate * p[ig]

    return F_total, R_scatter, p
