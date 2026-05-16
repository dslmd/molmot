"""
Numba JIT-compiled rate equation solver.

Drop-in replacement for solve_rate_equations() that runs ~10-20x faster
by compiling the inner loop to native code.

Usage:
    from molmot.obe.rate_equations_jit import solve_rate_equations_jit
    F, p, R = solve_rate_equations_jit(mol_data, beams, v, z, B_grad)
"""

import numpy as np
from numba import njit

from ..constants import hbar


@njit(cache=True)
def _rate_eq_core(energies, d_squared, zeeman_z_diag,
                  beam_dirs, beam_freqs, beam_pols, beam_s0s,
                  v, z, B_gradient, Gamma, k_wave, Gamma_eff,
                  n_ground, n_excited):
    """JIT-compiled inner loop."""
    n_g = n_ground
    n_e = n_excited
    n_beams = len(beam_dirs)
    hbar_val = 1.0545718e-34

    B_gauss = B_gradient * z * 1e4
    gamma_half = Gamma / 2.0
    gamma_half_sq = gamma_half * gamma_half

    Es = np.empty(n_g + n_e)
    for i in range(n_g + n_e):
        Es[i] = energies[i] + B_gauss * zeeman_z_diag[i] * Gamma / (2.0 * np.pi)

    # Branching ratios
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = 0.0
        for ig in range(n_g):
            for q in range(3):
                total += d_squared[ig, ie, q]
        if total > 1e-30:
            for ig in range(n_g):
                s = 0.0
                for q in range(3):
                    s += d_squared[ig, ie, q]
                BR[ie, ig] = s / total

    # Excitation rates and force per channel
    R_sum = np.zeros((n_g, n_e))
    F_total_arr = np.zeros((n_g, n_e))

    # Saturation totals per ground state
    s_total = np.zeros(n_g)

    for ib in range(n_beams):
        kdir = beam_dirs[ib]
        freq = beam_freqs[ib]
        pol = int(beam_pols[ib])
        s0 = beam_s0s[ib]
        doppler = -k_wave * kdir * v / (2.0 * np.pi)

        for ig in range(n_g):
            for ie in range(n_e):
                d2 = d_squared[ig, ie, pol]
                if d2 < 1e-15:
                    continue
                omega_trans = Es[n_g + ie] - Es[ig]
                delta_eff = (freq + doppler - omega_trans) * 2.0 * np.pi
                L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq)
                rate_unsaturated = gamma_half * s0 * d2 * L * Gamma_eff
                R_sum[ig, ie] += rate_unsaturated
                F_total_arr[ig, ie] += hbar_val * k_wave * kdir * rate_unsaturated
                s_total[ig] += s0 * d2 * L

    # Apply saturation correction
    for ig in range(n_g):
        sat_corr = 1.0 / (1.0 + s_total[ig])
        for ie in range(n_e):
            R_sum[ig, ie] *= sat_corr
            F_total_arr[ig, ie] *= sat_corr

    # Population matrix
    n = n_g
    M = np.zeros((n, n))
    for ig in range(n_g):
        for ie in range(n_e):
            M[ig, ig] -= R_sum[ig, ie]
        for ik in range(n_g):
            for ie in range(n_e):
                M[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    M[n-1, :] = 1.0
    b = np.zeros(n)
    b[n-1] = 1.0

    # Gaussian elimination with partial pivoting
    A = M.copy()
    for col in range(n):
        max_val = abs(A[col, col])
        max_row = col
        for row in range(col+1, n):
            if abs(A[row, col]) > max_val:
                max_val = abs(A[row, col])
                max_row = row
        if max_row != col:
            for j in range(n):
                A[col, j], A[max_row, j] = A[max_row, j], A[col, j]
            b[col], b[max_row] = b[max_row], b[col]
        for row in range(col+1, n):
            if abs(A[col, col]) < 1e-30:
                continue
            factor = A[row, col] / A[col, col]
            for j in range(col, n):
                A[row, j] -= factor * A[col, j]
            b[row] -= factor * b[col]

    p = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n):
            s -= A[i, j] * p[j]
        if abs(A[i, i]) > 1e-30:
            p[i] = s / A[i, i]

    for i in range(n):
        if p[i] < 0:
            p[i] = 0.0
    total_p = 0.0
    for i in range(n):
        total_p += p[i]
    if total_p > 0:
        for i in range(n):
            p[i] /= total_p

    force = 0.0
    R_scatter = 0.0
    for ig in range(n_g):
        for ie in range(n_e):
            force += p[ig] * F_total_arr[ig, ie]
            R_scatter += p[ig] * R_sum[ig, ie]

    return force, p, R_scatter


def solve_rate_equations_jit(mol_data, beams, v, z, B_gradient,
                             Gamma_eff_factor=1.0):
    """
    JIT-compiled rate equation solver. Same API as solve_rate_equations().

    ~10-20x faster than the pure Python version after JIT warmup.
    """
    n_beams = len(beams)
    beam_dirs = np.array([b.kdir_z for b in beams])
    beam_freqs = np.array([b.freq_offset for b in beams])
    beam_pols = np.array([b.polarization for b in beams], dtype=np.float64)
    beam_s0s = np.array([b.s0 for b in beams])

    return _rate_eq_core(
        mol_data.energies, mol_data.d_squared, mol_data.zeeman_z_diag,
        beam_dirs, beam_freqs, beam_pols, beam_s0s,
        v, z, B_gradient, mol_data.Gamma, mol_data.k, Gamma_eff_factor,
        mol_data.n_ground, mol_data.n_excited)
