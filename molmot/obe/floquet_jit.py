"""
Numba JIT-compiled Floquet Liouvillian block assembly.

The Floquet OBE builds a (2*n_max+1)*N^2 block-tridiagonal matrix.
The Liouvillian block construction (coherent part + dissipator) is
the most expensive part of each solve. This module JIT-compiles the
block assembly and the conversion from Hamiltonian to Liouvillian.

Usage:
    from molmot.obe.floquet_jit import build_floquet_matrix_jit
    M = build_floquet_matrix_jit(L0, L_plus, L_minus, kv, n_max, N)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def build_floquet_matrix_jit(L0, L_plus, L_minus, kv, n_max, N2):
    """
    Build the full block-tridiagonal Floquet matrix, JIT-compiled.

    Parameters
    ----------
    L0 : ndarray, shape (N2, N2), complex
        DC Liouvillian block.
    L_plus : ndarray, shape (N2, N2), complex
        Raising Liouvillian block.
    L_minus : ndarray, shape (N2, N2), complex
        Lowering Liouvillian block.
    kv : float
        k * v in rad/s.
    n_max : int
        Floquet truncation order.
    N2 : int
        N^2 where N is the number of states.

    Returns
    -------
    M : ndarray, shape (n_blocks * N2, n_blocks * N2), complex
    """
    n_blocks = 2 * n_max + 1
    total_size = n_blocks * N2
    M = np.zeros((total_size, total_size), dtype=np.complex128)

    for b in range(n_blocks):
        n = b - n_max  # Fourier order
        row_start = b * N2
        row_end = (b + 1) * N2

        # Diagonal block: L0 - i*n*kv*I
        for i in range(N2):
            for j in range(N2):
                M[row_start + i, row_start + j] = L0[i, j]
            M[row_start + i, row_start + i] -= 1j * n * kv

        # Sub-diagonal: L_plus @ rho_{n-1}
        if b > 0:
            col_start = (b - 1) * N2
            for i in range(N2):
                for j in range(N2):
                    M[row_start + i, col_start + j] = L_plus[i, j]

        # Super-diagonal: L_minus @ rho_{n+1}
        if b < n_blocks - 1:
            col_start = (b + 1) * N2
            for i in range(N2):
                for j in range(N2):
                    M[row_start + i, col_start + j] = L_minus[i, j]

    return M


@njit(cache=True)
def build_liouvillian_dissipator_jit(d_squared, Gamma, n_ground, n_excited, N):
    """
    Build the dissipator part of the Liouvillian, JIT-compiled.

    Constructs sum_c [L_c rho L_c^dag - 0.5 {L_c^dag L_c, rho}]
    in the column-major vectorized form.

    Parameters
    ----------
    d_squared : ndarray, shape (n_ground, n_excited, 3)
        |TDM|^2 array.
    Gamma : float
        Natural linewidth (rad/s).
    n_ground, n_excited : int
    N : int
        Total number of states.

    Returns
    -------
    L_dissipator : ndarray, shape (N^2, N^2), complex
    """
    N2 = N * N
    L = np.zeros((N2, N2), dtype=np.complex128)

    for ie in range(n_excited):
        ie_abs = ie + n_ground
        total = 0.0
        for ig in range(n_ground):
            for q in range(3):
                total += d_squared[ig, ie, q]

        if total < 1e-30:
            continue

        for ig in range(n_ground):
            for q in range(3):
                br = d_squared[ig, ie, q] / total
                if br < 1e-15:
                    continue
                Gc = Gamma * br

                # Jump operator L_c = sqrt(Gc) |ig><ie_abs|
                # L_c[ig, ie_abs] = sqrt(Gc), rest zero
                # L_c^dag L_c = Gc |ie_abs><ie_abs|
                sqrt_Gc = np.sqrt(Gc)

                # kron(conj(L_c), L_c) contribution:
                # L_c_conj[a, b] * L_c[c, d] at position [b*N+a, d*N+c]
                # Only nonzero when a=ig, b=ie_abs and c=ig, d=ie_abs
                row_idx = ie_abs * N + ig  # column-major: rho_{ig, ie_abs}
                col_idx = ie_abs * N + ig
                L[row_idx, col_idx] += Gc  # sqrt(Gc) * sqrt(Gc)

                # -0.5 * kron(I, L_c^dag L_c): acts on second index
                # L_c^dag L_c = Gc * |ie_abs><ie_abs|
                # kron(I, LdL)[a*N+b, c*N+d] = delta(a,c) * LdL[b,d]
                # = delta(a,c) * Gc * delta(b,ie_abs) * delta(d,ie_abs)
                for a in range(N):
                    row = a * N + ie_abs  # rho_{ie_abs, a} in col-major
                    # Wait: col-major vec: rho_{i,j} at index j*N + i
                    # So rho element (b_row, a_col) is at a_col*N + b_row
                    # kron(I, LdL) @ vec(rho): for each column a,
                    # element b of result is sum_d LdL[b,d]*rho[d,a]
                    # In vectorized form: result at index a*N+b = sum_d LdL[b,d] * vec[a*N+d]
                    # So L_kron[a*N+b, a*N+d] += LdL[b,d]
                    # Here LdL[b,d] = Gc * delta(b,ie_abs)*delta(d,ie_abs)
                    # So L_kron[a*N+ie_abs, a*N+ie_abs] += Gc
                    idx = a * N + ie_abs
                    L[idx, idx] -= 0.5 * Gc

                # -0.5 * kron(LdL^T, I): acts on first index
                # LdL^T = Gc * |ie_abs><ie_abs| (since LdL is diagonal/symmetric)
                # kron(LdL^T, I)[a*N+b, c*N+d] = LdL^T[a,c] * delta(b,d)
                # = Gc * delta(a,ie_abs) * delta(c,ie_abs) * delta(b,d)
                for b in range(N):
                    idx = ie_abs * N + b
                    L[idx, idx] -= 0.5 * Gc

    return L
