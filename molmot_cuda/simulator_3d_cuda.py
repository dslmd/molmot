"""
CUDA-accelerated 3D MOT simulator using rate equations.

GPU-batched version of ``molmot.mot.simulator_3d``.  All (r, v) points
are evaluated in parallel on the GPU, giving 100--1000x speedup for
force-map and ensemble-trajectory calculations.

The CUDA kernel for each particle/grid-point performs:
  1. Quadrupole B-field from position
  2. Ground-state Zeeman diagonalisation (12x12 Jacobi)
  3. TDM basis transformation
  4. Polarisation decomposition (Wigner d-matrix for j=1)
  5. Scattering rate loop (n_beams x n_ground x n_excited x 3 pol)
  6. Population solve (12x12 Gaussian elimination with partial pivoting)
  7. Force accumulation

One CUDA thread per particle.

References
----------
* Tarbutt, NJP 17, 015007 (2015)
* Lasner et al., PRL 134, 083401 (2025)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .utils import _require_cupy, to_gpu, to_cpu

try:
    import cupy as cp
    from cupy import RawKernel
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants embedded in the kernel
# ---------------------------------------------------------------------------

_HBAR = 1.054571817e-34
_PI = 3.14159265358979323846
_SQRT2 = 1.4142135623730951

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
extern "C" __global__
void rate_eq_force_3d_kernel(
    // Molecular data (constant)
    const double* __restrict__ energies,     // [N]
    const double* __restrict__ d_squared,    // [n_g, n_e, 3]
    const double* __restrict__ tdm_abs,      // [N, N, 3]  |d[ig, ie_abs, q]|
    const double* __restrict__ zeeman_x_re,  // [N, N]
    const double* __restrict__ zeeman_x_im,  // [N, N]
    const double* __restrict__ zeeman_y_re,  // [N, N]
    const double* __restrict__ zeeman_y_im,  // [N, N]
    const double* __restrict__ zeeman_z_re,  // [N, N]
    const double* __restrict__ zeeman_z_im,  // [N, N]
    double Gamma, double k_wave, double Gamma_eff_factor,
    int n_g, int n_e, int N,
    // Beam data (constant)
    const double* __restrict__ beam_k_hats,   // [n_beams, 3]
    const double* __restrict__ beam_freqs,    // [n_beams]
    const int*    __restrict__ beam_q_labs,    // [n_beams]
    const double* __restrict__ beam_s0s,      // [n_beams]
    const double* __restrict__ beam_radii,    // [n_beams] (0 = infinite)
    int n_beams,
    // Per-particle input
    const double* __restrict__ r_arr,   // [n_particles, 3]
    const double* __restrict__ v_arr,   // [n_particles, 3]
    double B_gradient,
    int n_particles,
    // Output
    double* __restrict__ F_out,         // [n_particles, 3]
    double* __restrict__ R_scatter_out, // [n_particles]
    double* __restrict__ pop_out        // [n_particles, n_g]
)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    const double hbar_val = 1.054571817e-34;
    const double PI = 3.14159265358979323846;
    const double SQRT2 = 1.4142135623730951;

    // --- Position and velocity ---
    double rx = r_arr[pid * 3 + 0];
    double ry = r_arr[pid * 3 + 1];
    double rz = r_arr[pid * 3 + 2];
    double vx = v_arr[pid * 3 + 0];
    double vy = v_arr[pid * 3 + 1];
    double vz = v_arr[pid * 3 + 2];

    // --- 1. Quadrupole B-field (Gauss) ---
    double Bx_G = -B_gradient * rx / 2.0 * 1e4;
    double By_G = -B_gradient * ry / 2.0 * 1e4;
    double Bz_G =  B_gradient * rz * 1e4;
    double B_mag = sqrt(Bx_G * Bx_G + By_G * By_G + Bz_G * Bz_G);

    double B_hat_x, B_hat_y, B_hat_z;
    if (B_mag > 1e-10) {
        B_hat_x = Bx_G / B_mag;
        B_hat_y = By_G / B_mag;
        B_hat_z = Bz_G / B_mag;
    } else {
        B_hat_x = 0.0; B_hat_y = 0.0; B_hat_z = 1.0;
    }

    // --- 2. Build ground-state Zeeman Hamiltonian H_Z (Hermitian, real after symmetrization) ---
    // H_Z = Bx*Zx + By*Zy + Bz*Zz  (ground block only: n_g x n_g)
    // We work with real symmetric part only (eigh returns real eigenvalues).
    // H_Z stored in row-major: H_Z[i*n_g + j]

    // Use shared memory or local arrays for small matrices.
    // For n_g <= 16, use stack arrays.
    double H_Z_re[16*16];
    double H_Z_im[16*16];
    for (int i = 0; i < n_g; i++) {
        for (int j = 0; j < n_g; j++) {
            int idx_full = i * N + j;  // index into full N x N matrix
            double re = Bx_G * zeeman_x_re[idx_full] - Bx_G * 0.0  // real part
                      + By_G * zeeman_y_re[idx_full]
                      + Bz_G * zeeman_z_re[idx_full];
            double im = Bx_G * zeeman_x_im[idx_full]
                      + By_G * zeeman_y_im[idx_full]
                      + Bz_G * zeeman_z_im[idx_full];
            // Hermitianize: H = 0.5*(H + H^dag)
            // We'll do this after filling.
            H_Z_re[i * n_g + j] = re;
            H_Z_im[i * n_g + j] = im;
        }
    }
    // Hermitianize
    for (int i = 0; i < n_g; i++) {
        for (int j = i + 1; j < n_g; j++) {
            double re_ij = H_Z_re[i * n_g + j];
            double im_ij = H_Z_im[i * n_g + j];
            double re_ji = H_Z_re[j * n_g + i];
            double im_ji = H_Z_im[j * n_g + i];
            // (H + H^dag)/2: H[i,j] = (H[i,j] + conj(H[j,i]))/2
            H_Z_re[i * n_g + j] = 0.5 * (re_ij + re_ji);
            H_Z_im[i * n_g + j] = 0.5 * (im_ij - im_ji);
            H_Z_re[j * n_g + i] = 0.5 * (re_ij + re_ji);
            H_Z_im[j * n_g + i] = 0.5 * (-im_ij + im_ji);
        }
        // Diagonal is real
        H_Z_im[i * n_g + i] = 0.0;
    }

    // --- Jacobi eigenvalue iteration for Hermitian matrix ---
    // Eigenvalues in eig_vals[], eigenvectors in U_re[], U_im[] (column-major)
    double eig_vals[16];
    double U_re[16*16];
    double U_im[16*16];

    // Initialize U = I
    for (int i = 0; i < n_g * n_g; i++) { U_re[i] = 0.0; U_im[i] = 0.0; }
    for (int i = 0; i < n_g; i++) { U_re[i * n_g + i] = 1.0; }

    // Copy H_Z into working matrix A (will be modified by Jacobi)
    double A_re[16*16], A_im[16*16];
    for (int i = 0; i < n_g * n_g; i++) {
        A_re[i] = H_Z_re[i];
        A_im[i] = H_Z_im[i];
    }

    // Jacobi iteration (classical, unitary 2x2 rotations)
    for (int sweep = 0; sweep < 50; sweep++) {
        // Check convergence: sum of off-diagonal |A[i,j]|^2
        double off_diag = 0.0;
        for (int i = 0; i < n_g; i++)
            for (int j = i + 1; j < n_g; j++)
                off_diag += A_re[i*n_g+j]*A_re[i*n_g+j] + A_im[i*n_g+j]*A_im[i*n_g+j];
        if (off_diag < 1e-24) break;

        for (int p = 0; p < n_g - 1; p++) {
            for (int q_idx = p + 1; q_idx < n_g; q_idx++) {
                double apq_re = A_re[p * n_g + q_idx];
                double apq_im = A_im[p * n_g + q_idx];
                double apq_abs = sqrt(apq_re * apq_re + apq_im * apq_im);
                if (apq_abs < 1e-15) continue;

                // Phase factor to make A[p,q] real
                double phase_re = apq_re / apq_abs;
                double phase_im = -apq_im / apq_abs;

                double app = A_re[p * n_g + p];
                double aqq = A_re[q_idx * n_g + q_idx];
                double tau = (aqq - app) / (2.0 * apq_abs);

                double t;
                if (tau >= 0.0)
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                else
                    t = -1.0 / (-tau + sqrt(1.0 + tau * tau));

                double c_val = 1.0 / sqrt(1.0 + t * t);
                double s_val = t * c_val;

                // Apply rotation to A: A <- G^dag A G
                // G is unitary: G[p,p]=c, G[q,q]=c, G[p,q]=-s*phase^*, G[q,p]=s*phase

                // Update rows and columns p and q of A
                for (int k = 0; k < n_g; k++) {
                    if (k == p || k == q_idx) continue;
                    // A[k,p] and A[k,q]
                    double akp_re = A_re[k * n_g + p];
                    double akp_im = A_im[k * n_g + p];
                    double akq_re = A_re[k * n_g + q_idx];
                    double akq_im = A_im[k * n_g + q_idx];

                    // new A[k,p] = c*A[k,p] + s*phase*A[k,q]
                    double sp_re = s_val * phase_re;
                    double sp_im = s_val * phase_im;
                    A_re[k * n_g + p] = c_val * akp_re + (sp_re * akq_re - sp_im * akq_im);
                    A_im[k * n_g + p] = c_val * akp_im + (sp_re * akq_im + sp_im * akq_re);
                    // new A[k,q] = c*A[k,q] - s*phase^**A[k,p_old]
                    double spc_re = s_val * phase_re;
                    double spc_im = -s_val * phase_im;  // conj
                    A_re[k * n_g + q_idx] = c_val * akq_re - (spc_re * akp_re - spc_im * akp_im);
                    A_im[k * n_g + q_idx] = c_val * akq_im - (spc_re * akp_im + spc_im * akp_re);

                    // Hermitian: A[p,k] = conj(A[k,p]), A[q,k] = conj(A[k,q])
                    A_re[p * n_g + k] = A_re[k * n_g + p];
                    A_im[p * n_g + k] = -A_im[k * n_g + p];
                    A_re[q_idx * n_g + k] = A_re[k * n_g + q_idx];
                    A_im[q_idx * n_g + k] = -A_im[k * n_g + q_idx];
                }

                // Update diagonal and (p,q) block
                double new_app = c_val * c_val * app + s_val * s_val * aqq + 2.0 * c_val * s_val * apq_abs;
                double new_aqq = c_val * c_val * aqq + s_val * s_val * app - 2.0 * c_val * s_val * apq_abs;
                A_re[p * n_g + p] = new_app;
                A_re[q_idx * n_g + q_idx] = new_aqq;
                A_re[p * n_g + q_idx] = 0.0;
                A_im[p * n_g + q_idx] = 0.0;
                A_re[q_idx * n_g + p] = 0.0;
                A_im[q_idx * n_g + p] = 0.0;

                // Update eigenvector matrix U: U <- U * G
                for (int k = 0; k < n_g; k++) {
                    double ukp_re = U_re[k * n_g + p];
                    double ukp_im = U_im[k * n_g + p];
                    double ukq_re = U_re[k * n_g + q_idx];
                    double ukq_im = U_im[k * n_g + q_idx];

                    double sp_re2 = s_val * phase_re;
                    double sp_im2 = s_val * phase_im;
                    U_re[k * n_g + p] = c_val * ukp_re + (sp_re2 * ukq_re - sp_im2 * ukq_im);
                    U_im[k * n_g + p] = c_val * ukp_im + (sp_re2 * ukq_im + sp_im2 * ukq_re);

                    double spc_re2 = s_val * phase_re;
                    double spc_im2 = -s_val * phase_im;
                    U_re[k * n_g + q_idx] = c_val * ukq_re - (spc_re2 * ukp_re - spc_im2 * ukp_im);
                    U_im[k * n_g + q_idx] = c_val * ukq_im - (spc_re2 * ukp_im + spc_im2 * ukp_re);
                }
            }
        }
    }
    for (int i = 0; i < n_g; i++) eig_vals[i] = A_re[i * n_g + i];

    // --- Zeeman-shifted energies ---
    double Es_g[16], Es_e[16];
    for (int i = 0; i < n_g; i++)
        Es_g[i] = energies[i] + eig_vals[i] * Gamma / (2.0 * PI);
    for (int i = 0; i < n_e; i++) {
        int ie_abs = i + n_g;
        Es_e[i] = energies[ie_abs] + Bz_G * zeeman_z_re[ie_abs * N + ie_abs] * Gamma / (2.0 * PI);
    }

    // --- 3. Transform TDMs to eigenstate basis ---
    // d_sq_eig[ig, ie, q] = |sum_jg conj(U[jg, ig]) * |tdm[jg, ie_abs, q]| |^2
    double d_sq_eig[16 * 4 * 3];
    for (int ig = 0; ig < n_g; ig++) {
        for (int ie = 0; ie < n_e; ie++) {
            int ie_abs = ie + n_g;
            for (int q = 0; q < 3; q++) {
                // d_eig = U^dag @ d_bare  (d_bare are real positive here)
                double sum_re = 0.0, sum_im = 0.0;
                for (int jg = 0; jg < n_g; jg++) {
                    double d_val = tdm_abs[jg * N * 3 + ie_abs * 3 + q];
                    // conj(U[jg, ig]) = U_re[jg*n_g+ig] - i*U_im[jg*n_g+ig]
                    sum_re += U_re[jg * n_g + ig] * d_val;
                    sum_im += -U_im[jg * n_g + ig] * d_val;
                }
                d_sq_eig[(ig * n_e + ie) * 3 + q] = sum_re * sum_re + sum_im * sum_im;
            }
        }
    }

    // --- Branching ratios ---
    double BR[4 * 16];  // BR[ie * n_g + ig]
    for (int ie = 0; ie < n_e; ie++) {
        double total = 0.0;
        for (int ig = 0; ig < n_g; ig++)
            for (int q = 0; q < 3; q++)
                total += d_sq_eig[(ig * n_e + ie) * 3 + q];
        for (int ig = 0; ig < n_g; ig++) {
            double br = 0.0;
            if (total > 1e-30)
                for (int q = 0; q < 3; q++)
                    br += d_sq_eig[(ig * n_e + ie) * 3 + q] / total;
            BR[ie * n_g + ig] = br;
        }
    }

    // --- 4. Polarisation decomposition ---
    double cos_theta = B_hat_z;
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Wigner d^1 squared matrix: d_sq[m_local, m_lab]
    // m indices: 0=sigma-, 1=pi, 2=sigma+
    double ct = cos_theta, st = sin_theta;
    double d_sq_wigner[9];
    d_sq_wigner[0*3+0] = ((1.0+ct)/2.0)*((1.0+ct)/2.0);
    d_sq_wigner[0*3+1] = (st/SQRT2)*(st/SQRT2);
    d_sq_wigner[0*3+2] = ((1.0-ct)/2.0)*((1.0-ct)/2.0);
    d_sq_wigner[1*3+0] = (st/SQRT2)*(st/SQRT2);
    d_sq_wigner[1*3+1] = ct*ct;
    d_sq_wigner[1*3+2] = (st/SQRT2)*(st/SQRT2);
    d_sq_wigner[2*3+0] = ((1.0-ct)/2.0)*((1.0-ct)/2.0);
    d_sq_wigner[2*3+1] = (st/SQRT2)*(st/SQRT2);
    d_sq_wigner[2*3+2] = ((1.0+ct)/2.0)*((1.0+ct)/2.0);

    // --- 5. Scattering rate loop ---
    double gamma_half = Gamma / 2.0;
    double gamma_half_sq = gamma_half * gamma_half;

    double s_total[16];
    for (int i = 0; i < n_g; i++) s_total[i] = 0.0;

    // beam_rate_sum[ib * n_g * n_e + ig * n_e + ie]
    // For n_beams up to ~48, n_g=12, n_e=4: 48*12*4=2304 doubles = 18 KB
    // This is too large for stack in some architectures, so we use
    // a two-pass approach storing per-beam totals more compactly.
    // Actually, we need per-(beam, g, e) rates for force computation.
    // We'll do a single-pass approach storing s_total and recomputing rates.

    // First pass: compute s_total
    for (int ib = 0; ib < n_beams; ib++) {
        double kx = beam_k_hats[ib * 3 + 0];
        double ky = beam_k_hats[ib * 3 + 1];
        double kz = beam_k_hats[ib * 3 + 2];
        double doppler_hz = -(k_wave * (kx * vx + ky * vy + kz * vz)) / (2.0 * PI);

        // Gaussian beam profile
        double r_parallel = rx * kx + ry * ky + rz * kz;
        double r_perp_sq = (rx*rx + ry*ry + rz*rz) - r_parallel * r_parallel;
        if (r_perp_sq < 0.0) r_perp_sq = 0.0;
        double gauss = 1.0;
        if (beam_radii[ib] > 0.0)
            gauss = exp(-2.0 * r_perp_sq / (beam_radii[ib] * beam_radii[ib]));

        double s_eff = beam_s0s[ib] * gauss;
        int q_lab = beam_q_labs[ib];

        // Polarisation weights for this beam
        double pw0 = d_sq_wigner[0 * 3 + q_lab];
        double pw1 = d_sq_wigner[1 * 3 + q_lab];
        double pw2 = d_sq_wigner[2 * 3 + q_lab];
        // Normalise
        double pw_sum = pw0 + pw1 + pw2;
        if (pw_sum > 1e-15) { pw0 /= pw_sum; pw1 /= pw_sum; pw2 /= pw_sum; }

        for (int ig = 0; ig < n_g; ig++) {
            for (int ie = 0; ie < n_e; ie++) {
                double omega_trans = Es_e[ie] - Es_g[ig];
                double rate_sum = 0.0;
                for (int q_local = 0; q_local < 3; q_local++) {
                    double d2 = d_sq_eig[(ig * n_e + ie) * 3 + q_local];
                    if (d2 < 1e-15) continue;
                    double pw_q;
                    if (q_local == 0) pw_q = pw0;
                    else if (q_local == 1) pw_q = pw1;
                    else pw_q = pw2;
                    double delta_eff = (beam_freqs[ib] + doppler_hz - omega_trans) * 2.0 * PI;
                    double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);
                    rate_sum += pw_q * d2 * L;
                }
                s_total[ig] += s_eff * rate_sum;
            }
        }
    }

    // --- 6. Build and solve population matrix ---
    // Recompute rates with saturation, accumulate R_exc_total
    double R_exc_total[16 * 4];  // [ig * n_e + ie]
    for (int i = 0; i < n_g * n_e; i++) R_exc_total[i] = 0.0;

    for (int ib = 0; ib < n_beams; ib++) {
        double kx = beam_k_hats[ib * 3 + 0];
        double ky = beam_k_hats[ib * 3 + 1];
        double kz = beam_k_hats[ib * 3 + 2];
        double doppler_hz = -(k_wave * (kx * vx + ky * vy + kz * vz)) / (2.0 * PI);

        double r_parallel = rx * kx + ry * ky + rz * kz;
        double r_perp_sq = (rx*rx + ry*ry + rz*rz) - r_parallel * r_parallel;
        if (r_perp_sq < 0.0) r_perp_sq = 0.0;
        double gauss = 1.0;
        if (beam_radii[ib] > 0.0)
            gauss = exp(-2.0 * r_perp_sq / (beam_radii[ib] * beam_radii[ib]));
        double s_eff = beam_s0s[ib] * gauss;
        int q_lab = beam_q_labs[ib];

        double pw0 = d_sq_wigner[0 * 3 + q_lab];
        double pw1 = d_sq_wigner[1 * 3 + q_lab];
        double pw2 = d_sq_wigner[2 * 3 + q_lab];
        double pw_sum = pw0 + pw1 + pw2;
        if (pw_sum > 1e-15) { pw0 /= pw_sum; pw1 /= pw_sum; pw2 /= pw_sum; }

        for (int ig = 0; ig < n_g; ig++) {
            double sat_denom = 1.0 + s_total[ig];
            for (int ie = 0; ie < n_e; ie++) {
                double omega_trans = Es_e[ie] - Es_g[ig];
                double rate_sum = 0.0;
                for (int q_local = 0; q_local < 3; q_local++) {
                    double d2 = d_sq_eig[(ig * n_e + ie) * 3 + q_local];
                    if (d2 < 1e-15) continue;
                    double pw_q;
                    if (q_local == 0) pw_q = pw0;
                    else if (q_local == 1) pw_q = pw1;
                    else pw_q = pw2;
                    double delta_eff = (beam_freqs[ib] + doppler_hz - omega_trans) * 2.0 * PI;
                    double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);
                    rate_sum += pw_q * d2 * L;
                }
                R_exc_total[ig * n_e + ie] += gamma_half * rate_sum * s_eff * Gamma_eff_factor / sat_denom;
            }
        }
    }

    // Population matrix M: dpi/dt = sum_j M[i,j] p[j] = 0
    double M[16 * 16];
    for (int i = 0; i < n_g * n_g; i++) M[i] = 0.0;

    for (int ig = 0; ig < n_g; ig++) {
        for (int ie = 0; ie < n_e; ie++)
            M[ig * n_g + ig] -= R_exc_total[ig * n_e + ie];
        for (int ik = 0; ik < n_g; ik++) {
            for (int ie = 0; ie < n_e; ie++)
                M[ig * n_g + ik] += R_exc_total[ik * n_e + ie] * BR[ie * n_g + ig];
        }
    }

    // Replace last row with normalisation: sum p_i = 1
    for (int j = 0; j < n_g; j++) M[(n_g - 1) * n_g + j] = 1.0;
    double b_vec[16];
    for (int i = 0; i < n_g; i++) b_vec[i] = 0.0;
    b_vec[n_g - 1] = 1.0;

    // Gaussian elimination with partial pivoting
    double A_mat[16 * 16];
    for (int i = 0; i < n_g * n_g; i++) A_mat[i] = M[i];

    for (int col = 0; col < n_g; col++) {
        // Find pivot
        double max_val = fabs(A_mat[col * n_g + col]);
        int max_row = col;
        for (int row = col + 1; row < n_g; row++) {
            double val = fabs(A_mat[row * n_g + col]);
            if (val > max_val) { max_val = val; max_row = row; }
        }
        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < n_g; j++) {
                double tmp = A_mat[col * n_g + j];
                A_mat[col * n_g + j] = A_mat[max_row * n_g + j];
                A_mat[max_row * n_g + j] = tmp;
            }
            double tmp = b_vec[col]; b_vec[col] = b_vec[max_row]; b_vec[max_row] = tmp;
        }
        // Eliminate
        for (int row = col + 1; row < n_g; row++) {
            if (fabs(A_mat[col * n_g + col]) < 1e-30) continue;
            double factor = A_mat[row * n_g + col] / A_mat[col * n_g + col];
            for (int j = col; j < n_g; j++)
                A_mat[row * n_g + j] -= factor * A_mat[col * n_g + j];
            b_vec[row] -= factor * b_vec[col];
        }
    }

    // Back substitution
    double p[16];
    for (int i = n_g - 1; i >= 0; i--) {
        double s = b_vec[i];
        for (int j = i + 1; j < n_g; j++) s -= A_mat[i * n_g + j] * p[j];
        if (fabs(A_mat[i * n_g + i]) > 1e-30) p[i] = s / A_mat[i * n_g + i];
        else p[i] = 0.0;
    }

    // Clamp and normalise
    double total_p = 0.0;
    for (int i = 0; i < n_g; i++) { if (p[i] < 0.0) p[i] = 0.0; total_p += p[i]; }
    if (total_p > 0.0) for (int i = 0; i < n_g; i++) p[i] /= total_p;
    else for (int i = 0; i < n_g; i++) p[i] = 1.0 / n_g;

    // --- 7. Compute force ---
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double R_total = 0.0;

    for (int ib = 0; ib < n_beams; ib++) {
        double kx = beam_k_hats[ib * 3 + 0];
        double ky = beam_k_hats[ib * 3 + 1];
        double kz = beam_k_hats[ib * 3 + 2];
        double doppler_hz = -(k_wave * (kx * vx + ky * vy + kz * vz)) / (2.0 * PI);

        double r_parallel = rx * kx + ry * ky + rz * kz;
        double r_perp_sq = (rx*rx + ry*ry + rz*rz) - r_parallel * r_parallel;
        if (r_perp_sq < 0.0) r_perp_sq = 0.0;
        double gauss = 1.0;
        if (beam_radii[ib] > 0.0)
            gauss = exp(-2.0 * r_perp_sq / (beam_radii[ib] * beam_radii[ib]));
        double s_eff = beam_s0s[ib] * gauss;
        int q_lab = beam_q_labs[ib];

        double pw0 = d_sq_wigner[0 * 3 + q_lab];
        double pw1 = d_sq_wigner[1 * 3 + q_lab];
        double pw2 = d_sq_wigner[2 * 3 + q_lab];
        double pw_sum2 = pw0 + pw1 + pw2;
        if (pw_sum2 > 1e-15) { pw0 /= pw_sum2; pw1 /= pw_sum2; pw2 /= pw_sum2; }

        for (int ig = 0; ig < n_g; ig++) {
            double sat_denom = 1.0 + s_total[ig];
            for (int ie = 0; ie < n_e; ie++) {
                double omega_trans = Es_e[ie] - Es_g[ig];
                double rate_sum = 0.0;
                for (int q_local = 0; q_local < 3; q_local++) {
                    double d2 = d_sq_eig[(ig * n_e + ie) * 3 + q_local];
                    if (d2 < 1e-15) continue;
                    double pw_q;
                    if (q_local == 0) pw_q = pw0;
                    else if (q_local == 1) pw_q = pw1;
                    else pw_q = pw2;
                    double delta_eff = (beam_freqs[ib] + doppler_hz - omega_trans) * 2.0 * PI;
                    double L = gamma_half_sq / (delta_eff * delta_eff + gamma_half_sq);
                    rate_sum += pw_q * d2 * L;
                }
                double rate = gamma_half * rate_sum * s_eff * Gamma_eff_factor / sat_denom;
                Fx += hbar_val * k_wave * kx * rate * p[ig];
                Fy += hbar_val * k_wave * ky * rate * p[ig];
                Fz += hbar_val * k_wave * kz * rate * p[ig];
                R_total += rate * p[ig];
            }
        }
    }

    // Write output
    F_out[pid * 3 + 0] = Fx;
    F_out[pid * 3 + 1] = Fy;
    F_out[pid * 3 + 2] = Fz;
    R_scatter_out[pid] = R_total;
    for (int i = 0; i < n_g; i++) pop_out[pid * n_g + i] = p[i];
}
"""


# ---------------------------------------------------------------------------
# Trajectory integration kernel (leapfrog)
# ---------------------------------------------------------------------------

_TRAJECTORY_KERNEL_SOURCE = r"""
extern "C" __global__
void leapfrog_step_kernel(
    // Same molecular + beam data as force kernel (omitted for brevity,
    // passed via a wrapper).  This kernel is called per time step.
    double* __restrict__ r_arr,    // [n_particles, 3]  in/out
    double* __restrict__ v_half,   // [n_particles, 3]  in/out (half-step velocity)
    double* __restrict__ v_full,   // [n_particles, 3]  out (centered velocity)
    const double* __restrict__ F,  // [n_particles, 3]  force at current position
    double dt, double inv_mass,
    int n_particles
)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    // Full position step: r = r + dt * v_half
    for (int d = 0; d < 3; d++)
        r_arr[pid * 3 + d] += dt * v_half[pid * 3 + d];

    // NOTE: Force at new position must be computed externally,
    // then velocity update is done in a second kernel call.
}

extern "C" __global__
void velocity_update_kernel(
    double* __restrict__ v_half,      // [n_particles, 3]  in/out
    double* __restrict__ v_centered,  // [n_particles, 3]  out
    const double* __restrict__ F,     // [n_particles, 3]  force at new position
    double dt, double inv_mass,
    int n_particles
)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    for (int d = 0; d < 3; d++) {
        double v_old = v_half[pid * 3 + d];
        double v_new = v_old + dt * F[pid * 3 + d] * inv_mass;
        v_centered[pid * 3 + d] = 0.5 * (v_old + v_new);
        v_half[pid * 3 + d] = v_new;
    }
}
"""


# ---------------------------------------------------------------------------
# Compiled kernel cache
# ---------------------------------------------------------------------------

_compiled_kernels = {}


def _get_kernel(name, source):
    """Compile and cache a RawKernel."""
    _require_cupy()
    if name not in _compiled_kernels:
        _compiled_kernels[name] = RawKernel(source, name)
    return _compiled_kernels[name]


# ---------------------------------------------------------------------------
# Helper: pack beam data into GPU arrays
# ---------------------------------------------------------------------------

def _pack_beams(beams):
    """
    Pack a list of Beam3D objects into flat NumPy arrays.

    Returns
    -------
    beam_k_hats : ndarray, (n_beams, 3)
    beam_freqs  : ndarray, (n_beams,)
    beam_q_labs : ndarray, (n_beams,), int32
    beam_s0s    : ndarray, (n_beams,)
    beam_radii  : ndarray, (n_beams,)
    """
    n = len(beams)
    k_hats = np.zeros((n, 3), dtype=np.float64)
    freqs = np.zeros(n, dtype=np.float64)
    q_labs = np.zeros(n, dtype=np.int32)
    s0s = np.zeros(n, dtype=np.float64)
    radii = np.zeros(n, dtype=np.float64)
    for i, b in enumerate(beams):
        k_hats[i] = b.k_hat
        freqs[i] = b.freq
        q_labs[i] = b.q_lab
        s0s[i] = b.s0
        radii[i] = b.beam_radius if b.beam_radius is not None else 0.0
    return k_hats, freqs, q_labs, s0s, radii


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rate_eq_force_3d_batch(mol_data, beams, r_arr, v_arr, B_gradient,
                           Gamma_eff_factor=1.0):
    """
    Batch force evaluation at multiple (r, v) points on the GPU.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data (CPU).
    beams : list of Beam3D
        Laser beams for the 3D MOT.
    r_arr : np.ndarray, shape (n_particles, 3)
        Positions (m).
    v_arr : np.ndarray, shape (n_particles, 3)
        Velocities (m/s).
    B_gradient : float
        Magnetic field gradient (T/m).
    Gamma_eff_factor : float
        Scattering rate reduction factor.

    Returns
    -------
    F_arr : np.ndarray, shape (n_particles, 3)
        Force vectors (N).
    R_scatter : np.ndarray, shape (n_particles,)
        Total scattering rates (rad/s).
    pop_arr : np.ndarray, shape (n_particles, n_ground)
        Steady-state ground-state populations.
    """
    _require_cupy()
    kernel = _get_kernel("rate_eq_force_3d_kernel", _KERNEL_SOURCE)

    r_arr = np.asarray(r_arr, dtype=np.float64).reshape(-1, 3)
    v_arr = np.asarray(v_arr, dtype=np.float64).reshape(-1, 3)
    n_particles = r_arr.shape[0]
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    N = n_g + n_e

    # Pack molecular data
    energies_d = cp.asarray(mol_data.energies, dtype=cp.float64)
    d_squared_d = cp.asarray(mol_data.d_squared, dtype=cp.float64)
    tdm_abs = np.abs(mol_data.tdm).astype(np.float64)
    tdm_abs_d = cp.asarray(tdm_abs)

    zeeman_x_re_d = cp.asarray(np.real(mol_data.zeeman_x).astype(np.float64))
    zeeman_x_im_d = cp.asarray(np.imag(mol_data.zeeman_x).astype(np.float64))
    zeeman_y_re_d = cp.asarray(np.real(mol_data.zeeman_y).astype(np.float64))
    zeeman_y_im_d = cp.asarray(np.imag(mol_data.zeeman_y).astype(np.float64))
    zeeman_z_re_d = cp.asarray(np.real(mol_data.zeeman_z).astype(np.float64))
    zeeman_z_im_d = cp.asarray(np.imag(mol_data.zeeman_z).astype(np.float64))

    # Pack beams
    bk, bf, bq, bs, br = _pack_beams(beams)
    bk_d = cp.asarray(bk)
    bf_d = cp.asarray(bf)
    bq_d = cp.asarray(bq)
    bs_d = cp.asarray(bs)
    br_d = cp.asarray(br)
    n_beams = len(beams)

    # Particle data
    r_d = cp.asarray(r_arr)
    v_d = cp.asarray(v_arr)

    # Output
    F_d = cp.zeros((n_particles, 3), dtype=cp.float64)
    R_d = cp.zeros(n_particles, dtype=cp.float64)
    pop_d = cp.zeros((n_particles, n_g), dtype=cp.float64)

    # Launch
    block = 128
    grid = (n_particles + block - 1) // block

    kernel(
        (grid,), (block,),
        (energies_d, d_squared_d, tdm_abs_d,
         zeeman_x_re_d, zeeman_x_im_d,
         zeeman_y_re_d, zeeman_y_im_d,
         zeeman_z_re_d, zeeman_z_im_d,
         np.float64(mol_data.Gamma), np.float64(mol_data.k),
         np.float64(Gamma_eff_factor),
         np.int32(n_g), np.int32(n_e), np.int32(N),
         bk_d, bf_d, bq_d, bs_d, br_d, np.int32(n_beams),
         r_d, v_d, np.float64(B_gradient), np.int32(n_particles),
         F_d, R_d, pop_d)
    )

    return to_cpu(F_d), to_cpu(R_d), to_cpu(pop_d)


def compute_3d_force_map_cuda(mol_data, beams, axes='xz', n_points=50,
                              v=None, B_gradient=0.0, extent=5e-3,
                              Gamma_eff_factor=1.0):
    """
    Compute a 2D force map on the GPU.

    All n_points^2 grid points are evaluated in a single kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of Beam3D
    axes : str
        Which 2D plane: 'xz', 'xy', or 'yz'.
    n_points : int
        Grid points per axis.
    v : array-like, shape (3,), optional
        Velocity vector (default: zero).
    B_gradient : float
        Magnetic field gradient (T/m).
    extent : float
        Half-size of the grid (m).
    Gamma_eff_factor : float

    Returns
    -------
    coords1, coords2 : np.ndarray, shape (n_points,)
    Fx_map, Fy_map : np.ndarray, shape (n_points, n_points)
    R_map : np.ndarray, shape (n_points, n_points)
    """
    if v is None:
        v = np.zeros(3)
    v = np.asarray(v, dtype=np.float64)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    a1 = axis_map[axes[0]]
    a2 = axis_map[axes[1]]

    coords1 = np.linspace(-extent, extent, n_points)
    coords2 = np.linspace(-extent, extent, n_points)

    # Build mesh of (r, v) pairs
    c1_grid, c2_grid = np.meshgrid(coords1, coords2, indexing='ij')
    n_total = n_points * n_points

    r_arr = np.zeros((n_total, 3), dtype=np.float64)
    r_arr[:, a1] = c1_grid.ravel()
    r_arr[:, a2] = c2_grid.ravel()

    v_arr = np.tile(v, (n_total, 1))

    F_arr, R_arr, _ = rate_eq_force_3d_batch(
        mol_data, beams, r_arr, v_arr, B_gradient, Gamma_eff_factor)

    Fx_map = F_arr[:, a1].reshape(n_points, n_points)
    Fy_map = F_arr[:, a2].reshape(n_points, n_points)
    R_map = R_arr.reshape(n_points, n_points)

    return coords1, coords2, Fx_map, Fy_map, R_map


def simulate_trajectory_3d_cuda(mol_data, beams, r0, v0, t_max, dt=1e-6,
                                B_gradient=0.0, Gamma_eff_factor=1.0,
                                r_escape=0.015):
    """
    Simulate a single 3D trajectory on the GPU using leapfrog integration.

    While a single trajectory does not benefit much from the GPU, the
    force evaluation at each step still uses the CUDA kernel.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of Beam3D
    r0, v0 : array-like, shape (3,)
    t_max, dt : float
    B_gradient : float
    Gamma_eff_factor : float
    r_escape : float

    Returns
    -------
    t_arr : np.ndarray, shape (n_steps,)
    r_arr : np.ndarray, shape (n_steps, 3)
    v_arr : np.ndarray, shape (n_steps, 3)
    """
    n_steps = int(t_max / dt)
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    mass = mol_data.mass

    t_arr = np.arange(n_steps, dtype=np.float64) * dt
    r_out = np.zeros((n_steps, 3), dtype=np.float64)
    v_out = np.zeros((n_steps, 3), dtype=np.float64)

    r_out[0] = r0
    v_out[0] = v0

    # Initial force
    F0, _, _ = rate_eq_force_3d_batch(
        mol_data, beams,
        r0.reshape(1, 3), v0.reshape(1, 3),
        B_gradient, Gamma_eff_factor)
    F0 = F0[0]

    v_half = v0 + 0.5 * dt * F0 / mass

    for j in range(1, n_steps):
        # Position step
        r_new = r_out[j - 1] + dt * v_half

        if np.linalg.norm(r_new) > r_escape:
            r_out[j:] = r_new
            v_out[j:] = v_half
            break

        r_out[j] = r_new

        # Force at new position
        F, _, _ = rate_eq_force_3d_batch(
            mol_data, beams,
            r_new.reshape(1, 3), v_half.reshape(1, 3),
            B_gradient, Gamma_eff_factor)
        F = F[0]

        # Velocity step
        v_half_new = v_half + dt * F / mass
        v_out[j] = 0.5 * (v_half + v_half_new)
        v_half = v_half_new

    return t_arr, r_out, v_out


def simulate_ensemble_3d_cuda(mol_data, beams, r0_list, v0_list,
                              t_max, dt=1e-6, B_gradient=0.0,
                              Gamma_eff_factor=1.0, r_escape=0.015):
    """
    Simulate an ensemble of 3D trajectories using GPU-batched force evaluation.

    All particles share the same time grid.  At each time step the force
    for ALL particles is evaluated in a single GPU kernel launch.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of Beam3D
    r0_list : list/array of shape (n_particles, 3)
    v0_list : list/array of shape (n_particles, 3)
    t_max, dt : float
    B_gradient : float
    Gamma_eff_factor : float
    r_escape : float

    Returns
    -------
    t_arr : np.ndarray, shape (n_steps,)
    r_all : np.ndarray, shape (n_particles, n_steps, 3)
    v_all : np.ndarray, shape (n_particles, n_steps, 3)
    """
    r0_arr = np.asarray(r0_list, dtype=np.float64).reshape(-1, 3)
    v0_arr = np.asarray(v0_list, dtype=np.float64).reshape(-1, 3)
    n_particles = r0_arr.shape[0]
    n_steps = int(t_max / dt)
    mass = mol_data.mass

    t_arr = np.arange(n_steps, dtype=np.float64) * dt
    r_all = np.zeros((n_particles, n_steps, 3), dtype=np.float64)
    v_all = np.zeros((n_particles, n_steps, 3), dtype=np.float64)

    r_all[:, 0, :] = r0_arr
    v_all[:, 0, :] = v0_arr

    # Current positions and half-step velocities
    r_cur = r0_arr.copy()
    v_cur = v0_arr.copy()

    # Initial force for all particles (single kernel launch)
    F_batch, _, _ = rate_eq_force_3d_batch(
        mol_data, beams, r_cur, v_cur, B_gradient, Gamma_eff_factor)

    v_half = v_cur + 0.5 * dt * F_batch / mass

    # Track which particles are still active
    active = np.ones(n_particles, dtype=bool)

    for j in range(1, n_steps):
        # Position step (all particles)
        r_cur[active] += dt * v_half[active]

        # Check escape
        r_norms = np.linalg.norm(r_cur, axis=1)
        escaped = active & (r_norms > r_escape)
        if np.any(escaped):
            for idx in np.where(escaped)[0]:
                r_all[idx, j:] = r_cur[idx]
                v_all[idx, j:] = v_half[idx]
            active[escaped] = False

        if not np.any(active):
            break

        r_all[active, j] = r_cur[active]

        # Batch force evaluation for active particles
        active_idx = np.where(active)[0]
        F_active, _, _ = rate_eq_force_3d_batch(
            mol_data, beams,
            r_cur[active_idx], v_half[active_idx],
            B_gradient, Gamma_eff_factor)

        # Velocity step
        for ii, idx in enumerate(active_idx):
            v_half_new = v_half[idx] + dt * F_active[ii] / mass
            v_all[idx, j] = 0.5 * (v_half[idx] + v_half_new)
            v_half[idx] = v_half_new

    return t_arr, r_all, v_all
