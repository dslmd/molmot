"""
CUDA-accelerated Monte Carlo Wavefunction / SSE solver for molecular MOT simulation.

Runs N_particles independent quantum trajectories IN PARALLEL on the GPU.
Each CUDA thread-block executes a single trajectory (full RK4 time integration
loop including quantum jumps), and many thread-blocks run concurrently.

The GPU kernel implements:
  1. RK4 time stepping with non-Hermitian H_eff
  2. 6-beam electric field with Gaussian profiles, polarization rotation, Doppler shifts
  3. Zeeman Hamiltonian (diagonal + off-diagonal ground-state coupling)
  4. Radiation pressure force from dipole expectation value
  5. Quantum jump detection (|psi|^2 decrease) and execution
  6. cuRAND per-thread RNG for quantum jumps and spontaneous emission kicks

Uses double-precision complex arithmetic throughout (cuDoubleComplex).

Requirements: CuPy (``pip install cupy-cuda12x`` or appropriate wheel).

Usage
-----
>>> from molmot.obe.stochastic import SSEProblem
>>> from molmot_cuda.stochastic_cuda import SSESolverCUDA, run_ensemble_cuda
>>>
>>> problem = SSEProblem(mol_data, beam_pairs, B_gradient, ...)
>>> solver = SSESolverCUDA(problem)
>>> result = solver.run(r0, v0, t_max, dt)
>>>
>>> # Batch mode (the whole point -- runs all trajectories in parallel on GPU):
>>> results = run_ensemble_cuda(problem, 1024, r0_sampler, v0_sampler, t_max, dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
    from cupy.cuda import runtime as cuda_runtime
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Import CPU SSEProblem (we reuse its problem setup and conversion logic)
import sys, os
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from molmot.obe.stochastic import SSEProblem, SSEResult, rotate_polarization


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_X_HAT = np.array([1.0, 0.0, 0.0])
_Y_HAT = np.array([0.0, 1.0, 0.0])
_Z_HAT = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# CUDA kernel source (complete, self-contained CUDA C)
# ---------------------------------------------------------------------------

# Parameterized at compile time via #define macros:
#   N_STATES  -- total number of quantum states (e.g. 16)
#   N_GROUND  -- number of ground states (e.g. 12)
#   N_EXCITED -- number of excited states (e.g. 4)
#   N_FREQS   -- number of laser frequency components
#   MAX_SAVE  -- maximum number of save points per trajectory

_CUDA_KERNEL_SOURCE = r"""
#include <curand_kernel.h>

// ---------- compile-time dimensions (injected by Python) ----------
// N_STATES, N_GROUND, N_EXCITED, N_FREQS, MAX_SAVE are #defined externally

// ---------- helper: cuDoubleComplex arithmetic ----------

__device__ __forceinline__
cuDoubleComplex make_cdouble(double re, double im) {
    cuDoubleComplex c;
    c.x = re; c.y = im;
    return c;
}

__device__ __forceinline__
cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cdouble(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__
cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cdouble(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__
cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cdouble(a.x * b.x - a.y * b.y,
                        a.x * b.y + a.y * b.x);
}

__device__ __forceinline__
cuDoubleComplex cconj(cuDoubleComplex a) {
    return make_cdouble(a.x, -a.y);
}

__device__ __forceinline__
cuDoubleComplex cscale(double s, cuDoubleComplex a) {
    return make_cdouble(s * a.x, s * a.y);
}

__device__ __forceinline__
double cnorm2(cuDoubleComplex a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__
cuDoubleComplex cexp_imag(double theta) {
    double s, c;
    sincos(theta, &s, &c);
    return make_cdouble(c, s);
}

// multiply by -i:  -i * (a + ib) = b - ia
__device__ __forceinline__
cuDoubleComplex cmul_neg_i(cuDoubleComplex a) {
    return make_cdouble(a.y, -a.x);
}

// ---------- 6-beam field computation ----------

__device__
void compute_fields_6beam(
    const double* __restrict__ r,           // position (3)
    double t,
    const double* __restrict__ omega_lasers,  // (N_FREQS)
    const double* __restrict__ sats,          // (N_FREQS)
    const cuDoubleComplex* __restrict__ eps,  // (6, N_FREQS, 3) row-major
    double beam_radius_k,
    cuDoubleComplex* __restrict__ E_kq       // output: (6, 3)
) {
    double denom = beam_radius_k * beam_radius_k * 0.5;

    // Gaussian beam profiles (perpendicular distance squared)
    double rp2_x = r[1]*r[1] + r[2]*r[2];
    double rp2_y = r[0]*r[0] + r[2]*r[2];
    double rp2_z = r[0]*r[0] + r[1]*r[1];

    double gauss_x = sqrt(exp(-rp2_x / denom));
    double gauss_y = sqrt(exp(-rp2_y / denom));
    double gauss_z = sqrt(exp(-rp2_z / denom));

    double gauss3[3] = {gauss_x, gauss_y, gauss_z};
    double kr[3] = {r[0], r[1], r[2]};

    // Zero output
    for (int i = 0; i < 18; i++) {
        E_kq[i] = make_cdouble(0.0, 0.0);
    }

    for (int f = 0; f < N_FREQS; f++) {
        double G = sqrt(sats[f]) / (2.0 * sqrt(2.0));
        double wt = omega_lasers[f] * t;

        for (int axis = 0; axis < 3; axis++) {
            double phase_fwd = -kr[axis] + wt;
            double phase_bwd =  kr[axis] + wt;

            cuDoubleComplex a_fwd = cscale(G * gauss3[axis], cexp_imag(phase_fwd));
            cuDoubleComplex a_bwd = cscale(G * gauss3[axis], cexp_imag(phase_bwd));

            for (int q = 0; q < 3; q++) {
                // eps layout: eps[beam*N_FREQS*3 + f*3 + q]
                cuDoubleComplex e_fwd = cconj(eps[axis * N_FREQS * 3 + f * 3 + q]);
                cuDoubleComplex e_bwd = cconj(eps[(axis + 3) * N_FREQS * 3 + f * 3 + q]);

                // E_kq[beam, q] += a * conj(eps)
                int idx_fwd = axis * 3 + q;
                int idx_bwd = (axis + 3) * 3 + q;
                E_kq[idx_fwd] = cadd(E_kq[idx_fwd], cmul(a_fwd, e_fwd));
                E_kq[idx_bwd] = cadd(E_kq[idx_bwd], cmul(a_bwd, e_bwd));
            }
        }
    }
}

// ---------- compute Zeeman-shifted energies ----------

__device__
void compute_omega_zeeman(
    const double* __restrict__ r,              // position (3) in natural units
    double r_nat,                              // 1/k in metres
    double B_gradient,                         // T/m
    const double* __restrict__ omega0s_base,   // (N_STATES) base energies
    const double* __restrict__ zeeman_x_g_re,  // (N_GROUND*N_GROUND) real parts
    const double* __restrict__ zeeman_x_g_im,  // (N_GROUND*N_GROUND) imag parts
    const double* __restrict__ zeeman_y_g_re,
    const double* __restrict__ zeeman_y_g_im,
    const double* __restrict__ zeeman_z_g_re,
    const double* __restrict__ zeeman_z_g_im,
    const double* __restrict__ zeeman_z_diag_excited,  // (N_EXCITED) diagonal Zeeman for excited
    double* __restrict__ omega0s,              // output (N_STATES)
    double* __restrict__ B_out                 // output (3) B field in Gauss
) {
    // Position in metres
    double rx_SI = r[0] * r_nat;
    double ry_SI = r[1] * r_nat;
    double rz_SI = r[2] * r_nat;

    // Anti-Helmholtz quadrupole B-field in Gauss
    double Bx_G = -B_gradient * rx_SI * 0.5 * 1e4;
    double By_G = -B_gradient * ry_SI * 0.5 * 1e4;
    double Bz_G =  B_gradient * rz_SI * 1e4;

    B_out[0] = Bx_G;
    B_out[1] = By_G;
    B_out[2] = Bz_G;

    // Ground states: diagonal of H_Z = Bx*Zx + By*Zy + Bz*Zz
    for (int i = 0; i < N_GROUND; i++) {
        int idx = i * N_GROUND + i;  // diagonal element [i,i]
        double shift = Bx_G * zeeman_x_g_re[idx]
                     + By_G * zeeman_y_g_re[idx]
                     + Bz_G * zeeman_z_g_re[idx];
        omega0s[i] = omega0s_base[i] + shift;
    }

    // Excited states: diagonal Zeeman from Bz only (g_J ~ 0 for Pi_1/2)
    for (int ie = 0; ie < N_EXCITED; ie++) {
        omega0s[N_GROUND + ie] = omega0s_base[N_GROUND + ie]
                                + Bz_G * zeeman_z_diag_excited[ie];
    }
}

// ---------- RHS of the SSE ODE ----------

__device__
void sse_rhs(
    const double* __restrict__ psi_re,    // (N_STATES)
    const double* __restrict__ psi_im,    // (N_STATES)
    const double* __restrict__ r,         // (3)
    const double* __restrict__ v,         // (3)
    double t,
    const double* __restrict__ omega0s,   // (N_STATES) with Zeeman
    const double* __restrict__ d_ge,      // (N_GROUND, N_EXCITED, 3) -- real
    const double* __restrict__ d_eg,      // (N_EXCITED, N_GROUND, 3) -- real
    const cuDoubleComplex* __restrict__ E_kq,  // (6, 3)
    double mass_nat,
    // Zeeman off-diagonal
    double Bx_G, double By_G, double Bz_G,
    const double* __restrict__ zx_re,    // (N_GROUND*N_GROUND)
    const double* __restrict__ zx_im,
    const double* __restrict__ zy_re,
    const double* __restrict__ zy_im,
    const double* __restrict__ zz_re,
    const double* __restrict__ zz_im,
    // outputs
    double* __restrict__ dpsi_re,        // (N_STATES)
    double* __restrict__ dpsi_im,        // (N_STATES)
    double* __restrict__ F_out           // (3) force
) {
    const int n_g = N_GROUND;
    const int n_e = N_EXCITED;
    const int N = N_STATES;

    // Reconstruct complex psi (DO NOT normalize -- MCWF requires unnormalized)
    cuDoubleComplex psi[N_STATES];
    for (int i = 0; i < N; i++)
        psi[i] = make_cdouble(psi_re[i], psi_im[i]);

    // Interaction picture: psi_int[i] = psi[i] * exp(-i * omega0s[i] * t)
    cuDoubleComplex psi_int[N_STATES];
    cuDoubleComplex eiw0t[N_STATES];
    for (int i = 0; i < N; i++) {
        eiw0t[i] = cexp_imag(-omega0s[i] * t);
        psi_int[i] = cmul(psi[i], eiw0t[i]);
    }

    // psi_q_g[ig, q] = sum_e d_ge[ig, ie, q] * psi_e_int[ie]
    cuDoubleComplex psi_q_g[N_GROUND * 3];
    for (int ig = 0; ig < n_g; ig++) {
        for (int q = 0; q < 3; q++) {
            cuDoubleComplex val = make_cdouble(0.0, 0.0);
            for (int ie = 0; ie < n_e; ie++) {
                double d = d_ge[ig * n_e * 3 + ie * 3 + q];
                val = cadd(val, cscale(d, psi_int[n_g + ie]));
            }
            psi_q_g[ig * 3 + q] = val;
        }
    }

    // psi_q_e[ie, q] = sum_g d_eg[ie, ig, q] * psi_g_int[ig]
    cuDoubleComplex psi_q_e[N_EXCITED * 3];
    for (int ie = 0; ie < n_e; ie++) {
        for (int q = 0; q < 3; q++) {
            cuDoubleComplex val = make_cdouble(0.0, 0.0);
            for (int ig = 0; ig < n_g; ig++) {
                double d = d_eg[ie * n_g * 3 + ig * 3 + q];
                val = cadd(val, cscale(d, psi_int[ig]));
            }
            psi_q_e[ie * 3 + q] = val;
        }
    }

    // Total field: E_total[q] = sum over 6 beams
    cuDoubleComplex E_total[3];
    for (int q = 0; q < 3; q++) {
        E_total[q] = make_cdouble(0.0, 0.0);
        for (int b = 0; b < 6; b++)
            E_total[q] = cadd(E_total[q], E_kq[b * 3 + q]);
    }

    // Dipole expectation: d_exp[q] = sum_g conj(psi_g_int[g]) * psi_q_g[g, q]
    cuDoubleComplex d_exp[3];
    for (int q = 0; q < 3; q++) {
        d_exp[q] = make_cdouble(0.0, 0.0);
        for (int ig = 0; ig < n_g; ig++)
            d_exp[q] = cadd(d_exp[q], cmul(cconj(psi_int[ig]),
                                            psi_q_g[ig * 3 + q]));
    }

    // Force: F[k] = -2 * Re(-i * sum_q d_exp[q] * (E_kq[k,q] - E_kq[k+3,q]))
    for (int k = 0; k < 3; k++) {
        double Fk = 0.0;
        for (int q = 0; q < 3; q++) {
            double Ediff_re = E_kq[k * 3 + q].x - E_kq[(k+3) * 3 + q].x;
            double Ediff_im = E_kq[k * 3 + q].y - E_kq[(k+3) * 3 + q].y;
            // Multiply by -i:  (-i)(a+ib) = b - ia
            double Ek_re =  Ediff_im;
            double Ek_im = -Ediff_re;
            double dr = d_exp[q].x;
            double di = d_exp[q].y;
            Fk -= 2.0 * (dr * Ek_re - di * Ek_im);
        }
        F_out[k] = Fk;
    }

    // dpsi/dt from laser coupling (interaction picture)
    cuDoubleComplex dpsi_int[N_STATES];
    for (int i = 0; i < N; i++)
        dpsi_int[i] = make_cdouble(0.0, 0.0);

    // Ground states: dpsi_g = -i * sum_q E_total[q] * psi_q_g[g, q]
    for (int ig = 0; ig < n_g; ig++) {
        cuDoubleComplex val = make_cdouble(0.0, 0.0);
        for (int q = 0; q < 3; q++)
            val = cadd(val, cmul(E_total[q], psi_q_g[ig * 3 + q]));
        dpsi_int[ig] = cmul_neg_i(val);
    }

    // Excited states: dpsi_e = -i * sum_q conj(E_total[q]) * psi_q_e[e, q]
    for (int ie = 0; ie < n_e; ie++) {
        cuDoubleComplex val = make_cdouble(0.0, 0.0);
        for (int q = 0; q < 3; q++)
            val = cadd(val, cmul(cconj(E_total[q]), psi_q_e[ie * 3 + q]));
        dpsi_int[n_g + ie] = cmul_neg_i(val);
    }

    // Non-Hermitian decay: -Gamma/2 * psi_e (Gamma=1 in natural units)
    for (int ie = 0; ie < n_e; ie++) {
        dpsi_int[n_g + ie] = csub(dpsi_int[n_g + ie],
                                   cscale(0.5, psi_int[n_g + ie]));
    }

    // Off-diagonal Zeeman coupling (ground states only)
    // H_Z = Bx*Zx + By*Zy + Bz*Zz
    // In interaction picture: H_Z_int[i,j] = H_Z[i,j] * exp(-i*(wi-wj)*t)
    // dpsi_Z = -i * H_Z_int @ psi_g_int (off-diagonal only)
    for (int ig = 0; ig < n_g; ig++) {
        cuDoubleComplex val = make_cdouble(0.0, 0.0);
        cuDoubleComplex phase_i = cexp_imag(-omega0s[ig] * t);
        for (int jg = 0; jg < n_g; jg++) {
            if (ig == jg) continue;
            int idx = ig * n_g + jg;
            // H_Z[ig,jg] = Bx*Zx[ig,jg] + By*Zy[ig,jg] + Bz*Zz[ig,jg]
            double hz_re = Bx_G * zx_re[idx] + By_G * zy_re[idx] + Bz_G * zz_re[idx];
            double hz_im = Bx_G * zx_im[idx] + By_G * zy_im[idx] + Bz_G * zz_im[idx];
            cuDoubleComplex Hz_ij = make_cdouble(hz_re, hz_im);

            // Phase: exp(-i*wi*t) * exp(+i*wj*t) = phase_i * conj(phase_j)
            cuDoubleComplex phase_j = cexp_imag(-omega0s[jg] * t);
            cuDoubleComplex phase_ij = cmul(phase_i, cconj(phase_j));
            cuDoubleComplex H_int_ij = cmul(Hz_ij, phase_ij);
            val = cadd(val, cmul(H_int_ij, psi_int[jg]));
        }
        // -i * val
        dpsi_int[ig] = cadd(dpsi_int[ig], cmul_neg_i(val));
    }

    // Transform back from interaction picture: dpsi = dpsi_int * exp(+i*omega0*t)
    for (int i = 0; i < N; i++) {
        cuDoubleComplex eiw0t_conj = cexp_imag(omega0s[i] * t);
        cuDoubleComplex dp = cmul(dpsi_int[i], eiw0t_conj);
        dpsi_re[i] = dp.x;
        dpsi_im[i] = dp.y;
    }
}


// ==========================================================================
// Main kernel: one trajectory per thread-block, 1 thread per block
// ==========================================================================

extern "C" __global__
void sse_trajectory_kernel(
    // --- per-trajectory arrays (indexed by traj_id) ---
    const double* __restrict__ r0_all,        // (n_traj, 3) initial positions (natural)
    const double* __restrict__ v0_all,        // (n_traj, 3) initial velocities (natural)
    const double* __restrict__ psi0_re_all,   // (n_traj, N_STATES) initial psi real
    const double* __restrict__ psi0_im_all,   // (n_traj, N_STATES) initial psi imag
    // --- problem parameters (shared across all trajectories) ---
    const double* __restrict__ omega0s_base,  // (N_STATES)
    const double* __restrict__ d_ge,          // (N_GROUND*N_EXCITED*3)
    const double* __restrict__ d_eg,          // (N_EXCITED*N_GROUND*3)
    const double* __restrict__ branching,     // (N_EXCITED*N_GROUND*3)
    const double* __restrict__ omega_lasers,  // (N_FREQS)
    const double* __restrict__ sats,          // (N_FREQS)
    const cuDoubleComplex* __restrict__ eps,  // (6*N_FREQS*3) rotated polarizations
    double beam_radius_k,
    double mass_nat,
    double v_recoil_nat,
    double r_nat_scale,                       // 1/k in metres
    double B_gradient,                        // T/m
    int add_spontaneous_kick,                 // bool
    const double* __restrict__ diffusion_nat, // (3)
    // Zeeman matrices (ground block, real+imag split)
    const double* __restrict__ zx_re,   // (N_GROUND*N_GROUND)
    const double* __restrict__ zx_im,
    const double* __restrict__ zy_re,
    const double* __restrict__ zy_im,
    const double* __restrict__ zz_re,
    const double* __restrict__ zz_im,
    const double* __restrict__ zeeman_z_diag_excited, // (N_EXCITED)
    // --- time stepping ---
    double dt_nat,
    int n_steps,
    int save_every,
    int n_save,
    double r_escape_nat,              // escape radius in natural units (0 = disabled)
    int renorm_interval,              // re-normalize psi every N steps (0 = disabled)
    // --- RNG seeds ---
    unsigned long long base_seed,
    // --- output arrays (indexed by traj_id) ---
    double* __restrict__ times_out,       // (n_traj, n_save)
    double* __restrict__ positions_out,   // (n_traj, n_save, 3)
    double* __restrict__ velocities_out,  // (n_traj, n_save, 3)
    double* __restrict__ populations_out, // (n_traj, n_save, N_STATES)
    int* __restrict__ n_scatters_out,     // (n_traj,)
    int* __restrict__ escaped_out,        // (n_traj,)  1 if escaped
    int* __restrict__ n_saved_out         // (n_traj,) actual save count
) {
    int tid = blockIdx.x;  // one trajectory per block

    // ---- per-trajectory RNG state ----
    curandStatePhilox4_32_10_t rng_state;
    curand_init(base_seed, (unsigned long long)tid, 0, &rng_state);

    // ---- local state arrays ----
    double psi_re[N_STATES], psi_im[N_STATES];
    double r[3], v_vel[3];

    // Load initial conditions
    for (int i = 0; i < N_STATES; i++) {
        psi_re[i] = psi0_re_all[tid * N_STATES + i];
        psi_im[i] = psi0_im_all[tid * N_STATES + i];
    }
    for (int i = 0; i < 3; i++) {
        r[i] = r0_all[tid * 3 + i];
        v_vel[i] = v0_all[tid * 3 + i];
    }

    // Quantum jump threshold: eta = -ln(r), r ~ U(0,1)
    double u_rand = curand_uniform_double(&rng_state);
    if (u_rand < 1e-15) u_rand = 1e-15;
    double threshold = -log(u_rand);

    int n_scatters = 0;
    int escaped = 0;
    int save_idx = 0;
    double t_current = 0.0;
    double last_decay_time = 0.0;

    // ---- RK4 workspace ----
    double k1_psi_re[N_STATES], k1_psi_im[N_STATES], k1_r[3], k1_v[3];
    double k2_psi_re[N_STATES], k2_psi_im[N_STATES], k2_r[3], k2_v[3];
    double k3_psi_re[N_STATES], k3_psi_im[N_STATES], k3_r[3], k3_v[3];
    double k4_psi_re[N_STATES], k4_psi_im[N_STATES], k4_r[3], k4_v[3];
    double tmp_psi_re[N_STATES], tmp_psi_im[N_STATES], tmp_r[3], tmp_v[3];

    double omega0s[N_STATES];
    double B_field[3];
    cuDoubleComplex E_kq[18];  // 6 beams x 3 polarization components
    double dpsi_re[N_STATES], dpsi_im[N_STATES], F[3];

    // ---- main time loop ----
    for (int step = 0; step <= n_steps; step++) {
        // ---- save state ----
        if (step % save_every == 0 && save_idx < n_save) {
            double norm_sq = 0.0;
            for (int i = 0; i < N_STATES; i++)
                norm_sq += psi_re[i]*psi_re[i] + psi_im[i]*psi_im[i];

            double inv_norm = (norm_sq > 1e-30) ? (1.0 / norm_sq) : 1.0;

            int t_off = tid * n_save + save_idx;
            times_out[t_off] = t_current;

            for (int i = 0; i < 3; i++) {
                positions_out[tid * n_save * 3 + save_idx * 3 + i] = r[i];
                velocities_out[tid * n_save * 3 + save_idx * 3 + i] = v_vel[i];
            }
            for (int i = 0; i < N_STATES; i++) {
                double pop = (psi_re[i]*psi_re[i] + psi_im[i]*psi_im[i]) * inv_norm;
                populations_out[tid * n_save * N_STATES + save_idx * N_STATES + i] = pop;
            }
            save_idx++;
        }

        if (step == n_steps) break;  // final save done, exit

        // ---- check escape ----
        if (r_escape_nat > 0.0) {
            double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
            if (r2 > r_escape_nat * r_escape_nat) {
                escaped = 1;
                break;
            }
        }

        // ==== RK4 step ====
        // Stage 1: k1 = f(t, y)
        compute_omega_zeeman(r, r_nat_scale, B_gradient, omega0s_base,
                             zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                             zeeman_z_diag_excited, omega0s, B_field);
        compute_fields_6beam(r, t_current, omega_lasers, sats, eps,
                             beam_radius_k, E_kq);
        sse_rhs(psi_re, psi_im, r, v_vel, t_current,
                omega0s, d_ge, d_eg, E_kq, mass_nat,
                B_field[0], B_field[1], B_field[2],
                zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                dpsi_re, dpsi_im, F);
        for (int i = 0; i < N_STATES; i++) { k1_psi_re[i] = dpsi_re[i]; k1_psi_im[i] = dpsi_im[i]; }
        for (int i = 0; i < 3; i++) { k1_r[i] = v_vel[i]; k1_v[i] = F[i] / mass_nat; }

        // Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        for (int i = 0; i < N_STATES; i++) {
            tmp_psi_re[i] = psi_re[i] + 0.5 * dt_nat * k1_psi_re[i];
            tmp_psi_im[i] = psi_im[i] + 0.5 * dt_nat * k1_psi_im[i];
        }
        for (int i = 0; i < 3; i++) {
            tmp_r[i] = r[i] + 0.5 * dt_nat * k1_r[i];
            tmp_v[i] = v_vel[i] + 0.5 * dt_nat * k1_v[i];
        }
        compute_omega_zeeman(tmp_r, r_nat_scale, B_gradient, omega0s_base,
                             zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                             zeeman_z_diag_excited, omega0s, B_field);
        compute_fields_6beam(tmp_r, t_current + 0.5*dt_nat, omega_lasers, sats, eps,
                             beam_radius_k, E_kq);
        sse_rhs(tmp_psi_re, tmp_psi_im, tmp_r, tmp_v, t_current + 0.5*dt_nat,
                omega0s, d_ge, d_eg, E_kq, mass_nat,
                B_field[0], B_field[1], B_field[2],
                zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                dpsi_re, dpsi_im, F);
        for (int i = 0; i < N_STATES; i++) { k2_psi_re[i] = dpsi_re[i]; k2_psi_im[i] = dpsi_im[i]; }
        for (int i = 0; i < 3; i++) { k2_r[i] = tmp_v[i]; k2_v[i] = F[i] / mass_nat; }

        // Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        for (int i = 0; i < N_STATES; i++) {
            tmp_psi_re[i] = psi_re[i] + 0.5 * dt_nat * k2_psi_re[i];
            tmp_psi_im[i] = psi_im[i] + 0.5 * dt_nat * k2_psi_im[i];
        }
        for (int i = 0; i < 3; i++) {
            tmp_r[i] = r[i] + 0.5 * dt_nat * k2_r[i];
            tmp_v[i] = v_vel[i] + 0.5 * dt_nat * k2_v[i];
        }
        compute_omega_zeeman(tmp_r, r_nat_scale, B_gradient, omega0s_base,
                             zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                             zeeman_z_diag_excited, omega0s, B_field);
        compute_fields_6beam(tmp_r, t_current + 0.5*dt_nat, omega_lasers, sats, eps,
                             beam_radius_k, E_kq);
        sse_rhs(tmp_psi_re, tmp_psi_im, tmp_r, tmp_v, t_current + 0.5*dt_nat,
                omega0s, d_ge, d_eg, E_kq, mass_nat,
                B_field[0], B_field[1], B_field[2],
                zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                dpsi_re, dpsi_im, F);
        for (int i = 0; i < N_STATES; i++) { k3_psi_re[i] = dpsi_re[i]; k3_psi_im[i] = dpsi_im[i]; }
        for (int i = 0; i < 3; i++) { k3_r[i] = tmp_v[i]; k3_v[i] = F[i] / mass_nat; }

        // Stage 4: k4 = f(t + dt, y + dt * k3)
        for (int i = 0; i < N_STATES; i++) {
            tmp_psi_re[i] = psi_re[i] + dt_nat * k3_psi_re[i];
            tmp_psi_im[i] = psi_im[i] + dt_nat * k3_psi_im[i];
        }
        for (int i = 0; i < 3; i++) {
            tmp_r[i] = r[i] + dt_nat * k3_r[i];
            tmp_v[i] = v_vel[i] + dt_nat * k3_v[i];
        }
        compute_omega_zeeman(tmp_r, r_nat_scale, B_gradient, omega0s_base,
                             zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                             zeeman_z_diag_excited, omega0s, B_field);
        compute_fields_6beam(tmp_r, t_current + dt_nat, omega_lasers, sats, eps,
                             beam_radius_k, E_kq);
        sse_rhs(tmp_psi_re, tmp_psi_im, tmp_r, tmp_v, t_current + dt_nat,
                omega0s, d_ge, d_eg, E_kq, mass_nat,
                B_field[0], B_field[1], B_field[2],
                zx_re, zx_im, zy_re, zy_im, zz_re, zz_im,
                dpsi_re, dpsi_im, F);
        for (int i = 0; i < N_STATES; i++) { k4_psi_re[i] = dpsi_re[i]; k4_psi_im[i] = dpsi_im[i]; }
        for (int i = 0; i < 3; i++) { k4_r[i] = tmp_v[i]; k4_v[i] = F[i] / mass_nat; }

        // RK4 update: y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        double dt6 = dt_nat / 6.0;
        for (int i = 0; i < N_STATES; i++) {
            psi_re[i] += dt6 * (k1_psi_re[i] + 2.0*k2_psi_re[i] + 2.0*k3_psi_re[i] + k4_psi_re[i]);
            psi_im[i] += dt6 * (k1_psi_im[i] + 2.0*k2_psi_im[i] + 2.0*k3_psi_im[i] + k4_psi_im[i]);
        }
        for (int i = 0; i < 3; i++) {
            r[i] += dt6 * (k1_r[i] + 2.0*k2_r[i] + 2.0*k3_r[i] + k4_r[i]);
            v_vel[i] += dt6 * (k1_v[i] + 2.0*k2_v[i] + 2.0*k3_v[i] + k4_v[i]);
        }

        t_current += dt_nat;

        // ==== Quantum jump check ====
        double norm_sq = 0.0;
        for (int i = 0; i < N_STATES; i++)
            norm_sq += psi_re[i]*psi_re[i] + psi_im[i]*psi_im[i];
        double dp = 1.0 - norm_sq;

        if (dp > threshold) {
            // ---- Select decay channel ----
            // Compute probabilities: prob[ie,ig,q] = branching[ie,ig,q] * |psi_e[ie]|^2
            double total_prob = 0.0;
            double cum_probs[N_EXCITED * N_GROUND * 3];
            int n_channels = 0;

            for (int ie = 0; ie < N_EXCITED; ie++) {
                double pe = psi_re[N_GROUND+ie]*psi_re[N_GROUND+ie]
                          + psi_im[N_GROUND+ie]*psi_im[N_GROUND+ie];
                if (pe < 1e-30) continue;
                for (int ig = 0; ig < N_GROUND; ig++) {
                    for (int q = 0; q < 3; q++) {
                        double br = branching[ie * N_GROUND * 3 + ig * 3 + q];
                        if (br < 1e-15) continue;
                        double prob = br * pe;
                        total_prob += prob;
                        cum_probs[n_channels] = total_prob;
                        // Encode channel index: pack (ie, ig, q)
                        // We reuse cum_probs for cumulative + store channel info
                        // in a parallel array below
                        n_channels++;
                    }
                }
            }

            if (n_channels > 0 && total_prob > 1e-30) {
                // Pick channel via cumulative probability
                double r_choice = curand_uniform_double(&rng_state) * total_prob;

                // Re-scan to find channel (need to reconstruct ie,ig,q)
                int selected_ig = 0;
                double cum = 0.0;
                int found = 0;
                for (int ie = 0; ie < N_EXCITED && !found; ie++) {
                    double pe = psi_re[N_GROUND+ie]*psi_re[N_GROUND+ie]
                              + psi_im[N_GROUND+ie]*psi_im[N_GROUND+ie];
                    if (pe < 1e-30) continue;
                    for (int ig = 0; ig < N_GROUND && !found; ig++) {
                        for (int q = 0; q < 3 && !found; q++) {
                            double br = branching[ie * N_GROUND * 3 + ig * 3 + q];
                            if (br < 1e-15) continue;
                            cum += br * pe;
                            if (r_choice <= cum) {
                                selected_ig = ig;
                                found = 1;
                            }
                        }
                    }
                }

                // Project onto selected ground state (normalized)
                for (int i = 0; i < N_STATES; i++) {
                    psi_re[i] = 0.0;
                    psi_im[i] = 0.0;
                }
                psi_re[selected_ig] = 1.0;

                n_scatters++;

                // Random photon recoil kick (isotropic emission)
                if (add_spontaneous_kick) {
                    double cos_theta = 2.0 * curand_uniform_double(&rng_state) - 1.0;
                    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                    double phi = 2.0 * M_PI * curand_uniform_double(&rng_state);
                    double kick_x = sin_theta * cos(phi);
                    double kick_y = sin_theta * sin(phi);
                    double kick_z = cos_theta;
                    v_vel[0] += v_recoil_nat * kick_x;
                    v_vel[1] += v_recoil_nat * kick_y;
                    v_vel[2] += v_recoil_nat * kick_z;
                }
            } else {
                // No excited population -- reset to first ground state
                for (int i = 0; i < N_STATES; i++) {
                    psi_re[i] = 0.0;
                    psi_im[i] = 0.0;
                }
                psi_re[0] = 1.0;
            }

            // Momentum diffusion kick
            double dt_since = (t_current - last_decay_time);
            for (int i = 0; i < 3; i++) {
                if (diffusion_nat[i] > 0.0) {
                    double kick_sigma = sqrt(2.0 * diffusion_nat[i] * dt_since);
                    double sign = (curand_uniform_double(&rng_state) > 0.5) ? 1.0 : -1.0;
                    v_vel[i] += sign * kick_sigma;
                }
            }

            last_decay_time = t_current;

            // Draw new threshold
            u_rand = curand_uniform_double(&rng_state);
            if (u_rand < 1e-15) u_rand = 1e-15;
            threshold = -log(u_rand);
        }

        // ==== Optional renormalization (prevent numerical drift) ====
        if (renorm_interval > 0 && (step + 1) % renorm_interval == 0) {
            // Only renormalize if norm has drifted above 1.0 (which is unphysical
            // for the non-Hermitian evolution -- it should only decrease).
            // Also renormalize if norm is very small (about to underflow).
            double ns = 0.0;
            for (int i = 0; i < N_STATES; i++)
                ns += psi_re[i]*psi_re[i] + psi_im[i]*psi_im[i];
            if (ns > 1.0 + 1e-10 || ns < 1e-15) {
                double inv_norm = 1.0 / sqrt(ns);
                for (int i = 0; i < N_STATES; i++) {
                    psi_re[i] *= inv_norm;
                    psi_im[i] *= inv_norm;
                }
                // Reset jump threshold since we modified the norm
                u_rand = curand_uniform_double(&rng_state);
                if (u_rand < 1e-15) u_rand = 1e-15;
                threshold = -log(u_rand);
            }
        }
    }

    // ---- write outputs ----
    n_scatters_out[tid] = n_scatters;
    escaped_out[tid] = escaped;
    n_saved_out[tid] = save_idx;
}
"""


# ---------------------------------------------------------------------------
# Python wrapper: SSESolverCUDA
# ---------------------------------------------------------------------------

def _compile_kernel(n_states, n_ground, n_excited, n_freqs, max_save):
    """Compile the CUDA kernel with the given dimensions."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for CUDA acceleration. "
                           "Install with: pip install cupy-cuda12x")

    defines = (
        f"#define N_STATES {n_states}\n"
        f"#define N_GROUND {n_ground}\n"
        f"#define N_EXCITED {n_excited}\n"
        f"#define N_FREQS {n_freqs}\n"
        f"#define MAX_SAVE {max_save}\n"
    )

    full_source = defines + _CUDA_KERNEL_SOURCE

    kernel = cp.RawKernel(
        full_source,
        'sse_trajectory_kernel',
        options=('--std=c++14', '-use_fast_math'),
        jitify=True,
    )
    return kernel


def _prepare_problem_buffers(problem: SSEProblem):
    """
    Convert SSEProblem data to flat GPU arrays suitable for the CUDA kernel.

    Returns a dict of CuPy arrays.
    """
    p = problem
    n_g = p.n_ground
    n_e = p.n_excited
    n_freqs = len(p.omega_lasers)

    # State energies (base, without Zeeman -- Zeeman is computed per-position on GPU)
    omega0s_base = cp.asarray(p.omega0s, dtype=cp.float64)

    # Transition dipole matrices (real, contiguous)
    d_ge = cp.asarray(p.d_ge.astype(np.float64).ravel(), dtype=cp.float64)
    d_eg = cp.asarray(p.d_eg.astype(np.float64).ravel(), dtype=cp.float64)

    # Branching ratios
    branching = cp.asarray(p.branching.astype(np.float64).ravel(), dtype=cp.float64)

    # Laser parameters
    omega_lasers = cp.asarray(p.omega_lasers, dtype=cp.float64)
    sats = cp.asarray(p.sats, dtype=cp.float64)

    # Pre-compute rotated polarizations (same as CPU)
    eps_np = np.zeros((6, n_freqs, 3), dtype=complex)
    dirs = [_X_HAT, _Y_HAT, _Z_HAT, -_X_HAT, -_Y_HAT, -_Z_HAT]
    for i, pol in enumerate(p.pols):
        for d_idx, d_vec in enumerate(dirs):
            eps_np[d_idx, i, :] = rotate_polarization(pol, d_vec)

    # Convert complex polarizations to cuDoubleComplex array (interleaved re/im)
    # CuPy complex128 maps directly to cuDoubleComplex
    eps = cp.asarray(eps_np.ravel(), dtype=cp.complex128)

    # Zeeman matrices (ground block) -- split into real and imag parts
    zx_re = cp.asarray(np.real(p.zeeman_x_ground).ravel(), dtype=cp.float64)
    zx_im = cp.asarray(np.imag(p.zeeman_x_ground).ravel(), dtype=cp.float64)
    zy_re = cp.asarray(np.real(p.zeeman_y_ground).ravel(), dtype=cp.float64)
    zy_im = cp.asarray(np.imag(p.zeeman_y_ground).ravel(), dtype=cp.float64)
    zz_re = cp.asarray(np.real(p.zeeman_z_ground).ravel(), dtype=cp.float64)
    zz_im = cp.asarray(np.imag(p.zeeman_z_ground).ravel(), dtype=cp.float64)

    # Diagonal Zeeman for excited states
    zeeman_z_diag_exc = cp.asarray(
        p.zeeman_z_diag[n_g:].astype(np.float64), dtype=cp.float64)

    # Diffusion constants
    diffusion_nat = cp.asarray(p.diffusion_nat, dtype=cp.float64)

    return {
        'omega0s_base': omega0s_base,
        'd_ge': d_ge,
        'd_eg': d_eg,
        'branching': branching,
        'omega_lasers': omega_lasers,
        'sats': sats,
        'eps': eps,
        'zx_re': zx_re, 'zx_im': zx_im,
        'zy_re': zy_re, 'zy_im': zy_im,
        'zz_re': zz_re, 'zz_im': zz_im,
        'zeeman_z_diag_excited': zeeman_z_diag_exc,
        'diffusion_nat': diffusion_nat,
        'beam_radius_k': float(p.beam_radius_k),
        'mass_nat': float(p.mass_nat),
        'v_recoil_nat': float(p.v_recoil_nat),
        'r_nat': float(p.r_nat),
        'B_gradient': float(p.B_gradient),
        'add_spontaneous_kick': int(p.add_spontaneous_kick),
    }


class SSESolverCUDA:
    """
    GPU-accelerated SSE solver using CuPy RawKernel.

    Same API as SSESolver but runs trajectories on the GPU.
    The primary advantage is ``run_ensemble`` which runs all trajectories
    in parallel (one trajectory per CUDA thread-block).

    Parameters
    ----------
    problem : SSEProblem
        The problem specification (same as CPU version).
    """

    def __init__(self, problem: SSEProblem):
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required. Install with: pip install cupy-cuda12x")
        self.prob = problem
        self._buffers = _prepare_problem_buffers(problem)
        self._kernel_cache = {}  # keyed by (n_save,)

    def _get_kernel(self, n_save):
        """Get (or compile) the CUDA kernel for the given save count."""
        p = self.prob
        key = (n_save,)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _compile_kernel(
                p.n_states, p.n_ground, p.n_excited,
                len(p.omega_lasers), n_save
            )
        return self._kernel_cache[key]

    def run(self, r0, v0, t_max, dt=None, psi0=None,
            save_every=100, rng_seed=None, r_escape=None,
            renorm_interval=0):
        """
        Run a single trajectory on the GPU.

        Parameters are identical to SSESolver.run(). For a single trajectory
        there is no speedup over CPU; use ``run_ensemble`` for batch mode.

        Parameters
        ----------
        r0 : array-like, shape (3,)
            Initial position in metres.
        v0 : array-like, shape (3,)
            Initial velocity in m/s.
        t_max : float
            Total simulation time in seconds.
        dt : float, optional
            Time step in seconds. Default: 0.01 / Gamma.
        psi0 : np.ndarray, optional
            Initial wavefunction.
        save_every : int
            Save interval in time steps.
        rng_seed : int, optional
            Random seed.
        r_escape : float, optional
            Escape radius in metres (trajectory stops if |r| > r_escape).
        renorm_interval : int
            Re-normalize psi every N steps to prevent numerical drift.
            0 disables (default).

        Returns
        -------
        SSEResult
        """
        results = self.run_ensemble(
            n_particles=1,
            r0_sampler=lambda: np.asarray(r0, dtype=float),
            v0_sampler=lambda: np.asarray(v0, dtype=float),
            t_max=t_max, dt=dt, psi0=psi0,
            save_every=save_every, rng_seed=rng_seed,
            r_escape=r_escape, renorm_interval=renorm_interval,
        )
        return results[0]

    def run_ensemble(self, n_particles, r0_sampler, v0_sampler,
                     t_max, dt=None, psi0=None, save_every=100,
                     rng_seed=None, r_escape=None,
                     renorm_interval=0, callback=None):
        """
        Run an ensemble of trajectories in parallel on the GPU.

        This is the primary method -- launches all trajectories simultaneously
        as independent CUDA thread-blocks.

        Parameters
        ----------
        n_particles : int
            Number of trajectories.
        r0_sampler : callable() -> ndarray (3,)
            Returns initial position in metres.
        v0_sampler : callable() -> ndarray (3,)
            Returns initial velocity in m/s.
        t_max : float
            Simulation time in seconds.
        dt : float, optional
            Time step in seconds.
        psi0 : ndarray, optional
            Initial wavefunction (shared for all trajectories).
        save_every : int
            Save interval.
        rng_seed : int, optional
            Base random seed.
        r_escape : float, optional
            Escape radius in metres.
        renorm_interval : int
            Re-normalize interval (0 = disabled).
        callback : callable(i, result), optional
            Called after GPU results are transferred for each trajectory.

        Returns
        -------
        list of SSEResult
        """
        p = self.prob
        buf = self._buffers
        N = p.n_states
        n_g = p.n_ground

        # Time step
        if dt is None:
            dt_SI = 0.01 / p.Gamma
        else:
            dt_SI = dt
        dt_nat = dt_SI / p.t_nat
        t_max_nat = t_max / p.t_nat
        n_steps = int(t_max_nat / dt_nat)
        n_save = n_steps // save_every + 1

        # Escape radius in natural units
        r_escape_nat = 0.0
        if r_escape is not None:
            r_escape_nat = r_escape / p.r_nat

        # Compile kernel
        kernel = self._get_kernel(n_save)

        # ----- Prepare initial conditions on host -----
        r0_all = np.zeros((n_particles, 3), dtype=np.float64)
        v0_all = np.zeros((n_particles, 3), dtype=np.float64)
        psi0_re_all = np.zeros((n_particles, N), dtype=np.float64)
        psi0_im_all = np.zeros((n_particles, N), dtype=np.float64)

        for i in range(n_particles):
            r0_SI = np.asarray(r0_sampler(), dtype=float)
            v0_SI = np.asarray(v0_sampler(), dtype=float)
            r0_all[i, :] = r0_SI / p.r_nat
            v0_all[i, :] = v0_SI / p.v_nat

            if psi0 is not None:
                psi_init = np.asarray(psi0, dtype=complex)
                psi_init = psi_init / np.linalg.norm(psi_init)
            else:
                psi_init = np.zeros(N, dtype=complex)
                psi_init[0] = 1.0

            psi0_re_all[i, :] = np.real(psi_init)
            psi0_im_all[i, :] = np.imag(psi_init)

        # Transfer to GPU
        d_r0 = cp.asarray(r0_all)
        d_v0 = cp.asarray(v0_all)
        d_psi0_re = cp.asarray(psi0_re_all)
        d_psi0_im = cp.asarray(psi0_im_all)

        # Allocate output arrays on GPU
        d_times = cp.zeros((n_particles, n_save), dtype=cp.float64)
        d_positions = cp.zeros((n_particles, n_save, 3), dtype=cp.float64)
        d_velocities = cp.zeros((n_particles, n_save, 3), dtype=cp.float64)
        d_populations = cp.zeros((n_particles, n_save, N), dtype=cp.float64)
        d_n_scatters = cp.zeros(n_particles, dtype=cp.int32)
        d_escaped = cp.zeros(n_particles, dtype=cp.int32)
        d_n_saved = cp.zeros(n_particles, dtype=cp.int32)

        # RNG seed
        if rng_seed is None:
            rng_seed = np.random.default_rng().integers(0, 2**62)
        base_seed = np.uint64(rng_seed)

        # ----- Launch kernel -----
        # One block per trajectory, 1 thread per block
        grid = (n_particles,)
        block = (1,)

        kernel(
            grid, block,
            (
                # Per-trajectory inputs
                d_r0, d_v0, d_psi0_re, d_psi0_im,
                # Problem parameters
                buf['omega0s_base'], buf['d_ge'], buf['d_eg'],
                buf['branching'],
                buf['omega_lasers'], buf['sats'], buf['eps'],
                np.float64(buf['beam_radius_k']),
                np.float64(buf['mass_nat']),
                np.float64(buf['v_recoil_nat']),
                np.float64(buf['r_nat']),
                np.float64(buf['B_gradient']),
                np.int32(buf['add_spontaneous_kick']),
                buf['diffusion_nat'],
                # Zeeman matrices
                buf['zx_re'], buf['zx_im'],
                buf['zy_re'], buf['zy_im'],
                buf['zz_re'], buf['zz_im'],
                buf['zeeman_z_diag_excited'],
                # Time stepping
                np.float64(dt_nat),
                np.int32(n_steps),
                np.int32(save_every),
                np.int32(n_save),
                np.float64(r_escape_nat),
                np.int32(renorm_interval),
                # RNG
                base_seed,
                # Outputs
                d_times, d_positions, d_velocities, d_populations,
                d_n_scatters, d_escaped, d_n_saved,
            ),
        )

        # Synchronize and transfer results back to CPU
        cp.cuda.Device().synchronize()

        h_times = d_times.get()
        h_positions = d_positions.get()
        h_velocities = d_velocities.get()
        h_populations = d_populations.get()
        h_n_scatters = d_n_scatters.get()
        h_escaped = d_escaped.get()
        h_n_saved = d_n_saved.get()

        # Build SSEResult objects
        results = []
        for i in range(n_particles):
            ns = int(h_n_saved[i])
            if ns == 0:
                ns = 1  # at least one save point

            result = SSEResult(
                times=h_times[i, :ns] * p.t_nat,           # -> seconds
                positions=h_positions[i, :ns, :] * p.r_nat, # -> metres
                velocities=h_velocities[i, :ns, :] * p.v_nat, # -> m/s
                populations=h_populations[i, :ns, :],
                photons_scattered=int(h_n_scatters[i]),
                jump_times=[],      # not tracked on GPU (too expensive)
                jump_channels=[],   # not tracked on GPU
                forces=None,
            )
            results.append(result)

            if callback is not None:
                callback(i, result)

        return results


# ---------------------------------------------------------------------------
# Convenience function (matches CPU run_ensemble API)
# ---------------------------------------------------------------------------

def run_ensemble_cuda(problem, n_particles, r0_sampler, v0_sampler,
                      t_max, dt=None, save_every=100, rng_seed=None,
                      r_escape=None, renorm_interval=0, callback=None):
    """
    Run an ensemble of SSE trajectories on the GPU.

    Drop-in replacement for ``molmot.obe.stochastic.run_ensemble`` but
    runs all trajectories in parallel on the GPU.

    Parameters
    ----------
    problem : SSEProblem
        Problem specification (from molmot.obe.stochastic).
    n_particles : int
        Number of trajectories.
    r0_sampler : callable() -> ndarray (3,)
        Returns initial position in metres.
    v0_sampler : callable() -> ndarray (3,)
        Returns initial velocity in m/s.
    t_max : float
        Simulation time in seconds.
    dt : float, optional
        Time step in seconds. Default: 0.01 / Gamma.
    save_every : int
        Save interval in time steps.
    rng_seed : int, optional
        Base random seed.
    r_escape : float, optional
        Escape radius in metres.
    renorm_interval : int
        Re-normalize psi every N steps (0 = disabled).
    callback : callable(i, result), optional
        Called after each trajectory result is available.

    Returns
    -------
    list of SSEResult
    """
    solver = SSESolverCUDA(problem)
    return solver.run_ensemble(
        n_particles=n_particles,
        r0_sampler=r0_sampler,
        v0_sampler=v0_sampler,
        t_max=t_max,
        dt=dt,
        save_every=save_every,
        rng_seed=rng_seed,
        r_escape=r_escape,
        renorm_interval=renorm_interval,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# Utility: get GPU info
# ---------------------------------------------------------------------------

def gpu_info():
    """Print GPU device information and return dict of properties."""
    if not HAS_CUPY:
        print("CuPy not available.")
        return {}

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    info = {
        'name': props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
        'compute_capability': f"{props['major']}.{props['minor']}",
        'total_memory_GB': props['totalGlobalMem'] / 1e9,
        'multiprocessors': props['multiProcessorCount'],
        'max_threads_per_block': props['maxThreadsPerBlock'],
        'max_blocks_per_grid_x': props['maxGridSize'][0],
        'shared_memory_per_block': props['sharedMemPerBlock'],
    }

    print(f"GPU: {info['name']}")
    print(f"  Compute capability: {info['compute_capability']}")
    print(f"  Memory: {info['total_memory_GB']:.1f} GB")
    print(f"  SMs: {info['multiprocessors']}")
    print(f"  Max blocks (x): {info['max_blocks_per_grid_x']}")
    print(f"  Shared memory/block: {info['shared_memory_per_block']} bytes")

    # Estimate max concurrent trajectories
    # Each trajectory uses ~4KB of register/local memory
    # Conservative: 256 blocks per SM
    max_traj = info['multiprocessors'] * 256
    print(f"  Estimated max concurrent trajectories: ~{max_traj}")

    return info
