#!/usr/bin/env python3
"""
SrOH DC Red MOT: Julia vs Python CPU vs Python GPU three-way comparison.

Uses the QuantumSimulations.jl SrOH_DC_redMOT example parameters:
  - 4 frequency components, detunings [-10.3, -5.9, -9.2, -29.9] MHz
  - Power ratios [0.50, 0.08, 0.15, 0.27], pols [s+, s-, s+, s-]
  - B' = 21.4 G/cm, total power 100 mW, beam radius 10 mm

Algorithms compared:
  * Julia (precomputed in ``julia_redmot_results.npz``): plain (unsaturated)
    rate equations -- matches the inline reference Python script.
  * Python inline (unsaturated):  same algorithm as Julia, used to
    independently validate the Julia/Python alignment.
  * molmot CPU ``solve_rate_equations``:  full rate equations *with*
    ``1/(1+s_total)`` saturation -- this is the production algorithm.
  * molmot CUDA ``force_vs_z_cuda``: GPU port of the saturated solver.

Expected:
  Julia vs Python-inline (unsat)   ~ 1e-12  (machine precision)
  CPU-saturated vs GPU-saturated   ~ 1e-8   (double precision FP ordering)
  Unsaturated vs saturated         systematic (saturation correction)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia, MASS_KG
from molmot.obe.fields import LaserBeam
from molmot.obe.rate_equations import solve_rate_equations
from molmot.constants import hbar, h, c

import molmot_cuda
from molmot_cuda import (
    force_vs_z_cuda,
    force_vs_v_cuda,
    spring_constant_cuda,
    damping_coefficient_cuda,
)


# =====================================================================
# Parameters (must match the inline Julia script used to produce the
# precomputed validation/julia_redmot_results.npz)
# =====================================================================

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia_sim")

DETUNINGS_MHZ = [-10.3, -5.9, -9.2, -29.9]
POWER_RATIOS = [0.50, 0.08, 0.15, 0.27]
POLS_Q = [2, 0, 2, 0]               # sigma+, sigma-, sigma+, sigma-
B_GRAD_GCM = 21.4
B_GRAD_SI = B_GRAD_GCM * 1e-4 / 1e-2  # T/m == G/cm * 1e-2 / 1e-4 = G/cm * 1e-2
# i.e. 21.4 G/cm = 0.214 T/m
TOTAL_POWER_MW = 100.0
BEAM_RADIUS_M = 10e-3
LAMBDA_M = 687e-9


def build_hallas_beams(mol):
    """Construct the 8 LaserBeam objects (4 frequencies, +/- z) used in
    the Hallas SrOH_DC_redMOT example."""

    Gamma = mol.Gamma
    n_g = mol.n_ground
    n_e = mol.n_excited
    E = mol.energies
    E_e_last = E[n_g + n_e - 1]

    detunings_Hz = np.array(DETUNINGS_MHZ) * 1e6
    freqs_Hz = [
        E_e_last - E[0] + detunings_Hz[0],
        E_e_last - E[0] + detunings_Hz[1],
        E_e_last - E[9] + detunings_Hz[2],
        E_e_last - E[9] + detunings_Hz[3],
    ]

    Isat = np.pi * h * c * Gamma / (3.0 * LAMBDA_M ** 3)
    ratios = np.array(POWER_RATIOS, dtype=float)
    ratios /= ratios.sum()
    powers_W = TOTAL_POWER_MW * 1e-3 * ratios
    intensities = powers_W * (2.0 / (np.pi * BEAM_RADIUS_M ** 2))
    sats = intensities / Isat

    beams = []
    for i_comp, (f, q_fwd, s0) in enumerate(zip(freqs_Hz, POLS_Q, sats)):
        for direction in [+1, -1]:
            q_eff = q_fwd if direction > 0 else (2 - q_fwd if q_fwd != 1 else 1)
            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=f,
                polarization=q_eff,
                s0=s0,
            ))

    return beams, freqs_Hz, sats


# =====================================================================
# Inline unsaturated rate-equation solver (matches Julia exactly)
# =====================================================================

def rate_eq_force_unsat(mol, freqs_hz, pols_q, sats_per_freq,
                        v_z, z_m, B_grad_Gcm):
    """Reference (unsaturated) rate-equation force -- ports
    sroh_redmot_comparison.py's algorithm and Julia's inline version."""
    Gamma = mol.Gamma
    k = mol.k
    n_g = mol.n_ground
    n_e = mol.n_excited

    B_gauss = B_grad_Gcm * z_m * 100.0
    Es = mol.energies + mol.zeeman_z_diag * B_gauss * Gamma / (2.0 * np.pi)

    d_sq = mol.d_squared
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = np.sum(d_sq[:, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq[ig, ie, :]) / total

    n_comp = len(freqs_hz)
    n_beams = 2 * n_comp
    R = np.zeros((n_g, n_e, n_beams))
    F_beam = np.zeros((n_g, n_e, n_beams))

    for i_comp in range(n_comp):
        omega_laser = freqs_hz[i_comp]
        q_pol = pols_q[i_comp]
        s0 = sats_per_freq[i_comp]
        for i_dir, kdir in enumerate([+1, -1]):
            i_beam = i_comp * 2 + i_dir
            doppler = -k * kdir * v_z / (2.0 * np.pi)
            q_eff = q_pol if kdir > 0 else (2 - q_pol if q_pol != 1 else 1)
            for ig in range(n_g):
                for ie in range(n_e):
                    d2 = d_sq[ig, ie, q_eff]
                    if d2 < 1e-15:
                        continue
                    omega_trans = Es[n_g + ie] - Es[ig]
                    delta_eff = (omega_laser + doppler - omega_trans) * 2.0 * np.pi
                    L = (Gamma / 2.0) ** 2 / (delta_eff ** 2 + (Gamma / 2.0) ** 2)
                    rate = (Gamma / 2.0) * s0 * d2 * L
                    R[ig, ie, i_beam] = rate
                    F_beam[ig, ie, i_beam] = hbar * k * kdir * rate

    R_sum = np.sum(R, axis=2)
    M_mat = np.zeros((n_g, n_g))
    for ig in range(n_g):
        M_mat[ig, ig] -= np.sum(R_sum[ig, :])
        for ik in range(n_g):
            for ie in range(n_e):
                M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    M_sol = M_mat.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(n_g); rhs[-1] = 1.0
    from scipy.linalg import solve as splsolve
    try:
        p = splsolve(M_sol, rhs)
    except Exception:
        p = np.ones(n_g) / n_g
    p = np.maximum(p, 0.0); p /= np.sum(p)
    force = sum(p[ig] * np.sum(F_beam[ig, :, :]) for ig in range(n_g))
    return force, p


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 78)
    print("  SrOH DC Red MOT: Julia vs Python CPU vs Python GPU")
    print("  Hallas SrOH_DC_redMOT example (4 freq components)")
    print("=" * 78)

    info = molmot_cuda.get_device_info()
    print(f"  GPU: {info['name']} (cc {info['compute_capability']}, "
          f"{info['free_memory_MB']:.0f}/{info['total_memory_MB']:.0f} MB free)")

    mol = load_sroh_from_julia(JULIA_DIR, verbose=False)
    print(f"  Molecule: SrOH ({mol.n_ground} ground + {mol.n_excited} excited)")
    print(f"  Gamma/(2pi) = {mol.Gamma / (2.0 * np.pi) / 1e6:.2f} MHz")

    beams, freqs_Hz, sats = build_hallas_beams(mol)
    n_g = mol.n_ground
    F_unit = hbar * mol.k * mol.Gamma / 2.0
    print(f"  Per-component saturation: {sats}")
    print(f"  4 freq + 8 beams configured")

    # ---- Grids
    z_arr = np.linspace(-5e-3, 5e-3, 101)
    v_arr = np.linspace(-6.0, 6.0, 101)
    nz = z_arr.size
    nv = v_arr.size

    # ---- Load precomputed Julia results
    jl = np.load(os.path.join(os.path.dirname(__file__),
                              "julia_redmot_results.npz"))
    Fz_jl = jl["Fz"]                       # already F/F_unit
    Fv_jl = jl["Fv"]
    Pz_jl = np.column_stack([jl[f"Pz_g{i+1}"] for i in range(n_g)])
    print(f"  Julia results: {Fz_jl.shape[0]} pts, "
          f"Fz max |..|={np.max(np.abs(Fz_jl)):.6f}")

    # ---- Inline UNSATURATED CPU (matches Julia algorithm)
    print("\nRunning Python inline (UNSATURATED)...")
    t0 = time.time()
    Fz_cpu_u = np.empty(nz)
    Pz_cpu_u = np.empty((nz, n_g))
    for i, z in enumerate(z_arr):
        Fz_cpu_u[i], Pz_cpu_u[i] = rate_eq_force_unsat(
            mol, freqs_Hz, POLS_Q, sats, 0.0, z, B_GRAD_GCM)
    Fv_cpu_u = np.empty(nv)
    for i, v in enumerate(v_arr):
        Fv_cpu_u[i], _ = rate_eq_force_unsat(
            mol, freqs_Hz, POLS_Q, sats, v, 0.0, B_GRAD_GCM)
    dz, dv = 0.3e-3, 0.1
    fp, _ = rate_eq_force_unsat(mol, freqs_Hz, POLS_Q, sats, 0, +dz, B_GRAD_GCM)
    fm, _ = rate_eq_force_unsat(mol, freqs_Hz, POLS_Q, sats, 0, -dz, B_GRAD_GCM)
    k_cpu_u = -(fp - fm) / (2.0 * dz)
    fvp, _ = rate_eq_force_unsat(mol, freqs_Hz, POLS_Q, sats, +dv, 0, B_GRAD_GCM)
    fvm, _ = rate_eq_force_unsat(mol, freqs_Hz, POLS_Q, sats, -dv, 0, B_GRAD_GCM)
    beta_cpu_u = -(fvp - fvm) / (2.0 * dv * MASS_KG)
    t_cpu_u = time.time() - t0
    print(f"  done in {t_cpu_u:.2f} s")

    # ---- molmot CPU (SATURATED) -- solve_rate_equations
    print("\nRunning molmot CPU (SATURATED)...")
    t0 = time.time()
    Fz_cpu_s = np.empty(nz)
    Pz_cpu_s = np.empty((nz, n_g))
    for i, z in enumerate(z_arr):
        F, P, _ = solve_rate_equations(mol, beams, 0.0, z, B_GRAD_SI)
        Fz_cpu_s[i] = F
        Pz_cpu_s[i] = P
    Fv_cpu_s = np.array([solve_rate_equations(mol, beams, v, 0.0, B_GRAD_SI)[0]
                         for v in v_arr])
    fp, _, _ = solve_rate_equations(mol, beams, 0.0, +dz, B_GRAD_SI)
    fm, _, _ = solve_rate_equations(mol, beams, 0.0, -dz, B_GRAD_SI)
    k_cpu_s = -(fp - fm) / (2.0 * dz)
    fvp, _, _ = solve_rate_equations(mol, beams, +dv, 0.0, B_GRAD_SI)
    fvm, _, _ = solve_rate_equations(mol, beams, -dv, 0.0, B_GRAD_SI)
    beta_cpu_s = -(fvp - fvm) / (2.0 * dv * MASS_KG)
    t_cpu_s = time.time() - t0
    print(f"  done in {t_cpu_s:.2f} s")

    # ---- molmot CUDA (SATURATED)
    print("\nRunning molmot CUDA (SATURATED)...")
    t0 = time.time()
    Fz_gpu, Pz_gpu, _ = force_vs_z_cuda(mol, beams, z_arr, 0.0, B_GRAD_SI)
    Fv_gpu = force_vs_v_cuda(mol, beams, v_arr, 0.0, B_GRAD_SI)
    k_gpu = spring_constant_cuda(mol, beams, B_GRAD_SI, dz=dz)
    beta_gpu = damping_coefficient_cuda(mol, beams, B_GRAD_SI, dv=dv) / MASS_KG
    t_gpu = time.time() - t0
    print(f"  done in {t_gpu:.3f} s  ({t_cpu_s / max(t_gpu, 1e-9):.1f}x faster than CPU)")

    # ---- Normalised force arrays
    Fz_cpu_u_n = Fz_cpu_u / F_unit
    Fv_cpu_u_n = Fv_cpu_u / F_unit
    Fz_cpu_s_n = Fz_cpu_s / F_unit
    Fv_cpu_s_n = Fv_cpu_s / F_unit
    Fz_gpu_n = Fz_gpu / F_unit
    Fv_gpu_n = Fv_gpu / F_unit

    def rel_rms(a, b):
        a = np.asarray(a); b = np.asarray(b)
        denom = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
        return np.sqrt(np.mean((a - b) ** 2)) / denom

    def rel_abs(a, b):
        denom = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / denom

    # ---- Compare and print
    print("\n" + "=" * 100)
    print("  PAIRWISE COMPARISON")
    print("=" * 100)

    print(f"\n  Julia (unsat) vs Python-inline (unsat)   -- algorithm sanity check:")
    print(f"    F(z) rel RMS: {rel_rms(Fz_jl, Fz_cpu_u_n):.2e}")
    print(f"    F(v) rel RMS: {rel_rms(Fv_jl, Fv_cpu_u_n):.2e}")
    print(f"    Populations rel RMS: {rel_rms(Pz_jl, Pz_cpu_u):.2e}")

    print(f"\n  molmot CPU (saturated) vs molmot CUDA (saturated)  -- CUDA fidelity:")
    print(f"    F(z) rel RMS: {rel_rms(Fz_cpu_s_n, Fz_gpu_n):.2e}")
    print(f"    F(v) rel RMS: {rel_rms(Fv_cpu_s_n, Fv_gpu_n):.2e}")
    print(f"    Populations rel RMS: {rel_rms(Pz_cpu_s, Pz_gpu):.2e}")
    print(f"    k rel error:    {rel_abs(k_cpu_s, k_gpu):.2e}")
    print(f"    beta rel error: {rel_abs(beta_cpu_s, beta_gpu):.2e}")

    print(f"\n  Julia (unsat) vs molmot CPU (saturated)  -- saturation correction:")
    print(f"    F(z) rel RMS: {rel_rms(Fz_jl, Fz_cpu_s_n):.2e}")
    print(f"    F(v) rel RMS: {rel_rms(Fv_jl, Fv_cpu_s_n):.2e}")
    print(f"    Populations rel RMS: {rel_rms(Pz_jl, Pz_cpu_s):.2e}")

    # ---- Summary table (per the prompt)
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)

    def fmt(x):
        return f"{x:>13.6e}"

    print(f"\n  {'Quantity':<22} {'Julia(unsat)':>14} {'CPU(unsat)':>14} "
          f"{'CPU(sat)':>14} {'GPU(sat)':>14}")
    print("  " + "-" * 90)
    print(f"  {'max |F(z)| (hbk Γ/2)':<22}"
          f" {np.max(np.abs(Fz_jl)):>14.6e}"
          f" {np.max(np.abs(Fz_cpu_u_n)):>14.6e}"
          f" {np.max(np.abs(Fz_cpu_s_n)):>14.6e}"
          f" {np.max(np.abs(Fz_gpu_n)):>14.6e}")
    print(f"  {'k_spring  (N/m)':<22}"
          f" {'n/a (not in npz)':>14}"
          f" {k_cpu_u:>14.6e}"
          f" {k_cpu_s:>14.6e}"
          f" {k_gpu:>14.6e}")
    print(f"  {'beta_damp (1/s)':<22}"
          f" {'n/a (not in npz)':>14}"
          f" {beta_cpu_u:>14.6e}"
          f" {beta_cpu_s:>14.6e}"
          f" {beta_gpu:>14.6e}")

    print(f"\n  {'Pairwise RMS':<22} {'CPU-J (unsat)':>14} {'GPU-CPU (sat)':>14} "
          f"{'GPU-J (mixed)':>14}")
    print("  " + "-" * 78)
    print(f"  {'F(z) rel RMS':<22}"
          f" {rel_rms(Fz_jl, Fz_cpu_u_n):>14.2e}"
          f" {rel_rms(Fz_cpu_s_n, Fz_gpu_n):>14.2e}"
          f" {rel_rms(Fz_jl, Fz_gpu_n):>14.2e}")
    print(f"  {'F(v) rel RMS':<22}"
          f" {rel_rms(Fv_jl, Fv_cpu_u_n):>14.2e}"
          f" {rel_rms(Fv_cpu_s_n, Fv_gpu_n):>14.2e}"
          f" {rel_rms(Fv_jl, Fv_gpu_n):>14.2e}")
    print(f"  {'Populations rel RMS':<22}"
          f" {rel_rms(Pz_jl, Pz_cpu_u):>14.2e}"
          f" {rel_rms(Pz_cpu_s, Pz_gpu):>14.2e}"
          f" {rel_rms(Pz_jl, Pz_gpu):>14.2e}")

    # ---- Persist results
    out = os.path.join(os.path.dirname(__file__), "three_way_results.npz")
    np.savez(out,
             z_arr=z_arr, v_arr=v_arr,
             Fz_jl=Fz_jl, Fv_jl=Fv_jl, Pz_jl=Pz_jl,
             Fz_cpu_u=Fz_cpu_u_n, Fv_cpu_u=Fv_cpu_u_n, Pz_cpu_u=Pz_cpu_u,
             k_cpu_u=k_cpu_u, beta_cpu_u=beta_cpu_u,
             Fz_cpu_s=Fz_cpu_s_n, Fv_cpu_s=Fv_cpu_s_n, Pz_cpu_s=Pz_cpu_s,
             k_cpu_s=k_cpu_s, beta_cpu_s=beta_cpu_s,
             Fz_gpu=Fz_gpu_n, Fv_gpu=Fv_gpu_n, Pz_gpu=Pz_gpu,
             k_gpu=k_gpu, beta_gpu=beta_gpu)
    print(f"\n  saved arrays -> {out}")

    # ---- Final verdict
    gpu_cpu_tol = 1e-6
    jl_cpu_tol = 1e-6
    fz_gpu_cpu = rel_rms(Fz_cpu_s_n, Fz_gpu_n)
    fz_jl_cpu_u = rel_rms(Fz_jl, Fz_cpu_u_n)
    print("\n" + "=" * 100)
    if fz_gpu_cpu < gpu_cpu_tol and fz_jl_cpu_u < jl_cpu_tol:
        print(f"  PASS: CPU/GPU agree (rel RMS {fz_gpu_cpu:.1e} < {gpu_cpu_tol:.0e})"
              f" AND Julia/CPU-unsat agree (rel RMS {fz_jl_cpu_u:.1e} < {jl_cpu_tol:.0e})")
    else:
        print(f"  Check needed -- CPU/GPU: {fz_gpu_cpu:.2e}, "
              f"Julia/CPU-unsat: {fz_jl_cpu_u:.2e}")
    print("=" * 100)


if __name__ == "__main__":
    main()
