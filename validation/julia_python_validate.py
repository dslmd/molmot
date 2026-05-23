#!/usr/bin/env python3
"""
Head-to-head validation: Julia run_dc_mot.jl vs Python molmot (16-level).

Uses the same Julia-exported CSV data (energies, TDM, Zeeman) and the same
unsaturated rate-equation force model to compare results point by point.
Also runs the full molmot.obe.rate_equations solver for cross-check.
"""

import sys, os, time
import numpy as np
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia, MolecularData, GAMMA_RAD, K_WAVE, MASS_KG
from molmot.constants import hbar, h
from molmot.obe.fields import LaserBeam, make_dc_mot_beams
from molmot.obe.rate_equations import solve_rate_equations

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia_sim")

# ═══════════════════════════════════════════════════════════
# Load molecular data (same CSV files that Julia uses)
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("  Julia vs Python Validation (16-level SrOH)")
print("=" * 70)

mol = load_sroh_from_julia(JULIA_DIR, verbose=True)

# ═══════════════════════════════════════════════════════════
# Unsaturated rate-equation force (matches Julia run_dc_mot.jl exactly)
# ═══════════════════════════════════════════════════════════

def rate_eq_force_julia_style(mol, freqs_hz, pols_q, s0, v_z, z_m, B_grad_Gcm):
    """
    Mirror of the Julia rate_eq_force() function.

    Parameters
    ----------
    freqs_hz : list of 4 floats, laser frequencies in Hz
    pols_q : list of 4 ints, polarisation index 0=sigma-, 1=pi, 2=sigma+
    s0 : float, saturation per component per beam
    v_z : float, velocity in m/s
    z_m : float, position in metres
    B_grad_Gcm : float, B gradient in Gauss/cm
    """
    Gamma = mol.Gamma
    k = mol.k
    n_g = mol.n_ground
    n_e = mol.n_excited
    n_s = mol.n_states

    B_gauss = B_grad_Gcm * z_m * 100  # z(m) -> z(cm) -> B(Gauss)

    # Zeeman-shifted energies
    zeeman_diag = mol.zeeman_z_diag  # per-Gauss factor
    E_Z = zeeman_diag * B_gauss * Gamma / (2 * np.pi)  # Hz
    Es = mol.energies + E_Z

    # Branching ratios
    d_sq = mol.d_squared
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = 0.0
        for ig in range(n_g):
            total += np.sum(d_sq[ig, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq[ig, ie, :]) / total

    n_comp = len(freqs_hz)
    n_beams = 2 * n_comp  # +z and -z for each component

    R = np.zeros((n_g, n_e, n_beams))
    F_beam = np.zeros((n_g, n_e, n_beams))
    hbar_val = hbar

    for i_comp in range(n_comp):
        omega_laser = freqs_hz[i_comp]
        q_pol = pols_q[i_comp]  # 0=sigma-, 1=pi, 2=sigma+

        for i_dir, kdir in enumerate([+1, -1]):
            i_beam = i_comp * 2 + i_dir
            doppler = -k * kdir * v_z / (2 * np.pi)  # Hz

            # Retro-reflected: sigma+ <-> sigma-
            if kdir > 0:
                q_eff = q_pol
            else:
                q_eff = 2 - q_pol if q_pol != 1 else 1  # swap 0<->2, keep 1

            for ig in range(n_g):
                for ie in range(n_e):
                    d2 = d_sq[ig, ie, q_eff]
                    if d2 < 1e-15:
                        continue

                    omega_trans = Es[n_g + ie] - Es[ig]  # Hz
                    delta_eff = (omega_laser + doppler - omega_trans) * 2 * np.pi  # rad/s

                    L = (Gamma / 2) ** 2 / (delta_eff ** 2 + (Gamma / 2) ** 2)
                    rate = (Gamma / 2) * s0 * d2 * L

                    R[ig, ie, i_beam] = rate
                    F_beam[ig, ie, i_beam] = hbar_val * k * kdir * rate

    # Steady-state populations
    R_sum = np.sum(R, axis=2)

    M_mat = np.zeros((n_g, n_g))
    for ig in range(n_g):
        M_mat[ig, ig] -= np.sum(R_sum[ig, :])
        for ik in range(n_g):
            for ie in range(n_e):
                M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    M_sol = M_mat.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(n_g)
    rhs[-1] = 1.0

    try:
        from scipy.linalg import solve
        p = solve(M_sol, rhs)
    except Exception:
        p = np.ones(n_g) / n_g

    p = np.maximum(p, 0.0)
    p /= np.sum(p)

    force = sum(p[ig] * np.sum(F_beam[ig, :, :]) for ig in range(n_g))
    R_scatter = sum(p[ig] * np.sum(R_sum[ig, :]) for ig in range(n_g))

    return force, p, R_scatter


# ═══════════════════════════════════════════════════════════
# Set up identical parameters to Julia
# ═══════════════════════════════════════════════════════════

B_gradient_Gcm = 16.0
s0 = 1.0
Delta_opt = -0.88   # in units of Gamma
delta_opt = 0.39     # in units of Gamma

Gamma = mol.Gamma
Delta_Hz = Delta_opt * Gamma / (2 * np.pi)
delta_Hz = delta_opt * Gamma / (2 * np.pi)

omega_J32 = mol.omega_J32
omega_J12 = mol.omega_J12

freqs_4f = [
    omega_J32 + Delta_Hz + delta_Hz,   # J=3/2, sigma+
    omega_J32 + Delta_Hz - delta_Hz,   # J=3/2, sigma-
    omega_J12 + Delta_Hz + delta_Hz,   # J=1/2, sigma+
    omega_J12 + Delta_Hz - delta_Hz,   # J=1/2, sigma-
]
pols_4f = [2, 0, 2, 0]  # 0=sigma-, 1=pi, 2=sigma+

print(f"\nTransition frequencies:")
print(f"  omega_J32 = {omega_J32/1e6:.3f} MHz")
print(f"  omega_J12 = {omega_J12/1e6:.3f} MHz")
print(f"  SR split  = {(omega_J32 - omega_J12)/1e6:.1f} MHz")
print(f"\nParameters:")
print(f"  Delta = {Delta_opt:.2f} Gamma = {Delta_Hz/1e6:.1f} MHz")
print(f"  delta = {delta_opt:.2f} Gamma = {delta_Hz/1e6:.1f}")
print(f"  s0 = {s0:.1f}, B' = {B_gradient_Gcm:.0f} G/cm")

# ═══════════════════════════════════════════════════════════
# F(z) at v=0
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Computing F(z) at v=0 (Python)")
print("=" * 70)

nz = 200
z_arr = np.linspace(-5e-3, 5e-3, nz)
Fz_py = np.zeros(nz)
Pz_py = np.zeros((nz, mol.n_ground))
Rz_py = np.zeros(nz)

t0 = time.time()
for i, z in enumerate(z_arr):
    Fz_py[i], Pz_py[i], Rz_py[i] = rate_eq_force_julia_style(
        mol, freqs_4f, pols_4f, s0, 0.0, z, B_gradient_Gcm)
t_fz = time.time() - t0

F_unit = hbar * mol.k * Gamma / 2
print(f"  Time: {t_fz:.2f} s")
print(f"  Max |F| = {np.max(np.abs(Fz_py))/F_unit:.4f} hbar*k*Gamma/2")

# Spring constant
dFdz = np.diff(Fz_py) / np.diff(z_arr)
k_spring = -dFdz[nz // 2]
omega_trap = np.sqrt(k_spring / MASS_KG) / (2 * np.pi) if k_spring > 0 else 0.0
print(f"  Spring constant k = {k_spring:.3e} N/m")
print(f"  Trap frequency omega = 2*pi * {omega_trap:.1f} Hz")

# ═══════════════════════════════════════════════════════════
# F(v) at z=0
# ═══════════════════════════════════════════════════════════

print("\nComputing F(v) at z=0...")
nv = 200
v_arr = np.linspace(-6.0, 6.0, nv)
Fv_py = np.zeros(nv)
Fv_1mm_py = np.zeros(nv)

t0 = time.time()
for i, v in enumerate(v_arr):
    Fv_py[i], _, _ = rate_eq_force_julia_style(
        mol, freqs_4f, pols_4f, s0, v, 0.0, B_gradient_Gcm)
    Fv_1mm_py[i], _, _ = rate_eq_force_julia_style(
        mol, freqs_4f, pols_4f, s0, v, 1e-3, B_gradient_Gcm)
t_fv = time.time() - t0

dFdv = np.diff(Fv_py) / np.diff(v_arr)
beta_damp = -dFdv[nv // 2] / MASS_KG
print(f"  Time: {t_fv:.2f} s")
print(f"  Damping beta = {beta_damp:.0f} s^-1")

# ═══════════════════════════════════════════════════════════
# Parameter optimisation: Delta vs delta
# ═══════════════════════════════════════════════════════════

print("\nOptimising: detuning vs split...")
Delta_scan = np.linspace(-3.0, -0.2, 25)
delta_scan = np.linspace(0.05, 2.0, 25)
K_map = np.zeros((len(Delta_scan), len(delta_scan)))
beta_map = np.zeros_like(K_map)
dz = 0.3e-3
dv = 0.1

t0 = time.time()
for iD, Dv in enumerate(Delta_scan):
    for id_, dv_ in enumerate(delta_scan):
        D_Hz = Dv * Gamma / (2 * np.pi)
        d_Hz = dv_ * Gamma / (2 * np.pi)
        fs = [
            omega_J32 + D_Hz + d_Hz, omega_J32 + D_Hz - d_Hz,
            omega_J12 + D_Hz + d_Hz, omega_J12 + D_Hz - d_Hz,
        ]

        fp, _, _ = rate_eq_force_julia_style(mol, fs, pols_4f, s0, 0.0, +dz, B_gradient_Gcm)
        fm, _, _ = rate_eq_force_julia_style(mol, fs, pols_4f, s0, 0.0, -dz, B_gradient_Gcm)
        K_map[iD, id_] = -(fp - fm) / (2 * dz)

        fvp, _, _ = rate_eq_force_julia_style(mol, fs, pols_4f, s0, +dv, 0.0, B_gradient_Gcm)
        fvm, _, _ = rate_eq_force_julia_style(mol, fs, pols_4f, s0, -dv, 0.0, B_gradient_Gcm)
        beta_map[iD, id_] = -(fvp - fvm) / (2 * dv * MASS_KG)

t_opt = time.time() - t0

idx_best = np.unravel_index(np.argmax(K_map), K_map.shape)
Delta_best = Delta_scan[idx_best[0]]
delta_best = delta_scan[idx_best[1]]
k_best = K_map[idx_best]
beta_best = beta_map[idx_best]
omega_best = np.sqrt(k_best / MASS_KG) / (2 * np.pi) if k_best > 0 else 0.0

print(f"  Time: {t_opt:.1f} s")
print(f"\n  OPTIMAL PARAMETERS:")
print(f"    Delta = {Delta_best:.2f} Gamma = {Delta_best*Gamma/2/np.pi/1e6:.1f} MHz")
print(f"    delta = {delta_best:.2f} Gamma = {delta_best*Gamma/2/np.pi/1e6:.1f} MHz")
print(f"    k = {k_best:.3e} N/m")
print(f"    omega = 2*pi * {omega_best:.1f} Hz")
print(f"    beta = {beta_best:.0f} s^-1")

# ═══════════════════════════════════════════════════════════
# Trajectories at optimum
# ═══════════════════════════════════════════════════════════

print("\nSimulating trajectories...")
dt = 1e-6
tmax = 0.03
nstep = int(tmax / dt)

D_Hz_best = Delta_best * Gamma / (2 * np.pi)
d_Hz_best = delta_best * Gamma / (2 * np.pi)
freqs_best = [
    omega_J32 + D_Hz_best + d_Hz_best, omega_J32 + D_Hz_best - d_Hz_best,
    omega_J12 + D_Hz_best + d_Hz_best, omega_J12 + D_Hz_best - d_Hz_best,
]

trajs = []
for z0, v0, lbl in [(3e-3, 0.0, "z0=3mm"),
                     (0.0, -2.0, "v0=-2m/s"),
                     (2e-3, -1.0, "mixed"),
                     (0.0, -5.0, "v0=-5m/s")]:
    zt = np.zeros(nstep)
    vt = np.zeros(nstep)
    tt = np.arange(nstep) * dt
    zt[0] = z0
    vt[0] = v0
    for j in range(1, nstep):
        f, _, _ = rate_eq_force_julia_style(
            mol, freqs_best, pols_4f, s0, vt[j - 1], zt[j - 1], B_gradient_Gcm)
        a = f / MASS_KG
        vt[j] = vt[j - 1] + a * dt
        zt[j] = zt[j - 1] + vt[j] * dt
        if abs(zt[j]) > 0.015:
            zt[j:] = zt[j]
            vt[j:] = vt[j]
            break
    trajs.append((tt, zt, vt, lbl))

# ═══════════════════════════════════════════════════════════
# Capture velocity
# ═══════════════════════════════════════════════════════════

print("Estimating capture velocity...")
v_cap = 0.0
for v0 in np.arange(0.5, 15.5, 0.5):
    zt_test = 3e-3
    vt_test = -v0
    trapped = True
    for _ in range(5000):
        f, _, _ = rate_eq_force_julia_style(
            mol, freqs_best, pols_4f, s0, vt_test, zt_test, B_gradient_Gcm)
        vt_test += (f / MASS_KG) * 1e-6
        zt_test += vt_test * 1e-6
        if abs(zt_test) > 0.015:
            trapped = False
            break
    if trapped:
        v_cap = v0
    else:
        break
print(f"  Capture velocity: {v_cap:.1f} m/s")

# ═══════════════════════════════════════════════════════════
# Save numerical data for comparison
# ═══════════════════════════════════════════════════════════

outdir = os.path.dirname(__file__)
np.savez(os.path.join(outdir, "python_results.npz"),
         z_arr=z_arr, Fz=Fz_py, Pz=Pz_py, Rz=Rz_py,
         v_arr=v_arr, Fv=Fv_py, Fv_1mm=Fv_1mm_py,
         Delta_scan=Delta_scan, delta_scan=delta_scan,
         K_map=K_map, beta_map=beta_map,
         k_spring=k_spring, omega_trap=omega_trap,
         beta_damp=beta_damp,
         Delta_best=Delta_best, delta_best=delta_best,
         k_best=k_best, omega_best=omega_best, beta_best=beta_best,
         v_cap=v_cap)

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PYTHON RESULTS SUMMARY")
print("=" * 70)
print(f"\n  At Delta={Delta_opt:.2f}G, delta={delta_opt:.2f}G:")
print(f"    Max |F(z)| = {np.max(np.abs(Fz_py))/F_unit:.4f} hbar*k*Gamma/2")
print(f"    k_spring   = {k_spring:.3e} N/m")
print(f"    omega_trap  = 2*pi * {omega_trap:.1f} Hz")
print(f"    beta_damp   = {beta_damp:.0f} s^-1")
print(f"\n  Optimised:")
print(f"    Delta_best  = {Delta_best:.2f} Gamma")
print(f"    delta_best  = {delta_best:.2f} Gamma")
print(f"    k_best      = {k_best:.3e} N/m")
print(f"    omega_best  = 2*pi * {omega_best:.1f} Hz")
print(f"    beta_best   = {beta_best:.0f} s^-1")
print(f"    v_capture   = {v_cap:.1f} m/s")

# ═══════════════════════════════════════════════════════════
# Also build from scratch and compare
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Cross-check: build_sroh_hamiltonian() from scratch")
print("=" * 70)

from molmot.molecules.sroh import build_sroh_hamiltonian
mol_scratch = build_sroh_hamiltonian(verbose=True)

# Compare energies
E_julia = mol.energies
E_python = mol_scratch.energies

print("\nEnergy comparison (Julia CSV vs Python from-scratch):")
E_g_mean_j = np.mean(E_julia[:12])
E_g_mean_p = np.mean(E_python[:12])
for i in range(12):
    ej = (E_julia[i] - E_g_mean_j) / 1e6
    ep = (E_python[i] - E_g_mean_p) / 1e6
    diff = abs(ej - ep)
    status = "OK" if diff < 0.5 else "DIFF"
    print(f"  g{i+1:2d}: Julia={ej:+8.3f} MHz  Python={ep:+8.3f} MHz  diff={diff:.3f} MHz [{status}]")

# Compare d^2
print("\nTDM comparison (nonzero |d|^2):")
d2_julia = mol.d_squared
d2_python = mol_scratch.d_squared
for ig in range(12):
    for ie in range(4):
        for q in range(3):
            dj = d2_julia[ig, ie, q]
            dp = d2_python[ig, ie, q]
            if dj > 1e-4 or dp > 1e-4:
                diff = abs(dj - dp) / max(dj, dp, 1e-30)
                status = "OK" if diff < 0.05 else "DIFF"
                pol = ["s-", "pi", "s+"][q]
                print(f"  g{ig+1}->e{ie+1} ({pol}): Julia={dj:.4f}  Python={dp:.4f}  "
                      f"rel_diff={diff:.1%} [{status}]")

print("\nDone!")
