#!/usr/bin/env python3
"""
SrOH MOT Simulation with Full OBE (Sub-Doppler Physics)

Three cases:
  1. RF red MOT — validate against experiment
  2. DC red MOT (4-frequency) — predict performance
  3. Blue DC MOT (Λ-BDM) — sub-Doppler cooling comparison

The OBE captures ground-state coherences that produce:
  - Sisyphus cooling in lin⊥lin polarization gradients
  - Velocity-selective dark states (Λ-cooling)
  - Sub-Doppler temperatures below T_Doppler = ℏΓ/(2k_B)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.obe_subdoppler import (
    obe_force_subdoppler,
    make_beam_pairs_rf, make_beam_pairs_dc4, make_beam_pairs_blue
)
from molmot.constants import hbar, k_B

# ═══════════════════════════════════════════════════════
# Load molecular data
# ═══════════════════════════════════════════════════════
mol = load_sroh_from_julia('julia_sim')
mass = mol.mass
Gamma = mol.Gamma
k = mol.k
FUNIT = hbar * k * Gamma / 2
TD = hbar * Gamma / (2 * k_B)

print("=" * 70)
print("  SrOH MOT — Full OBE with Sub-Doppler Physics")
print("=" * 70)
print(f"  T_Doppler = {TD*1e6:.0f} µK")
print(f"  v_Doppler = {Gamma/(2*k):.2f} m/s")

# ═══════════════════════════════════════════════════════
# Helper: force scans
# ═══════════════════════════════════════════════════════
def scan_z(make_pairs_fn, v, B_grad, Omega_G, gef, z_arr, n_sp=12):
    """Scan force vs position."""
    F = np.zeros(len(z_arr))
    R = np.zeros(len(z_arr))
    for i, z in enumerate(z_arr):
        bp = make_pairs_fn()
        f, rho, r = obe_force_subdoppler(mol, bp, v, z, B_grad, Omega_G,
                                          n_spatial=n_sp, Gamma_eff_factor=gef)
        F[i] = f
        R[i] = r
    return F, R

def scan_v(make_pairs_fn, z, B_grad, Omega_G, gef, v_arr, n_sp=12):
    """Scan force vs velocity."""
    F = np.zeros(len(v_arr))
    for i, v in enumerate(v_arr):
        bp = make_pairs_fn()
        f, _, _ = obe_force_subdoppler(mol, bp, v, z, B_grad, Omega_G,
                                        n_spatial=n_sp, Gamma_eff_factor=gef)
        F[i] = f
    return F

def scan_v_rf(delta_G, s0, B_grad, Omega_G, gef, v_arr, z=0, n_sp=12):
    """Scan force vs velocity for RF MOT (average over two phases)."""
    F = np.zeros(len(v_arr))
    for i, v in enumerate(v_arr):
        bp0 = make_beam_pairs_rf(mol, delta_G, s0, phase=0)
        bp1 = make_beam_pairs_rf(mol, delta_G, s0, phase=1)
        f0, _, _ = obe_force_subdoppler(mol, bp0, v, z, B_grad, Omega_G,
                                         n_spatial=n_sp, Gamma_eff_factor=gef)
        f1, _, _ = obe_force_subdoppler(mol, bp1, v, z, -B_grad, Omega_G,
                                         n_spatial=n_sp, Gamma_eff_factor=gef)
        F[i] = 0.5 * (f0 + f1)
    return F

# ═══════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════
B_grad = 16e-2        # 16 G/cm
s0 = 1.0
Omega_G = np.sqrt(s0 / 2)  # Rabi freq / Gamma
gef = 0.015           # repumping efficiency (calibrated to RF MOT expt)

nz = 40
nv = 60
n_sp = 12  # spatial averaging points (per wavelength)

z_arr = np.linspace(-5e-3, 5e-3, nz)
v_arr = np.linspace(-3, 3, nv)
v_fine = np.linspace(-0.5, 0.5, 40)  # fine scan for sub-Doppler

# ═══════════════════════════════════════════════════════
# Case 1: RF Red MOT
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Case 1: RF Red MOT (Δ = -1Γ)")
print("=" * 70)

t0 = time.time()
delta_rf = -1.0

print("  Computing F(z) at v=0...")
Fz_rf = np.zeros(nz)
for i, z in enumerate(z_arr):
    bp0 = make_beam_pairs_rf(mol, delta_rf, s0, phase=0)
    bp1 = make_beam_pairs_rf(mol, delta_rf, s0, phase=1)
    f0, _, _ = obe_force_subdoppler(mol, bp0, 0, z, B_grad, Omega_G,
                                     n_spatial=n_sp, Gamma_eff_factor=gef)
    f1, _, _ = obe_force_subdoppler(mol, bp1, 0, z, -B_grad, Omega_G,
                                     n_spatial=n_sp, Gamma_eff_factor=gef)
    Fz_rf[i] = 0.5 * (f0 + f1)
    if (i+1) % 10 == 0:
        print(f"    {i+1}/{nz} ({time.time()-t0:.0f}s)")

print("  Computing F(v) at z=0...")
Fv_rf = scan_v_rf(delta_rf, s0, B_grad, Omega_G, gef, v_arr, z=0, n_sp=n_sp)

print("  Computing F(v) fine scan for sub-Doppler...")
Fv_rf_fine = scan_v_rf(delta_rf, s0, B_grad, Omega_G, gef, v_fine, z=0, n_sp=n_sp)

# Spring constant and damping
dz = 0.3e-3
bp0p = make_beam_pairs_rf(mol, delta_rf, s0, phase=0)
bp1p = make_beam_pairs_rf(mol, delta_rf, s0, phase=1)
fp, _, _ = obe_force_subdoppler(mol, bp0p, 0, +dz, B_grad, Omega_G, n_sp, gef)
fm, _, _ = obe_force_subdoppler(mol, bp0p, 0, -dz, B_grad, Omega_G, n_sp, gef)
fp1, _, _ = obe_force_subdoppler(mol, bp1p, 0, +dz, -B_grad, Omega_G, n_sp, gef)
fm1, _, _ = obe_force_subdoppler(mol, bp1p, 0, -dz, -B_grad, Omega_G, n_sp, gef)
k_rf = -(0.5*(fp+fp1) - 0.5*(fm+fm1)) / (2*dz)

dv = 0.05
fvp = scan_v_rf(delta_rf, s0, B_grad, Omega_G, gef, np.array([+dv]), n_sp=n_sp)[0]
fvm = scan_v_rf(delta_rf, s0, B_grad, Omega_G, gef, np.array([-dv]), n_sp=n_sp)[0]
alpha_rf = -(fvp - fvm) / (2*dv)
beta_rf = alpha_rf / mass
omega_rf = np.sqrt(abs(k_rf)/mass)/(2*np.pi) if k_rf > 0 else 0

print(f"\n  RF MOT results (OBE):")
print(f"    k     = {k_rf:.3e} N/m")
print(f"    omega = 2pi x {omega_rf:.1f} Hz")
print(f"    beta  = {beta_rf:.0f} /s")
print(f"    (Expt: omega ~ 2pi x 45 Hz, beta ~ 100 /s)")
print(f"    Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════
# Case 2: DC Red MOT (4-frequency)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Case 2: DC Red MOT 4-Frequency (Δ = -2.7Γ, δ = 2.4Γ)")
print("=" * 70)

t0 = time.time()
delta_dc = -2.7
split_dc = 2.4

make_dc = lambda: make_beam_pairs_dc4(mol, delta_dc, split_dc, s0)

print("  Computing F(z) at v=0...")
Fz_dc, Rz_dc = scan_z(make_dc, 0, B_grad, Omega_G, gef, z_arr, n_sp)
print(f"    ({time.time()-t0:.0f}s)")

print("  Computing F(v) at z=0...")
Fv_dc = scan_v(make_dc, 0, B_grad, Omega_G, gef, v_arr, n_sp)

print("  Fine velocity scan...")
Fv_dc_fine = scan_v(make_dc, 0, B_grad, Omega_G, gef, v_fine, n_sp)

fp_dc, _, _ = obe_force_subdoppler(mol, make_dc(), 0, +dz, B_grad, Omega_G, n_sp, gef)
fm_dc, _, _ = obe_force_subdoppler(mol, make_dc(), 0, -dz, B_grad, Omega_G, n_sp, gef)
k_dc = -(fp_dc - fm_dc) / (2*dz)

fvp_dc, _, _ = obe_force_subdoppler(mol, make_dc(), +dv, 0, B_grad, Omega_G, n_sp, gef)
fvm_dc, _, _ = obe_force_subdoppler(mol, make_dc(), -dv, 0, B_grad, Omega_G, n_sp, gef)
alpha_dc = -(fvp_dc - fvm_dc) / (2*dv)
beta_dc = alpha_dc / mass
omega_dc = np.sqrt(abs(k_dc)/mass)/(2*np.pi) if k_dc > 0 else 0

print(f"\n  DC MOT results (OBE):")
print(f"    k     = {k_dc:.3e} N/m")
print(f"    omega = 2pi x {omega_dc:.1f} Hz")
print(f"    beta  = {beta_dc:.0f} /s")
print(f"    Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════
# Case 3: Blue DC MOT (Λ-BDM)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Case 3: Blue DC MOT / Λ-BDM (Δ = +3Γ, δ₂ = -0.1Γ)")
print("=" * 70)

t0 = time.time()
delta_blue = +3.0
delta2_blue = -0.1

make_blue = lambda: make_beam_pairs_blue(mol, delta_blue, delta2_blue, s0)

print("  Computing F(z) at v=0...")
Fz_blue, Rz_blue = scan_z(make_blue, 0, B_grad, Omega_G, gef, z_arr, n_sp)
print(f"    ({time.time()-t0:.0f}s)")

print("  Computing F(v) at z=0...")
Fv_blue = scan_v(make_blue, 0, B_grad, Omega_G, gef, v_arr, n_sp)

print("  Fine velocity scan...")
Fv_blue_fine = scan_v(make_blue, 0, B_grad, Omega_G, gef, v_fine, n_sp)

fp_b, _, _ = obe_force_subdoppler(mol, make_blue(), 0, +dz, B_grad, Omega_G, n_sp, gef)
fm_b, _, _ = obe_force_subdoppler(mol, make_blue(), 0, -dz, B_grad, Omega_G, n_sp, gef)
k_blue = -(fp_b - fm_b) / (2*dz)

fvp_b, _, _ = obe_force_subdoppler(mol, make_blue(), +dv, 0, B_grad, Omega_G, n_sp, gef)
fvm_b, _, _ = obe_force_subdoppler(mol, make_blue(), -dv, 0, B_grad, Omega_G, n_sp, gef)
alpha_blue = -(fvp_b - fvm_b) / (2*dv)
beta_blue = alpha_blue / mass
omega_blue = np.sqrt(abs(k_blue)/mass)/(2*np.pi) if k_blue > 0 else 0

print(f"\n  Blue MOT results (OBE):")
print(f"    k     = {k_blue:.3e} N/m")
print(f"    omega = 2pi x {omega_blue:.1f} Hz")
print(f"    beta  = {beta_blue:.0f} /s")
print(f"    Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  COMPARISON (all three MOT types)")
print("=" * 70)
print(f"{'':>20s} {'RF red':>12s} {'DC 4-freq':>12s} {'Blue Λ-BDM':>12s} {'Expt (RF)':>12s}")
print("-" * 70)
print(f"{'k (N/m)':>20s} {k_rf:12.2e} {k_dc:12.2e} {k_blue:12.2e} {'~1.4e-20':>12s}")
print(f"{'ω/(2π) (Hz)':>20s} {omega_rf:12.1f} {omega_dc:12.1f} {omega_blue:12.1f} {'45':>12s}")
print(f"{'β (s⁻¹)':>20s} {beta_rf:12.0f} {beta_dc:12.0f} {beta_blue:12.0f} {'100':>12s}")

# ═══════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════
print("\n  Generating plots...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

labels = ['RF Red MOT', 'DC 4-Freq MOT', 'Blue Λ-BDM']
colors = ['#1f77b4', '#d62728', '#2ca02c']
Fzs = [Fz_rf, Fz_dc, Fz_blue]
Fvs = [Fv_rf, Fv_dc, Fv_blue]
Fv_fines = [Fv_rf_fine, Fv_dc_fine, Fv_blue_fine]
ks = [k_rf, k_dc, k_blue]
betas = [beta_rf, beta_dc, beta_blue]
omegas = [omega_rf, omega_dc, omega_blue]

# Row 1: F(z) for each case
for col, (lbl, clr, Fz, kval) in enumerate(zip(labels, colors, Fzs, ks)):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(z_arr*1e3, Fz/FUNIT, color=clr, lw=2)
    ax.axhline(0, c='gray', ls='--', alpha=0.5)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('F / (ℏkΓ/2)')
    trap = 'TRAP' if kval > 0 else 'no trap'
    ax.set_title(f'{lbl}\nk={kval:.2e}, ω=2π×{omegas[col]:.0f}Hz [{trap}]')
    ax.grid(True, alpha=0.3)

# Row 2: F(v) broad + fine for sub-Doppler
for col, (lbl, clr, Fv, Fv_f) in enumerate(zip(labels, colors, Fvs, Fv_fines)):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(v_arr, Fv/FUNIT, color=clr, lw=2, label='broad')
    ax.axhline(0, c='gray', ls='--', alpha=0.5)
    ax.set_xlabel('v (m/s)')
    ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title(f'{lbl}: F(v) at z=0')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inset: fine velocity scan showing sub-Doppler structure
    axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    axins.plot(v_fine, Fv_f/FUNIT, color=clr, lw=1.5)
    axins.axhline(0, c='gray', ls='--', alpha=0.3)
    axins.set_xlabel('v (m/s)', fontsize=7)
    axins.set_ylabel('F/(ℏkΓ/2)', fontsize=7)
    axins.set_title('sub-Doppler region', fontsize=7)
    axins.tick_params(labelsize=6)

# Row 3: Comparison overlay + summary
ax = fig.add_subplot(gs[2, 0])
for lbl, clr, Fz in zip(labels, colors, Fzs):
    ax.plot(z_arr*1e3, Fz/FUNIT, color=clr, lw=2, label=lbl)
ax.axhline(0, c='gray', ls='--', alpha=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('F / (ℏkΓ/2)')
ax.set_title('Restoring Force Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[2, 1])
for lbl, clr, Fv in zip(labels, colors, Fvs):
    ax.plot(v_arr, Fv/FUNIT, color=clr, lw=2, label=lbl)
ax.axhline(0, c='gray', ls='--', alpha=0.5)
ax.set_xlabel('v (m/s)')
ax.set_ylabel('F / (ℏkΓ/2)')
ax.set_title('Damping Force Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Summary text
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
txt = (
    f"SrOH MOT — OBE with Sub-Doppler\n"
    f"{'─'*40}\n\n"
    f"{'':>14s} {'RF':>8s} {'DC-4f':>8s} {'Blue':>8s}\n"
    f"{'k (N/m)':>14s} {k_rf:8.1e} {k_dc:8.1e} {k_blue:8.1e}\n"
    f"{'ω/(2π) Hz':>14s} {omega_rf:8.0f} {omega_dc:8.0f} {omega_blue:8.0f}\n"
    f"{'β (1/s)':>14s} {beta_rf:8.0f} {beta_dc:8.0f} {beta_blue:8.0f}\n\n"
    f"Experiment (RF MOT):\n"
    f"  ω = 2π × 45 Hz\n"
    f"  β = 100 s⁻¹\n"
    f"  T = 1.2 mK\n\n"
    f"T_Doppler = {TD*1e6:.0f} µK\n"
    f"Sub-Doppler visible in F(v)\n"
    f"near v = 0 (inset plots)"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('SrOH MOT — Full OBE with Sub-Doppler Physics', fontsize=16, fontweight='bold')

outpath = '/Users/dslmd/Downloads/DC red MOT/sroh_obe_subdoppler.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"  Saved: {outpath}")
plt.close(fig)

print("\nDone!")
