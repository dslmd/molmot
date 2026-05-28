#!/usr/bin/env python3
"""
SrOH DC MOT Production Simulation
===================================

Complete DC MOT prediction for SrOH using the molmot package.

Steps:
  A. RF MOT validation (calibration against Lasner et al. 2024 experiment)
  B. DC MOT parameter optimization (2D scan over detuning and split)
  C. Detailed force analysis at optimal parameters
  D. Publication-quality plots
  E. Experimental recommendation printout

Uses the molmot package with Julia-validated molecular data.

Author: auto-generated production script
"""

import os
import sys
import time
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# ======================================================================
#  Configuration
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "julia_sim")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, "..", "julia_sim")
OUT_DIR = BASE_DIR

# Physical constants (also available from molmot.constants)
from molmot.constants import hbar, k_B, amu

# Molecular parameters
LAMBDA_NM = 688.0
GAMMA_HZ = 6.4e6
GAMMA_RAD = 2.0 * np.pi * GAMMA_HZ
MASS_AMU = 105.0
MASS_KG = MASS_AMU * amu
K_WAVE = 2.0 * np.pi / (LAMBDA_NM * 1e-9)
F_UNIT = hbar * K_WAVE * GAMMA_RAD / 2.0  # Force unit: hbar * k * Gamma / 2

# Plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ======================================================================
#  A. RF MOT Validation
# ======================================================================

def run_rf_validation(mol_data):
    """
    Run the RF MOT simulation at experimental conditions and compare
    with the Lasner et al. 2024 measurements.

    Experimental parameters from Lasner et al. PRL 134, 083401 (2025):
      - P = 4.7 mW per sideband
      - Gamma_eff factor ~ 0.015 (due to repumping cycle)
      - B' ~ 16 G/cm (RMS)
      - Detuning ~ -1 Gamma
    """
    from molmot.mot.simulator import RFMOTSimulator
    from molmot.mot.force_scan import (force_vs_z, force_vs_v,
                                       spring_constant, damping_coefficient,
                                       capture_velocity)

    print("\n" + "=" * 70)
    print("  A. RF MOT VALIDATION (Calibration)")
    print("=" * 70)

    # --- Experimental conditions ---
    # s0 ~ 7.3 from P = 4.7 mW per sideband in 20 mm beam
    # Gamma_eff = 0.015 accounts for repumping duty cycle
    rf_s0 = 7.3
    rf_Gamma_eff = 0.015
    rf_delta = -1.0   # Gamma
    rf_B_Tm = 0.16    # T/m = 16 G/cm

    print(f"\n  Experimental parameters:")
    print(f"    Detuning:         {rf_delta:.1f} Gamma = {rf_delta * GAMMA_HZ / 1e6:.1f} MHz")
    print(f"    s0 per sideband:  {rf_s0:.1f}")
    print(f"    Gamma_eff factor: {rf_Gamma_eff}")
    print(f"    B gradient:       {rf_B_Tm * 1e2:.0f} G/cm")

    sim_rf = RFMOTSimulator(mol_data, delta_Gamma=rf_delta, s0=rf_s0,
                             B_gradient=rf_B_Tm, Gamma_eff_factor=rf_Gamma_eff)

    # Force profiles
    print("\n  Computing RF MOT force profiles...")
    z_arr = np.linspace(-5e-3, 5e-3, 200)
    v_arr = np.linspace(-6, 6, 200)

    Fz_rf, pop_z_rf, Rz_rf = force_vs_z(sim_rf, z_arr, v=0.0)
    Fv_rf = force_vs_v(sim_rf, v_arr, z=0.0)

    # MOT characteristics
    k_rf = spring_constant(sim_rf, dz=0.3e-3)
    alpha_rf = damping_coefficient(sim_rf, dv=0.1)
    beta_rf = alpha_rf / MASS_KG
    omega_rf = np.sqrt(abs(k_rf) / MASS_KG) / (2 * np.pi) if k_rf > 0 else 0.0

    # Scattering rate at centre
    _, _, R_centre = sim_rf.force(0.0, 0.0)

    # Capture velocity
    print("  Estimating RF MOT capture velocity...")
    v_cap_rf = capture_velocity(sim_rf, z_start=3e-3, v_max=15, dv=0.5,
                                 dt=1e-6, n_steps=5000)

    # Print comparison table
    print(f"\n  {'='*60}")
    print(f"  RF MOT: Simulation vs Experiment (Lasner et al. 2024)")
    print(f"  {'='*60}")
    print(f"  {'Quantity':30s}  {'Simulation':>12s}  {'Experiment':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    print(f"  {'Damping beta (1/s)':30s}  {beta_rf:12.0f}  {'~100':>12s}")
    print(f"  {'Trap freq (Hz)':30s}  {omega_rf:12.1f}  {'~45':>12s}")
    print(f"  {'Spring constant (N/m)':30s}  {k_rf:12.2e}  {'--':>12s}")
    print(f"  {'Scattering rate (MHz)':30s}  {R_centre/(2*np.pi*1e6):12.3f}  {'--':>12s}")
    print(f"  {'Capture velocity (m/s)':30s}  {v_cap_rf:12.1f}  {'~10':>12s}")
    print(f"  {'='*60}")

    return {
        "z_arr": z_arr, "v_arr": v_arr,
        "Fz": Fz_rf, "Fv": Fv_rf,
        "pop_z": pop_z_rf, "Rz": Rz_rf,
        "k": k_rf, "alpha": alpha_rf, "beta": beta_rf,
        "omega": omega_rf, "v_cap": v_cap_rf,
        "R_centre": R_centre,
        "s0": rf_s0, "delta": rf_delta,
        "Gamma_eff": rf_Gamma_eff, "B_Tm": rf_B_Tm,
    }


# ======================================================================
#  B. DC MOT Parameter Optimization
# ======================================================================

def _find_equilibrium(sim, z_range=(-10e-3, 10e-3), n_scan=200):
    """Find the restoring equilibrium position where F=0 and dF/dz < 0."""
    z_arr = np.linspace(z_range[0], z_range[1], n_scan)
    F_arr = np.array([sim.force(0.0, z)[0] for z in z_arr])

    for i in range(len(F_arr) - 1):
        if F_arr[i] > 0 and F_arr[i + 1] < 0:
            z_eq = z_arr[i] + (z_arr[i+1] - z_arr[i]) * abs(F_arr[i]) / (abs(F_arr[i]) + abs(F_arr[i+1]))
            return z_eq
    return None


def run_dc_optimization(mol_data):
    """
    Scan over detuning, polarization split, B-gradient, and saturation
    to find optimal DC MOT parameters.

    Uses a physics-aware merit function that:
      1. Evaluates k and beta at the actual equilibrium position (not z=0)
      2. Penalizes large equilibrium offsets (which reduce effective trap volume)
      3. Requires both positive k and positive beta (trapping + cooling)
    """
    from molmot.mot.simulator import DCMOTSimulator
    from molmot.mot.force_scan import (spring_constant, damping_coefficient)

    print("\n" + "=" * 70)
    print("  B. DC MOT PARAMETER OPTIMIZATION")
    print("=" * 70)

    # --- Primary scan: detuning x split at reference B and s0 ---
    B_ref_Tm = 0.16     # 16 G/cm
    s0_ref = 1.0

    delta_scan = np.linspace(-3.0, -0.1, 30)
    split_scan = np.linspace(0.1, 3.0, 30)
    n_d = len(delta_scan)
    n_s = len(split_scan)

    print(f"\n  Primary 2D scan (with equilibrium finding):")
    print(f"    Detuning:  {delta_scan[0]:.1f} to {delta_scan[-1]:.1f} Gamma  ({n_d} pts)")
    print(f"    Split:     {split_scan[0]:.1f} to {split_scan[-1]:.1f} Gamma  ({n_s} pts)")
    print(f"    B':        {B_ref_Tm*1e2:.0f} G/cm,  s0 = {s0_ref}")

    K_map = np.zeros((n_d, n_s))
    beta_map = np.zeros((n_d, n_s))
    z_eq_map = np.full((n_d, n_s), np.nan)

    t0 = time.time()
    total = n_d * n_s
    count = 0

    for i_d, det in enumerate(delta_scan):
        for i_s, sp in enumerate(split_scan):
            sim = DCMOTSimulator(mol_data, delta_Gamma=det,
                                  split_Gamma=sp, s0=s0_ref,
                                  B_gradient=B_ref_Tm)

            # Find equilibrium
            z_eq = _find_equilibrium(sim)
            if z_eq is None or abs(z_eq) > 9e-3:
                K_map[i_d, i_s] = 0
                beta_map[i_d, i_s] = 0
                z_eq_map[i_d, i_s] = np.nan
            else:
                z_eq_map[i_d, i_s] = z_eq
                k = spring_constant(sim, z0=z_eq, dz=0.3e-3)
                alpha = damping_coefficient(sim, z0=z_eq, dv=0.1)
                K_map[i_d, i_s] = k
                beta_map[i_d, i_s] = alpha / MASS_KG

            count += 1
            if count % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"    {count}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"    Scan completed in {elapsed:.1f} s")

    # Merit function: maximize k subject to positive beta and small offset
    merit = K_map.copy()
    merit[K_map <= 0] = 0
    merit[beta_map <= 0] = 0
    # Penalize large offsets: multiply by exp(-|z_eq|/3mm)
    offset_penalty = np.where(np.isfinite(z_eq_map),
                              np.exp(-np.abs(z_eq_map) / 3e-3), 0.0)
    merit *= offset_penalty

    idx_best = np.unravel_index(np.argmax(merit), merit.shape)
    delta_best = delta_scan[idx_best[0]]
    split_best = split_scan[idx_best[1]]
    k_best = K_map[idx_best]
    beta_best = beta_map[idx_best]
    z_eq_best = z_eq_map[idx_best]

    # Store results in the same format as optimize_parameters for compatibility
    result_primary = {
        "delta_scan": delta_scan,
        "split_scan": split_scan,
        "K_map": K_map,
        "beta_map": beta_map,
        "z_eq_map": z_eq_map,
        "delta_best": float(delta_best),
        "split_best": float(split_best),
        "k_best": float(k_best),
        "beta_best": float(beta_best),
    }

    print(f"\n  Primary scan optimal:")
    print(f"    Delta = {delta_best:.2f} Gamma = {delta_best * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"    Split = {split_best:.2f} Gamma = {split_best * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"    k     = {k_best:.3e} N/m")
    print(f"    beta  = {beta_best:.0f} /s")
    if np.isfinite(z_eq_best):
        print(f"    z_eq  = {z_eq_best * 1e3:.2f} mm")

    # --- B-gradient scan ---
    B_scan_Gcm = np.array([5.0, 10.0, 16.0, 25.0, 40.0])
    B_scan_Tm = B_scan_Gcm / 100.0

    print(f"\n  B-gradient scan at optimal detuning/split:")
    k_vs_B = np.zeros(len(B_scan_Tm))
    beta_vs_B = np.zeros(len(B_scan_Tm))

    for i, B_Tm in enumerate(B_scan_Tm):
        sim = DCMOTSimulator(mol_data, delta_Gamma=delta_best,
                              split_Gamma=split_best, s0=s0_ref,
                              B_gradient=B_Tm)
        z_eq = _find_equilibrium(sim)
        if z_eq is not None and abs(z_eq) < 9e-3:
            k_vs_B[i] = spring_constant(sim, z0=z_eq, dz=0.3e-3)
            alpha = damping_coefficient(sim, z0=z_eq, dv=0.1)
            beta_vs_B[i] = alpha / MASS_KG
            print(f"    B' = {B_Tm*1e2:5.1f} G/cm:  k = {k_vs_B[i]:.3e} N/m,  "
                  f"beta = {beta_vs_B[i]:.0f} /s,  z_eq = {z_eq*1e3:.2f} mm")
        else:
            print(f"    B' = {B_Tm*1e2:5.1f} G/cm:  no equilibrium found")

    # --- Saturation scan ---
    s0_scan = np.array([0.5, 1.0, 2.0, 5.0])

    print(f"\n  Saturation scan at optimal detuning/split, B'={B_ref_Tm*1e2:.0f} G/cm:")
    k_vs_s0 = np.zeros(len(s0_scan))
    beta_vs_s0 = np.zeros(len(s0_scan))

    for i, s0 in enumerate(s0_scan):
        sim = DCMOTSimulator(mol_data, delta_Gamma=delta_best,
                              split_Gamma=split_best, s0=s0,
                              B_gradient=B_ref_Tm)
        z_eq = _find_equilibrium(sim)
        if z_eq is not None and abs(z_eq) < 9e-3:
            k_vs_s0[i] = spring_constant(sim, z0=z_eq, dz=0.3e-3)
            alpha = damping_coefficient(sim, z0=z_eq, dv=0.1)
            beta_vs_s0[i] = alpha / MASS_KG
            print(f"    s0 = {s0:4.1f}:  k = {k_vs_s0[i]:.3e} N/m,  "
                  f"beta = {beta_vs_s0[i]:.0f} /s,  z_eq = {z_eq*1e3:.2f} mm")
        else:
            print(f"    s0 = {s0:4.1f}:  no equilibrium found")

    return {
        "result_primary": result_primary,
        "delta_best": delta_best,
        "split_best": split_best,
        "k_best": k_best,
        "beta_best": beta_best,
        "z_eq_best": z_eq_best if np.isfinite(z_eq_best) else 0.0,
        "B_ref_Tm": B_ref_Tm,
        "s0_ref": s0_ref,
        "B_scan_Gcm": B_scan_Gcm,
        "B_scan_Tm": B_scan_Tm,
        "k_vs_B": k_vs_B,
        "beta_vs_B": beta_vs_B,
        "s0_scan": s0_scan,
        "k_vs_s0": k_vs_s0,
        "beta_vs_s0": beta_vs_s0,
    }


# ======================================================================
#  C. Detailed Analysis at Optimal Parameters
# ======================================================================

def run_detailed_analysis(mol_data, opt_result):
    """
    Detailed force analysis, 2D force map, trajectories, capture velocity,
    and power budget at the optimal parameters.
    """
    from molmot.mot.simulator import DCMOTSimulator
    from molmot.mot.force_scan import (force_vs_z, force_vs_v,
                                       spring_constant, damping_coefficient,
                                       capture_velocity)
    from molmot.propagation.trajectories import simulate_trajectory

    delta = opt_result["delta_best"]
    split = opt_result["split_best"]
    B_Tm = opt_result["B_ref_Tm"]
    s0 = opt_result["s0_ref"]
    z_eq = opt_result.get("z_eq_best", 0.0)
    if z_eq is None or not np.isfinite(z_eq):
        z_eq = 0.0

    print("\n" + "=" * 70)
    print("  C. DETAILED ANALYSIS AT OPTIMAL PARAMETERS")
    print("=" * 70)
    print(f"    Delta = {delta:.3f} Gamma = {delta * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"    Split = {split:.3f} Gamma = {split * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"    B'    = {B_Tm * 1e2:.0f} G/cm")
    print(f"    s0    = {s0}")
    print(f"    z_eq  = {z_eq * 1e3:.2f} mm")

    sim = DCMOTSimulator(mol_data, delta_Gamma=delta, split_Gamma=split,
                          s0=s0, B_gradient=B_Tm)

    # Find actual equilibrium
    z_eq_found = _find_equilibrium(sim)
    if z_eq_found is not None and abs(z_eq_found) < 9e-3:
        z_eq = z_eq_found
        print(f"    (refined z_eq = {z_eq * 1e3:.2f} mm)")

    # --- 1. High-resolution force profiles ---
    print("\n  1. Force profiles F(z) and F(v) at 200 points each...")
    z_arr = np.linspace(-10e-3, 10e-3, 200)
    v_arr = np.linspace(-8, 8, 200)

    Fz, pop_z, Rz = force_vs_z(sim, z_arr, v=0.0)
    Fv = force_vs_v(sim, v_arr, z=z_eq)

    # Also compute F(v) at z_eq + 1 mm for comparison
    Fv_1mm = force_vs_v(sim, v_arr, z=z_eq + 1e-3)

    # Spring constant and damping at equilibrium
    k_opt = spring_constant(sim, z0=z_eq, dz=0.3e-3)
    alpha_opt = damping_coefficient(sim, z0=z_eq, dv=0.1)
    beta_opt = alpha_opt / MASS_KG
    omega_opt = np.sqrt(abs(k_opt) / MASS_KG) / (2 * np.pi) if k_opt > 0 else 0.0
    _, _, R_centre = sim.force(0.0, z_eq)

    print(f"    k     = {k_opt:.3e} N/m")
    print(f"    omega = 2*pi * {omega_opt:.1f} Hz")
    print(f"    beta  = {beta_opt:.0f} /s")
    print(f"    R_sc  = {R_centre / (2*np.pi*1e6):.3f} MHz")

    # --- 2. 2D force map F(z, v) centred on equilibrium ---
    z_half = 5e-3
    print(f"\n  2. Computing 2D force map F(z,v) at 50x50 points "
          f"(centred on z_eq={z_eq*1e3:.1f}mm)...")
    z_2d = np.linspace(z_eq - z_half, z_eq + z_half, 50)
    v_2d = np.linspace(-5, 5, 50)
    F_2d = np.zeros((len(v_2d), len(z_2d)))

    t0 = time.time()
    for iv, v in enumerate(v_2d):
        for iz, z in enumerate(z_2d):
            F_2d[iv, iz], _, _ = sim.force(v, z)
    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f} s")

    # --- 3. Trajectory simulation (start near equilibrium) ---
    print(f"\n  3. Simulating 5 molecular trajectories (30 ms each, "
          f"near z_eq={z_eq*1e3:.1f}mm)...")
    traj_configs = [
        (z_eq + 3e-3,   0.0,  f"z0=z_eq+3mm, v0=0"),
        (z_eq,          -2.0,  f"z0=z_eq, v0=-2m/s"),
        (z_eq + 2e-3,  -1.0,  f"z0=z_eq+2mm, v0=-1m/s"),
        (z_eq - 1e-3,   3.0,  f"z0=z_eq-1mm, v0=+3m/s"),
        (z_eq,          -5.0,  f"z0=z_eq, v0=-5m/s"),
    ]

    trajectories = []
    for z0, v0, label in traj_configs:
        t_arr, z_traj, v_traj = simulate_trajectory(
            sim, z0, v0, t_max=0.030, dt=2e-6, z_escape=0.015)
        trajectories.append((t_arr, z_traj, v_traj, label))
        final_z = z_traj[-1]
        final_v = v_traj[-1]
        trapped = "TRAPPED" if abs(final_z) < 0.014 else "escaped"
        print(f"    {label:35s}: z_f = {final_z*1e3:+6.2f} mm, "
              f"v_f = {final_v:+6.2f} m/s  [{trapped}]")

    # --- 4. Capture velocity ---
    print("\n  4. Estimating capture velocity...")
    v_cap = capture_velocity(sim, z_start=z_eq + 3e-3, v_max=15, dv=0.5,
                              dt=1e-6, n_steps=10000)
    print(f"    Capture velocity = {v_cap:.1f} m/s")

    # --- 5. Power budget ---
    I_sat_val = np.pi * 6.62607015e-34 * 2.998e8 * GAMMA_HZ / (3.0 * (LAMBDA_NM * 1e-9) ** 3)
    beam_w = 10e-3  # 1/e^2 radius = 10 mm => diameter = 20 mm
    P_per_comp = s0 * I_sat_val * np.pi * beam_w ** 2 / 2.0  # Gaussian beam
    P_per_arm = 4 * P_per_comp
    P_total = 3 * P_per_arm  # 3 retro-reflected axes

    print(f"\n  5. Power budget:")
    print(f"    I_sat = {I_sat_val / 10:.1f} mW/cm^2")
    print(f"    Per component: {P_per_comp * 1e3:.1f} mW")
    print(f"    Per arm (4 components): {P_per_arm * 1e3:.1f} mW")
    print(f"    Total (3 axes): {P_total * 1e3:.0f} mW")

    return {
        "z_arr": z_arr, "v_arr": v_arr,
        "Fz": Fz, "Fv": Fv, "Fv_1mm": Fv_1mm,
        "pop_z": pop_z, "Rz": Rz,
        "z_2d": z_2d, "v_2d": v_2d, "F_2d": F_2d,
        "k": k_opt, "alpha": alpha_opt, "beta": beta_opt,
        "omega": omega_opt, "R_centre": R_centre,
        "v_cap": v_cap, "z_eq": z_eq,
        "trajectories": trajectories,
        "I_sat": I_sat_val, "beam_w": beam_w,
        "P_per_comp": P_per_comp, "P_per_arm": P_per_arm, "P_total": P_total,
    }


# ======================================================================
#  D. Generate Publication-Quality Plots
# ======================================================================

def plot_rf_validation(rf_result, out_dir):
    """Plot 1: RF MOT comparison with experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("SrOH RF MOT Validation (Lasner et al. 2024 comparison)",
                 fontsize=14, y=1.02)

    z_mm = rf_result["z_arr"] * 1e3
    v_ms = rf_result["v_arr"]

    # F(z)
    ax = axes[0]
    ax.plot(z_mm, rf_result["Fz"] / F_UNIT, "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"F / ($\hbar k \Gamma$/2)")
    ax.set_title(r"Restoring force F(z) at v=0")
    ax.grid(True, alpha=0.3)
    # Annotate spring constant
    k_val = rf_result["k"]
    omega_val = rf_result["omega"]
    ax.text(0.05, 0.95, f"k = {k_val:.2e} N/m\n"
            f"f = {omega_val:.0f} Hz",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # F(v)
    ax = axes[1]
    ax.plot(v_ms, rf_result["Fv"] / F_UNIT, "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel(r"F / ($\hbar k \Gamma$/2)")
    ax.set_title("Damping force F(v) at z=0")
    ax.grid(True, alpha=0.3)
    beta_val = rf_result["beta"]
    ax.text(0.05, 0.95, f"beta = {beta_val:.0f} /s\n(expt: ~100 /s)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Comparison bar chart
    ax = axes[2]
    labels = ["Damping\n(1/s)", "Trap freq\n(Hz)", "Capture v\n(m/s)"]
    sim_vals = [rf_result["beta"], rf_result["omega"], rf_result["v_cap"]]
    exp_vals = [100, 45, 10]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, sim_vals, w, label="Simulation", color="steelblue")
    bars2 = ax.bar(x + w / 2, exp_vals, w, label="Experiment", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Simulation vs Experiment")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "production_rf_validation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dc_optimization(opt_result, out_dir):
    """Plot 2: Parameter scan heatmaps."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("SrOH DC MOT Parameter Optimization", fontsize=14, y=0.98)

    rp = opt_result["result_primary"]
    delta_scan = rp["delta_scan"]
    split_scan = rp["split_scan"]
    K_map = rp["K_map"]
    beta_map = rp["beta_map"]
    delta_best = opt_result["delta_best"]
    split_best = opt_result["split_best"]

    # Row 1: Spring constant, damping, trap frequency heatmaps
    # Spring constant
    ax = axes[0, 0]
    K_plot = np.maximum(K_map, 0)
    im = ax.pcolormesh(split_scan, delta_scan, K_plot, cmap="hot", shading="auto")
    ax.plot(split_best, delta_best, "c*", ms=14, mew=1.5)
    ax.set_xlabel(r"Split $\delta$ ($\Gamma$)")
    ax.set_ylabel(r"Detuning $\Delta$ ($\Gamma$)")
    ax.set_title("Spring constant k (N/m)")
    fig.colorbar(im, ax=ax, pad=0.02)

    # Damping
    ax = axes[0, 1]
    bmax = max(abs(np.nanmin(beta_map)), abs(np.nanmax(beta_map)), 1.0)
    im = ax.pcolormesh(split_scan, delta_scan, beta_map, cmap="RdBu",
                       vmin=-bmax, vmax=bmax, shading="auto")
    ax.plot(split_best, delta_best, "k*", ms=14, mew=1.5)
    ax.set_xlabel(r"Split $\delta$ ($\Gamma$)")
    ax.set_ylabel(r"Detuning $\Delta$ ($\Gamma$)")
    ax.set_title(r"Damping $\beta$ (1/s)")
    fig.colorbar(im, ax=ax, pad=0.02)

    # Trap frequency
    omega_map = np.where(K_map > 0,
                         np.sqrt(np.maximum(K_map, 0) / MASS_KG) / (2 * np.pi), 0.0)
    ax = axes[0, 2]
    im = ax.pcolormesh(split_scan, delta_scan, omega_map, cmap="viridis", shading="auto")
    ax.plot(split_best, delta_best, "r*", ms=14, mew=1.5)
    ax.set_xlabel(r"Split $\delta$ ($\Gamma$)")
    ax.set_ylabel(r"Detuning $\Delta$ ($\Gamma$)")
    ax.set_title(r"Trap freq $\omega/(2\pi)$ (Hz)")
    fig.colorbar(im, ax=ax, pad=0.02)

    # Row 2: B-gradient scan, s0 scan, merit function
    # B-gradient scan
    ax = axes[1, 0]
    ax.plot(opt_result["B_scan_Gcm"], opt_result["k_vs_B"], "ro-", lw=2, ms=8)
    ax.set_xlabel("B' (G/cm)")
    ax.set_ylabel("k (N/m)")
    ax.set_title("Spring constant vs B-gradient")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(opt_result["B_scan_Gcm"], opt_result["beta_vs_B"], "bs--", lw=1.5, ms=6)
    ax2.set_ylabel(r"$\beta$ (1/s)", color="blue")

    # s0 scan
    ax = axes[1, 1]
    ax.semilogx(opt_result["s0_scan"], opt_result["k_vs_s0"], "ro-", lw=2, ms=8)
    ax.set_xlabel("s0")
    ax.set_ylabel("k (N/m)")
    ax.set_title("Spring constant vs s0")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogx(opt_result["s0_scan"], opt_result["beta_vs_s0"], "bs--", lw=1.5, ms=6)
    ax2.set_ylabel(r"$\beta$ (1/s)", color="blue")

    # Merit function (k * theta(k>0) * theta(beta>0))
    ax = axes[1, 2]
    merit = K_map.copy()
    merit[K_map <= 0] = 0
    merit[beta_map <= 0] = 0
    im = ax.pcolormesh(split_scan, delta_scan, merit, cmap="magma", shading="auto")
    ax.plot(split_best, delta_best, "g*", ms=14, mew=1.5)
    ax.set_xlabel(r"Split $\delta$ ($\Gamma$)")
    ax.set_ylabel(r"Detuning $\Delta$ ($\Gamma$)")
    ax.set_title("Merit = k (k>0, beta>0)")
    fig.colorbar(im, ax=ax, pad=0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "production_dc_optimization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dc_forces(detail_result, opt_result, out_dir):
    """Plot 3: Force profiles at optimal parameters."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle(
        f"SrOH DC MOT Force Analysis "
        f"($\\Delta$={opt_result['delta_best']:.2f}$\\Gamma$, "
        f"$\\delta$={opt_result['split_best']:.2f}$\\Gamma$, "
        f"B'={opt_result['B_ref_Tm']*1e2:.0f} G/cm)",
        fontsize=14, y=0.99)

    z_mm = detail_result["z_arr"] * 1e3
    v_ms = detail_result["v_arr"]

    # F(z)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(z_mm, detail_result["Fz"] / F_UNIT, "r-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"F / ($\hbar k \Gamma$/2)")
    ax.set_title("Restoring force F(z)")
    ax.grid(True, alpha=0.3)

    # F(v)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(v_ms, detail_result["Fv"] / F_UNIT, "r-", lw=2, label="z = 0")
    ax.plot(v_ms, detail_result["Fv_1mm"] / F_UNIT, "b--", lw=1.5, label="z = 1 mm")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel(r"F / ($\hbar k \Gamma$/2)")
    ax.set_title("Damping force F(v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D force map
    ax = fig.add_subplot(gs[0, 2])
    z_2d_mm = detail_result["z_2d"] * 1e3
    v_2d = detail_result["v_2d"]
    F_2d = detail_result["F_2d"]
    Fmax = max(abs(np.nanmin(F_2d / F_UNIT)), abs(np.nanmax(F_2d / F_UNIT)))
    if Fmax == 0:
        Fmax = 1.0
    im = ax.pcolormesh(z_2d_mm, v_2d, F_2d / F_UNIT, cmap="RdBu_r",
                       vmin=-Fmax, vmax=Fmax, shading="auto")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("v (m/s)")
    ax.set_title(r"F(z,v) / ($\hbar k \Gamma$/2)")
    fig.colorbar(im, ax=ax, pad=0.02)

    # Populations
    ax = fig.add_subplot(gs[1, 0])
    n_ground = detail_result["pop_z"].shape[1]
    for ig in range(min(n_ground, 12)):
        ls = "-" if ig < 4 else ("--" if ig < 7 else ":")
        ax.plot(z_mm, detail_result["pop_z"][:, ig], lw=1.2, ls=ls,
                label=f"g{ig+1}" if ig < 6 else None)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Population")
    ax.set_title("Ground state populations")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scattering rate
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(z_mm, detail_result["Rz"] / (2 * np.pi * 1e6), "g-", lw=2)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Scattering rate (MHz)")
    ax.set_title("Photon scattering rate")
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    text = (
        f"MOT Parameters\n"
        f"{'='*30}\n"
        f"k     = {detail_result['k']:.2e} N/m\n"
        f"f     = {detail_result['omega']:.0f} Hz\n"
        f"beta  = {detail_result['beta']:.0f} /s\n"
        f"R_sc  = {detail_result['R_centre']/(2*np.pi*1e6):.3f} MHz\n"
        f"v_cap = {detail_result['v_cap']:.1f} m/s\n"
        f"{'='*30}\n"
        f"Power Budget\n"
        f"{'='*30}\n"
        f"I_sat = {detail_result['I_sat']/10:.1f} mW/cm^2\n"
        f"Per component: {detail_result['P_per_comp']*1e3:.1f} mW\n"
        f"Per arm: {detail_result['P_per_arm']*1e3:.1f} mW\n"
        f"Total: {detail_result['P_total']*1e3:.0f} mW"
    )
    ax.text(0.1, 0.95, text, transform=ax.transAxes, va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="#f0f0f0", alpha=0.9))

    path = os.path.join(out_dir, "production_dc_forces.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dc_trajectories(detail_result, opt_result, out_dir):
    """Plot 4: Molecular trajectories and phase space."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"SrOH DC MOT Trajectories "
        f"($\\Delta$={opt_result['delta_best']:.2f}$\\Gamma$, "
        f"$\\delta$={opt_result['split_best']:.2f}$\\Gamma$)",
        fontsize=14, y=1.02)

    trajs = detail_result["trajectories"]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    # z vs t
    ax = axes[0]
    for i, (tt, zt, vt, lbl) in enumerate(trajs):
        ax.plot(tt * 1e3, zt * 1e3, lw=1.5, color=colors[i], label=lbl)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Position vs time")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # v vs t
    ax = axes[1]
    for i, (tt, zt, vt, lbl) in enumerate(trajs):
        ax.plot(tt * 1e3, vt, lw=1.5, color=colors[i], label=lbl)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("Velocity vs time")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Phase space (z, v)
    ax = axes[2]
    for i, (tt, zt, vt, lbl) in enumerate(trajs):
        ax.plot(zt * 1e3, vt, lw=1.5, color=colors[i], label=lbl)
        # Mark start
        ax.plot(zt[0] * 1e3, vt[0], "o", color=colors[i], ms=6)
        # Mark end
        ax.plot(zt[-1] * 1e3, vt[-1], "s", color=colors[i], ms=6)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("Phase space (circle=start, square=end)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "production_dc_trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary(rf_result, opt_result, detail_result, out_dir):
    """Plot 5: Summary panel with all key results."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("SrOH DC MOT -- Production Simulation Summary", fontsize=16, y=0.99)

    delta_best = opt_result["delta_best"]
    split_best = opt_result["split_best"]

    # Row 1: RF validation force, DC force, 2D force map, comparison bars
    # RF F(z)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(rf_result["z_arr"] * 1e3, rf_result["Fz"] / F_UNIT, "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"F / ($\hbar k\Gamma$/2)")
    ax.set_title("RF MOT F(z)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # DC F(z)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(detail_result["z_arr"] * 1e3, detail_result["Fz"] / F_UNIT, "r-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"F / ($\hbar k\Gamma$/2)")
    ax.set_title("DC MOT F(z)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # DC F(v)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(detail_result["v_arr"], detail_result["Fv"] / F_UNIT, "r-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel(r"F / ($\hbar k\Gamma$/2)")
    ax.set_title("DC MOT F(v)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Comparison bars
    ax = fig.add_subplot(gs[0, 3])
    labels = [r"$\beta$ (1/s)", "f (Hz)", r"$v_{cap}$ (m/s)"]
    rf_vals = [rf_result["beta"], rf_result["omega"], rf_result["v_cap"]]
    dc_vals = [detail_result["beta"], detail_result["omega"], detail_result["v_cap"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, rf_vals, w, label="RF MOT", color="steelblue")
    ax.bar(x + w/2, dc_vals, w, label="DC MOT", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("RF vs DC MOT", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: Optimization heatmap, B scan, s0 scan, 2D force map
    rp = opt_result["result_primary"]
    K_map = rp["K_map"]
    beta_map = rp["beta_map"]
    delta_scan = rp["delta_scan"]
    split_scan = rp["split_scan"]

    # Heatmap: spring constant
    ax = fig.add_subplot(gs[1, 0])
    K_plot = np.maximum(K_map, 0)
    im = ax.pcolormesh(split_scan, delta_scan, K_plot, cmap="hot", shading="auto")
    ax.plot(split_best, delta_best, "c*", ms=12)
    ax.set_xlabel(r"$\delta$ ($\Gamma$)")
    ax.set_ylabel(r"$\Delta$ ($\Gamma$)")
    ax.set_title("k (N/m)", fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.02, aspect=15)

    # Heatmap: damping
    ax = fig.add_subplot(gs[1, 1])
    bmax = max(abs(np.nanmin(beta_map)), abs(np.nanmax(beta_map)), 1.0)
    im = ax.pcolormesh(split_scan, delta_scan, beta_map, cmap="RdBu",
                       vmin=-bmax, vmax=bmax, shading="auto")
    ax.plot(split_best, delta_best, "k*", ms=12)
    ax.set_xlabel(r"$\delta$ ($\Gamma$)")
    ax.set_ylabel(r"$\Delta$ ($\Gamma$)")
    ax.set_title(r"$\beta$ (1/s)", fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.02, aspect=15)

    # 2D force map
    ax = fig.add_subplot(gs[1, 2])
    z_2d_mm = detail_result["z_2d"] * 1e3
    v_2d = detail_result["v_2d"]
    F_2d = detail_result["F_2d"]
    Fmax = max(abs(np.nanmin(F_2d / F_UNIT)), abs(np.nanmax(F_2d / F_UNIT)), 0.001)
    im = ax.pcolormesh(z_2d_mm, v_2d, F_2d / F_UNIT, cmap="RdBu_r",
                       vmin=-Fmax, vmax=Fmax, shading="auto")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("F(z,v)", fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.02, aspect=15)

    # B and s0 scan combined
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(opt_result["B_scan_Gcm"], opt_result["k_vs_B"], "ro-", lw=2, ms=6, label="k vs B'")
    ax.set_xlabel("B' (G/cm)")
    ax.set_ylabel("k (N/m)", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_title("B' and s0 scans", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Row 3: Trajectories (z vs t, phase space) + summary text
    ax = fig.add_subplot(gs[2, 0:2])
    trajs = detail_result["trajectories"]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i, (tt, zt, vt, lbl) in enumerate(trajs):
        ax.plot(tt * 1e3, zt * 1e3, lw=1.5, color=colors[i], label=lbl)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("z (mm)")
    ax.set_title("DC MOT Trajectories", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Phase space
    ax = fig.add_subplot(gs[2, 2])
    for i, (tt, zt, vt, lbl) in enumerate(trajs):
        ax.plot(zt * 1e3, vt, lw=1.5, color=colors[i])
        ax.plot(zt[0] * 1e3, vt[0], "o", color=colors[i], ms=5)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("Phase space", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    text = (
        "DC MOT Optimal Parameters\n"
        "=" * 28 + "\n"
        f"Delta = {delta_best:.2f} Gamma\n"
        f"      = {delta_best * GAMMA_HZ / 1e6:.2f} MHz\n"
        f"Split = {split_best:.2f} Gamma\n"
        f"      = {split_best * GAMMA_HZ / 1e6:.2f} MHz\n"
        f"B'    = {opt_result['B_ref_Tm']*1e2:.0f} G/cm\n"
        f"s0    = {opt_result['s0_ref']}\n\n"
        "Predicted Performance\n"
        "=" * 28 + "\n"
        f"k     = {detail_result['k']:.2e} N/m\n"
        f"f     = {detail_result['omega']:.0f} Hz\n"
        f"beta  = {detail_result['beta']:.0f} /s\n"
        f"v_cap = {detail_result['v_cap']:.1f} m/s\n\n"
        f"Power = {detail_result['P_total']*1e3:.0f} mW total"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#f0f0f0", alpha=0.9))

    path = os.path.join(out_dir, "production_dc_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
#  E. Print Experimental Recommendation
# ======================================================================

def print_recommendation(rf_result, opt_result, detail_result):
    """Print the complete experimental parameter recommendation."""

    delta = opt_result["delta_best"]
    split = opt_result["split_best"]
    B_Tm = opt_result["B_ref_Tm"]
    s0 = opt_result["s0_ref"]

    print("\n")
    print("=" * 70)
    print("  SrOH DC MOT -- Experimental Recommendation")
    print("=" * 70)
    print(f"  B-field gradient:    {B_Tm * 1e2:.0f} G/cm")
    print(f"  Beam diameter:       {detail_result['beam_w'] * 2e3:.0f} mm (1/e^2)")
    print(f"  Laser configuration: 4 frequencies per arm")
    print(f"    SR sideband 1 (J=3/2): sigma+ at Delta+delta, sigma- at Delta-delta")
    print(f"    SR sideband 2 (J=1/2): sigma+ at Delta+delta, sigma- at Delta-delta")
    print(f"  Detuning Delta:      {delta:.2f} Gamma = {delta * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"  Pol split delta:     {split:.2f} Gamma = {split * GAMMA_HZ / 1e6:.2f} MHz")
    print(f"  s0 per component:    {s0}")
    print(f"  Power per component: {detail_result['P_per_comp'] * 1e3:.1f} mW")
    print(f"  Power per arm:       {detail_result['P_per_arm'] * 1e3:.1f} mW (4 components)")
    print(f"  Total power:         {detail_result['P_total'] * 1e3:.0f} mW (3 retro-reflected axes)")
    print()
    z_eq = detail_result.get("z_eq", 0.0)
    print(f"  Equilibrium pos:     {z_eq * 1e3:.2f} mm")
    print()
    print(f"  Predicted performance:")
    print(f"    Spring constant:   {detail_result['k']:.2e} N/m")
    print(f"    Trap frequency:    2*pi x {detail_result['omega']:.0f} Hz")
    print(f"    Damping rate:      {detail_result['beta']:.0f} s^-1")
    print(f"    Scattering rate:   {detail_result['R_centre']/(2*np.pi*1e6):.3f} MHz")
    print(f"    Capture velocity:  {detail_result['v_cap']:.1f} m/s")
    print()
    print(f"  Comparison with RF MOT:")
    print(f"    {'Quantity':30s}  {'DC MOT':>12s}  {'RF MOT':>12s}  {'Expt (RF)':>12s}")
    print(f"    {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"    {'Damping beta (1/s)':30s}  {detail_result['beta']:12.0f}  {rf_result['beta']:12.0f}  {'~100':>12s}")
    print(f"    {'Trap freq (Hz)':30s}  {detail_result['omega']:12.0f}  {rf_result['omega']:12.0f}  {'~45':>12s}")
    print(f"    {'Spring const (N/m)':30s}  {detail_result['k']:12.2e}  {rf_result['k']:12.2e}  {'--':>12s}")
    print(f"    {'Capture velocity (m/s)':30s}  {detail_result['v_cap']:12.1f}  {rf_result['v_cap']:12.1f}  {'~10':>12s}")
    print(f"    {'Scattering rate (MHz)':30s}  {detail_result['R_centre']/(2*np.pi*1e6):12.3f}  {rf_result['R_centre']/(2*np.pi*1e6):12.3f}  {'--':>12s}")
    print()

    # Ratio comparison
    if rf_result["k"] > 0 and detail_result["k"] > 0:
        print(f"  DC/RF ratios:")
        print(f"    k ratio:       {detail_result['k']/rf_result['k']:.2f}")
        print(f"    omega ratio:   {detail_result['omega']/rf_result['omega']:.2f}" if rf_result['omega'] > 0 else "    omega ratio:   N/A")
        if rf_result['beta'] > 0:
            print(f"    beta ratio:    {detail_result['beta']/rf_result['beta']:.2f}")
        if rf_result['v_cap'] > 0:
            print(f"    v_cap ratio:   {detail_result['v_cap']/rf_result['v_cap']:.2f}")

    print(f"\n  {'='*60}")
    print(f"  NOTE: These predictions use the rate-equation model with")
    print(f"  Julia-validated molecular data. The DC MOT force depends")
    print(f"  critically on the polarization split; experimental tuning")
    print(f"  of delta around the optimal value is recommended.")
    print(f"  {'='*60}")


# ======================================================================
#  Main
# ======================================================================

def main():
    """Run the complete production DC MOT simulation."""

    print("=" * 70)
    print("  SrOH DC MOT PRODUCTION SIMULATION")
    print("  Using molmot package with Julia-validated molecular data")
    print("=" * 70)

    t_start = time.time()

    # ------------------------------------------------------------------
    #  Load molecular data
    # ------------------------------------------------------------------
    print("\n  Loading molecular data from:", DATA_DIR)
    from molmot.molecules.sroh import load_sroh_from_julia
    mol_data = load_sroh_from_julia(DATA_DIR, verbose=True)

    # ------------------------------------------------------------------
    #  A. RF MOT Validation
    # ------------------------------------------------------------------
    rf_result = run_rf_validation(mol_data)

    # ------------------------------------------------------------------
    #  B. DC MOT Parameter Optimization
    # ------------------------------------------------------------------
    opt_result = run_dc_optimization(mol_data)

    # ------------------------------------------------------------------
    #  C. Detailed Analysis
    # ------------------------------------------------------------------
    detail_result = run_detailed_analysis(mol_data, opt_result)

    # ------------------------------------------------------------------
    #  D. Generate Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  D. GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 70)

    plot_rf_validation(rf_result, OUT_DIR)
    plot_dc_optimization(opt_result, OUT_DIR)
    plot_dc_forces(detail_result, opt_result, OUT_DIR)
    plot_dc_trajectories(detail_result, opt_result, OUT_DIR)
    plot_summary(rf_result, opt_result, detail_result, OUT_DIR)

    # ------------------------------------------------------------------
    #  E. Experimental Recommendation
    # ------------------------------------------------------------------
    print_recommendation(rf_result, opt_result, detail_result)

    # ------------------------------------------------------------------
    #  Timing
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n  Total simulation time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print("  All outputs saved to:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
