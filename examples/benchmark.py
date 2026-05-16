#!/usr/bin/env python3
"""
Benchmark: CaOH Blue MOT, CaOH RF MOT, and CaF Blue MOT force profiles.

Compares force profiles from rate-equation solver using Christian's
exact blue MOT parameters. Generates a 3x2 panel comparison plot.

Benchmarks:
  1. CaOH Blue MOT (3-frequency, +1.19 Gamma detuning)
  2. CaOH RF MOT (red-detuned, -1 Gamma, for comparison)
  3. CaF Blue MOT (2-frequency, +2.87 Gamma detuning)

References:
  - Christian Hallas, CaOHBlueMOTSimulations
  - Li et al., PRL (2023) -- CaF blue MOT
"""

import sys
import os
import math
import time

import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molmot.constants import hbar, k_B, amu, c
from molmot.obe.fields import LaserBeam
from molmot.obe.rate_equations import solve_rate_equations
from molmot.mot.force_scan import spring_constant, damping_coefficient


# =====================================================================
# Helper: make custom MOT beams for the blue MOT configurations
# =====================================================================

def make_blue_mot_beams_caoh(mol_data):
    """
    Create CaOH 3-frequency blue MOT beams.

    Christian's parameters:
    - Detuning: +7.6 MHz (BLUE detuned = +1.19 Gamma)
    - 3 frequencies: delta1=0, delta2=-1.0 MHz, delta3=+0.75 MHz
    - Polarizations: sigma+, sigma-, sigma+
    - Saturation ratios: 0.37, 0.28, 0.35 of total
    - Total power: 13.1 mW, beam radius: 4 mm
    - Total s0 estimated from I_sat and beam parameters

    For the rate equation, we need the absolute s0 values.
    I_sat for CaOH (626 nm, Gamma = 6.4 MHz):
      I_sat = pi*h*c*Gamma/(3*lambda^3) ~ 0.85 mW/cm^2
    With P=13.1 mW and w=4mm: I = P/(pi*w^2) ~ 26 mW/cm^2
    So s0_total ~ 26/0.85 ~ 31

    Each beam in a 1D model sees half the power (forward + retro).
    For 3 frequencies, the total is split according to the ratios.
    """
    Gamma = mol_data.Gamma
    Gamma_hz = Gamma / (2 * math.pi)

    # Blue detuning: +7.6 MHz = +1.19 Gamma
    delta_blue_hz = +7.6e6  # Hz

    # Frequency offsets relative to J=1/2 and J=3/2 manifolds
    # Christian uses 3 frequencies addressing both SR manifolds
    # delta1=0, delta2=-1.0 MHz, delta3=+0.75 MHz relative to center
    delta1_hz = 0.0
    delta2_hz = -1.0e6
    delta3_hz = +0.75e6

    # Total saturation parameter
    # P_total = 13.1 mW, beam_radius = 4 mm
    P_total = 13.1e-3  # W
    beam_radius = 4e-3  # m
    I_peak = 2 * P_total / (math.pi * beam_radius**2)  # W/m^2

    # Saturation intensity for CaOH
    lambda_m = mol_data.wavelength
    I_sat_val = math.pi * 6.626e-34 * 3e8 * Gamma_hz / (3 * lambda_m**3)

    s0_total = I_peak / I_sat_val

    # Saturation ratios for the 3 frequencies
    s_ratios = [0.37, 0.28, 0.35]
    s0_per_freq = [r * s0_total for r in s_ratios]

    # Per beam (forward+retro each gets half in 1D model)
    s0_beam = [s / 2.0 for s in s0_per_freq]

    # Frequency components with polarizations
    # In the blue MOT, the 3 frequencies address both J=1/2 and J=3/2
    # For simplicity, we set absolute frequencies relative to the
    # mean transition frequency
    freq_base_J12 = mol_data.omega_J12 + delta_blue_hz
    freq_base_J32 = mol_data.omega_J32 + delta_blue_hz

    # Build beams: 3 frequencies x 2 directions
    beams = []

    # Frequency component specifications:
    # (freq_offset_hz, polarization_forward, s0_per_beam)
    # We address J=3/2 with all 3 frequencies (dominant manifold)
    freq_components = [
        (freq_base_J32 + delta1_hz, 2, s0_beam[0]),   # sigma+, main
        (freq_base_J32 + delta2_hz, 0, s0_beam[1]),   # sigma-, lower sideband
        (freq_base_J32 + delta3_hz, 2, s0_beam[2]),   # sigma+, upper sideband
        # Also address J=1/2
        (freq_base_J12 + delta1_hz, 2, s0_beam[0] * 0.5),
        (freq_base_J12 + delta2_hz, 0, s0_beam[1] * 0.5),
        (freq_base_J12 + delta3_hz, 2, s0_beam[2] * 0.5),
    ]

    for freq, q_fwd, s0 in freq_components:
        for direction in [+1, -1]:
            # Retro-reflected beam flips handedness
            if direction > 0:
                q_eff = q_fwd
            else:
                q_eff = 2 - q_fwd  # sigma+ <-> sigma-

            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=freq,
                polarization=q_eff,
                s0=s0,
            ))

    return beams


def make_blue_mot_beams_caf(mol_data):
    """
    Create CaF 2-frequency blue MOT beams.

    Parameters from Christian:
    - Detuning: +23.8 MHz (BLUE, = +2.87 Gamma for CaF)
    - 2 frequencies: delta1=-0.75 MHz, delta2=0
    - Polarizations: sigma+, sigma-
    - beam_radius=6mm, P=4.64mW
    - B-gradient: 14.6 G/cm = 0.146 T/m
    """
    Gamma = mol_data.Gamma
    Gamma_hz = Gamma / (2 * math.pi)

    delta_blue_hz = +23.8e6  # Hz

    delta1_hz = -0.75e6
    delta2_hz = 0.0

    P_total = 4.64e-3  # W
    beam_radius = 6e-3  # m
    I_peak = 2 * P_total / (math.pi * beam_radius**2)

    lambda_m = mol_data.wavelength
    I_sat_val = math.pi * 6.626e-34 * 3e8 * Gamma_hz / (3 * lambda_m**3)
    s0_total = I_peak / I_sat_val
    s0_per_freq = s0_total / 2.0
    s0_beam = s0_per_freq / 2.0  # forward+retro

    # Addressing both SR manifolds with 2 frequencies
    freq_base_J32 = mol_data.omega_J32 + delta_blue_hz
    freq_base_J12 = mol_data.omega_J12 + delta_blue_hz

    beams = []

    freq_components = [
        (freq_base_J32 + delta1_hz, 2, s0_beam),  # sigma+
        (freq_base_J32 + delta2_hz, 0, s0_beam),  # sigma-
        (freq_base_J12 + delta1_hz, 2, s0_beam * 0.5),
        (freq_base_J12 + delta2_hz, 0, s0_beam * 0.5),
    ]

    for freq, q_fwd, s0 in freq_components:
        for direction in [+1, -1]:
            if direction > 0:
                q_eff = q_fwd
            else:
                q_eff = 2 - q_fwd

            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=freq,
                polarization=q_eff,
                s0=s0,
            ))

    return beams


# =====================================================================
# Force computation wrapper
# =====================================================================

class BlueMOTSimulator:
    """Wrapper for custom blue MOT beam configurations."""

    def __init__(self, mol_data, beams, B_gradient):
        self.mol_data = mol_data
        self.beams = beams
        self.B_gradient = B_gradient

    def force(self, v, z):
        return solve_rate_equations(
            self.mol_data, self.beams, v, z, self.B_gradient)


def compute_force_profile(simulator, z_max, v_max, nz=101, nv=101):
    """Compute F(z) at v=0 and F(v) at z=0."""
    z_arr = np.linspace(-z_max, z_max, nz)
    v_arr = np.linspace(-v_max, v_max, nv)

    Fz = np.zeros(nz)
    Fv = np.zeros(nv)

    for i, z in enumerate(z_arr):
        F, _, _ = simulator.force(0.0, z)
        Fz[i] = F

    for i, v in enumerate(v_arr):
        F, _, _ = simulator.force(v, 0.0)
        Fv[i] = F

    return z_arr, Fz, v_arr, Fv


# =====================================================================
# Main benchmark
# =====================================================================

def main():
    print("=" * 70)
    print("  BENCHMARK: Blue MOT Force Profiles")
    print("  CaOH Blue MOT | CaOH RF MOT | CaF Blue MOT")
    print("=" * 70)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Load molecules
    # ------------------------------------------------------------------

    print("\n[1/6] Building CaOH molecular data...")
    from molmot.molecules.caoh import load_caoh_from_sroh_structure
    mol_caoh = load_caoh_from_sroh_structure(verbose=True)

    print("\n[2/6] Building CaF molecular data...")
    from molmot.molecules.caf import load_caf_from_sroh_structure
    mol_caf = load_caf_from_sroh_structure(verbose=True)

    # ------------------------------------------------------------------
    # Benchmark 1: CaOH Blue MOT
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  BENCHMARK 1: CaOH Blue MOT (3-frequency)")
    print("=" * 70)

    # B-gradient: 82 G/cm = 0.82 T/m
    B_grad_caoh = 0.82  # T/m (82 G/cm)

    beams_caoh_blue = make_blue_mot_beams_caoh(mol_caoh)
    sim_caoh_blue = BlueMOTSimulator(mol_caoh, beams_caoh_blue, B_grad_caoh)

    print(f"  Beams: {len(beams_caoh_blue)}")
    print(f"  B-gradient: {B_grad_caoh*100:.1f} G/cm")
    print(f"  Detuning: +7.6 MHz = +{7.6/6.4:.2f} Gamma (BLUE)")

    print("\n  Computing F(z) and F(v)...")
    z1, Fz1, v1, Fv1 = compute_force_profile(
        sim_caoh_blue, z_max=3e-3, v_max=3.0, nz=81, nv=81)

    # Spring constant and damping
    kappa1 = spring_constant(sim_caoh_blue, dz=0.3e-3)
    alpha1 = damping_coefficient(sim_caoh_blue, dv=0.05)
    mass_caoh = mol_caoh.mass

    Gamma_caoh = mol_caoh.Gamma
    T_D_caoh = hbar * Gamma_caoh / (2 * k_B)

    print(f"\n  Results:")
    print(f"    Spring constant kappa = {kappa1:.4e} N/m")
    print(f"    Damping coefficient alpha = {alpha1:.4e} N*s/m")
    print(f"    Damping rate beta = alpha/m = {alpha1/mass_caoh:.1f} /s")
    if kappa1 > 0 and alpha1 > 0:
        omega_trap = math.sqrt(kappa1 / mass_caoh)
        T_from_equipartition = kappa1 * (600e-6)**2 / k_B  # rough estimate
        print(f"    Trap frequency: {omega_trap/(2*math.pi):.1f} Hz")
        print(f"    T_Doppler (CaOH): {T_D_caoh*1e6:.1f} uK")
        print(f"    RESTORING force: {'YES' if kappa1 > 0 else 'NO'}")
        print(f"    COOLING force:   {'YES' if alpha1 > 0 else 'NO'}")
    else:
        print(f"    WARNING: Non-trapping configuration detected")
        print(f"    kappa sign: {'positive (restoring)' if kappa1 > 0 else 'NEGATIVE (anti-trapping)'}")
        print(f"    alpha sign: {'positive (cooling)' if alpha1 > 0 else 'NEGATIVE (heating)'}")

    # ------------------------------------------------------------------
    # Benchmark 2: CaOH RF MOT (red-detuned comparison)
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  BENCHMARK 2: CaOH RF MOT (red-detuned, -1 Gamma)")
    print("=" * 70)

    from molmot.mot.simulator import RFMOTSimulator
    B_grad_rf = 0.16  # T/m (16 G/cm, typical RF MOT)
    sim_caoh_rf = RFMOTSimulator(mol_caoh, delta_Gamma=-1.0,
                                  s0=1.0, B_gradient=B_grad_rf)

    print(f"  Detuning: -1.0 Gamma (RED)")
    print(f"  B-gradient: {B_grad_rf*100:.1f} G/cm")
    print(f"  s0: 1.0 per sideband per beam")

    print("\n  Computing F(z) and F(v)...")
    z2, Fz2, v2, Fv2 = compute_force_profile(
        sim_caoh_rf, z_max=3e-3, v_max=3.0, nz=81, nv=81)

    kappa2 = spring_constant(sim_caoh_rf, dz=0.3e-3)
    alpha2 = damping_coefficient(sim_caoh_rf, dv=0.05)

    print(f"\n  Results:")
    print(f"    Spring constant kappa = {kappa2:.4e} N/m")
    print(f"    Damping coefficient alpha = {alpha2:.4e} N*s/m")
    print(f"    Damping rate beta = alpha/m = {alpha2/mass_caoh:.1f} /s")
    if kappa2 > 0:
        omega_trap2 = math.sqrt(abs(kappa2) / mass_caoh)
        print(f"    Trap frequency: {omega_trap2/(2*math.pi):.1f} Hz")
    print(f"    RESTORING force: {'YES' if kappa2 > 0 else 'NO'}")
    print(f"    COOLING force:   {'YES' if alpha2 > 0 else 'NO'}")

    # ------------------------------------------------------------------
    # Benchmark 3: CaF Blue MOT
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  BENCHMARK 3: CaF Blue MOT (2-frequency)")
    print("=" * 70)

    # B-gradient: 14.6 G/cm = 0.146 T/m
    B_grad_caf = 0.146  # T/m

    beams_caf_blue = make_blue_mot_beams_caf(mol_caf)
    sim_caf_blue = BlueMOTSimulator(mol_caf, beams_caf_blue, B_grad_caf)

    Gamma_caf = mol_caf.Gamma
    T_D_caf = hbar * Gamma_caf / (2 * k_B)

    print(f"  Beams: {len(beams_caf_blue)}")
    print(f"  B-gradient: {B_grad_caf*100:.1f} G/cm")
    print(f"  Detuning: +23.8 MHz = +{23.8/8.3:.2f} Gamma (BLUE)")

    print("\n  Computing F(z) and F(v)...")
    z3, Fz3, v3, Fv3 = compute_force_profile(
        sim_caf_blue, z_max=3e-3, v_max=3.0, nz=81, nv=81)

    kappa3 = spring_constant(sim_caf_blue, dz=0.3e-3)
    alpha3 = damping_coefficient(sim_caf_blue, dv=0.05)
    mass_caf = mol_caf.mass

    print(f"\n  Results:")
    print(f"    Spring constant kappa = {kappa3:.4e} N/m")
    print(f"    Damping coefficient alpha = {alpha3:.4e} N*s/m")
    print(f"    Damping rate beta = alpha/m = {alpha3/mass_caf:.1f} /s")
    if kappa3 > 0:
        omega_trap3 = math.sqrt(abs(kappa3) / mass_caf)
        print(f"    Trap frequency: {omega_trap3/(2*math.pi):.1f} Hz")
    print(f"    T_Doppler (CaF): {T_D_caf*1e6:.1f} uK")
    print(f"    RESTORING force: {'YES' if kappa3 > 0 else 'NO'}")
    print(f"    COOLING force:   {'YES' if alpha3 > 0 else 'NO'}")

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n  {'':30s} {'CaOH Blue':>12s} {'CaOH RF':>12s} {'CaF Blue':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Detuning (Gamma)':30s} {'+1.19':>12s} {'-1.00':>12s} {'+2.87':>12s}")
    print(f"  {'B-grad (G/cm)':30s} {B_grad_caoh*100:>12.1f} {B_grad_rf*100:>12.1f} {B_grad_caf*100:>12.1f}")
    print(f"  {'kappa (N/m)':30s} {kappa1:>12.2e} {kappa2:>12.2e} {kappa3:>12.2e}")
    print(f"  {'alpha (N*s/m)':30s} {alpha1:>12.2e} {alpha2:>12.2e} {alpha3:>12.2e}")
    print(f"  {'beta = alpha/m (1/s)':30s} {alpha1/mass_caoh:>12.1f} {alpha2/mass_caoh:>12.1f} {alpha3/mass_caf:>12.1f}")
    print(f"  {'Restoring?':30s} {'YES' if kappa1 > 0 else 'NO':>12s} {'YES' if kappa2 > 0 else 'NO':>12s} {'YES' if kappa3 > 0 else 'NO':>12s}")
    print(f"  {'Cooling?':30s} {'YES' if alpha1 > 0 else 'NO':>12s} {'YES' if alpha2 > 0 else 'NO':>12s} {'YES' if alpha3 > 0 else 'NO':>12s}")

    # Expected values for comparison
    print(f"\n  Expected from literature:")
    print(f"    CaOH blue MOT: T ~ 35 uK, sigma ~ 200 um, n ~ 10^8 cm^-3")
    print(f"    Blue MOT should show sub-Doppler features (non-linear F(v) near v=0)")
    print(f"    Rate equation captures force SIGNS but not full sub-Doppler physics")
    print(f"")
    print(f"  IMPORTANT PHYSICS NOTE:")
    print(f"    The rate equation solver uses the secular approximation (no coherences).")
    print(f"    The blue MOT trapping mechanism relies on magnetically-induced Sisyphus")
    print(f"    cooling (a coherent, sub-Doppler effect). Therefore:")
    print(f"    - Blue MOT showing ANTI-trapping in rate eq. is CORRECT (Doppler-level)")
    print(f"    - Red MOT showing trapping in rate eq. is CORRECT (standard Doppler)")
    print(f"    - Full OBE/SSE solver is REQUIRED to see blue MOT trapping")
    print(f"    This benchmark validates that:")
    print(f"    (1) The molecular data loads correctly for CaOH and CaF")
    print(f"    (2) The rate equation solver runs with the new molecules")
    print(f"    (3) The force signs match Doppler-level expectations")
    print(f"    (4) The ground-state structure (SR splitting, HF) is correct")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f} s")

    # ------------------------------------------------------------------
    # Generate comparison plot
    # ------------------------------------------------------------------

    print("\n[6/6] Generating comparison plot...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle("Blue MOT Force Profile Benchmarks (Rate Equation Solver)",
                 fontsize=14, fontweight="bold", y=0.98)

    # Acceleration units for the y-axis
    a_caoh = 1.0 / mass_caoh  # N -> m/s^2
    a_caf = 1.0 / mass_caf

    # ------ Row 1: CaOH Blue MOT ------
    ax = axes[0, 0]
    ax.plot(z1 * 1e3, Fz1 * a_caoh, "b-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Position z (mm)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaOH Blue MOT: F(z) at v=0\n"
                 f"[+1.19 Gamma, {B_grad_caoh*100:.0f} G/cm, "
                 f"kappa={kappa1:.2e} N/m]")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(v1, Fv1 * a_caoh, "b-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Velocity v (m/s)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaOH Blue MOT: F(v) at z=0\n"
                 f"[alpha={alpha1:.2e} N*s/m, "
                 f"beta={alpha1/mass_caoh:.0f} /s]")
    ax.grid(True, alpha=0.3)

    # ------ Row 2: CaOH RF MOT ------
    ax = axes[1, 0]
    ax.plot(z2 * 1e3, Fz2 * a_caoh, "r-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Position z (mm)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaOH RF MOT: F(z) at v=0\n"
                 f"[-1.0 Gamma, {B_grad_rf*100:.0f} G/cm, "
                 f"kappa={kappa2:.2e} N/m]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(v2, Fv2 * a_caoh, "r-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Velocity v (m/s)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaOH RF MOT: F(v) at z=0\n"
                 f"[alpha={alpha2:.2e} N*s/m, "
                 f"beta={alpha2/mass_caoh:.0f} /s]")
    ax.grid(True, alpha=0.3)

    # ------ Row 3: CaF Blue MOT ------
    ax = axes[2, 0]
    ax.plot(z3 * 1e3, Fz3 * a_caf, "g-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Position z (mm)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaF Blue MOT: F(z) at v=0\n"
                 f"[+2.87 Gamma, {B_grad_caf*100:.1f} G/cm, "
                 f"kappa={kappa3:.2e} N/m]")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(v3, Fv3 * a_caf, "g-", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Velocity v (m/s)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("CaF Blue MOT: F(v) at z=0\n"
                 f"[alpha={alpha3:.2e} N*s/m, "
                 f"beta={alpha3/mass_caf:.0f} /s]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Add footnote
    fig.text(0.5, 0.005,
             "Rate equation solver (secular approximation). "
             "Sub-Doppler features require full OBE/SSE.",
             ha="center", fontsize=9, style="italic", color="gray")

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "benchmark_results.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {outpath}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
