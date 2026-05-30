#!/usr/bin/env python3
"""
SrOH DC red MOT -- molmot port of Christian Hallas's notebook
=============================================================

Reproduces the *fixed-parameter test run* from

    QuantumSimulations.jl/examples/SrOH_DC_redMOT/SrOH_DC_redMOT.ipynb

using the ``molmot`` package, and compares the result against the
notebook's reported value (mean ~115,165 photons scattered over a
60 ms diagonal-approach trajectory).

What is matched exactly
-----------------------
* Molecule + molecular data: the same Julia-generated SrOH data
  (12 ground + 4 excited states, lambda = 687 nm, Gamma = 2*pi*6.4 MHz,
  m = 105 amu) loaded from ``julia_sim/``.
* Laser scheme: 8 frequency components = 4 unique (detuning, polarisation)
  channels, each duplicated.  Detunings [-9.0, -6.2, -19.8, -2.9] MHz;
  channels 1,2 address the lower ground manifold (states[end]-states[1]),
  channels 3,4 the upper manifold (states[end]-states[10]); polarisations
  [sigma+, sigma-, sigma+, sigma-].
* Power -> saturation: per-component fractions [0.0095, 0.0195, 0.4095,
  0.0615] of 50 mW per beam, with I_sat = pi*h*c*Gamma/(3*lambda^3) and
  I = P * 2/(pi*w^2), w = 10 mm beam radius.  Each channel appears twice,
  so its rate-equation saturation is doubled.
* Magnetic field: anti-Helmholtz gradient B' = 8.8 G/cm.
* Initial conditions: r0 = [25, 25, 0]/sqrt(2) mm, v0 = [-11, -11, 0]/sqrt(2)
  m/s (the diagonal approach from the notebook's test cell), 60 ms window,
  escape boundary |r| = 20 mm.

The one fundamental difference
------------------------------
The notebook integrates the *full stochastic Schrodinger equation* (quantum
jump / MCWF) in the 3D standing-wave field, averaged over many trajectories
with random beam phases.  molmot's 3D MOT solver uses *rate equations*
(deterministic scattering rate).  A single rate-equation trajectory therefore
corresponds to the *ensemble mean* of the notebook's SSE trajectories -- which
is exactly the quantity the notebook reports (the mean photon count).  We
compare that mean photon number and the capture outcome.

Run from the repo root or from examples/:

    python examples/sroh_dc_redmot_notebook.py
"""

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Locate the Julia reference data (works from repo root or examples/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "julia_sim")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, "..", "julia_sim")

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.mot.simulator_3d import MOTSimulator3D, Beam3D
from molmot.constants import h, c

# ---------------------------------------------------------------------------
# Notebook parameters (cells 5, 10, 12, 14, 17 of SrOH_DC_redMOT.ipynb)
# ---------------------------------------------------------------------------
LAMBDA_M       = 687e-9
BEAM_RADIUS_M  = 10e-3
TOTAL_POWER_W  = 50e-3                                   # per beam
DETUNINGS_MHZ  = np.array([-9.0, -6.2, -19.8, -2.9])    # 4 unique channels
POWER_FRAC     = np.array([0.0095, 0.0195, 0.4095, 0.0615])
POLS_Q         = [2, 0, 2, 0]                            # sigma+, sigma-, sigma+, sigma-
B_GRAD_GCM     = 8.8                                     # anti-Helmholtz gradient
B_GRAD_TM      = B_GRAD_GCM / 100.0                      # G/cm -> T/m

R0 = np.array([25e-3, 25e-3, 0.0]) / np.sqrt(2)         # m
V0 = np.array([-11.0, -11.0, 0.0]) / np.sqrt(2)         # m/s
T_MAX   = 60e-3
DT      = 2e-6
R_ESCAPE = 20e-3
T_MIN_ESCAPE = 5e-3                                      # don't terminate before 5 ms

NOTEBOOK_MEAN_PHOTONS = 115165


def build_simulator(mol):
    """Build the molmot 3D rate-equation simulator for the notebook scheme."""
    Gamma = mol.Gamma
    E = mol.energies
    E_excited_last = E[mol.n_states - 1]
    # Notebook frequency references: channels 1,2 -> states[end]-states[1];
    # channels 3,4 -> states[end]-states[10].  (0-indexed: E[0] and E[9].)
    E_manifold = [E_excited_last - E[0], E_excited_last - E[0],
                  E_excited_last - E[9], E_excited_last - E[9]]
    freqs = [E_manifold[i] + DETUNINGS_MHZ[i] * 1e6 for i in range(4)]

    # Saturation per channel.  Each channel is duplicated in the 8-component
    # list, so the rate-equation saturation is 2x the single-component value.
    I_sat = np.pi * h * c * Gamma / (3 * LAMBDA_M ** 3)
    power_W = 2.0 * POWER_FRAC * TOTAL_POWER_W
    intensity = power_W * (2.0 / (np.pi * BEAM_RADIUS_M ** 2))
    s0 = intensity / I_sat

    # 6-beam retro-reflected geometry: forward beams (+x,+y,+z) carry the
    # channel polarisation; retro beams (-x,-y,-z) carry the flipped helicity.
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                  (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    beams = []
    for i in range(4):
        q_fwd = POLS_Q[i]
        q_bwd = 2 - q_fwd  # sigma+ <-> sigma-
        for j, d in enumerate(directions):
            q = q_fwd if j % 2 == 0 else q_bwd
            beams.append(Beam3D(d, freqs[i], q, s0[i], BEAM_RADIUS_M))

    sim = MOTSimulator3D(mol, beams, B_gradient=B_GRAD_TM,
                         beam_radius=BEAM_RADIUS_M)
    return sim, freqs, s0, I_sat


def integrate(sim, r0, v0, t_max, dt, r_escape, t_min_escape):
    """Euler integration that also accumulates scattered photons (int R dt)."""
    mass = sim.mass
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    n_steps = int(t_max / dt)

    ts = np.zeros(n_steps)
    rs = np.zeros((n_steps, 3))
    vs = np.zeros((n_steps, 3))
    n_photons = 0.0
    captured = True
    last = n_steps

    for j in range(n_steps):
        F, R, _ = sim.force_3d(r, v)
        n_photons += R * dt
        v = v + (F / mass) * dt
        r = r + v * dt
        ts[j] = j * dt
        rs[j] = r
        vs[j] = v
        if np.linalg.norm(r) >= r_escape and (j * dt) > t_min_escape:
            captured = False
            last = j + 1
            break

    if last == n_steps:
        captured = np.linalg.norm(r) < r_escape

    return ts[:last], rs[:last], vs[:last], n_photons, captured


def main():
    print("=" * 72)
    print("  SrOH DC red MOT -- molmot port of Christian's notebook test run")
    print("=" * 72)

    print(f"\n  Loading SrOH data from: {DATA_DIR}")
    mol = load_sroh_from_julia(DATA_DIR, verbose=False)
    print(f"    {mol.n_ground} ground + {mol.n_excited} excited states, "
          f"lambda = {mol.wavelength*1e9:.0f} nm, "
          f"Gamma/2pi = {mol.Gamma/2/np.pi/1e6:.1f} MHz, "
          f"m = {mol.mass/1.66053906660e-27:.0f} amu")

    sim, freqs, s0, I_sat = build_simulator(mol)

    print(f"\n  Laser configuration (4 unique channels x 6 beams):")
    print(f"    I_sat = {I_sat*1e-1:.2f} mW/cm^2")
    for i in range(4):
        pol = ["sigma-", "pi", "sigma+"][POLS_Q[i]]
        print(f"    ch{i+1}: detuning {DETUNINGS_MHZ[i]:+6.1f} MHz, "
              f"s0 = {s0[i]:7.3f}, pol = {pol}")
    print(f"    B' = {B_GRAD_GCM:.1f} G/cm,  beam radius = {BEAM_RADIUS_M*1e3:.0f} mm")

    print(f"\n  Trajectory:")
    print(f"    r0 = {np.round(R0*1e3, 2)} mm  (|r0| = {np.linalg.norm(R0)*1e3:.1f} mm)")
    print(f"    v0 = {np.round(V0, 3)} m/s  (|v0| = {np.linalg.norm(V0):.2f} m/s)")
    print(f"    t_max = {T_MAX*1e3:.0f} ms, dt = {DT*1e6:.1f} us "
          f"({int(T_MAX/DT)} steps)")

    print(f"\n  Integrating (rate-equation = ensemble mean of the SSE runs)...")
    t0 = time.time()
    ts, rs, vs, n_photons, captured = integrate(
        sim, R0, V0, T_MAX, DT, R_ESCAPE, T_MIN_ESCAPE)
    wall = time.time() - t0

    r_final = rs[-1]
    v_final = vs[-1]
    print(f"    done in {wall:.1f} s ({len(ts)} steps integrated)")

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"    final position : {np.round(r_final*1e3, 2)} mm  "
          f"(|r| = {np.linalg.norm(r_final)*1e3:.2f} mm)")
    print(f"    final velocity : {np.round(v_final, 3)} m/s  "
          f"(|v| = {np.linalg.norm(v_final):.3f} m/s)")
    print(f"    captured       : {captured}")
    print(f"    photons (molmot, rate eq) : {n_photons:>12,.0f}")
    print(f"    photons (notebook mean)   : {NOTEBOOK_MEAN_PHOTONS:>12,.0f}")
    ratio = n_photons / NOTEBOOK_MEAN_PHOTONS
    print(f"    ratio molmot / notebook   : {ratio:>12.2f}")
    print("=" * 72)

    # Save a plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
        for k, lbl in enumerate(["x", "y", "z"]):
            ax1.plot(ts * 1e3, rs[:, k] * 1e3, label=lbl)
            ax2.plot(ts * 1e3, vs[:, k], label=f"v{lbl}")
        ax1.axhline(R_ESCAPE * 1e3, ls="--", color="grey")
        ax1.axhline(-R_ESCAPE * 1e3, ls="--", color="grey")
        ax1.set_xlabel("t (ms)"); ax1.set_ylabel("position (mm)")
        ax1.set_title("SrOH DC red MOT trajectory (molmot rate-eq)")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.set_xlabel("t (ms)"); ax2.set_ylabel("velocity (m/s)")
        ax2.axhline(0, ls=":", color="grey")
        ax2.legend(); ax2.grid(alpha=0.3)
        out = os.path.join(BASE_DIR, "sroh_dc_redmot_notebook.png")
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\n  Saved trajectory plot: {out}")
    except Exception as e:
        print(f"\n  (plot skipped: {e})")


if __name__ == "__main__":
    main()
