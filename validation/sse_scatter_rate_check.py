#!/usr/bin/env python3
"""
SSE scattering-rate cross-check (regression guard for the quantum-jump bug)
===========================================================================

Background
----------
molmot's stochastic Schrodinger equation (SSE / MCWF) solver computes
single-molecule quantum trajectories with quantum jumps.  The jump is
triggered when the (un-normalized) survival probability drops below a
random threshold:

    jump when  dp = 1 - ||psi||^2  >  threshold

The threshold MUST be drawn uniformly on (0, 1) (the "direct" MCWF method),
because dp is bounded in [0, 1].  A previous version drew

    threshold = -ln(r),  r ~ U(0,1)        # range (0, +inf)

which is the *waiting-time* convention and must instead be compared against
-ln(||psi||^2).  Mixing the two made the jump condition unreachable whenever
-ln(r) > 1 (probability 1/e ~ 37%): those trajectories never jumped, never
scattered a photon, and silently froze -- driving the ensemble scattering
rate ~100-200x too low.

This script places a molecule at the trap centre (full laser intensity) of
the SrOH DC red MOT and measures the photon scattering rate over a short
window, comparing the SSE against the (independent) rate-equation solver.
With the fix, the two agree to ~Poisson noise.

    rate-eq reference (this configuration): ~3.4e6 /s
    SSE (fixed):                            ~1.8e6 /s  (lower: SSE captures
                                            coherent dark states; rate eq does not)
    notebook mean (Hallas SrOH_DC_redMOT):  ~1.9e6 /s

Run (CPU, ~3 min):
    python validation/sse_scatter_rate_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.stochastic import SSEProblem, SSESolver
from molmot.mot.simulator_3d import MOTSimulator3D, Beam3D
from molmot.constants import h, c

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia_sim")

# Notebook SrOH DC red MOT parameters (4 unique channels)
DETUNINGS_MHZ = np.array([-9.0, -6.2, -19.8, -2.9])
POWER_FRAC    = np.array([0.0095, 0.0195, 0.4095, 0.0615])
POLS_Q        = [2, 0, 2, 0]
B_GRAD_TM     = 0.088
BEAM_RADIUS_M = 10e-3
LAMBDA_M      = 687e-9


def build(mol):
    E = mol.energies
    Ee = E[mol.n_states - 1]
    base = [Ee - E[0], Ee - E[0], Ee - E[9], Ee - E[9]]
    freqs = [base[i] + DETUNINGS_MHZ[i] * 1e6 for i in range(4)]
    Isat = np.pi * h * c * mol.Gamma / (3 * LAMBDA_M ** 3)
    s0 = 2 * POWER_FRAC * 50e-3 * (2 / (np.pi * BEAM_RADIUS_M ** 2)) / Isat
    return freqs, s0


def main():
    mol = load_sroh_from_julia(JULIA_DIR, verbose=False)
    freqs, s0 = build(mol)

    r0 = np.array([0.5e-3, 0.5e-3, 0.0])   # near trap centre, inside beams
    v0 = np.array([1.0, 0.0, 0.0])
    t_max = 12e-6

    # --- rate-equation reference at the same point ---
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    beams3d = []
    for i in range(4):
        for j, d in enumerate(dirs):
            q = POLS_Q[i] if j % 2 == 0 else 2 - POLS_Q[i]
            beams3d.append(Beam3D(d, freqs[i], q, s0[i], BEAM_RADIUS_M))
    sim = MOTSimulator3D(mol, beams3d, B_gradient=B_GRAD_TM, beam_radius=BEAM_RADIUS_M)
    _, R_rate_eq, _ = sim.force_3d(r0, v0)

    # --- SSE over several seeds ---
    bp = [dict(freq=freqs[i], q_fwd=POLS_Q[i], q_bwd=2 - POLS_Q[i], s0=float(s0[i]))
          for i in range(4)]
    prob = SSEProblem(mol, bp, B_gradient=B_GRAD_TM, beam_radius=BEAM_RADIUS_M)
    solver = SSESolver(prob)

    print(f"rate-equation scattering rate at centre: {R_rate_eq:.3e} /s\n")
    print(f"SSE trajectories (r0={r0*1e3} mm, v0={v0} m/s, t_max={t_max*1e6:.0f} us):")
    rates = []
    n_frozen = 0
    for seed in range(1, 7):
        res = solver.run(r0, v0, t_max=t_max, save_every=200, rng_seed=seed)
        rate = res.photons_scattered / t_max
        rates.append(rate)
        if res.photons_scattered == 0:
            n_frozen += 1
        print(f"  seed {seed}: {res.photons_scattered:3d} photons -> {rate:.3e} /s")

    mean_rate = float(np.mean(rates))
    print(f"\nSSE mean scattering rate: {mean_rate:.3e} /s")
    print(f"rate-eq / SSE ratio:      {R_rate_eq / mean_rate:.2f}")
    print(f"frozen (0-photon) trajectories: {n_frozen}/6")

    # Regression assertions: the SSE rate must be the same order as the
    # rate-equation rate, and not all trajectories should freeze.
    assert mean_rate > 0.2 * R_rate_eq, (
        f"SSE scattering rate {mean_rate:.2e}/s is far below the rate-equation "
        f"reference {R_rate_eq:.2e}/s -- the quantum-jump threshold bug may have "
        f"regressed (trajectories freezing).")
    assert n_frozen <= 1, (
        f"{n_frozen}/6 SSE trajectories scattered zero photons -- the jump "
        f"threshold bug may have regressed.")
    print("\nOK: SSE scattering rate is consistent with the rate equations.")


if __name__ == "__main__":
    main()
