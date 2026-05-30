#!/usr/bin/env python3
"""
SrOH DC red MOT -- full SSE ensemble (GPU) reproduction of Christian's notebook
===============================================================================

This is the *quantum* counterpart to ``examples/sroh_dc_redmot_notebook.py``.
Where that script used the rate-equation 3D MOT solver (= ensemble mean), this
one runs the **full stochastic Schrodinger equation** (MCWF / quantum jumps),
the same method as Christian Hallas's notebook

    QuantumSimulations.jl/examples/SrOH_DC_redMOT/SrOH_DC_redMOT.ipynb

It launches an ensemble of independent trajectories in parallel on the GPU via
``molmot_cuda.stochastic_cuda`` and reports the mean number of photons
scattered and the capture fraction, to compare against the notebook's reported
mean of ~115,165 photons over a 60 ms diagonal-approach trajectory.

Why GPU
-------
A single 60 ms SSE trajectory needs ~2.4e8 RK4 steps (the time step must
resolve the laser dynamics).  On a CPU in pure Python that is ~75 hours *per
trajectory*; an ensemble is hopeless.  The GPU runs one trajectory per CUDA
thread-block, all in parallel, which is exactly what this problem needs (and
what Christian's 110-worker cluster does for the Julia code).

NOTE: requires a CUDA GPU + CuPy (`pip install -e ".[cuda]"`).  Without a GPU
this script prints the exact command to run on a GPU machine and (optionally)
runs a tiny CPU sanity check of the problem setup.

Matched configuration (identical to examples/sroh_dc_redmot_notebook.py):
  4 channels, detunings [-9.0, -6.2, -19.8, -2.9] MHz, sigma+/sigma-,
  states[end]-states[1] / states[end]-states[10] frequency references,
  I_sat = pi*h*c*Gamma/(3*lambda^3), 50 mW/beam, 10 mm beam radius,
  B' = 8.8 G/cm, r0 = [25,25,0]/sqrt2 mm, v0 = [-11,-11,0]/sqrt2 m/s.
"""

import argparse
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "julia_sim")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, "..", "julia_sim")

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.stochastic import SSEProblem, SSESolver
from molmot.constants import h, c

# ---- notebook parameters ----
LAMBDA_M       = 687e-9
BEAM_RADIUS_M  = 10e-3
TOTAL_POWER_W  = 50e-3
DETUNINGS_MHZ  = np.array([-9.0, -6.2, -19.8, -2.9])
POWER_FRAC     = np.array([0.0095, 0.0195, 0.4095, 0.0615])
POLS_Q         = [2, 0, 2, 0]
B_GRAD_TM      = 8.8 / 100.0

R0 = np.array([25e-3, 25e-3, 0.0]) / np.sqrt(2)
V0 = np.array([-11.0, -11.0, 0.0]) / np.sqrt(2)
R_ESCAPE = 20e-3
NOTEBOOK_MEAN_PHOTONS = 115165


def build_problem(mol):
    """Build the SSEProblem for the notebook's SrOH DC red MOT scheme."""
    E = mol.energies
    Ee = E[mol.n_states - 1]
    base = [Ee - E[0], Ee - E[0], Ee - E[9], Ee - E[9]]
    freqs = [base[i] + DETUNINGS_MHZ[i] * 1e6 for i in range(4)]

    Isat = np.pi * h * c * mol.Gamma / (3 * LAMBDA_M ** 3)
    # Each of the 4 channels appears twice in the notebook's 8-component list;
    # the two copies (independent random phases) average to ~2x intensity, so
    # the single-channel saturation is doubled here.
    s0 = 2.0 * POWER_FRAC * TOTAL_POWER_W * (2.0 / (np.pi * BEAM_RADIUS_M ** 2)) / Isat

    beam_pairs = [
        dict(freq=freqs[i], q_fwd=POLS_Q[i], q_bwd=2 - POLS_Q[i], s0=float(s0[i]))
        for i in range(4)
    ]
    prob = SSEProblem(mol, beam_pairs, B_gradient=B_GRAD_TM,
                      beam_radius=BEAM_RADIUS_M, add_spontaneous_kick=True)
    return prob, freqs, s0


def summarize(results, t_max):
    n = len(results)
    photons = np.array([r.photons_scattered for r in results], dtype=float)
    r_final = np.array([np.linalg.norm(r.positions[-1]) for r in results])
    captured = r_final < R_ESCAPE
    print("\n" + "=" * 72)
    print("  SSE ENSEMBLE RESULTS")
    print("=" * 72)
    print(f"    trajectories            : {n}")
    print(f"    mean photons scattered  : {photons.mean():,.0f}  "
          f"(std {photons.std():,.0f})")
    print(f"    notebook mean (target)  : {NOTEBOOK_MEAN_PHOTONS:,.0f}")
    if photons.mean() > 0:
        print(f"    ratio SSE / notebook    : {photons.mean()/NOTEBOOK_MEAN_PHOTONS:.2f}")
    print(f"    capture fraction        : {captured.mean()*100:.0f}%  "
          f"({captured.sum()}/{n})")
    print(f"    mean |r_final|          : {r_final.mean()*1e3:.2f} mm")
    print("=" * 72)


def run_gpu(prob, n_particles, t_max, dt, save_every, seed):
    from molmot_cuda.stochastic_cuda import SSESolverCUDA, gpu_info
    print("\n  GPU:")
    gpu_info()
    solver = SSESolverCUDA(prob)
    results = solver.run_ensemble(
        n_particles=n_particles,
        r0_sampler=lambda: R0.copy(),
        v0_sampler=lambda: V0.copy(),
        t_max=t_max, dt=dt, save_every=save_every,
        rng_seed=seed, r_escape=R_ESCAPE, renorm_interval=2000,
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=110,
                    help="number of ensemble trajectories (notebook used 110)")
    ap.add_argument("--t-max", type=float, default=60e-3, help="sim time (s)")
    ap.add_argument("--dt", type=float, default=None,
                    help="time step (s); default 0.01/Gamma")
    ap.add_argument("--save-every", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu-demo", action="store_true",
                    help="if no GPU, run a tiny CPU sanity check of the setup")
    args = ap.parse_args()

    print("=" * 72)
    print("  SrOH DC red MOT -- full SSE ensemble (GPU) -- notebook reproduction")
    print("=" * 72)

    mol = load_sroh_from_julia(DATA_DIR, verbose=False)
    prob, freqs, s0 = build_problem(mol)
    print(f"\n  {mol.n_ground}+{mol.n_excited} states, lambda={mol.wavelength*1e9:.0f} nm, "
          f"Gamma/2pi={mol.Gamma/2/np.pi/1e6:.1f} MHz")
    print(f"  4 channels, detunings {DETUNINGS_MHZ} MHz, s0={np.round(s0,3)}, "
          f"B'=8.8 G/cm")
    print(f"  ensemble: n={args.n}, t_max={args.t_max*1e3:.0f} ms, "
          f"r0={np.round(R0*1e3,1)} mm, v0={np.round(V0,2)} m/s")

    try:
        from molmot_cuda.stochastic_cuda import HAS_CUPY
    except Exception:
        HAS_CUPY = False

    if HAS_CUPY:
        results = run_gpu(prob, args.n, args.t_max, args.dt, args.save_every, args.seed)
        summarize(results, args.t_max)
    else:
        print("\n  [no CUDA GPU / CuPy here]")
        print("  Run this on a GPU machine with:")
        print("      pip install -e \".[cuda]\"")
        print(f"      python examples/{os.path.basename(__file__)} --n {args.n}")
        if args.cpu_demo:
            print("\n  CPU sanity check: 2 short trajectories at the trap centre")
            print("  (full 60 ms on CPU is ~75 h/traj -- demo only, NOT the notebook run)")
            solver = SSESolver(prob)
            for s in (1, 2):
                res = solver.run(np.array([0.5e-3, 0.5e-3, 0.0]),
                                 np.array([1.0, 0.0, 0.0]),
                                 t_max=6e-6, save_every=200, rng_seed=s)
                rate = res.photons_scattered / 6e-6
                print(f"    seed {s}: {res.photons_scattered} photons in 6 us "
                      f"-> {rate:.2e} /s  (expect ~2e6/s)")
            print("  Setup OK -- ready for the GPU ensemble run above.")


if __name__ == "__main__":
    main()
