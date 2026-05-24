#!/usr/bin/env python3
"""
Batch trajectory scaling benchmark (CPU JIT sequential vs GPU parallel).

Each particle integrates ``t_max / dt`` Euler steps of a 1D MOT trajectory
with the same DC-MOT configuration as ``trajectory_cpu_vs_gpu.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.mot.simulator import DCMOTSimulator
from molmot.propagation.trajectories_jit import simulate_trajectory_jit
from molmot_cuda import simulate_trajectories_cuda, CUDAMolData
import cupy as cp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", type=float, default=0.03)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--N", type=int, nargs="+",
                        default=[10, 100, 500, 1000, 2000, 5000, 10000])
    parser.add_argument("--cpu-max-N", type=int, default=500,
                        help="CPU sequential runs only for N <= this; "
                             "larger N is extrapolated from per-trajectory time.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mol = load_sroh_from_julia("julia_sim", verbose=False)
    sim = DCMOTSimulator(mol, delta_Gamma=-0.20, split_Gamma=0.70,
                         s0=1.0, B_gradient=0.16)
    cuda_data = CUDAMolData(mol, sim._beams, sim.Gamma_eff_factor)

    # warm-up
    _ = simulate_trajectory_jit(sim, 0.0, 0.0, t_max=1e-5, dt=args.dt)
    _ = simulate_trajectories_cuda(
        sim, np.array([0.0]), np.array([0.0]), t_max=1e-5, dt=args.dt,
        cuda_data=cuda_data)
    cp.cuda.Stream.null.synchronize()

    n_steps = int(args.t_max / args.dt)
    print("=" * 80)
    print(f"  Batch trajectory scaling, {n_steps} steps each (t_max={args.t_max} s)")
    print("=" * 80)
    print(f"  {'N':>7} {'CPU JIT (s)':>14} {'CPU mode':>12} "
          f"{'CUDA (s)':>12} {'speed-up':>10}")
    print("  " + "-" * 64)

    rng = np.random.default_rng(args.seed)
    results = []
    # Pre-measure per-trajectory JIT time for extrapolation
    per_traj_jit = None

    for N in args.N:
        z0 = rng.normal(0, 2e-3, N)
        v0 = rng.normal(0, 1.0, N)

        if N <= args.cpu_max_N:
            t0 = time.perf_counter()
            for i in range(N):
                simulate_trajectory_jit(sim, z0[i], v0[i],
                                        t_max=args.t_max, dt=args.dt)
            t_jit = time.perf_counter() - t0
            mode = "measured"
            per_traj_jit = t_jit / N
        else:
            if per_traj_jit is None:
                # fallback: use 50 samples
                t0 = time.perf_counter()
                for i in range(50):
                    simulate_trajectory_jit(sim, z0[i], v0[i],
                                            t_max=args.t_max, dt=args.dt)
                per_traj_jit = (time.perf_counter() - t0) / 50
            t_jit = per_traj_jit * N
            mode = "extrap."

        t0 = time.perf_counter()
        simulate_trajectories_cuda(
            sim, z0, v0, t_max=args.t_max, dt=args.dt,
            cuda_data=cuda_data, save_full=False)
        cp.cuda.Stream.null.synchronize()
        t_gpu = time.perf_counter() - t0

        spd = t_jit / max(t_gpu, 1e-12)
        results.append((N, t_jit, mode, t_gpu, spd))
        print(f"  {N:>7d} {t_jit:>14.3f} {mode:>12} "
              f"{t_gpu:>12.3f} {spd:>9.2f}x")

    out = os.path.join(os.path.dirname(__file__),
                       "trajectory_batch_scaling.npz")
    np.savez(out,
             N=np.array([r[0] for r in results]),
             t_jit=np.array([r[1] for r in results]),
             t_jit_mode=np.array([r[2] for r in results]),
             t_gpu=np.array([r[3] for r in results]),
             speedup=np.array([r[4] for r in results]),
             t_max=args.t_max, dt=args.dt)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
