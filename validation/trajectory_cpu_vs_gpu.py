#!/usr/bin/env python3
"""
Trajectory validation: molmot CPU (pure Python) vs CPU (numba JIT) vs CUDA.

Step 2 of the GPU validation plan -- compare 4 reference single-particle
trajectories (matching Julia's run_dc_mot.jl) across all three backends.
Step 3 follows: 100-particle batch trajectory timing CPU sequential vs
GPU parallel.

DC-MOT configuration (matches the prompt):
    DCMOTSimulator(mol, delta_Gamma=-0.20, split_Gamma=0.70,
                   s0=1.0, B_gradient=0.16)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.fields import make_dc_mot_beams
from molmot.mot.simulator import DCMOTSimulator
from molmot.propagation.trajectories import simulate_trajectory
from molmot.propagation.trajectories_jit import simulate_trajectory_jit

import molmot_cuda
from molmot_cuda import (
    simulate_trajectory_cuda,
    simulate_trajectories_cuda,
    CUDAMolData,
)
import cupy as cp


REF_TRAJECTORIES = [
    ("trapped oscillation  (z0=3 mm, v0=0)",     3e-3,  0.0),
    ("velocity damping     (z0=0,   v0=-2)",     0.0,  -2.0),
    ("mixed                (z0=2 mm, v0=-1)",    2e-3, -1.0),
    ("escaping             (z0=0,   v0=-5)",     0.0,  -5.0),
]


def fmt_time(t):
    if t < 1e-3:
        return f"{t*1e6:7.1f} µs"
    if t < 1.0:
        return f"{t*1e3:7.2f} ms"
    return f"{t:7.3f} s "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", type=float, default=0.03,
                        help="Total simulation time (s).")
    parser.add_argument("--dt", type=float, default=1e-6,
                        help="Time step (s).")
    parser.add_argument("--n-particles", type=int, default=100,
                        help="Particles in the batch benchmark.")
    parser.add_argument("--skip-pure-python", action="store_true",
                        help="Skip the slow pure-Python single trajectories.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_steps = int(args.t_max / args.dt)
    print("=" * 86)
    print("  Trajectory validation: CPU pure-Python  vs  CPU JIT  vs  CUDA")
    print("=" * 86)
    info = molmot_cuda.get_device_info()
    print(f"  GPU      : {info['name']} (cc {info['compute_capability']})")
    print(f"  t_max    : {args.t_max:.3g} s")
    print(f"  dt       : {args.dt:.0e} s")
    print(f"  n_steps  : {n_steps}")

    mol = load_sroh_from_julia("julia_sim", verbose=False)
    sim = DCMOTSimulator(mol, delta_Gamma=-0.20, split_Gamma=0.70,
                         s0=1.0, B_gradient=0.16)
    beams = sim._beams
    n_g, n_e = mol.n_ground, mol.n_excited
    print(f"  System   : SrOH, {n_g} ground × {n_e} excited, {len(beams)} beams")
    print()

    # ── Warm up the JIT compiler and CUDA kernel ──────────────────
    _ = simulate_trajectory_jit(sim, 0.0, 0.0, t_max=1e-5, dt=args.dt)
    _ = simulate_trajectory_cuda(sim, 0.0, 0.0, t_max=1e-5, dt=args.dt)
    cp.cuda.Stream.null.synchronize()

    cuda_data = CUDAMolData(mol, beams, sim.Gamma_eff_factor)

    # ====================================================================
    # STEP 2 -- single-trajectory three-way comparison
    # ====================================================================
    print("=" * 86)
    print("  STEP 2: per-trajectory final-state comparison "
          "(t_max=%.3g s, dt=%.0e s)" % (args.t_max, args.dt))
    print("=" * 86)

    rows = []
    for name, z0, v0 in REF_TRAJECTORIES:
        # ----- CPU pure Python -----
        if args.skip_pure_python:
            t_py = float("nan")
            z_py = v_py = None
        else:
            t0 = time.perf_counter()
            t_arr, z_py_arr, v_py_arr = simulate_trajectory(
                sim, z0, v0, t_max=args.t_max, dt=args.dt)
            t_py = time.perf_counter() - t0
            z_py, v_py = z_py_arr[-1], v_py_arr[-1]

        # ----- CPU JIT -----
        t0 = time.perf_counter()
        t_arr_j, z_jit_arr, v_jit_arr = simulate_trajectory_jit(
            sim, z0, v0, t_max=args.t_max, dt=args.dt)
        t_jit = time.perf_counter() - t0
        z_jit, v_jit = z_jit_arr[-1], v_jit_arr[-1]

        # ----- GPU -----
        t0 = time.perf_counter()
        t_arr_g, z_gpu_arr, v_gpu_arr = simulate_trajectory_cuda(
            sim, z0, v0, t_max=args.t_max, dt=args.dt,
            cuda_data=cuda_data)
        cp.cuda.Stream.null.synchronize()
        t_gpu = time.perf_counter() - t0
        z_gpu, v_gpu = z_gpu_arr[-1], v_gpu_arr[-1]

        # Endpoint diffs (use JIT as reference if pure-py skipped)
        ref_z = z_jit if z_py is None else z_py
        ref_v = v_jit if v_py is None else v_py
        rms_z_pyjit = (abs(z_py - z_jit) if z_py is not None else 0.0)
        rms_z_pygpu = (abs(z_py - z_gpu) if z_py is not None else 0.0)
        rms_v_pyjit = (abs(v_py - v_jit) if v_py is not None else 0.0)
        rms_z_jitgpu = abs(z_jit - z_gpu)
        rms_v_jitgpu = abs(v_jit - v_gpu)

        # Full-trajectory RMS (JIT vs GPU)
        rms_traj_z = np.sqrt(np.mean((z_jit_arr - z_gpu_arr) ** 2))
        rms_traj_v = np.sqrt(np.mean((v_jit_arr - v_gpu_arr) ** 2))

        rows.append((name, z0, v0,
                     z_py, v_py, z_jit, v_jit, z_gpu, v_gpu,
                     t_py, t_jit, t_gpu,
                     rms_traj_z, rms_traj_v))

        print(f"\n  {name}")
        print(f"    {'backend':<12} {'final z (m)':>16} {'final v (m/s)':>17}"
              f" {'wall (this run)':>17}")
        if z_py is not None:
            print(f"    {'Python':<12} {z_py:>16.10e} {v_py:>17.10e} "
                  f"{fmt_time(t_py):>17}")
        print(f"    {'JIT':<12} {z_jit:>16.10e} {v_jit:>17.10e} "
              f"{fmt_time(t_jit):>17}")
        print(f"    {'CUDA':<12} {z_gpu:>16.10e} {v_gpu:>17.10e} "
              f"{fmt_time(t_gpu):>17}")
        if z_py is not None:
            print(f"    |Py-JIT|  Δz = {rms_z_pyjit:.2e} m   "
                  f"Δv = {rms_v_pyjit:.2e} m/s")
            print(f"    |Py-GPU|  Δz = {rms_z_pygpu:.2e} m   "
                  f"Δv = {abs(v_py - v_gpu):.2e} m/s")
        print(f"    |JIT-GPU| Δz = {rms_z_jitgpu:.2e} m   "
              f"Δv = {rms_v_jitgpu:.2e} m/s")
        print(f"    full trajectory RMS  z: {rms_traj_z:.2e} m   "
              f"v: {rms_traj_v:.2e} m/s")

    # ====================================================================
    # STEP 3 -- 100-particle batch benchmark
    # ====================================================================
    print()
    print("=" * 86)
    print(f"  STEP 3: batch of {args.n_particles} particles "
          f"(z0 ~ N(0, 2 mm), v0 ~ N(0, 1 m/s)), each {n_steps} steps")
    print("=" * 86)

    rng = np.random.default_rng(args.seed)
    z0_arr = rng.normal(0.0, 2e-3, args.n_particles)
    v0_arr = rng.normal(0.0, 1.0, args.n_particles)

    # ----- CPU sequential with JIT -----
    t0 = time.perf_counter()
    zf_jit = np.empty(args.n_particles)
    vf_jit = np.empty(args.n_particles)
    for i in range(args.n_particles):
        _, z_arr, v_arr = simulate_trajectory_jit(
            sim, z0_arr[i], v0_arr[i], t_max=args.t_max, dt=args.dt)
        zf_jit[i] = z_arr[-1]
        vf_jit[i] = v_arr[-1]
    t_jit_batch = time.perf_counter() - t0

    # ----- GPU parallel -----
    t0 = time.perf_counter()
    zf_gpu, vf_gpu, n_actual, _, _ = simulate_trajectories_cuda(
        sim, z0_arr, v0_arr, t_max=args.t_max, dt=args.dt,
        save_full=False, cuda_data=cuda_data)
    cp.cuda.Stream.null.synchronize()
    t_gpu_batch = time.perf_counter() - t0

    rms_z_batch = np.sqrt(np.mean((zf_jit - zf_gpu) ** 2))
    rms_v_batch = np.sqrt(np.mean((vf_jit - vf_gpu) ** 2))
    max_z_batch = np.max(np.abs(zf_jit - zf_gpu))
    max_v_batch = np.max(np.abs(vf_jit - vf_gpu))

    n_trapped = np.sum(n_actual == n_steps)
    print(f"  CPU JIT (sequential) : {fmt_time(t_jit_batch)} "
          f"  ({args.n_particles} × {n_steps} = "
          f"{args.n_particles * n_steps:,} steps)")
    print(f"  CUDA   (parallel)    : {fmt_time(t_gpu_batch)}")
    print(f"  Speed-up GPU vs CPU  : {t_jit_batch / max(t_gpu_batch, 1e-12):.1f}×")
    print(f"  {n_trapped}/{args.n_particles} stayed inside z_escape.")
    print(f"  JIT vs GPU final state: "
          f"RMS Δz = {rms_z_batch:.2e} m, max |Δz| = {max_z_batch:.2e} m   "
          f"RMS Δv = {rms_v_batch:.2e} m/s, max |Δv| = {max_v_batch:.2e} m/s")

    # ----- Persist for plotting -----
    out = os.path.join(os.path.dirname(__file__),
                       "trajectory_cpu_vs_gpu_results.npz")
    np.savez(
        out,
        ref_z0=np.array([r[1] for r in rows]),
        ref_v0=np.array([r[2] for r in rows]),
        ref_zf_py=np.array([r[3] if r[3] is not None else np.nan for r in rows]),
        ref_vf_py=np.array([r[4] if r[4] is not None else np.nan for r in rows]),
        ref_zf_jit=np.array([r[5] for r in rows]),
        ref_vf_jit=np.array([r[6] for r in rows]),
        ref_zf_gpu=np.array([r[7] for r in rows]),
        ref_vf_gpu=np.array([r[8] for r in rows]),
        ref_t_py=np.array([r[9] for r in rows]),
        ref_t_jit=np.array([r[10] for r in rows]),
        ref_t_gpu=np.array([r[11] for r in rows]),
        ref_traj_rms_z=np.array([r[12] for r in rows]),
        ref_traj_rms_v=np.array([r[13] for r in rows]),
        batch_z0=z0_arr, batch_v0=v0_arr,
        batch_zf_jit=zf_jit, batch_vf_jit=vf_jit,
        batch_zf_gpu=zf_gpu, batch_vf_gpu=vf_gpu,
        batch_n_actual=n_actual,
        batch_t_jit=t_jit_batch, batch_t_gpu=t_gpu_batch,
        t_max=args.t_max, dt=args.dt,
    )
    print(f"\n  saved arrays -> {out}")

    # ----- Verdict -----
    all_jit_gpu_ok = all(r[12] < 1e-6 and r[13] < 1e-4 for r in rows)
    batch_ok = rms_z_batch < 1e-6 and rms_v_batch < 1e-4
    print()
    print("=" * 86)
    if all_jit_gpu_ok and batch_ok:
        print(f"  PASS  CPU/JIT and CUDA agree on all 4 reference trajectories"
              f" + {args.n_particles}-particle batch")
    else:
        print(f"  CHECK CPU/JIT vs CUDA agreement -- see RMS above")
    print("=" * 86)


if __name__ == "__main__":
    main()
