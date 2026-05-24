#!/usr/bin/env python3
"""
SrOH DC MOT speed benchmark: molmot CPU vs molmot CUDA.

Uses the Hallas SrOH_DC_redMOT 4-frequency / 8-beam configuration
and times several typical workloads at increasing grid sizes.

Each timing is the median of `--reps` trials (default 5) after one
warm-up run that compiles the CUDA kernel and primes any caches.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.fields import LaserBeam
from molmot.obe.rate_equations import solve_rate_equations
from molmot.constants import hbar, h, c

import molmot_cuda
from molmot_cuda import (
    force_vs_z_cuda,
    force_vs_v_cuda,
    force_map_2d_cuda,
    spring_constant_cuda,
    damping_coefficient_cuda,
)
import cupy as cp


JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia_sim")

# Hallas SrOH_DC_redMOT parameters
DETUNINGS_MHZ = [-10.3, -5.9, -9.2, -29.9]
POWER_RATIOS = [0.50, 0.08, 0.15, 0.27]
POLS_Q = [2, 0, 2, 0]
B_GRAD_SI = 0.214        # T/m  (= 21.4 G/cm)
TOTAL_POWER_MW = 100.0
BEAM_RADIUS_M = 10e-3
LAMBDA_M = 687e-9


def build_hallas_beams(mol):
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
    ratios = np.array(POWER_RATIOS, dtype=float); ratios /= ratios.sum()
    powers_W = TOTAL_POWER_MW * 1e-3 * ratios
    intensities = powers_W * (2.0 / (np.pi * BEAM_RADIUS_M ** 2))
    sats = intensities / Isat

    beams = []
    for f, q_fwd, s0 in zip(freqs_Hz, POLS_Q, sats):
        for direction in [+1, -1]:
            q_eff = q_fwd if direction > 0 else (2 - q_fwd if q_fwd != 1 else 1)
            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=f, polarization=q_eff, s0=s0))
    return beams


def time_median(fn, reps, sync_gpu=False):
    """Return (median_seconds, all_seconds)."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)), samples, out


def fmt_time(t):
    if t < 1e-6:
        return f"{t*1e9:7.1f} ns"
    if t < 1e-3:
        return f"{t*1e6:7.1f} µs"
    if t < 1.0:
        return f"{t*1e3:7.2f} ms"
    return f"{t:7.3f} s "


def fmt_speedup(cpu, gpu):
    if gpu <= 0:
        return "    inf"
    return f"{cpu/gpu:7.1f}x"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5,
                        help="Median over this many trials per case")
    parser.add_argument("--include-1d", action="store_true",
                        help="Also run the per-point loop benchmark")
    args = parser.parse_args()

    info = molmot_cuda.get_device_info()
    print("=" * 78)
    print("  SrOH DC red-MOT  --  CPU vs CUDA speed benchmark")
    print("=" * 78)
    print(f"  GPU : {info['name']} (cc {info['compute_capability']}, "
          f"{info['total_memory_MB']:.0f} MB)")
    try:
        import platform
        print(f"  CPU : {platform.processor() or platform.machine()} "
              f"({os.cpu_count()} logical cores)")
    except Exception:
        pass
    print(f"  Reps per case: {args.reps} (median reported, after warm-up)")

    mol = load_sroh_from_julia(JULIA_DIR, verbose=False)
    beams = build_hallas_beams(mol)
    print(f"  System: SrOH, {mol.n_ground} ground × {mol.n_excited} excited, "
          f"{len(beams)} beams")
    print()

    # Warm-up: compile CUDA kernel + prime caches
    _ = force_vs_z_cuda(mol, beams, np.linspace(-1e-3, 1e-3, 8),
                        0.0, B_GRAD_SI)
    cp.cuda.Stream.null.synchronize()

    header = (f"  {'Workload':<32} {'N points':>10} {'CPU':>11} "
              f"{'CUDA':>11} {'speed-up':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []

    # ----- 1D F(z) scans, increasing N -----
    for nz in (51, 101, 401, 1001, 4001):
        z_arr = np.linspace(-5e-3, 5e-3, nz)

        def fcpu():
            return np.array([solve_rate_equations(
                mol, beams, 0.0, z, B_GRAD_SI)[0] for z in z_arr])

        def fgpu():
            return force_vs_z_cuda(mol, beams, z_arr, 0.0, B_GRAD_SI)

        t_cpu, _, _ = time_median(fcpu, args.reps)
        t_gpu, _, _ = time_median(fgpu, args.reps, sync_gpu=True)
        row = ("F(z) scan",  nz, t_cpu, t_gpu)
        rows.append(row)
        print(f"  {row[0]:<32} {row[1]:>10d} {fmt_time(t_cpu):>11} "
              f"{fmt_time(t_gpu):>11} {fmt_speedup(t_cpu, t_gpu):>10}")

    # ----- 1D F(v) scans -----
    for nv in (51, 101, 401, 1001, 4001):
        v_arr = np.linspace(-6.0, 6.0, nv)

        def fcpu():
            return np.array([solve_rate_equations(
                mol, beams, v, 0.0, B_GRAD_SI)[0] for v in v_arr])

        def fgpu():
            return force_vs_v_cuda(mol, beams, v_arr, 0.0, B_GRAD_SI)

        t_cpu, _, _ = time_median(fcpu, args.reps)
        t_gpu, _, _ = time_median(fgpu, args.reps, sync_gpu=True)
        row = ("F(v) scan",  nv, t_cpu, t_gpu)
        rows.append(row)
        print(f"  {row[0]:<32} {row[1]:>10d} {fmt_time(t_cpu):>11} "
              f"{fmt_time(t_gpu):>11} {fmt_speedup(t_cpu, t_gpu):>10}")

    # ----- 2D F(v, z) maps (GPU should excel) -----
    for n in (32, 64, 128, 256, 512):
        v_arr = np.linspace(-6.0, 6.0, n)
        z_arr = np.linspace(-5e-3, 5e-3, n)

        # CPU is the slowest workload; for the largest grids skip CPU
        # repeats (extrapolate from one timing) to stay under ~minutes.
        if n <= 64:
            cpu_reps = args.reps
        elif n <= 128:
            cpu_reps = max(2, args.reps // 2)
        else:
            cpu_reps = 1   # one shot only

        def fcpu():
            Fmap = np.empty((n, n))
            for iv, v in enumerate(v_arr):
                for iz, z in enumerate(z_arr):
                    Fmap[iv, iz] = solve_rate_equations(
                        mol, beams, v, z, B_GRAD_SI)[0]
            return Fmap

        def fgpu():
            return force_map_2d_cuda(mol, beams, v_arr, z_arr, B_GRAD_SI)

        if cpu_reps == 1:
            t0 = time.perf_counter(); fcpu(); t_cpu = time.perf_counter() - t0
        else:
            t_cpu, _, _ = time_median(fcpu, cpu_reps)
        t_gpu, _, _ = time_median(fgpu, args.reps, sync_gpu=True)
        npts = n * n
        row = (f"F(v,z) map  ({n}x{n})", npts, t_cpu, t_gpu)
        rows.append(row)
        print(f"  {row[0]:<32} {row[1]:>10d} {fmt_time(t_cpu):>11} "
              f"{fmt_time(t_gpu):>11} {fmt_speedup(t_cpu, t_gpu):>10}")

    # ----- Spring constant + damping (2-point finite difference) -----
    def k_cpu():
        fp = solve_rate_equations(mol, beams, 0.0, +0.3e-3, B_GRAD_SI)[0]
        fm = solve_rate_equations(mol, beams, 0.0, -0.3e-3, B_GRAD_SI)[0]
        return -(fp - fm) / (2 * 0.3e-3)

    def k_gpu():
        return spring_constant_cuda(mol, beams, B_GRAD_SI, dz=0.3e-3)

    t_cpu, _, _ = time_median(k_cpu, args.reps)
    t_gpu, _, _ = time_median(k_gpu, args.reps, sync_gpu=True)
    row = ("spring_constant (2-pt FD)", 2, t_cpu, t_gpu)
    rows.append(row)
    print(f"  {row[0]:<32} {row[1]:>10d} {fmt_time(t_cpu):>11} "
          f"{fmt_time(t_gpu):>11} {fmt_speedup(t_cpu, t_gpu):>10}")

    def b_cpu():
        fp = solve_rate_equations(mol, beams, +0.1, 0.0, B_GRAD_SI)[0]
        fm = solve_rate_equations(mol, beams, -0.1, 0.0, B_GRAD_SI)[0]
        return -(fp - fm) / 0.2

    def b_gpu():
        return damping_coefficient_cuda(mol, beams, B_GRAD_SI, dv=0.1)

    t_cpu, _, _ = time_median(b_cpu, args.reps)
    t_gpu, _, _ = time_median(b_gpu, args.reps, sync_gpu=True)
    row = ("damping (2-pt FD)", 2, t_cpu, t_gpu)
    rows.append(row)
    print(f"  {row[0]:<32} {row[1]:>10d} {fmt_time(t_cpu):>11} "
          f"{fmt_time(t_gpu):>11} {fmt_speedup(t_cpu, t_gpu):>10}")

    # ----- Summary -----
    print()
    print("=" * 78)
    print("  Notes")
    print("=" * 78)
    print("  - Times exclude warm-up; CUDA times include host<->device copy and")
    print("    a cp.cuda.Stream.null.synchronize() at the end of each repetition.")
    print("  - The CPU path is pure NumPy/SciPy in molmot.obe.rate_equations.")
    print("    For larger CPU runs you can try numba JIT (see molmot/obe/")
    print("    rate_equations_jit.py) -- not used here for an apples-to-apples")
    print("    'pip install -e .[all]' comparison.")
    print("  - 2D maps with n >= 128 ran a single CPU trial to keep total")
    print("    runtime under a minute; GPU is always median of --reps.")


if __name__ == "__main__":
    run()
