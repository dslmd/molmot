#!/usr/bin/env python3
"""
GPU validation: compare molmot_cuda against CPU molmot.

Run this on a machine with a CUDA GPU and CuPy installed:
    python molmot_cuda/test_cuda.py

Each test computes the same quantity on CPU and GPU, then
reports the relative error. All should pass with < 1e-6.
"""

import sys
import time
import numpy as np

# ── Check GPU availability ──────────────────────────────────

import molmot_cuda
if not molmot_cuda.has_cuda():
    print("ERROR: No CUDA GPU detected. This script requires a GPU with CuPy.")
    print("Install CuPy: pip install cupy-cuda12x")
    sys.exit(1)

info = molmot_cuda.get_device_info()
print(f"GPU: {info['name']}")
print(f"Compute capability: {info['compute_capability']}")
print(f"Memory: {info['free_memory_MB']:.0f} / {info['total_memory_MB']:.0f} MB free")
print()

# ── Load molecular data ─────────────────────────────────────

from molmot.molecules.sroh import load_sroh_from_julia
from molmot.obe.fields import make_dc_mot_beams
from molmot.obe.rate_equations import solve_rate_equations
from molmot.mot.force_scan import force_vs_z, force_vs_v, spring_constant, damping_coefficient
from molmot.mot.simulator import DCMOTSimulator
from molmot.constants import hbar

mol = load_sroh_from_julia("julia_sim", verbose=False)
beams = make_dc_mot_beams(mol, delta_Gamma=-0.88, split_Gamma=0.39, s0=1.0)
B_grad = 0.16  # T/m = 16 G/cm

n_pass = 0
n_fail = 0

def check(name, cpu_val, gpu_val, tol=1e-6):
    global n_pass, n_fail
    cpu_val = np.asarray(cpu_val, dtype=float)
    gpu_val = np.asarray(gpu_val, dtype=float)
    denom = np.maximum(np.abs(cpu_val), np.abs(gpu_val))
    denom = np.where(denom < 1e-30, 1.0, denom)
    rel_err = np.max(np.abs(cpu_val - gpu_val) / denom)
    status = "PASS" if rel_err < tol else "FAIL"
    if status == "FAIL":
        n_fail += 1
    else:
        n_pass += 1
    print(f"  [{status}] {name}: rel_err = {rel_err:.2e}")
    return status == "PASS"


# ═══════════════════════════════════════════════════════════
# TEST 1: Rate equations -- F(z), F(v)
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1: Rate equation force profiles")
print("=" * 60)

z_arr = np.linspace(-5e-3, 5e-3, 101)
v_arr = np.linspace(-6, 6, 101)

# CPU
t0 = time.time()
Fz_cpu = np.array([solve_rate_equations(mol, beams, 0, z, B_grad)[0] for z in z_arr])
Fv_cpu = np.array([solve_rate_equations(mol, beams, v, 0, B_grad)[0] for v in v_arr])
t_cpu = time.time() - t0

# GPU
from molmot_cuda import force_vs_z_cuda, force_vs_v_cuda

t0 = time.time()
Fz_gpu, _, _ = force_vs_z_cuda(mol, beams, z_arr, 0.0, B_grad)
Fv_gpu = force_vs_v_cuda(mol, beams, v_arr, 0.0, B_grad)
t_gpu = time.time() - t0

check("F(z) 101 points", Fz_cpu, Fz_gpu)
check("F(v) 101 points", Fv_cpu, Fv_gpu)
print(f"  CPU: {t_cpu:.3f}s, GPU: {t_gpu:.3f}s, speedup: {t_cpu/t_gpu:.1f}x")


# ═══════════════════════════════════════════════════════════
# TEST 2: 2D force map
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 2: 2D force map F(v, z)")
print("=" * 60)

nv2, nz2 = 50, 50
v2 = np.linspace(-4, 4, nv2)
z2 = np.linspace(-3e-3, 3e-3, nz2)

# CPU
t0 = time.time()
Fmap_cpu = np.zeros((nv2, nz2))
for iv, v in enumerate(v2):
    for iz, z in enumerate(z2):
        Fmap_cpu[iv, iz], _, _ = solve_rate_equations(mol, beams, v, z, B_grad)
t_cpu = time.time() - t0

# GPU
from molmot_cuda import force_map_2d_cuda
t0 = time.time()
Fmap_gpu, _, _ = force_map_2d_cuda(mol, beams, v2, z2, B_grad)
t_gpu = time.time() - t0

check("F(v,z) 50x50 map", Fmap_cpu, Fmap_gpu)
print(f"  CPU: {t_cpu:.3f}s, GPU: {t_gpu:.3f}s, speedup: {t_cpu/t_gpu:.1f}x")


# ═══════════════════════════════════════════════════════════
# TEST 3: Spring constant and damping
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 3: Spring constant and damping coefficient")
print("=" * 60)

sim = DCMOTSimulator(mol, delta_Gamma=-0.88, split_Gamma=0.39,
                     s0=1.0, B_gradient=B_grad)

k_cpu = spring_constant(sim)
beta_cpu = damping_coefficient(sim)

from molmot_cuda import spring_constant_cuda, damping_coefficient_cuda
k_gpu = spring_constant_cuda(mol, beams, B_grad)
beta_gpu = damping_coefficient_cuda(mol, beams, B_grad)

check("spring constant k", k_cpu, k_gpu)
check("damping coeff alpha", beta_cpu, beta_gpu)


# ═══════════════════════════════════════════════════════════
# TEST 4: Parameter optimization scan
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 4: Parameter optimization (10x10 grid)")
print("=" * 60)

delta_arr = np.linspace(-2.0, -0.2, 10)
split_arr = np.linspace(0.1, 1.5, 10)

from molmot_cuda import optimize_parameters_cuda
from molmot.mot.force_scan import optimize_parameters

t0 = time.time()
result_cpu = optimize_parameters(mol, {"delta": delta_arr, "split": split_arr},
                                 B_grad, mot_type="dc", s0=1.0)
t_cpu = time.time() - t0

t0 = time.time()
result_gpu = optimize_parameters_cuda(mol, delta_arr, split_arr, 1.0, B_grad, verbose=False)
t_gpu = time.time() - t0

check("K_map 10x10", result_cpu["K"], result_gpu["K_map"])
check("beta_map 10x10", result_cpu["beta"], result_gpu["beta_map"])
print(f"  CPU: {t_cpu:.3f}s, GPU: {t_gpu:.3f}s, speedup: {t_cpu/t_gpu:.1f}x")


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = n_pass + n_fail
print(f"RESULTS: {n_pass}/{total} passed, {n_fail}/{total} failed")
if n_fail == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED -- see above")
print("=" * 60)

sys.exit(0 if n_fail == 0 else 1)
