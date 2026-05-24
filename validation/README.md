# Validation

## Motivation

`molmot` is a Python translation of Christian Hallas's Julia packages for
molecular MOT simulation
([QuantumStates.jl](https://github.com/hallaschristian/QuantumStates.jl),
[OpticalBlochEquations.jl](https://github.com/hallaschristian/OpticalBlochEquations.jl)).
The physics involves dozens of angular momentum coupling coefficients, Zeeman
matrix elements, and transition dipole moments — a single sign error can
silently produce wrong forces. We validate every backend (Julia, CPU Python,
CPU+JIT, CUDA) by running the same simulation and comparing point-by-point.

## Test parameters

From Hallas's `SrOH_DC_redMOT` example:

| Parameter | Value |
|---|---|
| Molecule | SrOH, 12 ground + 4 excited = 16 states |
| Detunings | [-10.3, -5.9, -9.2, -29.9] MHz |
| Power ratios | [0.50, 0.08, 0.15, 0.27] |
| Polarizations | [sigma+, sigma-, sigma+, sigma-] |
| B gradient | 21.4 G/cm |
| Total power | 100 mW, beam radius 10 mm |

## Results

### Julia vs Python (unsaturated rate equations, same model)

| Quantity | Relative Error |
|---|---|
| F(z) 101 points | **3.1 x 10^-12** |
| F(v) 101 points | **1.2 x 10^-11** |
| Populations (12 x 101) | **8.2 x 10^-12** |
| Laser frequencies (4) | **0.0 Hz** |

Machine precision agreement.

### CPU vs CPU+JIT (Numba)

| Quantity | Relative Error | Speedup |
|---|---|---|
| F(z) 101 points | 1.2 x 10^-8 | **20x** |
| Trajectories (30,000 steps) | **0.0** (bit-identical) | **3.7x** |

### CPU vs CUDA (RTX 3060)

| Quantity | Relative Error | Speedup |
|---|---|---|
| F(z) 101 points | 1.6 x 10^-8 | 3.8x |
| F(v,z) 50x50 map | 1.6 x 10^-8 | **780x** |
| Spring constant k | 1.6 x 10^-8 | — |
| Trajectory endpoints | 4 x 10^-18 m | **12x** (N=5000) |

### Three-way (Julia vs CPU(sat) vs CUDA(sat))

The `molmot` rate equation solver includes a saturation correction factor
`1/(1+s_total)` that the Julia reference does not. At s_total ~ 10 (the Hallas
parameters), this causes a ~7% systematic offset between Julia(unsat) and
molmot(sat). This is a deliberate physics choice, not a bug. When using the
same unsaturated formula, Julia and Python agree to 10^-12.

## Files

### Julia vs Python
- `sroh_redmot_comparison.py` — runs Julia subprocess + Python, compares arrays
- `julia_redmot_results.npz` — arrays from the Julia run
- `py_redmot_results.npz` — arrays from the Python run
- `sroh_redmot_julia_vs_python.png` — comparison figure
- `SrOH_RedMOT_Julia_vs_Python.pptx` — side-by-side presentation (Julia Plots.jl vs Python matplotlib)
- `julia_python_validate.py` — alternative comparison script

### Three-way (Julia vs CPU vs CUDA)
- `sroh_redmot_three_way.py` — all four solvers, pairwise RMS table
- `sroh_julia_cpu_gpu_comparison.png` — 6-panel comparison figure
- `three_way_results.npz` — saved arrays

### JIT benchmark
- `jit_benchmark.npz` — timing data for pure Python vs Numba JIT
- `SrOH_JIT_Validation.pptx` — JIT wiring presentation

### Trajectory (CPU vs CUDA)
- `trajectory_cpu_vs_gpu.py` — 4 reference trajectories, 3 backends
- `trajectory_cpu_vs_gpu.png` — comparison figure
- `trajectory_cpu_vs_gpu_results.npz` — endpoint data
- `trajectory_batch_scaling.py` — N-particle scaling sweep
- `trajectory_batch_scaling.npz` — timing data

## How to reproduce

```bash
# Julia vs Python (requires Julia + OpticalBlochEquations.jl)
python validation/sroh_redmot_comparison.py

# CPU test suite (no GPU needed)
pytest tests/ -v

# GPU tests (requires CUDA GPU + CuPy)
python molmot_cuda/test_cuda.py
python validation/trajectory_cpu_vs_gpu.py
python validation/trajectory_batch_scaling.py
```
