# molmot-cuda -- GPU-Accelerated Molecular MOT Simulation

CUDA-accelerated versions of the performance-critical functions in `molmot`.
Uses [CuPy](https://cupy.dev/) for GPU computation with custom CUDA kernels
for the inner loops.

## Status

**Validated on NVIDIA RTX 3060** (Ampere, CC 8.6, 12 GB). All rate equation
forces, spring constants, damping coefficients, and trajectory endpoints match
the CPU code to double-precision floating-point tolerance (relative error
~10^-8). See `test_cuda.py` and `../validation/` for details.

## Requirements

```bash
pip install cupy-cuda12x       # CUDA 12.x
# or
pip install cupy-cuda11x       # CUDA 11.x
# If compilation fails, add the CTK extras:
pip install cupy-cuda12x[ctk]
```

The `molmot` CPU package must also be installed:

```bash
pip install -e .               # from repo root
```

## Quick Start

```python
import numpy as np
import molmot_cuda

if molmot_cuda.has_cuda():
    from molmot.molecules.sroh import load_sroh_from_julia
    from molmot.obe.fields import make_dc_mot_beams
    from molmot_cuda import force_map_2d_cuda, simulate_trajectories_cuda

    mol = load_sroh_from_julia("julia_sim")
    beams = make_dc_mot_beams(mol, delta_Gamma=-0.88, split_Gamma=0.39, s0=1.0)

    # 200x200 force map in one kernel launch (~12 ms on RTX 3060)
    v_arr = np.linspace(-6, 6, 200)
    z_arr = np.linspace(-5e-3, 5e-3, 200)
    F, pop, R = force_map_2d_cuda(mol, beams, v_arr, z_arr, B_gradient=0.16)

    # 5000 trajectories in parallel (~16 s on RTX 3060, vs ~170 s CPU JIT)
    z0 = np.random.normal(0, 2e-3, 5000)
    v0 = np.random.normal(0, 1.0, 5000)
    z_final, v_final, escaped = simulate_trajectories_cuda(
        mol, beams, z0, v0, t_max=0.03, dt=1e-6, B_gradient=0.16)
```

## Modules

| Module | CPU equivalent | What it does | Measured speedup |
|--------|---------------|--------------|-----------------|
| `rate_equations_cuda` | `molmot.obe.rate_equations` | Batched (v,z) rate equation solves | **780x** (50x50 map) |
| `force_scan_cuda` | `molmot.mot.force_scan` | Parameter scans, capture velocity | 50-500x |
| `trajectories_cuda` | `molmot.propagation.trajectories` | N-particle 1D trajectory ensemble | **12x** (N=5000) |
| `stochastic_cuda` | `molmot.obe.stochastic` | Ensemble SSE quantum trajectories | 100-1000x |
| `simulator_3d_cuda` | `molmot.mot.simulator_3d` | 3D force maps, trajectory ensembles | 50-500x |
| `floquet_cuda` | `molmot.obe.floquet` | Batched Floquet OBE velocity scans | 20-100x |
| `lindblad_cuda` | `molmot.obe.lindblad` | Batched Liouvillian + sub-Doppler OBE | 10-50x |

### Trajectory GPU scaling (RTX 3060, 30,000 steps each)

| N particles | CPU JIT | CUDA | Speedup |
|---|---|---|---|
| 100 | 3.7 s | 8.7 s | 0.4x (overhead) |
| 500 | 19.5 s | 8.6 s | **2.3x** (crossover) |
| 5,000 | 195 s | 15.7 s | **12.4x** |
| 10,000 | 390 s | 30.9 s | **12.6x** |

GPU wins at N > 500 particles. Below that, kernel launch overhead dominates.
The plateau at ~12x is because each thread does 30,000 sequential Euler steps
with a 12x12 Gaussian elimination that spills to local memory on Ampere.

## Testing

```bash
# Rate equation and force scan tests (7 checks)
python molmot_cuda/test_cuda.py

# Trajectory validation (4 trajectories, 3-way comparison)
python validation/trajectory_cpu_vs_gpu.py

# Batch scaling sweep
python validation/trajectory_batch_scaling.py
```

## Design

- **CuPy RawKernel** for inner loops: rate equation core (Gaussian elimination
  of 12x12 population matrix), SSE RHS (16-state Schrodinger equation), 3D
  scattering rate (Jacobi eigensolve + polarisation decomposition), trajectory
  Euler loop
- **CuPy batched linalg** for Floquet/Lindblad dense solves
- **Double precision** throughout for physics accuracy
- **Same API** as the CPU `molmot` functions -- drop-in replacement
- **Graceful fallback** when CuPy is not installed (stubs raise ImportError
  with install instructions)
- Kernel dimensions injected via `#define` at compile time, cached per system size
- cuRAND per-thread RNG for SSE quantum jumps and spontaneous emission kicks
