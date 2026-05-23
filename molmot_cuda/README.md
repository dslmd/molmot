# molmot-cuda -- GPU-Accelerated Molecular MOT Simulation

CUDA-accelerated versions of the performance-critical functions in `molmot`.
Uses [CuPy](https://cupy.dev/) for GPU computation with custom CUDA kernels
for the inner loops.

## Status

**Untested on GPU.** The code compiles and imports cleanly on CPU (with
graceful fallback stubs). It has not yet been validated against the CPU
`molmot` package on an actual GPU. Run `test_cuda.py` (below) on your
GPU machine to validate.

Known items to verify on first GPU run:
- CUDA kernel compilation (CuPy RawKernel with `#define` macros)
- cuRAND initialization in the SSE kernel
- Double-precision complex arithmetic in the SSE RHS
- Gaussian elimination numerical stability for the 12x12 population matrix
- Memory layout (C-order vs Fortran-order) in all array transfers
- Kernel launch configuration (block/grid sizes)

## Requirements

```bash
pip install cupy-cuda12x   # for CUDA 12.x
# or
pip install cupy-cuda11x   # for CUDA 11.x
```

The `molmot` CPU package must also be installed (`pip install -e .` from
the repo root).

## Quick Start

```python
import molmot_cuda

if molmot_cuda.has_cuda():
    from molmot.molecules.sroh import load_sroh_from_julia
    from molmot.obe.fields import make_dc_mot_beams
    from molmot_cuda import force_map_2d_cuda

    mol = load_sroh_from_julia("julia_sim")
    beams = make_dc_mot_beams(mol, delta_Gamma=-0.88, split_Gamma=0.39, s0=1.0)

    v_arr = np.linspace(-6, 6, 200)
    z_arr = np.linspace(-5e-3, 5e-3, 200)
    F, pop, R = force_map_2d_cuda(mol, beams, v_arr, z_arr, B_gradient=0.16)
    # F is shape (200, 200) -- computed in ONE kernel launch
```

## Modules

| Module | CPU equivalent | What it accelerates | Expected speedup |
|--------|---------------|---------------------|------------------|
| `rate_equations_cuda` | `molmot.obe.rate_equations` | Batched (v,z) rate equation solves | 50-500x |
| `force_scan_cuda` | `molmot.mot.force_scan` | Force profiles, parameter scans, capture velocity | 50-500x |
| `stochastic_cuda` | `molmot.obe.stochastic` | Ensemble SSE trajectories with quantum jumps | 100-1000x |
| `simulator_3d_cuda` | `molmot.mot.simulator_3d` | 3D force maps, 3D trajectory ensembles | 50-500x |
| `floquet_cuda` | `molmot.obe.floquet` | Batched Floquet OBE velocity scans | 20-100x |
| `lindblad_cuda` | `molmot.obe.lindblad` | Batched Liouvillian construction + solve | 10-50x |

## How to Test

Run the test script on a machine with a CUDA GPU:

```bash
cd /path/to/DC\ red\ MOT
python molmot_cuda/test_cuda.py
```

The test script validates each CUDA module against the CPU `molmot` output:
1. **Rate equations**: compares F(z), F(v) at 101 points each
2. **Force scan**: compares spring constant and damping coefficient
3. **3D force**: compares force at several (r, v) points
4. **Floquet**: compares F(v) with sub-Doppler effects
5. **Lindblad**: compares steady-state density matrix
6. **SSE ensemble**: compares mean scattering rate from 100 trajectories

Each test reports PASS/FAIL with relative error.  All should pass with
relative error < 1e-6 (the GPU and CPU use the same algorithm, so
differences come only from floating-point operation ordering).

## Design

- **CuPy RawKernel** for inner loops (rate equation core, SSE RHS, 3D force)
- **CuPy batched linalg** for Floquet/Lindblad dense solves
- **Double precision** throughout for physics accuracy
- **Same API** as the CPU `molmot` functions -- drop-in replacement
- **Graceful fallback** when CuPy is not installed (stubs raise ImportError)
- Kernel dimensions injected via `#define` at compile time, cached per system size
