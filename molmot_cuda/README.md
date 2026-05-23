# molmot-cuda -- GPU-Accelerated Molecular MOT Simulation

CUDA-accelerated versions of the performance-critical functions in `molmot`.
Uses [CuPy](https://cupy.dev/) for GPU computation with custom CUDA kernels
for the inner loops.

## Requirements

```bash
pip install cupy-cuda12x   # for CUDA 12.x
# or
pip install cupy-cuda11x   # for CUDA 11.x
```

## Usage

```python
import molmot_cuda

if molmot_cuda.has_cuda():
    from molmot.molecules.sroh import load_sroh_from_julia
    from molmot_cuda import force_map_2d_cuda, optimize_parameters_cuda

    mol = load_sroh_from_julia("julia_sim")
    # ... build beams ...

    # 200x200 force map in one GPU kernel launch
    F = force_map_2d_cuda(mol, beams, v_arr, z_arr, B_gradient)

    # Full 2D parameter scan on GPU
    k_map, beta_map = optimize_parameters_cuda(mol, delta_arr, split_arr, s0, B_grad)
```

## Modules

| Module | CPU equivalent | What it accelerates | Expected speedup |
|--------|---------------|---------------------|------------------|
| `rate_equations_cuda` | `molmot.obe.rate_equations` | Batch (v,z) rate equation solves | 50-500x |
| `force_scan_cuda` | `molmot.mot.force_scan` | Force profiles, parameter scans, capture velocity | 50-500x |
| `stochastic_cuda` | `molmot.obe.stochastic` | Ensemble SSE trajectories with quantum jumps | 100-1000x |
| `simulator_3d_cuda` | `molmot.mot.simulator_3d` | 3D force maps, 3D trajectory ensembles | 50-500x |
| `floquet_cuda` | `molmot.obe.floquet` | Batched Floquet OBE velocity scans | 20-100x |
| `lindblad_cuda` | `molmot.obe.lindblad` | Batched Liouvillian construction + solve | 10-50x |

## Design

- **CuPy RawKernel** for inner loops (rate equation core, SSE RHS, 3D force)
- **CuPy batched linalg** for Floquet/Lindblad dense solves
- **Double precision** throughout for physics accuracy
- **Same API** as the CPU `molmot` functions -- drop-in replacement
- **Graceful fallback** when CuPy is not installed
