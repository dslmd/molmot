# Prompt for GPU Agent: Validate molmot-cuda

## Task

Clone https://github.com/dslmd/molmot.git and validate that **Julia**, **Python (molmot)**, and **Python CUDA (molmot_cuda)** all produce identical results for the SrOH DC red MOT simulation.

## Setup

```bash
git clone https://github.com/dslmd/molmot.git
cd molmot
pip install -e ".[all]"
pip install cupy-cuda12x   # adjust for your CUDA version
```

Verify GPU is detected:
```python
import molmot_cuda
print(molmot_cuda.has_cuda())       # should be True
print(molmot_cuda.get_device_info())
```

## What to validate

### Step 1: Run the existing CUDA test suite

```bash
python molmot_cuda/test_cuda.py
```

This compares rate equation forces (F(z), F(v), 2D maps, spring constant, damping) between CPU molmot and GPU molmot_cuda. All tests should PASS with relative error < 1e-6.

If any test FAILs, debug and fix the CUDA kernel. Common issues:
- Array memory layout (C vs Fortran order)
- Kernel launch configuration (block/grid dimensions)
- CUDA double-precision floating point ordering differences

### Step 2: Three-way comparison (Julia vs molmot vs molmot_cuda)

Run the SrOH DC red MOT simulation with the Hallas example parameters in all three:

**Parameters (from QuantumSimulations.jl SrOH_DC_redMOT example):**
- Molecule: SrOH, 12 ground + 4 excited = 16 states
- Detunings: [-10.3, -5.9, -9.2, -29.9] MHz
- Power ratios: [0.50, 0.08, 0.15, 0.27]
- Polarizations: [sigma+, sigma-, sigma+, sigma-]
- B gradient: 21.4 G/cm
- Total power: 100 mW, beam radius: 10 mm

**What to compute in each:**
1. F(z) at 101 positions (-5 to +5 mm), v = 0
2. F(v) at 101 velocities (-6 to +6 m/s), z = 0
3. Steady-state populations at each z point (12 x 101 values)
4. Spring constant k, damping rate beta

**For Julia:** The script `validation/sroh_redmot_comparison.py` already runs Julia as a subprocess and saves results to `validation/julia_redmot_results.npz`. If Julia is installed, just run:
```bash
python validation/sroh_redmot_comparison.py
```
If Julia is NOT installed on this machine, use the pre-computed `validation/julia_redmot_results.npz` that is already in the repo.

**For Python CPU (molmot):** The same script saves CPU results to `validation/py_redmot_results.npz`.

**For Python GPU (molmot_cuda):** Write a new script that:
1. Loads the same molecular data from `julia_sim/`
2. Builds the same 4-frequency beam configuration
3. Calls `force_vs_z_cuda()`, `force_vs_v_cuda()` to compute F(z) and F(v)
4. Calls `spring_constant_cuda()`, `damping_coefficient_cuda()`
5. Compares against the Julia and CPU results
6. Reports: relative error for every quantity, PASS/FAIL, and GPU speedup

**Expected results (from previous validation):**
- Julia vs Python CPU: relative error ~ 1e-12 (machine precision)
- Julia vs Python GPU: should also be ~ 1e-12 (same algorithm, same data)
- CPU vs GPU: should be < 1e-6 (floating-point ordering may differ slightly)

### Step 3: Generate comparison figure

Create a plot showing all three (Julia, Python CPU, Python GPU) force curves overlaid, plus residuals. Save as `validation/sroh_julia_cpu_gpu_comparison.png`.

### Step 4: Report

Print a summary table:

```
Quantity              Julia           CPU             GPU             CPU-Julia       GPU-Julia       GPU-CPU
Max |F(z)|            ...             ...             ...             ...             ...             ...
k_spring              ...             ...             ...             ...             ...             ...
beta_damp             ...             ...             ...             ...             ...             ...
F(z) RMS              -               -               -               ...             ...             ...
F(v) RMS              -               -               -               ...             ...             ...
Populations RMS       -               -               -               ...             ...             ...
```

## If bugs are found

If the CUDA code produces wrong results:
1. Identify which kernel is wrong (rate_equations_cuda, force_scan_cuda, etc.)
2. Read the CPU version in `molmot/obe/rate_equations.py` to understand the expected computation
3. Read the CUDA kernel source in the corresponding `molmot_cuda/*.py` file
4. Fix the bug in the CUDA kernel
5. Re-run the test to confirm
6. Commit and push the fix

## File locations

- `molmot/` — CPU Python package (validated against Julia, do NOT modify)
- `molmot_cuda/` — GPU Python package (this is what you're testing)
- `julia_sim/` — Julia project files and pre-computed molecular data (CSV)
- `validation/` — Julia vs Python comparison scripts and results
- `molmot_cuda/test_cuda.py` — existing GPU test script
- `molmot_cuda/README.md` — documents current status and known issues
