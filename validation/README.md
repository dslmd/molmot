# Julia vs Python Validation

## Motivation

This package (`molmot`) is a Python translation of Christian Hallas's Julia
packages for molecular MOT simulation:

- [QuantumStates.jl](https://github.com/hallaschristian/QuantumStates.jl)
- [OpticalBlochEquations.jl](https://github.com/hallaschristian/OpticalBlochEquations.jl)
  (also published as QuantumSimulations.jl)

Because the physics involves dozens of angular momentum coupling coefficients,
Zeeman matrix elements, and transition dipole moments, a single sign error or
index bug can silently produce wrong forces. We validate the Python code by
running the **exact same simulation** in both Julia and Python and comparing
every output array point-by-point.

## What we compare

Using the parameters from Hallas's `SrOH_DC_redMOT` example:

| Parameter | Value |
|---|---|
| Molecule | SrOH, 12 ground + 4 excited = 16 states |
| Detunings | [-10.3, -5.9, -9.2, -29.9] MHz |
| Power ratios | [0.50, 0.08, 0.15, 0.27] |
| Polarizations | [sigma+, sigma-, sigma+, sigma-] |
| B gradient | 21.4 G/cm |
| Total power | 100 mW, beam radius 10 mm |

Both codes load the same molecular data (energies, TDMs, Zeeman matrices
exported from the Julia package) and compute:

- F(z) at 101 positions (-5 to +5 mm), v = 0
- F(v) at 101 velocities (-6 to +6 m/s), z = 0
- Steady-state populations at each z point (12 x 101 values)
- Spring constant, damping rate, trap frequency

## Results

| Quantity | Julia | Python | Relative Error |
|---|---|---|---|
| Max F(z) | 8.732531e-03 | 8.732531e-03 | 1.9e-12 |
| k_spring | -6.752e-20 N/m | -6.752e-20 N/m | 1.6e-08 |
| beta_damp | 2743.2 s^-1 | 2743.2 s^-1 | 1.6e-08 |
| F(z) array RMS | | | 3.1e-12 |
| F(v) array RMS | | | 1.2e-11 |
| Populations RMS | | | 8.2e-12 |
| Frequencies | | | 0.0 Hz |

**All quantities match to machine precision.**

## Files

- `sroh_redmot_comparison.py` -- Main comparison script. Runs Julia as a
  subprocess and Python directly, saves both outputs, generates comparison
  plots.
- `julia_redmot_results.npz` -- Arrays from the Julia run.
- `py_redmot_results.npz` -- Arrays from the Python run.
- `sroh_redmot_julia_vs_python.png` -- Comparison figure.
- `SrOH_RedMOT_Julia_vs_Python.pptx` -- Presentation with side-by-side
  plots (Julia plots made by Plots.jl, Python plots made by matplotlib).

## How to reproduce

```bash
cd validation
python sroh_redmot_comparison.py
```

Requires Julia with `OpticalBlochEquations.jl` and `QuantumStates.jl`
installed (see `../julia_sim/Project.toml`).
