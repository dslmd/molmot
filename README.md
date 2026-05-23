# molmot -- Molecular Magneto-Optical Trap Simulation

A Python package for simulating magneto-optical traps (MOTs) of laser-coolable
molecules. Complete translation of Christian Hallas's Julia packages
([QuantumStates.jl](https://github.com/hallaschristian/QuantumStates.jl),
[OpticalBlochEquations.jl](https://github.com/hallaschristian/OpticalBlochEquations.jl))
with Numba JIT acceleration and additional solvers.

**Validated against Julia to machine precision** -- see
[validation/](validation/) for the comparison.

## Installation

```bash
pip install -e .
```

With JIT acceleration (recommended):

```bash
pip install -e ".[jit]"
```

## Quick Start

```python
import numpy as np
from molmot.molecules.sroh import load_sroh_from_julia
from molmot.mot.simulator import DCMOTSimulator
from molmot.mot.force_scan import spring_constant, damping_coefficient

# Load validated SrOH molecular data
mol = load_sroh_from_julia("julia_sim")

# Set up 4-frequency DC red MOT
sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                     s0=1.0, B_gradient=0.16)

# Compute trapping characteristics
k = spring_constant(sim)
beta = damping_coefficient(sim) / mol.mass
print(f"Trap frequency: {np.sqrt(k/mol.mass)/(2*np.pi):.0f} Hz")
print(f"Damping rate: {beta:.0f} s^-1")
```

## Features

### Quantum States (from QuantumStates.jl)

- Hund's case (a), (b), (c) basis states with all angular momentum operators
- Atomic states, angular momentum states, asymmetric top molecules
- Harmonic oscillator and triatomic vibrational states
- Product states and tensor product Hamiltonians
- Hamiltonian construction, diagonalisation, parameter scanning
- Transition dipole moments, basis conversion, state overlaps
- Wigner 3j/6j/9j symbols (Racah formula, LRU-cached)

### Optical Bloch Equations (from OpticalBlochEquations.jl)

- Rate equations: fast multi-level scattering rate solver
- Lindblad master equation: steady-state density matrix
- Stochastic Schrodinger equation: Monte Carlo wavefunction with quantum jumps
- Floquet sub-Doppler: Sisyphus cooling and VSCPT in standing waves
- Full OBE with spatial averaging for sub-Doppler forces
- Numba JIT kernels for all performance-critical paths

### MOT Simulation

- RF MOT and DC MOT simulators
- Full 3D MOT with quadrupole B-field and 6-beam geometry
- Force scanning, spring constants, damping coefficients
- Capture velocity estimation, parameter optimisation
- Trajectory integration (1D and 3D)
- Ensemble analysis: temperature, density, capture fraction

### Bayesian Optimisation (from BayesianOptimization.jl)

- Gaussian process regression (SE, Matern 5/2, ARD kernels)
- Acquisition functions (EI, PI, UCB, Thompson sampling)
- Batch optimisation with Kriging Believer heuristic

### Pre-built Molecules

- **SrOH**: 688 nm, Gamma = 6.4 MHz, 105 amu, 16-level system
- **CaOH**: 626 nm, Gamma = 6.4 MHz, 57 amu
- **CaF**: 606 nm, Gamma = 8.3 MHz, 59 amu

## Package Structure

```
molmot/
  constants.py          Physical constants (CODATA 2018)
  wigner.py             Wigner 3j/6j/9j symbols
  states/
    basis.py            BasisState, enumerate_states, subspace
    case_a.py           Hund's case (a): 19 operators
    case_b.py           Hund's case (b): 18 operators
    case_c.py           Hund's case (c): 7 operators
    case_b_uncoupled.py Uncoupled case (b) with overlap
    atomic.py           Atomic states with hyperfine + Zeeman
    angular_momentum.py 5 angular momentum state types
    asymmetric_top.py   Asymmetric top molecule
    vibrational.py      Harmonic oscillator + triatomic modes
    product.py          Tensor product states and Hamiltonians
    hamiltonian.py      Hamiltonian, CombinedHamiltonian, scanning
    overlaps.py         Case (a) <-> (b) overlaps
    tdm.py              Transition dipole moments
    transitions.py      Transition catalogue
  obe/
    fields.py           Laser beams, polarisation, 6-beam geometry
    rate_equations.py   Multi-level rate equation solver
    lindblad.py         Lindblad master equation solver
    stochastic.py       SSE / MCWF quantum jump solver
    floquet.py          Floquet sub-Doppler solver
    obe_subdoppler.py   Full OBE with standing waves
    force.py            Force from wavefunction / density matrix
    diffusion.py        Momentum diffusion and temperature
    *_jit.py            Numba JIT-compiled kernels
  mot/
    simulator.py        RF and DC MOT simulators
    simulator_3d.py     Full 3D MOT simulator
    force_scan.py       Force profiles and parameter optimisation
    analysis.py         Ensemble analysis tools
  propagation/
    trajectories.py     Trajectory integration and sampling
  molecules/
    sroh.py             SrOH data builder and loader
    caoh.py             CaOH data builder
    caf.py              CaF data builder
  optimization/
    kernels.py          GP kernels (SE, Matern52, ARD)
    gp.py               Gaussian process regression
    acquisition.py      Acquisition functions
    bopt.py             Bayesian optimiser
examples/               Example simulation scripts
validation/             Julia vs Python comparison (see validation/README.md)
julia_sim/              Julia reference data (CSV) and project files
tests/                  Test suite (77 tests)
```

## Validation

All rate equation forces, Zeeman shifts, transition dipole moments,
and steady-state populations match the Julia code to **machine precision**
(relative error < 10^-11). See [validation/](validation/) for details,
comparison plots, and a presentation.

## Solvers

| Solver | Speed | Physics | Use case |
|--------|-------|---------|----------|
| Rate equations | Fast (ms) | Doppler only | Parameter scans |
| Rate eq. (JIT) | Fastest | Doppler only | Trajectories |
| Lindblad OBE | Slow (s) | Sub-Doppler | Steady-state |
| Floquet OBE | Medium | Sub-Doppler | Force profiles |
| SSE / MCWF | Slow (min) | Full quantum | Single-molecule |
| 3D rate eq. | Medium | Full 3D Zeeman | Trap characterisation |

## Testing

```bash
pytest tests/ -v
```

## References

- Lasner et al., PRL 134, 083401 (2025) -- SrOH RF MOT
- Li et al., PRL 132, 233402 (2024) -- CaF blue MOT
- Hallas et al., QuantumStates.jl, OpticalBlochEquations.jl
- Dalibard, Castin, Molmer, PRL 68, 580 (1992) -- MCWF method
- Hirota, *High-Resolution Spectroscopy of Transient Molecules* (1985)
- Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules* (2003)

## License

MIT License. See [LICENSE](LICENSE).
