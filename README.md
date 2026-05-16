# molmot -- Molecular Magneto-Optical Trap Simulation

A comprehensive Python package for simulating magneto-optical traps (MOTs) 
of laser-coolable molecules. Faithful port of Christian Hallas's Julia 
packages with additional features and Numba JIT acceleration.

## Features

- **Molecular Hamiltonian builder**: Hund's case (a) and (b) operators with 
  23 angular momentum operators, Wigner 3j/6j/9j symbols
- **Optical Bloch Equations**: Full Lindblad master equation solver for 
  steady-state density matrices
- **Stochastic Schrodinger Equation**: Monte Carlo wavefunction method with 
  quantum jumps for single-molecule trajectories
- **Floquet sub-Doppler**: Velocity-dependent forces in standing waves, 
  capturing Sisyphus cooling and velocity-selective dark states
- **3D MOT simulation**: Full quadrupole B-field with 6-beam geometry
- **Rate equations**: Fast multi-level scattering rate solver with 
  Numba JIT acceleration (2x faster than Julia)
- **Pre-built molecules**: SrOH, CaOH, CaF with validated spectroscopic constants

## Installation

```bash
pip install -e .
```

For JIT acceleration (recommended):

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
mol = load_sroh_from_julia('julia_sim')

# Set up 4-frequency DC red MOT
sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40, 
                     s0=1.0, B_gradient=0.16)

# Compute trapping characteristics
k = spring_constant(sim)
beta = damping_coefficient(sim) / mol.mass
print(f"Trap frequency: {np.sqrt(k/mol.mass)/(2*np.pi):.0f} Hz")
print(f"Damping rate: {beta:.0f} s^-1")
```

## Validation

- RF MOT damping: beta = 99 s^-1 (experiment: 100 s^-1, Lasner et al. PRL 2025)
- All rate equation forces match Julia to machine precision (<10^-15)
- Wigner symbols, state energies, Zeeman shifts, TDMs: exact match with Julia
- Numba JIT: 2x faster than Julia for trajectory integration

## Package Structure

```
molmot/
  constants.py          Physical constants (CODATA 2018)
  wigner.py             Wigner 3j/6j/9j symbols (Racah formula, LRU-cached)
  states/
    basis.py            Abstract BasisState, enumerate_states()
    case_b.py           Hund's case (b): 12 operators (Rotation, Zeeman, TDM, ...)
    case_a.py           Hund's case (a): 11 operators (SpinOrbit, LambdaDoubling, ...)
    hamiltonian.py      Hamiltonian construction, diagonalization, extend_basis()
    overlaps.py         Case (a) <-> case (b) overlap integrals
    tdm.py              Transition dipole moment computation
  obe/
    fields.py           Laser beam configs, polarization rotation, 6-beam geometry
    rate_equations.py   Multi-level rate equation solver
    rate_equations_jit.py   Numba JIT-compiled rate equation solver
    lindblad.py         Lindblad master equation (OBE) solver
    stochastic.py       Monte Carlo wavefunction / quantum jump SSE solver
    stochastic_jit.py   Numba JIT-compiled SSE inner loop
    floquet.py          Floquet sub-Doppler force calculator
    floquet_jit.py      Numba JIT-compiled Floquet matrix assembly
    obe_subdoppler.py   Full OBE with standing-wave sub-Doppler effects
    force.py            Force from wavefunction / density matrix
    diffusion.py        Momentum diffusion and temperature estimation
  mot/
    simulator.py        RF and DC MOT simulator classes
    simulator_3d.py     Full 3D MOT simulator with quadrupole B-field
    simulator_3d_jit.py Numba JIT-compiled 3D scattering rate loop
    force_scan.py       Force profiles, spring constants, parameter optimization
    analysis.py         Gaussian/MB fits, cloud size, capture fraction
  propagation/
    trajectories.py     1D/3D trajectory integration, particle sampling
  molecules/
    sroh.py             SrOH: from-scratch builder + Julia data loader
    caoh.py             CaOH: from-scratch + SrOH-rescaled loader
    caf.py              CaF: from-scratch + SrOH-rescaled loader
```

## Molecules

### SrOH (Strontium monohydroxide)
- Transition: X~2Sigma+(000, N=1) -> A~2Pi_1/2(000, J'=1/2)
- Wavelength: 688 nm
- Linewidth: Gamma/(2pi) = 6.4 MHz
- Mass: 105 amu
- 12 ground states + 4 excited states = 16-level system
- Validated against Julia-computed reference data

### CaOH (Calcium monohydroxide)
- Transition: X~2Sigma+(000, N=1) -> A~2Pi_1/2(000, J'=1/2)
- Wavelength: 626 nm
- Linewidth: Gamma/(2pi) = 6.4 MHz
- Mass: 57 amu

### CaF (Calcium monofluoride)
- Transition: X~2Sigma+(v=0, N=1) -> A~2Pi_1/2(v=0, J'=1/2)
- Wavelength: 606 nm
- Linewidth: Gamma/(2pi) = 8.3 MHz
- Mass: 59 amu
- Large hyperfine splitting (~123 MHz) from F-19

## Solvers

| Solver | Speed | Physics | Use case |
|--------|-------|---------|----------|
| Rate equations | Fast (ms) | Doppler only | Parameter scans, optimization |
| Rate eq. (JIT) | Fastest | Doppler only | Trajectory integration, benchmarks |
| Lindblad OBE | Slow (s) | Sub-Doppler | Steady-state at fixed v, z |
| Floquet OBE | Medium | Sub-Doppler | Velocity-dependent force profiles |
| SSE / MCWF | Slow (min) | Full quantum | Single-molecule trajectories, diffusion |
| 3D rate eq. | Medium | Full 3D Zeeman | 3D trap characterization |

## Testing

```bash
pip install pytest
cd /path/to/DC\ red\ MOT
python -m pytest tests/ -v
```

## References

- Lasner et al., PRL 134, 083401 (2025) -- SrOH RF MOT
- Li et al., PRL 132, 233402 (2024) -- CaF blue MOT
- Hallas et al., QuantumStates.jl, OpticalBlochEquations.jl
- Dalibard, Castin, Molmer, PRL 68, 580 (1992) -- MCWF method
- Hirota, *High-Resolution Spectroscopy of Transient Molecules* (1985)
- Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules* (2003)

## License

MIT License. See LICENSE file.
