# molmot -- Molecular Magneto-Optical Trap Simulation

A Python package for simulating magneto-optical traps (MOTs) of laser-coolable
molecules such as SrOH, CaOH, and CaF. Complete translation of Christian
Hallas's Julia packages
([QuantumStates.jl](https://github.com/hallaschristian/QuantumStates.jl),
[OpticalBlochEquations.jl](https://github.com/hallaschristian/OpticalBlochEquations.jl)),
validated against Julia to machine precision. Includes Numba JIT acceleration
(~20x CPU speedup) and an optional CUDA backend (`molmot_cuda`, ~800x on GPU).

## Installation

```bash
git clone https://github.com/dslmd/molmot.git
cd molmot
pip install -e .
```

With Numba JIT acceleration (recommended — gives ~20x speedup):

```bash
pip install -e ".[jit]"
```

With CUDA GPU support:

```bash
pip install -e ".[cuda]"
```

## Quick Start

```python
import numpy as np
from molmot.molecules.sroh import load_sroh_from_julia
from molmot.mot.simulator import DCMOTSimulator
from molmot.mot.force_scan import force_vs_z, spring_constant, damping_coefficient

# Load SrOH molecular data (12 ground + 4 excited = 16 states)
mol = load_sroh_from_julia("julia_sim")

# Set up a 4-frequency DC red MOT
sim = DCMOTSimulator(mol, delta_Gamma=-0.88, split_Gamma=0.39,
                     s0=1.0, B_gradient=0.16)

# Compute force profile and trapping characteristics
z_arr = np.linspace(-5e-3, 5e-3, 200)
F_arr, pop_arr, R_arr = force_vs_z(sim, z_arr, v=0.0)
k = spring_constant(sim)
beta = damping_coefficient(sim) / mol.mass
print(f"Spring constant: {k:.2e} N/m")
print(f"Damping rate: {beta:.0f} s^-1")
```

## Validation

The package is validated against the original Julia code at three levels:

| Comparison | Method | Result |
|---|---|---|
| Julia vs Python (same model) | Unsaturated rate equations, 101-point F(z) and F(v) | **RMS = 3 x 10^-12** |
| Julia vs Python (populations) | 12 ground states x 101 z-points | **RMS = 8 x 10^-12** |
| Python CPU vs Python+JIT | Numba-accelerated rate equations | **RMS = 1 x 10^-8** |
| Python CPU vs CUDA GPU | CuPy batched rate equations | **RMS = 1 x 10^-8** |

Parameters used: Hallas `SrOH_DC_redMOT` example (detunings [-10.3, -5.9, -9.2, -29.9] MHz,
B' = 21.4 G/cm, 100 mW total power). See [validation/](validation/) for scripts, data, plots,
and a PowerPoint presentation.

## Performance

Benchmarked on the SrOH 16-level system with the Hallas SrOH_DC_redMOT parameters.

### Force evaluation

| Backend | F(z) 101 pts | F(v,z) 50x50 | vs Pure Python |
|---|---|---|---|
| Pure Python | 38 ms | 9.5 s | 1x |
| Python + Numba JIT | **2 ms** | 0.5 s | **20x** |
| CUDA (RTX 3060) | — | **12 ms** | **780x** |

### Trajectory integration (30,000 steps per particle)

| Backend | 1 particle | 100 particles | 5,000 particles |
|---|---|---|---|
| Pure Python | 125 ms | 12.5 s | 625 s |
| Python + Numba JIT | 34 ms | 3.4 s | 170 s |
| CUDA (RTX 3060) | — | 8.7 s | **16 s** |

JIT acceleration is automatic: when Numba is installed, all rate equation
solvers and trajectory integrators dispatch to the JIT-compiled version.
No code changes needed. GPU crossover for trajectories is at ~500 particles.

## What's Inside

### Quantum States (`molmot/states/`)

Complete translation of [QuantumStates.jl](https://github.com/hallaschristian/QuantumStates.jl):

- **Basis states**: Hund's case (a), (b), (c); atomic; angular momentum (5 types);
  asymmetric top; harmonic oscillator; triatomic vibrational; tensor product
- **Operators**: 37 angular momentum operators across case (a) and (b) — rotation,
  spin-rotation, spin-orbit, hyperfine (Fermi contact, dipolar, quadrupole),
  Lambda-doubling, Renner-Teller, Zeeman (electron, nuclear, rotational),
  Stark, polarizability, transition dipole moments (electric, magnetic, vibrational)
- **Hamiltonian**: construction, diagonalisation, parameter scanning with
  adiabatic state tracking, basis extension, save/load
- **Wigner symbols**: 3j, 6j, 9j via Racah formula with LRU caching (65536 entries)

### Optical Bloch Equations (`molmot/obe/`)

Complete translation of [OpticalBlochEquations.jl](https://github.com/hallaschristian/OpticalBlochEquations.jl):

| Solver | Physics | Speed | Use case |
|---|---|---|---|
| **Rate equations** | Doppler cooling only | Fast (ms) | Parameter scans, optimisation |
| **Lindblad master equation** | Sub-Doppler (Sisyphus, VSCPT) | Slow (s) | Steady-state at fixed v, z |
| **Floquet OBE** | Sub-Doppler in standing waves | Medium | Velocity-dependent force profiles |
| **Full OBE with spatial averaging** | Sub-Doppler + spatial | Slow | Accurate sub-Doppler forces |
| **SSE / MCWF** | Full quantum (stochastic) | Slow (min) | Single-molecule trajectories, diffusion |

All solvers support arbitrary multi-level molecules (not just SrOH).

### MOT Simulation (`molmot/mot/`)

- **RF MOT**: two-phase polarisation/B-field switching
- **DC MOT**: 2-frequency and 4-frequency configurations
- **3D MOT**: full quadrupole B-field with 6-beam geometry, Zeeman
  eigendecomposition, polarisation decomposition
- **Force scanning**: F(z), F(v), 2D force maps, spring constants,
  damping coefficients, capture velocity estimation
- **Parameter optimisation**: 2D detuning/split scans
- **Trajectory integration**: 1D and 3D with Euler/leapfrog, ensemble runs
- **Analysis**: Gaussian/Maxwell-Boltzmann fits, cloud size, temperature,
  density, capture fraction

### Bayesian Optimisation (`molmot/optimization/`)

Translation of [BayesianOptimization.jl](https://github.com/hallaschristian/BayesianOptimization.jl):

- Gaussian process regression (SE, Matern 5/2, ARD kernels)
- Acquisition functions: Expected Improvement, Probability of Improvement,
  Upper Confidence Bound, Thompson Sampling, Max Variance
- Batch optimisation with Kriging Believer heuristic
- Sobol / Latin Hypercube initialisation

### CUDA Acceleration (`molmot_cuda/`)

GPU-accelerated versions of all performance-critical solvers:

| Module | What it does | Measured speedup |
|---|---|---|
| `rate_equations_cuda` | Batched (v,z) rate equation solves | **780x** (50x50 map) |
| `trajectories_cuda` | N-particle 1D trajectory ensemble | **12x** (N=5000) |
| `stochastic_cuda` | Ensemble SSE quantum trajectories | 100-1000x |
| `simulator_3d_cuda` | 3D force maps, trajectory ensembles | 50-500x |
| `floquet_cuda` | Batched Floquet velocity scans | 20-100x |
| `lindblad_cuda` | Batched Liouvillian + sub-Doppler OBE | 10-50x |
| `force_scan_cuda` | Parameter scans, capture velocity | 50-500x |

Requires CuPy. Graceful fallback when no GPU is available. Validated on
NVIDIA RTX 3060. See [molmot_cuda/README.md](molmot_cuda/README.md) for
details and scaling data.

### Pre-built Molecules (`molmot/molecules/`)

| Molecule | Transition | Wavelength | Linewidth | Mass | States |
|---|---|---|---|---|---|
| **SrOH** | X²Sigma+ -> A²Pi_1/2 | 687 nm | 6.4 MHz | 105 amu | 12 + 4 = 16 |
| **CaOH** | X²Sigma+ -> A²Pi_1/2 | 626 nm | 6.4 MHz | 57 amu | 12 + 4 = 16 |
| **CaF** | X²Sigma+ -> A²Pi_1/2 | 606 nm | 8.3 MHz | 59 amu | 12 + 4 = 16 |

Each molecule can be built from spectroscopic constants (`build_*_hamiltonian()`)
or loaded from pre-computed Julia reference data (`load_*_from_julia()`).

## Package Structure

```
molmot/                     CPU Python package
  constants.py              Physical constants (CODATA 2018)
  wigner.py                 Wigner 3j/6j/9j symbols
  states/                   Quantum state framework
    basis.py                BasisState, enumerate_states, subspace
    case_a.py               Hund's case (a): 19 operators
    case_b.py               Hund's case (b): 18 operators
    case_c.py               Hund's case (c): 7 operators
    case_b_uncoupled.py     Uncoupled case (b) with overlap
    atomic.py               Atomic states (hyperfine + Zeeman)
    angular_momentum.py     5 angular momentum state types
    asymmetric_top.py       Asymmetric top molecule
    vibrational.py          Harmonic oscillator + triatomic modes
    product.py              Tensor product states + Hamiltonians
    hamiltonian.py          Hamiltonian, scanning, CombinedHamiltonian
    overlaps.py             Case (a) <-> (b) overlap integrals
    tdm.py                  Transition dipole moments
    transitions.py          Transition catalogue
  obe/                      Optical Bloch equation solvers
    rate_equations.py       Multi-level rate equation solver
    rate_equations_jit.py   Numba JIT-compiled rate equations (~20x faster)
    lindblad.py             Lindblad master equation solver
    stochastic.py           SSE / MCWF quantum jump solver
    stochastic_jit.py       Numba JIT-compiled SSE inner loop
    floquet.py              Floquet sub-Doppler solver (uses JIT when available)
    floquet_jit.py          Numba JIT-compiled Floquet matrix assembly
    obe_subdoppler.py       Full OBE with standing-wave averaging
    fields.py               Laser beams, polarisation, 6-beam geometry
    force.py                Force from wavefunction / density matrix
    diffusion.py            Momentum diffusion and temperature
  mot/                      MOT simulation
    simulator.py            RF and DC MOT simulators (auto-JIT dispatch)
    simulator_3d.py         Full 3D MOT simulator
    simulator_3d_jit.py     Numba JIT 3D scattering rate
    force_scan.py           Force profiles, parameter optimisation
    analysis.py             Ensemble analysis tools
  propagation/              Trajectory integration
    trajectories.py         1D/3D integration, sampling utilities
    trajectories_jit.py     Numba JIT trajectory loop (~3.7x faster)
  molecules/                Molecular data
    sroh.py                 SrOH builder + Julia data loader
    caoh.py                 CaOH builder
    caf.py                  CaF builder
  optimization/             Bayesian optimisation
    kernels.py              GP kernels (SE, Matern52, ARD)
    gp.py                   Gaussian process regression
    acquisition.py          Acquisition functions
    bopt.py                 Bayesian optimiser
molmot_cuda/                GPU CUDA package (optional)
  rate_equations_cuda.py    Batched rate equation solver (780x)
  trajectories_cuda.py      N-particle trajectory ensemble (12x at N=5000)
  stochastic_cuda.py        Ensemble SSE trajectory solver
  simulator_3d_cuda.py      3D MOT force + trajectories
  floquet_cuda.py           Batched Floquet solver
  lindblad_cuda.py          Batched Liouvillian + sub-Doppler
  force_scan_cuda.py        Parameter scans + capture velocity
  utils.py                  MolecularDataGPU, array transfer
  test_cuda.py              GPU validation script
validation/                 Julia vs Python comparison
julia_sim/                  Julia reference data (CSV)
examples/                   Example simulation scripts
tests/                      Test suite (77 tests)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# GPU tests (requires CUDA GPU + CuPy)
python molmot_cuda/test_cuda.py
```

## References

- Lasner et al., PRL 134, 083401 (2025) -- SrOH RF MOT
- Li et al., PRL 132, 233402 (2024) -- CaF blue MOT
- Hallas et al., QuantumStates.jl, OpticalBlochEquations.jl
- Dalibard, Castin, Molmer, PRL 68, 580 (1992) -- MCWF method
- Hirota, *High-Resolution Spectroscopy of Transient Molecules* (Springer, 1985)
- Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules* (Cambridge, 2003)

## License

MIT License. See [LICENSE](LICENSE).
