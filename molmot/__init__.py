"""
molmot — Molecular MOT Simulation Package
==========================================

A faithful Python port of Christian Hallas's Julia packages:
  - QuantumStates.jl
  - OpticalBlochEquations.jl
  - WignerSymbols_Simple.jl
  - UnitsToValue.jl
  - BeamPropagation.jl

Subpackages
-----------
constants
    CODATA 2018 physical constants.
wigner
    Wigner 3-j, 6-j, and 9-j symbols (Racah formula with LRU cache).
states
    Quantum state bases (Hund's cases a and b), Hamiltonian construction,
    operator matrix elements, basis overlaps, and TDM computation.
obe
    Optical Bloch Equations: rate equations and Lindblad master equation.
mot
    MOT simulators (RF and DC) with force scanning and optimisation.
propagation
    Trajectory integration and temperature estimation.
molecules
    Molecule-specific data (SrOH Hamiltonian builder and Julia data loader).
optimization
    Bayesian optimisation: GP regression, kernels, acquisition functions,
    and a batch-capable optimiser (port of BayesianOptimization.jl).

Quick start
-----------
>>> import molmot
>>> mol = molmot.molecules.load_sroh_from_julia("path/to/julia_sim/")
>>> from molmot.mot import DCMOTSimulator
>>> sim = DCMOTSimulator(mol, delta_Gamma=-0.88, split_Gamma=0.39, s0=1.0)
>>> F, pop, R = sim.force(v=0.0, z=1e-3)
"""

__version__ = "0.2.0"

from . import constants
from . import wigner
from . import states
from . import obe
from . import mot
from . import propagation
from . import molecules
from . import optimization
