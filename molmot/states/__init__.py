"""
Quantum states subpackage — basis definitions, Hamiltonians, and operators.

Port of QuantumStates.jl by Christian Hallas.
"""

from .basis import BasisState, delta, enumerate_states
from .case_b import HundsCaseB_LinearMolecule
from .case_a import HundsCaseA_LinearMolecule
from .overlaps import overlap_caseb_casea, convert_basis
from .hamiltonian import Hamiltonian, State, save_hamiltonian, load_hamiltonian, extend_basis
from .tdm import compute_tdms

__all__ = [
    "BasisState",
    "delta",
    "enumerate_states",
    "HundsCaseB_LinearMolecule",
    "HundsCaseA_LinearMolecule",
    "overlap_caseb_casea",
    "convert_basis",
    "Hamiltonian",
    "State",
    "save_hamiltonian",
    "load_hamiltonian",
    "compute_tdms",
    "extend_basis",
]
