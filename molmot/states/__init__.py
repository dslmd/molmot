"""
Quantum states subpackage — basis definitions, Hamiltonians, and operators.

Port of QuantumStates.jl by Christian Hallas.
"""

from .basis import BasisState, delta, enumerate_states
from .case_b import HundsCaseB_LinearMolecule
from .case_a import HundsCaseA_LinearMolecule
from .overlaps import overlap_caseb_casea, convert_basis
from .hamiltonian import (
    Hamiltonian, State, save_hamiltonian, load_hamiltonian,
    extend_basis, operator_to_matrix,
    CombinedHamiltonian, DiagonalOperator,
    scan_single_parameter, scan_parameters,
)
from .tdm import compute_tdms

# New state types
from .case_c import HundsCaseC_LinearMolecule
from .case_b_uncoupled import HundsCaseB_LinearMolecule_Uncoupled
from .atomic import AtomicState
from .angular_momentum import (
    AngularMomentumState,
    AngularMomentumState_Labelled,
    AngularMomentumState_Labelled_v1,
    AngularMomentumState_withSpinRotation,
    AngularMomentumState_withSpinRotation_Uncoupled,
)
from .asymmetric_top import AsymmetricTopMolecule
from .vibrational import (
    HarmonicOscillatorState,
    HarmonicOscillatorState_3D,
    TriatomicVibrationalState,
)
from .product import ProductState, ProductHamiltonian, make_product_basis
from .transitions import Transition, compute_transitions, transitions_table

__all__ = [
    # Core
    "BasisState", "delta", "enumerate_states",
    # State types
    "HundsCaseB_LinearMolecule",
    "HundsCaseA_LinearMolecule",
    "HundsCaseC_LinearMolecule",
    "HundsCaseB_LinearMolecule_Uncoupled",
    "AtomicState",
    "AngularMomentumState",
    "AngularMomentumState_Labelled",
    "AngularMomentumState_Labelled_v1",
    "AngularMomentumState_withSpinRotation",
    "AngularMomentumState_withSpinRotation_Uncoupled",
    "AsymmetricTopMolecule",
    "HarmonicOscillatorState",
    "HarmonicOscillatorState_3D",
    "TriatomicVibrationalState",
    "ProductState", "ProductHamiltonian", "make_product_basis",
    # Overlaps
    "overlap_caseb_casea", "convert_basis",
    # Hamiltonian
    "Hamiltonian", "State", "save_hamiltonian", "load_hamiltonian",
    "extend_basis", "operator_to_matrix",
    "CombinedHamiltonian", "DiagonalOperator",
    "scan_single_parameter", "scan_parameters",
    # TDM
    "compute_tdms",
    # Transitions
    "Transition", "compute_transitions", "transitions_table",
]
