"""
Product (tensor product) states and Hamiltonian.

Port of ProductState.jl and ProductHamiltonian.jl from QuantumStates.jl.

A ``ProductState`` represents the tensor product of two basis states
from different Hilbert spaces (e.g., rotation x vibration).
``ProductHamiltonian`` builds the full Hamiltonian in the product basis
by Kronecker-summing the individual Hamiltonians.

References
----------
* Sakurai & Napolitano, *Modern Quantum Mechanics* (Cambridge, 2021),
  Section 1.7 (Tensor Product Spaces).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable, TypeVar, Generic
import numpy as np

from .basis import BasisState


# =====================================================================
# ProductState
# =====================================================================

@dataclass
class ProductState(BasisState):
    """
    Tensor product of two basis states.

    Parameters
    ----------
    basis_state1 : BasisState
        First factor in the tensor product.
    basis_state2 : BasisState
        Second factor in the tensor product.
    """
    basis_state1: BasisState = None
    basis_state2: BasisState = None

    def __repr__(self):
        return f"|{self.basis_state1}> x |{self.basis_state2}>"

    def __eq__(self, other):
        if not isinstance(other, ProductState):
            return False
        return (self.basis_state1 == other.basis_state1 and
                self.basis_state2 == other.basis_state2)

    def __hash__(self):
        return hash((type(self), repr(self.basis_state1),
                     repr(self.basis_state2)))


def product_overlap(s1: ProductState, s2: ProductState,
                    overlap1=None, overlap2=None) -> float:
    """
    Overlap of two product states.

    overlap = overlap1(s1.state1, s2.state1) * overlap2(s1.state2, s2.state2)

    If overlap functions are not provided, uses equality comparison.

    Parameters
    ----------
    s1, s2 : ProductState
    overlap1 : callable, optional
        Overlap function for the first factor.
    overlap2 : callable, optional
        Overlap function for the second factor.

    Returns
    -------
    float
    """
    if overlap1 is not None:
        o1 = overlap1(s1.basis_state1, s2.basis_state1)
    else:
        o1 = 1.0 if s1.basis_state1 == s2.basis_state1 else 0.0

    if overlap2 is not None:
        o2 = overlap2(s1.basis_state2, s2.basis_state2)
    else:
        o2 = 1.0 if s1.basis_state2 == s2.basis_state2 else 0.0

    return o1 * o2


def product_identity(s1: ProductState, s2: ProductState) -> float:
    """Identity operator on product states."""
    if s1 == s2:
        return 1.0
    return 0.0


def make_product_basis(basis1: list, basis2: list) -> List[ProductState]:
    """
    Construct a product basis from two individual bases.

    Returns the list of all ProductState(b1, b2) for b1 in basis1
    and b2 in basis2.

    Parameters
    ----------
    basis1 : list of BasisState
    basis2 : list of BasisState

    Returns
    -------
    list of ProductState
    """
    product_basis = []
    for b1 in basis1:
        for b2 in basis2:
            product_basis.append(ProductState(basis_state1=b1,
                                              basis_state2=b2))
    return product_basis


# =====================================================================
# ProductHamiltonian
# =====================================================================

class ProductHamiltonian:
    """
    Hamiltonian in a tensor-product Hilbert space.

    Constructs the full matrix as
        H = H1 (x) I2  +  I1 (x) H2
    where (x) denotes the Kronecker product.

    Parameters
    ----------
    basis : list of ProductState
        Product basis states.
    H1_matrix : ndarray
        Hamiltonian matrix for the first factor space.
    H2_matrix : ndarray
        Hamiltonian matrix for the second factor space.
    basis1 : list of BasisState
        Basis for the first factor space.
    basis2 : list of BasisState
        Basis for the second factor space.
    """

    def __init__(self, basis: List[ProductState],
                 H1_matrix: np.ndarray,
                 H2_matrix: np.ndarray,
                 basis1: list,
                 basis2: list):
        self.basis = basis
        self.H1_matrix = H1_matrix
        self.H2_matrix = H2_matrix
        self.basis1 = basis1
        self.basis2 = basis2

        n1 = len(basis1)
        n2 = len(basis2)
        n_total = n1 * n2
        self.matrix = np.zeros((n_total, n_total), dtype=complex)

    def evaluate(self):
        """
        Compute the product Hamiltonian matrix.

        H = H1 (x) I2  +  I1 (x) H2

        The result is stored in ``self.matrix``.
        """
        n1 = len(self.basis1)
        n2 = len(self.basis2)

        I1 = np.eye(n1, dtype=complex)
        I2 = np.eye(n2, dtype=complex)

        self.matrix[:] = (np.kron(self.H1_matrix, I2) +
                          np.kron(I1, self.H2_matrix))

    def add_interaction(self, operator, coefficient: float = 1.0):
        """
        Add an interaction term to the product Hamiltonian.

        The operator function should take (ProductState, ProductState)
        and return the matrix element.

        Parameters
        ----------
        operator : callable
            Function (s1, s2) -> float/complex.
        coefficient : float
            Prefactor for this term.
        """
        n = len(self.basis)
        for i in range(n):
            for j in range(n):
                self.matrix[i, j] += coefficient * operator(
                    self.basis[i], self.basis[j])
