"""
Hamiltonian construction and diagonalisation.

Port of the Hamiltonian handling in QuantumStates.jl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class State:
    """
    An eigenstate of the Hamiltonian, expressed as a linear combination
    of basis states.

    Attributes
    ----------
    E : float
        Eigenenergy (in the units of the Hamiltonian, typically Hz or cm^-1).
    coeffs : np.ndarray
        Complex coefficients in the basis.
    basis : list
        Reference to the list of basis states.
    index : int
        Index within the sorted eigenstate list.
    """
    E: float = 0.0
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    basis: list = field(default_factory=list, repr=False)
    index: int = 0

    def dominant_state(self):
        """Return the basis state with the largest coefficient magnitude."""
        idx = np.argmax(np.abs(self.coeffs))
        return self.basis[idx], self.coeffs[idx]


class Hamiltonian:
    """
    Molecular Hamiltonian from a sum of operator terms.

    Usage
    -----
    >>> H = Hamiltonian(basis)
    >>> H.add_operator("B", 0.24920, Rotation)
    >>> H.add_operator("gamma", 0.00242748, SpinRotation)
    >>> H.evaluate()
    >>> H.solve()
    >>> print(H.states[0].E)  # lowest eigenenergy

    Parameters
    ----------
    basis : list of BasisState
        The basis states spanning the Hilbert space.
    """

    def __init__(self, basis: list):
        self.basis = basis
        self.n = len(basis)
        self.matrix = np.zeros((self.n, self.n), dtype=complex)
        self.operators: List[Tuple[str, float, Callable]] = []
        self.states: List[State] = []
        self._eigenvalues = None
        self._eigenvectors = None

    def add_operator(self, param_name: str, param_value: float,
                     operator_func: Callable) -> None:
        """
        Add an operator term: param_value * operator_func(s1, s2).

        Parameters
        ----------
        param_name : str
            Name of the spectroscopic constant (for bookkeeping).
        param_value : float
            Value of the constant (e.g., B in cm^-1).
        operator_func : callable
            Function(state1, state2) -> float/complex matrix element.
        """
        self.operators.append((param_name, param_value, operator_func))

    def evaluate(self) -> np.ndarray:
        """
        Build the Hamiltonian matrix from all added operator terms.

        Returns
        -------
        np.ndarray
            The Hamiltonian matrix (n x n, complex).
        """
        self.matrix = np.zeros((self.n, self.n), dtype=complex)

        for param_name, param_value, op_func in self.operators:
            for i in range(self.n):
                for j in range(i, self.n):
                    val = op_func(self.basis[i], self.basis[j])
                    if abs(val) > 1e-15:
                        self.matrix[i, j] += param_value * val
                        if i != j:
                            self.matrix[j, i] += param_value * np.conj(val)

        return self.matrix

    def solve(self) -> List[State]:
        """
        Diagonalise the Hamiltonian and store sorted eigenstates.

        Uses ``numpy.linalg.eigh`` for Hermitian matrices.

        Returns
        -------
        list of State
            Eigenstates sorted by energy (ascending).
        """
        if self.matrix is None or np.all(self.matrix == 0):
            self.evaluate()

        # Check if the matrix is Hermitian
        if np.allclose(self.matrix, self.matrix.conj().T):
            eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
            # Sort by real part
            idx = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        self._eigenvalues = eigenvalues.real
        self._eigenvectors = eigenvectors

        self.states = []
        for k in range(self.n):
            state = State(
                E=float(self._eigenvalues[k]),
                coeffs=eigenvectors[:, k].copy(),
                basis=self.basis,
                index=k,
            )
            self.states.append(state)

        return self.states

    @property
    def energies(self) -> np.ndarray:
        """Return eigenvalues as a 1-D array."""
        if self._eigenvalues is None:
            self.solve()
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        """Return eigenvectors as columns of a 2-D array."""
        if self._eigenvectors is None:
            self.solve()
        return self._eigenvectors


def save_hamiltonian(hamiltonian: Hamiltonian, filepath: str) -> None:
    """
    Save Hamiltonian eigenvalues, eigenvectors, and basis info to npz file.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        A solved Hamiltonian (must have called .solve() first).
    filepath : str
        Path to the output .npz file.
    """
    if hamiltonian._eigenvalues is None:
        raise ValueError("Hamiltonian has not been solved yet. Call .solve() first.")

    # Collect basis state quantum numbers as a structured representation
    basis_dicts = []
    for state in hamiltonian.basis:
        basis_dicts.append(str(state.qn_dict()))

    np.savez(filepath,
             eigenvalues=hamiltonian._eigenvalues,
             eigenvectors=hamiltonian._eigenvectors,
             matrix=hamiltonian.matrix,
             basis_repr=np.array(basis_dicts),
             n_states=hamiltonian.n)


def load_hamiltonian(filepath: str) -> dict:
    """
    Load saved Hamiltonian data from an npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file saved by save_hamiltonian.

    Returns
    -------
    dict with keys:
        'eigenvalues' : np.ndarray, shape (n,)
        'eigenvectors' : np.ndarray, shape (n, n)
        'matrix' : np.ndarray, shape (n, n), complex
        'basis_repr' : list of str
        'n_states' : int
    """
    data = np.load(filepath, allow_pickle=True)
    return {
        'eigenvalues': data['eigenvalues'],
        'eigenvectors': data['eigenvectors'],
        'matrix': data['matrix'],
        'basis_repr': list(data['basis_repr']),
        'n_states': int(data['n_states']),
    }


def extend_basis(hamiltonian: Hamiltonian,
                 new_basis_states: list) -> Hamiltonian:
    """
    Extend an existing Hamiltonian to include additional basis states.

    Creates a new Hamiltonian whose basis is the union of the original
    basis and the new states. The operator terms from the original
    Hamiltonian are preserved and re-evaluated over the extended basis.

    This is the Python equivalent of the ``extend_basis`` function in
    QuantumStates.jl, used for adding vibrational states or extending
    the rotational manifold without rebuilding from scratch.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        A Hamiltonian that has had operators added (via ``add_operator``).
        Does not need to be solved yet.
    new_basis_states : list of BasisState
        Additional basis states to include. Duplicates of existing
        basis states are silently ignored.

    Returns
    -------
    Hamiltonian
        A new Hamiltonian with the extended basis. The operator list
        is copied from the original, and ``evaluate()`` is called
        automatically so the matrix is ready for ``solve()``.

    Examples
    --------
    >>> H = Hamiltonian(ground_basis_N1)
    >>> H.add_operator("B", B_val, Rotation)
    >>> H_ext = extend_basis(H, ground_basis_N3)
    >>> H_ext.solve()
    """
    # Build the union of old and new basis states, preserving order
    existing_reprs = set()
    combined_basis = []
    for s in hamiltonian.basis:
        key = repr(s)
        if key not in existing_reprs:
            existing_reprs.add(key)
            combined_basis.append(s)
    for s in new_basis_states:
        key = repr(s)
        if key not in existing_reprs:
            existing_reprs.add(key)
            combined_basis.append(s)

    # Create a new Hamiltonian with the combined basis
    H_new = Hamiltonian(combined_basis)

    # Copy all operator terms
    for param_name, param_value, operator_func in hamiltonian.operators:
        H_new.add_operator(param_name, param_value, operator_func)

    # Evaluate the matrix over the extended basis
    H_new.evaluate()

    return H_new


def operator_to_matrix(basis: list, operator_func: Callable,
                       *args) -> np.ndarray:
    """
    Build the matrix representation of an operator in the given basis.

    Parameters
    ----------
    basis : list of BasisState
    operator_func : callable(s1, s2, *args) -> float/complex
    *args : additional arguments passed to operator_func (e.g., p for Zeeman)

    Returns
    -------
    np.ndarray
        The operator matrix (n x n).
    """
    n = len(basis)
    mat = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            val = operator_func(basis[i], basis[j], *args)
            if abs(val) > 1e-15:
                mat[i, j] = val

    return mat


# =====================================================================
# CombinedHamiltonian
# =====================================================================

class CombinedHamiltonian:
    """
    Multiple Hamiltonians sharing a common eigendecomposition.

    Useful when ground and excited states live in different Hilbert
    spaces but need to be diagonalised together for TDM computation.
    """

    def __init__(self, hamiltonians: List['Hamiltonian']):
        self.hamiltonians = list(hamiltonians)

    def evaluate(self):
        for h in self.hamiltonians:
            h.evaluate()

    def solve(self):
        states = []
        for h in self.hamiltonians:
            states.extend(h.solve())
        return states


# =====================================================================
# DiagonalOperator
# =====================================================================

def DiagonalOperator(basis: list, operator_func: Callable,
                     *args) -> np.ndarray:
    """
    Compute only diagonal elements of an operator in the given basis.

    Returns a 1-D array of diagonal values.
    """
    return np.array([operator_func(b, b, *args) for b in basis],
                    dtype=complex)


# =====================================================================
# Parameter scanning
# =====================================================================

def scan_single_parameter(
    hamiltonian: 'Hamiltonian',
    param_name: str,
    values: np.ndarray,
    track_idxs: Optional[List[int]] = None,
) -> dict:
    """
    Scan a single parameter and record eigenvalues and eigenstates.

    Parameters
    ----------
    hamiltonian : Hamiltonian
    param_name : str
        Name of the parameter to scan.
    values : array-like
        Values to scan over.
    track_idxs : list of int, optional
        Indices of states to track (adiabatic tracking).

    Returns
    -------
    dict with keys 'values', 'energies', 'states'
    """
    n = len(hamiltonian.basis)
    energies = np.zeros((len(values), n))
    all_states: List = []

    prev_vecs = None

    for iv, v in enumerate(values):
        for i, (pn, pv, op) in enumerate(hamiltonian.operators):
            if pn == param_name:
                hamiltonian.operators[i] = (pn, v, op)

        hamiltonian.evaluate()
        states = hamiltonian.solve()

        if track_idxs is not None and prev_vecs is not None:
            current_vecs = np.array([s.coeffs for s in states])
            order = _track_states(prev_vecs, current_vecs, track_idxs)
            states = [states[k] for k in order]

        for i, s in enumerate(states):
            energies[iv, i] = s.E
        all_states.append(states)

        if track_idxs is not None:
            prev_vecs = np.array([s.coeffs for s in states])

    return {"values": np.array(values), "energies": energies,
            "states": all_states}


def scan_parameters(
    hamiltonian: 'Hamiltonian',
    params: dict,
    track_idxs: Optional[List[int]] = None,
) -> dict:
    """
    Scan over a dictionary of parameters simultaneously.

    Parameters
    ----------
    params : dict
        {param_name: array_of_values}.  All arrays must have the same length.
    """
    names = list(params.keys())
    arrays = [np.asarray(params[n]) for n in names]
    n_steps = len(arrays[0])
    n = len(hamiltonian.basis)
    energies = np.zeros((n_steps, n))
    all_states: List = []
    prev_vecs = None

    for iv in range(n_steps):
        for name, arr in zip(names, arrays):
            for i, (pn, pv, op) in enumerate(hamiltonian.operators):
                if pn == name:
                    hamiltonian.operators[i] = (pn, arr[iv], op)

        hamiltonian.evaluate()
        states = hamiltonian.solve()

        if track_idxs is not None and prev_vecs is not None:
            current_vecs = np.array([s.coeffs for s in states])
            order = _track_states(prev_vecs, current_vecs, track_idxs)
            states = [states[k] for k in order]

        for i, s in enumerate(states):
            energies[iv, i] = s.E
        all_states.append(states)

        if track_idxs is not None:
            prev_vecs = np.array([s.coeffs for s in states])

    return {"values": {n: a for n, a in zip(names, arrays)},
            "energies": energies, "states": all_states}


def _track_states(prev_vecs: np.ndarray, current_vecs: np.ndarray,
                  track_idxs: List[int]) -> List[int]:
    """Adiabatic state tracking via maximum overlap."""
    n = len(current_vecs)
    overlaps = np.abs(prev_vecs.conj() @ current_vecs.T)
    order = list(range(n))
    used = set()

    for idx in track_idxs:
        if idx >= n:
            continue
        row = overlaps[idx]
        candidates = sorted(range(n), key=lambda j: -row[j])
        for c in candidates:
            if c not in used:
                order[idx] = c
                used.add(c)
                break

    remaining = [j for j in range(n) if j not in used]
    remaining_slots = [i for i in range(n) if i not in track_idxs]
    for slot, rem in zip(remaining_slots, remaining):
        order[slot] = rem

    return order
