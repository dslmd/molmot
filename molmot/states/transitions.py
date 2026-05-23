"""
Transition data structures and computation.

Port of Transitions.jl from QuantumStates.jl.

Provides a ``Transition`` dataclass to represent spectroscopic
transitions between eigenstates, and ``compute_transitions`` to
find all non-negligible transitions between two sets of states.

References
----------
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules*
  (Cambridge, 2003), Chapter 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional
import numpy as np


@dataclass
class Transition:
    """
    A spectroscopic transition between two eigenstates.

    Attributes
    ----------
    ground_state : object
        The lower-energy eigenstate (a State from hamiltonian.py).
    excited_state : object
        The higher-energy eigenstate.
    frequency : float
        Transition frequency (Hz), equal to excited_state.E - ground_state.E.
    tdm : complex
        Transition dipole moment matrix element.
    """
    ground_state: Any = None
    excited_state: Any = None
    frequency: float = 0.0
    tdm: complex = 0.0 + 0.0j


def compute_transitions(states: list,
                        states_prime: list,
                        epsilon: list,
                        TDM_func: Callable = None,
                        threshold: float = 1e-8) -> List[Transition]:
    """
    Compute all transitions between two sets of eigenstates.

    For each pair (state, state') with state'.E > state.E, compute the
    transition dipole moment by contracting the TDM operator in the
    basis representation with the eigenstate coefficients.

    The polarisation vector epsilon = [epsilon_{-1}, epsilon_0, epsilon_{+1}]
    weights the three spherical components of the TDM.

    Parameters
    ----------
    states : list
        Ground-state eigenstates, each having ``.E``, ``.coeffs``,
        and ``.basis`` attributes.
    states_prime : list
        Excited-state eigenstates.
    epsilon : list of float/complex, length 3
        Polarisation weights [epsilon_{-1}, epsilon_0, epsilon_{+1}].
    TDM_func : callable, optional
        The TDM operator function (basis_state, basis_state', p) -> float.
        If None, attempts to import TDM from case_b.
    threshold : float
        Minimum |tdm| to include a transition.

    Returns
    -------
    list of Transition
        All transitions above threshold, sorted by frequency.
    """
    if TDM_func is None:
        from .case_b import TDM as TDM_func

    basis = states[0].basis
    n_basis = len(basis)

    # Pre-compute the TDM matrix in the basis representation
    tdm_matrix = np.zeros((n_basis, n_basis), dtype=complex)
    for p_idx, p in enumerate([-1, 0, 1]):
        ep = epsilon[p_idx]
        if abs(ep) < 1e-15:
            continue
        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                tdm_matrix[i, j] += ep * TDM_func(b1, b2, p)

    transitions = []
    for state in states:
        for state_p in states_prime:
            if state_p.E <= state.E:
                continue

            # Contract: tdm = coeffs . (tdm_matrix . coeffs')
            tdm_val = np.dot(
                np.conj(state.coeffs),
                tdm_matrix @ state_p.coeffs)

            freq = state_p.E - state.E

            if abs(tdm_val) > threshold and abs(freq) > 1.0:
                transitions.append(
                    Transition(
                        ground_state=state,
                        excited_state=state_p,
                        frequency=freq,
                        tdm=tdm_val))

    return transitions


def transitions_table(transitions: List[Transition],
                      relabelling_states: Optional[list] = None,
                      threshold: float = 1e-8) -> list:
    """
    Build a summary table of transitions.

    Returns a list of dicts with keys:
        'ground_idx', 'excited_idx', 'frequency', 'tdm'
    and optionally ground/excited state info.

    Parameters
    ----------
    transitions : list of Transition
    relabelling_states : list, optional
        If provided, used to relabel ground/excited states by index.
    threshold : float
        Minimum coefficient magnitude for state labels.

    Returns
    -------
    list of dict
        Sorted by frequency.
    """
    table = []
    for t in transitions:
        row = {
            "frequency": t.frequency,
            "tdm_real": float(np.real(t.tdm)),
            "tdm_abs": float(np.abs(t.tdm)),
        }

        # Add ground state info
        gs = t.ground_state
        row["ground_E"] = gs.E
        if hasattr(gs, "idx"):
            row["ground_idx"] = gs.idx

        # Add excited state info
        es = t.excited_state
        row["excited_E"] = es.E
        if hasattr(es, "idx"):
            row["excited_idx"] = es.idx

        # Add dominant basis state labels
        if hasattr(gs, "coeffs") and hasattr(gs, "basis"):
            max_idx = int(np.argmax(np.abs(gs.coeffs)))
            row["ground_label"] = repr(gs.basis[max_idx])
        if hasattr(es, "coeffs") and hasattr(es, "basis"):
            max_idx = int(np.argmax(np.abs(es.coeffs)))
            row["excited_label"] = repr(es.basis[max_idx])

        table.append(row)

    table.sort(key=lambda r: r["frequency"])
    return table
