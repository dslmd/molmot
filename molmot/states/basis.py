"""
Abstract quantum state basis — enumeration and delta functions.

Port of Basis.jl from QuantumStates.jl.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any, Dict, List, Type, Union
import copy


@dataclass
class BasisState:
    """
    Abstract base for all angular-momentum-coupled quantum states.

    Every concrete subclass adds quantum number fields and an optional
    ``constraints`` class attribute (dict) that encodes the allowed
    quantum-number relationships.

    Attributes
    ----------
    E : float
        Energy of this state (Hz).  Populated during Hamiltonian diagonalisation.
    label : str
        Human-readable label.
    """
    E: float = 0.0
    label: str = ""

    # Subclasses override this with the list of quantum-number field names
    # in the order they should be iterated / displayed.
    _qn_names: list = field(default_factory=list, repr=False)

    def qn_dict(self) -> dict:
        """Return a dict of all quantum-number fields (excluding E, label)."""
        skip = {"E", "label", "_qn_names"}
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in skip}


def delta(state1: BasisState, state2: BasisState, *quantum_numbers: str) -> bool:
    """
    Kronecker delta over specified quantum numbers.

    Returns True if *all* listed quantum numbers have the same value
    in ``state1`` and ``state2``.

    Parameters
    ----------
    state1, state2 : BasisState
    *quantum_numbers : str
        Names of quantum number attributes to compare.
    """
    for qn in quantum_numbers:
        v1 = getattr(state1, qn, None)
        v2 = getattr(state2, qn, None)
        if v1 is None or v2 is None:
            return False
        # Use Fraction comparison for half-integer safety
        if Fraction(v1) != Fraction(v2):
            return False
    return True


def enumerate_states(state_class: Type[BasisState],
                     qn_bounds: Dict[str, Any]) -> List[BasisState]:
    """
    Generate all valid basis states satisfying quantum-number bounds and
    constraints.

    Parameters
    ----------
    state_class : type
        A concrete subclass of ``BasisState``.
    qn_bounds : dict
        Maps quantum-number names to one of:

        * A single value (``float`` / ``Fraction``) — fixed.
        * A ``range`` or list — iterable of allowed values.

    The ``state_class.constraints`` dict maps dependent QN names to
    callables.  Two forms are supported:

    * ``lambda s: value`` — the QN is computed as a fixed value from
      other QNs already assigned (e.g., ``K = Lambda + ell``).
    * ``lambda s: (lo, hi)`` — the QN ranges from *lo* to *hi* in
      integer steps.

    Returns
    -------
    list of state_class instances
    """
    constraints = getattr(state_class, "constraints", {})

    # Determine iteration order: explicit qn_bounds first, then constraints
    all_qn_names = [f.name for f in fields(state_class)
                    if f.name not in ("E", "label", "_qn_names")]

    # Separate into categories
    fixed_qns = {}       # name -> single value
    iterable_qns = {}    # name -> list of values
    constrained_qns = {} # name -> callable from constraints

    for name in all_qn_names:
        if name in qn_bounds:
            val = qn_bounds[name]
            if isinstance(val, (int, float, Fraction)):
                fixed_qns[name] = Fraction(val).limit_denominator(1000)
            elif isinstance(val, range):
                iterable_qns[name] = [Fraction(v).limit_denominator(1000)
                                      for v in val]
            elif hasattr(val, "__iter__"):
                iterable_qns[name] = [Fraction(v).limit_denominator(1000)
                                      for v in val]
            else:
                fixed_qns[name] = Fraction(val).limit_denominator(1000)
        elif name in constraints:
            constrained_qns[name] = constraints[name]
        # else: leave at default (0)

    # Build states via recursive enumeration
    states = []
    _enumerate_recursive(state_class, all_qn_names, 0,
                         fixed_qns, iterable_qns, constrained_qns,
                         {}, states)
    return states


def _enumerate_recursive(state_class, qn_names, idx,
                         fixed, iterable, constrained,
                         current, results):
    """Recursively enumerate valid quantum-number combinations."""
    if idx == len(qn_names):
        # All QNs assigned — create the state
        kwargs = {name: float(val) for name, val in current.items()}
        results.append(state_class(**kwargs))
        return

    name = qn_names[idx]

    if name in fixed:
        current[name] = fixed[name]
        _enumerate_recursive(state_class, qn_names, idx + 1,
                             fixed, iterable, constrained,
                             current, results)
    elif name in iterable:
        for val in iterable[name]:
            current[name] = val
            _enumerate_recursive(state_class, qn_names, idx + 1,
                                 fixed, iterable, constrained,
                                 current, results)
    elif name in constrained:
        # Build a temporary object so the constraint lambda can read QNs
        temp = _make_temp(state_class, current)
        try:
            result = constrained[name](temp)
        except (AttributeError, TypeError, ZeroDivisionError):
            return

        if isinstance(result, tuple) and len(result) == 2:
            lo, hi = Fraction(result[0]).limit_denominator(1000), \
                     Fraction(result[1]).limit_denominator(1000)
            # Determine step: 1 for integer QNs, 1 for half-integer ranges
            step = Fraction(1)
            val = lo
            while val <= hi:
                current[name] = val
                _enumerate_recursive(state_class, qn_names, idx + 1,
                                     fixed, iterable, constrained,
                                     current, results)
                val += step
        else:
            # Single computed value
            current[name] = Fraction(result).limit_denominator(1000)
            _enumerate_recursive(state_class, qn_names, idx + 1,
                                 fixed, iterable, constrained,
                                 current, results)
    else:
        # Not specified — use default (0)
        current[name] = Fraction(0)
        _enumerate_recursive(state_class, qn_names, idx + 1,
                             fixed, iterable, constrained,
                             current, results)


def _make_temp(state_class, current):
    """Create a temporary state object from the currently assigned QNs."""
    kwargs = {name: float(val) for name, val in current.items()}
    return state_class(**kwargs)
