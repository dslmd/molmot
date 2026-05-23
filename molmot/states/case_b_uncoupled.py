"""
Hund's case (b) uncoupled (decoupled) basis for linear molecules.

Port of UncoupledCaseB.jl from QuantumStates.jl.

In the uncoupled basis the angular momenta N, S, and I are
individually quantised along the space-fixed axis with projections
M_N, M_S, and M_I.  This basis is useful for describing molecules in
strong external fields where the Zeeman interaction dominates over
spin-rotation coupling.

The key function here is ``overlap``, which computes the Clebsch-Gordan
transformation between the coupled |N,S,J,I,F,M_F> basis and the
uncoupled |N,M_N,S,M_S,I,M_I> basis.

References
----------
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules*
  (Cambridge, 2003), eq. (5.187).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j


@dataclass
class HundsCaseB_LinearMolecule_Uncoupled(BasisState):
    """
    Uncoupled Hund's case (b) basis state for a linear molecule.

    Quantum numbers
    ---------------
    v1, v2, v3 : float
        Vibrational quantum numbers.
    S : float
        Total electron spin.
    I : float
        Nuclear spin.
    Lambda : float
        Projection of orbital angular momentum on the internuclear axis.
    ell : float
        Vibrational angular momentum.
    K : float
        K = Lambda + ell.
    N : float
        Rotational angular momentum quantum number (excluding spin).
    M_N : float
        Projection of N on space-fixed z-axis.
    M_S : float
        Projection of S on space-fixed z-axis.
    M_I : float
        Projection of I on space-fixed z-axis.
    """
    v1: float = 0
    v2: float = 0
    v3: float = 0
    S: float = 0
    I: float = 0
    Lambda: float = 0
    ell: float = 0
    K: float = 0
    N: float = 0
    M_N: float = 0
    M_S: float = 0
    M_I: float = 0

    constraints = {
        "K": lambda s: s.Lambda + s.ell,
        "N": lambda s: (abs(s.Lambda + s.ell), abs(s.Lambda + s.ell) + 100),
        "M_N": lambda s: (-s.N, s.N),
        "M_S": lambda s: (-s.S, s.S),
        "M_I": lambda s: (-s.I, s.I),
    }

    def __repr__(self):
        return (f"|v=({self.v1},{self.v2},{self.v3}), "
                f"N={Fraction(self.N).limit_denominator(10)}, "
                f"M_N={Fraction(self.M_N).limit_denominator(10)}, "
                f"M_S={Fraction(self.M_S).limit_denominator(10)}, "
                f"M_I={Fraction(self.M_I).limit_denominator(10)}>")


# =====================================================================
# Helper
# =====================================================================

def _f(x):
    """Convert to Fraction."""
    return Fraction(x).limit_denominator(1000)


# =====================================================================
# Overlap: coupled <-> uncoupled basis transformation
# =====================================================================

def overlap(coupled, uncoupled) -> float:
    """
    Compute the overlap between a coupled case (b) state and an
    uncoupled case (b) state.

    <N,S,J,I,F,M_F | N,M_N,S,M_S,I,M_I>

    This implements the Clebsch-Gordan decomposition:

        = sum_{M_J} (-1)^{J-I+M_F} * sqrt(2F+1)
              * W3j(J,I,F; M_J,M_I,-M_F)
              * (-1)^{N-S+M_J} * sqrt(2J+1)
              * W3j(N,S,J; M_N,M_S,-M_J)

    Reference: Brown & Carrington, eq. (5.187).

    Parameters
    ----------
    coupled : HundsCaseB_LinearMolecule
        Coupled basis state with quantum numbers N, S, J, I, F, M.
    uncoupled : HundsCaseB_LinearMolecule_Uncoupled
        Uncoupled basis state with quantum numbers N, M_N, S, M_S, I, M_I.

    Returns
    -------
    float
        Overlap (transformation coefficient).
    """
    # Import here to avoid circular imports
    from .case_b import HundsCaseB_LinearMolecule

    # Determine which is coupled and which is uncoupled
    if isinstance(coupled, HundsCaseB_LinearMolecule_Uncoupled):
        coupled, uncoupled = uncoupled, coupled

    if not delta(coupled, uncoupled, "v1", "v2", "v3", "Lambda", "ell", "N"):
        return 0.0

    N = _f(coupled.N)
    S = _f(coupled.S)
    J = _f(coupled.J)
    I = _f(coupled.I)
    F = _f(coupled.F)
    M_F = _f(coupled.M)

    M_N = _f(uncoupled.M_N)
    M_S = _f(uncoupled.M_S)
    M_I = _f(uncoupled.M_I)

    # Sum over M_J
    J_lo = max(-J, M_N + M_S - J)
    J_hi = min(J, M_N + M_S + J)

    val = 0.0
    M_J = -J
    while M_J <= J:
        val += ((-1) ** int(J - I + M_F) *
                math.sqrt(float(2 * F + 1)) *
                wigner3j(J, I, F, M_J, M_I, -M_F) *
                (-1) ** int(N - S + M_J) *
                math.sqrt(float(2 * J + 1)) *
                wigner3j(N, S, J, M_N, M_S, -M_J))
        M_J += 1

    return val
