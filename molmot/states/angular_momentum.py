"""
Angular momentum states and operators.

Port of AngularMomentumState.jl, AngularMomentumState_Labelled.jl,
AngularMomentumState_Labelled_v1.jl, AngularMomentumState_withSpinRotation.jl,
and AngularMomentumState_withSpinRotation_Uncoupled.jl from QuantumStates.jl.

These are simpler state types useful for building product-space
Hamiltonians or for modelling spin-free systems such as rigid rotors.

References
----------
* Zare, *Angular Momentum* (Wiley, 1988).
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules*
  (Cambridge, 2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j


# =====================================================================
# Helper
# =====================================================================

def _f(x):
    """Convert to Fraction."""
    return Fraction(x).limit_denominator(1000)


###############################################################################
#  AngularMomentumState  |N, M>
###############################################################################

@dataclass
class AngularMomentumState(BasisState):
    """
    Simple angular momentum state |N, M>.

    Quantum numbers
    ---------------
    N : float
        Angular momentum quantum number.
    M : float
        Projection of N on space-fixed z-axis.
    """
    N: float = 0
    M: float = 0

    constraints = {
        "M": lambda s: (-s.N, s.N),
    }

    def __repr__(self):
        return (f"|N={Fraction(self.N).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# ----- Operators for AngularMomentumState -----

def AM_Identity(s1: AngularMomentumState,
                s2: AngularMomentumState) -> float:
    """Identity operator."""
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)
    if N != Np or M != Mp:
        return 0.0
    return 1.0


def AM_Rotation(s1: AngularMomentumState,
                s2: AngularMomentumState) -> float:
    """Rotation operator: N(N+1)."""
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)
    if N != Np or M != Mp:
        return 0.0
    return float(N * (N + 1))


def AM_TDM(s1: AngularMomentumState,
           s2: AngularMomentumState,
           p: int) -> float:
    """
    Transition dipole moment operator for polarisation p.

    Only connects states with different labels.
    """
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)

    label_factor = 1.0 if s1.label != s2.label else 0.0

    val = (label_factor *
           (-1) ** int(p) *
           (-1) ** int(N - M) *
           wigner3j(N, 1, Np, -M, -p, Mp) *
           math.sqrt(float(2 * N + 1)))

    return float(val)


def AM_TDM_magnetic(s1: AngularMomentumState,
                    s2: AngularMomentumState,
                    p: int) -> float:
    """
    Magnetic transition dipole moment operator.

    Assumes magnetic moment aligned along z-axis of molecule-fixed frame.
    """
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)

    val = ((-1) ** int(p) *
           (-1) ** int(N - M) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


def AM_d0(s1: AngularMomentumState,
          s2: AngularMomentumState) -> float:
    """
    Permanent electric dipole moment operator (q=0 component).

    d_0 = (-1)^M * sqrt((2N+1)(2N'+1)) * W3j(N,1,N',-M,0,M')
          * W3j(N,1,N',0,0,0)
    """
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)

    val = ((-1) ** int(M) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp) *
           wigner3j(N, 1, Np, 0, 0, 0))

    return float(val)


def AM_Zeeman(s1: AngularMomentumState,
              s2: AngularMomentumState) -> float:
    """
    Zeeman operator (diagonal in N, M).

    Returns (-1)^(N-M) * sqrt(N(N+1)(2N+1)) * W3j(N,1,N',-M,0,M').
    """
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)

    if M != Mp or N != Np:
        return 0.0

    val = ((-1) ** int(N - M) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp))

    return float(val)


def AM_overlap(s1: AngularMomentumState,
               s2: AngularMomentumState) -> float:
    """Overlap between two AngularMomentumState: delta(N,N') * delta(M,M')."""
    N, M = _f(s1.N), _f(s1.M)
    Np, Mp = _f(s2.N), _f(s2.M)
    if N != Np or M != Mp:
        return 0.0
    return 1.0


###############################################################################
#  AngularMomentumState_Labelled  |L, N, M>
###############################################################################

@dataclass
class AngularMomentumState_Labelled(BasisState):
    """
    Angular momentum state with an electronic label L.

    Used for multi-level systems where states are distinguished by
    a label L (e.g., ground vs excited).

    Quantum numbers
    ---------------
    L : float
        Electronic state label (0 = ground, 1 = excited, ...).
    N : float
        Angular momentum quantum number.
    M : float
        Projection of N on space-fixed z-axis.
    """
    L: float = 0
    N: float = 0
    M: float = 0

    constraints = {
        "M": lambda s: (-s.N, s.N),
    }

    def __repr__(self):
        return (f"|L={self.L}, "
                f"N={Fraction(self.N).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# ----- Operators for AngularMomentumState_Labelled -----

def AML_L_op(s1: AngularMomentumState_Labelled,
             s2: AngularMomentumState_Labelled) -> float:
    """Returns the L label value, diagonal in all QNs."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)
    if L != Lp or N != Np or M != Mp:
        return 0.0
    return float(L)


def AML_Identity(s1: AngularMomentumState_Labelled,
                 s2: AngularMomentumState_Labelled) -> float:
    """Identity operator."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)
    if L != Lp or N != Np or M != Mp:
        return 0.0
    return 1.0


def AML_Rotation(s1: AngularMomentumState_Labelled,
                 s2: AngularMomentumState_Labelled) -> float:
    """Rotation operator: N(N+1), diagonal in L."""
    if AML_Identity(s1, s2) == 0.0:
        return 0.0
    N = _f(s1.N)
    return float(N * (N + 1))


def AML_TDM(s1: AngularMomentumState_Labelled,
            s2: AngularMomentumState_Labelled,
            p: int) -> float:
    """
    Transition dipole moment operator.

    Only connects states where L' > L and not (L < 1 and L' < 1)
    and not (L > 1 and L' > 1).
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if Lp <= L or (L < 1 and Lp < 1) or (L > 1 and Lp > 1):
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(N - M) *
           wigner3j(N, 1, Np, -M, -p, Mp) *
           math.sqrt(float(2 * Np + 1)))

    return float(val)


def AML_TDM_magnetic(s1: AngularMomentumState_Labelled,
                     s2: AngularMomentumState_Labelled,
                     p: int) -> float:
    """
    Magnetic transition dipole moment.

    Assumes magnetic moment aligned along z-axis of molecule-fixed frame.
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    val = -((-1) ** int(p) *
            (-1) ** int(N - M) *
            math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
            wigner3j(N, 1, Np, M, p, -Mp))

    return float(val)


def AML_d0(s1: AngularMomentumState_Labelled,
           s2: AngularMomentumState_Labelled) -> float:
    """Permanent electric dipole moment (q=0 component)."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    val = ((-1) ** int(M) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp) *
           wigner3j(N, 1, Np, 0, 0, 0))

    return float(val)


def AML_Zeeman_L0(s1: AngularMomentumState_Labelled,
                  s2: AngularMomentumState_Labelled,
                  p: int = None) -> float:
    """
    Zeeman operator for ground states (L < 1).

    If p is None, uses p=0 (pi polarisation).
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L >= 1:
        return 0.0

    if p is None:
        val = ((-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, 0, Mp))
    else:
        val = ((-1) ** int(p) *
               (-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


def AML_Zeeman_L1(s1: AngularMomentumState_Labelled,
                  s2: AngularMomentumState_Labelled,
                  p: int = None) -> float:
    """
    Zeeman operator for excited states (L >= 1).

    If p is None, uses p=0 (pi polarisation).
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L < 1:
        return 0.0

    if p is None:
        val = ((-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, 0, Mp))
    else:
        val = ((-1) ** int(p) *
               (-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


def AML_Zeeman_Lx(s1: AngularMomentumState_Labelled,
                  s2: AngularMomentumState_Labelled,
                  p: int,
                  x: float) -> float:
    """
    Zeeman operator restricted to states with label L == x.

    Parameters
    ----------
    s1, s2 : AngularMomentumState_Labelled
    p : int
        Polarisation component.
    x : float
        The label value to select.
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != _f(x):
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(N - M) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


###############################################################################
#  AngularMomentumState_Labelled_v1  |L, N, M>  (integer L)
###############################################################################

@dataclass
class AngularMomentumState_Labelled_v1(BasisState):
    """
    Angular momentum state with an integer electronic label L.

    Variant of AngularMomentumState_Labelled where L is strictly
    an integer label.

    Quantum numbers
    ---------------
    L : float
        Electronic state label (integer; 0, 1, 2, ...).
    N : float
        Angular momentum quantum number.
    M : float
        Projection of N on space-fixed z-axis.
    """
    L: float = 0
    N: float = 0
    M: float = 0

    constraints = {
        "M": lambda s: (-s.N, s.N),
    }

    def __repr__(self):
        return (f"|L={int(self.L)}, "
                f"N={Fraction(self.N).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# ----- Operators for AngularMomentumState_Labelled_v1 -----

def AMLv1_T(s1: AngularMomentumState_Labelled_v1,
            s2: AngularMomentumState_Labelled_v1) -> float:
    """Energy operator: returns state energy when L labels match."""
    if s1.L == s2.L:
        return s1.E
    return 0.0


def AMLv1_L_op(s1: AngularMomentumState_Labelled_v1,
               s2: AngularMomentumState_Labelled_v1) -> float:
    """Returns the L label value, diagonal in all QNs."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)
    if L != Lp or N != Np or M != Mp:
        return 0.0
    return float(L)


def AMLv1_Identity(s1: AngularMomentumState_Labelled_v1,
                   s2: AngularMomentumState_Labelled_v1) -> float:
    """Identity operator."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)
    if L != Lp or N != Np or M != Mp:
        return 0.0
    return 1.0


def AMLv1_Rotation(s1: AngularMomentumState_Labelled_v1,
                   s2: AngularMomentumState_Labelled_v1) -> float:
    """Rotation operator: N(N+1), diagonal in L."""
    if AMLv1_Identity(s1, s2) == 0.0:
        return 0.0
    N = _f(s1.N)
    return float(N * (N + 1))


def AMLv1_TDM(s1: AngularMomentumState_Labelled_v1,
              s2: AngularMomentumState_Labelled_v1,
              p: int) -> float:
    """
    Transition dipole moment: only connects L' > L.
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if Lp <= L:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(N - M) *
           wigner3j(N, 1, Np, -M, -p, Mp) *
           math.sqrt(float(2 * Np + 1)))

    return float(val)


def AMLv1_TDM_magnetic(s1: AngularMomentumState_Labelled_v1,
                       s2: AngularMomentumState_Labelled_v1,
                       p: int) -> float:
    """
    Magnetic TDM: assumes magnetic moment along z-axis of molecule frame.
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    val = -((-1) ** int(p) *
            (-1) ** int(N - M) *
            math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
            wigner3j(N, 1, Np, M, p, -Mp))

    return float(val)


def AMLv1_d0(s1: AngularMomentumState_Labelled_v1,
             s2: AngularMomentumState_Labelled_v1) -> float:
    """Permanent electric dipole moment (q=0 component)."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    val = ((-1) ** int(M) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp) *
           wigner3j(N, 1, Np, 0, 0, 0))

    return float(val)


def AMLv1_Zeeman_L0(s1: AngularMomentumState_Labelled_v1,
                    s2: AngularMomentumState_Labelled_v1,
                    p: int = None) -> float:
    """
    Zeeman operator restricted to L == 0 states.

    If p is None, uses p=0 (pi polarisation).
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != 0:
        return 0.0

    if p is None:
        val = ((-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, 0, Mp))
    else:
        val = ((-1) ** int(p) *
               (-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


def AMLv1_Zeeman_L1(s1: AngularMomentumState_Labelled_v1,
                    s2: AngularMomentumState_Labelled_v1,
                    p: int = None) -> float:
    """
    Zeeman operator restricted to L == 1 states.

    If p is None, uses p=0 (pi polarisation).
    """
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != 1:
        return 0.0

    if p is None:
        val = ((-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, 0, Mp))
    else:
        val = ((-1) ** int(p) *
               (-1) ** int(N - M) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               wigner3j(N, 1, Np, -M, -p, Mp))

    return float(val)


def AMLv1_Zeeman_L2(s1: AngularMomentumState_Labelled_v1,
                    s2: AngularMomentumState_Labelled_v1) -> float:
    """Zeeman operator restricted to L == 2 states (p=0 only)."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != 2:
        return 0.0

    val = ((-1) ** int(N - M) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp))

    return float(val)


def AMLv1_Zeeman_L0_N2(s1: AngularMomentumState_Labelled_v1,
                       s2: AngularMomentumState_Labelled_v1) -> float:
    """Zeeman operator restricted to L == 0 and N == 2."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != 0 or N != 2:
        return 0.0

    val = ((-1) ** int(N - M) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner3j(N, 1, Np, -M, 0, Mp))

    return float(val)


def AMLv1_Zeeman_L0_N1(s1: AngularMomentumState_Labelled_v1,
                       s2: AngularMomentumState_Labelled_v1) -> float:
    """Zeeman operator restricted to L == 0 and N == 1, with extra sign."""
    L, N, M = _f(s1.L), _f(s1.N), _f(s1.M)
    Lp, Np, Mp = _f(s2.L), _f(s2.N), _f(s2.M)

    if L != Lp or N != Np or L != 0 or N != 1:
        return 0.0

    val = -((-1) ** int(N - M) *
            math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
            wigner3j(N, 1, Np, -M, 0, Mp))

    return float(val)


###############################################################################
#  AngularMomentumState_withSpinRotation  |N, S, J, M>
###############################################################################

@dataclass
class AngularMomentumState_withSpinRotation(BasisState):
    """
    Angular momentum state with spin-rotation coupling |N, S, J, M>.

    Quantum numbers
    ---------------
    N : float
        Rotational angular momentum.
    S : float
        Electron spin.
    J : float
        Total angular momentum J = N + S.
    M : float
        Projection of J on the space-fixed z-axis.
    """
    N: float = 0
    S: float = 0
    J: float = 0
    M: float = 0

    constraints = {
        "J": lambda s: (abs(s.N - s.S), s.N + s.S),
        "M": lambda s: (-s.J, s.J),
    }

    def __repr__(self):
        return (f"|N={Fraction(self.N).limit_denominator(10)}, "
                f"S={Fraction(self.S).limit_denominator(10)}, "
                f"J={Fraction(self.J).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# ----- Operators for AngularMomentumState_withSpinRotation -----

def AMSR_Identity(s1: AngularMomentumState_withSpinRotation,
                  s2: AngularMomentumState_withSpinRotation) -> float:
    """Identity operator."""
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)
    if J != Jp or N != Np or M != Mp:
        return 0.0
    return 1.0


def AMSR_Rotation(s1: AngularMomentumState_withSpinRotation,
                  s2: AngularMomentumState_withSpinRotation) -> float:
    """Rotation operator: N(N+1)."""
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)
    if J != Jp or N != Np or M != Mp:
        return 0.0
    return float(N * (N + 1))


def AMSR_SpinRotation(s1: AngularMomentumState_withSpinRotation,
                      s2: AngularMomentumState_withSpinRotation) -> float:
    """
    Spin-rotation coupling: gamma * N.S.

    Matrix element in the |N,S,J,M> basis.
    """
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)

    if N != Np or J != Jp or M != Mp:
        return 0.0

    val = ((-1) ** int(N + S + J) *
           math.sqrt(float(S * (S + 1) * (2 * S + 1) *
                           N * (N + 1) * (2 * N + 1))) *
           wigner6j(S, N, J, N, S, 1))

    return float(val)


def AMSR_TDM(s1: AngularMomentumState_withSpinRotation,
             s2: AngularMomentumState_withSpinRotation,
             p: int) -> float:
    """
    Transition dipole moment.

    Note: the Julia code uses wigner3j_ (which may differ in sign
    convention); here we use the standard wigner3j.
    """
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)

    val = ((-1) ** int(p) *
           (-1) ** int(N - M) *
           wigner3j(N, 1, Np, M, p, -Mp) *
           math.sqrt(float(2 * N + 1)))

    return float(val)


def AMSR_TDM_magnetic(s1: AngularMomentumState_withSpinRotation,
                      s2: AngularMomentumState_withSpinRotation,
                      p: int) -> float:
    """
    Magnetic TDM: assumes magnetic moment along z-axis of molecule frame.
    """
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)

    val = -((-1) ** int(p) *
            (-1) ** int(N - M) *
            math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
            wigner3j(N, 1, Np, M, p, -Mp))

    return float(val)


def AMSR_d_moment(s1: AngularMomentumState_withSpinRotation,
                  s2: AngularMomentumState_withSpinRotation,
                  p: int) -> float:
    """
    Electric dipole moment matrix element for polarisation p.

    Reference: Brown & Carrington.
    """
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)

    val = (-(-1) ** int(p) *
           (-1) ** int(J - M) *
           wigner3j(J, 1, Jp, -M, p, Mp) *
           (-1) ** int(S + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner6j(N, J, S, Jp, Np, 1) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, 1, Np, 0, 0, 0))

    return float(val)


def AMSR_SplitMStates(s1: AngularMomentumState_withSpinRotation,
                      s2: AngularMomentumState_withSpinRotation) -> float:
    """
    Returns M (the projection quantum number) for diagonal states.
    Used for splitting M sub-levels.
    """
    J, N, S, M = _f(s1.J), _f(s1.N), _f(s1.S), _f(s1.M)
    Jp, Np, Sp, Mp = _f(s2.J), _f(s2.N), _f(s2.S), _f(s2.M)
    if J != Jp or M != Mp or N != Np:
        return 0.0
    return float(M)


###############################################################################
#  AngularMomentumState_withSpinRotation_Uncoupled  |N, M_N, S, M_S>
###############################################################################

@dataclass
class AngularMomentumState_withSpinRotation_Uncoupled(BasisState):
    """
    Uncoupled angular momentum state |N, M_N, S, M_S>.

    The N and S angular momenta are separately quantised along the
    space-fixed z-axis.

    Quantum numbers
    ---------------
    N : float
        Rotational angular momentum.
    M_N : float
        Projection of N on the space-fixed z-axis.
    S : float
        Electron spin.
    M_S : float
        Projection of S on the space-fixed z-axis.
    """
    N: float = 0
    M_N: float = 0
    S: float = 0
    M_S: float = 0

    constraints = {
        "M_N": lambda s: (-s.N, s.N),
        "M_S": lambda s: (-s.S, s.S),
    }

    def __repr__(self):
        return (f"|N={Fraction(self.N).limit_denominator(10)}, "
                f"M_N={Fraction(self.M_N).limit_denominator(10)}, "
                f"S={Fraction(self.S).limit_denominator(10)}, "
                f"M_S={Fraction(self.M_S).limit_denominator(10)}>")


# ----- Overlap: coupled <-> uncoupled -----

def AMSR_overlap(coupled: AngularMomentumState_withSpinRotation,
                 uncoupled: AngularMomentumState_withSpinRotation_Uncoupled) -> float:
    """
    Overlap between coupled |N,S,J,M_J> and uncoupled |N,M_N,S,M_S>.

    <N,S,J,M_J | N,M_N,S,M_S> = delta(N,N')
        * (-1)^{N - S + M_J} * sqrt(2J + 1)
        * W3j(N, S, J, M_N, M_S, -M_J)

    The transformation is symmetric: overlap(uncoupled, coupled) is
    the same value.

    Parameters
    ----------
    coupled : AngularMomentumState_withSpinRotation
    uncoupled : AngularMomentumState_withSpinRotation_Uncoupled

    Returns
    -------
    float
    """
    # Handle either argument order
    if isinstance(coupled, AngularMomentumState_withSpinRotation_Uncoupled):
        coupled, uncoupled = uncoupled, coupled

    J = _f(coupled.J)
    Nc = _f(coupled.N)
    S = _f(coupled.S)
    M_J = _f(coupled.M)

    N = _f(uncoupled.N)
    M_N = _f(uncoupled.M_N)
    M_S = _f(uncoupled.M_S)

    if Nc != N:
        return 0.0

    val = ((-1) ** int(N - S + M_J) *
           math.sqrt(float(2 * J + 1)) *
           wigner3j(N, S, J, M_N, M_S, -M_J))

    return float(val)
