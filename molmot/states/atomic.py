"""
Atomic state basis and operators.

Port of AtomicState.jl from QuantumStates.jl.

Defines basis states for atoms with quantum numbers L, S, J, I, F, M
and the associated operators for hyperfine structure and Zeeman effect.

References
----------
* Foot, *Atomic Physics* (Oxford, 2005).
* Arimondo, Inguscio, Violino, Rev. Mod. Phys. 49, 31 (1977).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j


@dataclass
class AtomicState(BasisState):
    """
    Atomic state in the |L,S,J,I,F,M> coupled basis.

    Quantum numbers
    ---------------
    L : float
        Orbital angular momentum.
    S : float
        Total electron spin.
    J : float
        Total electronic angular momentum J = L + S.
    I : float
        Nuclear spin.
    F : float
        Total angular momentum F = J + I.
    M : float
        Projection of F on the space-fixed z-axis.
    """
    L: float = 0
    S: float = 0
    J: float = 0
    I: float = 0
    F: float = 0
    M: float = 0

    constraints = {
        "J": lambda s: (abs(s.L - s.S), s.L + s.S),
        "F": lambda s: (abs(s.J - s.I), s.J + s.I),
        "M": lambda s: (-s.F, s.F),
    }

    def __repr__(self):
        return (f"|L={Fraction(self.L).limit_denominator(10)}, "
                f"S={Fraction(self.S).limit_denominator(10)}, "
                f"J={Fraction(self.J).limit_denominator(10)}, "
                f"F={Fraction(self.F).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# =====================================================================
# Helper
# =====================================================================

def _f(x):
    """Convert to Fraction."""
    return Fraction(x).limit_denominator(1000)


# =====================================================================
# Operator: Identity
# =====================================================================

def identity(s1: AtomicState, s2: AtomicState) -> float:
    """
    Identity operator: 1 if all quantum numbers match, 0 otherwise.
    """
    if not delta(s1, s2, "L", "J", "F", "M"):
        return 0.0
    return 1.0


# =====================================================================
# Operator: Hyperfine magnetic dipole  A * K / 2
# =====================================================================

def hyperfine_magnetic_dipole(s1: AtomicState, s2: AtomicState) -> float:
    """
    Magnetic dipole hyperfine interaction: A * K / 2.

    K = F(F+1) - I(I+1) - J(J+1)

    Diagonal in all quantum numbers.

    Reference: Foot, eq. (6.12).
    """
    if not delta(s1, s2, "L", "J", "F", "M"):
        return 0.0

    F = _f(s1.F)
    I = _f(s1.I)
    J = _f(s1.J)

    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    return float(K / 2)


# =====================================================================
# Operator: Hyperfine electric quadrupole
# =====================================================================

def hyperfine_electric_quadrupole(s1: AtomicState,
                                  s2: AtomicState) -> float:
    """
    Electric quadrupole hyperfine interaction.

    B * [3K(K+1)/2 - 2I(I+1)J(J+1)] / [4I(2I-1)J(2J-1)]

    where K = F(F+1) - I(I+1) - J(J+1).

    Diagonal in all quantum numbers.

    Reference: Foot, eq. (6.14).
    """
    if not delta(s1, s2, "L", "J", "F", "M"):
        return 0.0

    F = _f(s1.F)
    I = _f(s1.I)
    J = _f(s1.J)

    if I < 1 or J < 1:
        return 0.0

    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    numer = Fraction(3, 2) * K * (K + 1) - 2 * I * (I + 1) * J * (J + 1)
    denom = 4 * I * (2 * I - 1) * J * (2 * J - 1)

    if denom == 0:
        return 0.0

    return float(numer / denom)


# =====================================================================
# Operator: Zeeman B.S (electron spin)
# =====================================================================

def zeeman_BdotS(s1: AtomicState, s2: AtomicState, p: int) -> float:
    """
    Electron spin Zeeman: g_S * mu_B * B_p * T^1_p(S).

    Reference: Foot, Chapter 5; QuantumStates.jl AtomicState.jl.
    """
    if not delta(s1, s2, "L"):
        return 0.0

    L = _f(s1.L)
    S = _f(s1.S)
    J, Jp = _f(s1.J), _f(s2.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)

    if S == 0:
        return 0.0
    if abs(J - Jp) > 1 or abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(Fp - Mp) *
           wigner3j(Fp, 1, F, -Mp, p, M) *
           (-1) ** int(Jp + I + F + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(L + S + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1) *
                           S * (S + 1) * (2 * S + 1))) *
           wigner6j(S, Jp, L, J, S, 1))

    return float(val)


# =====================================================================
# Operator: Zeeman B.L (orbital)
# =====================================================================

def zeeman_BdotL(s1: AtomicState, s2: AtomicState, p: int) -> float:
    """
    Orbital Zeeman: g_L * mu_B * B_p * T^1_p(L).

    Reference: Foot, Chapter 5; QuantumStates.jl AtomicState.jl.
    """
    if not delta(s1, s2, "L"):
        return 0.0

    L = _f(s1.L)
    S = _f(s1.S)
    J, Jp = _f(s1.J), _f(s2.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)

    if L == 0:
        return 0.0
    if abs(J - Jp) > 1 or abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(Fp - Mp) *
           wigner3j(Fp, 1, F, -Mp, p, M) *
           (-1) ** int(Jp + I + F + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(L + S + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1) *
                           L * (L + 1) * (2 * L + 1))) *
           wigner6j(L, Jp, S, J, L, 1))

    return float(val)


# =====================================================================
# Operator: TDM (atomic electric dipole)
# =====================================================================

def TDM(s1: AtomicState, s2: AtomicState, p: int) -> float:
    """
    Transition dipole moment operator for polarisation p = -1, 0, +1.

    Connects states with different J (Delta_J = 0, +/-1, but J != J').

    Reference: QuantumStates.jl AtomicState.jl.
    """
    J = _f(s1.J)
    Jp = _f(s2.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)

    # Selection rule: J != J' (no elastic scattering in TDM)
    if J == Jp:
        return 0.0
    if abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(J, F, I, Fp, Jp, 1))

    return float(val)
