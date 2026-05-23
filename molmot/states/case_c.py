"""
Hund's case (c) basis for linear molecules.

In Hund's case (c), Lambda and Sigma are not individually good quantum
numbers; only their sum Omega = Lambda + Sigma is well-defined.
This is appropriate when spin-orbit coupling is large compared to
rotational spacing.

The quantum numbers are:
    Omega (projection of J_a on the internuclear axis),
    J, I, F, M, plus vibrational labels.

Operators implemented here follow Brown & Carrington, Chapter 9,
translated from HundsCaseA_LinearMolecule.jl in QuantumStates.jl
and adapted for the case (c) coupling scheme where Lambda and Sigma
are replaced by Omega.

References
----------
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules*
  (Cambridge, 2003), Chapters 6 and 9.
* Hirota, *High-Resolution Spectroscopy of Transient Molecules* (1985).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j


@dataclass
class HundsCaseC_LinearMolecule(BasisState):
    """
    Hund's case (c) basis state for a linear molecule.

    Quantum numbers
    ---------------
    v1, v2, v3 : float
        Vibrational quantum numbers.
    S : float
        Total electron spin (used for matrix element prefactors;
        Sigma is not a good QN in case c).
    I : float
        Nuclear spin.
    Omega : float
        Projection of total electronic angular momentum on the
        internuclear axis: Omega = Lambda + Sigma.
    J : float
        Total angular momentum excluding nuclear spin.
    F : float
        Total angular momentum including nuclear spin.
    M : float
        Space-fixed projection of F.
    """
    v1: float = 0
    v2: float = 0
    v3: float = 0
    S: float = 0
    I: float = 0
    Omega: float = 0
    J: float = 0
    F: float = 0
    M: float = 0

    constraints = {
        "J": lambda s: (abs(s.Omega), abs(s.Omega) + 100),
        "F": lambda s: (abs(s.J - s.I), s.J + s.I),
        "M": lambda s: (-s.F, s.F),
    }

    def __repr__(self):
        return (f"|v=({self.v1},{self.v2},{self.v3}), "
                f"Omega={Fraction(self.Omega).limit_denominator(10)}, "
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
# Operator: Rotation  J(J+1) - Omega^2
# =====================================================================

def Rotation(s1: HundsCaseC_LinearMolecule,
             s2: HundsCaseC_LinearMolecule) -> float:
    """
    Rigid rotor Hamiltonian in case (c): B * [J(J+1) - Omega^2].

    Diagonal in all quantum numbers.

    Reference: Brown & Carrington, eq. (6.149).
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Omega", "J", "F", "M"):
        return 0.0
    J = s1.J
    Omega = s1.Omega
    return J * (J + 1) - Omega ** 2


# =====================================================================
# Operator: Lambda-doubling / spin-orbit (connects +Omega and -Omega)
# =====================================================================

def Lambda_doubling_spin_orbit(s1: HundsCaseC_LinearMolecule,
                               s2: HundsCaseC_LinearMolecule) -> float:
    """
    Lambda-doubling matrix element connecting |Omega> with |-Omega>.

    In case (c), this manifests as an off-diagonal coupling between
    +Omega and -Omega states via the q-parameter.

    H_LD = q/2 * [J_+^2 + J_-^2] acting on the parity doublet.

    Reference: Brown & Carrington, eq. (9.63).
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I", "J", "F", "M"):
        return 0.0

    Omega1 = _f(s1.Omega)
    Omega2 = _f(s2.Omega)
    J = _f(s1.J)

    # Connects Omega with -Omega (selection rule Delta_Omega = +/-2*Omega)
    if Omega1 + Omega2 != 0 or Omega1 == 0:
        return 0.0

    # For |Omega| = 1/2: q * (J + 1/2)
    # For |Omega| = 1: q/2 * J(J+1)
    # General form using 3j symbols:
    val = ((-1) ** int(J - Omega1) *
           (Fraction(1, 2) / Fraction(math.sqrt(6.0)).limit_denominator(10000)) *
           math.sqrt(float((2 * J - 1) * 2 * J * (2 * J + 1) *
                           (2 * J + 2) * (2 * J + 3))) *
           wigner3j(J, 2, J, -Omega1, 2 * Omega1, Omega2))

    return float(val)


# =====================================================================
# Operator: Lambda-doubling / nuclear spin rotation
# =====================================================================

def Lambda_doubling_nuclear_spin_rotation(
        s1: HundsCaseC_LinearMolecule,
        s2: HundsCaseC_LinearMolecule) -> float:
    """
    Nuclear-spin-dependent Lambda-doubling connecting Omega and -Omega.

    This is the (p+2q)-type term adapted for case (c), where the
    coupling involves a rank-1 tensor in J and a rank-1 tensor in S.

    Reference: Brown & Carrington, eq. (9.66).
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I", "J", "F", "M"):
        return 0.0

    Omega1 = _f(s1.Omega)
    Omega2 = _f(s2.Omega)
    J = _f(s1.J)
    S = _f(s1.S)

    if S == 0:
        return 0.0

    # This operator connects Omega' = -Omega through a rank-1 coupling
    val = 0.0
    for q in [-1, 1]:
        if Omega2 != Omega1 + q:
            continue
        w3j_J = wigner3j(J, 1, J, -Omega1, -q, Omega2)
        # In case (c) the spin part is absorbed; use delta-like coupling
        val += ((-1) ** int(J - Omega1) *
                math.sqrt(float(J * (J + 1) * (2 * J + 1))) *
                w3j_J)

    return float(val)


# =====================================================================
# Operator: Magnetic hyperfine (nuclear spin 1)
# =====================================================================

def magnetic_hyperfine_1(s1: HundsCaseC_LinearMolecule,
                         s2: HundsCaseC_LinearMolecule) -> float:
    """
    Magnetic hyperfine interaction for nuclear spin I_1, projected
    onto case (c) basis.

    h_1 * Omega * <I.J_a/J(J+1)> — diagonal in Omega.

    Reference: Brown & Carrington, eq. (8.372) adapted for case (c).
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Omega", "J", "M"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    Omega = _f(s1.Omega)

    if I == 0:
        return 0.0
    if J != Jp or F != Fp:
        return 0.0

    # a * Omega * (-1)^(J+I+F) * (-1)^(J-Omega)
    # * sqrt(I(I+1)(2I+1)(2J+1)(2J'+1))
    # * W6j(I,J,F,J',I,1) * W3j(J,1,J',-Omega,0,Omega)
    val = (Omega *
           (-1) ** int(J + I + F) *
           (-1) ** int(J - Omega) *
           math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner6j(I, J, F, Jp, I, 1) *
           wigner3j(J, 1, Jp, -Omega, 0, Omega))

    return float(val)


# =====================================================================
# Operator: Magnetic hyperfine (nuclear spin 2)
# =====================================================================

def magnetic_hyperfine_2(s1: HundsCaseC_LinearMolecule,
                         s2: HundsCaseC_LinearMolecule) -> float:
    """
    Magnetic hyperfine interaction for a second nuclear spin.

    Same form as magnetic_hyperfine_1 but provided as a separate
    operator so the two nuclear spin couplings can carry different
    coefficients.
    """
    return magnetic_hyperfine_1(s1, s2)


# =====================================================================
# Operator: Nuclear spin-rotation (nuclear spin 1)
# =====================================================================

def nuclear_spin_rotation_1(s1: HundsCaseC_LinearMolecule,
                            s2: HundsCaseC_LinearMolecule) -> float:
    """
    Nuclear spin-rotation interaction c_I * I.N projected to case (c).

    In case (c), N = J - L - S is not a good quantum number, so this
    interaction is evaluated as I.J (up to corrections).

    Reference: Brown & Carrington, Section 8.6.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Omega", "J", "M"):
        return 0.0

    J = _f(s1.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)

    if I == 0:
        return 0.0
    if F != Fp:
        return 0.0

    # I.J = (1/2)[F(F+1) - I(I+1) - J(J+1)]
    val = Fraction(1, 2) * (F * (F + 1) - I * (I + 1) - J * (J + 1))
    return float(val)


# =====================================================================
# Operator: Nuclear spin-rotation (nuclear spin 2)
# =====================================================================

def nuclear_spin_rotation_2(s1: HundsCaseC_LinearMolecule,
                            s2: HundsCaseC_LinearMolecule) -> float:
    """
    Nuclear spin-rotation interaction for a second nuclear spin.

    Same functional form as nuclear_spin_rotation_1.
    """
    return nuclear_spin_rotation_1(s1, s2)


# =====================================================================
# Operator: DC Stark Effect
# =====================================================================

def Stark(s1: HundsCaseC_LinearMolecule,
          s2: HundsCaseC_LinearMolecule,
          p: int) -> float:
    """
    DC Stark matrix element for polarisation p = -1, 0, +1.

    H_Stark = -d * E * cos(theta)

    In case (c) the body-frame projection is Omega, so the coupling
    involves a single 3j symbol in (J, 1, J'; -Omega, 0, Omega).

    Reference: Brown & Carrington, Section 9.7.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I", "Omega"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)
    Omega = _f(s1.Omega)

    if abs(F - Fp) > 1 or abs(J - Jp) > 1:
        return 0.0
    if M - Mp != p:
        return 0.0

    val = (-(-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(J - Omega) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner3j(J, 1, Jp, -Omega, 0, Omega))

    return float(val)
