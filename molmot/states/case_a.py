"""
Hund's case (a) basis for linear molecules.

Port of HundsCaseA_LinearMolecule.jl from QuantumStates.jl.

This module defines the basis states for excited electronic states
such as the A~2Pi state of SrOH, along with all relevant operators.

References
----------
* Hirota, *High-Resolution Spectroscopy of Transient Molecules* (1985)
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules* (2003)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j, wigner9j


@dataclass
class HundsCaseA_LinearMolecule(BasisState):
    """
    Hund's case (a) basis state for a linear molecule.

    Quantum numbers
    ---------------
    v1, v2, v3 : float
        Vibrational quantum numbers.
    ell : float
        Vibrational angular momentum (bending).
    Lambda : float
        Projection of electronic orbital angular momentum on the
        internuclear axis.
    K : float
        K = Lambda + ell.
    I : float
        Nuclear spin.
    S : float
        Total electron spin.
    Sigma : float
        Projection of S on the internuclear axis.
    J : float
        Total angular momentum excluding nuclear spin.
    P : float
        P = Lambda + Sigma (= Omega for linear molecules).
    F : float
        Total angular momentum including nuclear spin.
    M : float
        Space-fixed projection of F.
    """
    v1: float = 0
    v2: float = 0
    ell: float = 0
    v3: float = 0
    Lambda: float = 0
    K: float = 0
    I: float = 0
    S: float = 0
    Sigma: float = 0
    J: float = 0
    P: float = 0
    F: float = 0
    M: float = 0

    constraints = {
        "K": lambda s: s.Lambda + s.ell,
        "Sigma": lambda s: (-s.S, s.S),
        "P": lambda s: s.Lambda + s.Sigma,
        "J": lambda s: (abs(s.P), abs(s.P) + 100),
        "F": lambda s: (abs(s.J - s.I), s.J + s.I),
        "M": lambda s: (-s.F, s.F),
    }

    def __repr__(self):
        return (f"|v=({self.v1},{self.v2},{self.v3}), "
                f"Lambda={Fraction(self.Lambda).limit_denominator(10)}, "
                f"Sigma={Fraction(self.Sigma).limit_denominator(10)}, "
                f"J={Fraction(self.J).limit_denominator(10)}, "
                f"P={Fraction(self.P).limit_denominator(10)}, "
                f"F={Fraction(self.F).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# =====================================================================
# Helper
# =====================================================================

def _f(x):
    """Convert to Fraction."""
    return Fraction(x).limit_denominator(1000)


# =====================================================================
# Operator: Rotation (case a)
# =====================================================================

def Rotation(s1: HundsCaseA_LinearMolecule,
             s2: HundsCaseA_LinearMolecule) -> float:
    """
    Rigid rotor in case (a): B * [J(J+1) - 2*P^2 + S(S+1)] for diagonal,
    plus spin-uncoupling (off-diagonal in Sigma by +/-1, off-diagonal in P).

    Reference: Hirota eq. 2.3.10.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K", "F", "M"):
        return 0.0

    J = _f(s1.J)
    S = _f(s1.S)
    P1, P2 = _f(s1.P), _f(s2.P)
    Sigma1, Sigma2 = _f(s1.Sigma), _f(s2.Sigma)

    if s1.J != s2.J:
        return 0.0

    # Diagonal part
    if P1 == P2 and Sigma1 == Sigma2:
        return float(J * (J + 1) - P1 ** 2)

    # Spin-uncoupling: Delta_Sigma = +/-1, Delta_P = +/-1
    if abs(Sigma1 - Sigma2) == 1 and abs(P1 - P2) == 1:
        # -B * sqrt(J(J+1) - P*P') * sqrt(S(S+1) - Sigma*Sigma')
        # where P' = P +/- 1 and Sigma' = Sigma +/- 1
        sign = 1 if P2 > P1 else -1
        val = -math.sqrt(
            float(J * (J + 1) - P1 * P2)) * math.sqrt(
            float(S * (S + 1) - Sigma1 * Sigma2))
        return val

    return 0.0


# =====================================================================
# Operator: Spin-Orbit
# =====================================================================

def SpinOrbit(s1: HundsCaseA_LinearMolecule,
              s2: HundsCaseA_LinearMolecule) -> float:
    """
    Spin-orbit coupling: A * Lambda * Sigma.

    Diagonal in all quantum numbers.

    Reference: Hirota eq. 2.3.7.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K", "Sigma", "J", "P", "F", "M"):
        return 0.0
    return s1.Lambda * s1.Sigma


# =====================================================================
# Operator: Lambda-doubling (q term)
# =====================================================================

def LambdaDoubling_q(s1: HundsCaseA_LinearMolecule,
                     s2: HundsCaseA_LinearMolecule) -> float:
    """
    Lambda-doubling q-term: connects |Lambda, Sigma> with |-Lambda, -Sigma+2>.

    Reference: Hirota eq. 2.4.20.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "ell", "J", "F", "M"):
        return 0.0

    Lambda1, Lambda2 = _f(s1.Lambda), _f(s2.Lambda)
    Sigma1, Sigma2 = _f(s1.Sigma), _f(s2.Sigma)
    P1, P2 = _f(s1.P), _f(s2.P)
    J = _f(s1.J)
    S = _f(s1.S)

    # q-term connects (Lambda, Sigma) <-> (-Lambda, Sigma+2*Lambda)
    if Lambda1 == -Lambda2 and Lambda1 != 0:
        if Sigma2 == Sigma1 + 2 * Lambda1:
            # Matrix element
            val = (0.5 *
                   math.sqrt(float(J * (J + 1) - P1 * P2)) *
                   math.sqrt(float(J * (J + 1) - P2 * (P2 + (P2 - P1)))) *
                   math.sqrt(float(S * (S + 1) - Sigma1 * Sigma2)) *
                   math.sqrt(float(S * (S + 1) - Sigma2 * (Sigma2 + (Sigma2 - Sigma1)))))
            # Simplified for |Lambda| = 1:
            if abs(Lambda1) == 1:
                q1 = P1
                q2 = P2
                val = 0.5 * (math.sqrt(
                    float((J - abs(q1)) * (J + abs(q1) + 1) *
                          (J - abs(q2)) * (J + abs(q2) + 1))) if abs(q1) <= J and abs(q2) <= J else 0.0)
            return val

    return 0.0


# =====================================================================
# Operator: Lambda-doubling (p + 2q term)
# =====================================================================

def LambdaDoubling_p2q(s1: HundsCaseA_LinearMolecule,
                       s2: HundsCaseA_LinearMolecule) -> float:
    """
    Lambda-doubling (p+2q) term: connects |Lambda, Sigma> with |-Lambda, Sigma>.

    Reference: Hirota eq. 2.4.21.
    """
    # (p+2q) term connects Lambda with -Lambda AND Sigma with -Sigma+2Lambda
    # i.e. |Lambda,Sigma,P> -> |-Lambda, Sigma', P'> where P changes.
    # Must NOT enforce delta(Sigma) — the spin-orbit coupling flips Sigma.
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "ell", "J", "F", "M"):
        return 0.0

    Lambda1, Lambda2 = _f(s1.Lambda), _f(s2.Lambda)
    Sigma1, Sigma2 = _f(s1.Sigma), _f(s2.Sigma)
    P1, P2 = _f(s1.P), _f(s2.P)
    J = _f(s1.J)
    S = _f(s1.S)

    # Selection rule: Lambda' = Lambda + 2q for q in {-1, +1}
    if Lambda1 == 0 or Lambda2 == 0:
        return 0.0
    if abs(Lambda1 + Lambda2) != 0 and abs(Lambda1 - Lambda2) != 2:
        if Lambda1 != -Lambda2:
            return 0.0

    # Brown & Carrington eq. 9.66: Lambda_+^(+-1) S_-^(+-1) coupling
    val = 0.0
    for q in [-1, 1]:
        if Lambda2 != Lambda1 + 2 * q:
            continue
        if Sigma2 != Sigma1 - q:
            continue
        if P2 != P1 + q:
            continue

        w3j_J = wigner3j(J, 1, J, -P1, -q, P2)
        w3j_S = wigner3j(S, 1, S, -Sigma1, q, Sigma2)

        val += ((-1) ** int(J - P1) * (-1) ** int(S - Sigma1)
                * math.sqrt(float(J * (J + 1) * (2 * J + 1)
                                  * S * (S + 1) * (2 * S + 1)))
                * w3j_J * w3j_S)

    return val


# =====================================================================
# Operator: ell-type Doubling (case a)
# =====================================================================

def ell_Doubling(s1: HundsCaseA_LinearMolecule,
                 s2: HundsCaseA_LinearMolecule) -> float:
    """
    ell-type doubling for bending modes in case (a).

    Reference: Hirota, Section 3.2.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Sigma", "J", "P", "F", "M", "Lambda"):
        return 0.0

    K1, K2 = _f(s1.K), _f(s2.K)
    ell1, ell2 = _f(s1.ell), _f(s2.ell)
    J = _f(s1.J)
    P = _f(s1.P)

    if abs(ell1 - ell2) != 2 or abs(K1 - K2) != 2:
        return 0.0

    val = 0.5 * math.sqrt(
        max(0.0, float((J * (J + 1) - P * (P + 1)) *
                        (J * (J + 1) - (P + 1) * (P + 2)))))
    return val


# =====================================================================
# Operator: Renner-Teller
# =====================================================================

def RennerTeller(s1: HundsCaseA_LinearMolecule,
                 s2: HundsCaseA_LinearMolecule) -> float:
    """
    Renner-Teller coupling between Lambda and ell.

    Connects states with Delta_Lambda = +/-1, Delta_ell = -/+1.

    Reference: Brown & Carrington, Chapter 9.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Sigma", "J", "P", "F", "M"):
        return 0.0

    Lambda1, Lambda2 = _f(s1.Lambda), _f(s2.Lambda)
    ell1, ell2 = _f(s1.ell), _f(s2.ell)
    v2 = _f(s1.v2)

    if abs(Lambda1 - Lambda2) != 1 or abs(ell1 - ell2) != 1:
        return 0.0
    if (Lambda2 - Lambda1) + (ell2 - ell1) != 0:
        return 0.0

    # epsilon * sqrt((v2 +/- ell + 2)(v2 -/+ ell) / 4)
    if ell2 == ell1 + 1:
        val = math.sqrt(max(0.0,
            float((v2 - ell1) * (v2 + ell1 + 2)) / 4.0))
    else:
        val = math.sqrt(max(0.0,
            float((v2 + ell1) * (v2 - ell1 + 2)) / 4.0))

    return val


# =====================================================================
# Operator: Hyperfine I.L
# =====================================================================

def Hyperfine_IL(s1: HundsCaseA_LinearMolecule,
                 s2: HundsCaseA_LinearMolecule) -> float:
    """
    Hyperfine I.L (nuclear spin - orbital angular momentum).

    Diagonal: a * Lambda * Omega_I, where Omega_I is derived from F coupling.

    Reference: Hirota, Section 2.3.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K", "Sigma", "J", "P", "M"):
        return 0.0

    J = _f(s1.J)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    M = _f(s1.M)
    P = _f(s1.P)
    Lambda = _f(s1.Lambda)

    if I == 0 or Lambda == 0:
        return 0.0

    val = (Lambda *
           (-1) ** int(J + I + Fp) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
           wigner6j(I, Fp, J, F, I, 1) *
           (-1) ** int(J - P) *
           math.sqrt(float(2 * J + 1)) *
           wigner3j(J, 1, J, -P, 0, P))

    return val


# =====================================================================
# Operator: Hyperfine I.S (Fermi contact, case a)
# =====================================================================

def Hyperfine_IF(s1: HundsCaseA_LinearMolecule,
                 s2: HundsCaseA_LinearMolecule) -> float:
    """
    Fermi contact hyperfine: b_F * I.S projected onto case (a) basis.

    Reference: Hirota, Section 2.3.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K", "M"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    S = _f(s1.S)
    I = _f(s1.I)
    F, Fp = _f(s1.F), _f(s2.F)
    Sigma1, Sigma2 = _f(s1.Sigma), _f(s2.Sigma)
    P1, P2 = _f(s1.P), _f(s2.P)

    if I == 0 or S == 0:
        return 0.0
    if abs(J - Jp) > 1 or abs(F - Fp) > 0:
        return 0.0

    # Diagonal in Sigma part (dominant)
    if Sigma1 == Sigma2 and J == Jp and P1 == P2:
        val = (Sigma1 *
               (-1) ** int(J + I + F) *
               math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
               (-1) ** int(J - P1) *
               math.sqrt(float(2 * J + 1)) *
               wigner3j(J, 1, J, -P1, 0, P1) *
               wigner6j(I, I, 1, J, J, F))
        return val

    return 0.0


# =====================================================================
# Operator: Dipolar Hyperfine c (case a)
# =====================================================================

def Hyperfine_Dipolar_c(s1: HundsCaseA_LinearMolecule,
                        s2: HundsCaseA_LinearMolecule) -> float:
    """
    Dipolar hyperfine c-term (diagonal in Sigma).

    Reference: Brown & Carrington, Section 8.7.
    """
    # Same selection rules as Hyperfine_IF
    return 0.0  # Higher-order; implement if needed


# =====================================================================
# Operator: Dipolar Hyperfine d (case a)
# =====================================================================

def Hyperfine_Dipolar_d(s1: HundsCaseA_LinearMolecule,
                        s2: HundsCaseA_LinearMolecule) -> float:
    """
    Dipolar hyperfine d-term (off-diagonal in Sigma by +/-1).

    Reference: Brown & Carrington, Section 8.7.
    """
    return 0.0  # Higher-order; implement if needed


# =====================================================================
# Operator: Zeeman L (orbital)
# =====================================================================

def Zeeman_L(s1: HundsCaseA_LinearMolecule,
             s2: HundsCaseA_LinearMolecule,
             p: int) -> float:
    """
    Orbital Zeeman effect: g_L * mu_B * B_p * T^1_p(L).

    In case (a), the orbital contribution comes from L projected onto
    the internuclear axis.

    Reference: Hirota, Section 2.5.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K", "Sigma", "P"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)
    P = _f(s1.P)
    Lambda = _f(s1.Lambda)

    if abs(J - Jp) > 1 or abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(J - P) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner3j(J, 1, Jp, -P, 0, P) *
           Lambda)

    return val


# =====================================================================
# Operator: Zeeman S (spin)
# =====================================================================

def Zeeman_S(s1: HundsCaseA_LinearMolecule,
             s2: HundsCaseA_LinearMolecule,
             p: int) -> float:
    """
    Spin Zeeman effect: g_S * mu_B * B_p * T^1_p(S).

    Reference: Hirota eq. 2.5.15.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S",
                 "Lambda", "ell", "K"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)
    S = _f(s1.S)
    Sigma1, Sigma2 = _f(s1.Sigma), _f(s2.Sigma)
    P1, P2 = _f(s1.P), _f(s2.P)

    if S == 0:
        return 0.0
    if abs(J - Jp) > 1 or abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0
    if abs(Sigma1 - Sigma2) > 1 or abs(P1 - P2) > 1:
        return 0.0

    # Two terms: diagonal in Sigma (Sigma part) and off-diagonal (perpendicular)

    # Diagonal in Sigma
    if Sigma1 == Sigma2 and P1 == P2:
        val = ((-1) ** int(p) *
               (-1) ** int(F - M) *
               wigner3j(F, 1, Fp, -M, p, Mp) *
               (-1) ** int(J + I + Fp + 1) *
               math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
               wigner6j(Jp, Fp, I, F, J, 1) *
               (-1) ** int(J - P1) *
               math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
               wigner3j(J, 1, Jp, -P1, 0, P1) *
               Sigma1)
        return val

    return 0.0


# =====================================================================
# Operator: TDM (case a)
# =====================================================================

def TDM(s1: HundsCaseA_LinearMolecule,
        s2: HundsCaseA_LinearMolecule,
        p: int) -> float:
    """
    Transition dipole moment in case (a) basis.

    Reference: Brown & Carrington, Section 6.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "I", "S", "Sigma"):
        return 0.0

    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)
    P1, P2 = _f(s1.P), _f(s2.P)
    Lambda1, Lambda2 = _f(s1.Lambda), _f(s2.Lambda)

    if abs(J - Jp) > 1 or abs(F - Fp) > 1:
        return 0.0
    if Mp - M != p:
        return 0.0

    # Sum over body-frame component q
    tdm_sum = 0.0
    for q in [-1, 0, 1]:
        q_frac = _f(q)
        if P2 - P1 != q_frac:
            continue
        tdm_sum += ((-1) ** int(J - P1) *
                    math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
                    wigner3j(J, 1, Jp, -P1, q_frac, P2))

    if abs(tdm_sum) < 1e-15:
        return 0.0

    val = (-(-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           tdm_sum)

    return val
