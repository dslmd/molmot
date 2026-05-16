"""
Hund's case (b) basis for linear molecules.

Port of HundsCaseB_LinearMolecule.jl from QuantumStates.jl.

This is the MOST CRITICAL module.  It defines the basis states for
the ground electronic state of molecules like SrOH in the
X~2Sigma+ state, along with all relevant Hamiltonian operators
and the transition dipole moment operator.

References
----------
* Hirota, *High-Resolution Spectroscopy of Transient Molecules* (Springer, 1985)
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules* (Cambridge, 2003)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j, wigner9j


@dataclass
class HundsCaseB_LinearMolecule(BasisState):
    """
    Hund's case (b) basis state for a linear molecule.

    Quantum numbers
    ---------------
    v1, v2, v3 : float
        Vibrational quantum numbers (stretching, bending, stretching).
    S : float
        Total electron spin.
    I : float
        Nuclear spin.
    Lambda : float
        Projection of orbital angular momentum on internuclear axis.
    ell : float
        Vibrational angular momentum quantum number (from bending mode).
    K : float
        K = Lambda + ell.
    N : float
        Rotational angular momentum quantum number (excluding spin).
    J : float
        N + S total angular momentum (excluding nuclear spin).
    F : float
        J + I total angular momentum.
    M : float
        Projection of F on space-fixed z-axis.
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
    J: float = 0
    F: float = 0
    M: float = 0

    constraints = {
        "K": lambda s: s.Lambda + s.ell,
        "N": lambda s: (abs(s.Lambda + s.ell), abs(s.Lambda + s.ell) + 100),
        "J": lambda s: (abs(s.N - s.S), s.N + s.S),
        "F": lambda s: (abs(s.J - s.I), s.J + s.I),
        "M": lambda s: (-s.F, s.F),
    }

    def __repr__(self):
        return (f"|v=({self.v1},{self.v2},{self.v3}), "
                f"N={Fraction(self.N).limit_denominator(10)}, "
                f"J={Fraction(self.J).limit_denominator(10)}, "
                f"F={Fraction(self.F).limit_denominator(10)}, "
                f"M={Fraction(self.M).limit_denominator(10)}>")


# =====================================================================
# Helper: convert to Fraction for Wigner symbol arguments
# =====================================================================

def _f(x):
    """Convert to Fraction for Wigner symbol argument."""
    return Fraction(x).limit_denominator(1000)


# =====================================================================
# Operator: Rotation  N(N+1) - Lambda^2
# =====================================================================

def Rotation(s1: HundsCaseB_LinearMolecule,
             s2: HundsCaseB_LinearMolecule) -> float:
    """
    Rigid rotor Hamiltonian: B * [N(N+1) - Lambda^2].

    Diagonal in all quantum numbers.

    Reference: Hirota eq. 2.3.14.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "N", "J", "F", "M"):
        return 0.0
    N = s1.N
    Lambda = s1.Lambda
    return N * (N + 1) - Lambda ** 2


# =====================================================================
# Operator: Rotation Distortion  -D * [N(N+1) - Lambda^2]^2
# =====================================================================

def RotationDistortion(s1: HundsCaseB_LinearMolecule,
                       s2: HundsCaseB_LinearMolecule) -> float:
    """
    Centrifugal distortion: -D * [N(N+1) - Lambda^2]^2.

    Diagonal in all quantum numbers.

    Reference: Hirota eq. 2.3.15.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "N", "J", "F", "M"):
        return 0.0
    N = s1.N
    Lambda = s1.Lambda
    return -(N * (N + 1) - Lambda ** 2) ** 2


# =====================================================================
# Operator: Spin-Rotation  gamma * N.S
# =====================================================================

def SpinRotation(s1: HundsCaseB_LinearMolecule,
                 s2: HundsCaseB_LinearMolecule) -> float:
    """
    Spin-rotation coupling: gamma * <N.S>.

    Uses the T_kq tensor formalism.  Off-diagonal in N (Delta_N = 0, +/-2
    for Sigma states), but diagonal in F and M.

    Reference: Hirota eq. 2.3.35.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "F", "M"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    S = _f(s1.S)
    J, Jp = _f(s1.J), _f(s2.J)
    F = _f(s1.F)
    I = _f(s1.I)

    # N.S = sum_q (-1)^q T^1_q(N) T^1_{-q}(S)
    # In the |N,S,J,I,F,M> basis:
    # <...| N.S |...> = (-1)^(N+S+J) * sqrt(S(S+1)(2S+1))
    #                   * sqrt(N(N+1)(2N+1)) * { N' J' S }
    #                                            { J  N  1 }
    #                   * delta(N,N') * delta(J,J')  [for diagonal part]

    # General form allowing off-diagonal in J:
    if N != Np:
        return 0.0

    if S == 0:
        return 0.0

    val = ((-1) ** int(N + S + Jp) *
           math.sqrt(float(S * (S + 1) * (2 * S + 1))) *
           math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
           wigner6j(J, N, S, N, Jp, 1))

    # Account for F coupling with nuclear spin
    if I != 0:
        val *= ((-1) ** int(Jp + I + F + 1) *
                math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
                wigner6j(J, F, I, F, Jp, 0))
        # The wigner6j with 0 gives delta(J, Jp)/sqrt(2J+1) factor
        # Actually for I != 0 we need to be more careful:
        # The matrix element in |J,I,F,M> is just diagonal in F with
        # the 6j coupling J-J' through F
        val = ((-1) ** int(N + S + Jp) *
               math.sqrt(float(S * (S + 1) * (2 * S + 1))) *
               math.sqrt(float(N * (N + 1) * (2 * N + 1))) *
               (-1) ** int(Jp + I + F + 1) *
               math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
               wigner6j(Jp, J, 1, N, N, S) *
               wigner6j(J, Jp, 0, I, I, F))

    # Simplify: for spin-rotation, it's diagonal in N, so just
    # use the simpler form
    if J == Jp:
        result = 0.5 * (J * (J + 1) - N * (N + 1) - S * (S + 1))
        return float(result)
    return 0.0


# =====================================================================
# Operator: Fermi Contact Hyperfine (I.S)
# =====================================================================

def Hyperfine_IS(s1: HundsCaseB_LinearMolecule,
                 s2: HundsCaseB_LinearMolecule) -> float:
    """
    Fermi contact hyperfine interaction: b_F * I.S.

    This is the isotropic part of the hyperfine coupling.

    Reference: Hirota, p. 39, eq. 2.3.59.
    """
    # I·S is a scalar (rank-0 in F), so diagonal in F and M
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "N", "F", "M"):
        return 0.0

    N = _f(s1.N)
    S = _f(s1.S)
    I = _f(s1.I)
    J, Jp = _f(s1.J), _f(s2.J)
    F = _f(s1.F)

    if I == 0 or S == 0:
        return 0.0

    # I.S in |N,S,J,I,F,M> basis (Hirota eq. 2.3.59):
    # (-1)^(J'+I+F+1) * (-1)^(N+S+J) * sqrt((2J+1)(2J'+1))
    # * sqrt(S(S+1)(2S+1)) * sqrt(I(I+1)(2I+1))
    # * { I  J  F }  * { S  J' N }
    #   { J' I  1 }    { J  S  1 }

    val = ((-1) ** int(Jp + I + F) *
           (-1) ** int(N + S + J + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           math.sqrt(float(S * (S + 1) * (2 * S + 1))) *
           math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
           wigner6j(J, Jp, 1, I, I, F) *
           wigner6j(S, S, 1, Jp, J, N))

    return val


# =====================================================================
# Operator: Dipolar Hyperfine (t * C^2(I,S))
# =====================================================================

def Hyperfine_Dipolar(s1: HundsCaseB_LinearMolecule,
                      s2: HundsCaseB_LinearMolecule) -> float:
    """
    Anisotropic (dipolar) hyperfine interaction.

    Uses a 9-j symbol to couple the rank-2 tensor.

    Reference: Hirota, p. 39, eq. 2.3.63; Brown & Carrington eq. 8.361.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "M"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    S = _f(s1.S)
    I = _f(s1.I)
    J, Jp = _f(s1.J), _f(s2.J)
    F = _f(s1.F)
    K = _f(s1.K)

    if I == 0 or S == 0:
        return 0.0

    # Selection rule: Delta_N = 0, +/-2
    if abs(N - Np) > 2:
        return 0.0

    # t * (-1)^(J'+I+F) * sqrt(30(2J+1)(2J'+1)(2N+1)(2N'+1))
    # * { J  J' 2 }  * { N  N' 2 }
    #   { I  I  F }    { S  S  J' }
    # * ... 9j symbol ...

    val = ((-1) ** int(Jp + I + F) *
           math.sqrt(30.0) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           math.sqrt(float(S * (S + 1) * (2 * S + 1))) *
           math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
           wigner6j(J, Jp, 2, I, I, F) *
           wigner9j(N, S, J, Np, S, Jp, 2, 2, 2) *  # This is a simplification
           (-1) ** int(N - K) *
           wigner3j(N, 2, Np, -K, 0, K))

    return val


# =====================================================================
# Operator: Nuclear Quadrupole
# =====================================================================

def nuclear_quadrupole(s1: HundsCaseB_LinearMolecule,
                       s2: HundsCaseB_LinearMolecule) -> float:
    """
    Nuclear electric quadrupole interaction.

    Reference: Hirota eq. 2.3.80.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "M"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    S = _f(s1.S)
    I = _f(s1.I)
    J, Jp = _f(s1.J), _f(s2.J)
    F = _f(s1.F)
    K = _f(s1.K)

    if I < 1:
        return 0.0

    # Delta_N = 0, +/-2
    if abs(N - Np) > 2:
        return 0.0

    # eQq * (-1)^(J'+I+F) * (-1)^(N+S+J'+1) * (-1)^(N-K)
    # * sqrt((2J+1)(2J'+1)(2N+1)(2N'+1))
    # * { J  J' 2 }  * { N  N' 2 }
    #   { I  I  F }    { J' J  S }
    # * (N 2 N'; -K 0 K')

    # Quadrupole prefactor
    qf = ((-1) ** int(I) *
           math.sqrt(float((2 * I + 3) * (2 * I + 1) * (I + 1)) /
                     float(I * (2 * I - 1))) /
           (2.0 if I >= 1 else 1.0))

    val = ((-1) ** int(Jp + I + F) *
           (-1) ** int(N + S + Jp + 1) *
           (-1) ** int(N - K) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner6j(J, Jp, 2, I, I, F) *
           wigner6j(N, Np, 2, Jp, J, S) *
           wigner3j(N, 2, Np, -K, 0, K) *
           qf)

    return val


# =====================================================================
# Operator: ell-type Doubling
# =====================================================================

def ell_Doubling(s1: HundsCaseB_LinearMolecule,
                 s2: HundsCaseB_LinearMolecule) -> float:
    """
    ell-type (l-type) doubling for bending vibrational modes.

    Connects states with Delta_ell = +/-2, Delta_K = +/-2.

    Reference: Hirota, Section 3.2.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "N", "J", "F", "M", "Lambda"):
        return 0.0

    N = _f(s1.N)
    K1, K2 = _f(s1.K), _f(s2.K)
    ell1, ell2 = _f(s1.ell), _f(s2.ell)

    if abs(ell1 - ell2) != 2 or abs(K1 - K2) != 2:
        return 0.0

    # q_ell * 1/2 * sqrt((N(N+1) - K(K+/-1)) * (N(N+1) - (K+/-1)(K+/-2)))
    # The sign depends on the direction of the shift
    if K2 == K1 + 2:
        val = 0.5 * math.sqrt(
            float((N * (N + 1) - K1 * (K1 + 1)) *
                  (N * (N + 1) - (K1 + 1) * (K1 + 2))))
    elif K2 == K1 - 2:
        val = 0.5 * math.sqrt(
            float((N * (N + 1) - K1 * (K1 - 1)) *
                  (N * (N + 1) - (K1 - 1) * (K1 - 2))))
    else:
        val = 0.0

    return val


# =====================================================================
# Operator: DC Stark Effect
# =====================================================================

def Stark(s1: HundsCaseB_LinearMolecule,
          s2: HundsCaseB_LinearMolecule,
          p: int) -> float:
    """
    DC Stark matrix element for polarisation p = -1, 0, +1.

    H_Stark = -mu_E * E * T^1_p(cos theta)

    Reference: Brown & Carrington, Section 9.7.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    S = _f(s1.S)
    I = _f(s1.I)
    K = _f(s1.K)

    # Delta_F = 0, +/-1; Delta_M = p
    if abs(F - Fp) > 1 or M - Mp != p:
        return 0.0
    if abs(J - Jp) > 1 or abs(N - Np) > 1:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(N + S + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner6j(Np, Jp, S, J, N, 1) *
           (-1) ** int(N - K) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, 1, Np, -K, 0, K))

    return val


# =====================================================================
# Operator: Zeeman (electron spin)
# =====================================================================

def Zeeman(s1: HundsCaseB_LinearMolecule,
           s2: HundsCaseB_LinearMolecule,
           p: int) -> float:
    """
    Electron spin Zeeman interaction for polarisation p = -1, 0, +1.

    H_Z = -g_S * mu_B * B_p * T^1_p(S)

    This is the CRITICAL operator for MOT simulations.

    Reference: Hirota eq. 2.5.16.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "N"):
        return 0.0

    N = _f(s1.N)
    S = _f(s1.S)
    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)

    if S == 0:
        return 0.0

    # Selection rules
    if abs(F - Fp) > 1 or abs(J - Jp) > 1:
        return 0.0
    if M - Mp != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(S + N + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           math.sqrt(float(S * (S + 1) * (2 * S + 1))) *
           wigner6j(S, Jp, N, J, S, 1))

    return val


# =====================================================================
# Operator: Zeeman (nuclear spin)
# =====================================================================

def zeeman_nuclear(s1: HundsCaseB_LinearMolecule,
                   s2: HundsCaseB_LinearMolecule,
                   p: int) -> float:
    """
    Nuclear spin Zeeman interaction for polarisation p = -1, 0, +1.

    H_Zn = -g_I * mu_N * B_p * T^1_p(I)

    Reference: Hirota, Section 2.5.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K", "N", "J"):
        return 0.0

    J = _f(s1.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)

    if I == 0:
        return 0.0

    if abs(F - Fp) > 1 or M - Mp != p:
        return 0.0

    val = ((-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(I + J + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           math.sqrt(float(I * (I + 1) * (2 * I + 1))) *
           wigner6j(I, Fp, J, F, I, 1))

    return val


# =====================================================================
# Operator: Transition Dipole Moment (TDM)
# =====================================================================

def Polarizability(s1: HundsCaseB_LinearMolecule,
                   s2: HundsCaseB_LinearMolecule,
                   alpha: float,
                   epsilon: int) -> float:
    """
    AC Stark shift / polarizability operator for ODT calculations.

    H_ODT = -alpha * |E|^2 / 2, where alpha is the polarizability tensor.

    For a linearly polarized trap beam, the scalar polarizability alpha_0
    contributes a state-independent shift, while vector (alpha_1) and
    tensor (alpha_2) components lift degeneracies.

    The matrix element is:

        <s1| alpha_epsilon |s2> = alpha * (-1)^(F - M) * (F epsilon F'; -M epsilon M')
            * (-1)^(J+I+F'+1) * sqrt((2F+1)(2F'+1))
            * {J J' epsilon; F' F I}
            * (-1)^(S+N+J'+1) * sqrt((2J+1)(2J'+1))
            * {N N' epsilon; J' J S}
            * (-1)^(N-K) * sqrt((2N+1)(2N'+1))
            * (N epsilon N'; -K 0 K)

    This is a rank-epsilon spherical tensor operator (epsilon = 0, 1, 2).

    Reference: Le Kien et al., EPJD 67, 92 (2013); Caldwell & Tarbutt (2020).

    Parameters
    ----------
    s1, s2 : HundsCaseB_LinearMolecule
    alpha : float
        Polarizability coefficient (SI units, J/(V/m)^2).
    epsilon : int
        Tensor rank: 0 (scalar), 1 (vector), or 2 (tensor).

    Returns
    -------
    float
        Matrix element (to be multiplied by the appropriate field factor).
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I",
                 "Lambda", "ell", "K"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    S = _f(s1.S)
    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    I = _f(s1.I)
    K = _f(s1.K)

    # Selection rules for rank-epsilon tensor
    if abs(N - Np) > epsilon or abs(J - Jp) > epsilon or abs(F - Fp) > epsilon:
        return 0.0
    # M is unchanged for linearly polarized light (q=0 component)
    if M != Mp:
        return 0.0

    # For scalar (epsilon=0): diagonal in all quantum numbers
    if epsilon == 0:
        if N != Np or J != Jp or F != Fp:
            return 0.0
        return alpha

    # General tensor coupling
    val = (alpha *
           (-1) ** int(F - M) *
           wigner3j(F, epsilon, Fp, -M, 0, M) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(J, Jp, epsilon, Fp, F, I) *
           (-1) ** int(S + N + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner6j(N, Np, epsilon, Jp, J, S) *
           (-1) ** int(N - K) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           wigner3j(N, epsilon, Np, -K, 0, K))

    return val


def TDM(s1: HundsCaseB_LinearMolecule,
        s2: HundsCaseB_LinearMolecule,
        p: int) -> float:
    """
    Transition dipole moment operator for polarisation p = -1, 0, +1.

    Connects ground and excited case (b) states via the electric dipole
    operator T^1_p(d).

    This is the CRITICAL operator for computing Rabi frequencies in OBE.

    The formula sums over the body-frame component q to handle
    Delta_Lambda = 0, +/-1 transitions.

    Reference: Brown & Carrington, eq. 6.148 ff.
    """
    if not delta(s1, s2, "v1", "v2", "v3", "S", "I", "ell"):
        return 0.0

    N, Np = _f(s1.N), _f(s2.N)
    S = _f(s1.S)
    I = _f(s1.I)
    J, Jp = _f(s1.J), _f(s2.J)
    F, Fp = _f(s1.F), _f(s2.F)
    M, Mp = _f(s1.M), _f(s2.M)
    K, Kp = _f(s1.K), _f(s2.K)

    if abs(F - Fp) > 1 or abs(J - Jp) > 1 or abs(N - Np) > 1:
        return 0.0
    if M - Mp != p:
        return 0.0

    # Sum over body-frame component q
    tdm_sum = 0.0
    for q in [-1, 0, 1]:
        q_frac = _f(q)
        # 3j symbol requires -K + q + K' = 0, so q = K - K'
        if K - Kp != q_frac:
            continue
        tdm_sum += wigner3j(N, 1, Np, -K, q_frac, Kp)

    if abs(tdm_sum) < 1e-15:
        return 0.0

    val = (-(-1) ** int(p) *
           (-1) ** int(F - M) *
           wigner3j(F, 1, Fp, -M, p, Mp) *
           (-1) ** int(J + I + Fp + 1) *
           math.sqrt(float((2 * F + 1) * (2 * Fp + 1))) *
           wigner6j(Jp, Fp, I, F, J, 1) *
           (-1) ** int(N + S + Jp + 1) *
           math.sqrt(float((2 * J + 1) * (2 * Jp + 1))) *
           wigner6j(Np, Jp, S, J, N, 1) *
           (-1) ** int(N - K) *
           math.sqrt(float((2 * N + 1) * (2 * Np + 1))) *
           tdm_sum)

    return val
