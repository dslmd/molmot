"""
Asymmetric top molecule basis and operators.

Port of AsymmetricTopMolecule.jl from QuantumStates.jl.

An asymmetric top molecule has three distinct rotational constants
A, B, C.  The Hamiltonian is written as a sum of rank-0 and rank-2
spherical tensors in the body-fixed angular momentum.  The tensors
T_A, T_B, T_C encode the coefficients for each rotational constant.

References
----------
* Comaker et al. (1973).
* Zare, *Angular Momentum* (Wiley, 1988), Chapter 6.
* Brown & Carrington, *Rotational Spectroscopy of Diatomic Molecules*
  (Cambridge, 2003), Chapter 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .basis import BasisState, delta
from ..wigner import wigner3j, wigner6j


@dataclass
class AsymmetricTopMolecule(BasisState):
    """
    Asymmetric top molecule basis state.

    Quantum numbers
    ---------------
    v1, v2, v3 : float
        Vibrational quantum numbers.
    S : float
        Total electron spin.
    I : float
        Nuclear spin.
    N : float
        Rotational angular momentum quantum number.
    K : float
        Projection of N on body-fixed z-axis (-N <= K <= N).
    J : float
        Total angular momentum J = N + S.
    F : float
        Total angular momentum F = J + I.
    M : float
        Projection of F on space-fixed z-axis.
    """
    v1: float = 0
    v2: float = 0
    v3: float = 0
    S: float = 0
    I: float = 0
    N: float = 0
    K: float = 0
    J: float = 0
    F: float = 0
    M: float = 0

    constraints = {
        "K": lambda s: (-s.N, s.N),
        "J": lambda s: (abs(s.N - s.S), s.N + s.S),
        "F": lambda s: (abs(s.J - s.I), s.J + s.I),
        "M": lambda s: (-s.F, s.F),
    }

    def __repr__(self):
        return (f"|v=({self.v1},{self.v2},{self.v3}), "
                f"N={Fraction(self.N).limit_denominator(10)}, "
                f"K={Fraction(self.K).limit_denominator(10)}, "
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
# Rotational constant tensors
# =====================================================================
#
# The rotational Hamiltonian for an asymmetric top is:
#   H_rot = A * N_a^2 + B * N_b^2 + C * N_c^2
#
# This can be rewritten using irreducible spherical tensors of rank 0
# and rank 2 in the body-fixed frame.
#
# The tensors T_Rotational, T_A, T_B, T_C encode the coefficients
# for each rotational constant.  They are indexed as T[k+1, q+3]
# where k is the tensor rank (0 or 2) and q runs from -2 to +2.
#
# Based on Comaker et al. (1973), the following matrices are used:
#
#   T_Rotational[q+2, k+1] for q in {-1, 0, 1} and k in {0, 1, 2}:
#
#       Row 0 (q=-1): [0,  0,  -1/sqrt(3),  0,  0]
#       Row 1 (q= 0): [0,  0,   0,          0,  0]
#       Row 2 (q=+1): [1/sqrt(6), 0, 0,     0, 1/2]
#
# The T_A, T_B, T_C are formed by element-wise multiplication with
# mask matrices specific to each principal axis.

# T_Rotational stored as T[q_idx][k_idx] where q_idx = q+1, k_idx = k
# q ranges over {-1, 0, 1}, k ranges over {0, 1, 2}
# In the Julia code: T_Rotational is a 3x5 matrix indexed [q+2, k+1]
# but the 5 columns correspond to the (k,q_body) = (0,0),(1,-1),(1,0),(1,1),(2,...)
# Actually the Julia code has rows indexed by q_body in {-1,0,1} and
# columns indexed by k in {0,1,2}, and within each k block, the different
# q_body components.
#
# Re-examining the Julia code:
# T_Rotational is 3 rows x 5 columns.
# Indexing: T_Rotational[k+1, q+3] where k in {0,1,2}, q in {-2,-1,0,1,2}
# Actually it's T_kq[q+2, k+1] — 3 rows (q=-1,0,1), 3 cols (k=0,1,2)
# But the AsymmetricTopMolecule.jl uses a different tensor.
#
# Let me re-read the Julia code carefully:
# T_Rotational is indexed as T_Rotational[row, col] with:
#   3 rows, 5 columns
# The sum in Rotation uses: T[k+1, q+3] for k in 0:2, q in -2:2
#
# So T_Rotational has shape (3, 5): rows are k=0,1,2; cols are q=-2,-1,0,1,2

def _make_T_Rotational():
    """
    Construct the base rotational tensor T_Rotational.

    T_Rotational[k][q+2] for k in {0,1,2} and q in {-2,-1,0,1,2}.
    """
    # 3 rows (k=0,1,2), 5 columns (q=-2,-1,0,1,2)
    T = [[0.0] * 5 for _ in range(3)]
    # k=0, q=0 => col index 2
    T[0][2] = -1.0 / math.sqrt(3.0)
    # k=1: all zero
    # k=2, q=-2 => col index 0
    T[2][0] = 1.0 / math.sqrt(6.0)
    # k=2, q=0 => col index 2: 0 (already)
    # k=2, q=+2 => col index 4
    T[2][4] = 0.5
    return T


def _make_T_mask(mask_type):
    """
    Construct mask matrices for T_A, T_B, T_C.

    Each mask is a 3x5 matrix element-wise multiplied with T_Rotational.
    """
    # Mask indexed as [k][q+2]
    mask = [[0.0] * 5 for _ in range(3)]

    if mask_type == "A":
        # From Julia: [0 0 1 0 0; 0 0 0 0 0; 0 0 2 0 0]
        mask[0][2] = 1.0
        mask[2][2] = 2.0
    elif mask_type == "B":
        # From Julia: [0 0 1 0 0; 0 0 0 0 0; 0 -1 0 1 0]
        mask[0][2] = 1.0
        mask[2][1] = -1.0  # q=-1
        mask[2][3] = 1.0   # q=+1
    elif mask_type == "C":
        # From Julia: [0 0 1 0 0; 0 0 0 0 0; 0 -1 0 -1 0]
        mask[0][2] = 1.0
        mask[2][1] = -1.0  # q=-1
        mask[2][3] = -1.0  # q=+1
    return mask


def _elementwise_multiply(T, mask):
    """Element-wise multiply two 3x5 matrices."""
    result = [[0.0] * 5 for _ in range(3)]
    for k in range(3):
        for q in range(5):
            result[k][q] = T[k][q] * mask[k][q]
    return result


_T_Rotational = _make_T_Rotational()
T_A = _elementwise_multiply(_T_Rotational, _make_T_mask("A"))
T_B = _elementwise_multiply(_T_Rotational, _make_T_mask("B"))
T_C = _elementwise_multiply(_T_Rotational, _make_T_mask("C"))


# =====================================================================
# Operator: Rotation
# =====================================================================

def Rotation(s1: AsymmetricTopMolecule,
             s2: AsymmetricTopMolecule,
             T) -> float:
    """
    General rotational Hamiltonian for an asymmetric top molecule.

    The rotation operator is parameterised by the tensor T, which
    should be one of T_A, T_B, or T_C (or a linear combination).

    H_rot = sum_{k=0}^{2} sum_{q=-k}^{k}
            (-1)^{N'-K} * N(N+1)(2N+1) * T[k][q+2]
            * W3j(N, k, N, -K, q, K')
            * (-1)^k * sqrt(2k+1) * W6j(N, N, 1, k, 1, N)

    Diagonal in N, J, F, M.

    Parameters
    ----------
    s1, s2 : AsymmetricTopMolecule
    T : list of lists
        3x5 tensor: T[k][q+2] for k in {0,1,2}, q in {-2,...,2}.

    Returns
    -------
    float
    """
    N = _f(s1.N)
    Np = _f(s2.N)
    J = _f(s1.J)
    Jp = _f(s2.J)
    F = _f(s1.F)
    Fp = _f(s2.F)
    M = _f(s1.M)
    Mp = _f(s2.M)
    K = _f(s1.K)
    Kp = _f(s2.K)

    if N != Np or J != Jp or F != Fp or M != Mp:
        return 0.0

    val = 0.0
    for k in range(3):  # k = 0, 1, 2
        for q_idx in range(5):  # q = -2, -1, 0, 1, 2
            q = q_idx - 2
            t_val = T[k][q_idx]
            if abs(t_val) < 1e-15:
                continue

            w3j = wigner3j(N, k, N, -K, q, Kp)
            if abs(w3j) < 1e-15:
                continue

            w6j = wigner6j(N, N, 1, k, 1, N)

            val += (t_val * w3j *
                    (-1) ** int(k) *
                    math.sqrt(float(2 * k + 1)) *
                    w6j)

    val *= ((-1) ** int(Np - K) *
            float(N * (N + 1) * (2 * N + 1)))

    return float(val)


def Rotation_A(s1: AsymmetricTopMolecule,
               s2: AsymmetricTopMolecule) -> float:
    """Rotation about the a-axis (largest rotational constant A)."""
    return Rotation(s1, s2, T_A)


def Rotation_B(s1: AsymmetricTopMolecule,
               s2: AsymmetricTopMolecule) -> float:
    """Rotation about the b-axis (intermediate rotational constant B)."""
    return Rotation(s1, s2, T_B)


def Rotation_C(s1: AsymmetricTopMolecule,
               s2: AsymmetricTopMolecule) -> float:
    """Rotation about the c-axis (smallest rotational constant C)."""
    return Rotation(s1, s2, T_C)
