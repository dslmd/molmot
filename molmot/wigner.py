"""
Wigner 3-j, 6-j, and 9-j symbols using the Racah formula.

Port of WignerSymbols_Simple.jl by Christian Hallas.
Uses ``fractions.Fraction`` internally for exact half-integer arithmetic
and ``functools.lru_cache`` for performance.

All angular-momentum arguments may be ``int``, ``float``, or ``Fraction``.
Half-integer values such as 0.5, 1.5, ... are accepted.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction
from typing import Union

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Number = Union[int, float, Fraction]


def _to_frac(x: _Number) -> Fraction:
    """Convert a number to an exact Fraction (handles 0.5, 1.5, …)."""
    if isinstance(x, Fraction):
        return x
    # Use limit_denominator to cleanly convert e.g. 0.5 -> 1/2
    return Fraction(x).limit_denominator(1000)


def _is_integer(f: Fraction) -> bool:
    return f.denominator == 1


def _fac(n: Fraction) -> int:
    """Factorial of a non-negative integer (given as Fraction)."""
    if not _is_integer(n) or n < 0:
        raise ValueError(f"factorial requires non-negative integer, got {n}")
    return math.factorial(int(n))


def _triangle_coefficient(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    """
    Triangle coefficient Delta(a,b,c).

    Delta(a,b,c) = factorial(a+b-c)*factorial(a-b+c)*factorial(-a+b+c)
                   / factorial(a+b+c+1)

    Returns exact Fraction.
    """
    s1 = a + b - c
    s2 = a - b + c
    s3 = -a + b + c
    s4 = a + b + c + 1
    if s1 < 0 or s2 < 0 or s3 < 0:
        return Fraction(0)
    if not (_is_integer(s1) and _is_integer(s2) and _is_integer(s3) and _is_integer(s4)):
        return Fraction(0)
    return Fraction(_fac(s1) * _fac(s2) * _fac(s3), _fac(s4))


def _check_triangle(j1: Fraction, j2: Fraction, j3: Fraction) -> bool:
    """Return True if (j1, j2, j3) satisfies the triangle inequality."""
    s1 = j1 + j2 - j3
    s2 = j1 - j2 + j3
    s3 = -j1 + j2 + j3
    return s1 >= 0 and s2 >= 0 and s3 >= 0 and _is_integer(s1)


# ---------------------------------------------------------------------------
# Wigner 3-j symbol
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def wigner3j(j1: _Number, j2: _Number, j3: _Number,
             m1: _Number, m2: _Number, m3: _Number) -> float:
    r"""
    Wigner 3-j symbol using the Racah formula.

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    Parameters
    ----------
    j1, j2, j3 : half-integer angular momenta
    m1, m2, m3 : projection quantum numbers

    Returns
    -------
    float
        Value of the 3-j symbol.

    References
    ----------
    Racah (1942); Edmonds, *Angular Momentum in Quantum Mechanics*, eq. 3.7.3.
    """
    j1, j2, j3 = _to_frac(j1), _to_frac(j2), _to_frac(j3)
    m1, m2, m3 = _to_frac(m1), _to_frac(m2), _to_frac(m3)

    # Selection rules
    if m1 + m2 + m3 != 0:
        return 0.0
    if not _check_triangle(j1, j2, j3):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    # Summation bounds
    t_min = max(Fraction(0), j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    if t_min > t_max:
        return 0.0

    # Triangle coefficient
    tri = _triangle_coefficient(j1, j2, j3)

    # Phase
    phase_exp = j1 - j2 - m3
    if not _is_integer(phase_exp):
        return 0.0
    phase = (-1) ** int(phase_exp)

    # Prefactor
    prefactor_num = _fac(j1 + m1) * _fac(j1 - m1) * \
                    _fac(j2 + m2) * _fac(j2 - m2) * \
                    _fac(j3 + m3) * _fac(j3 - m3)
    prefactor = float(tri) * prefactor_num

    # Sum over t
    total = Fraction(0)
    t = t_min
    while t <= t_max:
        denom = (_fac(t) *
                 _fac(j1 + j2 - j3 - t) *
                 _fac(j1 - m1 - t) *
                 _fac(j2 + m2 - t) *
                 _fac(j3 - j2 + m1 + t) *
                 _fac(j3 - j1 - m2 + t))
        sign = (-1) ** int(t)
        total += Fraction(sign, denom)
        t += 1

    result = phase * math.sqrt(prefactor) * float(total)
    return result


# ---------------------------------------------------------------------------
# Wigner 6-j symbol
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def wigner6j(j1: _Number, j2: _Number, j3: _Number,
             J1: _Number, J2: _Number, J3: _Number) -> float:
    r"""
    Wigner 6-j symbol using the Racah formula.

    .. math::

        \begin{Bmatrix} j_1 & j_2 & j_3 \\ J_1 & J_2 & J_3 \end{Bmatrix}

    Parameters
    ----------
    j1, j2, j3, J1, J2, J3 : half-integer angular momenta

    Returns
    -------
    float

    References
    ----------
    Edmonds, eq. 6.2.4; Varshalovich et al., ch. 9.
    """
    j1, j2, j3 = _to_frac(j1), _to_frac(j2), _to_frac(j3)
    J1, J2, J3 = _to_frac(J1), _to_frac(J2), _to_frac(J3)

    # Triangle conditions
    if not (_check_triangle(j1, j2, j3) and
            _check_triangle(j1, J2, J3) and
            _check_triangle(J1, j2, J3) and
            _check_triangle(J1, J2, j3)):
        return 0.0

    tri = (_triangle_coefficient(j1, j2, j3) *
           _triangle_coefficient(j1, J2, J3) *
           _triangle_coefficient(J1, j2, J3) *
           _triangle_coefficient(J1, J2, j3))

    # Summation bounds
    t_min = max(j1 + j2 + j3,
                j1 + J2 + J3,
                J1 + j2 + J3,
                J1 + J2 + j3)
    t_max = min(j1 + j2 + J1 + J2,
                j2 + j3 + J2 + J3,
                j1 + j3 + J1 + J3)

    if t_min > t_max:
        return 0.0

    total = Fraction(0)
    t = t_min
    while t <= t_max:
        num = (-1) ** int(t) * _fac(t + 1)
        denom = (_fac(t - j1 - j2 - j3) *
                 _fac(t - j1 - J2 - J3) *
                 _fac(t - J1 - j2 - J3) *
                 _fac(t - J1 - J2 - j3) *
                 _fac(j1 + j2 + J1 + J2 - t) *
                 _fac(j2 + j3 + J2 + J3 - t) *
                 _fac(j1 + j3 + J1 + J3 - t))
        total += Fraction(num, denom)
        t += 1

    return math.sqrt(float(tri)) * float(total)


# ---------------------------------------------------------------------------
# Wigner 9-j symbol
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def wigner9j(j1: _Number, j2: _Number, j3: _Number,
             j4: _Number, j5: _Number, j6: _Number,
             j7: _Number, j8: _Number, j9: _Number) -> float:
    r"""
    Wigner 9-j symbol via sum over products of 6-j symbols.

    .. math::

        \begin{Bmatrix}
        j_1 & j_2 & j_3 \\
        j_4 & j_5 & j_6 \\
        j_7 & j_8 & j_9
        \end{Bmatrix}
        = \sum_x (-1)^{2x}(2x+1)
          \begin{Bmatrix} j_1 & j_4 & j_7 \\ j_8 & j_9 & x \end{Bmatrix}
          \begin{Bmatrix} j_2 & j_5 & j_8 \\ j_4 & x & j_6 \end{Bmatrix}
          \begin{Bmatrix} j_3 & j_6 & j_9 \\ x & j_1 & j_2 \end{Bmatrix}

    References
    ----------
    Varshalovich et al., eq. 10.2.1.
    """
    j1, j2, j3 = _to_frac(j1), _to_frac(j2), _to_frac(j3)
    j4, j5, j6 = _to_frac(j4), _to_frac(j5), _to_frac(j6)
    j7, j8, j9 = _to_frac(j7), _to_frac(j8), _to_frac(j9)

    # x ranges from max(|j1-j9|, |j4-j8|, |j2-j6|)
    #           to   min( j1+j9,   j4+j8,   j2+j6 )
    x_min = max(abs(j1 - j9), abs(j4 - j8), abs(j2 - j6))
    x_max = min(j1 + j9, j4 + j8, j2 + j6)

    if x_min > x_max:
        return 0.0

    total = 0.0
    x = x_min
    while x <= x_max:
        total += ((-1) ** int(2 * x) * float(2 * x + 1) *
                  wigner6j(j1, j4, j7, j8, j9, x) *
                  wigner6j(j2, j5, j8, j4, x, j6) *
                  wigner6j(j3, j6, j9, x, j1, j2))
        x += 1

    return total


def wigner3j_safe(j1, j2, j3, m1, m2, m3) -> float:
    """wigner3j that returns 0.0 for invalid arguments instead of raising."""
    try:
        return wigner3j(j1, j2, j3, m1, m2, m3)
    except (ValueError, ZeroDivisionError):
        return 0.0


def wigner6j_safe(j1, j2, j3, J1, J2, J3) -> float:
    """wigner6j that returns 0.0 for invalid arguments instead of raising."""
    try:
        return wigner6j(j1, j2, j3, J1, J2, J3)
    except (ValueError, ZeroDivisionError):
        return 0.0
