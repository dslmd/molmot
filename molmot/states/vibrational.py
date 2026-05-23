"""
Vibrational state types and operators.

Port of HarmonicOscillatorState.jl, HarmonicOscillatorState_3D.jl,
and TriatomicVibrationalState.jl from QuantumStates.jl.

Defines basis states for quantum harmonic oscillators (1-D and 3-D)
and triatomic vibrational modes, with creation/annihilation-based
matrix elements for position and momentum operators.

References
----------
* Griffiths, *Introduction to Quantum Mechanics* (Prentice Hall, 2005).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .basis import BasisState


###############################################################################
#  HarmonicOscillatorState  |n>
###############################################################################

@dataclass
class HarmonicOscillatorState(BasisState):
    """
    One-dimensional quantum harmonic oscillator state |n>.

    Parameters
    ----------
    m : float
        Mass of the particle.
    omega : float
        Angular frequency of the oscillator.
    n : float
        Vibrational quantum number (integer >= 0).
    """
    m: float = 1.0
    omega: float = 1.0
    n: float = 0

    def __repr__(self):
        return f"|n={int(self.n)}>"


def HO_Identity(s1: HarmonicOscillatorState,
                s2: HarmonicOscillatorState) -> float:
    """Identity: delta(n, n')."""
    if int(s1.n) == int(s2.n):
        return 1.0
    return 0.0


def HO_T(s1: HarmonicOscillatorState,
         s2: HarmonicOscillatorState) -> float:
    """
    Kinetic + potential energy: omega * (n + 1/2).

    Diagonal in n.
    """
    if int(s1.n) != int(s2.n):
        return 0.0
    n = s1.n
    omega = s1.omega
    return omega * (n + 0.5)


###############################################################################
#  HarmonicOscillatorState_3D  |nx, ny, nz>
###############################################################################

@dataclass
class HarmonicOscillatorState_3D(BasisState):
    """
    Three-dimensional quantum harmonic oscillator state |nx, ny, nz>.

    Parameters
    ----------
    m : float
        Mass of the particle.
    omega_x, omega_y, omega_z : float
        Angular frequencies along x, y, z axes.
    nx, ny, nz : float
        Vibrational quantum numbers along each axis (integers >= 0).
    """
    m: float = 1.0
    omega_x: float = 1.0
    omega_y: float = 1.0
    omega_z: float = 1.0
    nx: float = 0
    ny: float = 0
    nz: float = 0

    def __repr__(self):
        return f"|nx={int(self.nx)}, ny={int(self.ny)}, nz={int(self.nz)}>"


def HO3D_Identity(s1: HarmonicOscillatorState_3D,
                  s2: HarmonicOscillatorState_3D) -> float:
    """Identity: product of deltas in nx, ny, nz."""
    if (int(s1.nx) == int(s2.nx) and
            int(s1.ny) == int(s2.ny) and
            int(s1.nz) == int(s2.nz)):
        return 1.0
    return 0.0


def HO3D_T(s1: HarmonicOscillatorState_3D,
           s2: HarmonicOscillatorState_3D) -> float:
    """
    Total energy: omega_x*(nx+1/2) + omega_y*(ny+1/2) + omega_z*(nz+1/2).

    Diagonal in all quantum numbers.
    """
    if (int(s1.nx) != int(s2.nx) or
            int(s1.ny) != int(s2.ny) or
            int(s1.nz) != int(s2.nz)):
        return 0.0

    nx, ny, nz = s1.nx, s1.ny, s1.nz
    ox, oy, oz = s1.omega_x, s1.omega_y, s1.omega_z
    return ox * (nx + 0.5) + oy * (ny + 0.5) + oz * (nz + 0.5)


###############################################################################
#  TriatomicVibrationalState  |v1, v2, v3>
###############################################################################

@dataclass
class TriatomicVibrationalState(BasisState):
    """
    Vibrational state of a triatomic molecule |v1, v2, v3>.

    The three modes correspond to the symmetric stretch (v1),
    bending (v2), and asymmetric stretch (v3).

    Operators I1, I2, I3 are identity operators for each mode.
    Operators x1, x2, x3 are dimensionless position operators.
    Operators p1, p2, p3 are dimensionless momentum operators.
    """
    v1: float = 0
    v2: float = 0
    v3: float = 0

    def __repr__(self):
        return f"|v1={int(self.v1)}, v2={int(self.v2)}, v3={int(self.v3)}>"


# ----- Identity operators for each vibrational mode -----

def TV_I1(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """Identity in v1: delta(v1, v1')."""
    return 1.0 if int(s1.v1) == int(s2.v1) else 0.0


def TV_I2(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """Identity in v2: delta(v2, v2')."""
    return 1.0 if int(s1.v2) == int(s2.v2) else 0.0


def TV_I3(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """Identity in v3: delta(v3, v3')."""
    return 1.0 if int(s1.v3) == int(s2.v3) else 0.0


# ----- Position operators (dimensionless) -----

def TV_x1(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """
    Dimensionless position operator for mode 1 (symmetric stretch).

    x = sqrt(n'+1) * delta(n, n'+1) + sqrt(n') * delta(n, n'-1)

    The overall prefactor sqrt(hbar / (2*m*omega)) is NOT included.
    """
    n = int(s1.v1)
    np_ = int(s2.v1)
    if int(s1.v2) != int(s2.v2) or int(s1.v3) != int(s2.v3):
        return 0.0
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val += math.sqrt(np_)
    return val


def TV_x2(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """
    Dimensionless position operator for mode 2 (bending).

    x = sqrt(n'+1) * delta(n, n'+1) + sqrt(n') * delta(n, n'-1)
    """
    n = int(s1.v2)
    np_ = int(s2.v2)
    if int(s1.v1) != int(s2.v1) or int(s1.v3) != int(s2.v3):
        return 0.0
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val += math.sqrt(np_)
    return val


def TV_x3(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> float:
    """
    Dimensionless position operator for mode 3 (asymmetric stretch).

    x = sqrt(n'+1) * delta(n, n'+1) + sqrt(n') * delta(n, n'-1)
    """
    n = int(s1.v3)
    np_ = int(s2.v3)
    if int(s1.v1) != int(s2.v1) or int(s1.v2) != int(s2.v2):
        return 0.0
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val += math.sqrt(np_)
    return val


# ----- Momentum operators (dimensionless, returns complex) -----

def TV_p1(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> complex:
    """
    Dimensionless momentum operator for mode 1.

    p = i * [sqrt(n'+1) * delta(n, n'+1) - sqrt(n') * delta(n, n'-1)]

    The overall prefactor sqrt(hbar*m*omega / 2) is NOT included.
    Returns a complex number.
    """
    n = int(s1.v1)
    np_ = int(s2.v1)
    if int(s1.v2) != int(s2.v2) or int(s1.v3) != int(s2.v3):
        return 0.0 + 0.0j
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val -= math.sqrt(np_)
    return 1j * val


def TV_p2(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> complex:
    """
    Dimensionless momentum operator for mode 2.

    p = i * [sqrt(n'+1) * delta(n, n'+1) - sqrt(n') * delta(n, n'-1)]
    """
    n = int(s1.v2)
    np_ = int(s2.v2)
    if int(s1.v1) != int(s2.v1) or int(s1.v3) != int(s2.v3):
        return 0.0 + 0.0j
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val -= math.sqrt(np_)
    return 1j * val


def TV_p3(s1: TriatomicVibrationalState,
          s2: TriatomicVibrationalState) -> complex:
    """
    Dimensionless momentum operator for mode 3.

    p = i * [sqrt(n'+1) * delta(n, n'+1) - sqrt(n') * delta(n, n'-1)]
    """
    n = int(s1.v3)
    np_ = int(s2.v3)
    if int(s1.v1) != int(s2.v1) or int(s1.v2) != int(s2.v2):
        return 0.0 + 0.0j
    val = 0.0
    if n == np_ + 1:
        val += math.sqrt(np_ + 1)
    if n == np_ - 1:
        val -= math.sqrt(np_)
    return 1j * val
