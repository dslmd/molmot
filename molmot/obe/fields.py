"""
Laser field and MOT beam configuration.

Port of the Field / laser handling from OpticalBlochEquations.jl.
Includes:
  - LaserBeam and MOTConfig data classes
  - RF / DC MOT beam constructors
  - Polarization rotation (rotate_polarization)
  - Gaussian beam profile computation
  - 6-beam field update functions (port of field_fast.jl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class LaserBeam:
    """
    Description of a single laser beam / frequency component.

    Attributes
    ----------
    direction : np.ndarray
        k-vector direction as a 3-component unit vector.
        For a 1-D model along z, use [0, 0, +1] or [0, 0, -1].
    freq_offset : float
        Laser frequency in Hz (absolute, referenced to zero-field
        transition frequency).
    polarization : int
        Polarisation index in the spherical basis:
        0 = sigma- (q = -1), 1 = pi (q = 0), 2 = sigma+ (q = +1).
    s0 : float
        Saturation parameter (I / I_sat) for this beam component.
    """
    direction: np.ndarray
    freq_offset: float
    polarization: int
    s0: float

    @property
    def kdir_z(self) -> float:
        """Projection of the propagation direction onto the z-axis."""
        if isinstance(self.direction, np.ndarray):
            return float(self.direction[2])
        return float(self.direction)

    def __post_init__(self):
        if isinstance(self.direction, (int, float)):
            # Convenience: scalar +1/-1 means beam along +z/-z
            d = float(self.direction)
            self.direction = np.array([0.0, 0.0, d])
        self.direction = np.asarray(self.direction, dtype=float)


@dataclass
class MOTConfig:
    """
    Complete MOT laser configuration.

    Attributes
    ----------
    beams : list of LaserBeam
    B_gradient : float
        Magnetic field gradient in T/m (for 1-D: dBz/dz).
    """
    beams: List[LaserBeam] = field(default_factory=list)
    B_gradient: float = 0.0


def make_rf_mot_beams(mol_data, delta_Gamma: float = -1.0,
                      s0: float = 1.0,
                      phase: int = 0,
                      n_sidebands: int = 2) -> List[LaserBeam]:
    """
    Create laser beams for the RF MOT configuration.

    The RF MOT uses two frequency sidebands addressing the J=1/2 and
    J=3/2 ground-state manifolds.  Polarisation and B-field direction
    switch synchronously at ~1.4 MHz.

    Parameters
    ----------
    mol_data : MolecularData (from molecules.sroh)
        Must have ``omega_J12``, ``omega_J32``, ``Gamma`` attributes.
    delta_Gamma : float
        Overall detuning in units of Gamma (negative = red).
    s0 : float
        Saturation parameter per sideband per beam.
    phase : int
        RF phase: 0 = sigma+ forward / +B; 1 = sigma- forward / -B.
    n_sidebands : int
        Number of sidebands (2 for standard SrOH RF MOT).

    Returns
    -------
    list of LaserBeam
    """
    import math
    Gamma = mol_data.Gamma
    delta_hz = delta_Gamma * Gamma / (2.0 * math.pi)

    freq_J12 = mol_data.omega_J12 + delta_hz
    freq_J32 = mol_data.omega_J32 + delta_hz

    beams = []

    # The two SR sidebands have OPPOSITE circular polarizations
    # (because gJ has opposite sign for J=3/2 vs J=1/2).
    # The Pockels cell switches ALL polarizations simultaneously.
    #
    # Phase 0: J=3/2 → sigma+ (q=2), J=1/2 → sigma- (q=0), +B gradient
    # Phase 1: J=3/2 → sigma- (q=0), J=1/2 → sigma+ (q=2), -B gradient
    if phase == 0:
        q_J32_fwd, q_J12_fwd = 2, 0  # sigma+, sigma-
    else:
        q_J32_fwd, q_J12_fwd = 0, 2  # sigma-, sigma+

    for freq, q_fwd in [(freq_J32, q_J32_fwd), (freq_J12, q_J12_fwd)]:
        for direction in [+1, -1]:
            # Retro-reflected beam flips handedness
            q_eff = q_fwd if direction > 0 else (2 - q_fwd)

            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=freq,
                polarization=q_eff,
                s0=s0,
            ))

    return beams


def make_dc_mot_beams(mol_data, delta_Gamma: float = -0.20,
                      split_Gamma: float = 0.70,
                      s0: float = 1.0,
                      n_freq: int = 4) -> List[LaserBeam]:
    """
    Create laser beams for the 4-frequency DC MOT configuration.

    For each spin-rotation manifold (J=1/2 and J=3/2), two frequency
    components are used with opposite circular polarisations:
        sigma+ at Delta + delta_split
        sigma- at Delta - delta_split

    Parameters
    ----------
    mol_data : MolecularData
        Must have ``omega_J12``, ``omega_J32``, ``Gamma`` attributes.
    delta_Gamma : float
        Overall detuning in units of Gamma.
    split_Gamma : float
        Polarisation split in units of Gamma.
    s0 : float
        Saturation parameter per frequency component per beam.
    n_freq : int
        Number of frequency components (4 for the standard scheme).

    Returns
    -------
    list of LaserBeam
    """
    import math
    Gamma = mol_data.Gamma
    delta_hz = delta_Gamma * Gamma / (2.0 * math.pi)
    split_hz = split_Gamma * Gamma / (2.0 * math.pi)

    # 4 frequency components
    components = [
        (mol_data.omega_J32 + delta_hz + split_hz, 2),   # J=3/2, sigma+
        (mol_data.omega_J32 + delta_hz - split_hz, 0),   # J=3/2, sigma-
        (mol_data.omega_J12 + delta_hz + split_hz, 2),   # J=1/2, sigma+
        (mol_data.omega_J12 + delta_hz - split_hz, 0),   # J=1/2, sigma-
    ]

    beams = []
    for freq, q_fwd in components:
        for direction in [+1, -1]:
            # Retro-reflected beam: sigma+ <-> sigma-
            if direction > 0:
                q_eff = q_fwd
            else:
                if q_fwd == 0:
                    q_eff = 2
                elif q_fwd == 2:
                    q_eff = 0
                else:
                    q_eff = 1  # pi unchanged

            beams.append(LaserBeam(
                direction=np.array([0.0, 0.0, float(direction)]),
                freq_offset=freq,
                polarization=q_eff,
                s0=s0,
            ))

    return beams


# ===================================================================
# Polarization rotation and field utilities
# (Port of rotate_pol, flip, and field_fast.jl)
# ===================================================================

def _wigner_D_j1(cos_beta, sin_beta, alpha, gamma):
    """
    Wigner D-matrix for j=1 with Euler angles (alpha, beta, gamma).

    Returns a 3x3 complex matrix in the spherical basis (q = -1, 0, +1).
    Uses the ZYZ convention matching the Julia implementation.
    """
    ea_p = np.exp(1j * alpha)
    ea_m = np.exp(-1j * alpha)
    eg_p = np.exp(1j * gamma)
    eg_m = np.exp(-1j * gamma)
    cb, sb = cos_beta, sin_beta

    return np.array([
        [0.5 * (1 + cb) * ea_m * eg_m,
         -sb / math.sqrt(2) * ea_m,
         0.5 * (1 - cb) * ea_m * eg_p],
        [sb / math.sqrt(2) * eg_m,
         cb + 0j,
         -sb / math.sqrt(2) * eg_p],
        [0.5 * (1 - cb) * ea_p * eg_m,
         sb / math.sqrt(2) * ea_p,
         0.5 * (1 + cb) * ea_p * eg_p],
    ], dtype=complex)


def flip_polarization(eps):
    """
    Flip the polarization for a retro-reflected beam.

    For a beam propagating in the -k direction, the handedness of
    circular polarization reverses: sigma+ <-> sigma-.
    In the spherical basis [q=-1, q=0, q=+1]:
      eps_new = [eps[+1], -eps[0], eps[-1]]

    Port of ``flip()`` from misc.jl.
    """
    eps = np.asarray(eps, dtype=complex)
    return np.array([eps[2], -eps[1], eps[0]], dtype=complex)


def rotate_polarization(pol, k_hat):
    """
    Rotate lab-frame polarization to the beam propagation frame.

    Given a polarization ``pol`` in the spherical basis (3 components:
    q = -1, 0, +1) defined with respect to the z-axis, compute the
    polarization in the frame where the quantization axis points along
    ``k_hat``.

    Uses the Wigner D-matrix for j=1. The rotation is parameterised by
    Euler angles extracted from the axis-angle representation of the
    rotation that takes z-hat to k_hat.

    Port of ``rotate_pol()`` from optical_bloch_equations.jl.

    Parameters
    ----------
    pol : array-like, shape (3,), complex
        Polarization vector [q=-1, q=0, q=+1].
    k_hat : array-like, shape (3,), float
        Beam propagation direction (will be normalised internally).

    Returns
    -------
    pol_rotated : np.ndarray, shape (3,), complex
    """
    pol = np.asarray(pol, dtype=complex)
    k_hat = np.asarray(k_hat, dtype=float)
    k_norm = np.linalg.norm(k_hat)
    if k_norm < 1e-15:
        return pol.copy()
    k_hat = k_hat / k_norm

    z_hat = np.array([0.0, 0.0, 1.0])
    cos_theta = float(np.clip(np.dot(k_hat, z_hat), -1.0, 1.0))
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-12:
        alpha, beta, gamma = 0.0, 0.0, 0.0
    elif abs(theta - math.pi) < 1e-12:
        alpha, beta, gamma = 0.0, math.pi, 0.0
    else:
        cross = np.cross(k_hat, z_hat)
        x, y, z = float(cross[0]), float(cross[1]), float(cross[2])
        y = -y  # sign convention matching Julia code

        sin_theta = math.sin(theta)
        cos_t = math.cos(theta)

        A33 = (1 - cos_t) * z ** 2 + cos_t
        A31 = (1 - cos_t) * z * x - y * sin_theta
        A32 = (1 - cos_t) * z * y + x * sin_theta
        A13 = (1 - cos_t) * x * z + y * sin_theta
        A23 = (1 - cos_t) * y * z - x * sin_theta

        alpha = math.atan2(A23, A13)
        beta = math.atan2(math.sqrt(max(0.0, 1.0 - A33 ** 2)), A33)
        gamma = math.atan2(A32, -A31)

    D = _wigner_D_j1(math.cos(beta), math.sin(beta), alpha, gamma)
    return np.linalg.solve(D, pol)


def gaussian_beam_profile(r, k_hat, beam_radius):
    """
    Compute the Gaussian beam intensity profile at position r.

    The profile is exp(-2 * r_perp^2 / w^2) where r_perp is the
    perpendicular distance from the beam axis.

    Parameters
    ----------
    r : np.ndarray, shape (3,)
        Position vector (metres).
    k_hat : np.ndarray, shape (3,)
        Beam propagation direction (unit vector).
    beam_radius : float
        1/e^2 intensity radius w (metres).

    Returns
    -------
    profile : float
        Intensity profile value in [0, 1].
    """
    r = np.asarray(r, dtype=float)
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    r_parallel = np.dot(r, k_hat)
    r_perp_sq = np.dot(r, r) - r_parallel ** 2
    r_perp_sq = max(r_perp_sq, 0.0)

    return math.exp(-2.0 * r_perp_sq / beam_radius ** 2)


def update_fields_6beam(r, t, omega_lasers, sats, pols, beam_radius_k,
                        k=1.0):
    """
    Compute electric field amplitudes for the 6-beam MOT geometry.

    Beams propagate along +x, +y, +z, -x, -y, -z. Each beam carries
    ``n_freqs`` frequency components.

    Port of ``update_fields_fast!`` from field_fast.jl.

    Parameters
    ----------
    r : np.ndarray, shape (3,)
        Position (natural units: 1/k, or SI: metres).
    t : float
        Time (natural units: 1/Gamma, or SI: seconds).
    omega_lasers : np.ndarray, shape (n_freqs,)
        Laser frequencies (angular, in matching units).
    sats : np.ndarray, shape (n_freqs,)
        Saturation parameters per frequency component.
    pols : list of np.ndarray, each shape (3,), complex
        Polarization vectors (spherical basis) for each frequency.
    beam_radius_k : float
        Beam radius in matching length units (e.g. w*k for natural units).
    k : float
        Wavenumber (1.0 for natural units).

    Returns
    -------
    E_kq : np.ndarray, shape (6, 3), complex
        Field amplitude per beam direction (6) and polarization component (3).
    E_total : np.ndarray, shape (3,), complex
        Total field summed over all 6 beam directions.
    """
    n_freqs = len(omega_lasers)
    denom = beam_radius_k ** 2 / 2.0

    # Perpendicular distances squared for each beam axis
    r_perp_sq = np.array([
        r[1] ** 2 + r[2] ** 2,  # x-axis beam
        r[0] ** 2 + r[2] ** 2,  # y-axis beam
        r[0] ** 2 + r[1] ** 2,  # z-axis beam
    ])
    gauss_amp = np.sqrt(np.exp(-r_perp_sq / denom))

    # Rotated polarizations for each beam direction
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])

    eps = np.zeros((6, n_freqs, 3), dtype=complex)
    for i, pol in enumerate(pols):
        # Forward beams: unflipped polarization
        eps[0, i, :] = rotate_polarization(pol, x_hat)
        eps[1, i, :] = rotate_polarization(pol, y_hat)
        eps[2, i, :] = rotate_polarization(pol, z_hat)
        # Backward beams: D-matrix rotation handles helicity flip automatically
        eps[3, i, :] = rotate_polarization(pol, -x_hat)
        eps[4, i, :] = rotate_polarization(pol, -y_hat)
        eps[5, i, :] = rotate_polarization(pol, -z_hat)

    kr = k * r[:3]

    # Amplitude: a[beam, freq] = G * gauss * exp(i * phase)
    a = np.zeros((6, n_freqs), dtype=complex)
    for f in range(n_freqs):
        G = math.sqrt(sats[f]) / (2.0 * math.sqrt(2.0))
        wt = omega_lasers[f] * t
        for axis in range(3):
            phase_fwd = -kr[axis] + wt
            phase_bwd = kr[axis] + wt
            a[axis, f] = G * gauss_amp[axis] * np.exp(1j * phase_fwd)
            a[axis + 3, f] = G * gauss_amp[axis] * np.exp(1j * phase_bwd)

    # E_kq[beam, q] = sum_f a[beam, f] * conj(eps[beam, f, q])
    E_kq = np.zeros((6, 3), dtype=complex)
    for q in range(3):
        for beam in range(6):
            for f in range(n_freqs):
                E_kq[beam, q] += a[beam, f] * np.conj(eps[beam, f, q])

    E_total = np.sum(E_kq, axis=0)
    return E_kq, E_total


def make_6beam_polarizations(pol, with_flip=True):
    """
    Compute rotated polarizations for all 6 beam directions.

    Parameters
    ----------
    pol : np.ndarray, shape (3,), complex
        Lab-frame polarization [q=-1, q=0, q=+1].
    with_flip : bool
        If True (default), backward beams on ALL axes use flipped
        polarization (helicity reversal for retro-reflected beams).

    Returns
    -------
    eps_all : np.ndarray, shape (6, 3), complex
        Rotated polarizations for +x, +y, +z, -x, -y, -z.
    """
    pol = np.asarray(pol, dtype=complex)
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])

    eps = np.zeros((6, 3), dtype=complex)
    # Forward beams
    eps[0] = rotate_polarization(pol, x_hat)
    eps[1] = rotate_polarization(pol, y_hat)
    eps[2] = rotate_polarization(pol, z_hat)
    # Backward beams: D-matrix rotation handles helicity flip automatically
    eps[3] = rotate_polarization(pol, -x_hat)
    eps[4] = rotate_polarization(pol, -y_hat)
    eps[5] = rotate_polarization(pol, -z_hat)

    return eps
