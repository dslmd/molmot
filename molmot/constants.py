"""
Physical constants — CODATA 2018 recommended values.

Port of UnitsToValue.jl by Christian Hallas.
All values in SI base units.
"""

# Electron spin g-factor
gS = 2.0023193043617

# Speed of light in vacuum (m/s)
c = 299792458.0

# Planck constant (J s)
h = 6.62607015e-34

# Reduced Planck constant (J s)
hbar = 1.054571817e-34

# Bohr magneton (J/T)
mu_B = 9.2740100783e-24

# Vacuum permittivity (F/m)
epsilon_0 = 8.8541878128e-12

# Vacuum permeability (N/A^2)
mu_0 = 1.25663706212e-6

# Boltzmann constant (J/K)
k_B = 1.380649e-23

# Atomic mass unit (kg)
amu = 1.66053906660e-27

# Derived constants frequently used in MOT simulations

# Wavenumber from wavelength
def wavenumber(wavelength):
    """Return k = 2*pi/lambda in 1/m."""
    import math
    return 2.0 * math.pi / wavelength

# Recoil velocity
def v_recoil(k, mass):
    """Return the photon recoil velocity hbar*k/m in m/s."""
    return hbar * k / mass

# Doppler temperature
def T_doppler(Gamma):
    """Return the Doppler temperature hbar*Gamma/(2*k_B) in K."""
    return hbar * Gamma / (2.0 * k_B)

# Saturation intensity
def I_sat(Gamma_hz, wavelength):
    """
    Return the saturation intensity in W/m^2.

    Parameters
    ----------
    Gamma_hz : float
        Natural linewidth Gamma/(2*pi) in Hz.
    wavelength : float
        Transition wavelength in metres.
    """
    import math
    return math.pi * h * c * Gamma_hz / (3.0 * wavelength ** 3)


# ───────────────────────────────────────────────────────────
# Polarization vectors (spherical basis)
# ───────────────────────────────────────────────────────────

import numpy as np

# Spherical basis components: index 0 = q=-1, index 1 = q=0, index 2 = q=+1
sigma_minus = np.array([1.0, 0.0, 0.0], dtype=complex)   # q = -1
sigma_0     = np.array([0.0, 1.0, 0.0], dtype=complex)   # q = 0 (pi)
sigma_plus  = np.array([0.0, 0.0, 1.0], dtype=complex)   # q = +1

# Cartesian ↔ spherical basis transformation matrices
# Spherical basis vectors in Cartesian: e_{+1} = -(x+iy)/√2, e_0 = z, e_{-1} = (x-iy)/√2
cart2sph = np.array([
    [ 1.0/np.sqrt(2), -1j/np.sqrt(2),  0.0],   # q = -1
    [ 0.0,             0.0,             1.0],   # q = 0
    [-1.0/np.sqrt(2), -1j/np.sqrt(2),  0.0],   # q = +1
], dtype=complex)

sph2cart = np.linalg.inv(cart2sph)

# Unit vectors
x_hat = np.array([1.0, 0.0, 0.0])
y_hat = np.array([0.0, 1.0, 0.0])
z_hat = np.array([0.0, 0.0, 1.0])


# ───────────────────────────────────────────────────────────
# cm^-1 to Hz conversion factor
# ───────────────────────────────────────────────────────────
CM_TO_HZ = c * 100.0  # multiply cm^-1 value by this to get Hz
