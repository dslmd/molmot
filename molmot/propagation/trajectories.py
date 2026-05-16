"""
Trajectory integration and particle sampling for MOT simulations.

Port of trajectory simulation from OpticalBlochEquations.jl /
BeamPropagation.jl by Christian Hallas, plus particle sampling
from particle.jl.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..constants import k_B, hbar


def simulate_trajectory(simulator,
                        z0: float,
                        v0: float,
                        t_max: float,
                        dt: float = 1e-6,
                        z_escape: float = 0.015) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D trajectory integration using the Euler method.

    Parameters
    ----------
    simulator : RFMOTSimulator or DCMOTSimulator
        Must have a ``force(v, z)`` method returning (F, pop, R)
        and a ``mol_data.mass`` attribute.
    z0 : float
        Initial position (m).
    v0 : float
        Initial velocity (m/s).
    t_max : float
        Total simulation time (s).
    dt : float
        Time step (s).
    z_escape : float
        If |z| exceeds this, the molecule is considered escaped and the
        trajectory is truncated.

    Returns
    -------
    t_arr : np.ndarray
        Time array (s).
    z_arr : np.ndarray
        Position array (m).
    v_arr : np.ndarray
        Velocity array (m/s).
    """
    mass = simulator.mol_data.mass
    n_steps = int(t_max / dt)

    t_arr = np.arange(n_steps) * dt
    z_arr = np.zeros(n_steps)
    v_arr = np.zeros(n_steps)

    z_arr[0] = z0
    v_arr[0] = v0

    for j in range(1, n_steps):
        F, _, _ = simulator.force(v_arr[j - 1], z_arr[j - 1])
        a = F / mass
        v_arr[j] = v_arr[j - 1] + a * dt
        z_arr[j] = z_arr[j - 1] + v_arr[j] * dt

        if abs(z_arr[j]) > z_escape:
            z_arr[j:] = z_arr[j]
            v_arr[j:] = v_arr[j]
            break

    return t_arr, z_arr, v_arr


def simulate_trajectories_3d(simulator,
                             r0_list: List[np.ndarray],
                             v0_list: List[np.ndarray],
                             t_max: float,
                             dt: float = 1e-6,
                             r_escape: float = 0.015) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    3D trajectory integration with full quadrupole B-field.

    The force along each axis is computed independently using the
    1D force model with the local B-field from the quadrupole:
        Bx = -B'*x/2,  By = -B'*y/2,  Bz = +B'*z

    Parameters
    ----------
    simulator : MOT simulator
        Must have ``force(v, z)`` and ``B_gradient`` attribute.
    r0_list : list of np.ndarray, shape (3,)
        Initial positions for each particle.
    v0_list : list of np.ndarray, shape (3,)
        Initial velocities for each particle.
    t_max : float
        Total simulation time (s).
    dt : float
        Time step (s).
    r_escape : float
        Escape radius (m).

    Returns
    -------
    trajectories : list of (t_arr, r_arr, v_arr)
        For each particle: time array, position (n, 3), velocity (n, 3).
    """
    mass = simulator.mol_data.mass
    B_grad = simulator.B_gradient  # T/m
    n_steps = int(t_max / dt)
    t_arr = np.arange(n_steps) * dt

    trajectories = []

    for r0, v0 in zip(r0_list, v0_list):
        r = np.zeros((n_steps, 3))
        v = np.zeros((n_steps, 3))
        r[0] = r0
        v[0] = v0

        for j in range(1, n_steps):
            F = np.zeros(3)

            # Along z-axis: full gradient
            Fz, _, _ = simulator.force(v[j - 1, 2], r[j - 1, 2])
            F[2] = Fz

            # Along x-axis: gradient is -B'/2 for anti-Helmholtz
            # Create a temporary simulator-like call at effective position
            # The effective B at position x is: B_eff = (-B_grad/2) * x
            # Map to 1D: z_eff such that B_grad * z_eff = (-B_grad/2) * x
            # => z_eff = -x/2
            z_eff_x = -r[j - 1, 0] / 2.0
            Fx, _, _ = simulator.force(v[j - 1, 0], z_eff_x)
            F[0] = Fx

            # Along y-axis: same as x
            z_eff_y = -r[j - 1, 1] / 2.0
            Fy, _, _ = simulator.force(v[j - 1, 1], z_eff_y)
            F[1] = Fy

            a = F / mass
            v[j] = v[j - 1] + a * dt
            r[j] = r[j - 1] + v[j] * dt

            if np.linalg.norm(r[j]) > r_escape:
                r[j:] = r[j]
                v[j:] = v[j]
                break

        trajectories.append((t_arr, r, v))

    return trajectories


def monte_carlo_temperature(simulator,
                            n_particles: int = 100,
                            t_max: float = 0.1,
                            dt: float = 1e-6,
                            z_init_sigma: float = 1e-3,
                            v_init_sigma: float = 1.0,
                            t_equilibrium: float = 0.05) -> float:
    """
    Estimate equilibrium temperature from an ensemble of 1D trajectories.

    Particles are initialised with Gaussian-distributed positions and
    velocities.  After an equilibration time, the kinetic energy spread
    is used to estimate the temperature.

    Parameters
    ----------
    simulator : MOT simulator
    n_particles : int
        Number of particles in the ensemble.
    t_max : float
        Total simulation time (s).
    dt : float
        Time step (s).
    z_init_sigma : float
        Standard deviation of initial positions (m).
    v_init_sigma : float
        Standard deviation of initial velocities (m/s).
    t_equilibrium : float
        Time after which to start averaging (s).

    Returns
    -------
    T : float
        Estimated temperature (K).  Uses T = m * <v^2> / k_B.
    """
    mass = simulator.mol_data.mass
    n_steps = int(t_max / dt)
    i_eq = int(t_equilibrium / dt)

    v_sq_sum = 0.0
    n_samples = 0

    rng = np.random.default_rng(42)

    for _ in range(n_particles):
        z0 = rng.normal(0, z_init_sigma)
        v0 = rng.normal(0, v_init_sigma)

        z = z0
        v = v0

        for j in range(1, n_steps):
            F, _, _ = simulator.force(v, z)
            a = F / mass
            v += a * dt
            z += v * dt

            if abs(z) > 0.015:
                break

            if j >= i_eq:
                v_sq_sum += v ** 2
                n_samples += 1

    if n_samples == 0:
        return float("inf")

    v_sq_mean = v_sq_sum / n_samples
    T = mass * v_sq_mean / k_B

    return T


# ===================================================================
# Particle sampling functions (port of particle.jl)
# ===================================================================

def sample_direction(rng=None):
    """
    Sample a random unit vector uniformly on the unit sphere.

    Used for photon recoil kick directions in the SSE solver.

    Port of ``sample_direction()`` from particle.jl.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    direction : np.ndarray, shape (3,)
        Unit vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = 2.0 * math.pi * rng.random()
    z = 2.0 * rng.random() - 1.0
    r_perp = math.sqrt(max(0.0, 1.0 - z * z))

    return np.array([r_perp * math.cos(theta),
                     r_perp * math.sin(theta),
                     z])


def sample_maxwell_boltzmann(temperature, mass, n_particles=1, rng=None):
    """
    Sample velocities from a 3D Maxwell-Boltzmann distribution.

    Each Cartesian component is drawn from N(0, sigma) where
    sigma = sqrt(k_B * T / m).

    Parameters
    ----------
    temperature : float
        Temperature (K).
    mass : float
        Particle mass (kg).
    n_particles : int
        Number of velocity vectors to sample.
    rng : np.random.Generator, optional

    Returns
    -------
    velocities : np.ndarray, shape (n_particles, 3) or (3,) if n_particles=1
        Velocity vectors (m/s).
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = math.sqrt(k_B * temperature / mass) if temperature > 0 else 0.0
    v = rng.normal(0.0, sigma, size=(n_particles, 3)) if sigma > 0 else np.zeros((n_particles, 3))

    if n_particles == 1:
        return v[0]
    return v


def sample_maxwell_boltzmann_speed(temperature, mass, n_particles=1, rng=None):
    """
    Sample speeds from the Maxwell-Boltzmann speed distribution.

    Uses the chi distribution with 3 degrees of freedom (since the
    speed is the magnitude of a 3D Gaussian vector).

    Parameters
    ----------
    temperature : float
        Temperature (K).
    mass : float
        Particle mass (kg).
    n_particles : int
    rng : np.random.Generator, optional

    Returns
    -------
    speeds : np.ndarray, shape (n_particles,) or float if n_particles=1
        Speeds (m/s).
    """
    v_3d = sample_maxwell_boltzmann(temperature, mass, n_particles, rng)
    if n_particles == 1:
        return float(np.linalg.norm(v_3d))
    return np.linalg.norm(v_3d, axis=1)


def sample_gaussian_position(sigma, n_particles=1, rng=None):
    """
    Sample positions from a 3D Gaussian distribution centred at the origin.

    Parameters
    ----------
    sigma : float
        Standard deviation of the position distribution (m).
        The same sigma is used for all three Cartesian components.
    n_particles : int
    rng : np.random.Generator, optional

    Returns
    -------
    positions : np.ndarray, shape (n_particles, 3) or (3,) if n_particles=1
        Position vectors (m).
    """
    if rng is None:
        rng = np.random.default_rng()

    r = rng.normal(0.0, sigma, size=(n_particles, 3))

    if n_particles == 1:
        return r[0]
    return r


def sample_uniform_sphere(radius, n_particles=1, rng=None):
    """
    Sample positions uniformly inside a sphere.

    Parameters
    ----------
    radius : float
        Sphere radius (m).
    n_particles : int
    rng : np.random.Generator, optional

    Returns
    -------
    positions : np.ndarray, shape (n_particles, 3) or (3,)
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = np.zeros((n_particles, 3))
    for i in range(n_particles):
        while True:
            r = rng.uniform(-radius, radius, size=3)
            if np.linalg.norm(r) <= radius:
                positions[i] = r
                break

    if n_particles == 1:
        return positions[0]
    return positions


def make_position_sampler(sigma=1e-3, method='gaussian'):
    """
    Create a position sampler callable for use with run_ensemble.

    Parameters
    ----------
    sigma : float
        Size parameter (standard deviation for Gaussian, radius for uniform).
    method : str
        'gaussian' or 'uniform_sphere'.

    Returns
    -------
    sampler : callable() -> np.ndarray, shape (3,)
    """
    rng = np.random.default_rng()

    if method == 'gaussian':
        def sampler():
            return sample_gaussian_position(sigma, 1, rng)
    elif method == 'uniform_sphere':
        def sampler():
            return sample_uniform_sphere(sigma, 1, rng)
    else:
        raise ValueError(f"Unknown method: {method}")

    return sampler


def make_velocity_sampler(temperature, mass, method='maxwell_boltzmann'):
    """
    Create a velocity sampler callable for use with run_ensemble.

    Parameters
    ----------
    temperature : float
        Temperature (K).
    mass : float
        Particle mass (kg).
    method : str
        'maxwell_boltzmann' or 'fixed'.

    Returns
    -------
    sampler : callable() -> np.ndarray, shape (3,)
    """
    rng = np.random.default_rng()

    if method == 'maxwell_boltzmann':
        def sampler():
            return sample_maxwell_boltzmann(temperature, mass, 1, rng)
    elif method == 'fixed':
        # Return zero velocity
        def sampler():
            return np.zeros(3)
    else:
        raise ValueError(f"Unknown method: {method}")

    return sampler
