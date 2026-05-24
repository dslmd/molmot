"""
Numba JIT-compiled trajectory integration.

The pure Python trajectory loop calls simulator.force() ~30,000 times
per trajectory, each call going through Python dispatch. By calling
the JIT rate equation kernel directly from within a JIT-compiled loop,
we eliminate all Python overhead.

Gives ~50-100x speedup over the pure Python trajectory loop.
"""

import numpy as np
from numba import njit

from ..obe.rate_equations_jit import _rate_eq_core


@njit(cache=True)
def _trajectory_1d_jit(
    energies, d_squared, zeeman_z_diag,
    beam_dirs, beam_freqs, beam_pols, beam_s0s,
    z0, v0, dt, n_steps, mass,
    B_gradient, Gamma, k_wave, Gamma_eff,
    n_ground, n_excited,
    z_escape,
):
    """
    1D trajectory integration with Euler method, fully JIT-compiled.

    Returns (z_arr, v_arr, n_actual) where n_actual is the step at which
    the particle escaped (or n_steps if it stayed trapped).
    """
    z_arr = np.empty(n_steps)
    v_arr = np.empty(n_steps)
    z_arr[0] = z0
    v_arr[0] = v0

    n_actual = n_steps
    for j in range(1, n_steps):
        force, _, _ = _rate_eq_core(
            energies, d_squared, zeeman_z_diag,
            beam_dirs, beam_freqs, beam_pols, beam_s0s,
            v_arr[j - 1], z_arr[j - 1],
            B_gradient, Gamma, k_wave, Gamma_eff,
            n_ground, n_excited)

        a = force / mass
        v_arr[j] = v_arr[j - 1] + a * dt
        z_arr[j] = z_arr[j - 1] + v_arr[j] * dt

        if abs(z_arr[j]) > z_escape:
            n_actual = j
            for k in range(j + 1, n_steps):
                z_arr[k] = z_arr[j]
                v_arr[k] = v_arr[j]
            break

    return z_arr, v_arr, n_actual


@njit(cache=True)
def _monte_carlo_temperature_jit(
    energies, d_squared, zeeman_z_diag,
    beam_dirs, beam_freqs, beam_pols, beam_s0s,
    n_particles, dt, n_steps, mass,
    B_gradient, Gamma, k_wave, Gamma_eff,
    n_ground, n_excited,
    z_escape, z_init_sigma, v_init_sigma,
    i_equilibrium, seed,
):
    """
    Monte Carlo temperature estimation, fully JIT-compiled.

    All n_particles trajectories are integrated sequentially inside
    this JIT function, with no Python overhead per step.
    """
    np.random.seed(seed)
    kB = 1.380649e-23

    v_sq_sum = 0.0
    n_samples = 0

    for ip in range(n_particles):
        z = np.random.normal(0.0, z_init_sigma)
        v = np.random.normal(0.0, v_init_sigma)

        for j in range(1, n_steps):
            force, _, _ = _rate_eq_core(
                energies, d_squared, zeeman_z_diag,
                beam_dirs, beam_freqs, beam_pols, beam_s0s,
                v, z,
                B_gradient, Gamma, k_wave, Gamma_eff,
                n_ground, n_excited)

            a = force / mass
            v += a * dt
            z += v * dt

            if abs(z) > z_escape:
                break

            if j >= i_equilibrium:
                v_sq_sum += v * v
                n_samples += 1

    if n_samples == 0:
        return 1e30

    return mass * v_sq_sum / (n_samples * kB)


def _prepare_beam_arrays(beams):
    """Extract flat beam arrays for JIT consumption."""
    return (
        np.array([b.kdir_z for b in beams], dtype=np.float64),
        np.array([b.freq_offset for b in beams], dtype=np.float64),
        np.array([b.polarization for b in beams], dtype=np.float64),
        np.array([b.s0 for b in beams], dtype=np.float64),
    )


def simulate_trajectory_jit(simulator, z0, v0, t_max, dt=1e-6,
                            z_escape=0.015):
    """
    1D trajectory integration, JIT-compiled. Same API as simulate_trajectory.

    ~50-100x faster than the pure Python version.
    """
    mol = simulator.mol_data
    beams = simulator._beams if hasattr(simulator, '_beams') else None
    if beams is None:
        raise ValueError("Simulator must have pre-built beams (_beams attribute)")

    dirs, freqs, pols, s0s = _prepare_beam_arrays(beams)
    n_steps = int(t_max / dt)
    Gamma_eff = getattr(simulator, 'Gamma_eff_factor', 1.0)

    z_arr, v_arr, _ = _trajectory_1d_jit(
        mol.energies, mol.d_squared, mol.zeeman_z_diag,
        dirs, freqs, pols, s0s,
        z0, v0, dt, n_steps, mol.mass,
        simulator.B_gradient, mol.Gamma, mol.k, Gamma_eff,
        mol.n_ground, mol.n_excited, z_escape)

    t_arr = np.arange(n_steps) * dt
    return t_arr, z_arr, v_arr


def monte_carlo_temperature_jit(simulator, n_particles=100, t_max=0.1,
                                dt=1e-6, z_init_sigma=1e-3,
                                v_init_sigma=1.0, t_equilibrium=0.05,
                                seed=42):
    """
    Monte Carlo temperature estimation, JIT-compiled.

    Same API as monte_carlo_temperature but ~50-100x faster.
    """
    mol = simulator.mol_data
    beams = simulator._beams if hasattr(simulator, '_beams') else None
    if beams is None:
        raise ValueError("Simulator must have pre-built beams")

    dirs, freqs, pols, s0s = _prepare_beam_arrays(beams)
    n_steps = int(t_max / dt)
    i_eq = int(t_equilibrium / dt)
    Gamma_eff = getattr(simulator, 'Gamma_eff_factor', 1.0)

    return _monte_carlo_temperature_jit(
        mol.energies, mol.d_squared, mol.zeeman_z_diag,
        dirs, freqs, pols, s0s,
        n_particles, dt, n_steps, mol.mass,
        simulator.B_gradient, mol.Gamma, mol.k, Gamma_eff,
        mol.n_ground, mol.n_excited,
        0.015, z_init_sigma, v_init_sigma, i_eq, seed)
