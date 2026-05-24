"""
Trajectory propagation subpackage.

Provides 1D and 3D trajectory integration for MOT simulations,
Monte Carlo temperature estimation, and particle sampling utilities.
"""

from .trajectories import (simulate_trajectory, simulate_trajectories_3d,
                           monte_carlo_temperature,
                           sample_direction, sample_maxwell_boltzmann,
                           sample_maxwell_boltzmann_speed,
                           sample_gaussian_position, sample_uniform_sphere,
                           make_position_sampler, make_velocity_sampler)

try:
    from .trajectories_jit import simulate_trajectory_jit, monte_carlo_temperature_jit
    _HAS_TRAJ_JIT = True
except ImportError:
    _HAS_TRAJ_JIT = False

__all__ = [
    "simulate_trajectory",
    "simulate_trajectories_3d",
    "monte_carlo_temperature",
    "sample_direction",
    "sample_maxwell_boltzmann",
    "sample_maxwell_boltzmann_speed",
    "sample_gaussian_position",
    "sample_uniform_sphere",
    "make_position_sampler",
    "make_velocity_sampler",
]
