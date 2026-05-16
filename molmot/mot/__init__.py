"""
MOT simulator subpackage.

Provides RF and DC MOT simulators that combine molecular data,
laser configurations, and force solvers.
"""

from .simulator import RFMOTSimulator, DCMOTSimulator
from .simulator_3d import (MOTSimulator3D, Beam3D, make_6beam_config,
                           rate_eq_force_3d, compute_3d_force_map)
from .force_scan import (force_vs_z, force_vs_v,
                         spring_constant, damping_coefficient,
                         capture_velocity, optimize_parameters)
from .analysis import (gaussian_fit, maxwell_boltzmann_fit_1d,
                       sigma_vs_time, temperature_vs_time,
                       density_vs_time, survived, capture_fraction)

__all__ = [
    "RFMOTSimulator",
    "DCMOTSimulator",
    # 3D simulator
    "MOTSimulator3D",
    "Beam3D",
    "make_6beam_config",
    "rate_eq_force_3d",
    "compute_3d_force_map",
    "force_vs_z",
    "force_vs_v",
    "spring_constant",
    "damping_coefficient",
    "capture_velocity",
    "optimize_parameters",
    # Analysis
    "gaussian_fit",
    "maxwell_boltzmann_fit_1d",
    "sigma_vs_time",
    "temperature_vs_time",
    "density_vs_time",
    "survived",
    "capture_fraction",
]
