"""
Optical Bloch Equations subpackage.

Port of OpticalBlochEquations.jl by Christian Hallas.
Provides rate-equation, Lindblad master equation, and stochastic
Schrodinger equation (Monte Carlo wavefunction) solvers for
multi-level atomic/molecular systems.
"""

from .fields import (LaserBeam, MOTConfig, make_rf_mot_beams, make_dc_mot_beams,
                     rotate_polarization, flip_polarization, gaussian_beam_profile,
                     update_fields_6beam, make_6beam_polarizations)
from .rate_equations import solve_rate_equations, solve_rate_equations_auto
from .lindblad import (build_liouvillian, steady_state_density_matrix,
                       compute_force_obe)
from .force import (force_from_wavefunction, force_from_density_matrix,
                    zeeman_gradient_force)
from .stochastic import (SSEProblem, SSESolver, SSEResult, run_ensemble)
from .floquet import FloquetOBE
from .diffusion import (compute_diffusion_from_ensemble,
                        compute_diffusion_temperature,
                        estimate_damping_rate,
                        compute_scattering_rate)

__all__ = [
    # Fields
    "LaserBeam",
    "MOTConfig",
    "make_rf_mot_beams",
    "make_dc_mot_beams",
    "rotate_polarization",
    "flip_polarization",
    "gaussian_beam_profile",
    "update_fields_6beam",
    "make_6beam_polarizations",
    # Rate equations
    "solve_rate_equations",
    "solve_rate_equations_auto",
    # Lindblad OBE
    "build_liouvillian",
    "steady_state_density_matrix",
    "compute_force_obe",
    # Force
    "force_from_wavefunction",
    "force_from_density_matrix",
    "zeeman_gradient_force",
    # Stochastic Schrodinger (SSE / MCWF)
    "SSEProblem",
    "SSESolver",
    "SSEResult",
    "run_ensemble",
    # Floquet OBE
    "FloquetOBE",
    # Diffusion
    "compute_diffusion_from_ensemble",
    "compute_diffusion_temperature",
    "estimate_damping_rate",
    "compute_scattering_rate",
]
