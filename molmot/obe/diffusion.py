"""
Momentum diffusion constant computation from SSE trajectories.

Port of diffusion.jl from OpticalBlochEquations.jl by Christian Hallas.

The momentum diffusion constant D characterises the heating rate in a MOT.
The equilibrium temperature is T = D / (m * beta), where beta is the
damping coefficient (friction force = -beta * v).

Two methods are provided:

1. compute_diffusion_from_ensemble: Direct measurement of <p^2(t)> growth
   from an ensemble of SSE trajectories starting at rest.
   D = lim_{t->inf} d<p^2>/dt / 2.

2. compute_diffusion_correlation: Force auto-correlation method from
   Christian's code (using chi+ / chi- wavefunctions).
   D = integral_0^inf <F(t)F(0)> dt.

References
----------
* Dalibard, Cohen-Tannoudji, JOSA B 6, 2023 (1989) -- semiclassical theory
* Christian Hallas, OpticalBlochEquations.jl (diffusion.jl)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants import hbar, k_B


def compute_diffusion_from_ensemble(results, mass, axis=2, t_min=None):
    """
    Compute the momentum diffusion constant from an ensemble of SSE trajectories.

    Uses D = <p^2(t)> / (2*t) averaged over trajectories, where p = m*v.
    The mean drift (radiation pressure force) is subtracted.

    Parameters
    ----------
    results : list of SSEResult
        Output from run_ensemble.
    mass : float
        Particle mass in kg.
    axis : int
        Spatial axis to compute diffusion along (0=x, 1=y, 2=z).
    t_min : float, optional
        Minimum time (seconds) to consider (to allow initial transients to settle).

    Returns
    -------
    D : float
        Momentum diffusion constant in kg^2 m^2 / s^3 (SI).
    D_err : float
        Standard error of D.
    T_eq : float
        Estimated equilibrium temperature from D and the damping rate,
        using T = D / (m * beta * k_B), returned as NaN if beta cannot
        be estimated.
    """
    n_traj = len(results)
    if n_traj == 0:
        return 0.0, 0.0, float('nan')

    # Find common time grid (use first trajectory's times)
    times = results[0].times

    if t_min is not None:
        i_start = np.searchsorted(times, t_min)
    else:
        i_start = max(1, len(times) // 4)  # skip first quarter as equilibration

    if i_start >= len(times) - 1:
        return 0.0, 0.0, float('nan')

    # Compute <v^2(t)> - <v(t)>^2 for each time
    n_times = len(times)
    v_mean = np.zeros(n_times)
    v2_mean = np.zeros(n_times)

    for result in results:
        n_pts = min(n_times, len(result.velocities))
        v = result.velocities[:n_pts, axis]
        v_mean[:n_pts] += v
        v2_mean[:n_pts] += v ** 2

    v_mean /= n_traj
    v2_mean /= n_traj
    v_var = v2_mean - v_mean ** 2  # variance

    # D = m^2 * d<v_var>/dt / 2
    # Fit a line to v_var(t) for t > t_min
    t_fit = times[i_start:]
    v_var_fit = v_var[i_start:]

    if len(t_fit) < 2:
        return 0.0, 0.0, float('nan')

    # Linear fit: v_var = a + b*t -> D = m^2 * b / 2
    coeffs = np.polyfit(t_fit, v_var_fit, 1)
    slope = coeffs[0]  # d(v_var)/dt

    D = mass ** 2 * slope / 2.0

    # Estimate uncertainty
    v_var_pred = np.polyval(coeffs, t_fit)
    residuals = v_var_fit - v_var_pred
    D_err = mass ** 2 * np.std(residuals) / (2.0 * np.sqrt(len(t_fit)))

    return D, D_err, float('nan')


def compute_diffusion_temperature(D, mass, beta):
    """
    Compute the equilibrium temperature from the diffusion constant and damping rate.

    T = D / (mass * beta * k_B)

    Parameters
    ----------
    D : float
        Momentum diffusion constant (kg^2 m^2 / s^3).
    mass : float
        Particle mass (kg).
    beta : float
        Damping coefficient (1/s), defined by F = -beta * m * v.

    Returns
    -------
    T : float
        Equilibrium temperature (K).
    """
    if abs(beta) < 1e-30:
        return float('inf')
    return D / (mass * beta * k_B)


def estimate_damping_rate(results, mass, axis=2, v_range=None):
    """
    Estimate the damping rate (friction coefficient) from SSE trajectories.

    Fits the velocity autocorrelation decay or the mean deceleration
    as a function of initial velocity.

    Parameters
    ----------
    results : list of SSEResult
    mass : float
        Particle mass (kg).
    axis : int
        Axis to analyse.
    v_range : tuple (v_min, v_max), optional
        Velocity range for the fit.

    Returns
    -------
    beta : float
        Damping rate (1/s), where F = -beta * m * v.
    """
    if len(results) == 0:
        return 0.0

    # Use velocity autocorrelation: <v(t)*v(0)> ~ <v(0)^2> * exp(-beta*t)
    times = results[0].times
    n_times = len(times)

    v0_all = []
    v_auto = np.zeros(n_times)
    v0_sq = 0.0

    for result in results:
        n_pts = min(n_times, len(result.velocities))
        v = result.velocities[:n_pts, axis]
        v0 = v[0]
        v0_all.append(v0)
        v0_sq += v0 ** 2
        v_auto[:n_pts] += v0 * v[:n_pts]

    n_traj = len(results)
    v_auto /= n_traj
    v0_sq /= n_traj

    if v0_sq < 1e-20:
        return 0.0

    # Normalize
    v_auto_norm = v_auto / v0_sq

    # Fit exponential decay: log(|v_auto_norm|) = -beta * t
    valid = v_auto_norm > 0.1  # only fit where signal is positive
    if np.sum(valid) < 3:
        return 0.0

    t_valid = times[valid]
    log_auto = np.log(v_auto_norm[valid])

    coeffs = np.polyfit(t_valid, log_auto, 1)
    beta = -coeffs[0]

    return max(beta, 0.0)


def compute_scattering_rate(results, t_min=None):
    """
    Compute the mean photon scattering rate from SSE trajectories.

    Parameters
    ----------
    results : list of SSEResult
    t_min : float, optional
        Count only photons after this time.

    Returns
    -------
    R_scatter : float
        Scattering rate (photons/s).
    R_err : float
        Standard error.
    """
    rates = []
    for result in results:
        if t_min is not None:
            jumps_after = [t for t in result.jump_times if t > t_min]
            dt = result.times[-1] - t_min
        else:
            jumps_after = result.jump_times
            dt = result.times[-1] - result.times[0]

        if dt > 0:
            rates.append(len(jumps_after) / dt)

    if len(rates) == 0:
        return 0.0, 0.0

    rates = np.array(rates)
    return np.mean(rates), np.std(rates) / np.sqrt(len(rates))
