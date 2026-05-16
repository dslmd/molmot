"""
Analysis functions for MOT simulation results.

Port of analysis routines from Christian Hallas's
misc/helper_functions.jl and misc/compute_size_temperature.jl.

Provides:
  - Gaussian fitting of position distributions
  - Maxwell-Boltzmann fitting of velocity distributions
  - Cloud size, temperature, and density vs time
  - Survival and capture fraction analysis
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from ..constants import k_B


# =====================================================================
# Gaussian fit to position distribution
# =====================================================================

def gaussian_fit(positions: np.ndarray,
                 bins: int = 200,
                 range_mm: Tuple[float, float] = (-2, 2)) -> float:
    """
    Fit a Gaussian to a position distribution and return sigma (m).

    Parameters
    ----------
    positions : np.ndarray
        Array of positions in metres.
    bins : int
        Number of histogram bins.
    range_mm : tuple of float
        Histogram range in mm (converted to m internally).

    Returns
    -------
    sigma : float
        Standard deviation of the Gaussian fit (m).
        Returns np.inf if the fit fails.
    """
    range_m = (range_mm[0] * 1e-3, range_mm[1] * 1e-3)
    counts, bin_edges = np.histogram(positions, bins=bins, range=range_m)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts.astype(float)

    if np.max(counts) < 1:
        return np.inf

    def _gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    try:
        p0 = [np.max(counts), np.mean(positions), np.std(positions)]
        if p0[2] < 1e-10:
            p0[2] = 1e-4
        popt, _ = curve_fit(_gauss, bin_centres, counts, p0=p0, maxfev=5000)
        return abs(popt[2])
    except (RuntimeError, ValueError):
        return np.inf


# =====================================================================
# Maxwell-Boltzmann 1D fit to velocity distribution
# =====================================================================

def maxwell_boltzmann_fit_1d(velocities: np.ndarray,
                              mass: float,
                              bins: int = 100,
                              range_ms: Tuple[float, float] = (-1, 1)) -> float:
    """
    Fit a 1D Maxwell-Boltzmann (Gaussian) to a velocity distribution
    and return the temperature T (K).

    For 1D: f(v) ~ exp(-m v^2 / (2 k_B T))
    => sigma_v = sqrt(k_B T / m)
    => T = m * sigma_v^2 / k_B

    Parameters
    ----------
    velocities : np.ndarray
        Array of velocities (m/s).
    mass : float
        Particle mass (kg).
    bins : int
        Number of histogram bins.
    range_ms : tuple of float
        Histogram range (m/s).

    Returns
    -------
    T : float
        Temperature (K). Returns np.inf if the fit fails.
    """
    counts, bin_edges = np.histogram(velocities, bins=bins, range=range_ms)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts.astype(float)

    if np.max(counts) < 1:
        return np.inf

    def _gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    try:
        p0 = [np.max(counts), np.mean(velocities), np.std(velocities)]
        if p0[2] < 1e-10:
            p0[2] = 0.01
        popt, _ = curve_fit(_gauss, bin_centres, counts, p0=p0, maxfev=5000)
        sigma_v = abs(popt[2])
        T = mass * sigma_v ** 2 / k_B
        return T
    except (RuntimeError, ValueError):
        return np.inf


# =====================================================================
# Cloud size (sigma) vs time
# =====================================================================

def sigma_vs_time(results: list,
                  time_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cloud size vs time from an ensemble of trajectories.

    Parameters
    ----------
    results : list of (t_arr, z_arr, v_arr) tuples
        Each element is the output of simulate_trajectory.
    time_bins : int
        Number of time bins.

    Returns
    -------
    t_centres : np.ndarray
        Time bin centres (s).
    sigma_z : np.ndarray
        Cloud size sigma_z (m) at each time bin.
    sigma_v : np.ndarray
        Velocity spread sigma_v (m/s) at each time bin.
    n_alive : np.ndarray
        Number of particles contributing at each time bin.
    """
    # Determine the common time range
    t_max = min(r[0][-1] for r in results)
    t_edges = np.linspace(0, t_max, time_bins + 1)
    t_centres = 0.5 * (t_edges[:-1] + t_edges[1:])

    sigma_z = np.zeros(time_bins)
    sigma_v = np.zeros(time_bins)
    n_alive = np.zeros(time_bins)

    for i_t in range(time_bins):
        t_lo, t_hi = t_edges[i_t], t_edges[i_t + 1]
        t_mid = t_centres[i_t]

        z_samples = []
        v_samples = []
        for t_arr, z_arr, v_arr in results:
            # Find the closest time index
            idx = np.searchsorted(t_arr, t_mid)
            if idx >= len(t_arr):
                idx = len(t_arr) - 1
            z_samples.append(z_arr[idx])
            v_samples.append(v_arr[idx])

        z_samples = np.array(z_samples)
        v_samples = np.array(v_samples)

        # Only count "alive" molecules (not escaped)
        alive = np.abs(z_samples) < 0.015
        n_alive[i_t] = np.sum(alive)

        if n_alive[i_t] > 1:
            sigma_z[i_t] = np.std(z_samples[alive])
            sigma_v[i_t] = np.std(v_samples[alive])
        else:
            sigma_z[i_t] = np.nan
            sigma_v[i_t] = np.nan

    return t_centres, sigma_z, sigma_v, n_alive


# =====================================================================
# Temperature vs time
# =====================================================================

def temperature_vs_time(results: list,
                        mass: float,
                        time_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute temperature vs time from the velocity distribution.

    Uses T = m * <v^2> / k_B for the 1D kinetic temperature.

    Parameters
    ----------
    results : list of (t_arr, z_arr, v_arr) tuples
    mass : float
        Particle mass (kg).
    time_bins : int

    Returns
    -------
    t_centres : np.ndarray
        Time bin centres (s).
    T_arr : np.ndarray
        Temperature (K) at each time bin.
    """
    t_max = min(r[0][-1] for r in results)
    t_edges = np.linspace(0, t_max, time_bins + 1)
    t_centres = 0.5 * (t_edges[:-1] + t_edges[1:])

    T_arr = np.zeros(time_bins)

    for i_t in range(time_bins):
        t_mid = t_centres[i_t]
        v_samples = []

        for t_arr, z_arr, v_arr in results:
            idx = np.searchsorted(t_arr, t_mid)
            if idx >= len(t_arr):
                idx = len(t_arr) - 1
            # Only include if molecule is still trapped
            if abs(z_arr[idx]) < 0.015:
                v_samples.append(v_arr[idx])

        if len(v_samples) > 1:
            v_samples = np.array(v_samples)
            v_sq_mean = np.mean(v_samples ** 2)
            T_arr[i_t] = mass * v_sq_mean / k_B
        else:
            T_arr[i_t] = np.nan

    return t_centres, T_arr


# =====================================================================
# Density vs time
# =====================================================================

def density_vs_time(results: list,
                    N_total: int = 2000,
                    time_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute peak density vs time assuming a Gaussian cloud:

        n0 = N / (sigma^3 * (2*pi)^(3/2))

    Parameters
    ----------
    results : list of (t_arr, z_arr, v_arr) tuples
    N_total : int
        Total number of molecules in the trap.
    time_bins : int

    Returns
    -------
    t_centres : np.ndarray
        Time bin centres (s).
    n0_arr : np.ndarray
        Peak density (1/m^3) at each time bin.
    """
    t_centres, sigma_z, _, n_alive = sigma_vs_time(results, time_bins)

    n0_arr = np.zeros(time_bins)
    for i_t in range(time_bins):
        s = sigma_z[i_t]
        if np.isfinite(s) and s > 0:
            # Scale N by survival fraction
            n_eff = N_total * (n_alive[i_t] / len(results)) if len(results) > 0 else 0
            n0_arr[i_t] = n_eff / (s ** 3 * (2 * np.pi) ** 1.5)
        else:
            n0_arr[i_t] = 0.0

    return t_centres, n0_arr


# =====================================================================
# Survival check
# =====================================================================

def survived(result: Tuple[np.ndarray, np.ndarray, np.ndarray],
             r_max: float = 3e-3) -> bool:
    """
    Check if a molecule is still trapped at the end of its trajectory.

    Parameters
    ----------
    result : (t_arr, z_arr, v_arr)
        Output of simulate_trajectory.
    r_max : float
        Maximum distance from origin to count as trapped (m).

    Returns
    -------
    bool
        True if |z_final| < r_max.
    """
    _, z_arr, _ = result
    return abs(z_arr[-1]) < r_max


# =====================================================================
# Capture fraction
# =====================================================================

def capture_fraction(results: list,
                     r_max: float = 3e-3,
                     t_min: float = 1e-3) -> float:
    """
    Fraction of molecules still trapped after t_min.

    Parameters
    ----------
    results : list of (t_arr, z_arr, v_arr) tuples
    r_max : float
        Trapping radius (m).
    t_min : float
        Minimum simulation time to consider (s).
        Only count trajectories that ran at least this long.

    Returns
    -------
    fraction : float
        Fraction of molecules with |z(t_end)| < r_max and t_end >= t_min.
    """
    if not results:
        return 0.0

    n_trapped = 0
    n_valid = 0

    for t_arr, z_arr, v_arr in results:
        if t_arr[-1] < t_min:
            continue
        n_valid += 1
        # Find the index closest to t_min
        idx = np.searchsorted(t_arr, t_min)
        if idx >= len(t_arr):
            idx = len(t_arr) - 1
        if abs(z_arr[idx]) < r_max:
            n_trapped += 1

    if n_valid == 0:
        return 0.0

    return n_trapped / n_valid
