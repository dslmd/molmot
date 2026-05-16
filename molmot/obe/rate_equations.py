"""
Multi-level rate-equation solver for molecular MOT force computation.

Port of the rate-equation solver from OpticalBlochEquations.jl.

This is faster than the full OBE (Lindblad) solver and suitable for
parameter scans.  It assumes that coherences between ground states
oscillate fast relative to the scattering rate and can be adiabatically
eliminated (secular approximation).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.linalg import solve

from ..constants import hbar
from .fields import LaserBeam


def solve_rate_equations(
    mol_data,
    beams: List[LaserBeam],
    v: float,
    z: float,
    B_gradient: float,
    Gamma_eff_factor: float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    """
    Solve steady-state rate equations for populations and force.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data container with attributes:
        - energies : (n_states,) array of state energies in Hz
        - d_squared : (n_ground, n_excited, 3) array of |TDM|^2
        - zeeman_z_diag : (n_states,) diagonal Zeeman shifts per Gauss
          (in units that give Hz when multiplied by B in Gauss and Gamma/(2*pi))
        - Gamma : float, natural linewidth in rad/s
        - k : float, wavenumber in 1/m
        - n_ground : int
        - n_excited : int
        - n_states : int
    beams : list of LaserBeam
    v : float
        Molecular velocity along z (m/s).
    z : float
        Position along z (m).
    B_gradient : float
        Magnetic field gradient dBz/dz in T/m.
        At position z, B = B_gradient * z (Tesla).
        Converted internally: 1 T = 10^4 Gauss.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor (default 1.0).
        Accounts for repumping cycle time in molecules.

    Returns
    -------
    force : float
        Net radiation pressure force in Newtons.
    populations : np.ndarray, shape (n_ground,)
        Steady-state ground state populations.
    R_scatter : float
        Total scattering rate in rad/s.
    """
    Gamma = mol_data.Gamma
    k = mol_data.k
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    n_s = mol_data.n_states

    # Magnetic field at position z
    B_tesla = B_gradient * z
    B_gauss = B_tesla * 1e4  # Convert T -> Gauss

    # Zeeman-shifted state energies
    zeeman_diag = mol_data.zeeman_z_diag  # shifts per Gauss * Gamma/(2*pi) -> Hz
    E_zeeman = B_gauss * zeeman_diag * Gamma / (2.0 * np.pi)
    Es = mol_data.energies + E_zeeman  # Hz

    # Branching ratios from |d|^2
    d_sq = mol_data.d_squared  # (n_g, n_e, 3)
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = 0.0
        for ig in range(n_g):
            total += np.sum(d_sq[ig, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq[ig, ie, :]) / total

    # Excitation rates and force contributions
    n_beams = len(beams)
    R_exc = np.zeros((n_g, n_e, n_beams))
    F_per = np.zeros((n_g, n_e, n_beams))

    gamma_half = Gamma / 2.0
    gamma_half_sq = gamma_half ** 2

    # First pass: compute unsaturated rates for saturation denominator
    # s_total[ig] = sum over all beams and excited states of s0 * d2 * L
    s_total = np.zeros(n_g)

    for ib, beam in enumerate(beams):
        kdir = beam.kdir_z
        doppler = -k * kdir * v / (2.0 * np.pi)  # Hz
        q_eff = beam.polarization

        for ig in range(n_g):
            for ie in range(n_e):
                d2 = d_sq[ig, ie, q_eff]
                if d2 < 1e-15:
                    continue

                # Transition frequency (Hz)
                omega_trans = Es[n_g + ie] - Es[ig]

                # Effective detuning (rad/s)
                delta_eff = (beam.freq_offset + doppler - omega_trans) * 2.0 * np.pi

                # Lorentzian
                L = gamma_half_sq / (delta_eff ** 2 + gamma_half_sq)

                s_total[ig] += beam.s0 * d2 * L

    # Second pass: compute rates with saturation correction
    for ib, beam in enumerate(beams):
        kdir = beam.kdir_z
        doppler = -k * kdir * v / (2.0 * np.pi)  # Hz
        q_eff = beam.polarization

        for ig in range(n_g):
            for ie in range(n_e):
                d2 = d_sq[ig, ie, q_eff]
                if d2 < 1e-15:
                    continue

                # Transition frequency (Hz)
                omega_trans = Es[n_g + ie] - Es[ig]

                # Effective detuning (rad/s)
                delta_eff = (beam.freq_offset + doppler - omega_trans) * 2.0 * np.pi

                # Lorentzian
                L = gamma_half_sq / (delta_eff ** 2 + gamma_half_sq)

                # Scattering rate with saturation correction
                rate = gamma_half * beam.s0 * d2 * L * Gamma_eff_factor / (1.0 + s_total[ig])

                R_exc[ig, ie, ib] = rate
                F_per[ig, ie, ib] = hbar * k * kdir * rate

    # Steady-state population
    R_sum = np.sum(R_exc, axis=2)  # (n_g, n_e)

    M = np.zeros((n_g, n_g))
    for ig in range(n_g):
        M[ig, ig] -= np.sum(R_sum[ig, :])  # loss from excitation
        for ik in range(n_g):
            for ie in range(n_e):
                M[ig, ik] += R_sum[ik, ie] * BR[ie, ig]  # gain from decay

    # Replace last equation with normalisation
    M_sol = M.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(n_g)
    rhs[-1] = 1.0

    try:
        p = solve(M_sol, rhs)
    except np.linalg.LinAlgError:
        p = np.ones(n_g) / n_g

    p = np.maximum(p, 0.0)
    p_sum = np.sum(p)
    if p_sum > 0:
        p /= p_sum
    else:
        p = np.ones(n_g) / n_g

    # Total force and scattering rate
    force = 0.0
    R_scatter = 0.0
    for ig in range(n_g):
        force += p[ig] * np.sum(F_per[ig, :, :])
        R_scatter += p[ig] * np.sum(R_sum[ig, :])

    return force, p, R_scatter
