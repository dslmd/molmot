"""
MOT simulator classes for RF and DC magneto-optical traps.

Port of the simulation logic from OpticalBlochEquations.jl examples.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..obe.fields import make_rf_mot_beams, make_dc_mot_beams
from ..obe.rate_equations import solve_rate_equations_auto as solve_rate_equations


class RFMOTSimulator:
    """
    RF MOT simulator with polarisation/B-field switching.

    The RF MOT alternates between two half-cycles:
      Phase 0: sigma+ forward, +B gradient
      Phase 1: sigma- forward, -B gradient

    The force is time-averaged over the two phases.

    Parameters
    ----------
    mol_data : MolecularData
    delta_Gamma : float
        Detuning in units of Gamma (negative = red).
    s0 : float
        Saturation parameter per sideband per beam.
    B_gradient : float
        Magnetic field gradient in T/m.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    """

    def __init__(self, mol_data, delta_Gamma: float = -1.0,
                 s0: float = 1.0, B_gradient: float = 0.16,
                 Gamma_eff_factor: float = 1.0):
        self.mol_data = mol_data
        self.delta_Gamma = delta_Gamma
        self.s0 = s0
        self.B_gradient = B_gradient
        self.Gamma_eff_factor = Gamma_eff_factor

    def force(self, v: float, z: float) -> Tuple[float, np.ndarray, float]:
        """
        Compute time-averaged RF MOT force.

        Parameters
        ----------
        v : float
            Molecular velocity (m/s).
        z : float
            Position (m).

        Returns
        -------
        force : float
            Time-averaged force (N).
        populations : np.ndarray
            Time-averaged ground-state populations.
        R_scatter : float
            Time-averaged scattering rate (rad/s).
        """
        # Phase 0: sigma+ with +B
        beams_0 = make_rf_mot_beams(self.mol_data,
                                     delta_Gamma=self.delta_Gamma,
                                     s0=self.s0, phase=0)
        F0, p0, R0 = solve_rate_equations(
            self.mol_data, beams_0, v, z, self.B_gradient,
            self.Gamma_eff_factor)

        # Phase 1: sigma- with -B
        beams_1 = make_rf_mot_beams(self.mol_data,
                                     delta_Gamma=self.delta_Gamma,
                                     s0=self.s0, phase=1)
        F1, p1, R1 = solve_rate_equations(
            self.mol_data, beams_1, v, z, -self.B_gradient,
            self.Gamma_eff_factor)

        # Time average
        force = 0.5 * (F0 + F1)
        pop = 0.5 * (p0 + p1)
        R_scatter = 0.5 * (R0 + R1)

        return force, pop, R_scatter


class DCMOTSimulator:
    """
    DC MOT simulator with static fields.

    Supports 2-frequency and 4-frequency configurations.

    Parameters
    ----------
    mol_data : MolecularData
    delta_Gamma : float
        Overall detuning in units of Gamma.
    split_Gamma : float
        Polarisation split in units of Gamma.
    s0 : float
        Saturation parameter per component per beam.
    B_gradient : float
        Magnetic field gradient in T/m.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    """

    def __init__(self, mol_data, delta_Gamma: float = -0.20,
                 split_Gamma: float = 0.70,
                 s0: float = 1.0, B_gradient: float = 0.16,
                 Gamma_eff_factor: float = 1.0):
        self.mol_data = mol_data
        self.delta_Gamma = delta_Gamma
        self.split_Gamma = split_Gamma
        self.s0 = s0
        self.B_gradient = B_gradient
        self.Gamma_eff_factor = Gamma_eff_factor

        # Pre-build the beam configuration
        self._beams = make_dc_mot_beams(
            self.mol_data,
            delta_Gamma=self.delta_Gamma,
            split_Gamma=self.split_Gamma,
            s0=self.s0)

    def force(self, v: float, z: float) -> Tuple[float, np.ndarray, float]:
        """
        Compute the DC MOT force.

        Parameters
        ----------
        v : float
            Molecular velocity (m/s).
        z : float
            Position (m).

        Returns
        -------
        force : float
            Force (N).
        populations : np.ndarray
            Ground-state populations.
        R_scatter : float
            Scattering rate (rad/s).
        """
        return solve_rate_equations(
            self.mol_data, self._beams, v, z, self.B_gradient,
            self.Gamma_eff_factor)

    def update_beams(self):
        """Rebuild beams after parameter changes."""
        self._beams = make_dc_mot_beams(
            self.mol_data,
            delta_Gamma=self.delta_Gamma,
            split_Gamma=self.split_Gamma,
            s0=self.s0)
