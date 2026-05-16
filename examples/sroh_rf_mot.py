#!/usr/bin/env python3
"""
SrOH Magneto-Optical Trap (MOT) Simulation
============================================

Comprehensive simulation package for SrOH molecular MOT, supporting:
  1. RF MOT  -- rapid polarization/B-field switching at 1.4 MHz
  2. DC MOT  -- static fields, 4-frequency polarization-split configuration

Molecular data is loaded from pre-computed CSV files (exported from the
validated Julia/OpticalBlochEquations.jl code by C. Hallas, Harvard).

The 16-level structure:
  - 12 ground states (X, N=1):
      g1        : J=1/2, F=0, M=0
      g2--g4    : J=1/2, F=1, M=+1, 0, -1
      g5--g7    : J=3/2, F=1, M=-1, 0, +1
      g8--g12   : J=3/2, F=2, M=-2, -1, 0, +1, +2
  - 4 excited states (A, J'=1/2):
      e1        : F'=0, M'=0
      e2        : F'=1, M'=0
      e3        : F'=1, M'=-1
      e4        : F'=1, M'=+1

Physical constants:
  lambda = 687 nm,  Gamma = 2*pi * 6.4 MHz,  mass = 105 amu,  k = 9.146e6 /m

Author: auto-generated from Julia reference code
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from scipy import constants as const
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ======================================================================
#  Physical constants
# ======================================================================

h_SI   = const.h                       # 6.626e-34 J s
hbar   = const.hbar                     # 1.055e-34 J s
c_SI   = const.c                        # 2.998e8 m/s
kB     = const.k                        # 1.381e-23 J/K
amu    = const.atomic_mass              # 1.661e-27 kg
muB    = const.value("Bohr magneton")   # 9.274e-24 J/T
gS     = 2.0023193                      # electron g-factor


# ======================================================================
#  SrOH molecular parameters
# ======================================================================

LAMBDA     = 687.0e-9                   # transition wavelength (m)
GAMMA      = 2.0 * np.pi * 6.4e6       # natural linewidth (rad/s)
GAMMA_HZ   = 6.4e6                      # Gamma / (2*pi) (Hz)
MASS       = 105.0 * amu                # molecular mass (kg)
K_WAVE     = 2.0 * np.pi / LAMBDA      # wavenumber (1/m)
PHOTON_BUDGET = 12600                   # approx photon budget before loss

N_GROUND   = 12
N_EXCITED  = 4
N_STATES   = 16

# Saturation intensity: I_sat = pi * h * c * Gamma_Hz / (3 * lambda^3)
I_SAT = np.pi * h_SI * c_SI * GAMMA_HZ / (3.0 * LAMBDA**3)   # W/m^2


# ======================================================================
#  A. Data Loading
# ======================================================================

@dataclass
class MolecularData:
    """Container for the SrOH molecular Hamiltonian data."""
    energies:   np.ndarray   # (16,) state energies in Hz
    tdm:        np.ndarray   # (16, 16, 3) complex transition dipole matrix
    zeeman_x:   np.ndarray   # (16, 16) complex Zeeman matrix, x-component
    zeeman_y:   np.ndarray   # (16, 16) complex Zeeman matrix, y-component
    zeeman_z:   np.ndarray   # (16, 16) complex Zeeman matrix, z-component

    # Derived quantities (populated by post_init)
    E_ground:      np.ndarray = field(default=None, repr=False)
    E_excited:     np.ndarray = field(default=None, repr=False)
    E_J12_mean:    float = 0.0
    E_J32_mean:    float = 0.0
    E_e_mean:      float = 0.0
    omega_J12:     float = 0.0   # mean transition freq for J=1/2 manifold (Hz)
    omega_J32:     float = 0.0   # mean transition freq for J=3/2 manifold (Hz)
    omega_mean:    float = 0.0   # overall mean transition freq (Hz)
    d_squared:     np.ndarray = field(default=None, repr=False)  # |d|^2 array

    def __post_init__(self):
        self.E_ground  = self.energies[:N_GROUND]
        self.E_excited = self.energies[N_GROUND:]
        self.E_e_mean  = np.mean(self.E_excited)

        # The Julia code groups ground states 0..3 as "J=1/2" (4 states)
        # and 4..11 as "J=3/2" (8 states).
        # Energy-based identification:
        #   g1--g4  : J=1/2 (F=0, F=1)  -- lower energy
        #   g5--g12 : J=3/2 (F=1, F=2)  -- higher energy
        self.E_J12_mean = np.mean(self.E_ground[:4])
        self.E_J32_mean = np.mean(self.E_ground[4:])

        self.omega_J12 = self.E_e_mean - self.E_J12_mean   # Hz
        self.omega_J32 = self.E_e_mean - self.E_J32_mean   # Hz
        self.omega_mean = self.E_e_mean - np.mean(self.E_ground)

        # Pre-compute |d|^2 for each (ground, excited, q)
        self.d_squared = np.zeros((N_GROUND, N_EXCITED, 3))
        for ig in range(N_GROUND):
            for ie in range(N_EXCITED):
                for q in range(3):
                    self.d_squared[ig, ie, q] = np.abs(
                        self.tdm[ig, N_GROUND + ie, q]
                    ) ** 2


def load_molecular_data(data_dir: str) -> MolecularData:
    """
    Load molecular structure data from CSV files exported by the Julia code.

    Expected files in data_dir:
        sroh_energies.csv          -- 16 energies, one per line (Hz)
        sroh_tdm_re.csv            -- 48 x 16 tab-separated (real part)
        sroh_tdm_im.csv            -- 48 x 16 tab-separated (imag part)
        sroh_zeeman_{x,y,z}_re.csv -- 16 x 16 tab-separated
        sroh_zeeman_{x,y,z}_im.csv -- 16 x 16 tab-separated

    The TDM files store d[i, j, q] as a 48x16 matrix where rows are
    (state_i * 3 + q) for state_i = 0..15 and q = 0..2 (sigma-, pi, sigma+).
    """
    print("=" * 70)
    print("  Loading SrOH molecular data")
    print("=" * 70)

    def _path(name):
        return os.path.join(data_dir, name)

    # --- Energies ---
    energies = np.loadtxt(_path("sroh_energies.csv"))
    assert energies.shape == (N_STATES,), f"Expected {N_STATES} energies, got {energies.shape}"
    print(f"  Loaded {N_STATES} state energies")

    # --- Transition dipole matrix ---
    tdm_re = np.loadtxt(_path("sroh_tdm_re.csv"))  # (48, 16)
    tdm_im = np.loadtxt(_path("sroh_tdm_im.csv"))  # (48, 16)
    assert tdm_re.shape == (48, 16), f"TDM shape mismatch: {tdm_re.shape}"

    # Reshape from (48, 16) → (16, 3, 16) → transpose to (16, 16, 3)
    tdm_flat = tdm_re + 1j * tdm_im   # (48, 16)
    tdm = np.zeros((N_STATES, N_STATES, 3), dtype=complex)
    for state_i in range(N_STATES):
        for q in range(3):
            row_idx = state_i * 3 + q
            tdm[state_i, :, q] = tdm_flat[row_idx, :]
    print(f"  Loaded TDM: {tdm.shape}")

    # --- Zeeman matrices ---
    def _load_zeeman(axis):
        re = np.loadtxt(_path(f"sroh_zeeman_{axis}_re.csv"))
        im = np.loadtxt(_path(f"sroh_zeeman_{axis}_im.csv"))
        assert re.shape == (N_STATES, N_STATES), f"Zeeman {axis} shape: {re.shape}"
        return re + 1j * im

    zeeman_x = _load_zeeman("x")
    zeeman_y = _load_zeeman("y")
    zeeman_z = _load_zeeman("z")
    print(f"  Loaded Zeeman matrices: {zeeman_z.shape}")

    mol = MolecularData(
        energies=energies,
        tdm=tdm,
        zeeman_x=zeeman_x,
        zeeman_y=zeeman_y,
        zeeman_z=zeeman_z,
    )

    # Print summary
    print(f"\n  lambda   = {LAMBDA*1e9:.1f} nm")
    print(f"  Gamma    = 2*pi * {GAMMA_HZ/1e6:.1f} MHz")
    print(f"  mass     = {MASS/amu:.0f} amu")
    print(f"  k        = {K_WAVE:.4e} /m")
    print(f"  I_sat    = {I_SAT/10:.1f} mW/cm^2")
    print(f"  SR split = {(mol.E_J32_mean - mol.E_J12_mean)/1e6:.1f} MHz")
    print(f"  omega_J12 = {mol.omega_J12/1e6:.2f} MHz")
    print(f"  omega_J32 = {mol.omega_J32/1e6:.2f} MHz")

    # Print ground state energies relative to mean
    E_g_mean = np.mean(mol.E_ground)
    print("\n  Ground state energies (MHz from mean):")
    for i in range(N_GROUND):
        print(f"    g{i+1:2d}: {(mol.E_ground[i] - E_g_mean)/1e6:+8.3f} MHz")

    # Print excited state energies relative to their mean
    print("\n  Excited state energies (MHz from mean):")
    for i in range(N_EXCITED):
        print(f"    e{i+1:2d}: {(mol.E_excited[i] - mol.E_e_mean)/1e6:+8.4f} MHz")

    # Print nonzero TDM elements
    print("\n  Nonzero TDM elements |d|^2 > 1e-4:")
    pol_labels = ["sigma-", "pi", "sigma+"]
    n_printed = 0
    for ig in range(N_GROUND):
        for ie in range(N_EXCITED):
            for q in range(3):
                d2 = mol.d_squared[ig, ie, q]
                if d2 > 1e-4:
                    print(f"    g{ig+1:2d} -> e{ie+1} ({pol_labels[q]:6s}): |d|^2 = {d2:.4f}")
                    n_printed += 1
    print(f"  ({n_printed} nonzero transitions)")

    return mol


# ======================================================================
#  B. Rate Equation Solver
# ======================================================================

@dataclass
class LaserBeam:
    """Description of a single laser beam / frequency component."""
    direction:    int     # +1 or -1 (propagation direction along z-axis)
    freq_offset:  float   # laser frequency in Hz (absolute)
    q_pol:        int     # polarization: 0=sigma-, 1=pi, 2=sigma+
    s0:           float   # saturation parameter (I / I_sat)


def solve_rate_equations(
    mol_data:    MolecularData,
    beams:       List[LaserBeam],
    v:           float,
    z:           float,
    B_gradient:  float,
) -> Tuple[float, np.ndarray, float]:
    """
    Solve steady-state rate equations for the 16-level SrOH system.

    Parameters
    ----------
    mol_data    : MolecularData  -- molecular Hamiltonian data
    beams       : list of LaserBeam  -- laser beam configurations
    v           : float  -- molecular velocity along z (m/s)
    z           : float  -- position along z-axis (m)
    B_gradient  : float  -- magnetic field gradient (G/cm)

    Returns
    -------
    force       : float  -- radiation pressure force (N)
    populations : np.ndarray (N_GROUND,)  -- steady-state ground populations
    R_scatter   : float  -- total scattering rate (rad/s)
    """
    # --- Zeeman-shifted energies ---
    # B-field at position z for anti-Helmholtz quadrupole (axial direction)
    B_gauss = B_gradient * z * 100.0   # z in m -> z_cm; B = B' * z_cm (Gauss)

    # Zeeman energy shifts: E_Z[i] = B * diag(Zeeman_z)[i] * Gamma/(2*pi) (Hz)
    # The Zeeman matrices are pre-scaled by 2*pi*gS*muB/Gamma in the Julia code,
    # so the energy shift in Hz is: B * Z_zz[i,i] * Gamma/(2*pi)
    zeeman_diag = np.real(np.diag(mol_data.zeeman_z))
    E_zeeman = B_gauss * zeeman_diag * GAMMA / (2.0 * np.pi)   # Hz

    Es = mol_data.energies + E_zeeman  # Zeeman-shifted energies (Hz)

    # --- Branching ratios ---
    BR = np.zeros((N_EXCITED, N_GROUND))
    for ie in range(N_EXCITED):
        total = 0.0
        for ig in range(N_GROUND):
            total += np.sum(mol_data.d_squared[ig, ie, :])
        if total < 1e-30:
            continue
        for ig in range(N_GROUND):
            BR[ie, ig] = np.sum(mol_data.d_squared[ig, ie, :]) / total

    # --- Excitation rates and forces per beam ---
    n_beams = len(beams)
    R_exc = np.zeros((N_GROUND, N_EXCITED, n_beams))
    F_per = np.zeros((N_GROUND, N_EXCITED, n_beams))

    gamma_half = GAMMA / 2.0
    gamma_half_sq = gamma_half ** 2

    for ib, beam in enumerate(beams):
        # Doppler shift (Hz)
        doppler = -K_WAVE * beam.direction * v / (2.0 * np.pi)

        q_eff = beam.q_pol

        for ig in range(N_GROUND):
            for ie in range(N_EXCITED):
                d2 = mol_data.d_squared[ig, ie, q_eff]
                if d2 < 1e-15:
                    continue

                # Transition frequency (Hz)
                omega_trans = Es[N_GROUND + ie] - Es[ig]

                # Effective detuning (rad/s)
                delta_eff = (beam.freq_offset + doppler - omega_trans) * 2.0 * np.pi

                # Lorentzian line shape
                L = gamma_half_sq / (delta_eff**2 + gamma_half_sq)

                # Scattering rate from this beam (rad/s)
                rate = gamma_half * beam.s0 * d2 * L

                R_exc[ig, ie, ib] = rate
                F_per[ig, ie, ib] = hbar * K_WAVE * beam.direction * rate

    # --- Steady-state population (ground states only) ---
    # Rate matrix: dp_ig/dt = -sum_ie R_ig_ie * p_ig + sum_ik,ie R_ik_ie * BR_ie_ig * p_ik
    R_sum = np.sum(R_exc, axis=2)   # (N_GROUND, N_EXCITED) total excitation rate

    M = np.zeros((N_GROUND, N_GROUND))
    for ig in range(N_GROUND):
        # Loss: excitation out of |ig>
        M[ig, ig] -= np.sum(R_sum[ig, :])
        # Gain: spontaneous emission from excited states pumped from |ik>
        for ik in range(N_GROUND):
            for ie in range(N_EXCITED):
                M[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    # Replace last equation with normalisation constraint: sum(p) = 1
    M_sol = M.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(N_GROUND)
    rhs[-1] = 1.0

    try:
        p = solve(M_sol, rhs)
    except np.linalg.LinAlgError:
        p = np.ones(N_GROUND) / N_GROUND

    # Clamp negative populations
    p = np.maximum(p, 0.0)
    p_sum = np.sum(p)
    if p_sum > 0:
        p /= p_sum
    else:
        p = np.ones(N_GROUND) / N_GROUND

    # --- Total force and scattering rate ---
    force = 0.0
    R_scatter = 0.0
    for ig in range(N_GROUND):
        force     += p[ig] * np.sum(F_per[ig, :, :])
        R_scatter += p[ig] * np.sum(R_sum[ig, :])

    return force, p, R_scatter


# ======================================================================
#  C. RF MOT Simulator
# ======================================================================

def make_rf_mot_beams(
    mol_data:    MolecularData,
    delta_gamma: float = -1.0,
    s0:          float = 1.0,
    phase:       int   = 0,
) -> List[LaserBeam]:
    """
    Create laser beams for the RF MOT configuration.

    The RF MOT uses two frequency sidebands (addressing J=1/2 and J=3/2
    ground-state manifolds).  The polarization and B-field direction switch
    synchronously at 1.4 MHz (faster than Gamma).

    At phase=0: all forward beams sigma+, B-field in +z direction
    At phase=1: all forward beams sigma-, B-field in -z direction

    In the RF MOT, at any given instant ALL beams have the SAME handedness
    (either all sigma+ or all sigma-).  The retro-reflected beam flips
    handedness as usual.

    Parameters
    ----------
    mol_data     : MolecularData
    delta_gamma  : overall detuning in units of Gamma (negative = red)
    s0           : saturation parameter per sideband per beam
    phase        : 0 or 1 (RF phase)

    Returns
    -------
    beams : list of LaserBeam
    """
    delta_hz = delta_gamma * GAMMA / (2.0 * np.pi)   # detuning in Hz

    # Two sideband frequencies: one for J=1/2 manifold, one for J=3/2
    freq_J12 = mol_data.omega_J12 + delta_hz
    freq_J32 = mol_data.omega_J32 + delta_hz

    beams = []

    if phase == 0:
        # All forward beams are sigma+ at phase 0
        q_fwd = 2   # sigma+
    else:
        # All forward beams are sigma- at phase 1
        q_fwd = 0   # sigma-

    for direction in [+1, -1]:
        # For retro-reflected beam, sigma+ <-> sigma-
        if direction > 0:
            q_eff = q_fwd
        else:
            q_eff = 2 - q_fwd   # flip sigma+ <-> sigma-

        beams.append(LaserBeam(direction=direction, freq_offset=freq_J12, q_pol=q_eff, s0=s0))
        beams.append(LaserBeam(direction=direction, freq_offset=freq_J32, q_pol=q_eff, s0=s0))

    return beams


def rf_mot_force(
    mol_data:    MolecularData,
    v:           float,
    z:           float,
    B_gradient:  float,
    delta_gamma: float = -1.0,
    s0:          float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    """
    Compute the time-averaged RF MOT force by averaging over the two
    half-cycles of the RF switching.

    Phase 0: sigma+ polarisation, +B gradient
    Phase 1: sigma- polarisation, -B gradient (equivalent to flipped B)

    Returns: (force, populations, scattering_rate)
    """
    # Phase 0: sigma+ with +B
    beams_0 = make_rf_mot_beams(mol_data, delta_gamma=delta_gamma, s0=s0, phase=0)
    F0, p0, R0 = solve_rate_equations(mol_data, beams_0, v, z, B_gradient)

    # Phase 1: sigma- with -B (negate B_gradient)
    beams_1 = make_rf_mot_beams(mol_data, delta_gamma=delta_gamma, s0=s0, phase=1)
    F1, p1, R1 = solve_rate_equations(mol_data, beams_1, v, z, -B_gradient)

    # Time-averaged quantities
    force      = 0.5 * (F0 + F1)
    pop        = 0.5 * (p0 + p1)
    R_scatter  = 0.5 * (R0 + R1)

    return force, pop, R_scatter


# ======================================================================
#  D. DC MOT Simulator
# ======================================================================

def make_dc_mot_beams(
    mol_data:      MolecularData,
    delta_gamma:   float = -0.20,
    split_gamma:   float = 0.70,
    s0:            float = 1.0,
) -> List[LaserBeam]:
    """
    Create laser beams for the 4-frequency DC MOT configuration.

    For each spin-rotation manifold (J=1/2 and J=3/2):
      - sigma+ at detuning Delta + delta_split
      - sigma- at detuning Delta - delta_split

    This creates 4 frequency components x 2 directions = 8 beams total
    (along z-axis; the full 3D MOT would have 3 retro-reflected axes).

    Parameters
    ----------
    mol_data     : MolecularData
    delta_gamma  : overall detuning in units of Gamma (negative = red)
    split_gamma  : polarisation split in units of Gamma
    s0           : saturation parameter per component per beam

    Returns
    -------
    beams : list of LaserBeam
    """
    delta_hz = delta_gamma * GAMMA / (2.0 * np.pi)
    split_hz = split_gamma * GAMMA / (2.0 * np.pi)

    # 4 frequency components:
    #   1. J=3/2 manifold, sigma+:  omega_J32 + Delta + delta_split
    #   2. J=3/2 manifold, sigma-:  omega_J32 + Delta - delta_split
    #   3. J=1/2 manifold, sigma+:  omega_J12 + Delta + delta_split
    #   4. J=1/2 manifold, sigma-:  omega_J12 + Delta - delta_split
    components = [
        (mol_data.omega_J32 + delta_hz + split_hz, 2),   # sigma+ (q=2)
        (mol_data.omega_J32 + delta_hz - split_hz, 0),   # sigma- (q=0)
        (mol_data.omega_J12 + delta_hz + split_hz, 2),   # sigma+
        (mol_data.omega_J12 + delta_hz - split_hz, 0),   # sigma-
    ]

    beams = []
    for freq, q_fwd in components:
        for direction in [+1, -1]:
            # For retro-reflected beam, sigma+ <-> sigma-
            if direction > 0:
                q_eff = q_fwd
            else:
                if q_fwd == 0:
                    q_eff = 2
                elif q_fwd == 2:
                    q_eff = 0
                else:
                    q_eff = 1  # pi unchanged
            beams.append(LaserBeam(
                direction=direction,
                freq_offset=freq,
                q_pol=q_eff,
                s0=s0,
            ))

    return beams


def dc_mot_force(
    mol_data:      MolecularData,
    v:             float,
    z:             float,
    B_gradient:    float,
    delta_gamma:   float = -0.20,
    split_gamma:   float = 0.70,
    s0:            float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    """
    Compute the DC MOT force for the 4-frequency configuration.

    Returns: (force, populations, scattering_rate)
    """
    beams = make_dc_mot_beams(mol_data, delta_gamma, split_gamma, s0)
    return solve_rate_equations(mol_data, beams, v, z, B_gradient)


# ======================================================================
#  E. Force Scanning Functions
# ======================================================================

def force_vs_z(
    solver:     Callable,
    z_range:    Tuple[float, float],
    n_points:   int,
    v:          float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force as a function of position.

    Parameters
    ----------
    solver    : callable(mol_data, v, z, B_gradient, **kw) -> (F, p, R)
    z_range   : (z_min, z_max) in metres
    n_points  : number of evaluation points
    v         : molecular velocity (m/s)
    **kwargs  : additional keyword arguments passed to solver

    Returns
    -------
    z_arr   : (n_points,) position array (m)
    F_arr   : (n_points,) force array (N)
    pop_arr : (n_points, N_GROUND) population array
    R_arr   : (n_points,) scattering rate array (rad/s)
    """
    z_arr = np.linspace(z_range[0], z_range[1], n_points)
    F_arr = np.zeros(n_points)
    pop_arr = np.zeros((n_points, N_GROUND))
    R_arr = np.zeros(n_points)

    for i, z in enumerate(z_arr):
        F_arr[i], pop_arr[i, :], R_arr[i] = solver(v=v, z=z, **kwargs)

    return z_arr, F_arr, pop_arr, R_arr


def force_vs_v(
    solver:     Callable,
    v_range:    Tuple[float, float],
    n_points:   int,
    z:          float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute force as a function of velocity.

    Returns
    -------
    v_arr : (n_points,) velocity array (m/s)
    F_arr : (n_points,) force array (N)
    """
    v_arr = np.linspace(v_range[0], v_range[1], n_points)
    F_arr = np.zeros(n_points)

    for i, v in enumerate(v_arr):
        F_arr[i], _, _ = solver(v=v, z=z, **kwargs)

    return v_arr, F_arr


# ======================================================================
#  F. Trajectory Simulation
# ======================================================================

def simulate_trajectory(
    solver:   Callable,
    z0:       float,
    v0:       float,
    t_max:    float,
    dt:       float,
    z_escape: float = 0.015,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a 1D molecular trajectory in the MOT.

    Parameters
    ----------
    solver    : callable(v=v, z=z, **kw) -> (F, p, R)
    z0        : initial position (m)
    v0        : initial velocity (m/s)
    t_max     : total simulation time (s)
    dt        : time step (s)
    z_escape  : escape boundary (m); trajectory stops if |z| > z_escape
    **kwargs  : passed to solver

    Returns
    -------
    t_arr : time array (s)
    z_arr : position array (m)
    v_arr : velocity array (m/s)
    """
    n_steps = int(t_max / dt)
    t_arr = np.arange(n_steps) * dt
    z_arr = np.zeros(n_steps)
    v_arr = np.zeros(n_steps)

    z_arr[0] = z0
    v_arr[0] = v0

    for j in range(1, n_steps):
        F, _, _ = solver(v=v_arr[j-1], z=z_arr[j-1], **kwargs)
        a = F / MASS
        v_arr[j] = v_arr[j-1] + a * dt
        z_arr[j] = z_arr[j-1] + v_arr[j] * dt

        if abs(z_arr[j]) > z_escape:
            z_arr[j:] = z_arr[j]
            v_arr[j:] = v_arr[j]
            break

    return t_arr, z_arr, v_arr


# ======================================================================
#  G. Parameter Optimisation
# ======================================================================

def optimise_dc_mot_parameters(
    mol_data:      MolecularData,
    B_gradient:    float = 16.0,
    s0:            float = 1.0,
    delta_range:   Tuple[float, float] = (-3.0, -0.1),
    split_range:   Tuple[float, float] = (0.05, 2.0),
    n_delta:       int = 30,
    n_split:       int = 30,
    dz:            float = 0.3e-3,
    dv:            float = 0.1,
) -> Dict:
    """
    Scan over detuning and polarisation split to find optimal DC MOT parameters.

    Returns a dict with scan results and optimal parameters.
    """
    print("\n" + "=" * 70)
    print("  DC MOT Parameter Optimisation")
    print("=" * 70)
    print(f"  Scanning Delta = [{delta_range[0]:.1f}, {delta_range[1]:.1f}] Gamma  ({n_delta} pts)")
    print(f"  Scanning split = [{split_range[0]:.2f}, {split_range[1]:.2f}] Gamma  ({n_split} pts)")
    print(f"  B' = {B_gradient} G/cm,  s0 = {s0}")

    delta_scan = np.linspace(delta_range[0], delta_range[1], n_delta)
    split_scan = np.linspace(split_range[0], split_range[1], n_split)

    K_map = np.zeros((n_delta, n_split))
    beta_map = np.zeros((n_delta, n_split))

    t0 = time.time()
    total = n_delta * n_split
    count = 0

    for i_d, delta_g in enumerate(delta_scan):
        for i_s, split_g in enumerate(split_scan):
            # Spring constant: k = -dF/dz at z=0
            Fp, _, _ = dc_mot_force(mol_data, 0.0, +dz, B_gradient,
                                    delta_gamma=delta_g, split_gamma=split_g, s0=s0)
            Fm, _, _ = dc_mot_force(mol_data, 0.0, -dz, B_gradient,
                                    delta_gamma=delta_g, split_gamma=split_g, s0=s0)
            K_map[i_d, i_s] = -(Fp - Fm) / (2.0 * dz)

            # Damping coefficient: beta = -(dF/dv)/m at v=0
            Fvp, _, _ = dc_mot_force(mol_data, +dv, 0.0, B_gradient,
                                     delta_gamma=delta_g, split_gamma=split_g, s0=s0)
            Fvm, _, _ = dc_mot_force(mol_data, -dv, 0.0, B_gradient,
                                     delta_gamma=delta_g, split_gamma=split_g, s0=s0)
            beta_map[i_d, i_s] = -(Fvp - Fvm) / (2.0 * dv * MASS)

            count += 1
            if count % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"    {count}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Find optimum (maximum spring constant with positive damping)
    # Mask out regions with negative spring constant or negative damping
    merit = K_map.copy()
    merit[K_map <= 0] = 0
    merit[beta_map <= 0] = 0

    idx_best = np.unravel_index(np.argmax(merit), merit.shape)
    delta_best = delta_scan[idx_best[0]]
    split_best = split_scan[idx_best[1]]
    k_best = K_map[idx_best]
    beta_best = beta_map[idx_best]
    omega_best = np.sqrt(k_best / MASS) / (2.0 * np.pi) if k_best > 0 else 0.0

    elapsed = time.time() - t0
    print(f"\n  Scan completed in {elapsed:.1f} s")
    print(f"\n  OPTIMAL PARAMETERS:")
    print(f"    Delta = {delta_best:.2f} Gamma = {delta_best * GAMMA_HZ/1e6:.2f} MHz")
    print(f"    split = {split_best:.2f} Gamma = {split_best * GAMMA_HZ/1e6:.2f} MHz")
    print(f"    k     = {k_best:.3e} N/m")
    print(f"    omega = 2*pi * {omega_best:.1f} Hz")
    print(f"    beta  = {beta_best:.0f} /s")

    return {
        "delta_scan": delta_scan,
        "split_scan": split_scan,
        "K_map": K_map,
        "beta_map": beta_map,
        "delta_best": delta_best,
        "split_best": split_best,
        "k_best": k_best,
        "beta_best": beta_best,
        "omega_best": omega_best,
    }


def optimise_rf_mot_parameters(
    mol_data:      MolecularData,
    B_gradient:    float = 16.0,
    s0:            float = 1.0,
    delta_range:   Tuple[float, float] = (-3.0, -0.1),
    n_delta:       int = 30,
    dz:            float = 0.3e-3,
    dv:            float = 0.1,
) -> Dict:
    """
    Scan over detuning to find optimal RF MOT parameters.

    Returns a dict with scan results and optimal parameters.
    """
    print("\n" + "=" * 70)
    print("  RF MOT Parameter Optimisation (detuning scan)")
    print("=" * 70)
    print(f"  Scanning Delta = [{delta_range[0]:.1f}, {delta_range[1]:.1f}] Gamma  ({n_delta} pts)")
    print(f"  B' = {B_gradient} G/cm,  s0 = {s0}")

    delta_scan = np.linspace(delta_range[0], delta_range[1], n_delta)
    K_arr = np.zeros(n_delta)
    beta_arr = np.zeros(n_delta)

    t0 = time.time()

    for i_d, delta_g in enumerate(delta_scan):
        # Spring constant
        Fp, _, _ = rf_mot_force(mol_data, 0.0, +dz, B_gradient,
                                delta_gamma=delta_g, s0=s0)
        Fm, _, _ = rf_mot_force(mol_data, 0.0, -dz, B_gradient,
                                delta_gamma=delta_g, s0=s0)
        K_arr[i_d] = -(Fp - Fm) / (2.0 * dz)

        # Damping
        Fvp, _, _ = rf_mot_force(mol_data, +dv, 0.0, B_gradient,
                                 delta_gamma=delta_g, s0=s0)
        Fvm, _, _ = rf_mot_force(mol_data, -dv, 0.0, B_gradient,
                                 delta_gamma=delta_g, s0=s0)
        beta_arr[i_d] = -(Fvp - Fvm) / (2.0 * dv * MASS)

        if (i_d + 1) % 10 == 0:
            print(f"    {i_d+1}/{n_delta}")

    idx_best = np.argmax(K_arr * (K_arr > 0) * (beta_arr > 0))
    delta_best = delta_scan[idx_best]
    k_best = K_arr[idx_best]
    beta_best = beta_arr[idx_best]
    omega_best = np.sqrt(k_best / MASS) / (2.0 * np.pi) if k_best > 0 else 0.0

    elapsed = time.time() - t0
    print(f"\n  Scan completed in {elapsed:.1f} s")
    print(f"\n  OPTIMAL RF MOT PARAMETERS:")
    print(f"    Delta = {delta_best:.2f} Gamma = {delta_best * GAMMA_HZ/1e6:.2f} MHz")
    print(f"    k     = {k_best:.3e} N/m")
    print(f"    omega = 2*pi * {omega_best:.1f} Hz")
    print(f"    beta  = {beta_best:.0f} /s")

    return {
        "delta_scan": delta_scan,
        "K_arr": K_arr,
        "beta_arr": beta_arr,
        "delta_best": delta_best,
        "k_best": k_best,
        "beta_best": beta_best,
        "omega_best": omega_best,
    }


# ======================================================================
#  Helper: capture velocity estimation
# ======================================================================

def estimate_capture_velocity(
    solver:     Callable,
    v_test:     np.ndarray = None,
    z_start:    float = 3e-3,
    dt:         float = 1e-6,
    n_steps:    int = 5000,
    z_escape:   float = 0.015,
    **kwargs,
) -> float:
    """
    Estimate capture velocity by simulating trajectories at increasing
    initial velocities until the molecule escapes.

    Parameters
    ----------
    solver  : callable(v=v, z=z, **kw) -> (F, p, R)
    v_test  : array of test velocities (m/s), default 0.5 to 15
    **kwargs: passed to solver

    Returns
    -------
    v_capture : float -- maximum captured velocity (m/s)
    """
    if v_test is None:
        v_test = np.arange(0.5, 15.5, 0.5)

    v_capture = 0.0

    for v0 in v_test:
        z = z_start
        v = -v0   # approaching from positive z with negative velocity
        trapped = True
        for _ in range(n_steps):
            F, _, _ = solver(v=v, z=z, **kwargs)
            v += (F / MASS) * dt
            z += v * dt
            if abs(z) > z_escape:
                trapped = False
                break
        if trapped:
            v_capture = v0
        else:
            break

    return v_capture


# ======================================================================
#  Helper: find equilibrium position
# ======================================================================

def find_equilibrium_position(
    solver:     Callable,
    z_range:    Tuple[float, float] = (-10e-3, 10e-3),
    n_scan:     int = 100,
    tol:        float = 1e-6,
    **kwargs,
) -> float:
    """
    Find the equilibrium position where F(z, v=0) = 0 with dF/dz < 0 (restoring).

    Uses a coarse scan followed by bisection refinement.

    Returns z_eq in metres, or 0.0 if no zero crossing found.
    """
    z_arr = np.linspace(z_range[0], z_range[1], n_scan)
    F_arr = np.array([solver(v=0.0, z=z, **kwargs)[0] for z in z_arr])

    # Look for sign changes (restoring: force should go from + to -)
    z_eq = 0.0
    for i in range(len(F_arr) - 1):
        if F_arr[i] > 0 and F_arr[i+1] < 0:
            # Restoring zero crossing found; refine by bisection
            z_lo, z_hi = z_arr[i], z_arr[i+1]
            for _ in range(50):
                z_mid = 0.5 * (z_lo + z_hi)
                F_mid, _, _ = solver(v=0.0, z=z_mid, **kwargs)
                if F_mid > 0:
                    z_lo = z_mid
                else:
                    z_hi = z_mid
                if abs(z_hi - z_lo) < tol * 1e-3:
                    break
            z_eq = 0.5 * (z_lo + z_hi)
            break

    return z_eq


# ======================================================================
#  Helper: compute MOT characteristics
# ======================================================================

def compute_mot_characteristics(
    solver:     Callable,
    B_gradient: float,
    dz:         float = 0.3e-3,
    dv:         float = 0.1,
    find_eq:    bool  = False,
    **kwargs,
) -> Dict:
    """
    Compute key MOT characteristics: spring constant, damping, trap frequency,
    Doppler temperature, scattering rate, and trap lifetime.

    If find_eq=True, first locates the equilibrium position z_eq where F=0,
    then evaluates all quantities around z_eq instead of z=0.

    Returns a dict of computed quantities.
    """
    z_eq = 0.0
    if find_eq:
        z_eq = find_equilibrium_position(solver, **kwargs)

    # Spring constant: k = -dF/dz at equilibrium
    Fp, _, Rp = solver(v=0.0, z=z_eq + dz, **kwargs)
    Fm, _, Rm = solver(v=0.0, z=z_eq - dz, **kwargs)
    k_spring = -(Fp - Fm) / (2.0 * dz)

    # Damping: beta = -(dF/dv) / m at equilibrium
    Fvp, _, _ = solver(v=+dv, z=z_eq, **kwargs)
    Fvm, _, _ = solver(v=-dv, z=z_eq, **kwargs)
    alpha = -(Fvp - Fvm) / (2.0 * dv)    # friction coefficient (N s/m)
    beta = alpha / MASS                    # damping rate (1/s)

    # Trap frequency
    omega_trap = np.sqrt(abs(k_spring) / MASS) / (2.0 * np.pi) if k_spring > 0 else 0.0

    # Scattering rate at equilibrium
    _, p_centre, R_centre = solver(v=0.0, z=z_eq, **kwargs)

    # Doppler temperature: T_D = hbar * Gamma / (2 * kB) for two-level;
    # for multilevel use T = hbar * R_scatter / (2 * kB * beta) if beta > 0
    T_doppler = hbar * GAMMA / (2.0 * kB)
    if beta > 0 and R_centre > 0:
        T_effective = hbar * R_centre / (2.0 * kB * (alpha / hbar / K_WAVE**2))
    else:
        T_effective = T_doppler

    # Trap lifetime from photon budget
    if R_centre > 0:
        tau_photon = PHOTON_BUDGET / (R_centre / (2.0 * np.pi))  # seconds
    else:
        tau_photon = np.inf

    return {
        "k_spring": k_spring,
        "alpha": alpha,
        "beta": beta,
        "omega_trap": omega_trap,
        "R_scatter": R_centre,
        "R_scatter_MHz": R_centre / (2.0 * np.pi * 1e6),
        "T_doppler_mK": T_doppler * 1e3,
        "T_effective_mK": T_effective * 1e3,
        "tau_photon_ms": tau_photon * 1e3,
        "populations": p_centre,
        "z_eq_mm": z_eq * 1e3,
    }


# ======================================================================
#  H. Main Analysis
# ======================================================================

def main():
    """Run the complete MOT simulation and analysis."""

    # ------------------------------------------------------------------
    #  1. Load molecular data
    # ------------------------------------------------------------------
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "julia_sim")
    out_dir  = os.path.dirname(os.path.abspath(__file__))

    mol = load_molecular_data(data_dir)

    # Force unit for normalisation
    F_unit = hbar * K_WAVE * GAMMA / 2.0   # hbar * k * Gamma / 2

    # ------------------------------------------------------------------
    #  2. RF MOT Simulation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RF MOT Simulation")
    print("=" * 70)

    rf_delta = -1.0   # typical detuning for RF MOT (Gamma)
    rf_s0 = 1.0       # saturation parameter
    rf_B = 16.0        # G/cm (RMS for RF, but we use it as the effective gradient)

    print(f"\n  Parameters:")
    print(f"    Delta     = {rf_delta:.1f} Gamma = {rf_delta * GAMMA_HZ/1e6:.1f} MHz")
    print(f"    s0        = {rf_s0}")
    print(f"    B'        = {rf_B} G/cm")

    # Create a solver closure for the RF MOT
    def rf_solver(v, z, **kw):
        return rf_mot_force(mol, v, z, rf_B,
                           delta_gamma=rf_delta, s0=rf_s0)

    # Force vs position
    print("\n  Computing F(z) at v=0...")
    z_arr_rf, Fz_rf, pop_rf, Rz_rf = force_vs_z(
        rf_solver, z_range=(-5e-3, 5e-3), n_points=100, v=0.0
    )

    # Force vs velocity
    print("  Computing F(v) at z=0...")
    v_arr_rf, Fv_rf = force_vs_v(
        rf_solver, v_range=(-6.0, 6.0), n_points=100, z=0.0
    )

    # MOT characteristics
    rf_chars = compute_mot_characteristics(rf_solver, rf_B)

    print(f"\n  RF MOT Characteristics:")
    print(f"    Spring constant k = {rf_chars['k_spring']:.3e} N/m")
    print(f"    Trap frequency    = 2*pi * {rf_chars['omega_trap']:.1f} Hz")
    print(f"    Damping beta      = {rf_chars['beta']:.0f} /s")
    print(f"    Scattering rate   = {rf_chars['R_scatter_MHz']:.2f} MHz")
    print(f"    T_Doppler         = {rf_chars['T_doppler_mK']:.2f} mK")
    print(f"    Photon lifetime   = {rf_chars['tau_photon_ms']:.1f} ms")

    # Comparison with experiment
    print(f"\n  Comparison with experiment:")
    print(f"    {'Quantity':25s}  {'Simulation':>12s}  {'Experiment':>12s}")
    print(f"    {'-'*25}  {'-'*12}  {'-'*12}")
    print(f"    {'Temperature (mK)':25s}  {rf_chars['T_doppler_mK']:12.2f}  {'~1.2':>12s}")
    print(f"    {'Damping beta (/s)':25s}  {rf_chars['beta']:12.0f}  {'~100':>12s}")
    print(f"    {'Trap freq (Hz)':25s}  {rf_chars['omega_trap']:12.1f}  {'~45':>12s}")

    # Capture velocity
    print("  Estimating capture velocity...")
    v_cap_rf = estimate_capture_velocity(rf_solver)
    print(f"    Capture velocity  = {v_cap_rf:.1f} m/s  (expt: ~10 m/s)")

    # RF MOT equilibrium and characteristics
    rf_chars_eq = compute_mot_characteristics(rf_solver, rf_B, find_eq=True)
    print(f"    Equilibrium pos   = {rf_chars_eq['z_eq_mm']:.2f} mm")

    # RF MOT detuning scan
    print("\n  Scanning RF MOT detuning...")
    rf_opt = optimise_rf_mot_parameters(
        mol, B_gradient=rf_B, s0=rf_s0,
        delta_range=(-3.0, -0.1), n_delta=30
    )

    # ------------------------------------------------------------------
    #  3. DC MOT Simulation (4-frequency configuration)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DC MOT Simulation (4-Frequency Configuration)")
    print("=" * 70)

    dc_delta = -0.20   # overall detuning (Gamma)
    dc_split = 0.70    # polarisation split (Gamma)
    dc_s0 = 1.0        # saturation parameter
    dc_B = 16.0         # G/cm

    print(f"\n  Parameters:")
    print(f"    Delta     = {dc_delta:.2f} Gamma = {dc_delta * GAMMA_HZ/1e6:.2f} MHz")
    print(f"    split     = {dc_split:.2f} Gamma = {dc_split * GAMMA_HZ/1e6:.2f} MHz")
    print(f"    s0        = {dc_s0}")
    print(f"    B'        = {dc_B} G/cm")

    # Create a solver closure for the DC MOT
    def dc_solver(v, z, **kw):
        return dc_mot_force(mol, v, z, dc_B,
                           delta_gamma=dc_delta, split_gamma=dc_split, s0=dc_s0)

    # Force vs position (wider range to see equilibrium)
    print("\n  Computing F(z) at v=0...")
    z_arr_dc, Fz_dc, pop_dc, Rz_dc = force_vs_z(
        dc_solver, z_range=(-10e-3, 10e-3), n_points=200, v=0.0
    )

    # Find equilibrium and compute characteristics there
    dc_chars = compute_mot_characteristics(dc_solver, dc_B, find_eq=True)

    z_eq_dc = dc_chars["z_eq_mm"] * 1e-3
    print(f"\n  DC MOT Characteristics (at equilibrium z_eq = {dc_chars['z_eq_mm']:.2f} mm):")
    print(f"    Spring constant k = {dc_chars['k_spring']:.3e} N/m")
    print(f"    Trap frequency    = 2*pi * {dc_chars['omega_trap']:.1f} Hz")
    print(f"    Damping beta      = {dc_chars['beta']:.0f} /s")
    print(f"    Scattering rate   = {dc_chars['R_scatter_MHz']:.2f} MHz")
    print(f"    T_Doppler         = {dc_chars['T_doppler_mK']:.2f} mK")
    print(f"    Photon lifetime   = {dc_chars['tau_photon_ms']:.1f} ms")

    # Force vs velocity at equilibrium
    print("  Computing F(v) at z=z_eq...")
    v_arr_dc, Fv_dc = force_vs_v(
        dc_solver, v_range=(-6.0, 6.0), n_points=200, z=z_eq_dc
    )
    _, Fv_dc_1mm = force_vs_v(
        dc_solver, v_range=(-6.0, 6.0), n_points=200, z=z_eq_dc + 1e-3
    )

    # Max force
    print(f"    Max |F|           = {np.max(np.abs(Fz_dc))/F_unit:.4f} hbar*k*Gamma/2")

    # DC MOT parameter optimisation
    dc_opt = optimise_dc_mot_parameters(
        mol, B_gradient=dc_B, s0=dc_s0,
        delta_range=(-3.0, -0.1), split_range=(0.05, 2.0),
        n_delta=25, n_split=25,
    )

    # Re-run with optimal parameters
    dc_delta_opt = dc_opt["delta_best"]
    dc_split_opt = dc_opt["split_best"]

    def dc_solver_opt(v, z, **kw):
        return dc_mot_force(mol, v, z, dc_B,
                           delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=dc_s0)

    print(f"\n  Re-running with optimal parameters:")
    print(f"    Delta = {dc_delta_opt:.2f} Gamma,  split = {dc_split_opt:.2f} Gamma")

    # Find equilibrium for optimal parameters
    dc_chars_opt = compute_mot_characteristics(dc_solver_opt, dc_B, find_eq=True)
    z_eq_opt = dc_chars_opt["z_eq_mm"] * 1e-3

    z_arr_opt, Fz_opt, pop_opt, Rz_opt = force_vs_z(
        dc_solver_opt, z_range=(-10e-3, 10e-3), n_points=200, v=0.0
    )
    v_arr_opt, Fv_opt = force_vs_v(
        dc_solver_opt, v_range=(-6.0, 6.0), n_points=200, z=z_eq_opt
    )

    print(f"    z_eq    = {dc_chars_opt['z_eq_mm']:.2f} mm")
    print(f"    k       = {dc_chars_opt['k_spring']:.3e} N/m")
    print(f"    omega   = 2*pi * {dc_chars_opt['omega_trap']:.1f} Hz")
    print(f"    beta    = {dc_chars_opt['beta']:.0f} /s")
    print(f"    R_sc    = {dc_chars_opt['R_scatter_MHz']:.2f} MHz")

    # Capture velocity (optimal) -- start near equilibrium
    print("  Estimating capture velocity (optimal DC)...")
    v_cap_dc = estimate_capture_velocity(
        dc_solver_opt, z_start=z_eq_opt + 3e-3,
    )
    print(f"    Capture velocity  = {v_cap_dc:.1f} m/s")

    # ------------------------------------------------------------------
    #  Trajectories (optimal DC MOT)
    # ------------------------------------------------------------------
    print("\n  Simulating trajectories (optimal DC MOT)...")
    print(f"    (equilibrium at z_eq = {z_eq_opt*1e3:.2f} mm)")
    traj_configs = [
        (z_eq_opt + 3e-3,  0.0, "z0=z_eq+3mm"),
        (z_eq_opt,        -2.0, "v0=-2m/s"),
        (z_eq_opt + 2e-3, -1.0, "mixed"),
        (z_eq_opt,        -5.0, "v0=-5m/s"),
    ]

    trajectories = []
    for z0, v0, label in traj_configs:
        t_arr, z_traj, v_traj = simulate_trajectory(
            dc_solver_opt, z0, v0,
            t_max=0.05, dt=1e-6,
        )
        trajectories.append((t_arr, z_traj, v_traj, label))
        print(f"    {label:16s}: final z = {z_traj[-1]*1e3:+.2f} mm, v = {v_traj[-1]:+.2f} m/s")

    # ------------------------------------------------------------------
    #  Power budget
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Experimental Parameters Summary")
    print("=" * 70)

    beam_w = 10e-3   # beam 1/e^2 radius (m)
    P_per_comp = dc_s0 * I_SAT * np.pi * beam_w**2 / 2.0   # W (Gaussian beam)
    P_per_arm = 4 * P_per_comp   # 4 frequency components
    P_total = 3 * P_per_arm       # 3 retro-reflected axes

    print(f"\n  B gradient:        {dc_B:.0f} G/cm")
    print(f"  Beam 1/e2 diam:    {beam_w*2e3:.0f} mm")
    print(f"  I_sat:             {I_SAT/10:.1f} mW/cm^2")
    print(f"  Power/component:   {P_per_comp*1e3:.1f} mW")
    print(f"  Power/arm:         {P_per_arm*1e3:.1f} mW (4 components)")
    print(f"  Total power:       {P_total*1e3:.0f} mW (3 retro axes)")
    print(f"")
    print(f"  DC MOT (optimal):")
    print(f"    Detuning Delta:    {dc_delta_opt:.2f} Gamma = {dc_delta_opt*GAMMA_HZ/1e6:.2f} MHz")
    print(f"    Pol split delta:   {dc_split_opt:.2f} Gamma = {dc_split_opt*GAMMA_HZ/1e6:.2f} MHz")
    print(f"    Equil. position:   {dc_chars_opt['z_eq_mm']:.2f} mm")
    print(f"    Spring constant:   {dc_chars_opt['k_spring']:.2e} N/m")
    print(f"    Trap frequency:    2*pi * {dc_chars_opt['omega_trap']:.0f} Hz")
    print(f"    Damping rate:      {dc_chars_opt['beta']:.0f} /s")
    print(f"    Capture velocity:  {v_cap_dc:.1f} m/s")
    print(f"")
    print(f"  RF MOT (for comparison):")
    print(f"    Detuning Delta:    {rf_delta:.1f} Gamma")
    print(f"    Equil. position:   {rf_chars_eq['z_eq_mm']:.2f} mm")
    print(f"    Spring constant:   {rf_chars_eq['k_spring']:.2e} N/m")
    print(f"    Trap frequency:    2*pi * {rf_chars_eq['omega_trap']:.0f} Hz")
    print(f"    Damping rate:      {rf_chars_eq['beta']:.0f} /s")
    print(f"    Capture velocity:  {v_cap_rf:.1f} m/s")

    # ------------------------------------------------------------------
    #  4. Generate comprehensive plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Generating plots...")
    print("=" * 70)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    # ---- Plot 1: DC MOT Force Profiles (initial parameters) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
    fig1.suptitle(
        f"SrOH 4-Frequency DC MOT "
        f"(Delta={dc_delta:.2f}Gamma, split={dc_split:.2f}Gamma, "
        f"s0={dc_s0}, B'={dc_B}G/cm)",
        fontsize=13, y=0.98
    )

    # F(z)
    ax = axes1[0, 0]
    ax.plot(z_arr_dc * 1e3, Fz_dc / F_unit, "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Restoring Force F(z) at v=0")
    ax.grid(True, alpha=0.3)

    # F(v)
    ax = axes1[0, 1]
    ax.plot(v_arr_dc, Fv_dc / F_unit, "b-", lw=2, label="z=0")
    ax.plot(v_arr_dc, Fv_dc_1mm / F_unit, "r--", lw=2, label="z=1mm")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Damping Force F(v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Populations
    ax = axes1[1, 0]
    for ig in range(N_GROUND):
        ls = "-" if ig < 4 else ("--" if ig < 7 else ":")
        ax.plot(z_arr_dc * 1e3, pop_dc[:, ig], lw=1.2, ls=ls,
                label=f"g{ig+1}" if ig < 6 else None)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Population")
    ax.set_title("Ground State Populations")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scattering rate
    ax = axes1[1, 1]
    ax.plot(z_arr_dc * 1e3, Rz_dc / (2.0 * np.pi * 1e6), "g-", lw=2)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Scattering rate (MHz)")
    ax.set_title("Photon Scattering Rate")
    ax.grid(True, alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(os.path.join(out_dir, "sroh_dc_mot_forces.png"), dpi=150)
    print(f"  Saved: sroh_dc_mot_forces.png")

    # ---- Plot 2: DC MOT Optimisation ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
    fig2.suptitle("DC MOT Parameter Optimisation (s0={}, B'={} G/cm)".format(
        dc_s0, dc_B), fontsize=13, y=1.02)

    delta_scan = dc_opt["delta_scan"]
    split_scan = dc_opt["split_scan"]
    K_map = dc_opt["K_map"]
    beta_map = dc_opt["beta_map"]

    # Spring constant
    ax = axes2[0]
    K_plot = np.maximum(K_map, 0)
    im0 = ax.pcolormesh(split_scan, delta_scan, K_plot, cmap="hot", shading="auto")
    ax.plot(dc_split_opt, dc_delta_opt, "c*", ms=12)
    ax.set_xlabel("Split (Gamma)")
    ax.set_ylabel("Detuning Delta (Gamma)")
    ax.set_title("Spring Constant k (N/m)")
    fig2.colorbar(im0, ax=ax, pad=0.02)

    # Damping
    ax = axes2[1]
    bmax = max(abs(np.nanmin(beta_map)), abs(np.nanmax(beta_map)))
    if bmax == 0:
        bmax = 1.0
    im1 = ax.pcolormesh(split_scan, delta_scan, beta_map, cmap="RdBu",
                        vmin=-bmax, vmax=bmax, shading="auto")
    ax.plot(dc_split_opt, dc_delta_opt, "k*", ms=12)
    ax.set_xlabel("Split (Gamma)")
    ax.set_ylabel("Detuning Delta (Gamma)")
    ax.set_title("Damping beta (1/s)")
    fig2.colorbar(im1, ax=ax, pad=0.02)

    # Trap frequency
    omega_map = np.where(K_map > 0, np.sqrt(np.maximum(K_map, 0) / MASS) / (2.0 * np.pi), 0.0)
    ax = axes2[2]
    im2 = ax.pcolormesh(split_scan, delta_scan, omega_map, cmap="viridis", shading="auto")
    ax.plot(dc_split_opt, dc_delta_opt, "r*", ms=12)
    ax.set_xlabel("Split (Gamma)")
    ax.set_ylabel("Detuning Delta (Gamma)")
    ax.set_title("Trap Frequency omega/(2*pi) (Hz)")
    fig2.colorbar(im2, ax=ax, pad=0.02)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "sroh_dc_mot_optimisation.png"),
                 dpi=150, bbox_inches="tight")
    print(f"  Saved: sroh_dc_mot_optimisation.png")

    # ---- Plot 3: DC MOT Trajectories ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    fig3.suptitle(
        f"DC MOT Trajectories "
        f"(Delta={dc_delta_opt:.2f}Gamma, split={dc_split_opt:.2f}Gamma)",
        fontsize=13
    )

    colors = ["C0", "C1", "C2", "C3"]
    for idx, (tt, zt, vt, lbl) in enumerate(trajectories):
        axes3[0].plot(tt * 1e3, zt * 1e3, lw=1.5, color=colors[idx], label=lbl)
        axes3[1].plot(zt * 1e3, vt, lw=1.5, color=colors[idx], label=lbl)

    axes3[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes3[0].set_xlabel("t (ms)")
    axes3[0].set_ylabel("z (mm)")
    axes3[0].set_title("Position vs Time")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    axes3[1].set_xlabel("z (mm)")
    axes3[1].set_ylabel("v (m/s)")
    axes3[1].set_title("Phase Space")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "sroh_dc_mot_trajectories.png"), dpi=150)
    print(f"  Saved: sroh_dc_mot_trajectories.png")

    # ---- Plot 4: RF vs DC MOT Comparison ----
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))
    fig4.suptitle("RF MOT vs DC MOT Force Comparison", fontsize=13)

    ax = axes4[0]
    ax.plot(z_arr_rf * 1e3, Fz_rf / F_unit, "b-", lw=2, label="RF MOT")
    ax.plot(z_arr_opt * 1e3, Fz_opt / F_unit, "r-", lw=2, label="DC MOT (opt)")
    ax.plot(z_arr_dc * 1e3, Fz_dc / F_unit, "r--", lw=1.5, alpha=0.6,
            label=f"DC MOT (D={dc_delta:.1f},s={dc_split:.1f})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Restoring Force F(z)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes4[1]
    ax.plot(v_arr_rf, Fv_rf / F_unit, "b-", lw=2, label="RF MOT")
    ax.plot(v_arr_opt, Fv_opt / F_unit, "r-", lw=2, label="DC MOT (opt)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Damping Force F(v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "sroh_rf_vs_dc_comparison.png"), dpi=150)
    print(f"  Saved: sroh_rf_vs_dc_comparison.png")

    # ---- Plot 5: RF MOT detuning scan ----
    fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4))
    fig5.suptitle("RF MOT Detuning Scan (s0={}, B'={} G/cm)".format(rf_s0, rf_B),
                  fontsize=13)

    ax = axes5[0]
    ax.plot(rf_opt["delta_scan"], rf_opt["K_arr"], "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(rf_opt["delta_best"], color="r", ls=":", lw=1.5, label="optimum")
    ax.set_xlabel("Detuning Delta (Gamma)")
    ax.set_ylabel("Spring constant k (N/m)")
    ax.set_title("Spring Constant")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes5[1]
    ax.plot(rf_opt["delta_scan"], rf_opt["beta_arr"], "b-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(rf_opt["delta_best"], color="r", ls=":", lw=1.5, label="optimum")
    ax.set_xlabel("Detuning Delta (Gamma)")
    ax.set_ylabel("Damping beta (1/s)")
    ax.set_title("Damping Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes5[2]
    omega_rf = np.where(rf_opt["K_arr"] > 0,
                        np.sqrt(np.maximum(rf_opt["K_arr"], 0) / MASS) / (2 * np.pi), 0)
    ax.plot(rf_opt["delta_scan"], omega_rf, "b-", lw=2)
    ax.axvline(rf_opt["delta_best"], color="r", ls=":", lw=1.5, label="optimum")
    ax.set_xlabel("Detuning Delta (Gamma)")
    ax.set_ylabel("Trap freq omega/(2*pi) (Hz)")
    ax.set_title("Trap Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5.savefig(os.path.join(out_dir, "sroh_rf_mot_scan.png"), dpi=150)
    print(f"  Saved: sroh_rf_mot_scan.png")

    # ---- Plot 6: Optimal DC MOT forces ----
    fig6, axes6 = plt.subplots(2, 2, figsize=(12, 9))
    fig6.suptitle(
        f"SrOH DC MOT (Optimal: "
        f"Delta={dc_delta_opt:.2f}Gamma, split={dc_split_opt:.2f}Gamma)",
        fontsize=13, y=0.98
    )

    ax = axes6[0, 0]
    ax.plot(z_arr_opt * 1e3, Fz_opt / F_unit, "r-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Restoring Force F(z) at v=0")
    ax.grid(True, alpha=0.3)

    ax = axes6[0, 1]
    ax.plot(v_arr_opt, Fv_opt / F_unit, "r-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("F / (hbar k Gamma/2)")
    ax.set_title("Damping Force F(v) at z=0")
    ax.grid(True, alpha=0.3)

    ax = axes6[1, 0]
    for ig in range(N_GROUND):
        ls = "-" if ig < 4 else ("--" if ig < 7 else ":")
        ax.plot(z_arr_opt * 1e3, pop_opt[:, ig], lw=1.2, ls=ls,
                label=f"g{ig+1}" if ig < 6 else None)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Population")
    ax.set_title("Ground State Populations")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes6[1, 1]
    ax.plot(z_arr_opt * 1e3, Rz_opt / (2.0 * np.pi * 1e6), "g-", lw=2)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Scattering rate (MHz)")
    ax.set_title("Photon Scattering Rate")
    ax.grid(True, alpha=0.3)

    fig6.tight_layout(rect=[0, 0, 1, 0.96])
    fig6.savefig(os.path.join(out_dir, "sroh_dc_mot_optimal_forces.png"), dpi=150)
    print(f"  Saved: sroh_dc_mot_optimal_forces.png")

    # ---- Plot 7: Saturation parameter scan ----
    print("\n  Running saturation parameter scan...")
    s0_scan = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    k_vs_s0 = np.zeros(len(s0_scan))
    beta_vs_s0 = np.zeros(len(s0_scan))
    omega_vs_s0 = np.zeros(len(s0_scan))
    dz_s = 0.3e-3
    dv_s = 0.1

    for i_s, s in enumerate(s0_scan):
        Fp, _, _ = dc_mot_force(mol, 0.0, +dz_s, dc_B,
                                delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=s)
        Fm, _, _ = dc_mot_force(mol, 0.0, -dz_s, dc_B,
                                delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=s)
        k_vs_s0[i_s] = -(Fp - Fm) / (2 * dz_s)

        Fvp, _, _ = dc_mot_force(mol, +dv_s, 0.0, dc_B,
                                 delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=s)
        Fvm, _, _ = dc_mot_force(mol, -dv_s, 0.0, dc_B,
                                 delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=s)
        beta_vs_s0[i_s] = -(Fvp - Fvm) / (2 * dv_s * MASS)
        omega_vs_s0[i_s] = (np.sqrt(k_vs_s0[i_s] / MASS) / (2 * np.pi)
                            if k_vs_s0[i_s] > 0 else 0.0)

    fig7, axes7 = plt.subplots(1, 3, figsize=(14, 4))
    fig7.suptitle("DC MOT: Saturation Parameter Dependence", fontsize=13)

    axes7[0].semilogx(s0_scan, k_vs_s0, "ro-", lw=2)
    axes7[0].set_xlabel("s0")
    axes7[0].set_ylabel("k (N/m)")
    axes7[0].set_title("Spring Constant")
    axes7[0].grid(True, alpha=0.3)

    axes7[1].semilogx(s0_scan, beta_vs_s0, "bo-", lw=2)
    axes7[1].set_xlabel("s0")
    axes7[1].set_ylabel("beta (1/s)")
    axes7[1].set_title("Damping Rate")
    axes7[1].grid(True, alpha=0.3)

    axes7[2].semilogx(s0_scan, omega_vs_s0, "go-", lw=2)
    axes7[2].set_xlabel("s0")
    axes7[2].set_ylabel("omega/(2*pi) (Hz)")
    axes7[2].set_title("Trap Frequency")
    axes7[2].grid(True, alpha=0.3)

    fig7.tight_layout()
    fig7.savefig(os.path.join(out_dir, "sroh_dc_mot_s0_scan.png"), dpi=150)
    print(f"  Saved: sroh_dc_mot_s0_scan.png")

    # ---- Plot 8: B-gradient scan ----
    print("  Running B-gradient scan...")
    B_scan = np.array([4, 8, 12, 16, 20, 25, 30, 40])
    k_vs_B = np.zeros(len(B_scan))
    beta_vs_B = np.zeros(len(B_scan))
    omega_vs_B = np.zeros(len(B_scan))

    for i_B, B in enumerate(B_scan):
        Fp, _, _ = dc_mot_force(mol, 0.0, +dz_s, B,
                                delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=dc_s0)
        Fm, _, _ = dc_mot_force(mol, 0.0, -dz_s, B,
                                delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=dc_s0)
        k_vs_B[i_B] = -(Fp - Fm) / (2 * dz_s)

        Fvp, _, _ = dc_mot_force(mol, +dv_s, 0.0, B,
                                 delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=dc_s0)
        Fvm, _, _ = dc_mot_force(mol, -dv_s, 0.0, B,
                                 delta_gamma=dc_delta_opt, split_gamma=dc_split_opt, s0=dc_s0)
        beta_vs_B[i_B] = -(Fvp - Fvm) / (2 * dv_s * MASS)
        omega_vs_B[i_B] = (np.sqrt(k_vs_B[i_B] / MASS) / (2 * np.pi)
                           if k_vs_B[i_B] > 0 else 0.0)

    fig8, axes8 = plt.subplots(1, 3, figsize=(14, 4))
    fig8.suptitle("DC MOT: B-Gradient Dependence", fontsize=13)

    axes8[0].plot(B_scan, k_vs_B, "ro-", lw=2)
    axes8[0].set_xlabel("B' (G/cm)")
    axes8[0].set_ylabel("k (N/m)")
    axes8[0].set_title("Spring Constant")
    axes8[0].grid(True, alpha=0.3)

    axes8[1].plot(B_scan, beta_vs_B, "bo-", lw=2)
    axes8[1].set_xlabel("B' (G/cm)")
    axes8[1].set_ylabel("beta (1/s)")
    axes8[1].set_title("Damping Rate")
    axes8[1].grid(True, alpha=0.3)

    axes8[2].plot(B_scan, omega_vs_B, "go-", lw=2)
    axes8[2].set_xlabel("B' (G/cm)")
    axes8[2].set_ylabel("omega/(2*pi) (Hz)")
    axes8[2].set_title("Trap Frequency")
    axes8[2].grid(True, alpha=0.3)

    fig8.tight_layout()
    fig8.savefig(os.path.join(out_dir, "sroh_dc_mot_Bgradient_scan.png"), dpi=150)
    print(f"  Saved: sroh_dc_mot_Bgradient_scan.png")

    plt.close("all")

    print("\n" + "=" * 70)
    print("  All simulations complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
