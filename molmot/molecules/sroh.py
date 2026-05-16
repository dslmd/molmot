"""
SrOH molecular data: Hamiltonian construction and data loading.

Builds the full 16-level SrOH molecular Hamiltonian from first
principles using spectroscopic constants, and provides a loader
for pre-computed data from the Julia simulation.

Transition: X~2Sigma+(000, N=1) -> A~2Pi_1/2(000, J'=1/2) at ~688 nm.

Level structure:
  - 12 ground states (X, N=1): Hund's case (b)
      4 states with J=1/2 (F=0,1) + 8 states with J=3/2 (F=1,2)
  - 4 excited states (A, J'=1/2): Hund's case (a) -> converted to case (b)
      F'=0 (1 state) + F'=1 (3 states)

Spectroscopic constants from:
  - Nguyen et al., JMS 297, 10 (2014) [ground state]
  - Kozyryev et al., NJP 17, 045003 (2015) [excited state]
  - Lasner et al., PRL 134, 083401 (2025) [MOT parameters]

References
----------
* Christian Hallas, OpticalBlochEquations.jl, SrOH_package.jl
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

from ..constants import (gS, c, h, hbar, mu_B, k_B, amu,
                         wavenumber, v_recoil, T_doppler, I_sat)
from ..states.basis import enumerate_states
from ..states.case_b import (HundsCaseB_LinearMolecule,
                              Rotation, RotationDistortion, SpinRotation,
                              Hyperfine_IS, Hyperfine_Dipolar,
                              Zeeman, zeeman_nuclear, TDM)
from ..states.case_a import (HundsCaseA_LinearMolecule,
                              Rotation as Rotation_a, SpinOrbit,
                              LambdaDoubling_p2q,
                              Zeeman_L, Zeeman_S,
                              TDM as TDM_a)
from ..states.hamiltonian import Hamiltonian, operator_to_matrix
from ..states.overlaps import overlap_caseb_casea, convert_basis
from ..states.tdm import compute_tdms


# =====================================================================
# Spectroscopic constants
# =====================================================================

# Ground state X~2Sigma+(000)
BX = 0.24920        # cm^-1, rotational constant
DX = 0.0            # cm^-1, centrifugal distortion (small, neglected)
gammaX = 0.00242748 # cm^-1, spin-rotation constant
bF_X = 1.713e6      # Hz, Fermi contact hyperfine
c_X = 1.673e6       # Hz, dipolar hyperfine

# Excited state A~2Pi_1/2(000)
T_A = 14542.573      # cm^-1, term energy
Be_A = 0.25389       # cm^-1, rotational constant
Aso = 263.51741       # cm^-1, spin-orbit coupling
p_A = -0.14320       # cm^-1, Lambda-doubling p
q_A = -0.00020       # cm^-1, Lambda-doubling q

# Transition properties
# Note: Christian's Julia package (SrOH_package.jl) uses 687 nm.
# Earlier literature sometimes cites 688 nm.  We use 687 nm to match
# the Julia code that generated the CSV data files.
LAMBDA_NM = 687.0                   # nm
LAMBDA_M = LAMBDA_NM * 1e-9        # m
GAMMA_HZ = 6.4e6                    # Hz (natural linewidth / 2*pi)
GAMMA_RAD = 2.0 * math.pi * GAMMA_HZ  # rad/s
MASS_AMU = 105.0                    # amu
MASS_KG = MASS_AMU * amu            # kg
K_WAVE = wavenumber(LAMBDA_M)       # 1/m

# Conversion factors
CM_TO_HZ = c * 100.0   # 1 cm^-1 = 2.998e10 Hz


@dataclass
class MolecularData:
    """
    Container for molecular Hamiltonian data used by the OBE solver.

    Attributes
    ----------
    energies : np.ndarray, shape (n_states,)
        State energies in Hz.
    tdm : np.ndarray, shape (n_states, n_states, 3)
        Complex transition dipole matrix elements.
    zeeman_x : np.ndarray, shape (n_states, n_states)
        Zeeman matrix for Bx, pre-scaled by 2*pi*gS*mu_B/(Gamma*h*1e4).
        Usage: E_zeeman(Hz) = B(Gauss) * diag(zeeman_z) * Gamma/(2*pi).
    zeeman_y : np.ndarray, shape (n_states, n_states)
        Zeeman matrix for By (same scaling as zeeman_x).
    zeeman_z : np.ndarray, shape (n_states, n_states)
        Zeeman matrix for Bz (same scaling as zeeman_x).
    Gamma : float
        Natural linewidth (rad/s).
    k : float
        Wavenumber (1/m).
    mass : float
        Molecular mass (kg).
    wavelength : float
        Transition wavelength (m).
    n_ground : int
    n_excited : int
    n_states : int
    """
    energies: np.ndarray
    tdm: np.ndarray
    zeeman_x: np.ndarray
    zeeman_y: np.ndarray
    zeeman_z: np.ndarray
    Gamma: float = GAMMA_RAD
    k: float = K_WAVE
    mass: float = MASS_KG
    wavelength: float = LAMBDA_M
    n_ground: int = 12
    n_excited: int = 4
    n_states: int = 16

    # Derived quantities
    d_squared: np.ndarray = field(default=None, repr=False)
    zeeman_z_diag: np.ndarray = field(default=None, repr=False)
    omega_J12: float = 0.0
    omega_J32: float = 0.0
    omega_mean: float = 0.0

    def __post_init__(self):
        n_g = self.n_ground
        n_e = self.n_excited

        # |d|^2 for each (ground, excited, polarisation)
        if self.d_squared is None:
            self.d_squared = np.zeros((n_g, n_e, 3))
            for ig in range(n_g):
                for ie in range(n_e):
                    for q in range(3):
                        self.d_squared[ig, ie, q] = abs(
                            self.tdm[ig, n_g + ie, q]) ** 2

        # Diagonal Zeeman shifts
        if self.zeeman_z_diag is None:
            self.zeeman_z_diag = np.real(np.diag(self.zeeman_z))

        # Mean transition frequencies
        E_g = self.energies[:n_g]
        E_e = self.energies[n_g:]
        E_e_mean = np.mean(E_e)

        # J manifold ordering.
        # The Julia SrOH package orders ground states as:
        #   indices 0-7  (8 states): J=3/2  (N=1, S=1/2, F=1,2)
        #   indices 8-11 (4 states): J=1/2  (N=1, S=1/2, F=0,1)
        # The J=3/2 manifold has lower mean energy than J=1/2
        # (SR splitting ~+55 MHz for J=1/2 above J=3/2).
        if n_g == 12:
            E_J32_mean = np.mean(E_g[:8])   # first 8 = J=3/2
            E_J12_mean = np.mean(E_g[8:])   # last 4  = J=1/2
        else:
            E_J12_mean = np.mean(E_g)
            E_J32_mean = np.mean(E_g)

        self.omega_J12 = E_e_mean - E_J12_mean
        self.omega_J32 = E_e_mean - E_J32_mean
        self.omega_mean = E_e_mean - np.mean(E_g)


def build_sroh_hamiltonian(verbose: bool = True) -> MolecularData:
    """
    Build the full SrOH molecular Hamiltonian from spectroscopic constants.

    Ground state X~2Sigma+(000, N=1): Hund's case (b)
        12 states: J=1/2 (F=0, F=1) + J=3/2 (F=1, F=2)

    Excited state A~2Pi_1/2(000, J'=1/2): Hund's case (a)
        4 states: F'=0 + F'=1 (selecting +parity / J'=1/2 only)

    Note
    ----
    For production simulations, use ``load_sroh_from_julia()`` with the
    pre-computed CSV files from the validated Julia code.  This from-scratch
    builder correctly produces the ground state energy structure and Zeeman
    operators, but the excited-state case (a) -> case (b) TDM conversion
    requires careful tensor coupling that matches the Julia implementation
    exactly.  The ground-state structure (12 states with correct hyperfine
    and spin-rotation splittings) is validated.

    Returns
    -------
    MolecularData
        Complete molecular data ready for OBE simulation.
    """
    if verbose:
        print("=" * 60)
        print("  Building SrOH Hamiltonian from spectroscopic constants")
        print("=" * 60)

    # ==================================================================
    # Step 1: Ground state (case b)
    # ==================================================================

    if verbose:
        print("\n  Ground state: X~2Sigma+(000, N=1)")

    # Enumerate ground basis states
    ground_basis = enumerate_states(
        HundsCaseB_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0,
            "S": 0.5, "I": 0.5,
            "Lambda": 0, "ell": 0,
            "N": [1],  # N=1 only
        }
    )

    if verbose:
        print(f"    {len(ground_basis)} basis states enumerated")
        for i, s in enumerate(ground_basis):
            print(f"      {i}: {s}")

    # Build ground Hamiltonian
    H_ground = Hamiltonian(ground_basis)
    H_ground.add_operator("B", BX * CM_TO_HZ, Rotation)
    H_ground.add_operator("D", DX * CM_TO_HZ, RotationDistortion)
    H_ground.add_operator("gamma", gammaX * CM_TO_HZ, SpinRotation)
    H_ground.add_operator("bF", bF_X, Hyperfine_IS)
    # Dipolar hyperfine (small correction)
    # H_ground.add_operator("c/3", c_X / 3.0, Hyperfine_Dipolar)

    H_ground.evaluate()
    ground_states = H_ground.solve()

    if verbose:
        print(f"    Diagonalised: {len(ground_states)} eigenstates")
        E_g_mean = np.mean([s.E for s in ground_states])
        for i, s in enumerate(ground_states):
            dom, coeff = s.dominant_state()
            print(f"      g{i+1}: E = {(s.E - E_g_mean)/1e6:+8.3f} MHz  "
                  f"(dominant: {dom})")

    # ==================================================================
    # Step 2: Excited state (case a)
    # ==================================================================

    if verbose:
        print(f"\n  Excited state: A~2Pi_1/2(000, J'=1/2)")

    # Enumerate case (a) basis for the A state
    # For Pi_1/2: Lambda=+/-1, Sigma=+/-1/2, select P=+/-1/2
    excited_basis_a = enumerate_states(
        HundsCaseA_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0, "ell": 0,
            "Lambda": [-1, 1],
            "I": 0.5,
            "S": 0.5,
            "J": [0.5],  # J'=1/2 only
        }
    )

    # Filter to keep only states with |P| = 1/2 (the Pi_1/2 component)
    excited_basis_a = [s for s in excited_basis_a
                       if abs(Fraction(s.P).limit_denominator(10)) == Fraction(1, 2)]

    if verbose:
        print(f"    {len(excited_basis_a)} case (a) basis states")
        for i, s in enumerate(excited_basis_a):
            print(f"      {i}: {s}")

    # Build excited Hamiltonian in case (a) basis
    H_excited = Hamiltonian(excited_basis_a)
    H_excited.add_operator("Aso", Aso * CM_TO_HZ, SpinOrbit)
    H_excited.add_operator("Be", Be_A * CM_TO_HZ, Rotation_a)
    H_excited.add_operator("p+2q", (p_A + 2 * q_A) * CM_TO_HZ, LambdaDoubling_p2q)

    H_excited.evaluate()

    # Add the term energy offset to the diagonal
    for i in range(len(excited_basis_a)):
        H_excited.matrix[i, i] += T_A * CM_TO_HZ

    excited_states_all = H_excited.solve()

    if verbose:
        print(f"    Diagonalised: {len(excited_states_all)} eigenstates")
        E_e_mean = np.mean([s.E for s in excited_states_all])
        for i, s in enumerate(excited_states_all):
            print(f"      e{i+1}: E = {(s.E - E_e_mean)/1e6:+8.4f} MHz")

    # Select the correct parity doublet (4 states out of 8).
    # The ground X²Σ⁺(N=1) has parity (-1)^N = -1 (negative).
    # Electric dipole transitions connect + ↔ - parity, so
    # the accessible A²Π₁/₂(J'=1/2) state has POSITIVE parity.
    # This is the UPPER Lambda-doublet (higher energy).
    excited_states_all.sort(key=lambda s: s.E)
    excited_states = excited_states_all[4:]  # upper doublet = positive parity

    if verbose:
        print(f"    Selected {len(excited_states)} states (lower parity doublet)")
        E_e_mean = np.mean([s.E for s in excited_states])
        for i, s in enumerate(excited_states):
            print(f"      e{i+1}: E = {(s.E - E_e_mean)/1e6:+8.4f} MHz")

    # ==================================================================
    # Step 3: Convert excited states to case (b) for TDM computation
    # ==================================================================

    if verbose:
        print(f"\n  Converting excited states to case (b) basis")

    # Enumerate a case (b) basis with matching quantum numbers
    excited_basis_b = enumerate_states(
        HundsCaseB_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0,
            "S": 0.5, "I": 0.5,
            "Lambda": [-1, 0, 1],
            "ell": 0,
            "N": [0, 1],  # N=0 and N=1 to span J'=1/2
        }
    )

    # Filter: J=1/2 only, AND enforce N >= |K| (constraint not auto-applied)
    excited_basis_b = [s for s in excited_basis_b
                       if abs(s.J - 0.5) < 0.01 and s.N >= abs(s.K)]

    if verbose:
        print(f"    {len(excited_basis_b)} case (b) basis states for excited manifold")

    # Compute overlap matrix
    n_e_a = len(excited_basis_a)
    n_e_b = len(excited_basis_b)
    overlap_matrix = np.zeros((n_e_b, n_e_a), dtype=complex)
    for i, sb in enumerate(excited_basis_b):
        for j, sa in enumerate(excited_basis_a):
            overlap_matrix[i, j] = overlap_caseb_casea(sb, sa)

    # Convert eigenstate coefficients to case (b)
    # excited_coeffs_a[k, j] = coefficient of eigenstate k in case (a) basis j
    excited_coeffs_a = np.array([s.coeffs for s in excited_states])
    # excited_coeffs_b[k, i] = coefficient of eigenstate k in case (b) basis i
    excited_coeffs_b = excited_coeffs_a @ overlap_matrix.conj().T

    if verbose:
        print(f"    Overlap matrix: {overlap_matrix.shape}")
        # Check unitarity
        check = overlap_matrix @ overlap_matrix.conj().T
        print(f"    Unitarity check (should be ~identity): "
              f"max off-diag = {np.max(np.abs(check - np.eye(n_e_b))):.2e}")

    # ==================================================================
    # Step 4: Compute TDMs
    # ==================================================================

    if verbose:
        print(f"\n  Computing transition dipole moments")

    n_g = len(ground_states)
    n_e = len(excited_states)
    n_total = n_g + n_e

    ground_coeffs = np.array([s.coeffs for s in ground_states])

    d = compute_tdms(
        ground_states, excited_states,
        ground_basis, excited_basis_b,
        TDM,
        ground_coeffs=ground_coeffs,
        excited_coeffs=excited_coeffs_b)

    if verbose:
        n_nonzero = np.sum(np.abs(d) > 1e-6)
        print(f"    TDM array shape: {d.shape}, nonzero elements: {n_nonzero}")

    # ==================================================================
    # Step 5: Compute Zeeman matrices
    # ==================================================================

    if verbose:
        print(f"\n  Computing Zeeman matrices")

    # Ground state Zeeman (case b)
    Z_ground = {}
    for p, label in [(-1, "x"), (0, "z"), (1, "y")]:
        mat = operator_to_matrix(ground_basis, Zeeman, p)
        # Transform to eigenstate basis
        Z_eigen = ground_coeffs.conj() @ mat @ ground_coeffs.T
        Z_ground[label] = Z_eigen

    # Excited state Zeeman (case a -> case b conversion)
    # For Pi_1/2 with J'=1/2, the g-factor is approximately zero
    # (g_J ~ g_L + g_S * [J(J+1)+S(S+1)-L(L+1)]/[2J(J+1)] ~ 0 for Omega=1/2)
    Z_excited = {}
    for label in ["x", "y", "z"]:
        Z_excited[label] = np.zeros((n_e, n_e), dtype=complex)

    # Assemble full Zeeman matrices
    Zeeman_x = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_y = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_z = np.zeros((n_total, n_total), dtype=complex)

    # The Zeeman operator for p=0 gives Bz coupling
    # For p=+/-1, we need to construct Bx and By from the spherical components:
    #   Bx ~ (T^1_{-1} - T^1_{+1}) / sqrt(2)
    #   By ~ i*(T^1_{-1} + T^1_{+1}) / sqrt(2)

    # p=0 -> Bz
    mat_g_0 = operator_to_matrix(ground_basis, Zeeman, 0)
    Zg_0 = ground_coeffs.conj() @ mat_g_0 @ ground_coeffs.T
    Zeeman_z[:n_g, :n_g] = Zg_0

    # p=-1 and p=+1 for Bx and By
    mat_g_m1 = operator_to_matrix(ground_basis, Zeeman, -1)
    mat_g_p1 = operator_to_matrix(ground_basis, Zeeman, 1)
    Zg_m1 = ground_coeffs.conj() @ mat_g_m1 @ ground_coeffs.T
    Zg_p1 = ground_coeffs.conj() @ mat_g_p1 @ ground_coeffs.T

    # Bx = (B_{-1} - B_{+1}) / sqrt(2),  By = i*(B_{-1} + B_{+1}) / sqrt(2)
    Zeeman_x[:n_g, :n_g] = (Zg_m1 - Zg_p1) / math.sqrt(2)
    Zeeman_y[:n_g, :n_g] = 1j * (Zg_m1 + Zg_p1) / math.sqrt(2)

    if verbose:
        print(f"    Zeeman matrices assembled: {Zeeman_z.shape}")

    # ==================================================================
    # Step 6: Assemble energies
    # ==================================================================

    energies = np.zeros(n_total)
    for i, s in enumerate(ground_states):
        energies[i] = s.E
    for i, s in enumerate(excited_states):
        energies[n_g + i] = s.E

    if verbose:
        print(f"\n  Energy summary:")
        E_g_mean = np.mean(energies[:n_g])
        E_e_mean = np.mean(energies[n_g:])
        print(f"    Ground mean: {E_g_mean/1e6:.2f} MHz")
        print(f"    Excited mean: {E_e_mean/1e6:.2f} MHz")
        print(f"    Transition: {(E_e_mean - E_g_mean)/1e12:.6f} THz "
              f"= {(E_e_mean - E_g_mean)/CM_TO_HZ:.3f} cm^-1")
        SR_split = np.mean(energies[4:n_g]) - np.mean(energies[:4])
        print(f"    SR splitting: {SR_split/1e6:.1f} MHz")

    # ==================================================================
    # Step 7: Create MolecularData
    # ==================================================================

    mol = MolecularData(
        energies=energies,
        tdm=d,
        zeeman_x=Zeeman_x,
        zeeman_y=Zeeman_y,
        zeeman_z=Zeeman_z,
        Gamma=GAMMA_RAD,
        k=K_WAVE,
        mass=MASS_KG,
        wavelength=LAMBDA_M,
        n_ground=n_g,
        n_excited=n_e,
        n_states=n_total,
    )

    if verbose:
        print(f"\n  MolecularData created successfully")
        print(f"    {n_g} ground + {n_e} excited = {n_total} states")
        print(f"    lambda = {LAMBDA_NM:.1f} nm")
        print(f"    Gamma/(2pi) = {GAMMA_HZ/1e6:.1f} MHz")
        print(f"    mass = {MASS_AMU:.0f} amu")
        print(f"    k = {K_WAVE:.4e} /m")
        print("=" * 60)

    return mol


def load_sroh_from_julia(data_dir: str,
                         verbose: bool = True) -> MolecularData:
    """
    Load pre-computed SrOH data from Julia CSV files for validation.

    Expected files:
        sroh_energies.csv            -- 16 energies, one per line (Hz)
        sroh_tdm_re.csv              -- 48 x 16 tab-separated (real part)
        sroh_tdm_im.csv              -- 48 x 16 tab-separated (imag part)
        sroh_zeeman_{x,y,z}_re.csv   -- 16 x 16 tab-separated
        sroh_zeeman_{x,y,z}_im.csv   -- 16 x 16 tab-separated

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files.
    verbose : bool
        Print loading information.

    Returns
    -------
    MolecularData
    """
    N_STATES = 16
    N_GROUND = 12
    N_EXCITED = 4

    if verbose:
        print("=" * 60)
        print("  Loading SrOH molecular data from Julia CSV files")
        print("=" * 60)

    def _path(name):
        return os.path.join(data_dir, name)

    # Energies
    energies = np.loadtxt(_path("sroh_energies.csv"))
    assert energies.shape == (N_STATES,), \
        f"Expected {N_STATES} energies, got {energies.shape}"
    if verbose:
        print(f"  Loaded {N_STATES} state energies")

    # Transition dipole matrix
    tdm_re = np.loadtxt(_path("sroh_tdm_re.csv"))
    tdm_im = np.loadtxt(_path("sroh_tdm_im.csv"))
    assert tdm_re.shape == (48, 16), f"TDM shape mismatch: {tdm_re.shape}"

    tdm_flat = tdm_re + 1j * tdm_im
    tdm = np.zeros((N_STATES, N_STATES, 3), dtype=complex)
    for state_i in range(N_STATES):
        for q in range(3):
            row_idx = state_i * 3 + q
            tdm[state_i, :, q] = tdm_flat[row_idx, :]
    if verbose:
        print(f"  Loaded TDM: {tdm.shape}")

    # Zeeman matrices
    def _load_zeeman(axis):
        re = np.loadtxt(_path(f"sroh_zeeman_{axis}_re.csv"))
        im = np.loadtxt(_path(f"sroh_zeeman_{axis}_im.csv"))
        assert re.shape == (N_STATES, N_STATES), \
            f"Zeeman {axis} shape: {re.shape}"
        return re + 1j * im

    zeeman_x = _load_zeeman("x")
    zeeman_y = _load_zeeman("y")
    zeeman_z = _load_zeeman("z")
    if verbose:
        print(f"  Loaded Zeeman matrices: {zeeman_z.shape}")

    mol = MolecularData(
        energies=energies,
        tdm=tdm,
        zeeman_x=zeeman_x,
        zeeman_y=zeeman_y,
        zeeman_z=zeeman_z,
        Gamma=GAMMA_RAD,
        k=K_WAVE,
        mass=MASS_KG,
        wavelength=LAMBDA_M,
        n_ground=N_GROUND,
        n_excited=N_EXCITED,
        n_states=N_STATES,
    )

    if verbose:
        E_g_mean = np.mean(mol.energies[:N_GROUND])
        E_e_mean = np.mean(mol.energies[N_GROUND:])
        print(f"\n  lambda   = {LAMBDA_NM:.1f} nm")
        print(f"  Gamma    = 2*pi * {GAMMA_HZ/1e6:.1f} MHz")
        print(f"  mass     = {MASS_AMU:.0f} amu")
        print(f"  k        = {K_WAVE:.4e} /m")
        print(f"  SR split = {(np.mean(mol.energies[4:N_GROUND]) - np.mean(mol.energies[:4]))/1e6:.1f} MHz")

        # Print nonzero TDM elements
        pol_labels = ["sigma-", "pi", "sigma+"]
        n_printed = 0
        for ig in range(N_GROUND):
            for ie in range(N_EXCITED):
                for q in range(3):
                    d2 = mol.d_squared[ig, ie, q]
                    if d2 > 1e-4:
                        n_printed += 1
        print(f"  Nonzero |d|^2 > 1e-4: {n_printed} transitions")
        print("=" * 60)

    return mol
