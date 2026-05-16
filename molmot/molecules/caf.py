"""
CaF molecular data: Hamiltonian construction and SrOH-structure-based loader.

Builds the full 16-level CaF molecular Hamiltonian from first
principles using spectroscopic constants, or loads it by rescaling
the validated SrOH data (preferred for benchmarking).

Transition: X~2Sigma+(v=0, N=1) -> A~2Pi_1/2(v=0, J'=1/2) at ~606 nm.

Level structure (same as SrOH/CaOH):
  - 12 ground states (X, N=1): Hund's case (b)
      4 states with J=1/2 (F=0,1) + 8 states with J=3/2 (F=1,2)
  - 4 excited states (A, J'=1/2): Hund's case (a) -> converted to case (b)
      F'=0 (1 state) + F'=1 (3 states)

Spectroscopic constants from Christian's CaOHBlueMOTSimulations/CaF/ files.

References
----------
* Christian Hallas, CaOHBlueMOTSimulations
* Anderegg et al., PRL 119, 103201 (2017) [CaF MOT]
* Li et al., PRL 131, 013001 (2023) [CaF blue MOT]
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

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
from .sroh import MolecularData


# =====================================================================
# Spectroscopic constants for CaF
# =====================================================================

# Ground state X~2Sigma+(v=0)
# CaF constants in MHz
BX_MHZ = 10303.988       # MHz, rotational constant
DX_MHZ = 0.014060        # MHz, centrifugal distortion
gammaX_MHZ = 39.65891    # MHz, spin-rotation constant
bF_X_MHZ = 122.5569      # MHz, Fermi contact hyperfine (F-19 has I=1/2)
c_X_MHZ = 40.1190        # MHz, dipolar hyperfine

# Convert to Hz for Hamiltonian
BX_HZ = BX_MHZ * 1e6
DX_HZ = DX_MHZ * 1e6
gammaX_HZ = gammaX_MHZ * 1e6
bF_X_HZ = bF_X_MHZ * 1e6
c_X_HZ = c_X_MHZ * 1e6

# Excited state A~2Pi_1/2(v=0) -- all in cm^-1
T_A = 16526.750           # cm^-1, term energy
Be_A = 0.348781           # cm^-1, rotational constant
Aso_A = 71.429            # cm^-1, spin-orbit coupling
p_A = -0.044517           # cm^-1, Lambda-doubling p
q_A = -2.916e-4           # cm^-1, Lambda-doubling q

# Transition properties
LAMBDA_NM = 606.0                        # nm
LAMBDA_M = LAMBDA_NM * 1e-9             # m
GAMMA_HZ = 8.3e6                         # Hz (natural linewidth / 2*pi)
GAMMA_RAD = 2.0 * math.pi * GAMMA_HZ    # rad/s
MASS_AMU = 59.0                          # amu
MASS_KG = MASS_AMU * amu                 # kg
K_WAVE = wavenumber(LAMBDA_M)            # 1/m

# Conversion factors
CM_TO_HZ = c * 100.0   # 1 cm^-1 = 2.998e10 Hz


def build_caf_hamiltonian(verbose: bool = True) -> MolecularData:
    """
    Build the full CaF molecular Hamiltonian from spectroscopic constants.

    Ground state X~2Sigma+(v=0, N=1): Hund's case (b)
        12 states: J=1/2 (F=0, F=1) + J=3/2 (F=1, F=2)

    Excited state A~2Pi_1/2(v=0, J'=1/2): Hund's case (a)
        4 states: F'=0 + F'=1

    Note
    ----
    CaF has much larger hyperfine constants than CaOH/SrOH because of
    the F-19 nuclear spin (I=1/2) with large Fermi contact interaction
    (~123 MHz vs ~2.6 MHz for CaOH).

    CaF uses the SAME Zeeman sign convention as SrOH (positive sign
    in Zeeman_x/y), unlike CaOH which uses negative sign.

    Returns
    -------
    MolecularData
    """
    if verbose:
        print("=" * 60)
        print("  Building CaF Hamiltonian from spectroscopic constants")
        print("=" * 60)

    # ==================================================================
    # Step 1: Ground state (case b)
    # ==================================================================

    if verbose:
        print("\n  Ground state: X~2Sigma+(v=0, N=1)")

    ground_basis = enumerate_states(
        HundsCaseB_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0,
            "S": 0.5, "I": 0.5,
            "Lambda": 0, "ell": 0,
            "N": [1],
        }
    )

    if verbose:
        print(f"    {len(ground_basis)} basis states enumerated")

    H_ground = Hamiltonian(ground_basis)
    H_ground.add_operator("B", BX_HZ, Rotation)
    H_ground.add_operator("D", DX_HZ, RotationDistortion)
    H_ground.add_operator("gamma", gammaX_HZ, SpinRotation)
    H_ground.add_operator("bF", bF_X_HZ, Hyperfine_IS)

    H_ground.evaluate()
    ground_states = H_ground.solve()
    ground_coeffs = np.array([s.coeffs for s in ground_states])

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
        print(f"\n  Excited state: A~2Pi_1/2(v=0, J'=1/2)")

    excited_basis_a = enumerate_states(
        HundsCaseA_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0, "ell": 0,
            "Lambda": [-1, 1],
            "I": 0.5,
            "S": 0.5,
            "J": [0.5],
        }
    )

    excited_basis_a = [s for s in excited_basis_a
                       if abs(Fraction(s.P).limit_denominator(10)) == Fraction(1, 2)]

    if verbose:
        print(f"    {len(excited_basis_a)} case (a) basis states")

    H_excited = Hamiltonian(excited_basis_a)
    H_excited.add_operator("Aso", Aso_A * CM_TO_HZ, SpinOrbit)
    H_excited.add_operator("Be", Be_A * CM_TO_HZ, Rotation_a)
    H_excited.add_operator("p+2q", (p_A + 2 * q_A) * CM_TO_HZ, LambdaDoubling_p2q)
    H_excited.evaluate()

    for i in range(len(excited_basis_a)):
        H_excited.matrix[i, i] += T_A * CM_TO_HZ

    excited_states_all = H_excited.solve()

    if verbose:
        print(f"    Diagonalised: {len(excited_states_all)} eigenstates")

    excited_states_all.sort(key=lambda s: s.E)
    excited_states = excited_states_all[4:]  # upper parity doublet

    if verbose:
        E_e_mean = np.mean([s.E for s in excited_states])
        for i, s in enumerate(excited_states):
            print(f"      e{i+1}: E = {(s.E - E_e_mean)/1e6:+8.4f} MHz")

    # ==================================================================
    # Step 3: Convert excited states to case (b) for TDM
    # ==================================================================

    if verbose:
        print(f"\n  Converting excited states to case (b) basis")

    excited_basis_b = enumerate_states(
        HundsCaseB_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0,
            "S": 0.5, "I": 0.5,
            "Lambda": [-1, 0, 1],
            "ell": 0,
            "N": [0, 1],
        }
    )

    excited_basis_b = [s for s in excited_basis_b
                       if abs(s.J - 0.5) < 0.01 and s.N >= abs(s.K)]

    n_e_a = len(excited_basis_a)
    n_e_b = len(excited_basis_b)
    overlap_matrix = np.zeros((n_e_b, n_e_a), dtype=complex)
    for i, sb in enumerate(excited_basis_b):
        for j, sa in enumerate(excited_basis_a):
            overlap_matrix[i, j] = overlap_caseb_casea(sb, sa)

    excited_coeffs_a = np.array([s.coeffs for s in excited_states])
    excited_coeffs_b = excited_coeffs_a @ overlap_matrix.conj().T

    if verbose:
        print(f"    Overlap matrix: {overlap_matrix.shape}")

    # ==================================================================
    # Step 4: Compute TDMs
    # ==================================================================

    if verbose:
        print(f"\n  Computing transition dipole moments")

    n_g = len(ground_states)
    n_e = len(excited_states)
    n_total = n_g + n_e

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
    # Step 5: Compute Zeeman matrices (SrOH convention, positive sign)
    # ==================================================================

    if verbose:
        print(f"\n  Computing Zeeman matrices (standard sign convention)")

    Zeeman_x = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_y = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_z = np.zeros((n_total, n_total), dtype=complex)

    mat_g_0 = operator_to_matrix(ground_basis, Zeeman, 0)
    Zg_0 = ground_coeffs.conj() @ mat_g_0 @ ground_coeffs.T
    Zeeman_z[:n_g, :n_g] = Zg_0

    mat_g_m1 = operator_to_matrix(ground_basis, Zeeman, -1)
    mat_g_p1 = operator_to_matrix(ground_basis, Zeeman, 1)
    Zg_m1 = ground_coeffs.conj() @ mat_g_m1 @ ground_coeffs.T
    Zg_p1 = ground_coeffs.conj() @ mat_g_p1 @ ground_coeffs.T

    # CaF uses the standard (SrOH-like) positive sign convention
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
        E_g_mean = np.mean(energies[:n_g])
        E_e_mean = np.mean(energies[n_g:])
        print(f"\n  Energy summary:")
        print(f"    Ground mean:  {E_g_mean/1e6:.2f} MHz")
        print(f"    Excited mean: {E_e_mean/1e6:.2f} MHz")
        print(f"    Transition:   {(E_e_mean - E_g_mean)/1e12:.6f} THz")
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
        print(f"    T_Doppler = {hbar * GAMMA_RAD / (2 * k_B) * 1e6:.1f} uK")
        print("=" * 60)

    return mol


def load_caf_from_sroh_structure(
    sroh_data_dir: str = None,
    verbose: bool = True,
) -> MolecularData:
    """
    Quick hack: use the same structure as SrOH but with CaF constants.

    Same approach as CaOH: build ground state from scratch, rescale
    SrOH TDM data for the transition dipole moments.

    Parameters
    ----------
    sroh_data_dir : str, optional
        Path to SrOH Julia CSV files.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    MolecularData
    """
    if verbose:
        print("=" * 60)
        print("  Loading CaF via SrOH structure rescaling")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Build CaF ground state from scratch
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 1: Building CaF ground state")

    ground_basis = enumerate_states(
        HundsCaseB_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0,
            "S": 0.5, "I": 0.5,
            "Lambda": 0, "ell": 0,
            "N": [1],
        }
    )

    H_ground = Hamiltonian(ground_basis)
    H_ground.add_operator("B", BX_HZ, Rotation)
    H_ground.add_operator("D", DX_HZ, RotationDistortion)
    H_ground.add_operator("gamma", gammaX_HZ, SpinRotation)
    H_ground.add_operator("bF", bF_X_HZ, Hyperfine_IS)

    H_ground.evaluate()
    ground_states = H_ground.solve()
    ground_coeffs = np.array([s.coeffs for s in ground_states])

    n_g = len(ground_states)
    n_e = 4
    n_total = n_g + n_e

    if verbose:
        E_g_mean = np.mean([s.E for s in ground_states])
        for i, s in enumerate(ground_states):
            print(f"    g{i+1}: E = {(s.E - E_g_mean)/1e6:+8.3f} MHz")
        SR_split = np.mean([s.E for s in ground_states[4:]]) - \
                   np.mean([s.E for s in ground_states[:4]])
        print(f"    SR splitting: {SR_split/1e6:.1f} MHz")

    # ------------------------------------------------------------------
    # Step 2: Compute CaF excited-state energies
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 2: Computing CaF excited-state energies")

    excited_basis_a = enumerate_states(
        HundsCaseA_LinearMolecule,
        {
            "v1": 0, "v2": 0, "v3": 0, "ell": 0,
            "Lambda": [-1, 1],
            "I": 0.5,
            "S": 0.5,
            "J": [0.5],
        }
    )

    excited_basis_a = [s for s in excited_basis_a
                       if abs(Fraction(s.P).limit_denominator(10)) == Fraction(1, 2)]

    H_excited = Hamiltonian(excited_basis_a)
    H_excited.add_operator("Aso", Aso_A * CM_TO_HZ, SpinOrbit)
    H_excited.add_operator("Be", Be_A * CM_TO_HZ, Rotation_a)
    H_excited.add_operator("p+2q", (p_A + 2 * q_A) * CM_TO_HZ, LambdaDoubling_p2q)
    H_excited.evaluate()

    for i in range(len(excited_basis_a)):
        H_excited.matrix[i, i] += T_A * CM_TO_HZ

    excited_states_all = H_excited.solve()
    excited_states_all.sort(key=lambda s: s.E)
    excited_states = excited_states_all[4:]

    if verbose:
        E_e_mean = np.mean([s.E for s in excited_states])
        for i, s in enumerate(excited_states):
            print(f"    e{i+1}: E = {(s.E - E_e_mean)/1e6:+8.4f} MHz")

    # ------------------------------------------------------------------
    # Step 3: Load SrOH TDM and rescale
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 3: Loading and rescaling SrOH TDM data")

    if sroh_data_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        sroh_data_dir = os.path.join(this_dir, "..", "..", "julia_sim")

    sroh_data_dir = os.path.abspath(sroh_data_dir)

    if os.path.exists(os.path.join(sroh_data_dir, "sroh_tdm_re.csv")):
        tdm_re = np.loadtxt(os.path.join(sroh_data_dir, "sroh_tdm_re.csv"))
        tdm_im = np.loadtxt(os.path.join(sroh_data_dir, "sroh_tdm_im.csv"))

        tdm_flat = tdm_re + 1j * tdm_im
        tdm_sroh = np.zeros((16, 16, 3), dtype=complex)
        for state_i in range(16):
            for q in range(3):
                row_idx = state_i * 3 + q
                tdm_sroh[state_i, :, q] = tdm_flat[row_idx, :]

        tdm = tdm_sroh.copy()

        if verbose:
            n_nonzero = np.sum(np.abs(tdm) > 1e-6)
            print(f"    SrOH TDM loaded and adopted: {n_nonzero} nonzero elements")
    else:
        if verbose:
            print(f"    WARNING: SrOH CSV files not found in {sroh_data_dir}")
            print(f"    Falling back to from-scratch TDM")

        excited_basis_b = enumerate_states(
            HundsCaseB_LinearMolecule,
            {
                "v1": 0, "v2": 0, "v3": 0,
                "S": 0.5, "I": 0.5,
                "Lambda": [-1, 0, 1],
                "ell": 0,
                "N": [0, 1],
            }
        )
        excited_basis_b = [s for s in excited_basis_b
                           if abs(s.J - 0.5) < 0.01 and s.N >= abs(s.K)]

        overlap_matrix = np.zeros((len(excited_basis_b), len(excited_basis_a)),
                                  dtype=complex)
        for i, sb in enumerate(excited_basis_b):
            for j, sa in enumerate(excited_basis_a):
                overlap_matrix[i, j] = overlap_caseb_casea(sb, sa)

        excited_coeffs_a = np.array([s.coeffs for s in excited_states])
        excited_coeffs_b = excited_coeffs_a @ overlap_matrix.conj().T

        tdm = compute_tdms(
            ground_states, excited_states,
            ground_basis, excited_basis_b,
            TDM,
            ground_coeffs=ground_coeffs,
            excited_coeffs=excited_coeffs_b)

    # ------------------------------------------------------------------
    # Step 4: Zeeman matrices (standard sign convention like SrOH)
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 4: Computing Zeeman matrices (standard sign convention)")

    Zeeman_x_mat = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_y_mat = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_z_mat = np.zeros((n_total, n_total), dtype=complex)

    mat_g_0 = operator_to_matrix(ground_basis, Zeeman, 0)
    Zg_0 = ground_coeffs.conj() @ mat_g_0 @ ground_coeffs.T
    Zeeman_z_mat[:n_g, :n_g] = Zg_0

    mat_g_m1 = operator_to_matrix(ground_basis, Zeeman, -1)
    mat_g_p1 = operator_to_matrix(ground_basis, Zeeman, 1)
    Zg_m1 = ground_coeffs.conj() @ mat_g_m1 @ ground_coeffs.T
    Zg_p1 = ground_coeffs.conj() @ mat_g_p1 @ ground_coeffs.T

    # Standard positive sign convention (like SrOH)
    Zeeman_x_mat[:n_g, :n_g] = (Zg_m1 - Zg_p1) / math.sqrt(2)
    Zeeman_y_mat[:n_g, :n_g] = 1j * (Zg_m1 + Zg_p1) / math.sqrt(2)

    if verbose:
        print(f"    Zeeman matrices: {Zeeman_z_mat.shape}")

    # ------------------------------------------------------------------
    # Step 5: Assemble energies
    # ------------------------------------------------------------------

    energies = np.zeros(n_total)
    for i, s in enumerate(ground_states):
        energies[i] = s.E
    for i, s in enumerate(excited_states):
        energies[n_g + i] = s.E

    if verbose:
        E_g_mean = np.mean(energies[:n_g])
        E_e_mean = np.mean(energies[n_g:])
        print(f"\n  Energy summary:")
        print(f"    Ground mean:  {E_g_mean/1e6:.2f} MHz")
        print(f"    Excited mean: {E_e_mean/1e6:.2f} MHz")
        print(f"    Transition:   {(E_e_mean - E_g_mean)/1e12:.6f} THz")

    # ------------------------------------------------------------------
    # Step 6: Create MolecularData
    # ------------------------------------------------------------------

    mol = MolecularData(
        energies=energies,
        tdm=tdm,
        zeeman_x=Zeeman_x_mat,
        zeeman_y=Zeeman_y_mat,
        zeeman_z=Zeeman_z_mat,
        Gamma=GAMMA_RAD,
        k=K_WAVE,
        mass=MASS_KG,
        wavelength=LAMBDA_M,
        n_ground=n_g,
        n_excited=n_e,
        n_states=n_total,
    )

    if verbose:
        print(f"\n  MolecularData created (CaF via SrOH rescaling)")
        print(f"    {n_g} ground + {n_e} excited = {n_total} states")
        print(f"    lambda = {LAMBDA_NM:.1f} nm")
        print(f"    Gamma/(2pi) = {GAMMA_HZ/1e6:.1f} MHz")
        print(f"    mass = {MASS_AMU:.0f} amu")
        print(f"    T_Doppler = {hbar * GAMMA_RAD / (2 * k_B) * 1e6:.1f} uK")
        print("=" * 60)

    return mol
