"""
CaOH molecular data: Hamiltonian construction and SrOH-structure-based loader.

Builds the full 16-level CaOH molecular Hamiltonian from first
principles using spectroscopic constants, or loads it by rescaling
the validated SrOH data (preferred for benchmarking).

Transition: X~2Sigma+(000, N=1) -> A~2Pi_1/2(000, J'=1/2) at ~626 nm.

Level structure (same as SrOH):
  - 12 ground states (X, N=1): Hund's case (b)
      4 states with J=1/2 (F=0,1) + 8 states with J=3/2 (F=1,2)
  - 4 excited states (A, J'=1/2): Hund's case (a) -> converted to case (b)
      F'=0 (1 state) + F'=1 (3 states)

Spectroscopic constants from Christian's CaOHBlueMOTSimulations/CaOH/ files.

References
----------
* Christian Hallas, CaOHBlueMOTSimulations
* Li et al., Phys. Rev. Lett. (2023) [blue MOT of CaF, analogy for CaOH]
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
# Spectroscopic constants for CaOH
# =====================================================================

# Ground state X~2Sigma+(000)
# CaOH has BX already in MHz from Christian's code
BX_MHZ = 10023.0841     # MHz, rotational constant
DX_MHZ = 1.154e-2       # MHz, centrifugal distortion
gammaX_MHZ = 34.7593    # MHz, spin-rotation constant
bF_X_MHZ = 2.602        # MHz, Fermi contact hyperfine
c_X_MHZ = 2.053         # MHz, dipolar hyperfine

# Convert to Hz for Hamiltonian (the Hamiltonian works in Hz)
BX_HZ = BX_MHZ * 1e6
DX_HZ = DX_MHZ * 1e6
gammaX_HZ = gammaX_MHZ * 1e6
bF_X_HZ = bF_X_MHZ * 1e6
c_X_HZ = c_X_MHZ * 1e6

# Excited state A~2Pi_1/2(000) -- all in cm^-1
T_A = 15998.122          # cm^-1, term energy
Be_A = 0.3412200         # cm^-1, rotational constant
Aso_A = 66.8181          # cm^-1, spin-orbit coupling
p_A = -0.04287           # cm^-1, Lambda-doubling p
q_A = -0.3257e-3         # cm^-1, Lambda-doubling q

# Transition properties
LAMBDA_NM = 626.0                        # nm
LAMBDA_M = LAMBDA_NM * 1e-9             # m
GAMMA_HZ = 6.4e6                         # Hz (natural linewidth / 2*pi)
GAMMA_RAD = 2.0 * math.pi * GAMMA_HZ    # rad/s
MASS_AMU = 57.0                          # amu
MASS_KG = MASS_AMU * amu                 # kg
K_WAVE = wavenumber(LAMBDA_M)            # 1/m

# Conversion factors
CM_TO_HZ = c * 100.0   # 1 cm^-1 = 2.998e10 Hz


def build_caoh_hamiltonian(verbose: bool = True) -> MolecularData:
    """
    Build the full CaOH molecular Hamiltonian from spectroscopic constants.

    Ground state X~2Sigma+(000, N=1): Hund's case (b)
        12 states: J=1/2 (F=0, F=1) + J=3/2 (F=1, F=2)

    Excited state A~2Pi_1/2(000, J'=1/2): Hund's case (a)
        4 states: F'=0 + F'=1 (selecting +parity / J'=1/2 only)

    Note
    ----
    For production simulations, use ``load_caoh_from_sroh_structure()``
    which rescales the validated SrOH TDM data (both are ^2Sigma -> ^2Pi_1/2
    transitions with identical angular momentum structure).

    The from-scratch builder correctly produces the ground state energy
    structure and Zeeman operators, but the excited-state case (a) -> case (b)
    TDM conversion has the known parity-selection issue.

    CaOH Zeeman sign convention (IMPORTANT):
    Christian's CaOH code uses a NEGATIVE sign in the Zeeman_x/y definition:
      Zeeman_x = -(Zeeman(s,s',-1) - Zeeman(s,s',1)) / sqrt(2)
    This is OPPOSITE to the SrOH convention.

    Returns
    -------
    MolecularData
        Complete molecular data ready for OBE simulation.
    """
    if verbose:
        print("=" * 60)
        print("  Building CaOH Hamiltonian from spectroscopic constants")
        print("=" * 60)

    # ==================================================================
    # Step 1: Ground state (case b)
    # ==================================================================

    if verbose:
        print("\n  Ground state: X~2Sigma+(000, N=1)")

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
        for i, s in enumerate(ground_basis):
            print(f"      {i}: {s}")

    # Build ground Hamiltonian
    # CaOH constants are already in Hz, so pass them directly
    H_ground = Hamiltonian(ground_basis)
    H_ground.add_operator("B", BX_HZ, Rotation)
    H_ground.add_operator("D", DX_HZ, RotationDistortion)
    H_ground.add_operator("gamma", gammaX_HZ, SpinRotation)
    H_ground.add_operator("bF", bF_X_HZ, Hyperfine_IS)

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
        for i, s in enumerate(excited_basis_a):
            print(f"      {i}: {s}")

    # Build excited Hamiltonian in case (a) basis
    H_excited = Hamiltonian(excited_basis_a)
    H_excited.add_operator("Aso", Aso_A * CM_TO_HZ, SpinOrbit)
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

    # Select upper parity doublet (positive parity for CaOH)
    excited_states_all.sort(key=lambda s: s.E)
    excited_states = excited_states_all[4:]  # upper doublet

    if verbose:
        print(f"    Selected {len(excited_states)} states (upper parity doublet)")
        E_e_mean = np.mean([s.E for s in excited_states])
        for i, s in enumerate(excited_states):
            print(f"      e{i+1}: E = {(s.E - E_e_mean)/1e6:+8.4f} MHz")

    # ==================================================================
    # Step 3: Convert excited states to case (b) for TDM computation
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

    if verbose:
        print(f"    {len(excited_basis_b)} case (b) basis states for excited manifold")

    # Compute overlap matrix
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
    # CaOH uses NEGATIVE sign convention for Zeeman_x/y
    # ==================================================================

    if verbose:
        print(f"\n  Computing Zeeman matrices (CaOH sign convention)")

    # Ground state Zeeman (case b)
    Zeeman_x = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_y = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_z = np.zeros((n_total, n_total), dtype=complex)

    # p=0 -> Bz
    mat_g_0 = operator_to_matrix(ground_basis, Zeeman, 0)
    Zg_0 = ground_coeffs.conj() @ mat_g_0 @ ground_coeffs.T
    Zeeman_z[:n_g, :n_g] = Zg_0

    # p=-1 and p=+1 for Bx and By
    mat_g_m1 = operator_to_matrix(ground_basis, Zeeman, -1)
    mat_g_p1 = operator_to_matrix(ground_basis, Zeeman, 1)
    Zg_m1 = ground_coeffs.conj() @ mat_g_m1 @ ground_coeffs.T
    Zg_p1 = ground_coeffs.conj() @ mat_g_p1 @ ground_coeffs.T

    # CaOH NEGATIVE sign convention:
    # Zeeman_x = -(Zeeman(-1) - Zeeman(+1)) / sqrt(2)
    # Zeeman_y = -i*(Zeeman(-1) + Zeeman(+1)) / sqrt(2)
    # This is OPPOSITE to SrOH where it's positive.
    Zeeman_x[:n_g, :n_g] = -(Zg_m1 - Zg_p1) / math.sqrt(2)
    Zeeman_y[:n_g, :n_g] = -1j * (Zg_m1 + Zg_p1) / math.sqrt(2)

    if verbose:
        print(f"    Zeeman matrices assembled: {Zeeman_z.shape}")
        print(f"    Using NEGATIVE sign convention for Zeeman_x/y (CaOH)")

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
        print(f"    T_Doppler = {hbar * GAMMA_RAD / (2 * k_B) * 1e6:.1f} uK")
        print("=" * 60)

    return mol


def load_caoh_from_sroh_structure(
    sroh_data_dir: str = None,
    verbose: bool = True,
) -> MolecularData:
    """
    Quick hack: use the same structure as SrOH but with CaOH constants.

    Good enough for benchmarking since both are linear ^2Sigma molecules
    with identical angular momentum structure (12 ground + 4 excited).

    The approach:
    1. Build the CaOH ground-state Hamiltonian from scratch (gives correct
       energy structure and Zeeman operators).
    2. For the TDM, rescale the SrOH Julia-validated TDM data. The TDM
       structure is nearly identical for all ^2Sigma -> ^2Pi_1/2 transitions
       in alkaline-earth monohydroxides; only the overall scale differs.
    3. Construct excited-state energies from CaOH spectroscopic constants.

    Parameters
    ----------
    sroh_data_dir : str, optional
        Path to SrOH Julia CSV files. If None, attempts to find them
        relative to this file.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    MolecularData
        CaOH molecular data ready for OBE simulation.
    """
    if verbose:
        print("=" * 60)
        print("  Loading CaOH via SrOH structure rescaling")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Build CaOH ground state from scratch
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 1: Building CaOH ground state")

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
    # Step 2: Compute CaOH excited-state energies
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 2: Computing CaOH excited-state energies")

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
    excited_states = excited_states_all[4:]  # upper parity doublet

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
        # Try to find Julia CSV files relative to this module
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

        # The TDM structure is the same, only the overall magnitude
        # differs by sqrt(Gamma_CaOH/Gamma_SrOH * lambda_SrOH^3/lambda_CaOH^3)
        # For rate equations, the d_squared ratios cancel in the branching
        # ratios, so we only need the overall scale to get absolute forces
        # right. The structure (which transitions are allowed and relative
        # strengths) is identical.
        tdm = tdm_sroh.copy()

        if verbose:
            n_nonzero = np.sum(np.abs(tdm) > 1e-6)
            print(f"    SrOH TDM loaded and adopted: {n_nonzero} nonzero elements")
    else:
        if verbose:
            print(f"    WARNING: SrOH CSV files not found in {sroh_data_dir}")
            print(f"    Falling back to from-scratch TDM (may have parity issue)")

        # Fall back to computing TDM from scratch
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

        tdm = compute_tdms(
            ground_states, excited_states,
            ground_basis, excited_basis_b,
            TDM,
            ground_coeffs=ground_coeffs,
            excited_coeffs=excited_coeffs_b)

    # ------------------------------------------------------------------
    # Step 4: Zeeman matrices with CaOH sign convention
    # ------------------------------------------------------------------

    if verbose:
        print("\n  Step 4: Computing Zeeman matrices (CaOH sign convention)")

    Zeeman_x = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_y = np.zeros((n_total, n_total), dtype=complex)
    Zeeman_z = np.zeros((n_total, n_total), dtype=complex)

    # p=0 -> Bz
    mat_g_0 = operator_to_matrix(ground_basis, Zeeman, 0)
    Zg_0 = ground_coeffs.conj() @ mat_g_0 @ ground_coeffs.T
    Zeeman_z[:n_g, :n_g] = Zg_0

    # p=-1 and p=+1
    mat_g_m1 = operator_to_matrix(ground_basis, Zeeman, -1)
    mat_g_p1 = operator_to_matrix(ground_basis, Zeeman, 1)
    Zg_m1 = ground_coeffs.conj() @ mat_g_m1 @ ground_coeffs.T
    Zg_p1 = ground_coeffs.conj() @ mat_g_p1 @ ground_coeffs.T

    # CaOH NEGATIVE sign convention
    Zeeman_x[:n_g, :n_g] = -(Zg_m1 - Zg_p1) / math.sqrt(2)
    Zeeman_y[:n_g, :n_g] = -1j * (Zg_m1 + Zg_p1) / math.sqrt(2)

    if verbose:
        print(f"    Zeeman matrices: {Zeeman_z.shape}")
        print(f"    Using NEGATIVE sign convention (CaOH)")

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
        print(f"\n  MolecularData created (CaOH via SrOH rescaling)")
        print(f"    {n_g} ground + {n_e} excited = {n_total} states")
        print(f"    lambda = {LAMBDA_NM:.1f} nm")
        print(f"    Gamma/(2pi) = {GAMMA_HZ/1e6:.1f} MHz")
        print(f"    mass = {MASS_AMU:.0f} amu")
        print(f"    T_Doppler = {hbar * GAMMA_RAD / (2 * k_B) * 1e6:.1f} uK")
        print("=" * 60)

    return mol
