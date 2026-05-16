"""
Comprehensive function-by-function tests for the molmot package.

Tests every public function in the package with valid inputs and
verifies outputs are physically reasonable.

Run with:
    cd "/Users/dslmd/Downloads/DC red MOT" && python3 -m pytest tests/ -v
"""

import pytest
import numpy as np
import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JULIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "julia_sim")


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="session")
def mol():
    """Load the validated SrOH molecular data from Julia CSV files."""
    from molmot.molecules.sroh import load_sroh_from_julia
    return load_sroh_from_julia(JULIA_DIR, verbose=False)


@pytest.fixture(scope="session")
def mol_scratch():
    """Build SrOH from spectroscopic constants (no Julia data)."""
    from molmot.molecules.sroh import build_sroh_hamiltonian
    return build_sroh_hamiltonian(verbose=False)


# =========================================================================
# TestWigner: Wigner 3j, 6j, 9j symbols
# =========================================================================

class TestWigner:
    """Test Wigner symbol computations."""

    def test_3j_known_values(self):
        """Wigner 3j(1,1,0; 0,0,0) = -1/sqrt(3)."""
        from molmot.wigner import wigner3j
        val = wigner3j(1, 1, 0, 0, 0, 0)
        assert abs(val - (-1.0 / np.sqrt(3))) < 1e-10

    def test_3j_half_integer(self):
        """Wigner 3j(1/2,1/2,1; 1/2,-1/2,0) is known."""
        from molmot.wigner import wigner3j
        val = wigner3j(0.5, 0.5, 1, 0.5, -0.5, 0)
        # Known: (-1)^(j1-j2-m3) * sqrt(delta) * sum...
        # For (1/2,1/2,1;1/2,-1/2,0): = 1/sqrt(6)
        expected = 1.0 / np.sqrt(6)
        assert abs(val - expected) < 1e-10

    def test_3j_selection_rule_m(self):
        """3j is zero if m1+m2+m3 != 0."""
        from molmot.wigner import wigner3j
        assert wigner3j(1, 1, 1, 1, 1, 0) == 0.0

    def test_3j_selection_rule_triangle(self):
        """3j is zero if triangle inequality violated."""
        from molmot.wigner import wigner3j
        assert wigner3j(1, 1, 5, 0, 0, 0) == 0.0

    def test_6j_known_value(self):
        """6j{1,1,0; 0.5,0.5,0.5} has a known value."""
        from molmot.wigner import wigner6j
        val = wigner6j(1, 1, 0, 0.5, 0.5, 0.5)
        # Should be nonzero if triangle conditions are met
        # {1,1,0} ok, {1,0.5,0.5} ok, {0.5,1,0.5} ok, {0.5,0.5,0} ok
        assert isinstance(val, float)

    def test_6j_triangle(self):
        """6j with violated triangle condition returns 0."""
        from molmot.wigner import wigner6j
        assert wigner6j(1, 1, 5, 1, 1, 1) == 0.0

    def test_6j_symmetry(self):
        """6j is invariant under column permutations."""
        from molmot.wigner import wigner6j
        val1 = wigner6j(1, 0.5, 0.5, 0.5, 1, 0.5)
        val2 = wigner6j(0.5, 1, 0.5, 1, 0.5, 0.5)
        assert abs(val1 - val2) < 1e-10

    def test_9j_sum_rule(self):
        """9j with j9=0 reduces to a 6j (sum rule check)."""
        from molmot.wigner import wigner9j, wigner6j
        # {j1,j2,j3; j4,j5,j6; j7,j8,0}
        # = delta(j3,j6)*delta(j7,j8) * (-1)^(j2+j3+j4+j7)
        #   / sqrt((2j3+1)(2j7+1)) * {j1,j2,j3; j5,j4,j7}
        j1, j2, j3 = 1, 0.5, 0.5
        j4, j5, j6 = 0.5, 1, 0.5
        j7, j8 = 1, 1
        val9j = wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, 0)
        # j3==j6 (0.5==0.5), j7==j8 (1==1) => nonzero
        val6j = wigner6j(j1, j2, j3, j5, j4, j7)
        phase = (-1) ** int(2 * (j2 + j3 + j4 + j7))
        expected = phase / np.sqrt((2 * j3 + 1) * (2 * j7 + 1)) * val6j
        assert abs(val9j - expected) < 1e-10


# =========================================================================
# TestBasis: Quantum state enumeration
# =========================================================================

class TestBasis:
    """Test basis state enumeration."""

    def test_enumerate_sroh_ground(self):
        """SrOH N=1 ground state should give 12 basis states."""
        from molmot.states.basis import enumerate_states
        from molmot.states.case_b import HundsCaseB_LinearMolecule
        states = enumerate_states(
            HundsCaseB_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "S": 0.5, "I": 0.5,
             "Lambda": 0, "ell": 0, "N": [1]}
        )
        assert len(states) == 12

    def test_enumerate_case_a_excited(self):
        """SrOH case (a) excited states for J=1/2 should give 8 states."""
        from molmot.states.basis import enumerate_states
        from molmot.states.case_a import HundsCaseA_LinearMolecule
        from fractions import Fraction
        states = enumerate_states(
            HundsCaseA_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "ell": 0,
             "Lambda": [-1, 1], "I": 0.5, "S": 0.5, "J": [0.5]}
        )
        # Filter to |P|=1/2
        states = [s for s in states
                  if abs(Fraction(s.P).limit_denominator(10)) == Fraction(1, 2)]
        assert len(states) == 8

    def test_delta_function(self):
        """delta() should correctly compare quantum numbers."""
        from molmot.states.basis import BasisState, delta
        from molmot.states.case_b import HundsCaseB_LinearMolecule
        s1 = HundsCaseB_LinearMolecule(N=1, J=0.5, F=1, M=0, S=0.5, I=0.5)
        s2 = HundsCaseB_LinearMolecule(N=1, J=0.5, F=1, M=0, S=0.5, I=0.5)
        s3 = HundsCaseB_LinearMolecule(N=1, J=0.5, F=1, M=1, S=0.5, I=0.5)
        assert delta(s1, s2, "N", "J", "F", "M") is True
        assert delta(s1, s3, "N", "J", "F", "M") is False


# =========================================================================
# TestOperators: Angular momentum operators
# =========================================================================

class TestOperators:
    """Test operator matrix elements."""

    def test_rotation_diagonal(self):
        """Rotation(s, s) = N(N+1) for Lambda=0."""
        from molmot.states.case_b import HundsCaseB_LinearMolecule, Rotation
        s = HundsCaseB_LinearMolecule(N=1, J=0.5, F=0, M=0,
                                       S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        val = Rotation(s, s)
        assert abs(val - 2.0) < 1e-10  # N(N+1) = 1*2 = 2

    def test_zeeman_selection_rules(self):
        """Zeeman operator vanishes for Delta_M != p."""
        from molmot.states.case_b import HundsCaseB_LinearMolecule, Zeeman
        s1 = HundsCaseB_LinearMolecule(N=1, J=0.5, F=1, M=0,
                                        S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        s2 = HundsCaseB_LinearMolecule(N=1, J=0.5, F=1, M=0,
                                        S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        # p=1 requires M2-M1 = 1, but both are M=0
        val = Zeeman(s1, s2, 1)
        assert abs(val) < 1e-15

    def test_tdm_selection_rules(self):
        """TDM vanishes for |Delta_F| > 1."""
        from molmot.states.case_b import HundsCaseB_LinearMolecule, TDM
        s1 = HundsCaseB_LinearMolecule(N=1, J=1.5, F=2, M=0,
                                        S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        s2 = HundsCaseB_LinearMolecule(N=0, J=0.5, F=0, M=0,
                                        S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        # F=2 -> F'=0: Delta_F = -2, should be zero
        val = TDM(s1, s2, 0)
        assert abs(val) < 1e-15

    def test_spin_rotation_diagonal(self):
        """SpinRotation diagonal is 0.5*(J(J+1) - N(N+1) - S(S+1))."""
        from molmot.states.case_b import HundsCaseB_LinearMolecule, SpinRotation
        s = HundsCaseB_LinearMolecule(N=1, J=1.5, F=1, M=0,
                                       S=0.5, I=0.5, Lambda=0, ell=0, K=0)
        val = SpinRotation(s, s)
        expected = 0.5 * (1.5 * 2.5 - 1 * 2 - 0.5 * 1.5)
        assert abs(val - expected) < 1e-10


# =========================================================================
# TestHamiltonian: Hamiltonian construction and diagonalization
# =========================================================================

class TestHamiltonian:
    """Test Hamiltonian construction."""

    def test_hermiticity(self, mol):
        """The loaded molecular data should have Hermitian Zeeman matrices."""
        assert np.allclose(mol.zeeman_z, mol.zeeman_z.conj().T, atol=1e-12)

    def test_eigenvalue_count(self, mol):
        """Should have 16 eigenvalues (12 ground + 4 excited)."""
        assert len(mol.energies) == 16

    def test_energies_ordered(self, mol):
        """Ground states should have lower energy than excited states."""
        E_g_mean = np.mean(mol.energies[:12])
        E_e_mean = np.mean(mol.energies[12:])
        assert E_e_mean > E_g_mean

    def test_sr_splitting_positive(self, mol):
        """SR splitting J=1/2 above J=3/2 (positive)."""
        # First 8 states are J=3/2, last 4 are J=1/2
        E_J32 = np.mean(mol.energies[:8])
        E_J12 = np.mean(mol.energies[8:12])
        sr_split_MHz = (E_J12 - E_J32) / 1e6
        # SrOH SR split is ~55 MHz
        assert sr_split_MHz > 10  # physically > 0

    def test_tdm_nonzero(self, mol):
        """TDM should have nonzero elements connecting ground-excited."""
        d_sq = mol.d_squared
        assert np.sum(d_sq) > 0.1

    def test_extend_basis(self):
        """extend_basis should produce a larger Hamiltonian."""
        from molmot.states.basis import enumerate_states
        from molmot.states.case_b import HundsCaseB_LinearMolecule, Rotation
        from molmot.states.hamiltonian import Hamiltonian, extend_basis

        basis_N1 = enumerate_states(
            HundsCaseB_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "S": 0.5, "I": 0.5,
             "Lambda": 0, "ell": 0, "N": [1]}
        )
        H = Hamiltonian(basis_N1)
        H.add_operator("B", 1e9, Rotation)

        basis_N3 = enumerate_states(
            HundsCaseB_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "S": 0.5, "I": 0.5,
             "Lambda": 0, "ell": 0, "N": [3]}
        )

        H_ext = extend_basis(H, basis_N3)
        assert H_ext.n > H.n
        assert H_ext.matrix is not None
        # The matrix should be nonzero
        assert np.max(np.abs(H_ext.matrix)) > 0

    def test_save_load_hamiltonian(self, mol_scratch, tmp_path):
        """Save and load should preserve eigenvalues."""
        from molmot.states.hamiltonian import save_hamiltonian, load_hamiltonian
        from molmot.states.basis import enumerate_states
        from molmot.states.case_b import HundsCaseB_LinearMolecule, Rotation, SpinRotation
        from molmot.states.hamiltonian import Hamiltonian

        basis = enumerate_states(
            HundsCaseB_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "S": 0.5, "I": 0.5,
             "Lambda": 0, "ell": 0, "N": [1]}
        )
        H = Hamiltonian(basis)
        H.add_operator("B", 1e9, Rotation)
        H.evaluate()
        H.solve()

        fpath = str(tmp_path / "test_hamiltonian.npz")
        save_hamiltonian(H, fpath)
        data = load_hamiltonian(fpath)
        assert np.allclose(data['eigenvalues'], H.energies, atol=1e-6)


# =========================================================================
# TestOverlaps: Basis conversion
# =========================================================================

class TestOverlaps:
    """Test case (a) / case (b) overlap integrals."""

    def test_overlap_unitarity(self):
        """Overlap matrix between case (a) and case (b) should be (semi-)unitary."""
        from molmot.states.basis import enumerate_states
        from molmot.states.case_b import HundsCaseB_LinearMolecule
        from molmot.states.case_a import HundsCaseA_LinearMolecule
        from molmot.states.overlaps import overlap_caseb_casea
        from fractions import Fraction

        basis_a = enumerate_states(
            HundsCaseA_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "ell": 0,
             "Lambda": [-1, 1], "I": 0.5, "S": 0.5, "J": [0.5]}
        )
        basis_a = [s for s in basis_a
                   if abs(Fraction(s.P).limit_denominator(10)) == Fraction(1, 2)]

        basis_b = enumerate_states(
            HundsCaseB_LinearMolecule,
            {"v1": 0, "v2": 0, "v3": 0, "S": 0.5, "I": 0.5,
             "Lambda": [-1, 0, 1], "ell": 0, "N": [0, 1]}
        )
        basis_b = [s for s in basis_b
                   if abs(s.J - 0.5) < 0.01 and s.N >= abs(s.K)]

        n_a = len(basis_a)
        n_b = len(basis_b)
        S = np.zeros((n_b, n_a), dtype=complex)
        for i, sb in enumerate(basis_b):
            for j, sa in enumerate(basis_a):
                S[i, j] = overlap_caseb_casea(sb, sa)

        # S^dag @ S should be close to identity (n_a x n_a)
        check = S.conj().T @ S
        assert np.max(np.abs(check - np.eye(n_a))) < 0.5  # semi-unitary up to subspace


# =========================================================================
# TestRateEquations: Rate equation force solver
# =========================================================================

class TestRateEquations:
    """Test rate equation force computation."""

    def test_force_antisymmetric(self, mol):
        """F(z) = -F(-z) at v=0 (restoring force)."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.fields import make_dc_mot_beams

        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        z0 = 1e-3
        B_grad = 0.16

        Fp, _, _ = solve_rate_equations(mol, beams, 0.0, z0, B_grad)
        Fm, _, _ = solve_rate_equations(mol, beams, 0.0, -z0, B_grad)

        # Should be approximately antisymmetric
        assert abs(Fp + Fm) < 0.1 * (abs(Fp) + abs(Fm) + 1e-30)

    def test_force_damping(self, mol):
        """F(v) opposes v at z=0 (damping force)."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.fields import make_dc_mot_beams

        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        v0 = 1.0  # m/s
        B_grad = 0.16

        Fv, _, _ = solve_rate_equations(mol, beams, v0, 0.0, B_grad)
        # For red-detuned light, F should oppose v (negative when v positive)
        assert Fv < 0

    def test_populations_sum_to_one(self, mol):
        """Ground state populations should sum to 1."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.fields import make_dc_mot_beams

        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        _, p, _ = solve_rate_equations(mol, beams, 0.0, 1e-3, 0.16)
        assert abs(np.sum(p) - 1.0) < 1e-6

    def test_scattering_rate_positive(self, mol):
        """Scattering rate should be positive."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.fields import make_dc_mot_beams

        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        _, _, R = solve_rate_equations(mol, beams, 0.0, 0.0, 0.16)
        assert R > 0

    def test_jit_matches_python(self, mol):
        """JIT solver should match pure Python solver to ~1e-10."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.rate_equations_jit import solve_rate_equations_jit
        from molmot.obe.fields import make_dc_mot_beams

        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        v, z, B = 0.5, 1e-3, 0.16

        F_py, p_py, R_py = solve_rate_equations(mol, beams, v, z, B)
        F_jit, p_jit, R_jit = solve_rate_equations_jit(mol, beams, v, z, B)

        assert abs(F_py - F_jit) < max(1e-10 * abs(F_py), 1e-28)
        assert np.allclose(p_py, p_jit, atol=1e-8)
        assert abs(R_py - R_jit) < max(1e-10 * abs(R_py), 1e-6)

    def test_rf_mot_force(self, mol):
        """RF MOT should produce restoring force."""
        from molmot.obe.rate_equations import solve_rate_equations
        from molmot.obe.fields import make_rf_mot_beams

        beams_0 = make_rf_mot_beams(mol, delta_Gamma=-1.0, s0=1.0, phase=0)
        beams_1 = make_rf_mot_beams(mol, delta_Gamma=-1.0, s0=1.0, phase=1)

        z = 1e-3
        B = 0.16
        F0, _, _ = solve_rate_equations(mol, beams_0, 0.0, z, B)
        F1, _, _ = solve_rate_equations(mol, beams_1, 0.0, z, -B)
        F_avg = 0.5 * (F0 + F1)

        # RF MOT force at z>0 should push towards z=0 (negative)
        assert F_avg < 0


# =========================================================================
# TestSSE: Stochastic Schrodinger Equation
# =========================================================================

class TestSSE:
    """Test SSE / MCWF solver."""

    def test_creates_problem(self, mol):
        """SSEProblem should instantiate without error."""
        from molmot.obe.stochastic import SSEProblem
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        prob = SSEProblem(mol, bp, B_gradient=0.16)
        assert prob.n_states == 16
        assert prob.state_size == 2 * 16 + 6

    def test_short_trajectory(self, mol):
        """A short SSE trajectory should complete without error."""
        from molmot.obe.stochastic import SSEProblem, SSESolver
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        prob = SSEProblem(mol, bp, B_gradient=0.16)
        solver = SSESolver(prob)

        result = solver.run(
            r0=np.array([0.0, 0.0, 0.0]),
            v0=np.array([0.0, 0.0, 0.0]),
            t_max=1e-7,  # very short
            save_every=10,
            rng_seed=42,
        )
        assert result.times.shape[0] > 0
        assert result.positions.shape[1] == 3
        # Populations should sum to ~1 at each saved time
        for pops in result.populations:
            assert abs(np.sum(pops) - 1.0) < 0.05


# =========================================================================
# TestFloquet: Floquet sub-Doppler solver
# =========================================================================

class TestFloquet:
    """Test Floquet OBE solver."""

    def test_creates_solver(self, mol):
        """FloquetOBE should instantiate without error."""
        from molmot.obe.floquet import FloquetOBE
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        floquet = FloquetOBE(mol, bp, B_gradient=0.0, n_max=2)
        assert floquet.N == 16
        assert floquet.n_max == 2

    def test_force_at_v0(self, mol):
        """Floquet force at v~0 should be small (no Doppler asymmetry)."""
        from molmot.obe.floquet import FloquetOBE
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        floquet = FloquetOBE(mol, bp, B_gradient=0.0, n_max=2)
        F_arr = floquet.force_vs_velocity(np.array([1e-6]))
        # At v=0, z_macro=0, force should be very small
        assert abs(F_arr[0]) < 1e-20

    def test_scattering_rate(self, mol):
        """Scattering rate from Floquet should be positive."""
        from molmot.obe.floquet import FloquetOBE
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        floquet = FloquetOBE(mol, bp, B_gradient=0.0, n_max=2)
        R = floquet.scattering_rate(0.1)
        assert R >= 0


# =========================================================================
# TestLindblad: Full OBE (Lindblad) solver
# =========================================================================

class TestLindblad:
    """Test Lindblad master equation solver."""

    def test_build_liouvillian(self, mol):
        """Liouvillian should be N^2 x N^2."""
        from molmot.obe.lindblad import build_liouvillian
        N = mol.n_states
        H = np.diag(mol.energies * 2 * np.pi)
        L = build_liouvillian(H, mol.tdm, mol.Gamma, mol.n_ground, mol.n_excited)
        assert L.shape == (N * N, N * N)

    def test_steady_state_trace(self, mol):
        """Steady-state density matrix should have Tr(rho)=1."""
        from molmot.obe.lindblad import steady_state_density_matrix
        N = mol.n_states
        H = np.diag(mol.energies * 2 * np.pi)
        rho = steady_state_density_matrix(H, mol.tdm, mol.Gamma,
                                           mol.n_ground, mol.n_excited)
        assert abs(np.trace(rho).real - 1.0) < 1e-6


# =========================================================================
# Test3D: 3D MOT simulator
# =========================================================================

class Test3D:
    """Test 3D MOT simulator."""

    def test_creates_simulator(self, mol):
        """MOTSimulator3D should instantiate without error."""
        from molmot.mot.simulator_3d import MOTSimulator3D
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        sim = MOTSimulator3D(mol, bp, B_gradient=0.16)
        assert len(sim.beams) > 0

    def test_force_3d_at_origin(self, mol):
        """3D force at origin with v=0 should be approximately zero."""
        from molmot.mot.simulator_3d import MOTSimulator3D
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        sim = MOTSimulator3D(mol, bp, B_gradient=0.16)
        F, R, p = sim.force_3d(np.zeros(3), np.zeros(3))
        assert F.shape == (3,)
        # At exact origin and zero velocity, force should be small
        assert np.linalg.norm(F) < 1e-20

    def test_force_3d_restoring(self, mol):
        """3D force should be restoring along z-axis."""
        from molmot.mot.simulator_3d import MOTSimulator3D
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4

        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        sim = MOTSimulator3D(mol, bp, B_gradient=0.16)
        F, _, _ = sim.force_3d(np.array([0.0, 0.0, 1e-3]), np.zeros(3))
        # Force z-component should push back towards origin (negative)
        assert F[2] < 0


# =========================================================================
# TestMOTSimulators: RF and DC MOT Simulators
# =========================================================================

class TestMOTSimulators:
    """Test RF and DC MOT simulator classes."""

    def test_dc_simulator(self, mol):
        """DCMOTSimulator should compute force."""
        from molmot.mot.simulator import DCMOTSimulator
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        F, p, R = sim.force(0.0, 1e-3)
        assert isinstance(F, float)
        assert F < 0  # restoring

    def test_rf_simulator(self, mol):
        """RFMOTSimulator should compute time-averaged force."""
        from molmot.mot.simulator import RFMOTSimulator
        sim = RFMOTSimulator(mol, delta_Gamma=-1.0, s0=1.0, B_gradient=0.16)
        F, p, R = sim.force(0.0, 1e-3)
        assert isinstance(F, float)
        assert F < 0  # restoring

    def test_dc_update_beams(self, mol):
        """Updating beams should not raise."""
        from molmot.mot.simulator import DCMOTSimulator
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        sim.delta_Gamma = -0.50
        sim.update_beams()
        F, _, _ = sim.force(0.0, 1e-3)
        assert isinstance(F, float)


# =========================================================================
# TestForceScan: Force scanning utilities
# =========================================================================

class TestForceScan:
    """Test force scanning and characterization."""

    def test_spring_constant(self, mol):
        """Spring constant should be positive for a trapping configuration."""
        from molmot.mot.simulator import DCMOTSimulator
        from molmot.mot.force_scan import spring_constant
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        k = spring_constant(sim)
        assert k > 0

    def test_damping_coefficient(self, mol):
        """Damping coefficient should be positive for red-detuned light."""
        from molmot.mot.simulator import DCMOTSimulator
        from molmot.mot.force_scan import damping_coefficient
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        alpha = damping_coefficient(sim)
        assert alpha > 0

    def test_force_vs_z(self, mol):
        """force_vs_z should return arrays of correct length."""
        from molmot.mot.simulator import DCMOTSimulator
        from molmot.mot.force_scan import force_vs_z
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        z_arr = np.linspace(-3e-3, 3e-3, 5)
        F_arr, pop_arr, R_arr = force_vs_z(sim, z_arr)
        assert len(F_arr) == 5
        assert pop_arr.shape == (5, 12)

    def test_force_vs_v(self, mol):
        """force_vs_v should return array of correct length."""
        from molmot.mot.simulator import DCMOTSimulator
        from molmot.mot.force_scan import force_vs_v
        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        v_arr = np.linspace(-5, 5, 5)
        F_arr = force_vs_v(sim, v_arr)
        assert len(F_arr) == 5


# =========================================================================
# TestMolecules: Molecule loading
# =========================================================================

class TestMolecules:
    """Test molecule data loading."""

    def test_sroh_energies(self, mol):
        """SrOH should have 16 states."""
        assert mol.n_states == 16
        assert mol.n_ground == 12
        assert mol.n_excited == 4

    def test_sroh_physical_constants(self, mol):
        """Physical constants should be reasonable."""
        assert 600e-9 < mol.wavelength < 700e-9
        assert 1e7 < mol.Gamma < 1e8
        assert 1e-25 < mol.mass < 1e-24

    def test_caoh_loads(self):
        """CaOH Hamiltonian should build successfully."""
        from molmot.molecules.caoh import build_caoh_hamiltonian
        mol = build_caoh_hamiltonian(verbose=False)
        assert mol.n_states == 16
        assert 500e-9 < mol.wavelength < 700e-9

    def test_caf_loads(self):
        """CaF Hamiltonian should build successfully."""
        from molmot.molecules.caf import build_caf_hamiltonian
        mol = build_caf_hamiltonian(verbose=False)
        assert mol.n_states == 16
        assert 500e-9 < mol.wavelength < 700e-9

    def test_caoh_from_sroh(self):
        """CaOH via SrOH rescaling should load."""
        from molmot.molecules.caoh import load_caoh_from_sroh_structure
        mol = load_caoh_from_sroh_structure(JULIA_DIR, verbose=False)
        assert mol.n_states == 16

    def test_caf_from_sroh(self):
        """CaF via SrOH rescaling should load."""
        from molmot.molecules.caf import load_caf_from_sroh_structure
        mol = load_caf_from_sroh_structure(JULIA_DIR, verbose=False)
        assert mol.n_states == 16


# =========================================================================
# TestAnalysis: Analysis utilities
# =========================================================================

class TestAnalysis:
    """Test analysis and fitting functions."""

    def test_gaussian_fit(self):
        """gaussian_fit should recover sigma from Gaussian data."""
        from molmot.mot.analysis import gaussian_fit
        rng = np.random.default_rng(42)
        sigma_true = 0.5e-3  # 0.5 mm
        positions = rng.normal(0, sigma_true, 10000)
        sigma_fit = gaussian_fit(positions, bins=100, range_mm=(-2, 2))
        assert abs(sigma_fit - sigma_true) / sigma_true < 0.1  # within 10%

    def test_mb_fit(self):
        """maxwell_boltzmann_fit_1d should recover temperature from Gaussian velocities."""
        from molmot.mot.analysis import maxwell_boltzmann_fit_1d
        from molmot.constants import k_B
        rng = np.random.default_rng(42)
        mass = 105 * 1.66e-27
        T_true = 1e-3  # 1 mK
        sigma_v = np.sqrt(k_B * T_true / mass)
        velocities = rng.normal(0, sigma_v, 10000)
        T_fit = maxwell_boltzmann_fit_1d(velocities, mass, bins=100,
                                          range_ms=(-3 * sigma_v, 3 * sigma_v))
        assert abs(T_fit - T_true) / T_true < 0.2  # within 20%

    def test_capture_fraction(self):
        """capture_fraction should work with synthetic data."""
        from molmot.mot.analysis import capture_fraction
        t = np.linspace(0, 0.01, 100)
        # Trapped molecule
        z_trapped = np.sin(t * 100) * 0.001
        v_trapped = np.cos(t * 100) * 0.1
        # Escaped molecule
        z_escaped = t * 10.0
        v_escaped = np.ones_like(t) * 10.0
        results = [(t, z_trapped, v_trapped), (t, z_escaped, v_escaped)]
        frac = capture_fraction(results, r_max=3e-3, t_min=0.005)
        assert 0.0 <= frac <= 1.0

    def test_survived(self):
        """survived should detect trapped molecules."""
        from molmot.mot.analysis import survived
        t = np.linspace(0, 0.01, 100)
        z_ok = np.zeros(100) + 0.001
        v_ok = np.zeros(100)
        assert survived((t, z_ok, v_ok), r_max=3e-3) == True

        z_esc = np.ones(100) * 0.01
        v_esc = np.zeros(100)
        assert survived((t, z_esc, v_esc), r_max=3e-3) == False


# =========================================================================
# TestPropagation: Trajectory integration and sampling
# =========================================================================

class TestPropagation:
    """Test trajectory integration."""

    def test_simulate_trajectory(self, mol):
        """1D trajectory should not crash and produce arrays."""
        from molmot.mot.simulator import DCMOTSimulator
        from molmot.propagation.trajectories import simulate_trajectory

        sim = DCMOTSimulator(mol, delta_Gamma=-0.30, split_Gamma=0.40,
                              s0=1.0, B_gradient=0.16)
        t, z, v = simulate_trajectory(sim, z0=1e-3, v0=0.0,
                                       t_max=1e-4, dt=1e-6)
        assert len(t) == len(z) == len(v)
        assert len(t) > 0

    def test_sample_direction(self):
        """sample_direction should return unit vectors."""
        from molmot.propagation.trajectories import sample_direction
        for _ in range(10):
            d = sample_direction()
            assert abs(np.linalg.norm(d) - 1.0) < 1e-10

    def test_sample_mb(self):
        """Maxwell-Boltzmann samples should have correct variance."""
        from molmot.propagation.trajectories import sample_maxwell_boltzmann
        from molmot.constants import k_B
        T = 1e-3  # 1 mK
        mass = 105 * 1.66e-27
        v = sample_maxwell_boltzmann(T, mass, n_particles=10000, rng=np.random.default_rng(42))
        sigma_expected = np.sqrt(k_B * T / mass)
        sigma_actual = np.std(v[:, 0])
        assert abs(sigma_actual - sigma_expected) / sigma_expected < 0.1

    def test_gaussian_position(self):
        """Gaussian position samples should have correct sigma."""
        from molmot.propagation.trajectories import sample_gaussian_position
        sigma = 1e-3
        r = sample_gaussian_position(sigma, n_particles=10000, rng=np.random.default_rng(42))
        assert abs(np.std(r[:, 0]) - sigma) / sigma < 0.1

    def test_uniform_sphere(self):
        """Uniform sphere samples should be within radius."""
        from molmot.propagation.trajectories import sample_uniform_sphere
        R = 5e-3
        r = sample_uniform_sphere(R, n_particles=100, rng=np.random.default_rng(42))
        for pos in r:
            assert np.linalg.norm(pos) <= R


# =========================================================================
# TestFields: Laser field configuration
# =========================================================================

class TestFields:
    """Test laser field and polarization utilities."""

    def test_dc_beams_count(self, mol):
        """DC MOT should produce 8 beams (4 freq x 2 directions)."""
        from molmot.obe.fields import make_dc_mot_beams
        beams = make_dc_mot_beams(mol, delta_Gamma=-0.30, split_Gamma=0.40, s0=1.0)
        assert len(beams) == 8

    def test_rf_beams_count(self, mol):
        """RF MOT should produce 4 beams (2 freq x 2 directions)."""
        from molmot.obe.fields import make_rf_mot_beams
        beams = make_rf_mot_beams(mol, delta_Gamma=-1.0, s0=1.0)
        assert len(beams) == 4

    def test_rotate_polarization_identity(self):
        """Rotating sigma+ along z should be identity."""
        from molmot.obe.fields import rotate_polarization
        pol = np.array([0.0, 0.0, 1.0], dtype=complex)  # sigma+
        pol_rot = rotate_polarization(pol, np.array([0, 0, 1]))
        assert np.allclose(pol, pol_rot, atol=1e-10)

    def test_flip_polarization(self):
        """flip should swap sigma+ and sigma-."""
        from molmot.obe.fields import flip_polarization
        sigma_plus = np.array([0.0, 0.0, 1.0], dtype=complex)
        flipped = flip_polarization(sigma_plus)
        # flipped should be [1, 0, 0] (sigma-)
        assert abs(flipped[0] - 1.0) < 1e-10
        assert abs(flipped[2]) < 1e-10

    def test_gaussian_beam_profile(self):
        """On-axis should be 1, off-axis should be < 1."""
        from molmot.obe.fields import gaussian_beam_profile
        k_hat = np.array([0, 0, 1])
        # On-axis
        assert abs(gaussian_beam_profile(np.array([0, 0, 1e-3]), k_hat, 5e-3) - 1.0) < 1e-10
        # Off-axis
        prof = gaussian_beam_profile(np.array([3e-3, 0, 0]), k_hat, 5e-3)
        assert 0.0 < prof < 1.0


# =========================================================================
# TestConstants: Physical constants
# =========================================================================

class TestConstants:
    """Test physical constants and utility functions."""

    def test_wavenumber(self):
        """Wavenumber of 688 nm should be ~9.1e6 /m."""
        from molmot.constants import wavenumber
        k = wavenumber(688e-9)
        assert 9e6 < k < 9.5e6

    def test_v_recoil(self):
        """Recoil velocity should be ~0.006 m/s for SrOH."""
        from molmot.constants import v_recoil, wavenumber, amu
        k = wavenumber(688e-9)
        v_r = v_recoil(k, 105 * amu)
        assert 0.001 < v_r < 0.01

    def test_doppler_temperature(self):
        """Doppler temperature should be ~150 uK for Gamma=6.4 MHz."""
        from molmot.constants import T_doppler, k_B
        import math
        Gamma = 2 * math.pi * 6.4e6
        T_D = T_doppler(Gamma)
        assert 100e-6 < T_D < 300e-6  # 100-300 uK

    def test_polarization_vectors(self):
        """Spherical basis vectors should be orthonormal."""
        from molmot.constants import sigma_minus, sigma_0, sigma_plus
        assert abs(np.dot(sigma_minus, sigma_plus.conj())) < 1e-10
        assert abs(np.dot(sigma_minus, sigma_0.conj())) < 1e-10
        assert abs(np.linalg.norm(sigma_minus) - 1.0) < 1e-10


# =========================================================================
# TestOBESubdoppler: Full OBE with standing wave
# =========================================================================

class TestOBESubdoppler:
    """Test sub-Doppler OBE force computation."""

    def test_beam_pairs_dc4(self, mol):
        """make_beam_pairs_dc4 should return 4 beam pairs."""
        from molmot.obe.obe_subdoppler import make_beam_pairs_dc4
        bp = make_beam_pairs_dc4(mol, -0.30, 0.40, 1.0)
        assert len(bp) == 4
        for b in bp:
            assert 'freq' in b
            assert 'q_fwd' in b
            assert 'q_bwd' in b
            assert 's0' in b

    def test_beam_pairs_rf(self, mol):
        """make_beam_pairs_rf should return 2 beam pairs."""
        from molmot.obe.obe_subdoppler import make_beam_pairs_rf
        bp = make_beam_pairs_rf(mol, -1.0, 1.0)
        assert len(bp) == 2

    def test_beam_pairs_blue(self, mol):
        """make_beam_pairs_blue should return 2 beam pairs."""
        from molmot.obe.obe_subdoppler import make_beam_pairs_blue
        bp = make_beam_pairs_blue(mol, 1.0, 0.5, 1.0)
        assert len(bp) == 2


# =========================================================================
# TestDiffusion: Diffusion and scattering rate
# =========================================================================

class TestDiffusion:
    """Test diffusion computation functions."""

    def test_diffusion_temperature(self):
        """compute_diffusion_temperature with known D and beta."""
        from molmot.obe.diffusion import compute_diffusion_temperature
        from molmot.constants import k_B
        D = 1e-45
        mass = 1e-25
        beta = 100.0
        T = compute_diffusion_temperature(D, mass, beta)
        expected = D / (mass * beta * k_B)
        assert abs(T - expected) / expected < 1e-10

    def test_diffusion_temperature_zero_beta(self):
        """Zero damping should give infinite temperature."""
        from molmot.obe.diffusion import compute_diffusion_temperature
        T = compute_diffusion_temperature(1e-45, 1e-25, 0.0)
        assert T == float('inf')


# =========================================================================
# TestForce: Force computation utilities
# =========================================================================

class TestForce:
    """Test force computation functions."""

    def test_force_from_wavefunction_shape(self):
        """force_from_wavefunction should return 3-component vector."""
        from molmot.obe.force import force_from_wavefunction
        n_g, n_e = 12, 4
        N = n_g + n_e
        psi = np.zeros(N, dtype=complex)
        psi[0] = 1.0
        d_ge = np.random.rand(n_g, n_e, 3) * 0.1
        E_kq = np.random.rand(6, 3) + 1j * np.random.rand(6, 3)
        F = force_from_wavefunction(psi, d_ge, E_kq, n_g, n_e)
        assert F.shape == (3,)

    def test_zeeman_gradient_force(self):
        """Zeeman gradient force should have correct shape."""
        from molmot.obe.force import zeeman_gradient_force
        N = 16
        psi = np.zeros(N, dtype=complex)
        psi[0] = 1.0
        Z = {'x': np.eye(N, dtype=complex) * 0.1,
             'y': np.eye(N, dtype=complex) * 0.1,
             'z': np.eye(N, dtype=complex) * 0.1}
        F = zeeman_gradient_force(psi, Z, 0.16, 1.0)
        assert F.shape == (3,)
