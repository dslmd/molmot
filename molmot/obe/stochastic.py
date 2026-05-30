"""
Monte Carlo wavefunction / quantum jump (stochastic Schrodinger equation) solver.

Port of stochastic_schrodinger_equations_fast.jl from OpticalBlochEquations.jl
by Christian Hallas.

Algorithm (MCWF / quantum trajectories, "direct" method):
    1. Draw random number eps ~ U(0,1) as the jump threshold.
    2. Evolve psi under the non-Hermitian effective Hamiltonian
       H_eff = H_0 + H_laser - i(Gamma/2) sum_c L_c^dag L_c
    3. Do NOT normalize psi between jumps -- the norm decrease encodes
       the accumulated no-jump probability.
    4. Monitor dp = 1 - ||psi||^2.
    5. When dp > eps (equivalently ||psi||^2 < 1 - eps): quantum jump occurs.
       - Select a decay channel c with probability p_c = ||L_c psi||^2 / sum_c' ||L_c' psi||^2.
       - Apply jump: psi -> L_c psi / ||L_c psi||  (normalize ONLY here).
       - Add a random momentum kick (photon recoil in random direction).
       - Draw a new threshold eps ~ U(0,1).
    6. Position and velocity are evolved classically using the dipole
       force expectation value.

Units:
    Internal: natural units (time in 1/Gamma, position in 1/k, velocity in Gamma/k).
    External API: SI (seconds, metres, m/s). Conversion is performed at the
    boundaries of run() and in the result container.

References
----------
* Dalibard, Castin, Molmer, PRL 68, 580 (1992) -- quantum jump method
* Christian Hallas, OpticalBlochEquations.jl (stochastic_schrodinger_equations_fast.jl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..constants import hbar, k_B

try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None


# ---------------------------------------------------------------------------
# Module-level constants (avoid repeated construction in hot paths)
# ---------------------------------------------------------------------------

_X_HAT = np.array([1.0, 0.0, 0.0])
_Y_HAT = np.array([0.0, 1.0, 0.0])
_Z_HAT = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Polarization rotation utilities (port of rotate_pol / flip from Julia)
# ---------------------------------------------------------------------------

def _wigner_D_1(cos_beta, sin_beta, alpha, gamma):
    """
    Wigner D-matrix for j=1 with Euler angles (alpha, beta, gamma).

    Uses the ZYZ convention. Returns a 3x3 complex matrix mapping
    spherical-basis components (q=-1,0,+1) from the z-frame to the
    rotated frame.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cb, sb = cos_beta, sin_beta

    ea_p = np.exp(1j * alpha)
    ea_m = np.exp(-1j * alpha)
    eg_p = np.exp(1j * gamma)
    eg_m = np.exp(-1j * gamma)

    D = np.array([
        [0.5 * (1 + cb) * ea_m * eg_m,
         -sb / np.sqrt(2) * ea_m,
         0.5 * (1 - cb) * ea_m * eg_p],
        [sb / np.sqrt(2) * eg_m,
         cb,
         -sb / np.sqrt(2) * eg_p],
        [0.5 * (1 - cb) * ea_p * eg_m,
         sb / np.sqrt(2) * ea_p,
         0.5 * (1 + cb) * ea_p * eg_p],
    ], dtype=complex)
    return D


def _flip_polarization(eps):
    """
    Flip the polarization for a retro-reflected beam.

    For a beam propagating along -k, the handedness of circular
    polarization flips: sigma+ <-> sigma-.  In the spherical basis
    (q=-1, 0, +1), this is: eps[-1] <-> eps[+1], eps[0] -> -eps[0].
    """
    return np.array([eps[2], -eps[1], eps[0]], dtype=complex)


def rotate_polarization(pol, k_hat):
    """
    Rotate lab-frame polarization to the quantization axis frame.

    Given a polarization vector ``pol`` (in the spherical basis, 3-component)
    defined relative to the z-axis, rotate it to the frame where the
    quantization axis is aligned with ``k_hat``.

    Uses the Wigner D-matrix for j=1 with Euler angles computed from the
    axis-angle rotation that takes z-hat to k_hat.

    Parameters
    ----------
    pol : np.ndarray, shape (3,), complex
        Polarization in the spherical basis [q=-1, q=0, q=+1].
    k_hat : np.ndarray, shape (3,), float
        Unit vector for the beam propagation direction.

    Returns
    -------
    pol_rotated : np.ndarray, shape (3,), complex
    """
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    z_hat = np.array([0.0, 0.0, 1.0])
    cos_theta = np.clip(np.dot(k_hat, z_hat), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-12:
        return pol.copy()
    if abs(theta - np.pi) < 1e-12:
        alpha, beta, gamma = 0.0, np.pi, 0.0
    else:
        # Axis-angle from cross product k_hat x z_hat
        cross = np.cross(k_hat, z_hat)
        x, y, z = cross
        y = -y  # sign convention matching Julia code

        sin_theta = np.sin(theta)

        # Rotation matrix elements needed for Euler angle extraction
        A33 = (1 - np.cos(theta)) * z ** 2 + np.cos(theta)
        A31 = (1 - np.cos(theta)) * z * x - y * sin_theta
        A32 = (1 - np.cos(theta)) * z * y + x * sin_theta
        A13 = (1 - np.cos(theta)) * x * z + y * sin_theta
        A23 = (1 - np.cos(theta)) * y * z - x * sin_theta

        alpha = np.arctan2(A23, A13)
        beta = np.arctan2(np.sqrt(max(0, 1 - A33 ** 2)), A33)
        gamma = np.arctan2(A32, -A31)

    D = _wigner_D_1(np.cos(beta), np.sin(beta), alpha, gamma)
    return np.linalg.solve(D, pol)


def sample_direction_fast(rng):
    """Sample a uniformly random direction on the unit sphere."""
    cos_theta = 2.0 * rng.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = 2.0 * np.pi * rng.random()
    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta])


# ---------------------------------------------------------------------------
# Field computation (port of field_fast.jl)
# ---------------------------------------------------------------------------

def compute_fields_6beam(r, t, omega_states, omega_lasers, sats, pols,
                         beam_radius_k, k=1.0, cached_eps=None,
                         _E_kq_buf=None, _a_buf=None):
    """
    Compute electric field amplitudes for the standard 6-beam MOT geometry.

    Beams propagate along +x, +y, +z, -x, -y, -z. Each beam has
    ``n_freqs`` frequency components with specified saturation parameters
    and polarizations.

    Port of update_fields_fast! from field_fast.jl.

    Parameters
    ----------
    r : np.ndarray, shape (3,)
        Position in natural units (1/k).
    t : float
        Time in natural units (1/Gamma).
    omega_states : np.ndarray, shape (n_states,)
        State energies in natural units (rad * Gamma).
    omega_lasers : np.ndarray, shape (n_freqs,)
        Laser frequencies in natural units.
    sats : np.ndarray, shape (n_freqs,)
        Saturation parameters per frequency component.
    pols : list of np.ndarray, each shape (3,)
        Polarization vectors (spherical basis) for each frequency component.
    beam_radius_k : float
        Beam radius in units of 1/k: w * k, where w is the 1/e^2 radius.
    k : float
        Wavenumber (=1 in natural units).
    cached_eps : np.ndarray, optional, shape (6, n_freqs, 3)
        Pre-computed rotated polarizations. If None, computed fresh.
    _E_kq_buf : np.ndarray, optional, shape (6, 3)
        Pre-allocated output buffer for E_kq.
    _a_buf : np.ndarray, optional, shape (6, n_freqs)
        Pre-allocated buffer for amplitudes.

    Returns
    -------
    E_kq : np.ndarray, shape (6, 3), complex
        Field amplitude for each beam direction (6) and polarization component (3).
    a : np.ndarray, shape (6, n_freqs), complex
        Complex amplitude for each beam and frequency (for use in force/coupling).
    """
    n_freqs = len(omega_lasers)
    denom = beam_radius_k ** 2 / 2.0

    # Gaussian beam profile: exp(-r_perp^2 / w^2)
    r_perp_sq = np.array([
        r[1] ** 2 + r[2] ** 2,  # beam along x
        r[0] ** 2 + r[2] ** 2,  # beam along y
        r[0] ** 2 + r[1] ** 2,  # beam along z
    ])
    gauss = np.sqrt(np.exp(-r_perp_sq / denom))  # sqrt for amplitude

    # Build rotated polarizations for each beam direction
    if cached_eps is not None:
        eps = cached_eps
    else:
        eps = np.zeros((6, n_freqs, 3), dtype=complex)
        for i, pol in enumerate(pols):
            # D-matrix rotation handles helicity flip for backward beams
            eps[0, i, :] = rotate_polarization(pol, _X_HAT)
            eps[1, i, :] = rotate_polarization(pol, _Y_HAT)
            eps[2, i, :] = rotate_polarization(pol, _Z_HAT)
            eps[3, i, :] = rotate_polarization(pol, -_X_HAT)
            eps[4, i, :] = rotate_polarization(pol, -_Y_HAT)
            eps[5, i, :] = rotate_polarization(pol, -_Z_HAT)

    # Phase: exp(i(-k.r + omega*t)) for each beam direction and frequency
    kr = k * r[:3]

    # Amplitude coefficients: a[beam_dir, freq] = sqrt(s)/(2*sqrt(2)) * gauss * exp(i*phase)
    if _a_buf is not None:
        a = _a_buf
        a[:] = 0.0
    else:
        a = np.zeros((6, n_freqs), dtype=complex)

    for f in range(n_freqs):
        G = np.sqrt(sats[f]) / (2.0 * np.sqrt(2.0))
        omega_t = omega_lasers[f] * t

        for axis in range(3):
            # Forward beam (+k direction): phase = -kr + omega*t
            phase_fwd = -kr[axis] + omega_t
            # Backward beam (-k direction): phase = +kr + omega*t
            phase_bwd = kr[axis] + omega_t

            a[axis, f] = G * gauss[axis] * np.exp(1j * phase_fwd)
            a[axis + 3, f] = G * gauss[axis] * np.exp(1j * phase_bwd)

    # E_kq[beam, q] = sum_f a[beam, f] * conj(eps[beam, f, q])
    if _E_kq_buf is not None:
        E_kq = _E_kq_buf
        E_kq[:] = 0.0
    else:
        E_kq = np.zeros((6, 3), dtype=complex)

    for q in range(3):
        for beam in range(6):
            for f in range(n_freqs):
                E_kq[beam, q] += a[beam, f] * np.conj(eps[beam, f, q])

    return E_kq, a


# ---------------------------------------------------------------------------
# SSEProblem: problem specification
# ---------------------------------------------------------------------------

class SSEProblem:
    """
    Stochastic Schrodinger Equation problem setup.

    Manages the non-Hermitian Hamiltonian evolution with quantum jumps
    for a single molecule moving in laser fields + magnetic field.

    The internal simulation is performed in *natural units*:
        time: 1/Gamma
        position: 1/k
        velocity: Gamma/k
        energy: hbar*Gamma
        force: hbar*k*Gamma

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data (from sroh.py).
    beam_pairs : list of dict
        Beam pair configurations from obe_subdoppler.py make_beam_pairs_*.
        Each dict has keys: 'freq' (Hz), 'q_fwd' (0/1/2), 'q_bwd' (0/1/2), 's0' (float).
    B_gradient : float
        Magnetic field gradient (T/m).
    Omega_Gamma : float
        Peak Rabi frequency in units of Gamma per beam.
        If sats are specified in beam_pairs, this can be set to 1.0 and the
        saturation parameters control the intensity.
    Gamma_eff_factor : float
        Effective scattering rate reduction factor (for repumping losses).
    beam_radius : float
        1/e^2 beam radius in metres.
    add_spontaneous_kick : bool
        Whether to add random photon recoil kicks at each quantum jump.
    diffusion_constants : np.ndarray, shape (3,), optional
        Additional momentum diffusion constants [Dx, Dy, Dz] in SI units.
        If provided, a random momentum kick proportional to
        sqrt(2*D*dt_since_last_jump) is added at each jump.
    """

    def __init__(self, mol_data, beam_pairs, B_gradient, Omega_Gamma=1.0,
                 Gamma_eff_factor=1.0, beam_radius=10e-3,
                 add_spontaneous_kick=True, diffusion_constants=None):
        self.mol_data = mol_data
        self.beam_pairs = beam_pairs
        self.B_gradient = B_gradient
        self.Omega_Gamma = Omega_Gamma
        self.Gamma_eff_factor = Gamma_eff_factor
        self.beam_radius = beam_radius
        self.add_spontaneous_kick = add_spontaneous_kick

        # Physical parameters
        self.Gamma = mol_data.Gamma  # rad/s
        self.k = mol_data.k          # 1/m
        self.mass = mol_data.mass     # kg
        self.wavelength = mol_data.wavelength

        self.n_ground = mol_data.n_ground
        self.n_excited = mol_data.n_excited
        self.n_states = mol_data.n_states

        # Photon recoil velocity in SI
        self.v_recoil_SI = hbar * self.k / self.mass  # m/s

        # Conversion factors: SI <-> natural
        self.t_nat = 1.0 / self.Gamma                     # 1 natural time unit in seconds
        self.r_nat = 1.0 / self.k                          # 1 natural length unit in metres
        self.v_nat = self.Gamma / self.k                   # 1 natural velocity unit in m/s
        self.F_nat = hbar * self.k * self.Gamma            # 1 natural force unit in N
        self.v_recoil_nat = self.v_recoil_SI / self.v_nat  # recoil in natural units

        # Natural mass: m_nat = mass * Gamma / (hbar * k^2)
        self.mass_nat = self.mass * self.Gamma / (hbar * self.k ** 2)

        # Diffusion constants (convert to natural units)
        if diffusion_constants is not None:
            self.diffusion_nat = np.asarray(diffusion_constants) / (
                (hbar * self.k) ** 2 * self.Gamma)
        else:
            self.diffusion_nat = np.zeros(3)

        # Pre-compute transition dipole matrix in the format needed
        d = mol_data.tdm  # (n_states, n_states, 3) complex
        n_g = self.n_ground
        n_e = self.n_excited

        # BUG FIX 6: Use absolute value, not negative real part
        # d_ge[ig, ie, q] = |d[ig, n_g + ie, q]|
        # The coupling strength is |d|, not -Re(d).
        self.d_ge = np.zeros((n_g, n_e, 3))
        for ig in range(n_g):
            for ie in range(n_e):
                for q in range(3):
                    self.d_ge[ig, ie, q] = np.abs(d[ig, n_g + ie, q])

        self.d_eg = np.transpose(self.d_ge, (1, 0, 2))  # (n_e, n_g, 3)

        # Branching ratios: BR[ie, ig, q] = |d[ig, n_g+ie, q]|^2 / sum_g',q' |d[ig', n_g+ie, q']|^2
        self.branching = np.zeros((n_e, n_g, 3))
        for ie in range(n_e):
            total = 0.0
            for ig in range(n_g):
                for q in range(3):
                    total += self.d_ge[ig, ie, q] ** 2
            if total > 1e-30:
                for ig in range(n_g):
                    for q in range(3):
                        self.branching[ie, ig, q] = self.d_ge[ig, ie, q] ** 2 / total

        # State energies in natural units (rad * Gamma -> divide by Gamma)
        # energies are in Hz; convert to angular frequency / Gamma
        self.omega0s = 2.0 * np.pi * mol_data.energies / self.Gamma

        # Zeeman shift coefficients
        self.zeeman_z_diag = mol_data.zeeman_z_diag  # shifts per Gauss * Gamma/(2pi)

        # ---- UPGRADE 1: Full 3D Zeeman Hamiltonian matrices ----
        # Store the ground-state Zeeman matrices (12x12 blocks) as numpy arrays.
        # The Zeeman matrices in mol_data are pre-scaled such that:
        #   E_zeeman(Hz) = B(Gauss) * diag(zeeman_z) * Gamma/(2*pi)
        # We need them in natural angular frequency units (rad * Gamma):
        #   omega_zeeman = 2*pi * E_zeeman / Gamma = 2*pi * B(G) * Z_matrix * Gamma/(2*pi) / Gamma
        #                = B(G) * Z_matrix
        # So the Zeeman matrices are already in the right scaling:
        # multiply by B(Gauss) to get the Zeeman shift in natural angular frequency units.
        self.zeeman_x_ground = np.array(mol_data.zeeman_x[:n_g, :n_g], dtype=complex)
        self.zeeman_y_ground = np.array(mol_data.zeeman_y[:n_g, :n_g], dtype=complex)
        self.zeeman_z_ground = np.array(mol_data.zeeman_z[:n_g, :n_g], dtype=complex)

        # Beam parameters: convert to natural units
        self._setup_beam_params()

        # Beam radius in natural units
        self.beam_radius_k = beam_radius * self.k

        # State vector layout (SIMPLIFIED -- no integrated pop, no force storage):
        # [psi_re(N), psi_im(N), rx, ry, rz, vx, vy, vz]
        self.n_psi = self.n_states
        self.r_idx = 2 * self.n_states
        self.v_idx = self.r_idx + 3
        self.state_size = 2 * self.n_states + 3 + 3

    def _setup_beam_params(self):
        """Convert beam_pairs to the internal format for the field computation."""
        n_freqs = len(self.beam_pairs)

        self.omega_lasers = np.zeros(n_freqs)
        self.sats = np.zeros(n_freqs)
        self.pols = []

        for i, bp in enumerate(self.beam_pairs):
            # freq is in Hz; convert to natural units
            self.omega_lasers[i] = 2.0 * np.pi * bp['freq'] / self.Gamma

            # Saturation parameter
            self.sats[i] = bp['s0']

            # Polarization: q_fwd specifies the spherical component
            pol = np.zeros(3, dtype=complex)
            q_fwd = bp['q_fwd']
            pol[q_fwd] = 1.0
            self.pols.append(pol)

    def build_initial_state(self, r0_SI, v0_SI, psi0=None):
        """
        Build the initial state vector.

        Parameters
        ----------
        r0_SI : np.ndarray, shape (3,)
            Initial position in metres.
        v0_SI : np.ndarray, shape (3,)
            Initial velocity in m/s.
        psi0 : np.ndarray, shape (n_states,), complex, optional
            Initial wavefunction. Default: ground state (first state).

        Returns
        -------
        u0 : np.ndarray, shape (state_size,)
            Initial state vector.
        """
        N = self.n_states

        u0 = np.zeros(self.state_size)

        # Wavefunction (real and imaginary parts)
        if psi0 is None:
            u0[0] = 1.0  # first ground state
        else:
            psi0 = np.asarray(psi0, dtype=complex)
            psi0 /= np.linalg.norm(psi0)
            u0[:N] = np.real(psi0)
            u0[N:2 * N] = np.imag(psi0)

        # Position and velocity in natural units
        r0_nat = np.asarray(r0_SI) / self.r_nat
        v0_nat = np.asarray(v0_SI) / self.v_nat

        u0[self.r_idx:self.r_idx + 3] = r0_nat
        u0[self.v_idx:self.v_idx + 3] = v0_nat

        return u0

    def get_omega0s_zeeman(self, r_nat):
        """
        Return state energies including DIAGONAL Zeeman shift at position r.

        Uses the full 3D quadrupole B-field for the diagonal part:
            B = B'(-x/2, -y/2, z) (anti-Helmholtz)

        The diagonal Zeeman shift uses the diagonal of the full Zeeman
        Hamiltonian H_Z = Bx*Zx + By*Zy + Bz*Zz.

        Parameters
        ----------
        r_nat : np.ndarray, shape (3,)
            Position in natural units (1/k).

        Returns
        -------
        omega0s : np.ndarray, shape (n_states,)
            State energies in natural units (including diagonal Zeeman).
        """
        # Position in SI
        r_SI = r_nat * self.r_nat  # metres

        # Full 3D quadrupole B-field (Gauss)
        B_x_G = -self.B_gradient * r_SI[0] / 2.0 * 1e4
        B_y_G = -self.B_gradient * r_SI[1] / 2.0 * 1e4
        B_z_G =  self.B_gradient * r_SI[2] * 1e4

        # Diagonal Zeeman shifts from the full Hamiltonian
        # H_Z = Bx*Zx + By*Zy + Bz*Zz
        # The diagonal of H_Z gives the first-order energy shifts per state.
        # zeeman_z_diag already has the right scaling for all states (ground+excited).
        # For the ground block, we use the full matrix diagonals.
        n_g = self.n_ground
        omega_zeeman = np.zeros(self.n_states)

        # Ground states: diagonal of full H_Z
        omega_zeeman[:n_g] = np.real(
            B_x_G * np.diag(self.zeeman_x_ground) +
            B_y_G * np.diag(self.zeeman_y_ground) +
            B_z_G * np.diag(self.zeeman_z_ground)
        )

        # Excited states: g_J ~ 0 for Pi_1/2, but use Bz diagonal for completeness
        omega_zeeman[n_g:] = B_z_G * self.zeeman_z_diag[n_g:]

        return self.omega0s + omega_zeeman

    def get_B_field_gauss(self, r_nat):
        """
        Return the 3D quadrupole B-field at position r in Gauss.

        B = B'(-x/2, -y/2, z) for anti-Helmholtz coils.

        Parameters
        ----------
        r_nat : np.ndarray, shape (3,)
            Position in natural units (1/k).

        Returns
        -------
        B_gauss : np.ndarray, shape (3,)
            B-field vector in Gauss.
        """
        r_SI = r_nat * self.r_nat  # metres
        return np.array([
            -self.B_gradient * r_SI[0] / 2.0 * 1e4,
            -self.B_gradient * r_SI[1] / 2.0 * 1e4,
             self.B_gradient * r_SI[2] * 1e4,
        ])


# ---------------------------------------------------------------------------
# SSEResult: trajectory result container
# ---------------------------------------------------------------------------

@dataclass
class SSEResult:
    """Container for a single SSE trajectory result."""
    times: np.ndarray              # (n_save,) -- seconds
    positions: np.ndarray          # (n_save, 3) -- metres
    velocities: np.ndarray         # (n_save, 3) -- m/s
    populations: np.ndarray        # (n_save, n_states) -- |psi_i|^2
    photons_scattered: int = 0
    jump_times: list = field(default_factory=list)    # seconds
    jump_channels: list = field(default_factory=list) # (ie, ig, q) tuples
    forces: Optional[np.ndarray] = None               # (n_save, 3) -- Newtons


# ---------------------------------------------------------------------------
# SSESolver: the main trajectory integrator
# ---------------------------------------------------------------------------

class SSESolver:
    """
    Solve the SSE for a single molecular trajectory using the quantum jump method.

    Correct MCWF algorithm (direct method):
        1. Draw eps ~ U(0,1) as the jump threshold.
        2. Evolve psi under H_eff (non-Hermitian) -- do NOT normalize.
        3. At each step, check dp = 1 - ||psi||^2.
        4. When dp > eps: quantum jump.
           - Select decay channel c with p_c = ||L_c psi||^2 / sum_c' ||L_c' psi||^2.
           - Apply jump: psi -> L_c psi / ||L_c psi|| (normalize ONLY here).
           - Add random photon recoil kick.
           - Draw new threshold eps ~ U(0,1).
        5. Position: dr/dt = v.
        6. Velocity: dv/dt = F/m.

    Parameters
    ----------
    problem : SSEProblem
        The problem specification.
    """

    def __init__(self, problem: SSEProblem):
        self.prob = problem
        N = problem.n_states
        n_g = problem.n_ground
        n_e = problem.n_excited
        n_freqs = len(problem.omega_lasers)

        # ------ MEMORY FIX: Pre-allocate ALL workspace arrays ------
        self._du = np.zeros(problem.state_size)
        self._psi = np.zeros(N, dtype=complex)
        self._dpsi = np.zeros(N, dtype=complex)
        self._dpsi_int = np.zeros(N, dtype=complex)
        self._eiw0t = np.zeros(N, dtype=complex)
        self._eiw0t_conj = np.zeros(N, dtype=complex)
        self._psi_int = np.zeros(N, dtype=complex)
        self._psi_q_g = np.zeros((n_g, 3), dtype=complex)
        self._psi_q_e = np.zeros((n_e, 3), dtype=complex)
        self._d_exp = np.zeros(3, dtype=complex)
        self._F = np.zeros(3)
        self._omega0s_buf = np.zeros(N)

        # RK4 workspace
        self._k1 = np.zeros(problem.state_size)
        self._k2 = np.zeros(problem.state_size)
        self._k3 = np.zeros(problem.state_size)
        self._k4 = np.zeros(problem.state_size)
        self._u_temp = np.zeros(problem.state_size)

        # Field workspace
        self._E_kq_buf = np.zeros((6, 3), dtype=complex)
        self._a_buf = np.zeros((6, n_freqs), dtype=complex)

        # UPGRADE 1: Workspace for full Zeeman Hamiltonian
        self._H_Z_ground = np.zeros((n_g, n_g), dtype=complex)
        self._dpsi_zeeman = np.zeros(n_g, dtype=complex)
        self._B_gauss = np.zeros(3)

        # Pre-compute constant beam polarizations (computed once, not per _rhs call)
        self._cached_eps = self._compute_cached_polarizations()

    def _compute_cached_polarizations(self):
        """
        Pre-compute rotated polarizations for all 6 beam directions.

        The Wigner D rotation from lab-frame polarization to each beam
        direction ALREADY handles the helicity reversal for backward beams.
        Do NOT apply _flip_polarization — that causes a double-flip.
        """
        p = self.prob
        n_freqs = len(p.omega_lasers)
        eps = np.zeros((6, n_freqs, 3), dtype=complex)
        for i, pol in enumerate(p.pols):
            eps[0, i, :] = rotate_polarization(pol, _X_HAT)    # +x
            eps[1, i, :] = rotate_polarization(pol, _Y_HAT)    # +y
            eps[2, i, :] = rotate_polarization(pol, _Z_HAT)    # +z (identity)
            eps[3, i, :] = rotate_polarization(pol, -_X_HAT)   # -x (includes helicity flip)
            eps[4, i, :] = rotate_polarization(pol, -_Y_HAT)   # -y
            eps[5, i, :] = rotate_polarization(pol, -_Z_HAT)   # -z (maps σ⁺↔σ⁻)
        return eps

    def run(self, r0, v0, t_max, dt=None, psi0=None,
            save_every=100, rng_seed=None):
        """
        Run a single molecular trajectory.

        Parameters
        ----------
        r0 : array-like, shape (3,)
            Initial position in metres.
        v0 : array-like, shape (3,)
            Initial velocity in m/s.
        t_max : float
            Total simulation time in seconds.
        dt : float, optional
            Time step in seconds. Default: 0.01 / Gamma.
        psi0 : np.ndarray, optional
            Initial wavefunction. Default: first ground state.
        save_every : int
            Save state every N steps.
        rng_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        result : SSEResult
        """
        p = self.prob
        rng = np.random.default_rng(rng_seed)

        # Default time step
        if dt is None:
            dt_SI = 0.01 / p.Gamma
        else:
            dt_SI = dt
        dt_nat = dt_SI / p.t_nat

        # Total steps
        t_max_nat = t_max / p.t_nat
        n_steps = int(t_max_nat / dt_nat)

        # Initialize state
        r0 = np.asarray(r0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        u = p.build_initial_state(r0, v0, psi0)

        N = p.n_states
        n_g = p.n_ground
        n_e = p.n_excited

        # Quantum jump threshold (direct method): draw eps ~ U(0,1).
        # A jump occurs when the (un-normalized) survival probability
        # ||psi||^2 drops below eps, i.e. when dp = 1 - ||psi||^2 > 1 - eps.
        # Since eps is uniform on (0,1), so is (1 - eps); we therefore compare
        # dp against a uniform threshold.  (Using -ln(r) here would be the
        # *waiting-time* convention, which must be compared against
        # -ln(||psi||^2), NOT against 1 - ||psi||^2.  Mixing the two makes the
        # jump condition unreachable whenever -ln(r) > 1 -- ~37% of draws --
        # which silently freezes those trajectories.)
        threshold = rng.random()

        # Storage
        n_save = n_steps // save_every + 1
        times_save = np.zeros(n_save)
        positions_save = np.zeros((n_save, 3))
        velocities_save = np.zeros((n_save, 3))
        populations_save = np.zeros((n_save, N))
        forces_save = np.zeros((n_save, 3))

        jump_times = []
        jump_channels = []
        n_scatters = 0
        last_decay_time = 0.0

        save_idx = 0
        t_current = 0.0

        for step in range(n_steps):
            # Save state
            if step % save_every == 0 and save_idx < n_save:
                psi_re = u[:N]
                psi_im = u[N:2 * N]
                norm_sq = np.sum(psi_re ** 2 + psi_im ** 2)
                # Report populations normalized (for visualization)
                if norm_sq > 1e-30:
                    pops = (psi_re ** 2 + psi_im ** 2) / norm_sq
                else:
                    pops = psi_re ** 2 + psi_im ** 2

                times_save[save_idx] = t_current * p.t_nat
                positions_save[save_idx] = u[p.r_idx:p.r_idx + 3] * p.r_nat
                velocities_save[save_idx] = u[p.v_idx:p.v_idx + 3] * p.v_nat
                populations_save[save_idx] = pops
                save_idx += 1

            # RK4 step (does NOT normalize -- the norm decrease IS the physics)
            u = self._rk4_step(u, t_current, dt_nat)
            t_current += dt_nat

            # Check for quantum jump via norm decrease
            dp = self._check_jump(u)
            if dp > threshold:
                u, channel = self._do_jump(u, rng)
                if channel is not None:
                    n_scatters += 1
                    jump_times.append(t_current * p.t_nat)
                    jump_channels.append(channel)

                # Add diffusion kick
                dt_since_decay = (t_current - last_decay_time) * p.t_nat
                for i in range(3):
                    if p.diffusion_nat[i] > 0:
                        kick_sigma = np.sqrt(
                            2 * p.diffusion_nat[i] * (t_current - last_decay_time))
                        u[p.v_idx + i] += rng.choice([-1, 1]) * kick_sigma

                last_decay_time = t_current

                # Draw new threshold (uniform; see note at first draw)
                threshold = rng.random()

        # Final save
        if save_idx < n_save:
            psi_re = u[:N]
            psi_im = u[N:2 * N]
            norm_sq = np.sum(psi_re ** 2 + psi_im ** 2)
            if norm_sq > 1e-30:
                pops = (psi_re ** 2 + psi_im ** 2) / norm_sq
            else:
                pops = psi_re ** 2 + psi_im ** 2

            times_save[save_idx] = t_current * p.t_nat
            positions_save[save_idx] = u[p.r_idx:p.r_idx + 3] * p.r_nat
            velocities_save[save_idx] = u[p.v_idx:p.v_idx + 3] * p.v_nat
            populations_save[save_idx] = pops
            save_idx += 1

        # Trim arrays
        times_save = times_save[:save_idx]
        positions_save = positions_save[:save_idx]
        velocities_save = velocities_save[:save_idx]
        populations_save = populations_save[:save_idx]

        return SSEResult(
            times=times_save,
            positions=positions_save,
            velocities=velocities_save,
            populations=populations_save,
            photons_scattered=n_scatters,
            jump_times=jump_times,
            jump_channels=jump_channels,
            forces=None,
        )

    def _rhs(self, t, u):
        """
        Right-hand side for the ODE: du/dt = f(t, u).

        Works with UN-normalized psi. The norm decrease encodes the
        accumulated no-jump probability.

        State vector layout (all in natural units):
            u[0      : N]        = psi_re
            u[N      : 2*N]      = psi_im
            u[r_idx  : r_idx+3]  = position (rx, ry, rz)
            u[v_idx  : v_idx+3]  = velocity (vx, vy, vz)

        Returns
        -------
        du : np.ndarray, same shape as u
        """
        p = self.prob
        N = p.n_states
        n_g = p.n_ground
        n_e = p.n_excited

        du = self._du
        du[:] = 0.0

        # Extract psi (DO NOT NORMALIZE -- this is the key MCWF fix)
        psi = self._psi
        psi[:] = u[:N] + 1j * u[N:2*N]

        # Position and velocity
        r = u[p.r_idx:p.r_idx + 3]
        v = u[p.v_idx:p.v_idx + 3]

        # State energies with Zeeman shift
        omega0s = p.get_omega0s_zeeman(r)

        # Interaction picture transformation: psi -> psi * exp(-i*omega0*t)
        np.exp(-1j * omega0s * t, out=self._eiw0t)
        np.multiply(psi, self._eiw0t, out=self._psi_int)

        psi_g_int = self._psi_int[:n_g]
        psi_e_int = self._psi_int[n_g:]

        # Compute psi_q = d @ psi (for ground and excited parts)
        # psi_q for ground states: sum_e d_ge[g,e,q] * psi_e
        np.einsum('geq,e->gq', p.d_ge, psi_e_int, out=self._psi_q_g)
        # psi_q for excited states: sum_g d_eg[e,g,q] * psi_g
        np.einsum('egq,g->eq', p.d_eg, psi_g_int, out=self._psi_q_e)

        # Compute electric fields (using cached polarizations and pre-allocated buffers)
        E_kq, a = compute_fields_6beam(
            r, t, omega0s, p.omega_lasers, p.sats, p.pols,
            p.beam_radius_k, cached_eps=self._cached_eps,
            _E_kq_buf=self._E_kq_buf, _a_buf=self._a_buf)

        # Total field
        E_total = np.sum(E_kq, axis=0)

        # Compute dipole expectation for force
        # d_exp[q] = sum_g conj(psi_g_int) * psi_q_g[g,q]
        np.einsum('g,gq->q', np.conj(psi_g_int), self._psi_q_g, out=self._d_exp)

        # Force
        F = self._F
        F[:] = 0.0
        for k_dir in range(3):
            F_k = 0.0
            for q in range(3):
                E_diff_re = np.real(E_kq[k_dir, q]) - np.real(E_kq[k_dir + 3, q])
                E_diff_im = np.imag(E_kq[k_dir, q]) - np.imag(E_kq[k_dir + 3, q])
                # Multiply by -i
                Ekq_re = E_diff_im
                Ekq_im = -E_diff_re
                d_re = np.real(self._d_exp[q])
                d_im = np.imag(self._d_exp[q])
                F_k -= 2.0 * (d_re * Ekq_re - d_im * Ekq_im)
            F[k_dir] = F_k

        # dpsi/dt from laser coupling (using UN-normalized psi consistently)
        dpsi_int = self._dpsi_int
        dpsi_int[:] = 0.0

        # Ground states: dpsi_g = -i * sum_q E_total[q] * psi_q_g[g,q]
        for ig in range(n_g):
            val = 0.0 + 0.0j
            for q in range(3):
                val += E_total[q] * self._psi_q_g[ig, q]
            # Multiply by -i: (-i)(a+ib) = b - ia
            dpsi_int[ig] = complex(np.imag(val), -np.real(val))

        # Excited states: dpsi_e = -i * sum_q conj(E_total[q]) * psi_q_e[e,q]
        for ie in range(n_e):
            val = 0.0 + 0.0j
            for q in range(3):
                val += np.conj(E_total[q]) * self._psi_q_e[ie, q]
            dpsi_int[n_g + ie] = complex(np.imag(val), -np.real(val))

        # Non-Hermitian decay: -Gamma/2 * psi_e for excited states
        # In natural units where Gamma=1: coefficient is 0.5
        for ie in range(n_e):
            dpsi_int[n_g + ie] -= 0.5 * psi_e_int[ie]

        # ---- UPGRADE 1: Full off-diagonal Zeeman coupling ----
        # The full Zeeman Hamiltonian H_Z = Bx*Zx + By*Zy + Bz*Zz
        # has OFF-DIAGONAL elements that mix different mF ground states.
        # In the interaction picture, the off-diagonal terms oscillate:
        #   dpsi_Z_int[ig] = -i * sum_jg H_Z[ig,jg] * exp(-i*(w_ig - w_jg)*t) * psi_jg_int
        # The diagonal part is already included in omega0s (via get_omega0s_zeeman).
        # Here we add ONLY the off-diagonal Zeeman coupling.
        B_gauss = p.get_B_field_gauss(r)
        B_x_G, B_y_G, B_z_G = B_gauss[0], B_gauss[1], B_gauss[2]

        # Build full ground-state Zeeman Hamiltonian (in natural freq units)
        H_Z_g = self._H_Z_ground
        H_Z_g[:] = (B_x_G * p.zeeman_x_ground +
                     B_y_G * p.zeeman_y_ground +
                     B_z_G * p.zeeman_z_ground)

        # Apply off-diagonal Zeeman in the interaction picture (vectorized)
        # H_Z_int[ig,jg] = H_Z[ig,jg] * exp(-i*(omega_ig - omega_jg)*t)
        # dpsi_Z = -i * H_Z_int @ psi_g_int  (off-diagonal only)
        omega_g = omega0s[:n_g]
        # Phase matrix: exp(-i*(omega_i - omega_j)*t) for all pairs
        phase_vec = np.exp(-1j * omega_g * t)  # exp(-i*omega_i*t)
        # H_Z_int[i,j] = H_Z[i,j] * exp(-i*omega_i*t) * exp(+i*omega_j*t)
        #              = H_Z[i,j] * phase_vec[i] * conj(phase_vec[j])
        H_Z_int = H_Z_g * np.outer(phase_vec, np.conj(phase_vec))
        # Zero the diagonal (already in omega0s)
        np.fill_diagonal(H_Z_int, 0.0)
        # dpsi_Z = -i * H_Z_int @ psi_g_int
        val_vec = H_Z_int @ psi_g_int
        # Multiply by -i: (-i)(a+ib) = b - ia
        dpsi_int[:n_g] += np.imag(val_vec) - 1j * np.real(val_vec)

        # Transform back from interaction picture
        np.exp(1j * omega0s * t, out=self._eiw0t_conj)
        np.multiply(dpsi_int, self._eiw0t_conj, out=self._dpsi)

        # Write derivatives
        du[:N] = self._dpsi.real
        du[N:2 * N] = self._dpsi.imag

        # Position: dr/dt = v
        du[p.r_idx:p.r_idx + 3] = v

        # Velocity: dv/dt = F/m_nat
        du[p.v_idx:p.v_idx + 3] = F / p.mass_nat

        return du

    def _rk4_step(self, u, t, dt):
        """
        Perform one RK4 step.

        CRITICAL: Do NOT normalize psi after the step. The norm decrease
        encodes the accumulated no-jump probability in the MCWF algorithm.
        """
        k1 = self._rhs(t, u)
        self._k1[:] = k1

        self._u_temp[:] = u + 0.5 * dt * self._k1
        k2 = self._rhs(t + 0.5 * dt, self._u_temp)
        self._k2[:] = k2

        self._u_temp[:] = u + 0.5 * dt * self._k2
        k3 = self._rhs(t + 0.5 * dt, self._u_temp)
        self._k3[:] = k3

        self._u_temp[:] = u + dt * self._k3
        k4 = self._rhs(t + dt, self._u_temp)
        self._k4[:] = k4

        u_new = u + (dt / 6.0) * (self._k1 + 2 * self._k2 + 2 * self._k3 + self._k4)

        # DO NOT NORMALIZE -- the norm decrease IS the physics
        return u_new

    def _check_jump(self, u):
        """
        Check if a quantum jump should occur.

        Returns dp = 1 - ||psi||^2, which is the accumulated probability
        of a decay event having occurred.
        """
        N = self.prob.n_states
        psi_re = u[:N]
        psi_im = u[N:2*N]
        norm_sq = np.sum(psi_re**2 + psi_im**2)
        dp = 1.0 - norm_sq
        return dp

    def _do_jump(self, u, rng):
        """
        Perform quantum jump and normalize.

        Selects a decay channel (ie, ig, q) with probability proportional
        to ||L_c psi||^2, then projects psi onto the resulting ground state
        and normalizes.

        Returns
        -------
        u : np.ndarray
            Updated state vector (psi normalized after jump).
        channel : tuple (ie, ig, q) or None
            The selected decay channel, or None if no excited population.
        """
        p = self.prob
        N = p.n_states
        n_g = p.n_ground
        n_e = p.n_excited

        psi = u[:N] + 1j * u[N:2*N]

        # Compute ||L_c psi||^2 for each decay channel
        # L_c = sqrt(Gamma * BR) |g><e|, so L_c psi = sqrt(Gamma * BR) * psi_e * |g>
        # ||L_c psi||^2 = Gamma * BR * |psi_e|^2
        # In natural units Gamma=1, so ||L_c psi||^2 = BR * |psi_e|^2
        probs = []
        channels = []
        for ie in range(n_e):
            psi_e = psi[n_g + ie]
            pe = abs(psi_e)**2
            if pe < 1e-30:
                continue
            for ig in range(n_g):
                for q in range(3):
                    br = p.branching[ie, ig, q]
                    if br < 1e-15:
                        continue
                    prob = br * pe
                    probs.append(prob)
                    channels.append((ie, ig, q))

        if len(probs) == 0:
            # No excited population -- reset to ground state
            psi_new = np.zeros(N, dtype=complex)
            psi_new[0] = 1.0
            u[:N] = psi_new.real
            u[N:2*N] = psi_new.imag
            return u, None

        probs = np.array(probs)
        probs /= probs.sum()

        idx = rng.choice(len(probs), p=probs)
        ie, ig, q = channels[idx]

        # Project: psi_new = L_c psi / ||L_c psi||
        # L_c |psi> = sqrt(BR * Gamma) * psi_e[ie] * |ig>
        # After normalization this is just |ig>
        psi_new = np.zeros(N, dtype=complex)
        psi_new[ig] = 1.0  # pure ground state

        u[:N] = psi_new.real
        u[N:2*N] = psi_new.imag

        # Random photon recoil kick
        if p.add_spontaneous_kick:
            kick_dir = sample_direction_fast(rng)
            u[p.v_idx:p.v_idx + 3] += p.v_recoil_nat * kick_dir

        return u, (ie, ig, q)


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble(problem, n_particles, r0_sampler, v0_sampler,
                 t_max, dt=None, save_every=100, rng_seed=None,
                 callback=None):
    """
    Run an ensemble of SSE trajectories.

    Parameters
    ----------
    problem : SSEProblem
    n_particles : int
        Number of trajectories to run.
    r0_sampler : callable() -> np.ndarray, shape (3,)
        Returns an initial position in metres.
    v0_sampler : callable() -> np.ndarray, shape (3,)
        Returns an initial velocity in m/s.
    t_max : float
        Simulation time in seconds.
    dt : float, optional
        Time step.
    save_every : int
        Save interval.
    rng_seed : int, optional
        Base random seed. Each trajectory uses seed + i.
    callback : callable(i, result), optional
        Called after each trajectory completes.

    Returns
    -------
    results : list of SSEResult
    """
    solver = SSESolver(problem)
    results = []

    for i in range(n_particles):
        r0 = r0_sampler()
        v0 = v0_sampler()
        seed = (rng_seed + i) if rng_seed is not None else None

        result = solver.run(r0, v0, t_max, dt=dt,
                            save_every=save_every, rng_seed=seed)
        results.append(result)

        if callback is not None:
            callback(i, result)

    return results
