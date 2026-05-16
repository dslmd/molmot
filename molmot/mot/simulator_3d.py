"""
3D MOT simulator using rate equations with full quadrupole B-field.

At each position (x, y, z), computes the local B-field vector, applies
the full Zeeman Hamiltonian, and computes scattering forces from all 6
beams.

The 3D quadrupole field B = B'(-x/2, -y/2, z) causes:
  - Position-dependent Zeeman shifts (trapping)
  - B-field direction rotation (dark state remixing)
  - Different beam-molecule coupling at each position

References
----------
* Tarbutt, NJP 17, 015007 (2015)
* Lasner et al., PRL 134, 083401 (2025)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants import hbar, k_B


# ---------------------------------------------------------------------------
# Beam description for 3D geometry
# ---------------------------------------------------------------------------

class Beam3D:
    """
    A single laser beam in 3D.

    Parameters
    ----------
    k_hat : array-like, shape (3,)
        Propagation direction (unit vector).
    freq : float
        Laser frequency (Hz, absolute).
    q_lab : int
        Polarization index in the lab spherical basis:
        0 = sigma- (q=-1), 1 = pi (q=0), 2 = sigma+ (q=+1).
    s0 : float
        Saturation parameter.
    beam_radius : float
        1/e^2 beam radius (m). If None, infinite beam is assumed.
    """
    def __init__(self, k_hat, freq, q_lab, s0, beam_radius=None):
        self.k_hat = np.asarray(k_hat, dtype=float)
        self.k_hat = self.k_hat / np.linalg.norm(self.k_hat)
        self.freq = freq
        self.q_lab = q_lab
        self.s0 = s0
        self.beam_radius = beam_radius


def make_6beam_config(mol_data, beam_pairs, beam_radius=10e-3):
    """
    Create a list of 6 Beam3D objects from beam_pairs (z-axis) by
    extending to x and y axes.

    For each beam pair (two frequencies addressing J=1/2 and J=3/2),
    creates counter-propagating beams along +x, -x, +y, -y, +z, -z
    with appropriate polarization helicity flips for retro-reflected beams.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
        From make_beam_pairs_* functions.
    beam_radius : float
        Beam 1/e^2 radius in metres.

    Returns
    -------
    beams : list of Beam3D
        All beams for the 3D MOT.
    """
    directions = [
        np.array([1, 0, 0]),   # +x
        np.array([-1, 0, 0]),  # -x
        np.array([0, 1, 0]),   # +y
        np.array([0, -1, 0]),  # -y
        np.array([0, 0, 1]),   # +z
        np.array([0, 0, -1]),  # -z
    ]

    beams = []
    for bp in beam_pairs:
        freq = bp['freq']
        q_fwd = bp['q_fwd']
        q_bwd = bp['q_bwd']
        s0 = bp['s0']

        for i, d in enumerate(directions):
            # Forward beams (even index) use q_fwd; backward (odd) use q_bwd
            if i % 2 == 0:
                q = q_fwd
            else:
                q = q_bwd
            beams.append(Beam3D(d, freq, q, s0, beam_radius))

    return beams


# ---------------------------------------------------------------------------
# 3D rate equation force
# ---------------------------------------------------------------------------

def _decompose_polarization_local(q_lab, B_hat):
    """
    Decompose a lab-frame spherical polarization component into the
    local B-field frame.

    For a beam with polarization q_lab (in the lab z-axis frame),
    project onto the spherical basis defined by B_hat.

    Returns the squared weights |<q_local|q_lab>|^2 for q_local = 0, 1, 2
    (sigma-, pi, sigma+).
    """
    # If B ~ 0, return equal weights
    B_norm = np.linalg.norm(B_hat)
    if B_norm < 1e-15:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    B_hat = B_hat / B_norm

    # Rotation angle from z-hat to B-hat
    cos_theta = np.clip(B_hat[2], -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    # Wigner d^1 matrix elements (j=1, small d-matrix for ZYZ with beta=theta)
    # d^1_{m'm}(theta) gives the rotation mixing
    # For simplicity, use the squared modulus of the d-matrix elements
    # d^1 matrix:
    #   d[-1,-1] = (1+cos)/2    d[-1,0] = -sin/sqrt2    d[-1,+1] = (1-cos)/2
    #   d[0,-1]  = sin/sqrt2    d[0,0]  = cos            d[0,+1]  = -sin/sqrt2
    #   d[+1,-1] = (1-cos)/2    d[+1,0] = sin/sqrt2     d[+1,+1] = (1+cos)/2

    ct = cos_theta
    st = sin_theta

    # Map q_lab to m value: q=0 -> m=-1, q=1 -> m=0, q=2 -> m=+1
    # d_matrix[m_local + 1, m_lab + 1] gives the element
    d_matrix_sq = np.array([
        [((1 + ct) / 2)**2, (st / np.sqrt(2))**2, ((1 - ct) / 2)**2],
        [(st / np.sqrt(2))**2, ct**2, (st / np.sqrt(2))**2],
        [((1 - ct) / 2)**2, (st / np.sqrt(2))**2, ((1 + ct) / 2)**2],
    ])

    # Column index for m_lab
    m_lab_idx = q_lab  # q=0->sigma-(m=-1), q=1->pi(m=0), q=2->sigma+(m=+1)

    # Row gives weights for m_local = -1, 0, +1 (i.e., q_local = 0, 1, 2)
    weights = d_matrix_sq[:, m_lab_idx]

    # Normalize (should already be ~1, but be safe)
    w_sum = np.sum(weights)
    if w_sum > 1e-15:
        weights /= w_sum

    return weights


def rate_eq_force_3d(mol_data, beams, r, v, B_gradient,
                     Gamma_eff_factor=1.0, beam_radius=None):
    """
    Compute 3D radiation pressure force using rate equations.

    At position r = (x, y, z):
    1. B = B'(-x/2, -y/2, z) [Gauss]
    2. Diagonalize the full Zeeman Hamiltonian H_Z = Bx*Zx + By*Zy + Bz*Zz
    3. For each beam: compute Doppler shift, polarization decomposition in
       local B-field frame, absorption rates, and force contribution.
    4. Sum over all beams for total 3D force vector.

    Parameters
    ----------
    mol_data : MolecularData
    beams : list of Beam3D
    r : array-like, shape (3,)
        Position (m).
    v : array-like, shape (3,)
        Velocity (m/s).
    B_gradient : float
        Magnetic field gradient (T/m).
    Gamma_eff_factor : float
        Scattering rate reduction factor.
    beam_radius : float, optional
        Override beam radius for all beams.

    Returns
    -------
    F : np.ndarray, shape (3,)
        Force vector (N).
    R_scatter : float
        Total scattering rate (rad/s).
    populations : np.ndarray, shape (n_ground,)
        Steady-state ground state populations.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    Gamma = mol_data.Gamma
    k = mol_data.k
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    N = n_g + n_e

    # 1. Quadrupole B-field
    B_vec_G = np.array([
        -B_gradient * r[0] / 2.0 * 1e4,
        -B_gradient * r[1] / 2.0 * 1e4,
         B_gradient * r[2] * 1e4,
    ])
    B_mag = np.linalg.norm(B_vec_G)

    if B_mag > 1e-10:
        B_hat = B_vec_G / B_mag
    else:
        B_hat = np.array([0.0, 0.0, 1.0])

    # 2. Full Zeeman Hamiltonian for the ground state
    H_Z = (B_vec_G[0] * mol_data.zeeman_x[:n_g, :n_g] +
            B_vec_G[1] * mol_data.zeeman_y[:n_g, :n_g] +
            B_vec_G[2] * mol_data.zeeman_z[:n_g, :n_g])

    # Diagonalize the ground-state Zeeman Hamiltonian
    H_Z_herm = 0.5 * (H_Z + H_Z.conj().T)
    eig_vals_g, eig_vecs_g = np.linalg.eigh(H_Z_herm)

    # Zeeman-shifted ground state energies (Hz)
    # The eigenvalues are in units of Gamma/(2*pi) * Gauss,
    # so multiply by Gamma/(2*pi) to get Hz
    Es_g = mol_data.energies[:n_g] + eig_vals_g * Gamma / (2 * np.pi)

    # Excited state: g_J ~ 0, use diagonal Bz shift
    Es_e = mol_data.energies[n_g:].copy()
    B_z_G = B_vec_G[2]
    Es_e += B_z_G * np.real(np.diag(mol_data.zeeman_z[n_g:, n_g:])) * Gamma / (2 * np.pi)

    # 3. Transform TDMs to the Zeeman eigenstate basis
    # d_new[ig_eig, ie, q] = sum_ig_bare eig_vecs_g[ig_bare, ig_eig]^* * d[ig_bare, ie_abs, q]
    d_sq_eig = np.zeros((n_g, n_e, 3))
    for ie in range(n_e):
        ie_abs = ie + n_g
        for q in range(3):
            d_bare = np.array([abs(mol_data.tdm[ig, ie_abs, q]) for ig in range(n_g)])
            # Transform: d_eig = U^dag @ d_bare (element-wise transformation)
            d_eig = eig_vecs_g.conj().T @ d_bare
            d_sq_eig[:, ie, q] = np.abs(d_eig) ** 2

    # Branching ratios in the eigenstate basis
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = np.sum(d_sq_eig[:, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq_eig[ig, ie, :]) / total

    # 4. Compute absorption rates and forces from each beam
    gamma_half = Gamma / 2.0
    gamma_half_sq = gamma_half ** 2

    # First pass: total saturation per ground state
    s_total = np.zeros(n_g)

    beam_rates = []  # store (R_exc_ig_ie, F_contribution) per beam

    for beam in beams:
        # Doppler shift
        doppler_hz = -k * np.dot(beam.k_hat, v) / (2 * np.pi)

        # Gaussian beam profile
        if beam.beam_radius is not None:
            r_parallel = np.dot(r, beam.k_hat)
            r_perp_sq = np.dot(r, r) - r_parallel ** 2
            r_perp_sq = max(r_perp_sq, 0.0)
            gauss_factor = np.exp(-2.0 * r_perp_sq / beam.beam_radius ** 2)
        else:
            gauss_factor = 1.0

        s_eff = beam.s0 * gauss_factor

        # Polarization decomposition in the local B-field frame
        pol_weights = _decompose_polarization_local(beam.q_lab, B_hat)

        R_exc_beam = np.zeros((n_g, n_e))
        for ig in range(n_g):
            for ie in range(n_e):
                omega_trans = Es_e[ie] - Es_g[ig]  # Hz

                # Sum over polarization components in local frame
                rate_sum = 0.0
                for q_local in range(3):
                    d2 = d_sq_eig[ig, ie, q_local]
                    if d2 < 1e-15:
                        continue
                    delta_eff = (beam.freq + doppler_hz - omega_trans) * 2 * np.pi
                    L = gamma_half_sq / (delta_eff ** 2 + gamma_half_sq)
                    rate_sum += pol_weights[q_local] * d2 * L

                s_total[ig] += s_eff * rate_sum
                R_exc_beam[ig, ie] = rate_sum * s_eff

        beam_rates.append(R_exc_beam)

    # Second pass: apply saturation correction and compute forces
    R_exc_total = np.zeros((n_g, n_e))
    F_total = np.zeros(3)

    for ib, beam in enumerate(beams):
        R_exc_beam = beam_rates[ib]
        for ig in range(n_g):
            sat_denom = 1.0 + s_total[ig]
            for ie in range(n_e):
                rate = gamma_half * R_exc_beam[ig, ie] * Gamma_eff_factor / sat_denom
                R_exc_total[ig, ie] += rate

    # Steady-state populations
    R_sum = R_exc_total  # (n_g, n_e) -- total excitation rate from each (g, e)

    M = np.zeros((n_g, n_g))
    for ig in range(n_g):
        M[ig, ig] -= np.sum(R_sum[ig, :])
        for ik in range(n_g):
            for ie in range(n_e):
                M[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    M_sol = M.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(n_g)
    rhs[-1] = 1.0

    try:
        from scipy.linalg import solve
        p = solve(M_sol, rhs)
    except Exception:
        p = np.ones(n_g) / n_g

    p = np.maximum(p, 0.0)
    p_sum = np.sum(p)
    if p_sum > 0:
        p /= p_sum
    else:
        p = np.ones(n_g) / n_g

    # Compute force per beam with populations
    F_total = np.zeros(3)
    R_scatter = 0.0

    for ib, beam in enumerate(beams):
        R_exc_beam = beam_rates[ib]
        for ig in range(n_g):
            sat_denom = 1.0 + s_total[ig]
            for ie in range(n_e):
                rate = gamma_half * R_exc_beam[ig, ie] * Gamma_eff_factor / sat_denom
                F_beam_contrib = hbar * k * beam.k_hat * rate * p[ig]
                F_total += F_beam_contrib
                R_scatter += rate * p[ig]

    return F_total, R_scatter, p


# ---------------------------------------------------------------------------
# MOTSimulator3D
# ---------------------------------------------------------------------------

class MOTSimulator3D:
    """
    3D MOT simulator using rate equations.

    At each position (x, y, z), computes the local B-field vector,
    applies the full Zeeman Hamiltonian, and computes scattering
    forces from all 6 beams.

    Parameters
    ----------
    mol_data : MolecularData
    beam_config : list of Beam3D or list of dict (beam_pairs + beam_radius)
        If list of Beam3D, used directly.
        If list of dict (beam_pairs format), converted via make_6beam_config.
    B_gradient : float
        Magnetic field gradient (T/m).
    s0 : float
        Saturation parameter (used if beam_config is beam_pairs).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor.
    beam_radius : float
        1/e^2 beam radius (m).
    """

    def __init__(self, mol_data, beam_config, B_gradient,
                 s0=1.0, Gamma_eff_factor=1.0, beam_radius=10e-3):
        self.mol_data = mol_data
        self.B_gradient = B_gradient
        self.Gamma_eff_factor = Gamma_eff_factor
        self.beam_radius = beam_radius
        self.mass = mol_data.mass

        # Convert beam_pairs to Beam3D list if needed
        if isinstance(beam_config, list) and len(beam_config) > 0:
            if isinstance(beam_config[0], Beam3D):
                self.beams = beam_config
            elif isinstance(beam_config[0], dict):
                self.beams = make_6beam_config(mol_data, beam_config,
                                               beam_radius=beam_radius)
            else:
                raise ValueError("beam_config must be list of Beam3D or beam_pairs dicts")
        else:
            raise ValueError("beam_config must be a non-empty list")

    def force_3d(self, r, v):
        """
        Compute the 3D force vector at position r with velocity v.

        Parameters
        ----------
        r : array-like, shape (3,)
            Position (m).
        v : array-like, shape (3,)
            Velocity (m/s).

        Returns
        -------
        F : np.ndarray, shape (3,)
            Force vector (N).
        R_scatter : float
            Total scattering rate (rad/s).
        populations : np.ndarray
            Ground-state populations.
        """
        return rate_eq_force_3d(
            self.mol_data, self.beams,
            np.asarray(r, dtype=float),
            np.asarray(v, dtype=float),
            self.B_gradient,
            self.Gamma_eff_factor)

    def simulate_trajectory_3d(self, r0, v0, t_max, dt=1e-6,
                               r_escape=0.015):
        """
        Integrate a 3D molecular trajectory using leapfrog integration.

        Leapfrog (Stormer-Verlet) provides better energy conservation
        than Euler and is symplectic.

        Parameters
        ----------
        r0 : array-like, shape (3,)
            Initial position (m).
        v0 : array-like, shape (3,)
            Initial velocity (m/s).
        t_max : float
            Total simulation time (s).
        dt : float
            Time step (s).
        r_escape : float
            If |r| exceeds this, trajectory is truncated.

        Returns
        -------
        t_arr : np.ndarray, shape (n_steps,)
        r_arr : np.ndarray, shape (n_steps, 3)
        v_arr : np.ndarray, shape (n_steps, 3)
        """
        mass = self.mass
        n_steps = int(t_max / dt)

        t_arr = np.arange(n_steps) * dt
        r_arr = np.zeros((n_steps, 3))
        v_arr = np.zeros((n_steps, 3))

        r_arr[0] = np.asarray(r0, dtype=float)
        v_arr[0] = np.asarray(v0, dtype=float)

        # Initial half-step for velocity (leapfrog)
        F0, _, _ = self.force_3d(r_arr[0], v_arr[0])
        v_half = v_arr[0] + 0.5 * dt * F0 / mass

        for j in range(1, n_steps):
            # Full position step
            r_arr[j] = r_arr[j - 1] + dt * v_half

            # Check escape
            if np.linalg.norm(r_arr[j]) > r_escape:
                r_arr[j:] = r_arr[j]
                v_arr[j:] = v_half  # approximate
                break

            # Force at new position
            F, _, _ = self.force_3d(r_arr[j], v_half)

            # Full velocity step
            v_half_new = v_half + dt * F / mass

            # Store the centered velocity
            v_arr[j] = 0.5 * (v_half + v_half_new)
            v_half = v_half_new

        return t_arr, r_arr, v_arr

    def simulate_ensemble_3d(self, r0_list, v0_list, t_max, dt=1e-6,
                             r_escape=0.015, callback=None):
        """
        Simulate an ensemble of 3D trajectories.

        Parameters
        ----------
        r0_list : list of array-like
            Initial positions.
        v0_list : list of array-like
            Initial velocities.
        t_max : float
        dt : float
        r_escape : float
        callback : callable(i, t_arr, r_arr, v_arr), optional

        Returns
        -------
        trajectories : list of (t_arr, r_arr, v_arr) tuples
        """
        trajectories = []
        for i, (r0, v0) in enumerate(zip(r0_list, v0_list)):
            t, r, v = self.simulate_trajectory_3d(r0, v0, t_max, dt, r_escape)
            trajectories.append((t, r, v))
            if callback is not None:
                callback(i, t, r, v)
        return trajectories


# ---------------------------------------------------------------------------
# Force map computation
# ---------------------------------------------------------------------------

def compute_3d_force_map(simulator, axes='xz', n_points=50,
                         v=None, extent=5e-3):
    """
    Compute force on a 2D slice through the trap.

    Parameters
    ----------
    simulator : MOTSimulator3D
    axes : str
        Which 2D plane to compute: 'xz', 'xy', or 'yz'.
    n_points : int
        Number of grid points per axis.
    v : array-like, shape (3,), optional
        Velocity vector. Default: zero.
    extent : float
        Half-size of the grid (m).

    Returns
    -------
    coords1 : np.ndarray, shape (n_points,)
        First axis coordinates (m).
    coords2 : np.ndarray, shape (n_points,)
        Second axis coordinates (m).
    Fx_map : np.ndarray, shape (n_points, n_points)
        Force along first axis (N).
    Fy_map : np.ndarray, shape (n_points, n_points)
        Force along second axis (N).
    R_map : np.ndarray, shape (n_points, n_points)
        Scattering rate (rad/s).
    """
    if v is None:
        v = np.zeros(3)
    v = np.asarray(v, dtype=float)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    a1 = axis_map[axes[0]]
    a2 = axis_map[axes[1]]

    coords1 = np.linspace(-extent, extent, n_points)
    coords2 = np.linspace(-extent, extent, n_points)

    Fx_map = np.zeros((n_points, n_points))
    Fy_map = np.zeros((n_points, n_points))
    R_map = np.zeros((n_points, n_points))

    for i, c1 in enumerate(coords1):
        for j, c2 in enumerate(coords2):
            r = np.zeros(3)
            r[a1] = c1
            r[a2] = c2

            F, R, _ = simulator.force_3d(r, v)
            Fx_map[i, j] = F[a1]
            Fy_map[i, j] = F[a2]
            R_map[i, j] = R

    return coords1, coords2, Fx_map, Fy_map, R_map
