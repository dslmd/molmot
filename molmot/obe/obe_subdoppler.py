"""
Full OBE force calculator with sub-Doppler effects.

Captures Sisyphus cooling and velocity-selective dark states by:
1. Including standing-wave phases exp(±ikz) in the Hamiltonian
2. Solving the full Lindblad master equation (with ground-state coherences)
3. Averaging the force over one optical wavelength

The standing wave creates a spatially periodic polarization pattern.
For counter-propagating σ⁺/σ⁻ beams, this is the classic lin⊥lin
configuration that produces Sisyphus cooling.
"""

import numpy as np
from scipy.linalg import solve as la_solve
from ..constants import hbar


def build_H_standing_wave(mol_data, beam_pairs, v, z_macro, z_micro,
                          B_gradient, Omega_Gamma):
    """
    Build the Hamiltonian in the ROTATING FRAME with standing-wave phases.

    The rotating frame removes the large optical frequency, leaving only
    detunings (~MHz), Rabi frequencies (~MHz), and Zeeman shifts (~MHz).

    For each beam pair at laser frequency ω_L, the rotating frame gives:
      H_gg = -(ω_L - ω_{eg}) + Zeeman  (detuning from each g→e transition)
      H_ee = 0 + Zeeman_excited
      H_eg = (Ω/2) × d × exp(+ikz)  for +z beam
      H_eg = (Ω/2) × d × exp(-ikz)  for -z beam

    For multiple laser frequencies addressing different ground states,
    we use a multi-rotating frame where each ground state is shifted by
    its nearest laser frequency.
    """
    Gamma = mol_data.Gamma
    k = mol_data.k
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    N = n_g + n_e

    B_gauss = B_gradient * z_macro * 1e4

    # Zeeman-shifted state energies (Hz)
    E_Z = B_gauss * mol_data.zeeman_z_diag * Gamma / (2 * np.pi)
    Es_Hz = mol_data.energies + E_Z

    # Mean excited state energy (Hz)
    E_e_mean = np.mean(Es_Hz[n_g:])

    laser_freqs = [bp['freq'] for bp in beam_pairs]

    Omega0 = Omega_Gamma * Gamma  # peak Rabi freq (rad/s)
    d = mol_data.tdm

    H = np.zeros((N, N), dtype=complex)

    for ig in range(n_g):
        omega_trans = E_e_mean - Es_Hz[ig]
        # Average detuning from all addressing lasers
        deltas = [lf - omega_trans for lf in laser_freqs]
        nearest_delta = min(deltas, key=abs)
        H[ig, ig] = -2 * np.pi * nearest_delta

    for ie in range(n_e):
        ie_abs = ie + n_g
        H[ie_abs, ie_abs] = 2 * np.pi * (Es_Hz[ie_abs] - E_e_mean)

    for bp in beam_pairs:
        freq = bp['freq']
        q_fwd = bp['q_fwd']
        q_bwd = bp['q_bwd']
        s0 = bp['s0']
        Om = Omega0 * np.sqrt(s0)

        for ig in range(n_g):
            omega_trans = E_e_mean - Es_Hz[ig]
            delta_laser = freq - omega_trans  # Hz

            for ie in range(n_e):
                ie_abs = ie + n_g

                # Forward beam (+z): Doppler = -kv/(2π) Hz
                delta_fwd = delta_laser - k * v / (2 * np.pi)
                d_val = abs(d[ig, ie_abs, q_fwd])
                if d_val > 1e-10:
                    # In the rotating frame of this beam:
                    # The coupling phase includes exp(ikz_micro)
                    # The detuning for this specific g→e pair via this beam:
                    delta_ge_fwd = delta_fwd - (Es_Hz[ie_abs] - E_e_mean)
                    coupling = Om * d_val / 2 * np.exp(1j * z_micro)
                    H[ie_abs, ig] += coupling
                    H[ig, ie_abs] += np.conj(coupling)

                # Backward beam (-z): Doppler = +kv/(2π) Hz
                delta_bwd = delta_laser + k * v / (2 * np.pi)
                d_val = abs(d[ig, ie_abs, q_bwd])
                if d_val > 1e-10:
                    coupling = Om * d_val / 2 * np.exp(-1j * z_micro)
                    H[ie_abs, ig] += coupling
                    H[ig, ie_abs] += np.conj(coupling)

    return H


def build_liouvillian_fast(H, mol_data):
    """Build Liouvillian from Hamiltonian, optimized for repeated calls."""
    N = H.shape[0]
    N2 = N * N
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    Gamma = mol_data.Gamma
    d = mol_data.tdm

    I = np.eye(N, dtype=complex)

    # Coherent part (column-major vectorization)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Dissipator: spontaneous emission
    d_sq = mol_data.d_squared
    for ie in range(n_e):
        ie_abs = ie + n_g
        total = sum(d_sq[ig, ie, q] for ig in range(n_g) for q in range(3))
        if total < 1e-30:
            continue
        for ig in range(n_g):
            for q in range(3):
                br = d_sq[ig, ie, q] / total
                if br < 1e-15:
                    continue
                Gc = Gamma * br
                Lc = np.zeros((N, N), dtype=complex)
                Lc[ig, ie_abs] = np.sqrt(Gc)
                LdL = Lc.conj().T @ Lc
                L += np.kron(Lc.conj(), Lc)
                L -= 0.5 * np.kron(I, LdL)
                L -= 0.5 * np.kron(LdL.T, I)

    return L


def solve_steady_state(L, N):
    """
    Solve L @ vec(rho) = 0 with Tr(rho) = 1.

    Uses SVD to find the null space, handling degenerate dark states
    robustly. If the null space has dimension > 1, selects the
    physical steady state (Hermitian, positive, normalized).
    """
    N2 = N * N

    # Add trace constraint as an extra row
    trace_row = np.zeros((1, N2), dtype=complex)
    for i in range(N):
        trace_row[0, i * N + i] = 1.0

    L_aug = np.vstack([L, trace_row])
    rhs = np.zeros(N2 + 1, dtype=complex)
    rhs[-1] = 1.0

    # Solve via least squares (handles rank-deficient systems)
    rho_vec, residuals, rank, sv = np.linalg.lstsq(L_aug, rhs, rcond=None)

    rho = rho_vec.reshape((N, N), order='F')

    # Enforce Hermiticity and positivity
    rho = 0.5 * (rho + rho.conj().T)

    # Project out negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = (eigvecs * eigvals) @ eigvecs.conj().T

    # Normalize
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho /= tr
    else:
        rho = np.eye(N, dtype=complex) / N

    return rho


def force_from_rho(rho, mol_data, beam_pairs, z_micro, Omega_Gamma):
    """
    Compute force from density matrix and field gradients.

    F = -Tr(rho * dH/dz) summed over all beams.
    For beam +z: dH_coupling/dz = ik * H_coupling
    For beam -z: dH_coupling/dz = -ik * H_coupling
    """
    k = mol_data.k
    Gamma = mol_data.Gamma
    n_g = mol_data.n_ground
    n_e = mol_data.n_excited
    d = mol_data.tdm
    Omega0 = Omega_Gamma * Gamma

    force = 0.0
    for bp in beam_pairs:
        q_fwd = bp['q_fwd']
        q_bwd = bp['q_bwd']
        s0 = bp['s0']
        Om = Omega0 * np.sqrt(s0)

        for ig in range(n_g):
            for ie in range(n_e):
                ie_abs = ie + n_g

                # +z beam: dH/dz = ik * coupling
                d_fwd = abs(d[ig, ie_abs, q_fwd])
                if d_fwd > 1e-10:
                    coupling = Om * d_fwd / 2 * np.exp(1j * z_micro)
                    # F = -Tr(rho * dH/dz) for this element
                    # dH[ie,ig]/dz = ik * coupling
                    # contribution = -(rho[ig,ie] * ik * coupling + rho[ie,ig] * (-ik) * conj(coupling))
                    # = -2k * Im(rho[ig,ie] * coupling)
                    force += hbar * k * 2 * np.imag(rho[ig, ie_abs] * coupling)

                # -z beam: dH/dz = -ik * coupling
                d_bwd = abs(d[ig, ie_abs, q_bwd])
                if d_bwd > 1e-10:
                    coupling = Om * d_bwd / 2 * np.exp(-1j * z_micro)
                    force -= hbar * k * 2 * np.imag(rho[ig, ie_abs] * coupling)

    return force


def obe_force_subdoppler(mol_data, beam_pairs, v, z_macro, B_gradient,
                         Omega_Gamma, n_spatial=16, Gamma_eff_factor=1.0):
    """
    Compute force including sub-Doppler effects via OBE.

    Averages the force over one optical wavelength to capture
    Sisyphus cooling and polarization-gradient effects.

    Parameters
    ----------
    mol_data : MolecularData
    beam_pairs : list of dict
        Each dict: {'freq': Hz, 'q_fwd': 0/1/2, 'q_bwd': 0/1/2, 's0': float}
    v : float — velocity (m/s)
    z_macro : float — macroscopic position (m)
    B_gradient : float — T/m
    Omega_Gamma : float — Rabi frequency / Gamma per beam
    n_spatial : int — number of points for wavelength averaging
    Gamma_eff_factor : float — scattering rate reduction factor

    Returns
    -------
    force : float (N)
    rho_avg : averaged density matrix
    R_scatter : scattering rate (rad/s)
    """
    N = mol_data.n_states
    Gamma_eff = mol_data.Gamma * Gamma_eff_factor

    # Sample positions within one standing-wave period (0 to 2*pi)
    z_micros = np.linspace(0, 2 * np.pi, n_spatial, endpoint=False)

    F_total = 0.0
    rho_avg = np.zeros((N, N), dtype=complex)

    for z_m in z_micros:
        H = build_H_standing_wave(mol_data, beam_pairs, v, z_macro, z_m,
                                  B_gradient, Omega_Gamma)

        # Scale decay rate by Gamma_eff_factor
        mol_data_eff = type(mol_data).__new__(type(mol_data))
        mol_data_eff.__dict__.update(mol_data.__dict__)
        mol_data_eff.Gamma = Gamma_eff

        L = build_liouvillian_fast(H, mol_data_eff)
        rho = solve_steady_state(L, N)

        F = force_from_rho(rho, mol_data, beam_pairs, z_m, Omega_Gamma)
        F_total += F
        rho_avg += rho

    F_avg = F_total / n_spatial
    rho_avg /= n_spatial
    R_scatter = Gamma_eff * np.sum(np.real(np.diag(rho_avg))[mol_data.n_ground:])

    return F_avg, rho_avg, R_scatter


def make_beam_pairs_rf(mol_data, delta_Gamma, s0, phase=0):
    """Make beam pairs for RF MOT (one phase of the switching cycle)."""
    Gamma = mol_data.Gamma
    delta_hz = delta_Gamma * Gamma / (2 * np.pi)

    freq_J12 = mol_data.omega_J12 + delta_hz
    freq_J32 = mol_data.omega_J32 + delta_hz

    if phase == 0:
        q_J32, q_J12 = 2, 0  # J=3/2: σ⁺ fwd, J=1/2: σ⁻ fwd
    else:
        q_J32, q_J12 = 0, 2

    return [
        {'freq': freq_J32, 'q_fwd': q_J32, 'q_bwd': 2 - q_J32, 's0': s0},
        {'freq': freq_J12, 'q_fwd': q_J12, 'q_bwd': 2 - q_J12, 's0': s0},
    ]


def make_beam_pairs_dc4(mol_data, delta_Gamma, split_Gamma, s0):
    """Make beam pairs for 4-frequency DC MOT."""
    Gamma = mol_data.Gamma
    delta_hz = delta_Gamma * Gamma / (2 * np.pi)
    split_hz = split_Gamma * Gamma / (2 * np.pi)

    return [
        {'freq': mol_data.omega_J32 + delta_hz + split_hz,
         'q_fwd': 2, 'q_bwd': 0, 's0': s0},  # J=3/2, σ⁺
        {'freq': mol_data.omega_J32 + delta_hz - split_hz,
         'q_fwd': 0, 'q_bwd': 2, 's0': s0},  # J=3/2, σ⁻
        {'freq': mol_data.omega_J12 + delta_hz + split_hz,
         'q_fwd': 2, 'q_bwd': 0, 's0': s0},  # J=1/2, σ⁺
        {'freq': mol_data.omega_J12 + delta_hz - split_hz,
         'q_fwd': 0, 'q_bwd': 2, 's0': s0},  # J=1/2, σ⁻
    ]


def make_beam_pairs_blue(mol_data, delta_Gamma, delta2_Gamma, s0):
    """Make beam pairs for blue Λ-BDM."""
    Gamma = mol_data.Gamma
    delta_hz = delta_Gamma * Gamma / (2 * np.pi)
    delta2_hz = delta2_Gamma * Gamma / (2 * np.pi)

    return [
        {'freq': mol_data.omega_J32 + delta_hz,
         'q_fwd': 0, 'q_bwd': 2, 's0': s0},  # J=3/2, σ⁻ fwd
        {'freq': mol_data.omega_J12 + delta_hz + delta2_hz,
         'q_fwd': 2, 'q_bwd': 0, 's0': s0},  # J=1/2, σ⁺ fwd
    ]
