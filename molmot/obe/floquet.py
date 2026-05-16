"""
Floquet OBE sub-Doppler force calculator.

Computes velocity-dependent forces in a standing wave using the Floquet
expansion of the Lindblad master equation.  This captures Sisyphus cooling
and velocity-selective coherent population trapping (VSCPT) effects that
rate equations miss.

For a molecule moving at velocity v through a standing wave with wavevector k,
the Hamiltonian is periodic: H(z) = H(z + lambda/2) where z = z0 + v*t.
In the rotating frame:
    H(t) = H0 + V+ exp(+ikvt) + V- exp(-ikvt) + ...

The density matrix Fourier expansion:
    rho(t) = sum_n rho_n exp(i*n*k*v*t)

The Lindblad equation becomes a block-tridiagonal system for the Fourier
components {rho_n}, which is solved to get the steady-state ρ_0 and the
velocity-dependent force.

References
----------
* Dalibard, Cohen-Tannoudji, JOSA B 6, 2023 (1989)
* Javanainen, PRE 47, 3547 (1993) -- Floquet approach to laser cooling
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.sparse import linalg as sp_linalg

from ..constants import hbar


class FloquetOBE:
    """
    Compute velocity-dependent forces using the Floquet expansion
    of the Lindblad master equation in a standing wave.

    This captures Sisyphus cooling and velocity-selective coherent
    population trapping (VSCPT) effects that rate equations miss.

    Parameters
    ----------
    mol_data : MolecularData
        Molecular data container.
    beam_pairs : list of dict
        Beam pair configurations from obe_subdoppler.py.
        Each dict: {'freq': Hz, 'q_fwd': int, 'q_bwd': int, 's0': float}
    B_gradient : float
        Magnetic field gradient (T/m).
    Omega_Gamma : float
        Peak Rabi frequency in units of Gamma per beam.
    n_max : int
        Floquet truncation order (2-5; higher = more accurate but slower).
    Gamma_eff_factor : float
        Effective scattering rate reduction factor (for repumping losses).
    """

    def __init__(self, mol_data, beam_pairs, B_gradient, Omega_Gamma=1.0,
                 n_max=3, Gamma_eff_factor=1.0):
        self.mol_data = mol_data
        self.beam_pairs = beam_pairs
        self.B_gradient = B_gradient
        self.Omega_Gamma = Omega_Gamma
        self.n_max = n_max
        self.Gamma_eff_factor = Gamma_eff_factor

        self.Gamma = mol_data.Gamma
        self.k = mol_data.k
        self.n_g = mol_data.n_ground
        self.n_e = mol_data.n_excited
        self.N = self.n_g + self.n_e
        self.N2 = self.N ** 2

        # Pre-build the Liouvillian Floquet blocks
        self._build_floquet_blocks()

    def _build_floquet_blocks(self):
        """
        Decompose the Hamiltonian into Fourier components and build
        the corresponding Liouvillian blocks.

        For counter-propagating beams with polarizations q_fwd and q_bwd:
            H(z) = H0 + H_plus * exp(+ikz) + H_minus * exp(-ikz)

        where:
            H0 = diagonal (detunings + Zeeman at z_macro)
            H_plus = (Omega/2) * d_{ge,q_fwd}  (coupling from +z beam)
            H_minus = (Omega/2) * d_{ge,q_bwd}  (coupling from -z beam)

        The Liouvillian blocks are:
            L0 = -i[H0, .] + Lindblad decay
            L_plus = -i[H_plus, .]   (raising by one Fourier order)
            L_minus = -i[H_minus, .]  (lowering by one Fourier order)
        """
        N = self.N
        n_g = self.n_g
        n_e = self.n_e
        Gamma = self.Gamma
        Gamma_eff = Gamma * self.Gamma_eff_factor
        d = self.mol_data.tdm
        Omega0 = self.Omega_Gamma * Gamma

        # --- H0: diagonal detunings ---
        # Use mean detuning at z_macro = 0 (no Zeeman at trap center)
        Es_Hz = self.mol_data.energies.copy()
        E_e_mean = np.mean(Es_Hz[n_g:])

        # Build H0 in the rotating frame
        H0 = np.zeros((N, N), dtype=complex)
        laser_freqs = [bp['freq'] for bp in self.beam_pairs]

        for ig in range(n_g):
            omega_trans = E_e_mean - Es_Hz[ig]
            deltas = [lf - omega_trans for lf in laser_freqs]
            nearest_delta = min(deltas, key=abs)
            H0[ig, ig] = -2 * np.pi * nearest_delta

        for ie in range(n_e):
            ie_abs = ie + n_g
            H0[ie_abs, ie_abs] = 2 * np.pi * (Es_Hz[ie_abs] - E_e_mean)

        # --- H_plus and H_minus: coupling from forward/backward beams ---
        H_plus = np.zeros((N, N), dtype=complex)
        H_minus = np.zeros((N, N), dtype=complex)

        for bp in self.beam_pairs:
            q_fwd = bp['q_fwd']
            q_bwd = bp['q_bwd']
            s0 = bp['s0']
            Om = Omega0 * np.sqrt(s0)

            for ig in range(n_g):
                for ie in range(n_e):
                    ie_abs = ie + n_g

                    # Forward beam (+z): exp(+ikz)
                    d_fwd = abs(d[ig, ie_abs, q_fwd])
                    if d_fwd > 1e-10:
                        coupling = Om * d_fwd / 2
                        H_plus[ie_abs, ig] += coupling
                        H_plus[ig, ie_abs] += np.conj(coupling)

                    # Backward beam (-z): exp(-ikz)
                    d_bwd = abs(d[ig, ie_abs, q_bwd])
                    if d_bwd > 1e-10:
                        coupling = Om * d_bwd / 2
                        H_minus[ie_abs, ig] += coupling
                        H_minus[ig, ie_abs] += np.conj(coupling)

        self._H0 = H0
        self._H_plus = H_plus
        self._H_minus = H_minus

        # --- Build Liouvillian blocks ---
        I = np.eye(N, dtype=complex)

        # L0: coherent part from H0 + dissipator
        L0 = -1j * (np.kron(I, H0) - np.kron(H0.T, I))

        # Dissipator: spontaneous emission
        d_sq = self.mol_data.d_squared
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
                    Gc = Gamma_eff * br
                    Lc = np.zeros((N, N), dtype=complex)
                    Lc[ig, ie_abs] = np.sqrt(Gc)
                    LdL = Lc.conj().T @ Lc
                    L0 += np.kron(Lc.conj(), Lc)
                    L0 -= 0.5 * np.kron(I, LdL)
                    L0 -= 0.5 * np.kron(LdL.T, I)

        # L_plus: coupling from H_plus (raises Fourier order by 1)
        L_plus = -1j * (np.kron(I, H_plus) - np.kron(H_plus.T, I))

        # L_minus: coupling from H_minus (lowers Fourier order by 1)
        L_minus = -1j * (np.kron(I, H_minus) - np.kron(H_minus.T, I))

        self._L0 = L0
        self._L_plus = L_plus
        self._L_minus = L_minus

    def _solve_floquet(self, v, z_macro=0.0):
        """
        Solve the Floquet-Lindblad equation for the steady-state
        density matrix Fourier components.

        The equation for each Fourier order n:
            (i*n*k*v) rho_n = L0 @ rho_n + L_plus @ rho_{n-1} + L_minus @ rho_{n+1}

        Rearranged:
            (L0 - i*n*k*v*I) @ rho_n + L_plus @ rho_{n-1} + L_minus @ rho_{n+1} = 0

        This is a block-tridiagonal system of size (2*n_max+1) * N^2.

        Parameters
        ----------
        v : float
            Velocity in m/s.
        z_macro : float
            Macroscopic position for Zeeman shift (m). Default 0.

        Returns
        -------
        rho_components : list of np.ndarray
            Fourier components rho_{-n_max}, ..., rho_0, ..., rho_{+n_max},
            each reshaped to (N, N).
        """
        N = self.N
        N2 = self.N2
        n_max = self.n_max
        n_blocks = 2 * n_max + 1

        # Angular frequency spacing for Fourier modes
        kv = self.k * v  # rad/s (angular)

        # Add Zeeman shift to H0 if z_macro != 0
        if abs(z_macro) > 1e-15:
            B_gauss = self.B_gradient * z_macro * 1e4
            E_Z = B_gauss * self.mol_data.zeeman_z_diag * self.Gamma / (2 * np.pi)
            H0_shifted = self._H0.copy()
            for i in range(N):
                H0_shifted[i, i] += 2 * np.pi * E_Z[i]
            I_N = np.eye(N, dtype=complex)
            L0_shifted = self._L0.copy()
            L0_shifted += -1j * (np.kron(I_N, np.diag(2 * np.pi * E_Z)) -
                                  np.kron(np.diag(2 * np.pi * E_Z), I_N))
        else:
            L0_shifted = self._L0

        # Build the big block-tridiagonal matrix
        total_size = n_blocks * N2
        M = np.zeros((total_size, total_size), dtype=complex)

        for b in range(n_blocks):
            n = b - n_max  # Fourier order: -n_max, ..., 0, ..., +n_max
            row_start = b * N2
            row_end = (b + 1) * N2

            # Diagonal block: L0 - i*n*kv*I
            M[row_start:row_end, row_start:row_end] = (
                L0_shifted - 1j * n * kv * np.eye(N2, dtype=complex))

            # Sub-diagonal block: L_plus @ rho_{n-1}
            if b > 0:
                col_start = (b - 1) * N2
                col_end = b * N2
                M[row_start:row_end, col_start:col_end] = self._L_plus

            # Super-diagonal block: L_minus @ rho_{n+1}
            if b < n_blocks - 1:
                col_start = (b + 1) * N2
                col_end = (b + 2) * N2
                M[row_start:row_end, col_start:col_end] = self._L_minus

        # Replace trace constraint for the n=0 block
        # The physical constraint is Tr(rho_0) = 1; all other Tr(rho_n) = 0
        # Replace the last row of the n=0 block with the trace condition
        b0 = n_max  # index of the n=0 block
        row_trace = b0 * N2 + N2 - 1  # last row of n=0 block
        M[row_trace, :] = 0.0
        for i in range(N):
            # Column-major: rho_{i,i} is at index i*N + i within the n=0 block
            col_idx = b0 * N2 + i * N + i
            M[row_trace, col_idx] = 1.0

        rhs = np.zeros(total_size, dtype=complex)
        rhs[row_trace] = 1.0

        # Solve the linear system
        try:
            rho_vec = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Fallback: least squares
            rho_vec, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Extract Fourier components
        rho_components = []
        for b in range(n_blocks):
            rho_n = rho_vec[b * N2:(b + 1) * N2].reshape((N, N), order='F')
            rho_components.append(rho_n)

        return rho_components

    def force_vs_velocity(self, v_arr, z_macro=0.0, n_points_z=1):
        """
        Compute the force as a function of velocity.

        For each velocity v:
        1. Build and solve the Floquet system
        2. Extract rho_0, rho_+1, rho_-1
        3. Compute the scattering force from the Fourier components

        Parameters
        ----------
        v_arr : np.ndarray
            Array of velocities (m/s).
        z_macro : float
            Macroscopic position for Zeeman shift (m).
        n_points_z : int
            Number of macroscopic z points to average over (for spatial
            averaging in a trap). Default: 1 (no averaging).

        Returns
        -------
        F_arr : np.ndarray
            Force at each velocity (N).
        """
        v_arr = np.asarray(v_arr, dtype=float)
        F_arr = np.zeros(len(v_arr))
        k = self.k
        N = self.N
        n_g = self.n_g
        n_e = self.n_e
        n_max = self.n_max

        for iv, v in enumerate(v_arr):
            if abs(v) < 1e-12:
                # At exactly v=0, the Floquet matrix is singular (all blocks
                # have the same diagonal). Use a small offset.
                v_eff = 1e-6
            else:
                v_eff = v

            rho_components = self._solve_floquet(v_eff, z_macro)

            # rho_0 is the DC component (index n_max in the list)
            rho_0 = rho_components[n_max]

            # rho_{+1} and rho_{-1}
            rho_p1 = rho_components[n_max + 1] if n_max + 1 < len(rho_components) else np.zeros((N, N), dtype=complex)
            rho_m1 = rho_components[n_max - 1] if n_max - 1 >= 0 else np.zeros((N, N), dtype=complex)

            # Force from the scattering:
            # F = -Tr(rho * dH/dz) = -Tr(rho_0 * dH0/dz)
            #     - Tr(rho_{-1} * ik * H_plus) - Tr(rho_{+1} * (-ik) * H_minus)
            # The H0 has no z-dependence at fixed z_macro.
            # The standing-wave gradient: dH/dz = ik*H_plus*exp(ikz) - ik*H_minus*exp(-ikz)
            #
            # F = hbar * { -ik * Tr(rho_{-1} * H_plus) + ik * Tr(rho_{+1} * H_minus) }
            # Taking real part for physical force:

            F_plus = np.trace(rho_m1 @ self._H_plus)   # Tr(rho_{-1} * H_+)
            F_minus = np.trace(rho_p1 @ self._H_minus)  # Tr(rho_{+1} * H_-)

            # Force in SI: F = hbar * k * 2 * Im(F_plus - F_minus)
            # More precisely:
            # F = -hbar * Re( i*k * Tr(rho_{-1} H_+) - i*k * Tr(rho_{+1} H_-) )
            #   = -hbar * k * ( -Im(Tr(rho_{-1} H_+)) + Im(Tr(rho_{+1} H_-)) )
            #   = hbar * k * ( Im(Tr(rho_{-1} H_+)) - Im(Tr(rho_{+1} H_-)) )
            force_val = hbar * k * (np.imag(F_plus) - np.imag(F_minus))

            F_arr[iv] = force_val

        return F_arr

    def scattering_rate(self, v, z_macro=0.0):
        """
        Compute the scattering rate at velocity v.

        Parameters
        ----------
        v : float
            Velocity (m/s).
        z_macro : float
            Position (m).

        Returns
        -------
        R_scatter : float
            Scattering rate (rad/s).
        """
        v_eff = v if abs(v) > 1e-12 else 1e-6
        rho_components = self._solve_floquet(v_eff, z_macro)
        rho_0 = rho_components[self.n_max]

        # Scattering rate = Gamma_eff * sum of excited-state populations
        Gamma_eff = self.Gamma * self.Gamma_eff_factor
        n_g = self.n_g
        R = Gamma_eff * np.sum(np.real(np.diag(rho_0))[n_g:])
        return max(R, 0.0)

    def steady_state_rho(self, v, z_macro=0.0):
        """
        Return the DC component of the steady-state density matrix.

        Parameters
        ----------
        v : float
            Velocity (m/s).
        z_macro : float
            Position (m).

        Returns
        -------
        rho_0 : np.ndarray, shape (N, N)
        """
        v_eff = v if abs(v) > 1e-12 else 1e-6
        rho_components = self._solve_floquet(v_eff, z_macro)
        return rho_components[self.n_max]
