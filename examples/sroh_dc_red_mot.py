#!/usr/bin/env python3
"""
SrOH DC Red MOT Simulation — Optical Bloch Equations

Simulates the magneto-optical trapping force on SrOH molecules in a
DC (static B-field, static polarization) red-detuned MOT using
multi-level Optical Bloch Equations (Lindblad master equation).

Transition: X̃²Σ⁺(000, N=1) → Ã²Π₁/₂(000, J'=1/2) at 688 nm

Level structure:
  Ground (6 states):  J=3/2 (mJ=-3/2..+3/2), J=1/2 (mJ=-1/2,+1/2)
  Excited (2 states): J'=1/2 (mJ'=-1/2,+1/2)

The spin-rotation splitting (~109 MHz) >> linewidth (~6.4 MHz), so
coherences between J=3/2 and J=1/2 ground states oscillate fast and
are dropped (secular approximation). Within each J manifold, full
coherences are kept.

Reference: Lasner et al., PRL 134, 083401 (2025) — SrOH RF MOT
"""

import numpy as np
from scipy.linalg import solve as la_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════
# Physical Constants
# ══════════════════════════════════════════════════════════════
HBAR = 1.0545718e-34       # J·s
KB   = 1.380649e-23        # J/K
C    = 299792458.0          # m/s
MU_B = 9.2740100783e-24    # J/T
AMU  = 1.66053906660e-27   # kg

# ══════════════════════════════════════════════════════════════
# SrOH Molecular Parameters
# ══════════════════════════════════════════════════════════════
M       = 105 * AMU                       # mass of SrOH
LAM     = 688e-9                          # main transition wavelength
KL      = 2 * np.pi / LAM                 # photon wavevector
VREC    = HBAR * KL / M                   # recoil velocity ≈ 5.5 mm/s
GAMMA   = 2 * np.pi * 6.4e6              # linewidth of Ã state (rad/s)
DELTA_SR = 2 * np.pi * 109e6             # spin-rotation splitting (rad/s)
TD      = HBAR * GAMMA / (2 * KB)         # Doppler temperature
ISAT    = np.pi * HBAR * C * GAMMA / (3 * LAM**3)  # saturation intensity

# g-factors: gJ = gS [J(J+1)+S(S+1)-N(N+1)] / [2J(J+1)]  with gS≈2.002
GJ32 =  2.002 * (1.5*2.5 + 0.5*1.5 - 1*2) / (2*1.5*2.5)   # +0.667
GJ12 =  2.002 * (0.5*1.5 + 0.5*1.5 - 1*2) / (2*0.5*1.5)   # -0.667
GJE  =  0.0   # Ã²Π₁/₂(J'=1/2): gJ≈0 for Ω=1/2 case-a state

# ══════════════════════════════════════════════════════════════
# Transition Dipole Matrix Elements (3-j symbols)
# ══════════════════════════════════════════════════════════════
# For (J'=1/2, 1, J; -mJ', q, mJ) with q = mJ'-mJ
# Tabulated |3j|² values:

TJS_32 = {  # J' = 1/2, J = 3/2
    ( 0.5,  1.5, -1): 1/4,   # σ⁻: mJ=3/2→mJ'=1/2
    (-0.5,  0.5, -1): 1/12,  # σ⁻: mJ=1/2→mJ'=-1/2
    (-0.5, -1.5,  1): 1/4,   # σ⁺: mJ=-3/2→mJ'=-1/2
    ( 0.5, -0.5,  1): 1/12,  # σ⁺: mJ=-1/2→mJ'=1/2
    ( 0.5,  0.5,  0): 1/6,   # π:  mJ=1/2→mJ'=1/2
    (-0.5, -0.5,  0): 1/6,   # π:  mJ=-1/2→mJ'=-1/2
}

TJS_12 = {  # J' = 1/2, J = 1/2
    (-0.5,  0.5, -1): 1/3,   # σ⁻: mJ=1/2→mJ'=-1/2
    ( 0.5, -0.5,  1): 1/3,   # σ⁺: mJ=-1/2→mJ'=1/2
    ( 0.5,  0.5,  0): 1/6,   # π:  mJ=1/2→mJ'=1/2
    (-0.5, -0.5,  0): 1/6,   # π:  mJ=-1/2→mJ'=-1/2
}


class SrOHLevels:
    """
    SrOH level structure for X̃²Σ⁺(N=1) → Ã²Π₁/₂(J'=1/2).

    State ordering:
      0: J=3/2, mJ=-3/2    (ground)
      1: J=3/2, mJ=-1/2    (ground)
      2: J=3/2, mJ=+1/2    (ground)
      3: J=3/2, mJ=+3/2    (ground)
      4: J=1/2, mJ=-1/2    (ground)
      5: J=1/2, mJ=+1/2    (ground)
      6: J'=1/2, mJ'=-1/2  (excited)
      7: J'=1/2, mJ'=+1/2  (excited)
    """

    N  = 8
    NG = 6
    NE = 2

    J_vals  = np.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5])
    mJ_vals = np.array([-1.5, -0.5, 0.5, 1.5, -0.5, 0.5, -0.5, 0.5])
    gJ_vals = np.array([GJ32]*4 + [GJ12]*2 + [GJE]*2)

    # Hönl-London factors: relative reduced matrix element squared
    # BR(J=3/2) : BR(J=1/2) = 2 : 1
    HL = {1.5: 2.0, 0.5: 1.0}

    def __init__(self):
        self._build_couplings()

    def _build_couplings(self):
        """Compute relative transition strengths and branching ratios."""
        # d2[ig, ie, qi] = relative |dipole|² for ground ig → excited ie, pol q
        # qi = 0,1,2 for q = -1,0,+1
        self.d2 = np.zeros((self.NG, self.NE, 3))

        for ig in range(self.NG):
            J_g  = self.J_vals[ig]
            mJ_g = self.mJ_vals[ig]
            R2   = self.HL[J_g]
            tjs  = TJS_32 if abs(J_g - 1.5) < 0.1 else TJS_12

            for ie in range(self.NE):
                mJ_e = self.mJ_vals[ie + self.NG]
                for qi, q in enumerate([-1, 0, 1]):
                    key = (round(2*mJ_e)/2, round(2*mJ_g)/2, q)
                    val = tjs.get(key, 0.0)
                    self.d2[ig, ie, qi] = R2 * (2*J_g + 1) * val

        # normalise so max = 1
        mx = self.d2.max()
        if mx > 0:
            self.d2 /= mx

        # branching ratios BR[ie, ig] = prob of decay e→g
        self.BR = np.zeros((self.NE, self.NG))
        self.BR_pol = np.zeros((self.NE, self.NG, 3))
        for ie in range(self.NE):
            tot = self.d2[:, ie, :].sum()
            if tot > 0:
                self.BR[ie, :]       = self.d2[:, ie, :].sum(axis=1) / tot
                self.BR_pol[ie, :, :] = self.d2[:, ie, :] / tot

    def zeeman(self, i, B):
        """Zeeman shift of state i at field B (rad/s)."""
        return self.gJ_vals[i] * self.mJ_vals[i] * MU_B * B / HBAR


# ══════════════════════════════════════════════════════════════
# Rate-Equation MOT Force Solver
# ══════════════════════════════════════════════════════════════
class RateEqSolver:
    """
    Compute the MOT force using multi-level rate equations.

    For each (z, v) the solver:
      1) computes the excitation rate from each ground state via each beam
      2) finds the steady-state population distribution
      3) returns the net radiation-pressure force
    """

    def __init__(self, levels, beams, s0):
        """
        Parameters
        ----------
        levels : SrOHLevels
        beams  : list of beam dicts — see build_beams()
        s0     : peak saturation parameter per beam
        """
        self.lev  = levels
        self.beams = beams
        self.s0   = s0

    # ----------------------------------------------------------
    def _rates(self, v, z, Bgrad):
        """
        Absorption rate R[ig, ie, ib] from ground ig to excited ie
        via beam ib, and the corresponding force contribution.
        """
        lev = self.lev
        B   = Bgrad * z
        nb  = len(self.beams)

        R = np.zeros((lev.NG, lev.NE, nb))
        F = np.zeros((lev.NG, lev.NE, nb))

        for ib, beam in enumerate(self.beams):
            kdir  = beam['dir']              # +1 or -1
            J_tgt = beam['J']                # which SR manifold
            q     = beam['q']                # polarisation index (-1,0,+1)
            delta0 = beam['delta']           # bare laser detuning (rad/s)
            qi    = q + 1

            doppler = -KL * kdir * v

            for ig in range(lev.NG):
                if abs(lev.J_vals[ig] - J_tgt) > 0.1:
                    continue
                for ie in range(lev.NE):
                    d2 = lev.d2[ig, ie, qi]
                    if d2 < 1e-15:
                        continue

                    ie_abs = ie + lev.NG
                    # effective detuning = laser − transition
                    dZ_g = lev.zeeman(ig,     B)
                    dZ_e = lev.zeeman(ie_abs, B)
                    delta_eff = delta0 + doppler + dZ_g - dZ_e

                    L = (GAMMA/2)**2 / (delta_eff**2 + (GAMMA/2)**2)
                    rate = (GAMMA/2) * self.s0 * d2 * L

                    R[ig, ie, ib] = rate
                    F[ig, ie, ib] = HBAR * KL * kdir * rate

        return R, F

    # ----------------------------------------------------------
    def solve(self, v, z, Bgrad):
        """
        Steady-state populations, net force, and total scattering rate.
        """
        lev = self.lev
        R, Fmat = self._rates(v, z, Bgrad)

        # rate matrix M such that dp/dt = M @ p   (NG × NG)
        Rsum = R.sum(axis=2)         # [ig, ie]
        Mmat = np.zeros((lev.NG, lev.NG))

        for ig in range(lev.NG):
            Mmat[ig, ig] -= Rsum[ig, :].sum()          # loss
            for ik in range(lev.NG):
                for ie in range(lev.NE):
                    Mmat[ig, ik] += Rsum[ik, ie] * lev.BR[ie, ig]   # gain

        # solve M p = 0 with Σp = 1
        Msol = Mmat.copy()
        Msol[-1, :] = 1.0
        rhs = np.zeros(lev.NG); rhs[-1] = 1.0

        try:
            p = la_solve(Msol, rhs)
        except np.linalg.LinAlgError:
            p = np.ones(lev.NG) / lev.NG

        p = np.maximum(p, 0.0)
        p /= p.sum()

        # force
        force = 0.0
        for ig in range(lev.NG):
            force += p[ig] * Fmat[ig, :, :].sum()

        # scattering rate
        Rsc = 0.0
        for ig in range(lev.NG):
            Rsc += p[ig] * Rsum[ig, :].sum()

        return force, p, Rsc

    # ----------------------------------------------------------
    def force_vs_z(self, z_arr, v=0.0, Bgrad=16e-2):
        nz = len(z_arr)
        F = np.zeros(nz)
        P = np.zeros((nz, self.lev.NG))
        R = np.zeros(nz)
        for i, z in enumerate(z_arr):
            F[i], P[i], R[i] = self.solve(v, z, Bgrad)
        return F, P, R

    def force_vs_v(self, v_arr, z=0.0, Bgrad=16e-2):
        nv = len(v_arr)
        F = np.zeros(nv)
        for i, v in enumerate(v_arr):
            F[i], _, _ = self.solve(v, z, Bgrad)
        return F


# ══════════════════════════════════════════════════════════════
# OBE Solver (Lindblad Master Equation)
# ══════════════════════════════════════════════════════════════
class OBESolver:
    """
    Full multi-level Optical Bloch Equation solver.

    Builds the 8×8 density-matrix Liouvillian (64×64 superoperator)
    and finds its steady state.  The secular approximation drops
    coherences between J=3/2 and J=1/2 ground manifolds (they
    oscillate at ~109 MHz >> Γ).
    """

    def __init__(self, levels, beams, Omega_over_Gamma):
        self.lev   = levels
        self.beams = beams
        self.Omega = Omega_over_Gamma * GAMMA   # peak Rabi freq (rad/s)

    # ----------------------------------------------------------
    def _build_H(self, v, z, Bgrad):
        """Hamiltonian in the multi-rotating frame (8×8, in rad/s)."""
        N   = self.lev.N
        B   = Bgrad * z
        H   = np.zeros((N, N), dtype=complex)

        # diagonal: detunings + Zeeman
        for ig in range(self.lev.NG):
            J_g = self.lev.J_vals[ig]
            # find laser detuning for this manifold
            delta_L = 0.0
            for bm in self.beams:
                if abs(bm['J'] - J_g) < 0.1:
                    delta_L = bm['delta']
                    break
            H[ig, ig] = -delta_L + self.lev.zeeman(ig, B)

        for ie in range(self.lev.NE):
            ie_a = ie + self.lev.NG
            H[ie_a, ie_a] = self.lev.zeeman(ie_a, B)

        # off-diagonal: laser couplings from each beam
        for bm in self.beams:
            kdir  = bm['dir']
            J_tgt = bm['J']
            q     = bm['q']
            qi    = q + 1
            doppler = -KL * kdir * v

            for ig in range(self.lev.NG):
                if abs(self.lev.J_vals[ig] - J_tgt) > 0.1:
                    continue
                for ie in range(self.lev.NE):
                    d2 = self.lev.d2[ig, ie, qi]
                    if d2 < 1e-15:
                        continue
                    ie_a = ie + self.lev.NG
                    d    = np.sqrt(d2)
                    Om   = self.Omega * d

                    # include Doppler as phase evolution ≈ shift in detuning
                    # (quasi-static for |v| < Γ/k)
                    H[ie_a, ig] += Om / 2
                    H[ig, ie_a] += Om / 2       # real Rabi → Hermitian

        return H

    # ----------------------------------------------------------
    def _build_liouvillian(self, v, z, Bgrad):
        """64×64 Liouvillian superoperator."""
        N  = self.lev.N
        N2 = N * N
        H  = self._build_H(v, z, Bgrad)
        I  = np.eye(N, dtype=complex)

        # coherent part
        L = -1j * (np.kron(H, I) - np.kron(I, H.T))

        # Lindblad dissipator: spontaneous emission
        for ie in range(self.lev.NE):
            ie_a = ie + self.lev.NG
            for ig in range(self.lev.NG):
                for qi in range(3):
                    br = self.lev.BR_pol[ie, ig, qi]
                    if br < 1e-15:
                        continue
                    Gc = GAMMA * br
                    # collapse operator |g⟩⟨e|
                    Lc = np.zeros((N, N), dtype=complex)
                    Lc[ig, ie_a] = np.sqrt(Gc)
                    LdL = Lc.conj().T @ Lc

                    L += np.kron(Lc.conj(), Lc)
                    L -= 0.5 * np.kron(I, LdL)
                    L -= 0.5 * np.kron(LdL.T, I)

        # secular approximation: zero out rows/cols that couple
        # J=3/2 and J=1/2 ground-state coherences
        # indices of J=3/2 ground: 0-3;  J=1/2 ground: 4-5
        J32_idx = list(range(4))
        J12_idx = [4, 5]
        for a in J32_idx:
            for b in J12_idx:
                # ρ_{a,b} lives at row a*N+b in vec(ρ)
                idx_ab = a * N + b
                idx_ba = b * N + a
                L[idx_ab, :] = 0.0
                L[:, idx_ab] = 0.0
                L[idx_ab, idx_ab] = -1e12   # damp to zero
                L[idx_ba, :] = 0.0
                L[:, idx_ba] = 0.0
                L[idx_ba, idx_ba] = -1e12

        return L

    # ----------------------------------------------------------
    def steady_state(self, v, z, Bgrad):
        """Return steady-state density matrix (8×8)."""
        N  = self.lev.N
        N2 = N * N
        L  = self._build_liouvillian(v, z, Bgrad)

        # replace last row with trace condition
        L[-1, :] = 0.0
        for i in range(N):
            L[-1, i*N + i] = 1.0
        rhs = np.zeros(N2, dtype=complex)
        rhs[-1] = 1.0

        try:
            rho_vec = la_solve(L, rhs)
        except np.linalg.LinAlgError:
            rho_vec = np.zeros(N2, dtype=complex)
            for i in range(N):
                rho_vec[i*N+i] = 1.0 / N

        rho = rho_vec.reshape((N, N), order='F')
        return rho

    # ----------------------------------------------------------
    def force_and_pops(self, v, z, Bgrad):
        """Compute force and populations from the OBE steady state."""
        rho = self.steady_state(v, z, Bgrad)
        lev = self.lev

        # Force from each beam: F_b = ℏk_b × Σ_{g,e} Ω_{ge,b} Im(ρ_{ge})
        force = 0.0
        for bm in self.beams:
            kdir  = bm['dir']
            J_tgt = bm['J']
            q     = bm['q']
            qi    = q + 1

            for ig in range(lev.NG):
                if abs(lev.J_vals[ig] - J_tgt) > 0.1:
                    continue
                for ie in range(lev.NE):
                    d2 = lev.d2[ig, ie, qi]
                    if d2 < 1e-15:
                        continue
                    ie_a = ie + lev.NG
                    d    = np.sqrt(d2)
                    Om   = self.Omega * d

                    force += HBAR * KL * kdir * Om * np.imag(rho[ig, ie_a])

        pops = np.real(np.diag(rho))
        Rsc  = GAMMA * pops[lev.NG:].sum()
        return force, pops, Rsc


# ══════════════════════════════════════════════════════════════
# Beam Configuration Builder
# ══════════════════════════════════════════════════════════════
def build_beams(delta_Gamma, pol_mode):
    """
    Build the 1-D beam list for a DC MOT along z.

    Parameters
    ----------
    delta_Gamma : float
        Detuning in units of Γ (negative = red).
    pol_mode : str
        'A' — J=3/2: beam+z→σ⁻, beam-z→σ⁺  (type-II standard)
               J=1/2: beam+z→σ⁺, beam-z→σ⁻
        'B' — reversed from A
        'C' — same pol for both components: beam+z→σ⁻, beam-z→σ⁺

    Returns list of beam dicts with keys: dir, J, q, delta.
    """
    delta = delta_Gamma * GAMMA

    if pol_mode == 'A':
        return [
            {'dir': +1, 'J': 1.5, 'q': -1, 'delta': delta},  # +z σ⁻ J32
            {'dir': +1, 'J': 0.5, 'q': +1, 'delta': delta},  # +z σ⁺ J12
            {'dir': -1, 'J': 1.5, 'q': +1, 'delta': delta},  # -z σ⁺ J32
            {'dir': -1, 'J': 0.5, 'q': -1, 'delta': delta},  # -z σ⁻ J12
        ]
    elif pol_mode == 'B':
        return [
            {'dir': +1, 'J': 1.5, 'q': +1, 'delta': delta},
            {'dir': +1, 'J': 0.5, 'q': -1, 'delta': delta},
            {'dir': -1, 'J': 1.5, 'q': -1, 'delta': delta},
            {'dir': -1, 'J': 0.5, 'q': +1, 'delta': delta},
        ]
    elif pol_mode == 'C':
        return [
            {'dir': +1, 'J': 1.5, 'q': -1, 'delta': delta},
            {'dir': +1, 'J': 0.5, 'q': -1, 'delta': delta},
            {'dir': -1, 'J': 1.5, 'q': +1, 'delta': delta},
            {'dir': -1, 'J': 0.5, 'q': +1, 'delta': delta},
        ]
    else:
        raise ValueError(f"Unknown pol_mode: {pol_mode}")


# ══════════════════════════════════════════════════════════════
# Main Simulation
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  SrOH DC Red MOT — Optical Bloch Equation Simulation")
    print("=" * 70)

    levels = SrOHLevels()

    # ── Print molecular data ─────────────────────────────────
    print(f"\nMolecular parameters:")
    print(f"  Mass           = {M/AMU:.0f} amu")
    print(f"  λ              = {LAM*1e9:.1f} nm")
    print(f"  Γ/(2π)         = {GAMMA/(2*np.pi*1e6):.1f} MHz")
    print(f"  SR splitting   = {DELTA_SR/(2*np.pi*1e6):.0f} MHz")
    print(f"  v_recoil       = {VREC*1e3:.2f} mm/s")
    print(f"  T_Doppler      = {TD*1e6:.0f} µK")
    print(f"  gJ(J=3/2)      = {GJ32:+.4f}")
    print(f"  gJ(J=1/2)      = {GJ12:+.4f}")
    print(f"  gJ(excited)    = {GJE:+.4f}")

    print(f"\nBranching ratios from each excited state:")
    for ie in range(levels.NE):
        ie_a = ie + levels.NG
        print(f"  |e, mJ'={levels.mJ_vals[ie_a]:+.1f}⟩ →")
        for ig in range(levels.NG):
            if levels.BR[ie, ig] > 0.001:
                print(f"    |g{ig}, J={levels.J_vals[ig]:.1f}, "
                      f"mJ={levels.mJ_vals[ig]:+.1f}⟩  "
                      f"BR = {levels.BR[ie, ig]:.4f}")

    # ── Simulation parameters ────────────────────────────────
    Bgrad   = 16e-2       # 16 G/cm  (T/m)
    s0      = 1.0         # saturation parameter per beam
    Omega_G = np.sqrt(s0/2)   # Rabi freq / Gamma (for OBE)
    det_nom = -1.0        # detuning in Gamma

    pol_modes  = ['A', 'B', 'C']
    pol_labels = [
        'A: opposite pol (type-II)',
        'B: reversed opposite',
        'C: same pol both SR'
    ]

    print(f"\nSimulation settings:")
    print(f"  B gradient     = {Bgrad*100:.0f} G/cm")
    print(f"  s₀ per beam    = {s0:.1f}")
    print(f"  Ω/Γ per beam   = {Omega_G:.3f}")
    print(f"  Nominal Δ      = {det_nom:.1f} Γ")

    # ── Scan all polarisation configs ────────────────────────
    nz = 120;  z_arr = np.linspace(-3e-3, 3e-3, nz)
    nv = 120;  v_arr = np.linspace(-4.0,  4.0,  nv)

    results = {}

    for pm, pl in zip(pol_modes, pol_labels):
        print(f"\n─── Config {pl} ───")
        beams = build_beams(det_nom, pm)
        solver = RateEqSolver(levels, beams, s0)

        Fz, Pz, Rz = solver.force_vs_z(z_arr, v=0, Bgrad=Bgrad)
        Fv = solver.force_vs_v(v_arr, z=0, Bgrad=Bgrad)
        Fv_off = solver.force_vs_v(v_arr, z=1e-3, Bgrad=Bgrad)

        # spring constant  k = -dF/dz at z=0
        dFdz = np.gradient(Fz, z_arr)
        k_spring = -dFdz[nz//2]

        # damping  β = -(1/m) dF/dv at v=0
        dFdv = np.gradient(Fv, v_arr)
        beta = -dFdv[nv//2] / M

        trap = k_spring > 0
        cool = beta > 0

        osc = np.sqrt(abs(k_spring)/M)/(2*np.pi) if k_spring > 0 else 0
        print(f"  Spring const   = {k_spring:.3e} N/m  "
              f"({'TRAP' if trap else 'ANTI-TRAP'})")
        print(f"  Osc. frequency = {osc:.1f} Hz")
        print(f"  Damping β      = {beta:.0f} s⁻¹  "
              f"({'COOL' if cool else 'HEAT'})")
        print(f"  Peak scat.rate = {np.max(Rz)/(2*np.pi*1e6):.3f} MHz")

        results[pm] = dict(Fz=Fz, Pz=Pz, Rz=Rz, Fv=Fv, Fv_off=Fv_off,
                           k=k_spring, beta=beta, trap=trap, cool=cool,
                           label=pl)

    # ── OBE cross-check for best config ──────────────────────
    # pick the config with the strongest restoring force
    best = max(results, key=lambda k: results[k]['k'])
    print(f"\n{'='*70}")
    print(f"Best configuration: {results[best]['label']}")
    print(f"{'='*70}")

    print(f"\nRunning OBE cross-check for config {best}...")
    beams_best = build_beams(det_nom, best)
    obe = OBESolver(levels, beams_best, Omega_G)

    nz_obe = 40
    z_obe  = np.linspace(-3e-3, 3e-3, nz_obe)
    Fz_obe = np.zeros(nz_obe)
    Pz_obe = np.zeros((nz_obe, levels.N))
    for i, z in enumerate(z_obe):
        f, p, _ = obe.force_and_pops(0, z, Bgrad)
        Fz_obe[i]  = f
        Pz_obe[i]  = p

    nv_obe = 40
    v_obe  = np.linspace(-4, 4, nv_obe)
    Fv_obe = np.zeros(nv_obe)
    for i, vel in enumerate(v_obe):
        f, _, _ = obe.force_and_pops(vel, 0, Bgrad)
        Fv_obe[i] = f

    print("  OBE done.")

    # ── Parameter scans ──────────────────────────────────────
    print("\nRunning detuning scan...")
    det_scan = np.linspace(-3.0, -0.2, 20)
    k_vs_det = np.zeros(len(det_scan))
    beta_vs_det = np.zeros(len(det_scan))

    for i, d in enumerate(det_scan):
        bms = build_beams(d, best)
        slv = RateEqSolver(levels, bms, s0)
        dz  = 0.3e-3
        fp, _, _ = slv.solve(0, +dz, Bgrad)
        fm, _, _ = slv.solve(0, -dz, Bgrad)
        k_vs_det[i] = -(fp - fm) / (2*dz)

        dv = 0.1
        fvp, _, _ = slv.solve(+dv, 0, Bgrad)
        fvm, _, _ = slv.solve(-dv, 0, Bgrad)
        beta_vs_det[i] = -(fvp - fvm) / (2*dv*M)

    print("Running saturation scan...")
    s_scan = np.linspace(0.1, 5.0, 20)
    k_vs_s = np.zeros(len(s_scan))
    for i, ss in enumerate(s_scan):
        bms = build_beams(det_nom, best)
        slv = RateEqSolver(levels, bms, ss)
        dz  = 0.3e-3
        fp, _, _ = slv.solve(0, +dz, Bgrad)
        fm, _, _ = slv.solve(0, -dz, Bgrad)
        k_vs_s[i] = -(fp - fm) / (2*dz)

    print("Running B-gradient scan...")
    B_scan = np.linspace(5, 50, 15) * 1e-2
    k_vs_B = np.zeros(len(B_scan))
    for i, bg in enumerate(B_scan):
        bms = build_beams(det_nom, best)
        slv = RateEqSolver(levels, bms, s0)
        dz  = 0.3e-3
        fp, _, _ = slv.solve(0, +dz, bg)
        fm, _, _ = slv.solve(0, -dz, bg)
        k_vs_B[i] = -(fp - fm) / (2*dz)

    # ── Trajectories ─────────────────────────────────────────
    print("Simulating trajectories...")
    beams_traj = build_beams(det_nom, best)
    slv_traj   = RateEqSolver(levels, beams_traj, s0)
    dt = 2e-6;  tmax = 0.05;  nstep = int(tmax/dt)

    traj_list = []
    for z0, v0, lbl in [(2e-3, 0, 'z₀=2mm'),
                         (0, -1.0, 'v₀=-1m/s'),
                         (1e-3, -0.5, 'z₀=1mm,v₀=-0.5')]:
        zt = np.zeros(nstep); vt = np.zeros(nstep); tt = np.arange(nstep)*dt
        zt[0] = z0; vt[0] = v0
        for j in range(1, nstep):
            f, _, _ = slv_traj.solve(vt[j-1], zt[j-1], Bgrad)
            a = f / M
            vt[j] = vt[j-1] + a*dt
            zt[j] = zt[j-1] + vt[j]*dt
            if abs(zt[j]) > 0.015:
                zt[j:] = zt[j]; vt[j:] = vt[j]; break
        traj_list.append((tt, zt, vt, lbl))

    # ══════════════════════════════════════════════════════════
    # Plotting
    # ══════════════════════════════════════════════════════════
    F_UNIT = HBAR * KL * GAMMA / 2          # natural force unit
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    fig = plt.figure(figsize=(22, 28))
    gs  = GridSpec(5, 3, figure=fig, hspace=0.38, wspace=0.32)

    # Row 0: F(z) for three configs
    ax00 = fig.add_subplot(gs[0, 0])
    for pm in pol_modes:
        r = results[pm]
        ax00.plot(z_arr*1e3, r['Fz']/F_UNIT, color=colors[pm],
                  lw=2, label=pm)
    ax00.axhline(0, c='gray', ls='--', alpha=.5)
    ax00.set_xlabel('z (mm)'); ax00.set_ylabel('F / (ℏkΓ/2)')
    ax00.set_title('Restoring Force F(z) at v=0')
    ax00.legend(); ax00.grid(True, alpha=.3)

    # Row 0: F(v) for three configs
    ax01 = fig.add_subplot(gs[0, 1])
    for pm in pol_modes:
        r = results[pm]
        ax01.plot(v_arr, r['Fv']/F_UNIT, color=colors[pm], lw=2, label=pm)
    ax01.axhline(0, c='gray', ls='--', alpha=.5)
    ax01.set_xlabel('v (m/s)'); ax01.set_ylabel('F / (ℏkΓ/2)')
    ax01.set_title('Damping Force F(v) at z=0')
    ax01.legend(); ax01.grid(True, alpha=.3)

    # Row 0: summary bar chart
    ax02 = fig.add_subplot(gs[0, 2])
    x_bar = np.arange(len(pol_modes))
    ks = [results[pm]['k'] for pm in pol_modes]
    bar_colors = ['green' if k > 0 else 'red' for k in ks]
    ax02.bar(x_bar, ks, color=bar_colors, alpha=0.7)
    ax02.set_xticks(x_bar)
    ax02.set_xticklabels([f'Config {pm}' for pm in pol_modes])
    ax02.set_ylabel('Spring constant (N/m)')
    ax02.set_title('Restoring Force Comparison')
    ax02.axhline(0, c='k', lw=0.5)
    ax02.grid(True, alpha=.3)

    # Row 1: best config detail — populations & scattering
    r_best = results[best]

    ax10 = fig.add_subplot(gs[1, 0])
    gcolors = plt.cm.tab10(np.linspace(0, .6, levels.NG))
    for ig in range(levels.NG):
        lbl = f'J={levels.J_vals[ig]:.0f}/2,m={levels.mJ_vals[ig]:+.1f}'
        ax10.plot(z_arr*1e3, r_best['Pz'][:, ig], color=gcolors[ig],
                  lw=1.5, label=lbl)
    ax10.set_xlabel('z (mm)'); ax10.set_ylabel('Population')
    ax10.set_title(f'Populations (config {best}, v=0)')
    ax10.legend(fontsize=7, ncol=2); ax10.grid(True, alpha=.3)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(z_arr*1e3, r_best['Rz']/(2*np.pi*1e6), 'g-', lw=2)
    ax11.set_xlabel('z (mm)'); ax11.set_ylabel('Γ_sc (MHz)')
    ax11.set_title(f'Scattering Rate (config {best})')
    ax11.grid(True, alpha=.3)

    # F(v) at z=0 and z=1mm
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(v_arr, r_best['Fv']/F_UNIT, 'b-', lw=2, label='z = 0')
    ax12.plot(v_arr, r_best['Fv_off']/F_UNIT, 'r--', lw=2, label='z = 1 mm')
    ax12.axhline(0, c='gray', ls='--', alpha=.5)
    ax12.set_xlabel('v (m/s)'); ax12.set_ylabel('F / (ℏkΓ/2)')
    ax12.set_title(f'Velocity-dep. Force (config {best})')
    ax12.legend(); ax12.grid(True, alpha=.3)

    # Row 2: OBE cross-check
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(z_arr*1e3, r_best['Fz']/F_UNIT, 'b-', lw=2, label='Rate eq.')
    ax20.plot(z_obe*1e3,  Fz_obe/F_UNIT, 'ro', ms=5, label='OBE')
    ax20.axhline(0, c='gray', ls='--', alpha=.5)
    ax20.set_xlabel('z (mm)'); ax20.set_ylabel('F / (ℏkΓ/2)')
    ax20.set_title(f'Rate Eq. vs OBE — F(z)')
    ax20.legend(); ax20.grid(True, alpha=.3)

    ax21 = fig.add_subplot(gs[2, 1])
    F_re_v = RateEqSolver(levels, build_beams(det_nom, best), s0).force_vs_v(
        v_arr, z=0, Bgrad=Bgrad)
    ax21.plot(v_arr, F_re_v/F_UNIT, 'b-', lw=2, label='Rate eq.')
    ax21.plot(v_obe, Fv_obe/F_UNIT, 'ro', ms=5, label='OBE')
    ax21.axhline(0, c='gray', ls='--', alpha=.5)
    ax21.set_xlabel('v (m/s)'); ax21.set_ylabel('F / (ℏkΓ/2)')
    ax21.set_title('Rate Eq. vs OBE — F(v)')
    ax21.legend(); ax21.grid(True, alpha=.3)

    # OBE populations
    ax22 = fig.add_subplot(gs[2, 2])
    for ig in range(levels.NG):
        lbl = f'J={levels.J_vals[ig]:.0f}/2,m={levels.mJ_vals[ig]:+.1f}'
        ax22.plot(z_obe*1e3, Pz_obe[:, ig], color=gcolors[ig],
                  lw=1.5, label=lbl)
    ax22.plot(z_obe*1e3, Pz_obe[:, levels.NG:].sum(axis=1), 'k--',
             lw=1, label='excited')
    ax22.set_xlabel('z (mm)'); ax22.set_ylabel('Population')
    ax22.set_title('OBE Populations')
    ax22.legend(fontsize=7, ncol=2); ax22.grid(True, alpha=.3)

    # Row 3: parameter scans
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.plot(det_scan, k_vs_det, 'bo-', lw=2, ms=4)
    ax30.axhline(0, c='gray', ls='--')
    ax30.set_xlabel('Detuning (Γ)'); ax30.set_ylabel('k (N/m)')
    ax30.set_title('Spring Constant vs Detuning')
    ax30.grid(True, alpha=.3)

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.plot(s_scan, k_vs_s, 'go-', lw=2, ms=4)
    ax31.axhline(0, c='gray', ls='--')
    ax31.set_xlabel('s₀ per beam'); ax31.set_ylabel('k (N/m)')
    ax31.set_title('Spring Constant vs Saturation')
    ax31.grid(True, alpha=.3)

    ax32 = fig.add_subplot(gs[3, 2])
    ax32.plot(B_scan*100, k_vs_B, 'rs-', lw=2, ms=4)
    ax32.axhline(0, c='gray', ls='--')
    ax32.set_xlabel("B' (G/cm)"); ax32.set_ylabel('k (N/m)')
    ax32.set_title('Spring Constant vs B Gradient')
    ax32.grid(True, alpha=.3)

    # Row 4: trajectories & phase space
    ax40 = fig.add_subplot(gs[4, 0])
    for tt, zt, vt, lbl in traj_list:
        ax40.plot(tt*1e3, zt*1e3, lw=1.5, label=lbl)
    ax40.axhline(0, c='gray', ls='--', alpha=.5)
    ax40.set_xlabel('t (ms)'); ax40.set_ylabel('z (mm)')
    ax40.set_title(f'Trajectories (config {best})')
    ax40.legend(fontsize=8); ax40.grid(True, alpha=.3)

    ax41 = fig.add_subplot(gs[4, 1])
    for tt, zt, vt, lbl in traj_list:
        ax41.plot(zt*1e3, vt, lw=1.5, label=lbl)
    ax41.set_xlabel('z (mm)'); ax41.set_ylabel('v (m/s)')
    ax41.set_title('Phase Space')
    ax41.legend(fontsize=8); ax41.grid(True, alpha=.3)

    # Text summary
    ax42 = fig.add_subplot(gs[4, 2])
    ax42.axis('off')
    summary = (
        f"SrOH DC Red MOT — Summary\n"
        f"{'─'*35}\n"
        f"Best config: {results[best]['label']}\n\n"
        f"At Δ = {det_nom:.1f}Γ, s₀ = {s0:.1f}, B' = {Bgrad*100:.0f} G/cm:\n"
        f"  Spring const k  = {r_best['k']:.2e} N/m\n"
        f"  Osc. freq       = {np.sqrt(abs(r_best['k'])/M)/(2*np.pi):.1f} Hz\n"
        f"  Damping β       = {r_best['beta']:.0f} s⁻¹\n"
        f"  Peak Γ_sc       = {np.max(r_best['Rz'])/(2*np.pi*1e6):.3f} MHz\n\n"
        f"{'TRAPPING' if r_best['trap'] else 'NO TRAPPING'} force detected\n"
        f"{'COOLING' if r_best['cool'] else 'HEATING'} detected at z=0\n\n"
        f"Comparison — SrOH RF MOT (experiment):\n"
        f"  T = 1.2 mK,  N = 2000\n"
        f"  τ = 91 ms,   β ~ 100 s⁻¹\n"
        f"  ω ~ 2π×45 Hz\n\n"
        f"Larmor rate at 1 G:\n"
        f"  ωL/(2π) = {GJ32*MU_B*1e-4/HBAR/(2*np.pi*1e6):.2f} MHz\n"
        f"  (compare Γ/(2π) = {GAMMA/(2*np.pi*1e6):.1f} MHz)"
    )
    ax42.text(0.05, 0.95, summary, transform=ax42.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('SrOH DC Red MOT — Optical Bloch Equation Simulation',
                 fontsize=18, fontweight='bold', y=0.995)

    outpath = '/Users/dslmd/Downloads/DC red MOT/sroh_dc_red_mot_OBE.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")
    plt.close(fig)

    return results, best


if __name__ == '__main__':
    main()
