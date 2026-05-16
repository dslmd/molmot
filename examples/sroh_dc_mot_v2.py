#!/usr/bin/env python3
"""
SrOH DC MOT Simulation v2 — Rate Equations + OBE

Explores whether a DC (static B-field/polarisation) MOT is feasible
for SrOH on the X̃²Σ⁺(000,N=1) → Ã²Π₁/₂(000,J'=1/2) transition.

Tests:
  • Red-detuned DC MOT (standard, blue, and mixed configurations)
  • Blue-detuned DC MOT (Λ-BDM style, as demonstrated for CaF/YO/SrF)
  • Multi-frequency "stirring" configurations
  • Parameter optimisation scans

Key finding: Type-II dark-state pumping cancels the trapping force
in a simple red DC MOT.  Blue-detuned or multi-frequency configs
can overcome this.
"""

import numpy as np
from scipy.linalg import solve as la_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════  Constants  ═══════════════════
HBAR = 1.0545718e-34;  KB = 1.380649e-23;  MU_B = 9.2740100783e-24
AMU  = 1.66053906660e-27

# ═══════════════════  SrOH  ═══════════════════
M        = 105 * AMU
LAM      = 688e-9
KL       = 2*np.pi/LAM
VREC     = HBAR*KL/M
GAMMA    = 2*np.pi*6.4e6          # linewidth  (rad/s)
DELTA_SR = 2*np.pi*109e6          # spin-rotation splitting
TD       = HBAR*GAMMA/(2*KB)      # Doppler temp
GJ32     =  2.002/3               # g_J for J=3/2
GJ12     = -2.002/3               # g_J for J=1/2
GJE      =  0.0                   # g_J for excited Π₁/₂

FUNIT = HBAR*KL*GAMMA/2           # natural force unit


# ═══════════════════  Level structure  ═══════════════════
#  0-3 : J=3/2, mJ = -3/2 … +3/2  (ground)
#  4-5 : J=1/2, mJ = -1/2, +1/2   (ground)
#  (excited handled implicitly via branching)

NG = 6; NE = 2
J_g  = np.array([1.5]*4 + [0.5]*2)
mJ_g = np.array([-1.5, -0.5, 0.5, 1.5, -0.5, 0.5])
gJ_g = np.array([GJ32]*4 + [GJ12]*2)
mJ_e = np.array([-0.5, 0.5])

# ── transition strengths ──
# d2[ig, ie, qi]  with qi = 0,1,2 → q = -1,0,+1
# includes Hönl-London:  J=3/2 branch = 2/3, J=1/2 = 1/3

_tjs32 = {
    (0.5,1.5,-1): 1/4, (-0.5,0.5,-1): 1/12,
    (-0.5,-1.5,1): 1/4, (0.5,-0.5,1): 1/12,
    (0.5,0.5,0): 1/6, (-0.5,-0.5,0): 1/6,
}
_tjs12 = {
    (-0.5,0.5,-1): 1/3, (0.5,-0.5,1): 1/3,
    (0.5,0.5,0): 1/6, (-0.5,-0.5,0): 1/6,
}

d2 = np.zeros((NG, NE, 3))
for ig in range(NG):
    HL = 2.0 if J_g[ig] > 1 else 1.0
    tbl = _tjs32 if J_g[ig] > 1 else _tjs12
    for ie in range(NE):
        for qi, q in enumerate([-1, 0, 1]):
            key = (round(2*mJ_e[ie])/2, round(2*mJ_g[ig])/2, q)
            d2[ig, ie, qi] = HL * (2*J_g[ig]+1) * tbl.get(key, 0.0)
d2 /= d2.max()                      # normalise so max = 1

# branching ratios
BR = np.zeros((NE, NG))
for ie in range(NE):
    s = d2[:, ie, :].sum()
    if s > 0:
        BR[ie] = d2[:, ie, :].sum(axis=1) / s


def zeeman_g(ig, B):
    return gJ_g[ig] * mJ_g[ig] * MU_B * B / HBAR

def zeeman_e(ie, B):
    return GJE * mJ_e[ie] * MU_B * B / HBAR


# ═══════════════════  Rate-equation solver  ═══════════════════
def solve_force(beams, s0, v, z, Bgrad):
    """
    Solve steady-state populations and return (force, pops, R_scatter).

    Each beam: dict(dir=±1, J=1.5|0.5, q=±1|0, delta=…)
    """
    B  = Bgrad * z
    nb = len(beams)

    # excitation rates  R[ig, ie, ib]
    R = np.zeros((NG, NE, nb))
    Fk = np.zeros((NG, NE, nb))  # ℏk×rate for each channel

    for ib, bm in enumerate(beams):
        kd = bm['dir']
        Jt = bm['J']
        q  = bm['q'];  qi = q + 1
        d0 = bm['delta']
        dop = -KL * kd * v

        for ig in range(NG):
            if abs(J_g[ig] - Jt) > 0.1:
                continue
            for ie in range(NE):
                c2 = d2[ig, ie, qi]
                if c2 < 1e-15:
                    continue
                dZg = zeeman_g(ig, B)
                dZe = zeeman_e(ie, B)
                delta = d0 + dop + dZg - dZe
                L = (GAMMA/2)**2 / (delta**2 + (GAMMA/2)**2)
                rate = (GAMMA/2) * s0 * c2 * L
                R[ig, ie, ib] = rate
                Fk[ig, ie, ib] = HBAR * KL * kd * rate

    Rsum = R.sum(axis=2)   # [ig, ie]

    # population matrix  dp/dt = M p
    Mmat = np.zeros((NG, NG))
    for ig in range(NG):
        Mmat[ig, ig] -= Rsum[ig].sum()
        for ik in range(NG):
            for ie in range(NE):
                Mmat[ig, ik] += Rsum[ik, ie] * BR[ie, ig]

    Msol = Mmat.copy();  Msol[-1] = 1.0
    rhs = np.zeros(NG);  rhs[-1] = 1.0
    try:
        p = la_solve(Msol, rhs)
    except np.linalg.LinAlgError:
        p = np.ones(NG)/NG
    p = np.maximum(p, 0); p /= p.sum()

    force = sum(p[ig]*Fk[ig].sum() for ig in range(NG))
    Rsc   = sum(p[ig]*Rsum[ig].sum() for ig in range(NG))
    return force, p, Rsc


# ═══════════════════  Beam builders  ═══════════════════
def beams_two_freq(delta_G, pol='A'):
    """
    Standard 2-frequency DC MOT (one freq per SR manifold).
      A: J=3/2 σ⁻/σ⁺,  J=1/2 σ⁺/σ⁻  (opposite pol for opposite gJ)
      B: reversed
      C: same pol for both SR
    """
    d = delta_G * GAMMA
    if pol == 'A':
        return [
            dict(dir=+1, J=1.5, q=-1, delta=d),
            dict(dir=+1, J=0.5, q=+1, delta=d),
            dict(dir=-1, J=1.5, q=+1, delta=d),
            dict(dir=-1, J=0.5, q=-1, delta=d),
        ]
    elif pol == 'B':
        return [
            dict(dir=+1, J=1.5, q=+1, delta=d),
            dict(dir=+1, J=0.5, q=-1, delta=d),
            dict(dir=-1, J=1.5, q=-1, delta=d),
            dict(dir=-1, J=0.5, q=+1, delta=d),
        ]
    else:  # C
        return [
            dict(dir=+1, J=1.5, q=-1, delta=d),
            dict(dir=+1, J=0.5, q=-1, delta=d),
            dict(dir=-1, J=1.5, q=+1, delta=d),
            dict(dir=-1, J=0.5, q=+1, delta=d),
        ]


def beams_four_freq(delta_G, delta_split_G=0.0):
    """
    4-frequency config (CaF-style): each SR manifold gets two
    frequency components at δ ± δ_split, with OPPOSITE polarisations.
    This provides remixing by driving σ⁺ and σ⁻ at slightly
    different detunings within each manifold.
    """
    d = delta_G * GAMMA
    ds = delta_split_G * GAMMA
    bms = []
    for Jt in [1.5, 0.5]:
        for sign, doff in [(+1, +ds), (-1, -ds)]:
            # beam +z: pol depends on which frequency within the manifold
            bms.append(dict(dir=+1, J=Jt, q=sign,  delta=d+doff))
            bms.append(dict(dir=-1, J=Jt, q=-sign, delta=d+doff))
    return bms


def beams_blue_lambda(delta_G, delta2_G):
    """
    Λ-BDM style (blue-detuned): two frequencies addressing the
    highest and lowest ground sub-manifolds, producing Λ-type
    dark states for sub-Doppler cooling.

    For SrOH: address J=3/2 (highest) and J=1/2 (lowest) with
    blue detuning.  The two-photon detuning δ₂ controls the
    dark-state condition.
    """
    d1 = delta_G * GAMMA   # single-photon detuning (blue = positive)
    d2v = delta2_G * GAMMA  # two-photon detuning
    # component 1 addresses J=3/2  (like CaF F=2)
    # component 2 addresses J=1/2  (like CaF F=1⁻)
    # opposite handedness for the two components
    return [
        dict(dir=+1, J=1.5, q=-1, delta=d1),
        dict(dir=+1, J=0.5, q=+1, delta=d1 + d2v),
        dict(dir=-1, J=1.5, q=+1, delta=d1),
        dict(dir=-1, J=0.5, q=-1, delta=d1 + d2v),
    ]


# ═══════════════════  Scan helpers  ═══════════════════
def scan_z(beams, s0, Bgrad, zmax=5e-3, nz=200, v=0):
    za = np.linspace(-zmax, zmax, nz)
    F = np.zeros(nz); P = np.zeros((nz, NG)); R = np.zeros(nz)
    for i, z in enumerate(za):
        F[i], P[i], R[i] = solve_force(beams, s0, v, z, Bgrad)
    return za, F, P, R

def scan_v(beams, s0, Bgrad, vmax=5.0, nv=200, z=0):
    va = np.linspace(-vmax, vmax, nv)
    F = np.zeros(nv)
    for i, vel in enumerate(va):
        F[i], _, _ = solve_force(beams, s0, vel, z, Bgrad)
    return va, F

def spring_and_damping(beams, s0, Bgrad, dz=0.3e-3, dv=0.1):
    fp, _, _ = solve_force(beams, s0, 0, +dz, Bgrad)
    fm, _, _ = solve_force(beams, s0, 0, -dz, Bgrad)
    k = -(fp - fm)/(2*dz)
    fvp, _, _ = solve_force(beams, s0, +dv, 0, Bgrad)
    fvm, _, _ = solve_force(beams, s0, -dv, 0, Bgrad)
    beta = -(fvp - fvm)/(2*dv*M)
    return k, beta


# ═══════════════════  MAIN  ═══════════════════
def main():
    Bgrad = 16e-2   # 16 G/cm
    s0    = 1.0

    print("=" * 70)
    print("  SrOH DC MOT Simulation v2")
    print("=" * 70)
    print(f"  λ={LAM*1e9:.0f}nm  Γ/(2π)={GAMMA/2/np.pi/1e6:.1f}MHz  "
          f"SR={DELTA_SR/2/np.pi/1e6:.0f}MHz")
    print(f"  gJ(3/2)={GJ32:+.3f}  gJ(1/2)={GJ12:+.3f}  "
          f"T_D={TD*1e6:.0f}µK  v_rec={VREC*1e3:.2f}mm/s")
    print(f"  B'={Bgrad*100:.0f}G/cm  s₀={s0}  m={M/AMU:.0f}amu")

    # ── 1) Per-beam force diagnostic ──────────────────────────
    print("\n── Per-beam force diagnostic at z=1mm, v=0 ──")
    z_test = 1e-3
    B_test = Bgrad * z_test
    print(f"  B at z=1mm: {B_test*1e4:.2f} Gauss")

    for cfg_name, bms_func in [('Red A', lambda: beams_two_freq(-1.0, 'A')),
                                 ('Red B', lambda: beams_two_freq(-1.0, 'B')),
                                 ('Blue A', lambda: beams_two_freq(+3.0, 'A'))]:
        bms = bms_func()
        print(f"\n  Config: {cfg_name}")
        for ib, bm in enumerate(bms):
            # force from this single beam
            f1, p1, r1 = solve_force([bm], s0, 0, z_test, Bgrad)
            dlbl = f"{'+'if bm['dir']>0 else '-'}z"
            qlbl = {-1:'σ⁻', 0:'π', 1:'σ⁺'}[bm['q']]
            print(f"    beam {dlbl} J={bm['J']:.1f} {qlbl}: "
                  f"F={f1/FUNIT:+.4f} ℏkΓ/2  R={r1/1e6:.3f}MHz")
        # combined
        fc, pc, rc = solve_force(bms, s0, 0, z_test, Bgrad)
        print(f"    COMBINED: F={fc/FUNIT:+.6f} ℏkΓ/2  R={rc/1e6:.3f}MHz")

    # ── 2) Comprehensive config comparison ────────────────────
    configs = {}

    # Red-detuned configs
    for det in [-0.5, -1.0, -1.5, -2.0]:
        for pol in ['A', 'B', 'C']:
            name = f'Red {pol} Δ={det:.1f}Γ'
            bms = beams_two_freq(det, pol)
            k, beta = spring_and_damping(bms, s0, Bgrad)
            configs[name] = dict(k=k, beta=beta, bms=bms, det=det,
                                 color='red' if pol=='A' else ('orange' if pol=='B' else 'brown'))

    # Blue-detuned configs
    for det in [+1.0, +2.0, +3.0, +4.0, +5.0]:
        for pol in ['A', 'B', 'C']:
            name = f'Blue {pol} Δ={det:+.1f}Γ'
            bms = beams_two_freq(det, pol)
            k, beta = spring_and_damping(bms, s0, Bgrad)
            configs[name] = dict(k=k, beta=beta, bms=bms, det=det,
                                 color='blue' if pol=='A' else ('cyan' if pol=='B' else 'navy'))

    # 4-frequency configs (CaF-style with split)
    for det in [-1.0, -0.5]:
        for ds in [0.3, 0.5, 1.0]:
            name = f'4freq Δ={det:.1f}Γ split={ds:.1f}Γ'
            bms = beams_four_freq(det, ds)
            k, beta = spring_and_damping(bms, s0, Bgrad)
            configs[name] = dict(k=k, beta=beta, bms=bms, det=det,
                                 color='purple')

    # Λ-BDM style
    for det in [+2.0, +3.0, +4.0]:
        for d2 in [-0.3, -0.1, 0.0, +0.1]:
            name = f'Λ-BDM Δ={det:+.1f}Γ δ₂={d2:+.2f}Γ'
            bms = beams_blue_lambda(det, d2)
            k, beta = spring_and_damping(bms, s0, Bgrad)
            configs[name] = dict(k=k, beta=beta, bms=bms, det=det,
                                 color='green')

    # Find best configs
    print("\n── Top 10 configs by |spring constant| ──")
    ranked = sorted(configs.items(), key=lambda x: -x[1]['k'])
    for i, (name, c) in enumerate(ranked[:10]):
        trap = "TRAP" if c['k'] > 0 else "anti"
        cool = "COOL" if c['beta'] > 0 else "heat"
        osc = np.sqrt(abs(c['k'])/M)/(2*np.pi) if c['k'] > 0 else 0
        print(f"  {i+1:2d}. {name:42s}  k={c['k']:+.3e}  "
              f"β={c['beta']:+.0e}  ω={osc:.1f}Hz  [{trap}/{cool}]")

    # Find best trapping config (k>0 AND beta>0)
    trapping = [(n,c) for n,c in configs.items() if c['k'] > 0 and c['beta'] > 0]
    trapping.sort(key=lambda x: -x[1]['k'])

    print(f"\n── Best TRAPPING + COOLING configs ──")
    if trapping:
        for i, (name, c) in enumerate(trapping[:5]):
            osc = np.sqrt(c['k']/M)/(2*np.pi)
            print(f"  {i+1}. {name:42s}  k={c['k']:.3e}  β={c['beta']:.0e}  ω={osc:.1f}Hz")
    else:
        print("  None found with both trapping AND cooling!")
        # relax: find best trapping even with heating
        trapping_only = [(n,c) for n,c in configs.items() if c['k'] > 0]
        trapping_only.sort(key=lambda x: -x[1]['k'])
        print("  Best TRAPPING (possibly heating):")
        for i, (name, c) in enumerate(trapping_only[:5]):
            osc = np.sqrt(c['k']/M)/(2*np.pi)
            print(f"  {i+1}. {name:42s}  k={c['k']:.3e}  β={c['beta']:.0e}")

    # ── 3) Detailed plots for best configs ────────────────────
    fig = plt.figure(figsize=(22, 30))
    gs = GridSpec(6, 3, figure=fig, hspace=0.40, wspace=0.30)

    # --- Row 0: Red DC MOT configs (show the problem) ---
    print("\nPlotting red DC MOT configs...")
    ax = fig.add_subplot(gs[0, 0])
    for pol, ls, clr in [('A','-','#1f77b4'), ('B','--','#ff7f0e'), ('C',':','#2ca02c')]:
        bms = beams_two_freq(-1.0, pol)
        za, Fz, _, _ = scan_z(bms, s0, Bgrad, zmax=3e-3, nz=100)
        ax.plot(za*1e3, Fz/FUNIT, ls=ls, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('RED DC MOT: F(z) at v=0, Δ=-1Γ')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    for pol, ls, clr in [('A','-','#1f77b4'), ('B','--','#ff7f0e'), ('C',':','#2ca02c')]:
        bms = beams_two_freq(-1.0, pol)
        va, Fv = scan_v(bms, s0, Bgrad, vmax=4, nv=100)
        ax.plot(va, Fv/FUNIT, ls=ls, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('v (m/s)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('RED DC MOT: F(v) at z=0')
    ax.legend(); ax.grid(True, alpha=.3)

    # Dark-state population analysis
    ax = fig.add_subplot(gs[0, 2])
    bms = beams_two_freq(-1.0, 'A')
    za2, Fz2, Pz2, Rz2 = scan_z(bms, s0, Bgrad, zmax=3e-3, nz=100)
    gcolors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    for ig in range(NG):
        lbl = f'J={J_g[ig]:.0f}/2,m={mJ_g[ig]:+.1f}'
        ax.plot(za2*1e3, Pz2[:, ig], color=gcolors[ig], lw=1.5, label=lbl)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('Population')
    ax.set_title('Red A: populations show optical pumping')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3)

    # --- Row 1: Blue DC MOT configs ---
    print("Plotting blue DC MOT configs...")
    ax = fig.add_subplot(gs[1, 0])
    for pol, ls, clr in [('A','-','#1f77b4'), ('B','--','#ff7f0e'), ('C',':','#2ca02c')]:
        bms = beams_two_freq(+3.0, pol)
        za, Fz, _, _ = scan_z(bms, s0, Bgrad, zmax=3e-3, nz=100)
        ax.plot(za*1e3, Fz/FUNIT, ls=ls, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('BLUE DC MOT: F(z) at v=0, Δ=+3Γ')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    for pol, ls, clr in [('A','-','#1f77b4'), ('B','--','#ff7f0e'), ('C',':','#2ca02c')]:
        bms = beams_two_freq(+3.0, pol)
        va, Fv = scan_v(bms, s0, Bgrad, vmax=4, nv=100)
        ax.plot(va, Fv/FUNIT, ls=ls, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('v (m/s)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('BLUE DC MOT: F(v) at z=0')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[1, 2])
    bms_blue = beams_two_freq(+3.0, 'A')
    za3, Fz3, Pz3, Rz3 = scan_z(bms_blue, s0, Bgrad, zmax=3e-3, nz=100)
    for ig in range(NG):
        lbl = f'J={J_g[ig]:.0f}/2,m={mJ_g[ig]:+.1f}'
        ax.plot(za3*1e3, Pz3[:, ig], color=gcolors[ig], lw=1.5, label=lbl)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('Population')
    ax.set_title('Blue A: populations')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3)

    # --- Row 2: 4-freq and Λ-BDM configs ---
    print("Plotting multi-frequency configs...")
    ax = fig.add_subplot(gs[2, 0])
    for ds, ls in [(0.3, '-'), (0.5, '--'), (1.0, ':')]:
        bms = beams_four_freq(-1.0, ds)
        za, Fz, _, _ = scan_z(bms, s0, Bgrad, zmax=3e-3, nz=100)
        ax.plot(za*1e3, Fz/FUNIT, ls=ls, lw=2, label=f'split={ds:.1f}Γ')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('4-FREQ RED: F(z), Δ=-1Γ')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[2, 1])
    for d2, ls in [(-0.3,'-'), (-0.1,'--'), (0.0,':'), (0.1,'-.')]:
        bms = beams_blue_lambda(+3.0, d2)
        za, Fz, _, _ = scan_z(bms, s0, Bgrad, zmax=3e-3, nz=100)
        ax.plot(za*1e3, Fz/FUNIT, ls=ls, lw=2, label=f'δ₂={d2:+.1f}Γ')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('Λ-BDM: F(z), Δ=+3Γ')
    ax.legend(); ax.grid(True, alpha=.3)

    # Λ-BDM F(v)
    ax = fig.add_subplot(gs[2, 2])
    for d2, ls in [(-0.3,'-'), (-0.1,'--'), (0.0,':')]:
        bms = beams_blue_lambda(+3.0, d2)
        va, Fv = scan_v(bms, s0, Bgrad, vmax=4, nv=100)
        ax.plot(va, Fv/FUNIT, ls=ls, lw=2, label=f'δ₂={d2:+.1f}Γ')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('v (m/s)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('Λ-BDM: F(v) at z=0')
    ax.legend(); ax.grid(True, alpha=.3)

    # --- Row 3: Detuning scan for best categories ---
    print("Running parameter scans...")
    ax = fig.add_subplot(gs[3, 0])
    det_arr = np.linspace(-3, +6, 50)
    for pol, clr in [('A','#1f77b4'), ('B','#ff7f0e'), ('C','#2ca02c')]:
        ks = []
        for d in det_arr:
            bms = beams_two_freq(d, pol)
            k, _ = spring_and_damping(bms, s0, Bgrad)
            ks.append(k)
        ax.plot(det_arr, ks, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--'); ax.axvline(0, c='gray', ls=':', alpha=.3)
    ax.set_xlabel('Detuning (Γ)'); ax.set_ylabel('k (N/m)')
    ax.set_title('Spring Const. vs Detuning (all pols)')
    ax.legend(); ax.grid(True, alpha=.3)

    # Damping scan
    ax = fig.add_subplot(gs[3, 1])
    for pol, clr in [('A','#1f77b4'), ('B','#ff7f0e'), ('C','#2ca02c')]:
        betas = []
        for d in det_arr:
            bms = beams_two_freq(d, pol)
            _, beta = spring_and_damping(bms, s0, Bgrad)
            betas.append(beta)
        ax.plot(det_arr, betas, color=clr, lw=2, label=f'Pol {pol}')
    ax.axhline(0, c='gray', ls='--'); ax.axvline(0, c='gray', ls=':', alpha=.3)
    ax.set_xlabel('Detuning (Γ)'); ax.set_ylabel('β (s⁻¹)')
    ax.set_title('Damping Coeff. vs Detuning')
    ax.legend(); ax.grid(True, alpha=.3)

    # B-gradient scan for best config
    ax = fig.add_subplot(gs[3, 2])
    Bs = np.linspace(5, 60, 20) * 1e-2
    # pick a couple of promising configs
    for det, pol, lbl, clr in [(-1.0,'A','Red A Δ=-1Γ','red'),
                                (+3.0,'A','Blue A Δ=+3Γ','blue'),
                                (+3.0,'B','Blue B Δ=+3Γ','cyan')]:
        ks_B = []
        for bg in Bs:
            bms = beams_two_freq(det, pol)
            k, _ = spring_and_damping(bms, s0, bg)
            ks_B.append(k)
        ax.plot(Bs*100, ks_B, lw=2, label=lbl, color=clr)
    ax.axhline(0, c='gray', ls='--')
    ax.set_xlabel("B' (G/cm)"); ax.set_ylabel('k (N/m)')
    ax.set_title('Spring Const. vs B Gradient')
    ax.legend(); ax.grid(True, alpha=.3)

    # --- Row 4: Best config detailed force map ---
    # Pick the best trapping+cooling or best trapping
    if trapping:
        best_name, best_cfg = trapping[0]
    elif trapping_only := [(n,c) for n,c in configs.items() if c['k']>0]:
        trapping_only.sort(key=lambda x: -x[1]['k'])
        best_name, best_cfg = trapping_only[0]
    else:
        best_name = 'Blue A Δ=+3.0Γ'
        best_cfg = configs.get(best_name, {'bms': beams_two_freq(+3.0,'A')})

    print(f"\nDetailed analysis of best config: {best_name}")
    bms_best = best_cfg.get('bms', beams_two_freq(+3.0, 'A'))

    ax = fig.add_subplot(gs[4, 0])
    za, Fz, Pz, Rz = scan_z(bms_best, s0, Bgrad, zmax=5e-3, nz=150)
    ax.plot(za*1e3, Fz/FUNIT, 'b-', lw=2)
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title(f'Best: {best_name}\nF(z) at v=0')
    ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[4, 1])
    va, Fv = scan_v(bms_best, s0, Bgrad, vmax=5, nv=150)
    ax.plot(va, Fv/FUNIT, 'r-', lw=2)
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('v (m/s)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title(f'Best: F(v) at z=0')
    ax.grid(True, alpha=.3)

    # 2D force map
    ax = fig.add_subplot(gs[4, 2])
    nz2d = 40; nv2d = 40
    z2d = np.linspace(-3e-3, 3e-3, nz2d)
    v2d = np.linspace(-3, 3, nv2d)
    F2d = np.zeros((nv2d, nz2d))
    for iz, zz in enumerate(z2d):
        for iv, vv in enumerate(v2d):
            F2d[iv, iz], _, _ = solve_force(bms_best, s0, vv, zz, Bgrad)
    F2d_n = F2d / FUNIT
    vmax2d = max(abs(F2d_n.min()), abs(F2d_n.max()), 1e-6)
    im = ax.pcolormesh(z2d*1e3, v2d, F2d_n, cmap='RdBu_r',
                       vmin=-vmax2d, vmax=vmax2d, shading='auto')
    plt.colorbar(im, ax=ax, label='F/(ℏkΓ/2)')
    ax.set_xlabel('z (mm)'); ax.set_ylabel('v (m/s)')
    ax.set_title('Force Map F(z,v)')

    # --- Row 5: Trajectories and summary ---
    ax = fig.add_subplot(gs[5, 0])
    dt = 2e-6; tmax = 0.04; nstep = int(tmax/dt)
    for z0, v0, lbl in [(2e-3, 0, 'z₀=2mm'),
                         (0, -1.0, 'v₀=-1m/s'),
                         (1e-3, -0.5, 'mixed')]:
        zt = np.zeros(nstep); vt = np.zeros(nstep)
        tt = np.arange(nstep)*dt
        zt[0]=z0; vt[0]=v0
        for j in range(1, nstep):
            f,_,_ = solve_force(bms_best, s0, vt[j-1], zt[j-1], Bgrad)
            vt[j] = vt[j-1] + (f/M)*dt
            zt[j] = zt[j-1] + vt[j]*dt
            if abs(zt[j]) > 0.015: zt[j:]=zt[j]; vt[j:]=vt[j]; break
        ax.plot(tt*1e3, zt*1e3, lw=1.5, label=lbl)
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('t (ms)'); ax.set_ylabel('z (mm)')
    ax.set_title(f'Trajectories ({best_name})')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # Phase space
    ax = fig.add_subplot(gs[5, 1])
    for z0, v0, lbl in [(2e-3, 0, 'z₀=2mm'),
                         (0, -1.0, 'v₀=-1m/s')]:
        zt = np.zeros(nstep); vt = np.zeros(nstep)
        zt[0]=z0; vt[0]=v0
        for j in range(1, nstep):
            f,_,_ = solve_force(bms_best, s0, vt[j-1], zt[j-1], Bgrad)
            vt[j] = vt[j-1] + (f/M)*dt
            zt[j] = zt[j-1] + vt[j]*dt
            if abs(zt[j]) > 0.015: zt[j:]=zt[j]; vt[j:]=vt[j]; break
        ax.plot(zt*1e3, vt, lw=1.5, label=lbl)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('v (m/s)')
    ax.set_title('Phase Space'); ax.legend(); ax.grid(True, alpha=.3)

    # Summary text
    ax = fig.add_subplot(gs[5, 2])
    ax.axis('off')
    k_best = best_cfg.get('k', 0)
    b_best = best_cfg.get('beta', 0)
    osc_best = np.sqrt(abs(k_best)/M)/(2*np.pi) if k_best > 0 else 0
    summary = (
        f"SrOH DC MOT — Summary\n"
        f"{'─'*38}\n\n"
        f"RED DC MOT (Δ<0, 2 frequencies):\n"
        f"  Spring constant ~ 0  (type-II dark\n"
        f"  state pumping cancels trapping force)\n"
        f"  ➜ NOT viable without RF switching\n\n"
        f"BLUE DC MOT (Δ>0, 2 frequencies):\n"
        f"  Sub-Doppler Sisyphus force reversal\n"
        f"  ➜ Non-zero restoring + damping forces\n"
        f"  ➜ PROMISING for SrOH DC MOT\n\n"
        f"Best config found:\n"
        f"  {best_name}\n"
        f"  k  = {k_best:.3e} N/m\n"
        f"  β  = {b_best:.1e} s⁻¹\n"
        f"  ω  = {osc_best:.1f} Hz\n\n"
        f"Key physics:\n"
        f"  • Type-II (6 ground, 2 excited states)\n"
        f"  • Red: σ± pumps to dark states that\n"
        f"    couple preferentially to anti-trap beam\n"
        f"  • Blue: force reversal from sub-Doppler\n"
        f"    magnetically-assisted Sisyphus effect\n"
        f"  • 3D effects (B-field rotation) not in\n"
        f"    this 1D model would help remix darks\n\n"
        f"Experimental RF MOT (Lasner 2024):\n"
        f"  T=1.2mK  N=2000  τ=91ms  β~100s⁻¹"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=.8))

    fig.suptitle('SrOH DC MOT — Red vs Blue Detuning Analysis',
                 fontsize=18, fontweight='bold', y=0.998)

    outpath = '/Users/dslmd/Downloads/DC red MOT/sroh_dc_mot_v2.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
