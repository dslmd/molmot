#!/usr/bin/env python3
"""
SrOH 4-Frequency DC Red MOT — Parameter Optimisation

The 4-frequency scheme works by providing both σ⁺ and σ⁻ at slightly
different detunings within each spin-rotation manifold.  This breaks
the type-II dark-state symmetry that kills the standard 2-frequency
DC red MOT.

Laser configuration per beam direction:
  For each SR manifold (J=3/2 and J=1/2):
    • σ⁺ component at Δ + δ_split
    • σ⁻ component at Δ − δ_split
  Counter-propagating beam: opposite polarizations for each frequency.

This is analogous to the CaF DC red MOT where 4 hyperfine frequencies
are used with the F=2 component at opposite handedness.
"""

import numpy as np
from scipy.linalg import solve as la_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════  Constants & Molecule  ═══════════════
HBAR = 1.0545718e-34; KB = 1.380649e-23; MU_B = 9.2740100783e-24
AMU = 1.66053906660e-27
M = 105*AMU; LAM = 688e-9; KL = 2*np.pi/LAM; VREC = HBAR*KL/M
GAMMA = 2*np.pi*6.4e6; TD = HBAR*GAMMA/(2*KB)
GJ32 = 2.002/3; GJ12 = -2.002/3; GJE = 0.0
FUNIT = HBAR*KL*GAMMA/2

NG=6; NE=2
J_g = np.array([1.5]*4+[0.5]*2)
mJ_g = np.array([-1.5,-0.5,0.5,1.5,-0.5,0.5])
gJ_g = np.array([GJ32]*4+[GJ12]*2)
mJ_e = np.array([-0.5,0.5])

_t32 = {(0.5,1.5,-1):1/4,(-0.5,0.5,-1):1/12,(-0.5,-1.5,1):1/4,
        (0.5,-0.5,1):1/12,(0.5,0.5,0):1/6,(-0.5,-0.5,0):1/6}
_t12 = {(-0.5,0.5,-1):1/3,(0.5,-0.5,1):1/3,(0.5,0.5,0):1/6,(-0.5,-0.5,0):1/6}

d2 = np.zeros((NG,NE,3))
for ig in range(NG):
    HL = 2.0 if J_g[ig]>1 else 1.0
    tbl = _t32 if J_g[ig]>1 else _t12
    for ie in range(NE):
        for qi,q in enumerate([-1,0,1]):
            d2[ig,ie,qi] = HL*(2*J_g[ig]+1)*tbl.get((round(2*mJ_e[ie])/2,round(2*mJ_g[ig])/2,q),0.)
d2 /= d2.max()

BR = np.zeros((NE,NG))
for ie in range(NE):
    s = d2[:,ie,:].sum()
    if s > 0: BR[ie] = d2[:,ie,:].sum(axis=1)/s

def zg(ig,B): return gJ_g[ig]*mJ_g[ig]*MU_B*B/HBAR
def ze(ie,B): return GJE*mJ_e[ie]*MU_B*B/HBAR

def solve(beams, s0, v, z, Bg):
    B = Bg*z; nb = len(beams)
    R = np.zeros((NG,NE,nb)); Fk = np.zeros((NG,NE,nb))
    for ib,bm in enumerate(beams):
        kd=bm['dir']; Jt=bm['J']; q=bm['q']; qi=q+1; d0=bm['delta']
        dop = -KL*kd*v
        for ig in range(NG):
            if abs(J_g[ig]-Jt)>0.1: continue
            for ie in range(NE):
                c2 = d2[ig,ie,qi]
                if c2<1e-15: continue
                delta = d0+dop+zg(ig,B)-ze(ie,B)
                L = (GAMMA/2)**2/(delta**2+(GAMMA/2)**2)
                rate = (GAMMA/2)*s0*c2*L
                R[ig,ie,ib] = rate; Fk[ig,ie,ib] = HBAR*KL*kd*rate
    Rsum = R.sum(axis=2)
    Mm = np.zeros((NG,NG))
    for ig in range(NG):
        Mm[ig,ig] -= Rsum[ig].sum()
        for ik in range(NG):
            for ie in range(NE): Mm[ig,ik] += Rsum[ik,ie]*BR[ie,ig]
    Ms = Mm.copy(); Ms[-1]=1.; rhs=np.zeros(NG); rhs[-1]=1.
    try: p = la_solve(Ms,rhs)
    except: p = np.ones(NG)/NG
    p = np.maximum(p,0); p/=p.sum()
    force = sum(p[ig]*Fk[ig].sum() for ig in range(NG))
    Rsc = sum(p[ig]*Rsum[ig].sum() for ig in range(NG))
    return force, p, Rsc


def beams_4freq(delta_G, split_G):
    d = delta_G*GAMMA; ds = split_G*GAMMA
    bms = []
    for Jt in [1.5, 0.5]:
        for sign, doff in [(+1,+ds),(-1,-ds)]:
            bms.append(dict(dir=+1, J=Jt, q=sign,  delta=d+doff))
            bms.append(dict(dir=-1, J=Jt, q=-sign, delta=d+doff))
    return bms


def spring_damp(beams, s0, Bg, dz=0.3e-3, dv=0.1):
    fp,_,_ = solve(beams,s0,0,+dz,Bg)
    fm,_,_ = solve(beams,s0,0,-dz,Bg)
    k = -(fp-fm)/(2*dz)
    fvp,_,_ = solve(beams,s0,+dv,0,Bg)
    fvm,_,_ = solve(beams,s0,-dv,0,Bg)
    beta = -(fvp-fvm)/(2*dv*M)
    return k, beta


def main():
    print("=" * 70)
    print("  SrOH 4-Frequency DC Red MOT — Parameter Optimisation")
    print("=" * 70)

    # ── 2D parameter scan: Δ vs split ────────────────────────
    print("\n── 2D scan: detuning vs split ──")
    Bg = 16e-2; s0 = 1.0
    dets = np.linspace(-3.0, -0.2, 30)
    splits = np.linspace(0.05, 2.0, 30)
    K = np.zeros((len(dets), len(splits)))
    B_arr = np.zeros_like(K)

    for i, det in enumerate(dets):
        for j, sp in enumerate(splits):
            bms = beams_4freq(det, sp)
            k, beta = spring_damp(bms, s0, Bg)
            K[i,j] = k
            B_arr[i,j] = beta

    # find optimum
    idx = np.unravel_index(np.argmax(K), K.shape)
    det_opt = dets[idx[0]]; sp_opt = splits[idx[1]]
    k_opt = K[idx]; beta_opt = B_arr[idx]
    print(f"  Optimum: Δ = {det_opt:.2f}Γ, split = {sp_opt:.2f}Γ")
    print(f"           k = {k_opt:.3e} N/m, β = {beta_opt:.0f} s⁻¹")
    print(f"           ω = {np.sqrt(k_opt/M)/(2*np.pi):.1f} Hz")

    # ── s₀ scan at optimum ───────────────────────────────────
    print("\n── Saturation scan at optimum detuning ──")
    s_arr = np.linspace(0.1, 10, 30)
    ks_s = []; bs_s = []
    for ss in s_arr:
        bms = beams_4freq(det_opt, sp_opt)
        k, beta = spring_damp(bms, ss, Bg)
        ks_s.append(k); bs_s.append(beta)

    # ── B gradient scan ──────────────────────────────────────
    print("── B gradient scan ──")
    Bg_arr = np.linspace(5, 60, 25)*1e-2
    ks_B = []; bs_B = []
    for bg in Bg_arr:
        bms = beams_4freq(det_opt, sp_opt)
        k, beta = spring_damp(bms, s0, bg)
        ks_B.append(k); bs_B.append(beta)

    # ── Force profiles at optimum ────────────────────────────
    print("── Computing force profiles ──")
    bms_opt = beams_4freq(det_opt, sp_opt)

    nz = 200; zmax = 5e-3
    za = np.linspace(-zmax, zmax, nz)
    Fz = np.zeros(nz); Pz = np.zeros((nz,NG)); Rz = np.zeros(nz)
    for i,z in enumerate(za):
        Fz[i], Pz[i], Rz[i] = solve(bms_opt, s0, 0, z, Bg)

    nv = 200; vmax = 6
    va = np.linspace(-vmax, vmax, nv)
    Fv = np.zeros(nv); Fv1 = np.zeros(nv)
    for i,v in enumerate(va):
        Fv[i],_,_ = solve(bms_opt, s0, v, 0, Bg)
        Fv1[i],_,_ = solve(bms_opt, s0, v, 1e-3, Bg)

    # ── 2D force map ─────────────────────────────────────────
    print("── 2D force map ──")
    nz2 = 50; nv2 = 50
    z2 = np.linspace(-4e-3, 4e-3, nz2)
    v2 = np.linspace(-4, 4, nv2)
    F2 = np.zeros((nv2, nz2))
    for iz,zz in enumerate(z2):
        for iv,vv in enumerate(v2):
            F2[iv,iz],_,_ = solve(bms_opt, s0, vv, zz, Bg)

    # ── Trajectories ─────────────────────────────────────────
    print("── Trajectories ──")
    dt = 1e-6; tmax = 0.03; nstep = int(tmax/dt)
    trajs = []
    for z0,v0,lbl in [(3e-3,0,'z₀=3mm'), (0,-2,'v₀=-2m/s'),
                       (2e-3,-1,'z₀=2mm,v₀=-1'), (0,-5,'v₀=-5m/s')]:
        zt = np.zeros(nstep); vt = np.zeros(nstep); tt = np.arange(nstep)*dt
        zt[0]=z0; vt[0]=v0
        for j in range(1,nstep):
            f,_,_ = solve(bms_opt, s0, vt[j-1], zt[j-1], Bg)
            vt[j] = vt[j-1]+(f/M)*dt; zt[j] = zt[j-1]+vt[j]*dt
            if abs(zt[j])>0.015: zt[j:]=zt[j]; vt[j:]=vt[j]; break
        trajs.append((tt, zt, vt, lbl))

    # ── Capture velocity estimate ────────────────────────────
    print("── Estimating capture velocity ──")
    v_tests = np.arange(0.5, 15, 0.5)
    captured = []
    for v0 in v_tests:
        zt = np.zeros(5000); vt = np.zeros(5000)
        zt[0] = 3e-3; vt[0] = -v0
        for j in range(1,5000):
            f,_,_ = solve(bms_opt, s0, vt[j-1], zt[j-1], Bg)
            vt[j] = vt[j-1]+(f/M)*1e-6
            zt[j] = zt[j-1]+vt[j]*1e-6
            if abs(zt[j])>0.015: break
        final_pos = abs(zt[min(j, 4999)])
        if final_pos < 0.005:
            captured.append(v0)

    v_cap = max(captured) if captured else 0
    print(f"  Estimated capture velocity: {v_cap:.1f} m/s")

    # ═══════════════  PLOTTING  ═══════════════
    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(5, 3, figure=fig, hspace=0.38, wspace=0.32)

    # Row 0: 2D optimisation maps
    ax = fig.add_subplot(gs[0, 0])
    K_n = K.copy(); K_n[K_n < 0] = 0
    im = ax.pcolormesh(splits, dets, K_n, cmap='hot', shading='auto')
    ax.plot(sp_opt, det_opt, 'c*', ms=15, mew=2)
    ax.set_xlabel('Split (Γ)'); ax.set_ylabel('Detuning (Γ)')
    ax.set_title('Spring Constant k (N/m)')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(splits, dets, B_arr, cmap='RdBu_r',
                       vmin=-max(abs(B_arr.min()),B_arr.max()),
                       vmax=max(abs(B_arr.min()),B_arr.max()), shading='auto')
    ax.plot(sp_opt, det_opt, 'k*', ms=15, mew=2)
    ax.set_xlabel('Split (Γ)'); ax.set_ylabel('Detuning (Γ)')
    ax.set_title('Damping β (s⁻¹)')
    plt.colorbar(im, ax=ax)

    # Oscillation frequency map
    ax = fig.add_subplot(gs[0, 2])
    omega_map = np.where(K > 0, np.sqrt(K/M)/(2*np.pi), 0)
    im = ax.pcolormesh(splits, dets, omega_map, cmap='viridis', shading='auto')
    ax.plot(sp_opt, det_opt, 'r*', ms=15, mew=2)
    ax.set_xlabel('Split (Γ)'); ax.set_ylabel('Detuning (Γ)')
    ax.set_title('Trap Frequency (Hz)')
    plt.colorbar(im, ax=ax)

    # Row 1: 1D parameter scans
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_arr, ks_s, 'b-', lw=2)
    ax.axhline(0, c='gray', ls='--')
    ax.set_xlabel('s₀ per beam'); ax.set_ylabel('k (N/m)')
    ax.set_title(f'k vs Saturation (Δ={det_opt:.2f}Γ, split={sp_opt:.2f}Γ)')
    ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(Bg_arr*100, ks_B, 'r-', lw=2)
    ax.axhline(0, c='gray', ls='--')
    ax.set_xlabel("B' (G/cm)"); ax.set_ylabel('k (N/m)')
    ax.set_title('k vs B Gradient')
    ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(s_arr, bs_s, 'g-', lw=2)
    ax.axhline(0, c='gray', ls='--')
    ax.set_xlabel('s₀ per beam'); ax.set_ylabel('β (s⁻¹)')
    ax.set_title('Damping vs Saturation')
    ax.grid(True, alpha=.3)

    # Row 2: Force profiles
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(za*1e3, Fz/FUNIT, 'b-', lw=2)
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title(f'Restoring Force F(z)\nΔ={det_opt:.2f}Γ, split={sp_opt:.2f}Γ')
    ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(va, Fv/FUNIT, 'r-', lw=2, label='z=0')
    ax.plot(va, Fv1/FUNIT, 'b--', lw=2, label='z=1mm')
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('v (m/s)'); ax.set_ylabel('F / (ℏkΓ/2)')
    ax.set_title('Velocity-Dependent Force')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[2, 2])
    gcolors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    for ig in range(NG):
        lbl = f'J={J_g[ig]:.0f}/2,m={mJ_g[ig]:+.1f}'
        ax.plot(za*1e3, Pz[:,ig], color=gcolors[ig], lw=1.5, label=lbl)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('Population')
    ax.set_title('Populations (v=0)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3)

    # Row 3: 2D force map + trajectories
    ax = fig.add_subplot(gs[3, 0])
    F2n = F2/FUNIT
    vm = max(abs(F2n.min()), abs(F2n.max()))
    im = ax.pcolormesh(z2*1e3, v2, F2n, cmap='RdBu_r', vmin=-vm, vmax=vm,
                       shading='auto')
    plt.colorbar(im, ax=ax, label='F/(ℏkΓ/2)')
    ax.set_xlabel('z (mm)'); ax.set_ylabel('v (m/s)')
    ax.set_title('Force Map F(z,v)')

    ax = fig.add_subplot(gs[3, 1])
    for tt,zt,vt,lbl in trajs:
        ax.plot(tt*1e3, zt*1e3, lw=1.5, label=lbl)
    ax.axhline(0, c='gray', ls='--', alpha=.5)
    ax.set_xlabel('t (ms)'); ax.set_ylabel('z (mm)')
    ax.set_title('Molecular Trajectories')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    ax = fig.add_subplot(gs[3, 2])
    for tt,zt,vt,lbl in trajs:
        ax.plot(zt*1e3, vt, lw=1.5, label=lbl)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('v (m/s)')
    ax.set_title('Phase Space')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # Row 4: Scattering rate + summary
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(za*1e3, Rz/(2*np.pi*1e6), 'g-', lw=2)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('Γ_sc / (2π MHz)')
    ax.set_title('Scattering Rate')
    ax.grid(True, alpha=.3)

    # Capture velocity plot
    ax = fig.add_subplot(gs[4, 1])
    cap_results = []
    for bg_test in [10, 16, 25, 40]:
        bg_T = bg_test * 1e-2
        vcaps = []
        for sp_test in np.linspace(0.1, 1.5, 15):
            bms_t = beams_4freq(det_opt, sp_test)
            # quick trajectory test
            vc = 0
            for v0 in np.arange(0.5, 12, 0.5):
                zt_t = 3e-3; vt_t = -v0
                trapped = True
                for _ in range(3000):
                    f,_,_ = solve(bms_t, s0, vt_t, zt_t, bg_T)
                    vt_t += (f/M)*1e-6; zt_t += vt_t*1e-6
                    if abs(zt_t)>0.015: trapped=False; break
                if trapped: vc = v0
                else: break
            vcaps.append(vc)
        ax.plot(np.linspace(0.1,1.5,15), vcaps, 'o-', lw=1.5,
                label=f"B'={bg_test}G/cm")
    ax.set_xlabel('Split (Γ)'); ax.set_ylabel('Capture velocity (m/s)')
    ax.set_title('Capture Velocity vs Split')
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # Summary
    ax = fig.add_subplot(gs[4, 2])
    ax.axis('off')
    omega_opt = np.sqrt(k_opt/M)/(2*np.pi)
    summary = (
        f"SrOH 4-Frequency DC Red MOT\n"
        f"{'═'*40}\n\n"
        f"OPTIMISED PARAMETERS:\n"
        f"  Detuning    Δ = {det_opt:.2f} Γ = {det_opt*GAMMA/2/np.pi/1e6:.1f} MHz\n"
        f"  Pol. split  δ = {sp_opt:.2f} Γ = {sp_opt*GAMMA/2/np.pi/1e6:.1f} MHz\n"
        f"  Sat. param  s₀ = {s0:.1f}\n"
        f"  B gradient  B' = {Bg*100:.0f} G/cm\n\n"
        f"PREDICTED PERFORMANCE:\n"
        f"  Spring const    k  = {k_opt:.2e} N/m\n"
        f"  Trap freq       ω  = 2π × {omega_opt:.0f} Hz\n"
        f"  Damping rate    β  = {beta_opt:.0f} s⁻¹\n"
        f"  Capture vel     v_c ≈ {v_cap:.1f} m/s\n"
        f"  Scatt. rate     Γ_sc = {Rz[nz//2+10]/2/np.pi/1e6:.2f} MHz\n\n"
        f"LASER CONFIGURATION (per axis):\n"
        f"  Beam +z:  4 frequency components\n"
        f"    J=3/2, σ⁺ at Δ+δ = {(det_opt+sp_opt)*GAMMA/2/np.pi/1e6:+.1f} MHz\n"
        f"    J=3/2, σ⁻ at Δ−δ = {(det_opt-sp_opt)*GAMMA/2/np.pi/1e6:+.1f} MHz\n"
        f"    J=1/2, σ⁺ at Δ+δ = {(det_opt+sp_opt)*GAMMA/2/np.pi/1e6:+.1f} MHz\n"
        f"    J=1/2, σ⁻ at Δ−δ = {(det_opt-sp_opt)*GAMMA/2/np.pi/1e6:+.1f} MHz\n"
        f"  Beam −z: opposite σ for each freq.\n\n"
        f"  SR sideband spacing: 109 MHz\n"
        f"  Pol. split spacing:  {sp_opt*GAMMA/2/np.pi/1e6:.1f} MHz\n\n"
        f"RF MOT COMPARISON (Lasner 2024):\n"
        f"  T=1.2mK  N=2000  τ=91ms\n"
        f"  β~100s⁻¹  ω~2π×45Hz"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=.8))

    fig.suptitle('SrOH 4-Frequency DC Red MOT — Optimised Configuration',
                 fontsize=18, fontweight='bold', y=0.998)

    outpath = '/Users/dslmd/Downloads/DC red MOT/sroh_4freq_optimised.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
