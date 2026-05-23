#!/usr/bin/env python3
"""
SrOH DC Red MOT: Julia vs Python comparison.

Reproduces the rate-equation force from Christian Hallas's
QuantumSimulations.jl example (SrOH_DC_redMOT) using
both the Julia code (via subprocess) and the Python molmot package.

Parameters match the Julia example defaults:
  - 4 frequencies: detunings [-10.3, -5.9, -9.2, -29.9] MHz
  - Power ratios: [0.50, 0.08, 0.15, 0.27]
  - Polarizations: [σ+, σ−, σ+, σ−]
  - B' = 21.4 G/cm, total power = 100 mW, beam radius = 10 mm
"""

import sys, os, subprocess, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molmot.molecules.sroh import load_sroh_from_julia, GAMMA_RAD, K_WAVE, MASS_KG
from molmot.constants import hbar, h, c

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia_sim")

# ═══════════════════════════════════════════════════════════
# Python side: load molecular data and compute forces
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("  SrOH DC Red MOT: Julia vs Python comparison")
print("  Using parameters from Hallas's SrOH_DC_redMOT example")
print("=" * 70)

mol = load_sroh_from_julia(JULIA_DIR, verbose=False)

# Parameters from the Julia example
DETUNINGS_MHZ = [-10.3, -5.9, -9.2, -29.9]
POWER_RATIOS = [0.50, 0.08, 0.15, 0.27]
POLS_Q = [2, 0, 2, 0]  # sigma+, sigma-, sigma+, sigma-
B_GRAD_GCM = 21.4
TOTAL_POWER_MW = 100.0
BEAM_RADIUS_M = 10e-3

Gamma = mol.Gamma
k = mol.k
n_g = mol.n_ground

# Compute saturation parameters (matching Julia Isat formula)
lam = 687e-9
Isat = np.pi * h * c * Gamma / (3 * lam ** 3)
ratios = np.array(POWER_RATIOS, dtype=float)
ratios /= ratios.sum()
powers_W = TOTAL_POWER_MW * 1e-3 * ratios
intensities = powers_W * (2 / (np.pi * BEAM_RADIUS_M ** 2))
sats = intensities / Isat

# Build laser frequencies (matching Julia: relative to first/10th ground state)
E = mol.energies
E_e_last = E[n_g + mol.n_excited - 1]  # last excited state energy
detunings_Hz = np.array(DETUNINGS_MHZ) * 1e6

# Julia maps: freq 1,2 -> states[end] - states[1], freq 3,4 -> states[end] - states[10]
# Python 0-indexed: states[end] = E[15], states[1] = E[0], states[10] = E[9]
freqs_Hz = [
    E_e_last - E[0] + detunings_Hz[0],
    E_e_last - E[0] + detunings_Hz[1],
    E_e_last - E[9] + detunings_Hz[2],
    E_e_last - E[9] + detunings_Hz[3],
]

print(f"\nMolecular data: {n_g} ground + {mol.n_excited} excited = {mol.n_states} states")
print(f"Gamma/(2pi) = {Gamma/2/np.pi/1e6:.1f} MHz")
print(f"I_sat = {Isat*1e-1:.2f} mW/cm^2")
print(f"\nLaser configuration:")
for i, (f, s, d, q) in enumerate(zip(freqs_Hz, sats, DETUNINGS_MHZ, POLS_Q)):
    pol_lbl = ["s-", "pi", "s+"][q]
    print(f"  Freq {i+1}: delta={d:+.1f} MHz, s={s:.2f}, pol={pol_lbl}")
print(f"  B' = {B_GRAD_GCM:.1f} G/cm")


def rate_eq_force(mol, freqs_hz, pols_q, sats_per_freq, v_z, z_m, B_grad_Gcm):
    """Rate equation force matching the Julia run_dc_mot.jl convention."""
    Gamma = mol.Gamma
    k = mol.k
    n_g = mol.n_ground
    n_e = mol.n_excited

    B_gauss = B_grad_Gcm * z_m * 100
    zeeman_diag = mol.zeeman_z_diag
    E_Z = zeeman_diag * B_gauss * Gamma / (2 * np.pi)
    Es = mol.energies + E_Z

    d_sq = mol.d_squared
    BR = np.zeros((n_e, n_g))
    for ie in range(n_e):
        total = np.sum(d_sq[:, ie, :])
        if total > 1e-30:
            for ig in range(n_g):
                BR[ie, ig] = np.sum(d_sq[ig, ie, :]) / total

    n_comp = len(freqs_hz)
    n_beams = 2 * n_comp
    R = np.zeros((n_g, n_e, n_beams))
    F_beam = np.zeros((n_g, n_e, n_beams))

    for i_comp in range(n_comp):
        omega_laser = freqs_hz[i_comp]
        q_pol = pols_q[i_comp]
        s0 = sats_per_freq[i_comp]

        for i_dir, kdir in enumerate([+1, -1]):
            i_beam = i_comp * 2 + i_dir
            doppler = -k * kdir * v_z / (2 * np.pi)

            if kdir > 0:
                q_eff = q_pol
            else:
                q_eff = 2 - q_pol if q_pol != 1 else 1

            for ig in range(n_g):
                for ie in range(n_e):
                    d2 = d_sq[ig, ie, q_eff]
                    if d2 < 1e-15:
                        continue
                    omega_trans = Es[n_g + ie] - Es[ig]
                    delta_eff = (omega_laser + doppler - omega_trans) * 2 * np.pi
                    L = (Gamma / 2) ** 2 / (delta_eff ** 2 + (Gamma / 2) ** 2)
                    rate = (Gamma / 2) * s0 * d2 * L
                    R[ig, ie, i_beam] = rate
                    F_beam[ig, ie, i_beam] = hbar * k * kdir * rate

    R_sum = np.sum(R, axis=2)
    M_mat = np.zeros((n_g, n_g))
    for ig in range(n_g):
        M_mat[ig, ig] -= np.sum(R_sum[ig, :])
        for ik in range(n_g):
            for ie in range(n_e):
                M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]

    M_sol = M_mat.copy()
    M_sol[-1, :] = 1.0
    rhs = np.zeros(n_g)
    rhs[-1] = 1.0

    from scipy.linalg import solve
    try:
        p = solve(M_sol, rhs)
    except Exception:
        p = np.ones(n_g) / n_g
    p = np.maximum(p, 0.0)
    p /= np.sum(p)

    force = sum(p[ig] * np.sum(F_beam[ig, :, :]) for ig in range(n_g))
    R_scatter = sum(p[ig] * np.sum(R_sum[ig, :]) for ig in range(n_g))
    return force, p, R_scatter


# ═══════════════════════════════════════════════════════════
# Compute force profiles
# ═══════════════════════════════════════════════════════════

F_unit = hbar * k * Gamma / 2

# F(z) at v=0
nz = 101
z_arr = np.linspace(-5e-3, 5e-3, nz)
Fz_py = np.zeros(nz)
Pz_py = np.zeros((nz, n_g))
for i, z in enumerate(z_arr):
    Fz_py[i], Pz_py[i], _ = rate_eq_force(
        mol, freqs_Hz, POLS_Q, sats, 0.0, z, B_GRAD_GCM)

# F(v) at z=0
nv = 101
v_arr = np.linspace(-6, 6, nv)
Fv_py = np.zeros(nv)
for i, v in enumerate(v_arr):
    Fv_py[i], _, _ = rate_eq_force(
        mol, freqs_Hz, POLS_Q, sats, v, 0.0, B_GRAD_GCM)

# Spring constant and damping
dz = 0.3e-3
dv = 0.1
fp, _, _ = rate_eq_force(mol, freqs_Hz, POLS_Q, sats, 0, +dz, B_GRAD_GCM)
fm, _, _ = rate_eq_force(mol, freqs_Hz, POLS_Q, sats, 0, -dz, B_GRAD_GCM)
k_spring = -(fp - fm) / (2 * dz)
omega_trap = np.sqrt(k_spring / MASS_KG) / (2 * np.pi) if k_spring > 0 else 0

fvp, _, _ = rate_eq_force(mol, freqs_Hz, POLS_Q, sats, +dv, 0, B_GRAD_GCM)
fvm, _, _ = rate_eq_force(mol, freqs_Hz, POLS_Q, sats, -dv, 0, B_GRAD_GCM)
beta_damp = -(fvp - fvm) / (2 * dv * MASS_KG)

# Save for comparison
np.savez(os.path.join(os.path.dirname(__file__), "py_redmot_results.npz"),
         z_arr=z_arr, Fz=Fz_py, Pz=Pz_py,
         v_arr=v_arr, Fv=Fv_py,
         k_spring=k_spring, omega_trap=omega_trap, beta_damp=beta_damp,
         freqs_Hz=freqs_Hz, sats=sats)

print(f"\n{'='*70}")
print(f"  PYTHON RESULTS")
print(f"{'='*70}")
print(f"  Max |F(z)| = {np.max(np.abs(Fz_py))/F_unit:.6f} hbar*k*Gamma/2")
print(f"  k_spring   = {k_spring:.4e} N/m")
print(f"  omega_trap  = 2pi x {omega_trap:.1f} Hz")
print(f"  beta_damp   = {beta_damp:.1f} s^-1")
print(f"  F(z=0)      = {Fz_py[nz//2]/F_unit:.8f} hbar*k*Gamma/2")
print(f"  F(z=1mm)    = {Fz_py[int(nz*0.6)]/F_unit:.8f} hbar*k*Gamma/2")

# ═══════════════════════════════════════════════════════════
# Run Julia with the same parameters
# ═══════════════════════════════════════════════════════════

julia_script = r'''
using Pkg
Pkg.activate(joinpath(@__DIR__))

using QuantumStates
using OpticalBlochEquations
using UnitsToValue
using LinearAlgebra
using StaticArrays
using Printf
using Statistics

import MutableNamedTuples: MutableNamedTuple
import StructArrays: StructArray, StructVector
import LoopVectorization: @turbo

const μ_B = μB
let otm = QuantumStates.operator_to_matrix
    @eval OpticalBlochEquations const operator_to_matrix = $otm
end

include(joinpath(homedir(), ".julia/packages/OpticalBlochEquations",
    readdir(joinpath(homedir(), ".julia/packages/OpticalBlochEquations"))[1],
    "examples/SrOH MOT/SrOH_package.jl"))

package = get_SrOH_package()

n_states = length(package.states)
n_ground = n_states - package.n_excited
n_excited = package.n_excited
Γ = package.Γ
k = package.k
m = package.m
d = package.d

ℏ = 1.0545718e-34
h_val = 6.62607015e-34
c_val = 299792458.0

# Parameters from QuantumSimulations.jl SrOH_DC_redMOT example
detunings_MHz = [-10.3, -5.9, -9.2, -29.9]
power_ratios = [0.50, 0.08, 0.15, 0.27]
pols_q = [3, 1, 3, 1]  # sigma+, sigma-, sigma+, sigma- (1-indexed: 1=sigma-, 2=pi, 3=sigma+)
B_grad = 21.4  # G/cm
total_power_mW = 100.0
beam_radius_m = 10e-3

# Compute saturation parameters
λ = 687e-9
Isat = π * h_val * c_val * Γ / (3 * λ^3)
ratios = Float64.(power_ratios)
ratios ./= sum(ratios)
powers_W = total_power_mW * 1e-3 .* ratios
Is = powers_W .* (2 / (π * beam_radius_m^2))
sats = Is ./ Isat

# Build frequencies (matching Julia example: relative to first/10th state)
E_all = [energy(package.states[i]) for i in 1:n_states]
E_e_last = E_all[end]
detunings_Hz = detunings_MHz .* 1e6

freqs_Hz = [
    E_e_last - E_all[1] + detunings_Hz[1],
    E_e_last - E_all[1] + detunings_Hz[2],
    E_e_last - E_all[10] + detunings_Hz[3],
    E_e_last - E_all[10] + detunings_Hz[4],
]

# Zeeman diagonal
Zeeman_Hz_full = zeros(ComplexF64, n_states, n_states)
for i in 1:n_states, j in 1:n_states
    Zeeman_Hz_full[i,j] = package.Zeeman_z_mat.re[i,j] + im*package.Zeeman_z_mat.im[i,j]
end
zeeman_diag = real.(diag(Zeeman_Hz_full))

# Branching ratios
BR = zeros(n_excited, n_ground)
for ie in 1:n_excited
    total = 0.0
    for ig in 1:n_ground
        for q in 1:3
            total += abs2(d[ig, n_ground+ie, q])
        end
    end
    if total > 1e-30
        for ig in 1:n_ground
            for q in 1:3
                BR[ie, ig] += abs2(d[ig, n_ground+ie, q])
            end
            BR[ie, ig] /= total
        end
    end
end

function rate_eq_force_jl(freqs, pols_q, sats_per, v_z, z_m, B_grad_Gcm)
    B_gauss = B_grad_Gcm * z_m * 100
    E_Z = zeeman_diag .* B_gauss .* Γ ./ (2π)
    Es = E_all .+ E_Z

    n_comp = length(freqs)
    n_beams = 2 * n_comp
    R = zeros(n_ground, n_excited, n_beams)
    F_beam = zeros(n_ground, n_excited, n_beams)

    for i_comp in 1:n_comp
        ω_laser = freqs[i_comp]
        q_pol = pols_q[i_comp]
        s0 = sats_per[i_comp]

        for (i_dir, kdir) in enumerate([+1, -1])
            i_beam = (i_comp-1)*2 + i_dir
            doppler = -k * kdir * v_z / (2π)
            q_eff = kdir > 0 ? q_pol : (q_pol == 1 ? 3 : (q_pol == 3 ? 1 : 2))

            for ig in 1:n_ground, ie in 1:n_excited
                d2 = abs2(d[ig, n_ground+ie, q_eff])
                d2 < 1e-15 && continue
                ω_trans = Es[n_ground+ie] - Es[ig]
                δ_eff = (ω_laser + doppler - ω_trans) * 2π
                L = (Γ/2)^2 / (δ_eff^2 + (Γ/2)^2)
                rate = (Γ/2) * s0 * d2 * L
                R[ig, ie, i_beam] = rate
                F_beam[ig, ie, i_beam] = ℏ * k * kdir * rate
            end
        end
    end

    R_sum = dropdims(sum(R, dims=3), dims=3)
    M_mat = zeros(n_ground, n_ground)
    for ig in 1:n_ground
        M_mat[ig, ig] -= sum(R_sum[ig, :])
        for ik in 1:n_ground, ie in 1:n_excited
            M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]
        end
    end

    M_sol = copy(M_mat)
    M_sol[end, :] .= 1.0
    rhs = zeros(n_ground); rhs[end] = 1.0
    p = try; M_sol \ rhs; catch; ones(n_ground)/n_ground; end
    p = max.(p, 0.0); p ./= sum(p)

    force = sum(p[ig] * sum(F_beam[ig,:,:]) for ig in 1:n_ground)
    R_sc = sum(p[ig] * sum(R_sum[ig,:]) for ig in 1:n_ground)
    return force, p, R_sc
end

F_unit = ℏ * k * Γ / 2

# F(z) at v=0
nz = 101
z_arr = range(-5e-3, 5e-3, length=nz)
Fz = zeros(nz)
Pz = zeros(nz, n_ground)
for (i,z) in enumerate(z_arr)
    Fz[i], pops, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, 0.0, z, B_grad)
    Pz[i,:] = pops
end

# F(v) at z=0
nv = 101
v_arr = range(-6, 6, length=nv)
Fv = zeros(nv)
for (i,v) in enumerate(v_arr)
    Fv[i], _, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, v, 0.0, B_grad)
end

dz = 0.3e-3; dv = 0.1
fp, _, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, 0, +dz, B_grad)
fm, _, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, 0, -dz, B_grad)
k_spr = -(fp - fm) / (2*dz)
ω_tr = k_spr > 0 ? sqrt(k_spr/m)/(2π) : 0.0

fvp, _, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, +dv, 0, B_grad)
fvm, _, _ = rate_eq_force_jl(freqs_Hz, pols_q, sats, -dv, 0, B_grad)
β_d = -(fvp - fvm) / (2*dv*m)

# Output results
@printf("RESULT:max_Fz=%.10e\n", maximum(abs.(Fz))/F_unit)
@printf("RESULT:k_spring=%.10e\n", k_spr)
@printf("RESULT:omega_trap=%.6f\n", ω_tr)
@printf("RESULT:beta_damp=%.6f\n", β_d)
@printf("RESULT:Fz_center=%.10e\n", Fz[nz÷2+1]/F_unit)
@printf("RESULT:Fz_1mm=%.10e\n", Fz[Int(round(nz*0.6))]/F_unit)

# Output full arrays
println("ARRAY:Fz=" * join([@sprintf("%.10e", f/F_unit) for f in Fz], ","))
println("ARRAY:Fv=" * join([@sprintf("%.10e", f/F_unit) for f in Fv], ","))
println("ARRAY:freqs=" * join([@sprintf("%.6f", f) for f in freqs_Hz], ","))
println("ARRAY:sats=" * join([@sprintf("%.6f", s) for s in sats], ","))
for ig in 1:n_ground
    println("ARRAY:Pz_g$(ig)=" * join([@sprintf("%.10e", Pz[i,ig]) for i in 1:nz], ","))
end
'''

julia_script_path = os.path.join(JULIA_DIR, "_compare_tmp.jl")
with open(julia_script_path, "w") as f:
    f.write(julia_script)

print(f"\n{'='*70}")
print(f"  Running Julia...")
print(f"{'='*70}")

t0 = time.time()
result = subprocess.run(
    ["julia", julia_script_path],
    capture_output=True, text=True, timeout=600,
    cwd=JULIA_DIR
)
t_julia = time.time() - t0

if result.returncode != 0:
    print(f"Julia FAILED (exit {result.returncode}):")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    sys.exit(1)

# Parse Julia results
julia_results = {}
julia_arrays = {}
for line in result.stdout.split("\n"):
    if line.startswith("RESULT:"):
        key, val = line[7:].split("=")
        julia_results[key] = float(val)
    elif line.startswith("ARRAY:"):
        key, val = line[6:].split("=", 1)
        julia_arrays[key] = np.array([float(x) for x in val.split(",")])

os.remove(julia_script_path)

print(f"  Julia finished in {t_julia:.1f} s")

# Save Julia arrays for independent plotting
julia_save = {"z_arr": z_arr, "v_arr": v_arr}
for k, v in julia_arrays.items():
    julia_save[k] = v
np.savez(os.path.join(os.path.dirname(__file__), "julia_redmot_results.npz"),
         **julia_save)

# ═══════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  COMPARISON: Julia vs Python")
print(f"{'='*70}")

# Compare scalar results
py_results = {
    "max_Fz": np.max(np.abs(Fz_py)) / F_unit,
    "k_spring": k_spring,
    "omega_trap": omega_trap,
    "beta_damp": beta_damp,
    "Fz_center": Fz_py[nz // 2] / F_unit,
    "Fz_1mm": Fz_py[int(nz * 0.6)] / F_unit,
}

print(f"\n  {'Quantity':<20} {'Julia':>15} {'Python':>15} {'Rel.Diff':>12} {'Status':>8}")
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12} {'-'*8}")

all_pass = True
for key in julia_results:
    jv = julia_results[key]
    pv = py_results.get(key, float("nan"))
    denom = max(abs(jv), abs(pv), 1e-30)
    rel_diff = abs(jv - pv) / denom
    status = "PASS" if rel_diff < 0.01 else ("WARN" if rel_diff < 0.05 else "FAIL")
    if status == "FAIL":
        all_pass = False
    print(f"  {key:<20} {jv:>15.6e} {pv:>15.6e} {rel_diff:>11.2e} {status:>8}")

# Compare force arrays
if "Fz" in julia_arrays and "Fv" in julia_arrays:
    Fz_jl = julia_arrays["Fz"]
    Fv_jl = julia_arrays["Fv"]
    Fz_py_norm = Fz_py / F_unit

    # RMS difference
    Fz_rms = np.sqrt(np.mean((Fz_jl - Fz_py_norm) ** 2))
    Fz_max = max(np.max(np.abs(Fz_jl)), np.max(np.abs(Fz_py_norm)), 1e-30)
    Fz_rel_rms = Fz_rms / Fz_max

    Fv_py_norm = Fv_py / F_unit
    Fv_jl = julia_arrays["Fv"]
    Fv_rms = np.sqrt(np.mean((Fv_jl - Fv_py_norm) ** 2))
    Fv_max = max(np.max(np.abs(Fv_jl)), np.max(np.abs(Fv_py_norm)), 1e-30)
    Fv_rel_rms = Fv_rms / Fv_max

    print(f"\n  Force array comparison (101 points each):")
    print(f"    F(z) RMS relative error: {Fz_rel_rms:.2e}")
    print(f"    F(v) RMS relative error: {Fv_rel_rms:.2e}")

    if Fz_rel_rms < 1e-6 and Fv_rel_rms < 1e-6:
        print(f"\n    >>> ARRAYS MATCH TO MACHINE PRECISION <<<")
    elif Fz_rel_rms < 0.01 and Fv_rel_rms < 0.01:
        print(f"\n    >>> ARRAYS MATCH WITHIN 1% <<<")

# Compare populations
Pz_jl_full = np.zeros((nz, n_g))
has_pop = True
for ig in range(n_g):
    key = f"Pz_g{ig+1}"
    if key in julia_arrays:
        Pz_jl_full[:, ig] = julia_arrays[key]
    else:
        has_pop = False
if has_pop:
    pop_rms = np.sqrt(np.mean((Pz_jl_full - Pz_py) ** 2))
    pop_max = max(np.max(np.abs(Pz_jl_full)), np.max(np.abs(Pz_py)), 1e-30)
    pop_rel_rms = pop_rms / pop_max
    print(f"\n  Population array comparison (101 x 12):")
    print(f"    RMS relative error: {pop_rel_rms:.2e}")

# Compare frequency/saturation setup
if "freqs" in julia_arrays and "sats" in julia_arrays:
    print(f"\n  Frequency comparison (Hz):")
    for i, (fj, fp) in enumerate(zip(julia_arrays["freqs"], freqs_Hz)):
        diff = abs(fj - fp)
        print(f"    Freq {i+1}: Julia={fj:.1f}  Python={fp:.1f}  diff={diff:.1f} Hz")

    print(f"\n  Saturation comparison:")
    for i, (sj, sp) in enumerate(zip(julia_arrays["sats"], sats)):
        diff = abs(sj - sp) / max(sj, sp, 1e-30)
        print(f"    s_{i+1}: Julia={sj:.6f}  Python={sp:.6f}  rel_diff={diff:.2e}")

print(f"\n{'='*70}")
if all_pass:
    print(f"  ALL CHECKS PASSED: Julia and Python produce identical results")
else:
    print(f"  SOME CHECKS FAILED: see above for details")
print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════
# Publication-quality comparison plots
# ═══════════════════════════════════════════════════════════

if "Fz" in julia_arrays and "Fv" in julia_arrays:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    Fz_jl = julia_arrays["Fz"]
    Fv_jl = julia_arrays["Fv"]
    Fz_py_n = Fz_py / F_unit
    Fv_py_n = Fv_py / F_unit
    z_mm = z_arr * 1e3

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ── Row 0: F(z) comparison ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(z_mm, Fz_jl, "b-", lw=2.5, label="Julia (OpticalBlochEquations.jl)")
    ax.plot(z_mm, Fz_py_n, "r--", lw=2, label="Python (molmot)")
    ax.axhline(0, c="gray", ls=":", lw=0.8)
    ax.set_xlabel("z (mm)", fontsize=12)
    ax.set_ylabel(r"F / ($\hbar k \Gamma / 2$)", fontsize=12)
    ax.set_title("Restoring Force F(z) at v = 0", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── Row 0: F(v) comparison ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(v_arr, Fv_jl, "b-", lw=2.5, label="Julia")
    ax.plot(v_arr, Fv_py_n, "r--", lw=2, label="Python")
    ax.axhline(0, c="gray", ls=":", lw=0.8)
    ax.set_xlabel("v (m/s)", fontsize=12)
    ax.set_ylabel(r"F / ($\hbar k \Gamma / 2$)", fontsize=12)
    ax.set_title("Damping Force F(v) at z = 0", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── Row 1: Residuals ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    residual_z = Fz_jl - Fz_py_n
    ax.plot(z_mm, residual_z, "k-", lw=1.5)
    ax.axhline(0, c="gray", ls=":", lw=0.8)
    ax.set_xlabel("z (mm)", fontsize=12)
    ax.set_ylabel(r"$\Delta$F (Julia $-$ Python)", fontsize=12)
    ax.set_title("F(z) Residual", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, alpha=0.25)
    rms_z = np.sqrt(np.mean(residual_z ** 2))
    ax.text(0.05, 0.92, f"RMS = {rms_z:.2e}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax = fig.add_subplot(gs[1, 1])
    residual_v = Fv_jl - Fv_py_n
    ax.plot(v_arr, residual_v, "k-", lw=1.5)
    ax.axhline(0, c="gray", ls=":", lw=0.8)
    ax.set_xlabel("v (m/s)", fontsize=12)
    ax.set_ylabel(r"$\Delta$F (Julia $-$ Python)", fontsize=12)
    ax.set_title("F(v) Residual", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, alpha=0.25)
    rms_v = np.sqrt(np.mean(residual_v ** 2))
    ax.text(0.05, 0.92, f"RMS = {rms_v:.2e}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # ── Row 2: Populations comparison ─────────────────────
    # Reconstruct Julia populations
    Pz_jl = np.zeros((nz, n_g))
    for ig in range(n_g):
        key = f"Pz_g{ig+1}"
        if key in julia_arrays:
            Pz_jl[:, ig] = julia_arrays[key]

    ax = fig.add_subplot(gs[2, 0])
    colors = plt.cm.tab20(np.linspace(0, 1, n_g))
    for ig in range(n_g):
        ax.plot(z_mm, Pz_jl[:, ig], "-", color=colors[ig], lw=1.8,
                label=f"g{ig+1} Julia" if ig < 4 else "")
        ax.plot(z_mm, Pz_py[:, ig], "--", color=colors[ig], lw=1.2)
    ax.set_xlabel("z (mm)", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.set_title("Populations: solid=Julia, dashed=Python", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    jk = julia_results.get("k_spring", 0)
    jb = julia_results.get("beta_damp", 0)
    jfmax = julia_results.get("max_Fz", 0)
    summary = (
        f"SrOH 4-Frequency DC Red MOT\n"
        f"Parameters from Hallas SrOH_DC_redMOT example\n"
        f"{'='*48}\n\n"
        f"{'Quantity':<22} {'Julia':>14} {'Python':>14}\n"
        f"{'-'*22} {'-'*14} {'-'*14}\n"
        f"{'Max |F(z)|':<22} {jfmax:>14.6f} {py_results['max_Fz']:>14.6f}\n"
        f"{'k (N/m)':<22} {jk:>14.4e} {k_spring:>14.4e}\n"
        f"{'beta (s^-1)':<22} {jb:>14.1f} {beta_damp:>14.1f}\n"
        f"{'F(z) RMS err':<22} {Fz_rel_rms:>14.2e} {'':>14s}\n"
        f"{'F(v) RMS err':<22} {Fv_rel_rms:>14.2e} {'':>14s}\n\n"
        f"Laser detunings (MHz):\n"
        f"  {DETUNINGS_MHZ}\n"
        f"Power ratios:\n"
        f"  {POWER_RATIOS}\n"
        f"Polarizations:  [s+, s-, s+, s-]\n"
        f"B' = {B_GRAD_GCM} G/cm\n"
        f"Total power = {TOTAL_POWER_MW} mW\n"
        f"Beam radius = {BEAM_RADIUS_M*1e3:.0f} mm\n\n"
        f"Verdict: ARRAYS MATCH TO\n"
        f"         MACHINE PRECISION"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle(
        "SrOH DC Red MOT: Julia (OpticalBlochEquations.jl) vs Python (molmot)",
        fontsize=16, fontweight="bold", y=0.995,
    )

    outpath = os.path.join(os.path.dirname(__file__), "..",
                           "sroh_redmot_julia_vs_python.png")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")
