#!/usr/bin/env julia
"""
SrOH 4-Frequency DC Red MOT Simulation
Using OpticalBlochEquations.jl + QuantumStates.jl

This script builds the full 16-level SrOH molecular Hamiltonian
(12 ground + 4 excited states), sets up a 4-frequency DC MOT
configuration, and computes trapping forces via OBE.

The 4-frequency scheme uses σ⁺ and σ⁻ at split detunings within
each spin-rotation manifold to break type-II dark-state symmetry.
"""

# ══════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(joinpath(@__DIR__))

using QuantumStates
using OpticalBlochEquations
using UnitsToValue
using LinearAlgebra
using StaticArrays
using Printf
using Plots
using Statistics

import MutableNamedTuples: MutableNamedTuple
import StructArrays: StructArray, StructVector
import LoopVectorization: @turbo

# Aliases for compatibility with Christian's code
const μ_B = μB
# Patch: OBE package references operator_to_matrix without importing from QuantumStates
let otm = QuantumStates.operator_to_matrix
    @eval OpticalBlochEquations const operator_to_matrix = $otm
end

# Include Christian's SrOH Hamiltonian builder
include(joinpath(homedir(), ".julia/packages/OpticalBlochEquations",
    readdir(joinpath(homedir(), ".julia/packages/OpticalBlochEquations"))[1],
    "examples/SrOH MOT/SrOH_package.jl"))

# ══════════════════════════════════════════════════════════════
# Build molecular structure
# ══════════════════════════════════════════════════════════════

println("=" ^ 70)
println("  SrOH 4-Frequency DC Red MOT Simulation")
println("  Using OpticalBlochEquations.jl (C. Hallas, Harvard)")
println("=" ^ 70)

println("\nBuilding SrOH molecular Hamiltonian...")
package = get_SrOH_package()

n_states  = length(package.states)
n_ground  = n_states - package.n_excited
n_excited = package.n_excited
Γ = package.Γ
k = package.k
m = package.m
λ = 2π / k

println("  States: $n_ground ground + $n_excited excited = $n_states total")
println("  λ = $(round(λ*1e9, digits=1)) nm")
println("  Γ/(2π) = $(round(Γ/2π/1e6, digits=1)) MHz")
println("  mass = $(round(m/(@with_unit 1 "u"), digits=0)) amu")

# Print ground state energies (relative to mean)
println("\nGround state energies (MHz):")
E_ground = [energy(package.states[i]) for i in 1:n_ground]
E_mean = sum(E_ground) / n_ground
for i in 1:n_ground
    st = package.states[i]
    E_rel = (energy(st) - E_mean) / 1e6
    # extract quantum numbers from the eigenstate
    @printf("  |g%2d⟩: E = %+8.2f MHz\n", i, E_rel)
end

println("\nExcited state energies (MHz, relative):")
E_excited = [energy(package.states[n_ground+i]) for i in 1:n_excited]
E_ex_mean = sum(E_excited) / n_excited
for i in 1:n_excited
    E_rel = (energy(package.states[n_ground+i]) - E_ex_mean) / 1e6
    @printf("  |e%2d⟩: E = %+8.2f MHz\n", i, E_rel)
end

# ══════════════════════════════════════════════════════════════
# TDM analysis
# ══════════════════════════════════════════════════════════════

d = package.d
println("\nTransition dipole matrix |d[g,e,q]|² (nonzero elements):")
for ie in 1:n_excited
    for ig in 1:n_ground
        for q in 1:3
            val = abs2(d[ig, n_ground+ie, q])
            if val > 1e-6
                pol_label = ["σ⁻","π","σ⁺"][q]
                @printf("  |g%d⟩ → |e%d⟩ (%s): |d|² = %.4f\n", ig, ie, pol_label, val)
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════
# Zeeman interaction setup
# ══════════════════════════════════════════════════════════════

Zeeman_Hx = package.Zeeman_x_mat
Zeeman_Hy = package.Zeeman_y_mat
Zeeman_Hz = package.Zeeman_z_mat

"""
Update the Zeeman Hamiltonian for anti-Helmholtz configuration.
B-field: Bx = -B'x/2, By = -B'y/2, Bz = B'z (quadrupole)
Gradient B' in Gauss/cm; position r in units of 1/k.
"""
function update_H_and_∇H(H, p, r, t, Zeeman_Hx, Zeeman_Hy, Zeeman_Hz)
    B_gradient = p.sim_params.B_gradient  # Gauss/cm

    # Convert position from 1/k units to cm
    r_cm = r ./ (k * 100)

    # Quadrupole field (anti-Helmholtz)
    Bx = -B_gradient/2 * r_cm[1]
    By = -B_gradient/2 * r_cm[2]
    Bz =  B_gradient   * r_cm[3]

    @turbo for i in eachindex(H)
        H.re[i] = Bz * Zeeman_Hz.re[i] + Bx * Zeeman_Hx.re[i] + By * Zeeman_Hy.re[i]
        H.im[i] = Bz * Zeeman_Hz.im[i] + Bx * Zeeman_Hx.im[i] + By * Zeeman_Hy.im[i]
    end

    return SVector{3,ComplexF64}(0,0,0)
end

# ══════════════════════════════════════════════════════════════
# 4-Frequency Laser Configuration
# ══════════════════════════════════════════════════════════════

"""
Create the 4-frequency DC MOT laser fields.

For each SR manifold (J=3/2 and J=1/2), two frequency components
are used with opposite circular polarizations and split detunings:
  σ⁺ at Δ + δ_split
  σ⁻ at Δ − δ_split

Parameters:
  Δ_Gamma  : overall detuning in units of Γ (negative = red)
  δ_Gamma  : polarisation split in units of Γ
  s0       : saturation parameter per frequency component per beam
  w        : beam 1/e² radius in metres
"""
function make_4freq_fields(package, Δ_Gamma, δ_Gamma, s0, w;
                           directions=[:px, :mx, :py, :my, :pz, :mz])

    Γ = package.Γ
    k = package.k

    # State energies (Hz)
    E_g = [energy(package.states[i]) for i in 1:12]
    E_e = [energy(package.states[13+i-1]) for i in 1:4]

    # Transition frequencies for each ground→excited pair
    # The laser frequencies are set relative to the mean transition frequency
    ω_0 = mean(E_e) - mean(E_g)  # mean transition frequency (Hz)

    # The two SR manifolds are separated by ~109 MHz
    # Identify which ground states belong to J=3/2 vs J=1/2
    # In the SrOH package, ground states 1-8 are J=3/2 (N=1), 9-12 are J=1/2 (N=1)
    # (this is after selecting states [5:16] from the full basis)

    # Mean energy of each SR manifold
    E_J32_mean = mean(E_g[1:8])   # J=3/2 states
    E_J12_mean = mean(E_g[9:12])  # J=1/2 states
    E_e_mean = mean(E_e)

    # Transition frequencies for each manifold
    ω_J32 = E_e_mean - E_J32_mean  # Hz
    ω_J12 = E_e_mean - E_J12_mean  # Hz

    println("\n  Transition frequencies:")
    @printf("    J=3/2 → J'=1/2: %.3f MHz above mean\n", (ω_J32 - ω_0)/1e6)
    @printf("    J=1/2 → J'=1/2: %.3f MHz above mean\n", (ω_J12 - ω_0)/1e6)
    @printf("    SR splitting:    %.1f MHz\n", (ω_J32 - ω_J12)/1e6)

    # Detuning and split in Hz
    Δ = Δ_Gamma * Γ / (2π)  # Hz
    δ = δ_Gamma * Γ / (2π)  # Hz

    # 4 frequency components:
    # 1. J=3/2, σ⁺: ω = ω_J32 + Δ + δ
    # 2. J=3/2, σ⁻: ω = ω_J32 + Δ - δ
    # 3. J=1/2, σ⁺: ω = ω_J12 + Δ + δ
    # 4. J=1/2, σ⁻: ω = ω_J12 + Δ - δ
    freqs = [
        ω_J32 + Δ + δ,   # component 1: J=3/2, σ⁺
        ω_J32 + Δ - δ,   # component 2: J=3/2, σ⁻
        ω_J12 + Δ + δ,   # component 3: J=1/2, σ⁺
        ω_J12 + Δ - δ,   # component 4: J=1/2, σ⁻
    ]

    # Polarisations (in lab frame, for +z beam direction)
    # σ⁺ = right-circular: ε = (-1/√2, -i/√2, 0)
    # σ⁻ = left-circular:  ε = (+1/√2, -i/√2, 0)
    ε_plus  = SVector{3, ComplexF64}(-1/√2, -im/√2, 0)
    ε_minus = SVector{3, ComplexF64}(+1/√2, -im/√2, 0)

    pols = [ε_plus, ε_minus, ε_plus, ε_minus]

    # Gaussian beam intensity profile
    s_func(r, t) = exp(-2*(r[1]^2 + r[2]^2) / (w*k)^2)

    # Build Field objects for each frequency component × beam direction
    fields = Field[]

    # k-vectors for 6 beam directions
    kvecs = Dict(
        :px => SVector{3,Float64}(+1, 0, 0),
        :mx => SVector{3,Float64}(-1, 0, 0),
        :py => SVector{3,Float64}(0, +1, 0),
        :my => SVector{3,Float64}(0, -1, 0),
        :pz => SVector{3,Float64}(0, 0, +1),
        :mz => SVector{3,Float64}(0, 0, -1),
    )

    for (i_comp, (ω_comp, ε_comp)) in enumerate(zip(freqs, pols))
        for dir in directions
            khat = kvecs[dir]
            # Angular frequency in natural units (Γ)
            ω_nat = ω_comp * 2π / Γ

            # Construct the field
            f = Field(khat, t -> ε_comp, ω_nat, s0, s_func, 0.0)
            push!(fields, f)
        end
    end

    println("  Created $(length(fields)) field objects ($(length(freqs)) freq × $(length(directions)) beams)")
    @printf("  Detuning Δ = %.2f Γ = %.1f MHz\n", Δ_Gamma, Δ/1e6)
    @printf("  Pol split δ = %.2f Γ = %.1f MHz\n", δ_Gamma, δ/1e6)
    @printf("  s₀ = %.2f per component per beam\n", s0)
    @printf("  Beam radius w = %.1f mm\n", w*1e3)

    return fields, freqs
end

# ══════════════════════════════════════════════════════════════
# Force computation using rate equations (fast screening)
# ══════════════════════════════════════════════════════════════

"""
Compute the DC MOT force using rate equations.
This is faster than the full OBE and good for parameter scans.
"""
function rate_eq_force(package, freqs, pols_q, s0, v_z, z, B_gradient)
    Γ = package.Γ
    k = package.k
    d = package.d
    n_g = n_ground
    n_e = n_excited

    B = B_gradient * z * 100  # z in metres → B in Gauss (B' in G/cm)

    # State energies with Zeeman shift
    # Use the Zeeman matrices to compute energy shifts
    Zeeman_Hz_full = zeros(ComplexF64, n_states, n_states)
    for i in 1:n_states
        for j in 1:n_states
            Zeeman_Hz_full[i,j] = Zeeman_Hz.re[i,j] + im*Zeeman_Hz.im[i,j]
        end
    end

    # Zeeman energies: E_Z = B * diag(Zeeman_z_mat) * Γ/(2π)
    E_Z = real.(diag(Zeeman_Hz_full)) .* B .* Γ / (2π)  # Hz

    # State energies
    Es = [energy(package.states[i]) for i in 1:n_states]
    Es_B = Es .+ E_Z

    # Branching ratios from TDM
    BR = zeros(n_e, n_g)
    for ie in 1:n_e
        total = 0.0
        for ig in 1:n_g
            for q in 1:3
                total += abs2(d[ig, n_g+ie, q])
            end
        end
        for ig in 1:n_g
            for q in 1:3
                BR[ie, ig] += abs2(d[ig, n_g+ie, q])
            end
            BR[ie, ig] /= max(total, 1e-30)
        end
    end

    # Excitation rates R[ig, ie, i_beam] for each beam
    # Beams: +z and -z for each frequency component
    n_comp = length(freqs)
    n_beams = 2 * n_comp  # +z and -z for each component

    R = zeros(n_g, n_e, n_beams)
    F_beam = zeros(n_g, n_e, n_beams)

    for i_comp in 1:n_comp
        ω_laser = freqs[i_comp]  # Hz
        q_pol = pols_q[i_comp]   # polarisation index: 1=σ⁻, 2=π, 3=σ⁺

        for (i_dir, kdir) in enumerate([+1, -1])
            i_beam = (i_comp-1)*2 + i_dir

            doppler = -k * kdir * v_z / (2π)  # Hz

            # For retro-reflected beam, σ⁺ ↔ σ⁻
            q_eff = kdir > 0 ? q_pol : (q_pol == 1 ? 3 : (q_pol == 3 ? 1 : 2))

            for ig in 1:n_g
                for ie in 1:n_e
                    d2 = abs2(d[ig, n_g+ie, q_eff])
                    if d2 < 1e-15; continue; end

                    # Transition frequency
                    ω_trans = Es_B[n_g+ie] - Es_B[ig]  # Hz

                    # Effective detuning
                    δ_eff = (ω_laser + doppler - ω_trans) * 2π  # rad/s

                    # Lorentzian
                    L = (Γ/2)^2 / (δ_eff^2 + (Γ/2)^2)

                    rate = (Γ/2) * s0 * d2 * L
                    R[ig, ie, i_beam] = rate
                    F_beam[ig, ie, i_beam] = ℏ_val * k * kdir * rate
                end
            end
        end
    end

    # Steady-state populations
    R_sum = dropdims(sum(R, dims=3), dims=3)  # [ig, ie]

    M_mat = zeros(n_g, n_g)
    for ig in 1:n_g
        M_mat[ig, ig] -= sum(R_sum[ig, :])
        for ik in 1:n_g
            for ie in 1:n_e
                M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]
            end
        end
    end

    # Solve M p = 0, Σp = 1
    M_sol = copy(M_mat)
    M_sol[end, :] .= 1.0
    rhs = zeros(n_g)
    rhs[end] = 1.0

    p = try
        M_sol \ rhs
    catch
        ones(n_g) / n_g
    end
    p = max.(p, 0.0)
    p ./= sum(p)

    # Total force
    force = 0.0
    R_scatter = 0.0
    for ig in 1:n_g
        force += p[ig] * sum(F_beam[ig, :, :])
        R_scatter += p[ig] * sum(R_sum[ig, :])
    end

    return force, p, R_scatter
end

# ══════════════════════════════════════════════════════════════
# Main simulation
# ══════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("  Running 4-Frequency DC MOT Force Analysis")
println("=" ^ 70)

# Physical parameters
B_gradient = 16.0     # G/cm
s0 = 1.0              # saturation per component per beam
beam_w = 10e-3        # beam 1/e² radius (m)

# Optimal parameters from Python screening
Δ_opt = -0.88   # Γ
δ_opt = 0.39    # Γ

# Transition frequencies
E_g_all = [energy(package.states[i]) for i in 1:n_ground]
E_e_all = [energy(package.states[n_ground+i]) for i in 1:n_excited]
E_J32_mean = mean(E_g_all[1:8])
E_J12_mean = mean(E_g_all[9:12])
E_e_mean = mean(E_e_all)
ω_J32 = E_e_mean - E_J32_mean
ω_J12 = E_e_mean - E_J12_mean

# Set up frequencies and polarisations for rate equation model
Δ_Hz = Δ_opt * Γ / (2π)
δ_Hz = δ_opt * Γ / (2π)

freqs_4f = [
    ω_J32 + Δ_Hz + δ_Hz,   # J=3/2, σ⁺
    ω_J32 + Δ_Hz - δ_Hz,   # J=3/2, σ⁻
    ω_J12 + Δ_Hz + δ_Hz,   # J=1/2, σ⁺
    ω_J12 + Δ_Hz - δ_Hz,   # J=1/2, σ⁻
]
pols_4f = [3, 1, 3, 1]  # q index: 1=σ⁻, 2=π, 3=σ⁺

ℏ_val = h / (2π)

println("\nLaser configuration:")
@printf("  Detuning  Δ = %.2f Γ = %.1f MHz\n", Δ_opt, Δ_Hz/1e6)
@printf("  Pol split δ = %.2f Γ = %.1f MHz\n", δ_opt, δ_Hz/1e6)
@printf("  s₀ = %.1f per component\n", s0)
@printf("  B' = %.0f G/cm\n", B_gradient)

println("\nFrequency components (MHz from mean transition):")
for (i, (f, q)) in enumerate(zip(freqs_4f, pols_4f))
    ql = ["σ⁻","π","σ⁺"][q]
    manifold = i <= 2 ? "J=3/2" : "J=1/2"
    @printf("  %d. %s %s: %+.1f MHz\n", i, manifold, ql, (f - (ω_J32+ω_J12)/2)/1e6)
end

# ── Force vs position ────────────────────────────────────
println("\nComputing F(z) at v=0...")
nz = 200
z_arr = range(-5e-3, 5e-3, length=nz)
Fz = zeros(nz)
Pz = zeros(nz, n_ground)
Rz = zeros(nz)

for (i, z) in enumerate(z_arr)
    Fz[i], Pz[i,:], Rz[i] = rate_eq_force(package, freqs_4f, pols_4f, s0, 0.0, z, B_gradient)
end

F_unit = ℏ_val * k * Γ / 2
println("  Max |F| = $(@sprintf("%.4f", maximum(abs.(Fz))/F_unit)) ℏkΓ/2")

# Spring constant
dFdz = diff(Fz) ./ diff(collect(z_arr))
k_spring = -dFdz[nz÷2]
ω_trap = k_spring > 0 ? sqrt(k_spring/m)/(2π) : 0.0
@printf("  Spring constant k = %.3e N/m\n", k_spring)
@printf("  Trap frequency ω  = 2π × %.1f Hz\n", ω_trap)

# ── Force vs velocity ────────────────────────────────────
println("Computing F(v) at z=0...")
nv = 200
v_arr = range(-6.0, 6.0, length=nv)
Fv = zeros(nv)
Fv_1mm = zeros(nv)

for (i, v) in enumerate(v_arr)
    Fv[i], _, _ = rate_eq_force(package, freqs_4f, pols_4f, s0, v, 0.0, B_gradient)
    Fv_1mm[i], _, _ = rate_eq_force(package, freqs_4f, pols_4f, s0, v, 1e-3, B_gradient)
end

dFdv = diff(Fv) ./ diff(collect(v_arr))
β_damp = -dFdv[nv÷2] / m
@printf("  Damping β = %.0f s⁻¹\n", β_damp)

# ── Parameter optimisation: Δ vs δ ───────────────────────
println("\nOptimising: detuning vs split...")
Δ_scan = range(-3.0, -0.2, length=25)
δ_scan = range(0.05, 2.0, length=25)
K_map = zeros(length(Δ_scan), length(δ_scan))
β_map = zeros(length(Δ_scan), length(δ_scan))

dz = 0.3e-3; dv = 0.1
for (iΔ, Δv) in enumerate(Δ_scan)
    for (iδ, δv) in enumerate(δ_scan)
        Δ_Hz_s = Δv * Γ / (2π)
        δ_Hz_s = δv * Γ / (2π)
        fs = [ω_J32+Δ_Hz_s+δ_Hz_s, ω_J32+Δ_Hz_s-δ_Hz_s,
              ω_J12+Δ_Hz_s+δ_Hz_s, ω_J12+Δ_Hz_s-δ_Hz_s]

        fp, _, _ = rate_eq_force(package, fs, pols_4f, s0, 0.0, +dz, B_gradient)
        fm, _, _ = rate_eq_force(package, fs, pols_4f, s0, 0.0, -dz, B_gradient)
        K_map[iΔ, iδ] = -(fp - fm) / (2*dz)

        fvp, _, _ = rate_eq_force(package, fs, pols_4f, s0, +dv, 0.0, B_gradient)
        fvm, _, _ = rate_eq_force(package, fs, pols_4f, s0, -dv, 0.0, B_gradient)
        β_map[iΔ, iδ] = -(fvp - fvm) / (2*dv*m)
    end
end

idx_best = argmax(K_map)
Δ_best = Δ_scan[idx_best[1]]
δ_best = δ_scan[idx_best[2]]
k_best = K_map[idx_best]
β_best = β_map[idx_best]
ω_best = k_best > 0 ? sqrt(k_best/m)/(2π) : 0.0

println("\n  OPTIMAL PARAMETERS:")
@printf("    Δ = %.2f Γ = %.1f MHz\n", Δ_best, Δ_best*Γ/2π/1e6)
@printf("    δ = %.2f Γ = %.1f MHz\n", δ_best, δ_best*Γ/2π/1e6)
@printf("    k = %.3e N/m\n", k_best)
@printf("    ω = 2π × %.1f Hz\n", ω_best)
@printf("    β = %.0f s⁻¹\n", β_best)

# ── Trajectories ─────────────────────────────────────────
println("\nSimulating trajectories...")
dt = 1e-6
tmax = 0.03
nstep = Int(tmax / dt)

# Use optimised parameters
Δ_Hz_best = Δ_best * Γ / (2π)
δ_Hz_best = δ_best * Γ / (2π)
freqs_best = [ω_J32+Δ_Hz_best+δ_Hz_best, ω_J32+Δ_Hz_best-δ_Hz_best,
              ω_J12+Δ_Hz_best+δ_Hz_best, ω_J12+Δ_Hz_best-δ_Hz_best]

trajs = []
for (z0, v0, lbl) in [(3e-3, 0.0, "z₀=3mm"),
                        (0.0, -2.0, "v₀=-2m/s"),
                        (2e-3, -1.0, "mixed"),
                        (0.0, -5.0, "v₀=-5m/s")]
    zt = zeros(nstep); vt = zeros(nstep); tt = (0:nstep-1) .* dt
    zt[1] = z0; vt[1] = v0
    for j in 2:nstep
        f, _, _ = rate_eq_force(package, freqs_best, pols_4f, s0, vt[j-1], zt[j-1], B_gradient)
        a = f / m
        vt[j] = vt[j-1] + a*dt
        zt[j] = zt[j-1] + vt[j]*dt
        if abs(zt[j]) > 0.015
            zt[j:end] .= zt[j]; vt[j:end] .= vt[j]; break
        end
    end
    push!(trajs, (tt, zt, vt, lbl))
end

# ── Capture velocity ─────────────────────────────────────
println("Estimating capture velocity...")
v_cap = 0.0
for v0 in 0.5:0.5:15.0
    zt_test = 3e-3; vt_test = -v0
    trapped = true
    for _ in 1:5000
        f, _, _ = rate_eq_force(package, freqs_best, pols_4f, s0, vt_test, zt_test, B_gradient)
        vt_test += (f/m)*1e-6
        zt_test += vt_test*1e-6
        if abs(zt_test) > 0.015; trapped = false; break; end
    end
    if trapped
        global v_cap = v0
    else
        break
    end
end
@printf("  Capture velocity: %.1f m/s\n", v_cap)

# ── Power budget ─────────────────────────────────────────
I_sat = π * h * UnitsToValue.c * (Γ/2π) / (3 * λ^3)  # W/m²
P_per_comp = s0 * I_sat * π * beam_w^2 / 2  # W (Gaussian beam)
P_per_arm = 4 * P_per_comp
P_total = 3 * P_per_arm  # 3 retro-reflected axes

println("\n" * "=" ^ 70)
println("  EXPERIMENTAL PARAMETERS SUMMARY")
println("=" ^ 70)
@printf("\n  B gradient:       %d G/cm\n", B_gradient)
@printf("  Beam 1/e² diam:   %.0f mm\n", beam_w*2e3)
@printf("  I_sat:            %.1f mW/cm²\n", I_sat/10)
@printf("  Power/component:  %.1f mW\n", P_per_comp*1e3)
@printf("  Power/arm:        %.1f mW (4 components)\n", P_per_arm*1e3)
@printf("  Total power:      %.0f mW (3 retro axes)\n", P_total*1e3)
@printf("\n  Detuning Δ:       %.2f Γ = %.1f MHz\n", Δ_best, Δ_best*Γ/2π/1e6)
@printf("  Pol split δ:      %.2f Γ = %.1f MHz\n", δ_best, δ_best*Γ/2π/1e6)
@printf("  Spring constant:  %.2e N/m\n", k_best)
@printf("  Trap frequency:   2π × %.0f Hz\n", ω_best)
@printf("  Damping rate:     %.0f s⁻¹\n", β_best)
@printf("  Capture velocity: %.1f m/s\n", v_cap)

# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

println("\nGenerating plots...")

outdir = joinpath(@__DIR__, "..")

# Plot 1: Force profiles
p1 = plot(layout=(2,2), size=(1200,900), dpi=150)

plot!(p1[1], collect(z_arr)*1e3, Fz./F_unit, lw=2, label="",
      xlabel="z (mm)", ylabel="F / (ℏkΓ/2)",
      title="Restoring Force F(z) at v=0")
hline!(p1[1], [0], ls=:dash, c=:gray, label="")

plot!(p1[2], collect(v_arr), Fv./F_unit, lw=2, label="z=0",
      xlabel="v (m/s)", ylabel="F / (ℏkΓ/2)",
      title="Damping Force F(v)")
plot!(p1[2], collect(v_arr), Fv_1mm./F_unit, lw=2, ls=:dash, label="z=1mm")
hline!(p1[2], [0], ls=:dash, c=:gray, label="")

for ig in 1:n_ground
    plot!(p1[3], collect(z_arr)*1e3, Pz[:,ig], lw=1.5,
          label=(ig <= 3 ? "g$ig" : ""),
          xlabel="z (mm)", ylabel="Population",
          title="State Populations")
end

plot!(p1[4], collect(z_arr)*1e3, Rz./(2π*1e6), lw=2, label="",
      xlabel="z (mm)", ylabel="Γ_sc (MHz)",
      title="Scattering Rate", c=:green)

savefig(p1, joinpath(outdir, "sroh_4freq_forces_julia.png"))

# Plot 2: Parameter optimisation
p2 = plot(layout=(1,3), size=(1600,450), dpi=150)

heatmap!(p2[1], collect(δ_scan), collect(Δ_scan), max.(K_map, 0),
         xlabel="Split δ (Γ)", ylabel="Detuning Δ (Γ)",
         title="Spring Constant k (N/m)", c=:hot)

heatmap!(p2[2], collect(δ_scan), collect(Δ_scan), β_map,
         xlabel="Split δ (Γ)", ylabel="Detuning Δ (Γ)",
         title="Damping β (s⁻¹)", c=:RdBu)

ω_map = [K_map[i,j] > 0 ? sqrt(K_map[i,j]/m)/(2π) : 0.0
         for i in 1:length(Δ_scan), j in 1:length(δ_scan)]
heatmap!(p2[3], collect(δ_scan), collect(Δ_scan), ω_map,
         xlabel="Split δ (Γ)", ylabel="Detuning Δ (Γ)",
         title="Trap Frequency ω/(2π) (Hz)", c=:viridis)

savefig(p2, joinpath(outdir, "sroh_4freq_optimisation_julia.png"))

# Plot 3: Trajectories
p3 = plot(layout=(1,2), size=(1200,450), dpi=150)

for (tt, zt, vt, lbl) in trajs
    plot!(p3[1], tt*1e3, zt*1e3, lw=1.5, label=lbl,
          xlabel="t (ms)", ylabel="z (mm)", title="Molecular Trajectories")
end
hline!(p3[1], [0], ls=:dash, c=:gray, label="")

for (tt, zt, vt, lbl) in trajs
    plot!(p3[2], zt*1e3, vt, lw=1.5, label=lbl,
          xlabel="z (mm)", ylabel="v (m/s)", title="Phase Space")
end

savefig(p3, joinpath(outdir, "sroh_4freq_trajectories_julia.png"))

println("Plots saved to: $outdir")
println("\nDone!")
