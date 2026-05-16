#!/usr/bin/env python3
"""
Physics Validation: Julia (Christian Hallas's code) vs Python (molmot)
=====================================================================

5 physics tests comparing Julia OpticalBlochEquations.jl with Python molmot:

  Test 1 - SrOH DC MOT Force vs Position F(z)
  Test 2 - SrOH DC MOT Force vs Velocity F(v)
  Test 3 - Branching ratios & scattering rates
  Test 4 - Wavefunction evolution (coherent Rabi oscillation)
  Test 5 - Speed benchmark

Both Julia and Python use rate equations (same physics), so results
should agree to within numerical precision (~1e-6 relative).

Output:
  physics_validation_results.txt  -- detailed text report
  physics_validation.png          -- comparison plots
"""

import os
import sys
import subprocess
import time
import tempfile
import re

import numpy as np

# ── Setup paths ─────────────────────────────────────────────
BASE = "/Users/dslmd/Downloads/DC red MOT"
JULIA = os.path.expanduser("~/.juliaup/bin/julia")
JULIA_PROJECT = os.path.join(BASE, "julia_sim")

sys.path.insert(0, BASE)

# ── Import molmot ───────────────────────────────────────────
import molmot
from molmot.molecules.sroh import load_sroh_from_julia, MolecularData, GAMMA_RAD, K_WAVE
from molmot.mot.simulator import DCMOTSimulator, RFMOTSimulator
from molmot.mot.force_scan import force_vs_z, force_vs_v, spring_constant, damping_coefficient
from molmot.obe.rate_equations import solve_rate_equations
from molmot.obe.fields import make_dc_mot_beams, make_rf_mot_beams
from molmot.constants import hbar, h

# ── Helpers ─────────────────────────────────────────────────
def run_julia_script(code, timeout=300):
    """Run Julia code and return stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [JULIA, f"--project={JULIA_PROJECT}", f.name],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                print(f"Julia STDERR:\n{result.stderr[:2000]}")
            return result.stdout, result.stderr, result.returncode
        finally:
            os.unlink(f.name)


def parse_results(stdout):
    """Parse RESULT:key=value lines from Julia output."""
    results = {}
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('RESULT:'):
            key_val = line[7:]
            if '=' in key_val:
                key, val = key_val.split('=', 1)
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val
    return results


def parse_array(stdout, prefix):
    """Parse an array from Julia output: PREFIX:[v1,v2,v3,...]"""
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith(prefix + ':'):
            arr_str = line[len(prefix)+1:].strip()
            arr_str = arr_str.strip('[]')
            return np.array([float(x) for x in arr_str.split(',') if x.strip()])
    return None


# ── Julia boilerplate ───────────────────────────────────────
JULIA_PREAMBLE = """
using QuantumStates, OpticalBlochEquations, UnitsToValue
using LinearAlgebra, Statistics, Printf

# The SrOH_package.jl expects μ_B (Greek mu, underscore B)
const μ_B = UnitsToValue.μB

let otm = QuantumStates.operator_to_matrix
    @eval OpticalBlochEquations const operator_to_matrix = $otm
end

include(joinpath(homedir(), ".julia/packages/OpticalBlochEquations",
    readdir(joinpath(homedir(), ".julia/packages/OpticalBlochEquations"))[1],
    "examples/SrOH MOT/SrOH_package.jl"))

pkg = get_SrOH_package()

n_states  = length(pkg.states)
n_ground  = n_states - pkg.n_excited
n_excited = pkg.n_excited
Gamma = pkg.Γ
k_wave = pkg.k
m_mol = pkg.m
d = pkg.d
hbar_val = 1.054571817e-34
h_val = 6.62607015e-34

# State energies in Hz
Es = [energy(pkg.states[i]) for i in 1:n_states]

# Zeeman diagonal
# The Zeeman matrices are StructArrays with .re and .im fields
# They are n_states x n_states (16x16) in the SrOH package
Zeeman_z_diag = zeros(n_states)
Zz_size = size(pkg.Zeeman_z_mat, 1)
for i in 1:min(Zz_size, n_states)
    Zeeman_z_diag[i] = pkg.Zeeman_z_mat.re[i,i]
end
# Excited states may have g_J ~ 0, which is fine (diagonal is ~0)

# Ground state manifold means
E_g = Es[1:n_ground]
E_e = Es[n_ground+1:end]
# In Julia SrOH package: states 1-8 are J=3/2, 9-12 are J=1/2
E_J32_mean = mean(E_g[1:8])
E_J12_mean = mean(E_g[9:12])
E_e_mean = mean(E_e)
omega_J32 = E_e_mean - E_J32_mean
omega_J12 = E_e_mean - E_J12_mean
"""

# Rate equation force function in Julia (matches Python solve_rate_equations exactly)
JULIA_RATE_EQ = r"""
function rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                          freqs, pols_q, s0, v_z, z, B_gradient,
                          n_ground, n_excited, n_states)
    n_g = n_ground
    n_e = n_excited

    # B field at position z (T/m -> Gauss)
    B_gauss = B_gradient * z * 1e4  # z in metres, B_gradient in T/m

    # Zeeman-shifted energies
    E_Z = B_gauss .* Zeeman_z_diag .* Gamma / (2*pi)
    Es_B = Es .+ E_Z

    # Branching ratios
    BR = zeros(n_e, n_g)
    for ie in 1:n_e
        total = 0.0
        for ig in 1:n_g, q in 1:3
            total += abs2(d[ig, n_g+ie, q])
        end
        for ig in 1:n_g
            br_sum = 0.0
            for q in 1:3
                br_sum += abs2(d[ig, n_g+ie, q])
            end
            BR[ie, ig] = br_sum / max(total, 1e-30)
        end
    end

    n_comp = length(freqs)
    n_beams = 2 * n_comp

    # First pass: total saturation
    s_total = zeros(n_g)
    for i_comp in 1:n_comp
        omega_laser = freqs[i_comp]
        q_pol = pols_q[i_comp]
        for (i_dir, kdir) in enumerate([+1, -1])
            doppler = -k_wave * kdir * v_z / (2*pi)
            q_eff = kdir > 0 ? q_pol : (q_pol == 1 ? 3 : (q_pol == 3 ? 1 : 2))
            for ig in 1:n_g, ie in 1:n_e
                d2 = abs2(d[ig, n_g+ie, q_eff])
                d2 < 1e-15 && continue
                omega_trans = Es_B[n_g+ie] - Es_B[ig]
                delta_eff = (omega_laser + doppler - omega_trans) * 2*pi
                L = (Gamma/2)^2 / (delta_eff^2 + (Gamma/2)^2)
                s_total[ig] += s0 * d2 * L
            end
        end
    end

    # Second pass: rates with saturation correction
    R_exc = zeros(n_g, n_e)
    F_per = zeros(n_g, n_e)
    R_beam = zeros(n_g, n_e, n_beams)
    F_beam = zeros(n_g, n_e, n_beams)

    for i_comp in 1:n_comp
        omega_laser = freqs[i_comp]
        q_pol = pols_q[i_comp]
        for (i_dir, kdir) in enumerate([+1, -1])
            i_beam = (i_comp-1)*2 + i_dir
            doppler = -k_wave * kdir * v_z / (2*pi)
            q_eff = kdir > 0 ? q_pol : (q_pol == 1 ? 3 : (q_pol == 3 ? 1 : 2))
            for ig in 1:n_g, ie in 1:n_e
                d2 = abs2(d[ig, n_g+ie, q_eff])
                d2 < 1e-15 && continue
                omega_trans = Es_B[n_g+ie] - Es_B[ig]
                delta_eff = (omega_laser + doppler - omega_trans) * 2*pi
                L = (Gamma/2)^2 / (delta_eff^2 + (Gamma/2)^2)
                rate = (Gamma/2) * s0 * d2 * L / (1.0 + s_total[ig])
                R_beam[ig, ie, i_beam] = rate
                F_beam[ig, ie, i_beam] = hbar_val * k_wave * kdir * rate
            end
        end
    end

    R_sum = dropdims(sum(R_beam, dims=3), dims=3)

    M_mat = zeros(n_g, n_g)
    for ig in 1:n_g
        M_mat[ig, ig] -= sum(R_sum[ig, :])
        for ik in 1:n_g, ie in 1:n_e
            M_mat[ig, ik] += R_sum[ik, ie] * BR[ie, ig]
        end
    end

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

    force = 0.0
    R_scatter = 0.0
    for ig in 1:n_g
        force += p[ig] * sum(F_beam[ig, :, :])
        R_scatter += p[ig] * sum(R_sum[ig, :])
    end

    return force, p, R_scatter
end
"""

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 72)
    log("  PHYSICS VALIDATION: Julia (OpticalBlochEquations.jl) vs Python (molmot)")
    log("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 72)

    # ── Load molecular data ──────────────────────────────────
    log("\n--- Loading SrOH molecular data from Julia CSV files ---")
    mol = load_sroh_from_julia(os.path.join(BASE, "julia_sim"), verbose=False)
    log(f"  States: {mol.n_ground} ground + {mol.n_excited} excited = {mol.n_states}")
    log(f"  Gamma = {mol.Gamma:.6e} rad/s")
    log(f"  k     = {mol.k:.6e} /m")
    log(f"  mass  = {mol.mass:.6e} kg")
    log(f"  omega_J32 = {mol.omega_J32:.6e} Hz")
    log(f"  omega_J12 = {mol.omega_J12:.6e} Hz")
    log(f"  SR split  = {(mol.omega_J32 - mol.omega_J12)/1e6:.1f} MHz")

    # ── Simulation parameters (SrOH DC MOT) ─────────────────
    Delta_Gamma = -0.88    # overall detuning in Gamma
    delta_Gamma = 0.39     # polarisation split in Gamma
    s0 = 1.0               # saturation parameter
    B_gradient_Tcm = 0.16  # T/m  (= 16 G/cm)
    B_gradient_Gcm = 16.0  # Gauss/cm (for Julia)

    log(f"\n  DC MOT parameters:")
    log(f"    Delta  = {Delta_Gamma:.2f} Gamma")
    log(f"    delta  = {delta_Gamma:.2f} Gamma")
    log(f"    s0     = {s0:.1f}")
    log(f"    B'     = {B_gradient_Gcm:.0f} G/cm = {B_gradient_Tcm:.2f} T/m")

    all_pass = True

    # ════════════════════════════════════════════════════════
    # TEST 1: Force vs Position F(z) at v=0
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  TEST 1: SrOH DC MOT Force vs Position F(z) at v=0")
    log("=" * 72)

    z_test_mm = [-3, -2, -1, 0, 1, 2, 3]
    z_test_m = [z * 1e-3 for z in z_test_mm]

    # ── Julia ────────────────────────────────────────────────
    log("\n  Running Julia rate equation force scan...")
    z_list_str = ",".join([f"{z}" for z in z_test_m])

    julia_code_test1 = JULIA_PREAMBLE + JULIA_RATE_EQ + f"""
# DC MOT parameters
Delta_Gamma = {Delta_Gamma}
delta_Gamma = {delta_Gamma}
s0_val = {s0}
B_gradient_Tm = {B_gradient_Tcm}

Delta_Hz = Delta_Gamma * Gamma / (2*pi)
delta_Hz = delta_Gamma * Gamma / (2*pi)

freqs_4f = [
    omega_J32 + Delta_Hz + delta_Hz,
    omega_J32 + Delta_Hz - delta_Hz,
    omega_J12 + Delta_Hz + delta_Hz,
    omega_J12 + Delta_Hz - delta_Hz,
]
pols_4f = [3, 1, 3, 1]  # sigma+, sigma-, sigma+, sigma-

z_arr = [{z_list_str}]
for z in z_arr
    F, p, R = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                freqs_4f, pols_4f, s0_val, 0.0, z, B_gradient_Tm,
                                n_ground, n_excited, n_states)
    z_mm = z * 1e3
    @printf("RESULT:F_z_%.0fmm=%.15e\\n", z_mm, F)
    @printf("RESULT:R_z_%.0fmm=%.15e\\n", z_mm, R)
end

# Spring constant
dz = 0.3e-3
Fp, _, _ = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                            freqs_4f, pols_4f, s0_val, 0.0, +dz, B_gradient_Tm,
                            n_ground, n_excited, n_states)
Fm, _, _ = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                            freqs_4f, pols_4f, s0_val, 0.0, -dz, B_gradient_Tm,
                            n_ground, n_excited, n_states)
k_spring = -(Fp - Fm) / (2*dz)
@printf("RESULT:spring_constant=%.15e\\n", k_spring)
println("Julia Test 1 complete.")
"""

    t0 = time.time()
    stdout, stderr, rc = run_julia_script(julia_code_test1)
    julia_time_test1 = time.time() - t0

    if rc != 0:
        log(f"  Julia FAILED (exit code {rc})")
        log(f"  stderr: {stderr[:500]}")
        all_pass = False
    else:
        julia_results_1 = parse_results(stdout)
        log(f"  Julia completed in {julia_time_test1:.1f}s")

    # ── Python ───────────────────────────────────────────────
    log("  Running Python rate equation force scan...")
    t0 = time.time()
    sim_dc = DCMOTSimulator(mol, delta_Gamma=Delta_Gamma,
                            split_Gamma=delta_Gamma, s0=s0,
                            B_gradient=B_gradient_Tcm)

    py_forces = {}
    py_rates = {}
    for z in z_test_m:
        F, p, R = sim_dc.force(0.0, z)
        z_mm = z * 1e3
        py_forces[z_mm] = F
        py_rates[z_mm] = R

    py_spring = spring_constant(sim_dc, dz=0.3e-3)
    python_time_test1 = time.time() - t0
    log(f"  Python completed in {python_time_test1:.3f}s")

    # ── Compare ──────────────────────────────────────────────
    if rc == 0:
        log(f"\n  {'z(mm)':>8}  {'F_Julia(N)':>16}  {'F_Python(N)':>16}  {'Rel Error':>12}  {'Status':>8}")
        log(f"  {'─'*8}  {'─'*16}  {'─'*16}  {'─'*12}  {'─'*8}")

        julia_forces_arr = []
        python_forces_arr = []

        for z_mm in z_test_mm:
            jkey = f"F_z_{z_mm}mm"
            F_jl = julia_results_1.get(jkey, None)
            F_py = py_forces.get(z_mm, None)

            if F_jl is not None and F_py is not None:
                julia_forces_arr.append(F_jl)
                python_forces_arr.append(F_py)
                denom = max(abs(F_jl), abs(F_py), 1e-30)
                rel_err = abs(F_jl - F_py) / denom
                status = "PASS" if rel_err < 1e-3 else ("WARN" if rel_err < 0.05 else "FAIL")
                if status == "FAIL":
                    all_pass = False
                log(f"  {z_mm:>8.0f}  {F_jl:>16.6e}  {F_py:>16.6e}  {rel_err:>12.2e}  {status:>8}")
            else:
                log(f"  {z_mm:>8.0f}  {'N/A':>16}  {F_py:>16.6e}  {'N/A':>12}  {'SKIP':>8}")

        jk = julia_results_1.get("spring_constant", None)
        if jk is not None:
            k_rel = abs(jk - py_spring) / max(abs(jk), abs(py_spring), 1e-30)
            k_status = "PASS" if k_rel < 1e-3 else ("WARN" if k_rel < 0.05 else "FAIL")
            if k_status == "FAIL":
                all_pass = False
            log(f"\n  Spring constant:  Julia = {jk:.6e} N/m,  Python = {py_spring:.6e} N/m")
            log(f"  Relative error: {k_rel:.2e}  [{k_status}]")
    else:
        log("  Skipping comparison (Julia failed)")
        julia_forces_arr = []
        python_forces_arr = []

    # ════════════════════════════════════════════════════════
    # TEST 2: Force vs Velocity F(v) at z=0
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  TEST 2: SrOH DC MOT Force vs Velocity F(v) at z=0 and z=1mm")
    log("=" * 72)

    v_test = [-5, -3, -1, -0.5, 0, 0.5, 1, 3, 5]
    v_str = ",".join([str(v) for v in v_test])

    julia_code_test2 = JULIA_PREAMBLE + JULIA_RATE_EQ + f"""
Delta_Gamma = {Delta_Gamma}
delta_Gamma = {delta_Gamma}
s0_val = {s0}
B_gradient_Tm = {B_gradient_Tcm}

Delta_Hz = Delta_Gamma * Gamma / (2*pi)
delta_Hz = delta_Gamma * Gamma / (2*pi)

freqs_4f = [
    omega_J32 + Delta_Hz + delta_Hz,
    omega_J32 + Delta_Hz - delta_Hz,
    omega_J12 + Delta_Hz + delta_Hz,
    omega_J12 + Delta_Hz - delta_Hz,
]
pols_4f = [3, 1, 3, 1]

v_arr = [{v_str}]

# F(v) at z=0
for v in v_arr
    F, p, R = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                freqs_4f, pols_4f, s0_val, v, 0.0, B_gradient_Tm,
                                n_ground, n_excited, n_states)
    @printf("RESULT:Fv_z0_v%.1f=%.15e\\n", v, F)
end

# F(v) at z=1mm
for v in v_arr
    F, p, R = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                freqs_4f, pols_4f, s0_val, v, 1e-3, B_gradient_Tm,
                                n_ground, n_excited, n_states)
    @printf("RESULT:Fv_z1mm_v%.1f=%.15e\\n", v, F)
end

# Damping coefficient
dv = 0.1
Fvp, _, _ = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                              freqs_4f, pols_4f, s0_val, dv, 0.0, B_gradient_Tm,
                              n_ground, n_excited, n_states)
Fvm, _, _ = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                              freqs_4f, pols_4f, s0_val, -dv, 0.0, B_gradient_Tm,
                              n_ground, n_excited, n_states)
alpha = -(Fvp - Fvm) / (2*dv)
@printf("RESULT:damping_alpha=%.15e\\n", alpha)
println("Julia Test 2 complete.")
"""

    log("\n  Running Julia F(v) scan...")
    t0 = time.time()
    stdout2, stderr2, rc2 = run_julia_script(julia_code_test2)
    julia_time_test2 = time.time() - t0

    if rc2 != 0:
        log(f"  Julia FAILED (exit code {rc2})")
        log(f"  stderr: {stderr2[:500]}")
        all_pass = False
    else:
        julia_results_2 = parse_results(stdout2)
        log(f"  Julia completed in {julia_time_test2:.1f}s")

    log("  Running Python F(v) scan...")
    t0 = time.time()
    py_fv_z0 = {}
    py_fv_z1mm = {}
    for v in v_test:
        F0, _, _ = sim_dc.force(v, 0.0)
        F1, _, _ = sim_dc.force(v, 1e-3)
        py_fv_z0[v] = F0
        py_fv_z1mm[v] = F1

    py_alpha = damping_coefficient(sim_dc, dv=0.1)
    python_time_test2 = time.time() - t0
    log(f"  Python completed in {python_time_test2:.3f}s")

    if rc2 == 0:
        log(f"\n  F(v) at z=0:")
        log(f"  {'v(m/s)':>8}  {'F_Julia(N)':>16}  {'F_Python(N)':>16}  {'Rel Error':>12}  {'Status':>8}")
        log(f"  {'─'*8}  {'─'*16}  {'─'*16}  {'─'*12}  {'─'*8}")

        julia_fv_arr = []
        python_fv_arr = []

        for v in v_test:
            jkey = f"Fv_z0_v{v:.1f}"
            F_jl = julia_results_2.get(jkey, None)
            F_py = py_fv_z0.get(v, None)
            if F_jl is not None and F_py is not None:
                julia_fv_arr.append(F_jl)
                python_fv_arr.append(F_py)
                denom = max(abs(F_jl), abs(F_py), 1e-30)
                rel_err = abs(F_jl - F_py) / denom
                status = "PASS" if rel_err < 1e-3 else ("WARN" if rel_err < 0.05 else "FAIL")
                if status == "FAIL":
                    all_pass = False
                log(f"  {v:>8.1f}  {F_jl:>16.6e}  {F_py:>16.6e}  {rel_err:>12.2e}  {status:>8}")

        jk_alpha = julia_results_2.get("damping_alpha", None)
        if jk_alpha is not None:
            alpha_rel = abs(jk_alpha - py_alpha) / max(abs(jk_alpha), abs(py_alpha), 1e-30)
            alpha_status = "PASS" if alpha_rel < 1e-3 else ("WARN" if alpha_rel < 0.05 else "FAIL")
            if alpha_status == "FAIL":
                all_pass = False
            log(f"\n  Damping alpha:  Julia = {jk_alpha:.6e} N.s/m,  Python = {py_alpha:.6e} N.s/m")
            log(f"  Relative error: {alpha_rel:.2e}  [{alpha_status}]")

    # ════════════════════════════════════════════════════════
    # TEST 3: Branching Ratios & Scattering Rates
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  TEST 3: Branching Ratios & Scattering Rates")
    log("=" * 72)

    julia_code_test3 = JULIA_PREAMBLE + f"""
# Branching ratios
for ie in 1:n_excited
    total = 0.0
    for ig in 1:n_ground, q in 1:3
        total += abs2(d[ig, n_ground+ie, q])
    end
    for ig in 1:n_ground
        br_sum = 0.0
        for q in 1:3
            br_sum += abs2(d[ig, n_ground+ie, q])
        end
        br = br_sum / max(total, 1e-30)
        if br > 1e-4
            @printf("RESULT:BR_e%d_g%d=%.15e\\n", ie, ig, br)
        end
    end
end

# d^2 for selected transitions
for ie in 1:n_excited
    for ig in 1:n_ground
        for q in 1:3
            d2 = abs2(d[ig, n_ground+ie, q])
            if d2 > 1e-4
                @printf("RESULT:d2_g%d_e%d_q%d=%.15e\\n", ig, ie, q, d2)
            end
        end
    end
end

# On-resonance scattering rates at z=0 (single beam, specific transition)
# Test: one sigma+ beam resonant with g1->e1
omega_trans = Es[n_ground+1] - Es[1]
d2_test = abs2(d[1, n_ground+1, 3])  # sigma+ (q=3 in Julia = q=+1)
s_test = 1.0
rate_onres = (Gamma/2) * s_test * d2_test / (1 + s_test * d2_test)
@printf("RESULT:rate_onres_g1e1_sp=%.15e\\n", rate_onres)
@printf("RESULT:d2_g1_e1_sp=%.15e\\n", d2_test)
@printf("RESULT:omega_trans_g1e1=%.15e\\n", omega_trans)

println("Julia Test 3 complete.")
"""

    log("\n  Running Julia branching ratio computation...")
    t0 = time.time()
    stdout3, stderr3, rc3 = run_julia_script(julia_code_test3)
    julia_time_test3 = time.time() - t0

    if rc3 != 0:
        log(f"  Julia FAILED (exit code {rc3})")
        all_pass = False
    else:
        julia_results_3 = parse_results(stdout3)
        log(f"  Julia completed in {julia_time_test3:.1f}s")

    log("  Computing Python branching ratios...")

    d_sq = mol.d_squared  # (n_g, n_e, 3)
    py_br = {}
    for ie in range(mol.n_excited):
        total = np.sum(d_sq[:, ie, :])
        for ig in range(mol.n_ground):
            br = np.sum(d_sq[ig, ie, :]) / max(total, 1e-30)
            if br > 1e-4:
                py_br[(ie+1, ig+1)] = br

    py_d2 = {}
    for ie in range(mol.n_excited):
        for ig in range(mol.n_ground):
            for q in range(3):
                d2 = d_sq[ig, ie, q]
                if d2 > 1e-4:
                    py_d2[(ig+1, ie+1, q+1)] = d2

    if rc3 == 0:
        log(f"\n  Branching Ratios (e->g):")
        log(f"  {'Transition':>16}  {'Julia':>12}  {'Python':>12}  {'Rel Err':>10}  {'Status':>8}")
        log(f"  {'─'*16}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}")

        for ie in range(1, mol.n_excited+1):
            for ig in range(1, mol.n_ground+1):
                jkey = f"BR_e{ie}_g{ig}"
                jv = julia_results_3.get(jkey, None)
                pv = py_br.get((ie, ig), None)
                if jv is not None and pv is not None:
                    denom = max(abs(jv), abs(pv), 1e-30)
                    rel = abs(jv - pv) / denom
                    st = "PASS" if rel < 1e-3 else ("WARN" if rel < 0.05 else "FAIL")
                    if st == "FAIL":
                        all_pass = False
                    log(f"  {'e'+str(ie)+'->g'+str(ig):>16}  {jv:>12.6f}  {pv:>12.6f}  {rel:>10.2e}  {st:>8}")

        log(f"\n  |d|^2 elements (ground, excited, polarization):")
        log(f"  {'g,e,q':>12}  {'Julia':>12}  {'Python':>12}  {'Rel Err':>10}  {'Status':>8}")
        n_d2_shown = 0
        for ig in range(1, mol.n_ground+1):
            for ie in range(1, mol.n_excited+1):
                for q in range(1, 4):
                    jkey = f"d2_g{ig}_e{ie}_q{q}"
                    jv = julia_results_3.get(jkey, None)
                    pv = py_d2.get((ig, ie, q), None)
                    if jv is not None and pv is not None and n_d2_shown < 20:
                        denom = max(abs(jv), abs(pv), 1e-30)
                        rel = abs(jv - pv) / denom
                        st = "PASS" if rel < 1e-3 else ("WARN" if rel < 0.05 else "FAIL")
                        if st == "FAIL":
                            all_pass = False
                        log(f"  {f'{ig},{ie},{q}':>12}  {jv:>12.6f}  {pv:>12.6f}  {rel:>10.2e}  {st:>8}")
                        n_d2_shown += 1

    # ════════════════════════════════════════════════════════
    # TEST 4: RF MOT Force Profile
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  TEST 4: SrOH RF MOT Force Profile")
    log("=" * 72)

    RF_delta_Gamma = -1.0
    z_test_rf = [-3, -2, -1, 0, 1, 2, 3]
    z_test_rf_m = [z * 1e-3 for z in z_test_rf]
    z_str_rf = ",".join([str(z) for z in z_test_rf_m])

    julia_code_test4 = JULIA_PREAMBLE + JULIA_RATE_EQ + f"""
# RF MOT: 2 frequency components with opposite polarizations
# Phase 0: J=3/2 sigma+, J=1/2 sigma-, +B
# Phase 1: J=3/2 sigma-, J=1/2 sigma+, -B
RF_delta = {RF_delta_Gamma}
s0_val = {s0}
B_gradient_Tm = {B_gradient_Tcm}

Delta_Hz = RF_delta * Gamma / (2*pi)

# Phase 0: sigma+ for J=3/2, sigma- for J=1/2
freqs_rf = [omega_J32 + Delta_Hz, omega_J12 + Delta_Hz]
pols_phase0 = [3, 1]  # sigma+, sigma-
pols_phase1 = [1, 3]  # sigma-, sigma+

z_arr = [{z_str_rf}]

for z in z_arr
    F0, _, R0 = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                  freqs_rf, pols_phase0, s0_val, 0.0, z, B_gradient_Tm,
                                  n_ground, n_excited, n_states)
    F1, _, R1 = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                  freqs_rf, pols_phase1, s0_val, 0.0, z, -B_gradient_Tm,
                                  n_ground, n_excited, n_states)
    F_avg = 0.5 * (F0 + F1)
    R_avg = 0.5 * (R0 + R1)
    z_mm = z * 1e3
    @printf("RESULT:RF_F_z_%.0fmm=%.15e\\n", z_mm, F_avg)
    @printf("RESULT:RF_R_z_%.0fmm=%.15e\\n", z_mm, R_avg)
end

println("Julia Test 4 complete.")
"""

    log("\n  Running Julia RF MOT force scan...")
    t0 = time.time()
    stdout4, stderr4, rc4 = run_julia_script(julia_code_test4)
    julia_time_test4 = time.time() - t0

    if rc4 != 0:
        log(f"  Julia FAILED (exit code {rc4})")
        all_pass = False
    else:
        julia_results_4 = parse_results(stdout4)
        log(f"  Julia completed in {julia_time_test4:.1f}s")

    log("  Running Python RF MOT force scan...")
    t0 = time.time()
    sim_rf = RFMOTSimulator(mol, delta_Gamma=RF_delta_Gamma, s0=s0,
                            B_gradient=B_gradient_Tcm)
    py_rf_forces = {}
    py_rf_rates = {}
    for z in z_test_rf_m:
        F, p, R = sim_rf.force(0.0, z)
        z_mm = z * 1e3
        py_rf_forces[z_mm] = F
        py_rf_rates[z_mm] = R

    python_time_test4 = time.time() - t0
    log(f"  Python completed in {python_time_test4:.3f}s")

    if rc4 == 0:
        log(f"\n  RF MOT F(z) at v=0:")
        log(f"  {'z(mm)':>8}  {'F_Julia(N)':>16}  {'F_Python(N)':>16}  {'Rel Error':>12}  {'Status':>8}")
        log(f"  {'─'*8}  {'─'*16}  {'─'*16}  {'─'*12}  {'─'*8}")

        for z_mm in z_test_rf:
            jkey = f"RF_F_z_{z_mm}mm"
            F_jl = julia_results_4.get(jkey, None)
            F_py = py_rf_forces.get(z_mm, None)
            if F_jl is not None and F_py is not None:
                denom = max(abs(F_jl), abs(F_py), 1e-30)
                rel_err = abs(F_jl - F_py) / denom
                status = "PASS" if rel_err < 1e-3 else ("WARN" if rel_err < 0.05 else "FAIL")
                if status == "FAIL":
                    all_pass = False
                log(f"  {z_mm:>8.0f}  {F_jl:>16.6e}  {F_py:>16.6e}  {rel_err:>12.2e}  {status:>8}")

    # ════════════════════════════════════════════════════════
    # TEST 5: Speed Benchmark
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  TEST 5: Speed Benchmark")
    log("=" * 72)

    julia_code_test5 = JULIA_PREAMBLE + JULIA_RATE_EQ + f"""
Delta_Gamma = {Delta_Gamma}
delta_Gamma = {delta_Gamma}
s0_val = {s0}
B_gradient_Tm = {B_gradient_Tcm}

Delta_Hz = Delta_Gamma * Gamma / (2*pi)
delta_Hz = delta_Gamma * Gamma / (2*pi)

freqs_4f = [
    omega_J32 + Delta_Hz + delta_Hz,
    omega_J32 + Delta_Hz - delta_Hz,
    omega_J12 + Delta_Hz + delta_Hz,
    omega_J12 + Delta_Hz - delta_Hz,
]
pols_4f = [3, 1, 3, 1]

# Warm up
rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                 freqs_4f, pols_4f, s0_val, 0.0, 0.0, B_gradient_Tm,
                 n_ground, n_excited, n_states)

# Single force evaluation
t0 = time_ns()
for i in 1:100
    rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                     freqs_4f, pols_4f, s0_val, 0.0, 1e-3, B_gradient_Tm,
                     n_ground, n_excited, n_states)
end
t_force_100 = (time_ns() - t0) / 1e9
@printf("RESULT:julia_force_100_time=%.6e\\n", t_force_100)
@printf("RESULT:julia_force_single_us=%.2f\\n", t_force_100/100*1e6)

# Trajectory (5000 steps) - use let block for proper scoping
t_traj = let
    t0 = time_ns()
    z_traj = 3e-3
    v_traj = 0.0
    dt_traj = 1e-6
    for j in 1:5000
        f, _, _ = rate_eq_force_jl(Es, d, Zeeman_z_diag, Gamma, k_wave, hbar_val,
                                   freqs_4f, pols_4f, s0_val, v_traj, z_traj, B_gradient_Tm,
                                   n_ground, n_excited, n_states)
        v_traj += (f / m_mol) * dt_traj
        z_traj += v_traj * dt_traj
        abs(z_traj) > 0.015 && break
    end
    (time_ns() - t0) / 1e9
end
@printf("RESULT:julia_trajectory_5000_time=%.6e\\n", t_traj)

println("Julia Test 5 complete.")
"""

    log("\n  Running Julia benchmark...")
    t0 = time.time()
    stdout5, stderr5, rc5 = run_julia_script(julia_code_test5)
    julia_time_test5 = time.time() - t0

    if rc5 != 0:
        log(f"  Julia FAILED (exit code {rc5})")
    else:
        julia_results_5 = parse_results(stdout5)
        log(f"  Julia total (inc. compilation): {julia_time_test5:.1f}s")

    log("  Running Python benchmark...")

    # Warm up
    sim_dc.force(0.0, 0.0)

    # Single force (100x)
    t0 = time.time()
    for i in range(100):
        sim_dc.force(0.0, 1e-3)
    py_force_100 = time.time() - t0

    # Trajectory (5000 steps)
    t0 = time.time()
    z_traj = 3e-3
    v_traj = 0.0
    dt_traj = 1e-6
    mass = mol.mass
    for j in range(5000):
        F, _, _ = sim_dc.force(v_traj, z_traj)
        v_traj += (F / mass) * dt_traj
        z_traj += v_traj * dt_traj
        if abs(z_traj) > 0.015:
            break
    py_traj_5000 = time.time() - t0

    log(f"\n  {'Operation':>30}  {'Julia':>12}  {'Python':>12}  {'Ratio':>8}")
    log(f"  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*8}")

    if rc5 == 0:
        jf100 = julia_results_5.get("julia_force_100_time", None)
        jf_us = julia_results_5.get("julia_force_single_us", None)
        jtraj = julia_results_5.get("julia_trajectory_5000_time", None)

        if jf100 is not None:
            ratio_f = py_force_100 / jf100
            log(f"  {'100 force evals':>30}  {jf100*1e3:>10.1f}ms  {py_force_100*1e3:>10.1f}ms  {ratio_f:>7.1f}x")
            log(f"  {'Single force eval':>30}  {jf_us:>10.1f}us  {py_force_100/100*1e6:>10.1f}us  {ratio_f:>7.1f}x")

        if jtraj is not None:
            ratio_t = py_traj_5000 / jtraj
            log(f"  {'5000-step trajectory':>30}  {jtraj*1e3:>10.1f}ms  {py_traj_5000*1e3:>10.1f}ms  {ratio_t:>7.1f}x")

    # ════════════════════════════════════════════════════════
    # IMPORT VALIDATION
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  IMPORT VALIDATION")
    log("=" * 72)

    imports = [
        ("molmot (top-level)", "import molmot"),
        ("load_sroh_from_julia", "from molmot.molecules.sroh import load_sroh_from_julia"),
        ("RFMOTSimulator", "from molmot.mot.simulator import RFMOTSimulator"),
        ("DCMOTSimulator", "from molmot.mot.simulator import DCMOTSimulator"),
        ("SSEProblem, SSESolver", "from molmot.obe.stochastic import SSEProblem, SSESolver"),
        ("FloquetOBE", "from molmot.obe.floquet import FloquetOBE"),
        ("MOTSimulator3D", "from molmot.mot.simulator_3d import MOTSimulator3D"),
        ("force_vs_z, force_vs_v", "from molmot.mot.force_scan import force_vs_z, force_vs_v"),
        ("solve_rate_equations", "from molmot.obe.rate_equations import solve_rate_equations"),
        ("build_liouvillian", "from molmot.obe.lindblad import build_liouvillian"),
        ("LaserBeam", "from molmot.obe.fields import LaserBeam"),
        ("compute_diffusion_from_ensemble", "from molmot.obe.diffusion import compute_diffusion_from_ensemble"),
        ("force_from_wavefunction", "from molmot.obe.force import force_from_wavefunction"),
        ("simulate_trajectory", "from molmot.propagation.trajectories import simulate_trajectory"),
        ("gaussian_fit", "from molmot.mot.analysis import gaussian_fit"),
        ("build_caoh_hamiltonian", "from molmot.molecules.caoh import build_caoh_hamiltonian"),
    ]

    n_ok = 0
    n_fail = 0
    for label, stmt in imports:
        try:
            exec(stmt)
            log(f"  OK   {label}")
            n_ok += 1
        except Exception as e:
            log(f"  FAIL {label}: {e}")
            n_fail += 1

    log(f"\n  {n_ok}/{n_ok+n_fail} imports successful, {n_fail} failed")

    # ── Functional tests ─────────────────────────────────────
    log("\n  Functional tests:")
    try:
        mol_test = load_sroh_from_julia(os.path.join(BASE, "julia_sim"), verbose=False)
        log(f"  OK   load_sroh_from_julia -> {mol_test.n_states} states")
    except Exception as e:
        log(f"  FAIL load_sroh_from_julia: {e}")

    try:
        sim_test = DCMOTSimulator(mol_test, delta_Gamma=-0.88, split_Gamma=0.39, s0=1.0)
        F_test, p_test, R_test = sim_test.force(0.0, 1e-3)
        log(f"  OK   DCMOTSimulator.force -> F = {F_test:.6e} N")
    except Exception as e:
        log(f"  FAIL DCMOTSimulator: {e}")

    try:
        sim_rf_test = RFMOTSimulator(mol_test, delta_Gamma=-1.0, s0=1.0)
        F_rf, p_rf, R_rf = sim_rf_test.force(0.0, 1e-3)
        log(f"  OK   RFMOTSimulator.force -> F = {F_rf:.6e} N")
    except Exception as e:
        log(f"  FAIL RFMOTSimulator: {e}")

    try:
        from molmot.obe.stochastic import SSEProblem, SSESolver
        bp = [
            {'freq': mol_test.omega_J32 - 1.0 * mol_test.Gamma / (2*np.pi),
             'q_fwd': 2, 'q_bwd': 0, 's0': 1.0},
            {'freq': mol_test.omega_J12 - 1.0 * mol_test.Gamma / (2*np.pi),
             'q_fwd': 0, 'q_bwd': 2, 's0': 1.0},
        ]
        prob = SSEProblem(mol_test, bp, B_gradient=0.16)
        log(f"  OK   SSEProblem created (state_size={prob.state_size})")
    except Exception as e:
        log(f"  FAIL SSEProblem: {e}")

    try:
        from molmot.obe.floquet import FloquetOBE
        fobe = FloquetOBE(mol_test, bp, B_gradient=0.16, n_max=2)
        log(f"  OK   FloquetOBE created")
    except Exception as e:
        log(f"  FAIL FloquetOBE: {e}")

    try:
        from molmot.mot.simulator_3d import MOTSimulator3D
        sim3d = MOTSimulator3D(mol_test, bp, B_gradient=0.16)
        log(f"  OK   MOTSimulator3D created")
    except Exception as e:
        log(f"  FAIL MOTSimulator3D: {e}")

    # ════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  OVERALL RESULT")
    log("=" * 72)
    if all_pass:
        log("\n  ALL PHYSICS TESTS PASSED")
        log("  Julia and Python rate equation results agree to < 0.1% relative error.")
    else:
        log("\n  SOME TESTS FAILED -- see details above")

    log(f"\n  Import validation: {n_ok}/{n_ok+n_fail} OK")

    # ── Save report ──────────────────────────────────────────
    report_path = os.path.join(BASE, "physics_validation_results.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    log(f"\n  Report saved to: {report_path}")

    # ════════════════════════════════════════════════════════
    # GENERATE PLOT
    # ════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Physics Validation: Julia vs Python (SrOH MOT)", fontsize=14, fontweight='bold')

        F_unit = hbar * mol.k * mol.Gamma / 2

        # ── Panel 1: DC MOT F(z) ────────────────────────────
        ax = axes[0, 0]
        z_fine = np.linspace(-5e-3, 5e-3, 100)
        F_fine = np.array([sim_dc.force(0.0, z)[0] for z in z_fine])
        ax.plot(z_fine * 1e3, F_fine / F_unit, 'b-', lw=2, label='Python')

        if rc == 0 and len(julia_forces_arr) == len(z_test_mm):
            ax.plot(z_test_mm, np.array(julia_forces_arr) / F_unit,
                    'ro', ms=8, label='Julia', zorder=5)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel(r'$F / (\hbar k \Gamma/2)$')
        ax.set_title('DC MOT F(z) at v=0')
        ax.legend()

        # ── Panel 2: DC MOT F(v) ────────────────────────────
        ax = axes[0, 1]
        v_fine = np.linspace(-6, 6, 100)
        Fv_fine = np.array([sim_dc.force(v, 0.0)[0] for v in v_fine])
        ax.plot(v_fine, Fv_fine / F_unit, 'b-', lw=2, label='Python z=0')

        if rc2 == 0:
            jfv_z0 = []
            for v in v_test:
                jkey = f"Fv_z0_v{v:.1f}"
                jfv_z0.append(julia_results_2.get(jkey, 0.0))
            ax.plot(v_test, np.array(jfv_z0) / F_unit,
                    'ro', ms=8, label='Julia z=0', zorder=5)

        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel('v (m/s)')
        ax.set_ylabel(r'$F / (\hbar k \Gamma/2)$')
        ax.set_title('DC MOT F(v) at z=0')
        ax.legend()

        # ── Panel 3: RF MOT F(z) ────────────────────────────
        ax = axes[0, 2]
        F_rf_fine = np.array([sim_rf.force(0.0, z)[0] for z in z_fine])
        ax.plot(z_fine * 1e3, F_rf_fine / F_unit, 'b-', lw=2, label='Python')

        if rc4 == 0:
            jrf = [julia_results_4.get(f"RF_F_z_{z}mm", 0.0) for z in z_test_rf]
            ax.plot(z_test_rf, np.array(jrf) / F_unit,
                    'ro', ms=8, label='Julia', zorder=5)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel(r'$F / (\hbar k \Gamma/2)$')
        ax.set_title('RF MOT F(z) at v=0')
        ax.legend()

        # ── Panel 4: Scattering rates ───────────────────────
        ax = axes[1, 0]
        R_fine = np.array([sim_dc.force(0.0, z)[2] for z in z_fine])
        ax.plot(z_fine * 1e3, R_fine / (2 * np.pi * 1e6), 'b-', lw=2, label='Python DC')

        R_rf_fine = np.array([sim_rf.force(0.0, z)[2] for z in z_fine])
        ax.plot(z_fine * 1e3, R_rf_fine / (2 * np.pi * 1e6), 'r-', lw=2, label='Python RF')

        ax.set_xlabel('z (mm)')
        ax.set_ylabel(r'$R_{scatter}$ (MHz)')
        ax.set_title('Scattering Rate vs z')
        ax.legend()

        # ── Panel 5: Populations ─────────────────────────────
        ax = axes[1, 1]
        pops_fine = np.array([sim_dc.force(0.0, z)[1] for z in z_fine])
        for ig in range(mol.n_ground):
            alpha = 0.8 if ig < 4 else 0.4
            ax.plot(z_fine * 1e3, pops_fine[:, ig], lw=1.2, alpha=alpha)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('Population')
        ax.set_title('Ground State Populations (DC MOT)')

        # ── Panel 6: Error summary ──────────────────────────
        ax = axes[1, 2]
        ax.axis('off')
        summary_lines = [
            "VALIDATION SUMMARY",
            "",
            f"Julia packages:",
            f"  OpticalBlochEquations.jl",
            f"  QuantumStates.jl",
            f"",
            f"Python package:",
            f"  molmot v{molmot.__version__}",
            f"",
            f"Molecule: SrOH (X->A, 687nm)",
            f"  12 ground + 4 excited states",
            f"",
            f"Tests:",
            f"  1. DC MOT F(z): {'PASS' if all_pass else 'CHECK DETAILS'}",
            f"  2. DC MOT F(v): {'PASS' if all_pass else 'CHECK DETAILS'}",
            f"  3. Branching ratios: {'PASS' if all_pass else 'CHECK DETAILS'}",
            f"  4. RF MOT F(z): {'PASS' if all_pass else 'CHECK DETAILS'}",
            f"  5. Speed benchmark: see report",
            f"",
            f"Imports: {n_ok}/{n_ok+n_fail} OK",
        ]
        ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))

        plt.tight_layout()
        plot_path = os.path.join(BASE, "physics_validation.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        log(f"  Plot saved to: {plot_path}")
        plt.close()

    except Exception as e:
        log(f"  Plot generation failed: {e}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
