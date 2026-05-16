"""
Force scanning and MOT characterisation utilities.

Provides functions to compute force profiles, spring constants,
damping coefficients, capture velocities, and parameter optimisation.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..constants import hbar, k_B


def force_vs_z(simulator, z_arr: np.ndarray,
               v: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force as a function of position.

    Parameters
    ----------
    simulator : RFMOTSimulator or DCMOTSimulator
    z_arr : np.ndarray
        Array of positions (m).
    v : float
        Fixed velocity (m/s).

    Returns
    -------
    F_arr : np.ndarray
        Force at each position (N).
    pop_arr : np.ndarray, shape (len(z_arr), n_ground)
        Population at each position.
    R_arr : np.ndarray
        Scattering rate at each position (rad/s).
    """
    nz = len(z_arr)
    F_arr = np.zeros(nz)
    R_arr = np.zeros(nz)
    pop_arr = None

    for i, z in enumerate(z_arr):
        F, p, R = simulator.force(v, z)
        F_arr[i] = F
        R_arr[i] = R
        if pop_arr is None:
            pop_arr = np.zeros((nz, len(p)))
        pop_arr[i, :] = p

    return F_arr, pop_arr, R_arr


def force_vs_v(simulator, v_arr: np.ndarray,
               z: float = 0.0) -> np.ndarray:
    """
    Compute force as a function of velocity.

    Parameters
    ----------
    simulator : RFMOTSimulator or DCMOTSimulator
    v_arr : np.ndarray
        Array of velocities (m/s).
    z : float
        Fixed position (m).

    Returns
    -------
    F_arr : np.ndarray
        Force at each velocity (N).
    """
    nv = len(v_arr)
    F_arr = np.zeros(nv)

    for i, v in enumerate(v_arr):
        F, _, _ = simulator.force(v, z)
        F_arr[i] = F

    return F_arr


def spring_constant(simulator, z0: float = 0.0,
                    dz: float = 0.3e-3) -> float:
    """
    Compute the spring constant k = -dF/dz at position z0.

    Parameters
    ----------
    simulator : MOT simulator
    z0 : float
        Equilibrium position (m).
    dz : float
        Finite difference step (m).

    Returns
    -------
    k : float
        Spring constant (N/m).  Positive = restoring (trapping).
    """
    Fp, _, _ = simulator.force(0.0, z0 + dz)
    Fm, _, _ = simulator.force(0.0, z0 - dz)
    return -(Fp - Fm) / (2.0 * dz)


def damping_coefficient(simulator, z0: float = 0.0,
                        dv: float = 0.1) -> float:
    """
    Compute the damping coefficient beta = -(dF/dv)/m at position z0.

    Parameters
    ----------
    simulator : MOT simulator
    z0 : float
        Position (m).
    dv : float
        Finite difference step (m/s).

    Returns
    -------
    alpha : float
        Friction coefficient dF/dv (N*s/m).  To get damping rate beta
        in 1/s, divide by the molecular mass.
    """
    Fvp, _, _ = simulator.force(dv, z0)
    Fvm, _, _ = simulator.force(-dv, z0)
    return -(Fvp - Fvm) / (2.0 * dv)


def capture_velocity(simulator, z_start: float = 3e-3,
                     v_max: float = 15.0,
                     dv: float = 0.5,
                     dt: float = 1e-6,
                     n_steps: int = 5000,
                     z_escape: float = 0.015) -> float:
    """
    Estimate the capture velocity by trajectory simulation.

    A molecule starts at ``z_start`` with velocity ``-v0`` and is
    considered captured if |z| < z_escape after n_steps.

    Parameters
    ----------
    simulator : MOT simulator
    z_start : float
        Initial position (m).
    v_max : float
        Maximum test velocity (m/s).
    dv : float
        Velocity step (m/s).
    dt : float
        Time step for trajectory (s).
    n_steps : int
        Number of time steps.
    z_escape : float
        Escape boundary (m).

    Returns
    -------
    v_capture : float
        Maximum captured velocity (m/s).
    """
    mass = simulator.mol_data.mass

    v_capture = 0.0
    v_test = dv
    while v_test <= v_max:
        z = z_start
        v = -v_test
        trapped = True

        for _ in range(n_steps):
            F, _, _ = simulator.force(v, z)
            v += (F / mass) * dt
            z += v * dt
            if abs(z) > z_escape:
                trapped = False
                break

        if trapped:
            v_capture = v_test
        else:
            break

        v_test += dv

    return v_capture


def optimize_parameters(mol_data,
                        param_ranges: Dict[str, np.ndarray],
                        B_gradient: float,
                        mot_type: str = "dc",
                        s0: float = 1.0,
                        dz: float = 0.3e-3,
                        dv: float = 0.1,
                        verbose: bool = True) -> Dict:
    """
    Optimise MOT parameters by scanning detuning and split.

    Parameters
    ----------
    mol_data : MolecularData
    param_ranges : dict
        Must contain:
        - "delta": np.ndarray of detunings (in Gamma)
        - "split": np.ndarray of splits (in Gamma), only for DC MOT
    B_gradient : float
        Magnetic field gradient (T/m).
    mot_type : str
        "dc" or "rf".
    s0 : float
        Saturation parameter.
    dz, dv : float
        Finite difference steps.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with scan results and optimal parameters.
    """
    from .simulator import DCMOTSimulator, RFMOTSimulator

    delta_scan = param_ranges["delta"]
    mass = mol_data.mass

    if mot_type == "dc":
        split_scan = param_ranges.get("split", np.array([0.0]))
        n_d = len(delta_scan)
        n_s = len(split_scan)

        K_map = np.zeros((n_d, n_s))
        beta_map = np.zeros((n_d, n_s))

        t0 = time.time()
        total = n_d * n_s
        count = 0

        for i_d, det in enumerate(delta_scan):
            for i_s, sp in enumerate(split_scan):
                sim = DCMOTSimulator(mol_data, delta_Gamma=det,
                                     split_Gamma=sp, s0=s0,
                                     B_gradient=B_gradient)
                k = spring_constant(sim, dz=dz)
                alpha = damping_coefficient(sim, dv=dv)
                K_map[i_d, i_s] = k
                beta_map[i_d, i_s] = alpha / mass

                count += 1
                if verbose and count % 100 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  {count}/{total} ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

        # Find optimum
        merit = K_map.copy()
        merit[K_map <= 0] = 0
        merit[beta_map <= 0] = 0

        idx_best = np.unravel_index(np.argmax(merit), merit.shape)
        delta_best = delta_scan[idx_best[0]]
        split_best = split_scan[idx_best[1]]

        return {
            "delta_scan": delta_scan,
            "split_scan": split_scan,
            "K_map": K_map,
            "beta_map": beta_map,
            "delta_best": float(delta_best),
            "split_best": float(split_best),
            "k_best": float(K_map[idx_best]),
            "beta_best": float(beta_map[idx_best]),
        }

    elif mot_type == "rf":
        n_d = len(delta_scan)
        K_arr = np.zeros(n_d)
        beta_arr = np.zeros(n_d)

        for i_d, det in enumerate(delta_scan):
            sim = RFMOTSimulator(mol_data, delta_Gamma=det,
                                  s0=s0, B_gradient=B_gradient)
            k = spring_constant(sim, dz=dz)
            alpha = damping_coefficient(sim, dv=dv)
            K_arr[i_d] = k
            beta_arr[i_d] = alpha / mass

        idx_best = np.argmax(K_arr * (K_arr > 0) * (beta_arr > 0))
        return {
            "delta_scan": delta_scan,
            "K_arr": K_arr,
            "beta_arr": beta_arr,
            "delta_best": float(delta_scan[idx_best]),
            "k_best": float(K_arr[idx_best]),
            "beta_best": float(beta_arr[idx_best]),
        }

    else:
        raise ValueError(f"Unknown mot_type: {mot_type}")
