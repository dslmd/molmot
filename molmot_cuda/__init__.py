"""
molmot_cuda -- CUDA-accelerated MOT simulation.

Provides GPU-batched versions of the rate equation solver, Floquet OBE,
Lindblad master equation solver, sub-Doppler OBE, 3D MOT simulator,
force scanning utilities, and Monte Carlo wavefunction (SSE) trajectory
solver from the molmot package.  Uses CuPy for GPU computation with
double precision throughout.

Modules
-------
- rate_equations_cuda: GPU-batched 1D rate equation solver for force scans
- simulator_3d_cuda: GPU-batched 3D MOT rate equation force + trajectories
- floquet_cuda: GPU-batched Floquet OBE sub-Doppler force
- lindblad_cuda: GPU-batched Lindblad solver + sub-Doppler OBE
- stochastic_cuda: GPU-parallel SSE / quantum trajectory solver
- utils: GPU array transfer, device info, MolecularDataGPU container

Quick start
-----------
>>> from molmot_cuda import has_cuda, get_device_info
>>> if has_cuda():
...     print(get_device_info())
...     from molmot_cuda import force_vs_velocity_cuda
...     F = force_vs_velocity_cuda(mol_data, beam_pairs, v_arr)
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# CuPy availability check
# ---------------------------------------------------------------------------

_CUPY_AVAILABLE = False
_CUPY_IMPORT_ERROR = None

try:
    import cupy as _cp
    _CUPY_AVAILABLE = True
except ImportError as e:
    _CUPY_IMPORT_ERROR = e


def has_cuda() -> bool:
    """
    Check whether CuPy is installed and a CUDA GPU is available.

    Returns
    -------
    bool
        True if CuPy is importable and at least one GPU device exists.
    """
    if not _CUPY_AVAILABLE:
        return False
    try:
        _cp.cuda.Device(0)
        return True
    except _cp.cuda.runtime.CUDARuntimeError:
        return False


# ---------------------------------------------------------------------------
# Utilities (always importable, raise on use if no CuPy)
# ---------------------------------------------------------------------------

from .utils import to_gpu, to_cpu, get_device_info, MolecularDataGPU

# ---------------------------------------------------------------------------
# CUDA module imports (guarded: only when CuPy is present)
# ---------------------------------------------------------------------------

if _CUPY_AVAILABLE:
    # Existing 1D rate equation solver
    from .rate_equations_cuda import (
        CUDAMolData,
        solve_rate_equations_batch,
        force_vs_z_cuda,
        force_vs_v_cuda,
        force_map_2d_cuda,
        spring_constant_cuda,
        damping_coefficient_cuda,
        optimize_parameters_cuda,
    )

    # 3D MOT simulator
    from .simulator_3d_cuda import (
        rate_eq_force_3d_batch,
        compute_3d_force_map_cuda,
        simulate_trajectory_3d_cuda,
        simulate_ensemble_3d_cuda,
    )

    # Floquet OBE
    from .floquet_cuda import (
        force_vs_velocity_cuda,
        force_vs_velocity_cuda_fast,
        scattering_rate_vs_velocity_cuda,
    )

    # Lindblad / sub-Doppler OBE
    from .lindblad_cuda import (
        build_liouvillian_single,
        build_liouvillian_batch,
        steady_state_batch,
        obe_force_subdoppler_cuda,
        obe_force_map_cuda,
    )

    # Stochastic Schrodinger equation (SSE / MCWF)
    from .stochastic_cuda import (
        SSESolverCUDA,
        run_ensemble_cuda,
    )

else:
    # Define stubs that raise informative errors
    def _make_stub(name):
        def stub(*args, **kwargs):
            raise ImportError(
                f"molmot_cuda.{name}() requires CuPy. "
                f"Install with: pip install cupy-cuda12x  "
                f"(adjust suffix for your CUDA version). "
                f"Original error: {_CUPY_IMPORT_ERROR}"
            )
        stub.__name__ = name
        stub.__doc__ = f"Stub for {name} (CuPy not available)."
        return stub

    # Existing 1D rate equation stubs
    CUDAMolData = _make_stub("CUDAMolData")
    solve_rate_equations_batch = _make_stub("solve_rate_equations_batch")
    force_vs_z_cuda = _make_stub("force_vs_z_cuda")
    force_vs_v_cuda = _make_stub("force_vs_v_cuda")
    force_map_2d_cuda = _make_stub("force_map_2d_cuda")
    spring_constant_cuda = _make_stub("spring_constant_cuda")
    damping_coefficient_cuda = _make_stub("damping_coefficient_cuda")
    optimize_parameters_cuda = _make_stub("optimize_parameters_cuda")

    # 3D MOT simulator stubs
    rate_eq_force_3d_batch = _make_stub("rate_eq_force_3d_batch")
    compute_3d_force_map_cuda = _make_stub("compute_3d_force_map_cuda")
    simulate_trajectory_3d_cuda = _make_stub("simulate_trajectory_3d_cuda")
    simulate_ensemble_3d_cuda = _make_stub("simulate_ensemble_3d_cuda")

    # Floquet OBE stubs
    force_vs_velocity_cuda = _make_stub("force_vs_velocity_cuda")
    force_vs_velocity_cuda_fast = _make_stub("force_vs_velocity_cuda_fast")
    scattering_rate_vs_velocity_cuda = _make_stub("scattering_rate_vs_velocity_cuda")

    # Lindblad / sub-Doppler OBE stubs
    build_liouvillian_single = _make_stub("build_liouvillian_single")
    build_liouvillian_batch = _make_stub("build_liouvillian_batch")
    steady_state_batch = _make_stub("steady_state_batch")
    obe_force_subdoppler_cuda = _make_stub("obe_force_subdoppler_cuda")
    obe_force_map_cuda = _make_stub("obe_force_map_cuda")

    # SSE stubs
    SSESolverCUDA = _make_stub("SSESolverCUDA")
    run_ensemble_cuda = _make_stub("run_ensemble_cuda")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Meta
    "has_cuda",
    "__version__",
    # Utilities
    "to_gpu",
    "to_cpu",
    "get_device_info",
    "MolecularDataGPU",
    # Existing 1D rate equation solver
    "CUDAMolData",
    "solve_rate_equations_batch",
    "force_vs_z_cuda",
    "force_vs_v_cuda",
    "force_map_2d_cuda",
    "spring_constant_cuda",
    "damping_coefficient_cuda",
    "optimize_parameters_cuda",
    # 3D MOT simulator
    "rate_eq_force_3d_batch",
    "compute_3d_force_map_cuda",
    "simulate_trajectory_3d_cuda",
    "simulate_ensemble_3d_cuda",
    # Floquet OBE
    "force_vs_velocity_cuda",
    "force_vs_velocity_cuda_fast",
    "scattering_rate_vs_velocity_cuda",
    # Lindblad / sub-Doppler OBE
    "build_liouvillian_single",
    "build_liouvillian_batch",
    "steady_state_batch",
    "obe_force_subdoppler_cuda",
    "obe_force_map_cuda",
    # Stochastic SSE
    "SSESolverCUDA",
    "run_ensemble_cuda",
]
