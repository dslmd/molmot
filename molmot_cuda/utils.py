"""
Utility functions for molmot_cuda.

Provides GPU array transfer helpers, device info, and a container
class that holds molecular data arrays on the GPU to avoid repeated
host-to-device transfers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _require_cupy():
    """Raise ImportError if CuPy is not installed."""
    if not _CUPY_AVAILABLE:
        raise ImportError(
            "CuPy is required for CUDA-accelerated computation. "
            "Install with: pip install cupy-cuda12x  (adjust for your CUDA version)"
        )


def to_gpu(arr: np.ndarray) -> "cp.ndarray":
    """
    Transfer a NumPy array to GPU memory.

    Parameters
    ----------
    arr : np.ndarray
        Host array (any dtype).

    Returns
    -------
    cp.ndarray
        Device array with the same dtype.
    """
    _require_cupy()
    return cp.asarray(arr)


def to_cpu(arr) -> np.ndarray:
    """
    Transfer a CuPy array back to host memory.

    Parameters
    ----------
    arr : cp.ndarray or np.ndarray
        Device (or host) array.

    Returns
    -------
    np.ndarray
        Host array.
    """
    _require_cupy()
    if isinstance(arr, np.ndarray):
        return arr
    return cp.asnumpy(arr)


def get_device_info() -> dict:
    """
    Return a dictionary of GPU device information.

    Returns
    -------
    info : dict
        Keys: 'name', 'compute_capability', 'total_memory_MB',
        'free_memory_MB', 'driver_version', 'num_devices'.
    """
    _require_cupy()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    free, total = dev.mem_info
    info = {
        "name": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "total_memory_MB": total / (1024 ** 2),
        "free_memory_MB": free / (1024 ** 2),
        "num_devices": cp.cuda.runtime.getDeviceCount(),
    }
    return info


class MolecularDataGPU:
    """
    Container that holds molecular data arrays on the GPU.

    Mirrors the attributes of ``MolecularData`` but stores all
    large arrays as CuPy device arrays.  Scalar parameters
    (Gamma, k, mass, n_ground, ...) remain on the host.

    Parameters
    ----------
    mol_data : MolecularData
        CPU molecular data container.

    Attributes
    ----------
    energies : cp.ndarray, shape (n_states,)
    tdm_abs : cp.ndarray, shape (n_states, n_states, 3)
        ``|tdm|`` -- absolute values of TDMs (real, double).
    tdm : cp.ndarray, shape (n_states, n_states, 3)
        Full complex TDMs on GPU.
    d_squared : cp.ndarray, shape (n_ground, n_excited, 3)
    zeeman_x, zeeman_y, zeeman_z : cp.ndarray, shape (n_states, n_states)
    zeeman_z_diag : cp.ndarray, shape (n_states,)
    Gamma, k, mass, wavelength : float
    n_ground, n_excited, n_states : int
    omega_J12, omega_J32, omega_mean : float
    """

    def __init__(self, mol_data):
        _require_cupy()

        # Scalars stay on host
        self.Gamma = float(mol_data.Gamma)
        self.k = float(mol_data.k)
        self.mass = float(mol_data.mass)
        self.wavelength = float(mol_data.wavelength)
        self.n_ground = int(mol_data.n_ground)
        self.n_excited = int(mol_data.n_excited)
        self.n_states = int(mol_data.n_states)
        self.omega_J12 = float(mol_data.omega_J12)
        self.omega_J32 = float(mol_data.omega_J32)
        self.omega_mean = float(mol_data.omega_mean)

        # Arrays on GPU
        self.energies = cp.asarray(mol_data.energies, dtype=cp.float64)
        self.tdm = cp.asarray(mol_data.tdm, dtype=cp.complex128)
        self.tdm_abs = cp.abs(self.tdm).astype(cp.float64)
        self.d_squared = cp.asarray(mol_data.d_squared, dtype=cp.float64)
        self.zeeman_x = cp.asarray(mol_data.zeeman_x, dtype=cp.complex128)
        self.zeeman_y = cp.asarray(mol_data.zeeman_y, dtype=cp.complex128)
        self.zeeman_z = cp.asarray(mol_data.zeeman_z, dtype=cp.complex128)
        self.zeeman_z_diag = cp.asarray(mol_data.zeeman_z_diag, dtype=cp.float64)

        # Keep a reference to the CPU data for convenience
        self._cpu = mol_data

    @property
    def cpu(self):
        """Return the original CPU MolecularData."""
        return self._cpu

    def __repr__(self):
        return (f"MolecularDataGPU(n_ground={self.n_ground}, "
                f"n_excited={self.n_excited}, n_states={self.n_states})")
