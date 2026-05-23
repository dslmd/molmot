"""
Gaussian process kernels (covariance functions).

Port of the kernel functionality used by BayesianOptimization.jl.
Provides Squared Exponential, Matern 5/2, and ARD Matern 5/2 kernels
with methods for kernel matrix and kernel vector computation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Kernel(ABC):
    """Abstract base class for GP kernels."""

    @abstractmethod
    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Evaluate the kernel between two points.

        Parameters
        ----------
        x1, x2 : np.ndarray, shape (D,)
            Input points.

        Returns
        -------
        float
            k(x1, x2).
        """

    def kernelmatrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel (Gram) matrix for a set of points.

        Parameters
        ----------
        X : np.ndarray, shape (D, N)
            Input data, each column is one observation.

        Returns
        -------
        K : np.ndarray, shape (N, N)
            Symmetric positive-definite kernel matrix.
        """
        N = X.shape[1]
        K = np.empty((N, N))
        for i in range(N):
            K[i, i] = self.evaluate(X[:, i], X[:, i])
            for j in range(i + 1, N):
                val = self.evaluate(X[:, i], X[:, j])
                K[i, j] = val
                K[j, i] = val
        return K

    def kernelmatrix_diag(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the diagonal of the kernel matrix.

        Parameters
        ----------
        X : np.ndarray, shape (D, N)

        Returns
        -------
        diag : np.ndarray, shape (N,)
        """
        N = X.shape[1]
        return np.array([self.evaluate(X[:, i], X[:, i]) for i in range(N)])

    def kernelvector(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute the kernel vector between training data and a test point.

        Parameters
        ----------
        X : np.ndarray, shape (D, N)
            Training data.
        x : np.ndarray, shape (D,)
            Test point.

        Returns
        -------
        k : np.ndarray, shape (N,)
            k[i] = kernel(X[:, i], x).
        """
        N = X.shape[1]
        return np.array([self.evaluate(X[:, i], x) for i in range(N)])

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return kernel hyperparameters as a flat array (log-space)."""

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set kernel hyperparameters from a flat array (log-space)."""

    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""


# ---------------------------------------------------------------------------
# Squared Exponential (RBF) kernel
# ---------------------------------------------------------------------------

@dataclass
class SEKernel(Kernel):
    """
    Squared Exponential (RBF) kernel.

    k(x1, x2) = sigma_f^2 * exp(-||x1 - x2||^2 / (2 * l^2))

    Attributes
    ----------
    log_sigma_f : float
        Log signal standard deviation.
    log_l : float
        Log length scale (isotropic).
    """
    log_sigma_f: float = 0.0
    log_l: float = 0.0

    @property
    def sigma_f(self) -> float:
        return math.exp(self.log_sigma_f)

    @property
    def l(self) -> float:
        return math.exp(self.log_l)

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        diff = x1 - x2
        sq_dist = np.dot(diff, diff)
        return self.sigma_f ** 2 * math.exp(-sq_dist / (2.0 * self.l ** 2))

    def kernelmatrix(self, X: np.ndarray) -> np.ndarray:
        # Vectorised implementation for performance.
        N = X.shape[1]
        # Squared pairwise distances via (a-b)^2 = a^2 - 2ab + b^2
        XtX = X.T @ X
        sq_norms = np.diag(XtX)
        sq_dists = sq_norms[:, None] - 2.0 * XtX + sq_norms[None, :]
        np.maximum(sq_dists, 0.0, out=sq_dists)
        return self.sigma_f ** 2 * np.exp(-sq_dists / (2.0 * self.l ** 2))

    def kernelvector(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        diff = X - x[:, None]  # (D, N)
        sq_dists = np.sum(diff ** 2, axis=0)
        return self.sigma_f ** 2 * np.exp(-sq_dists / (2.0 * self.l ** 2))

    def get_params(self) -> np.ndarray:
        return np.array([self.log_sigma_f, self.log_l])

    def set_params(self, params: np.ndarray) -> None:
        self.log_sigma_f = float(params[0])
        self.log_l = float(params[1])

    def n_params(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Matern 5/2 kernel (isotropic)
# ---------------------------------------------------------------------------

@dataclass
class Matern52Kernel(Kernel):
    """
    Matern 5/2 kernel (isotropic).

    k(r) = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)

    where r = ||x1 - x2||.

    Attributes
    ----------
    log_sigma_f : float
        Log signal standard deviation.
    log_l : float
        Log length scale (isotropic).
    """
    log_sigma_f: float = 0.0
    log_l: float = 0.0

    @property
    def sigma_f(self) -> float:
        return math.exp(self.log_sigma_f)

    @property
    def l(self) -> float:
        return math.exp(self.log_l)

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        diff = x1 - x2
        r = math.sqrt(np.dot(diff, diff))
        s5 = math.sqrt(5.0)
        z = s5 * r / self.l
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * math.exp(-z)

    def kernelmatrix(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[1]
        XtX = X.T @ X
        sq_norms = np.diag(XtX)
        sq_dists = sq_norms[:, None] - 2.0 * XtX + sq_norms[None, :]
        np.maximum(sq_dists, 0.0, out=sq_dists)
        r = np.sqrt(sq_dists)
        s5 = math.sqrt(5.0)
        z = s5 * r / self.l
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    def kernelvector(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        diff = X - x[:, None]
        r = np.sqrt(np.sum(diff ** 2, axis=0))
        s5 = math.sqrt(5.0)
        z = s5 * r / self.l
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    def get_params(self) -> np.ndarray:
        return np.array([self.log_sigma_f, self.log_l])

    def set_params(self, params: np.ndarray) -> None:
        self.log_sigma_f = float(params[0])
        self.log_l = float(params[1])

    def n_params(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# ARD Matern 5/2 kernel (anisotropic / automatic relevance determination)
# ---------------------------------------------------------------------------

@dataclass
class ARDMatern52Kernel(Kernel):
    """
    Matern 5/2 kernel with Automatic Relevance Determination (ARD).

    Each input dimension has its own length scale, enabling the model
    to learn which dimensions are most relevant.

    k(x1, x2) = sigma_f^2 * (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)

    where r = sqrt(sum_d ((x1_d - x2_d)^2 / l_d^2)).

    Attributes
    ----------
    log_sigma_f : float
        Log signal standard deviation.
    log_l : np.ndarray, shape (D,)
        Log length scales, one per input dimension.
    """
    log_sigma_f: float = 0.0
    log_l: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def __post_init__(self):
        self.log_l = np.asarray(self.log_l, dtype=float)

    @property
    def sigma_f(self) -> float:
        return math.exp(self.log_sigma_f)

    @property
    def l(self) -> np.ndarray:
        return np.exp(self.log_l)

    @property
    def dim(self) -> int:
        return len(self.log_l)

    def _scaled_dist(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute ARD-scaled distance: sqrt(sum((x1-x2)^2 / l^2))."""
        diff = (x1 - x2) / self.l
        return math.sqrt(np.dot(diff, diff))

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        r = self._scaled_dist(x1, x2)
        s5 = math.sqrt(5.0)
        z = s5 * r
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * math.exp(-z)

    def kernelmatrix(self, X: np.ndarray) -> np.ndarray:
        # Scale each dimension by its length scale, then compute distances.
        L = self.l[:, None]  # (D, 1)
        X_scaled = X / L     # (D, N)
        N = X_scaled.shape[1]
        XtX = X_scaled.T @ X_scaled
        sq_norms = np.diag(XtX)
        sq_dists = sq_norms[:, None] - 2.0 * XtX + sq_norms[None, :]
        np.maximum(sq_dists, 0.0, out=sq_dists)
        r = np.sqrt(sq_dists)
        s5 = math.sqrt(5.0)
        z = s5 * r
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    def kernelvector(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        L = self.l[:, None]
        X_scaled = X / L
        x_scaled = x / self.l
        diff = X_scaled - x_scaled[:, None]
        r = np.sqrt(np.sum(diff ** 2, axis=0))
        s5 = math.sqrt(5.0)
        z = s5 * r
        return self.sigma_f ** 2 * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    def get_params(self) -> np.ndarray:
        return np.concatenate([[self.log_sigma_f], self.log_l])

    def set_params(self, params: np.ndarray) -> None:
        self.log_sigma_f = float(params[0])
        self.log_l = params[1:].copy()

    def n_params(self) -> int:
        return 1 + len(self.log_l)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_default_kernel(dim: int, kernel_type: str = "ard_matern52") -> Kernel:
    """
    Create a kernel with sensible defaults for a given input dimension.

    Parameters
    ----------
    dim : int
        Number of input dimensions.
    kernel_type : str
        One of 'se', 'matern52', 'ard_matern52'.

    Returns
    -------
    Kernel
    """
    if kernel_type == "se":
        return SEKernel()
    elif kernel_type == "matern52":
        return Matern52Kernel()
    elif kernel_type == "ard_matern52":
        return ARDMatern52Kernel(log_sigma_f=0.0, log_l=np.zeros(dim))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type!r}. "
                         f"Choose from 'se', 'matern52', 'ard_matern52'.")
