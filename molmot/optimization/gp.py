"""
Gaussian Process regression.

Port of the GP model used by BayesianOptimization.jl, built on
GaussianProcesses.jl.  Uses Cholesky decomposition with jitter for
numerical stability and scipy for hyperparameter optimisation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import minimize

from .kernels import Kernel, ARDMatern52Kernel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JITTER_INITIAL = 1e-8
_JITTER_MAX = 1e-2
_JITTER_FACTOR = 10.0


# ---------------------------------------------------------------------------
# GaussianProcess
# ---------------------------------------------------------------------------

@dataclass
class GaussianProcess:
    """
    Gaussian Process regression model with a configurable kernel.

    Implements exact GP inference with Cholesky decomposition, log
    marginal likelihood computation, and hyperparameter optimisation.

    Attributes
    ----------
    kernel : Kernel
        Covariance function.
    log_noise : float
        Log observation noise standard deviation.  The noise variance
        added to the diagonal of the kernel matrix is exp(2 * log_noise).
    mean : float
        Constant prior mean function.
    X : np.ndarray or None
        Training inputs, shape (D, N).  Each column is one observation.
    y : np.ndarray or None
        Training targets, shape (N,).
    """
    kernel: Kernel
    log_noise: float = -2.0
    mean: float = 0.0
    X: Optional[np.ndarray] = field(default=None, repr=False)
    y: Optional[np.ndarray] = field(default=None, repr=False)

    # Cached Cholesky factor and alpha vector (set by fit).
    _L: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    _alpha: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    _log_marginal: Optional[float] = field(default=None, repr=False, init=False)

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def noise_variance(self) -> float:
        """Observation noise variance sigma_n^2."""
        return math.exp(2.0 * self.log_noise)

    @property
    def n_obs(self) -> int:
        """Number of training observations."""
        if self.y is None:
            return 0
        return len(self.y)

    @property
    def input_dim(self) -> int:
        """Dimensionality of the input space."""
        if self.X is None:
            return 0
        return self.X.shape[0]

    # ---------------------------------------------------------------
    # Fit
    # ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP to training data (replaces any existing data).

        Parameters
        ----------
        X : np.ndarray, shape (D, N)
            Training inputs.
        y : np.ndarray, shape (N,)
            Training targets.
        """
        self.X = np.array(X, dtype=float, order="C")
        self.y = np.array(y, dtype=float).ravel()
        self._update_cache()

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Append new observations and refit.

        Parameters
        ----------
        X_new : np.ndarray, shape (D, M)
            New inputs.
        y_new : np.ndarray, shape (M,)
            New targets.
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new).ravel()
        if self.X is None or self.y is None or len(self.y) == 0:
            self.fit(X_new, y_new)
        else:
            self.X = np.hstack([self.X, X_new])
            self.y = np.concatenate([self.y, y_new])
            self._update_cache()

    # ---------------------------------------------------------------
    # Cache (Cholesky decomposition)
    # ---------------------------------------------------------------

    def _update_cache(self) -> None:
        """Recompute Cholesky factor and alpha from current data."""
        if self.X is None or self.y is None or len(self.y) == 0:
            self._L = None
            self._alpha = None
            self._log_marginal = None
            return

        K = self.kernel.kernelmatrix(self.X)
        N = K.shape[0]
        noise_var = self.noise_variance

        # Add noise + jitter on the diagonal for numerical stability.
        jitter = _JITTER_INITIAL
        while jitter <= _JITTER_MAX:
            try:
                Ky = K + (noise_var + jitter) * np.eye(N)
                L, lower = cho_factor(Ky, lower=True, check_finite=False)
                break
            except LinAlgError:
                jitter *= _JITTER_FACTOR
        else:
            # Last resort: use eigenvalue repair.
            Ky = K + noise_var * np.eye(N)
            eigvals, eigvecs = np.linalg.eigh(Ky)
            eigvals = np.maximum(eigvals, _JITTER_INITIAL)
            Ky = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L, lower = cho_factor(Ky, lower=True, check_finite=False)

        residual = self.y - self.mean
        alpha = cho_solve((L, True), residual, check_finite=False)

        # Log marginal likelihood
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_ml = -0.5 * (residual @ alpha + log_det + N * math.log(2.0 * math.pi))

        self._L = L
        self._alpha = alpha
        self._log_marginal = log_ml

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    def predict(self, X_star: np.ndarray,
                return_var: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Predict at test points.

        Parameters
        ----------
        X_star : np.ndarray, shape (D,) or (D, M)
            Test inputs.
        return_var : bool
            If True, also return predictive variance.

        Returns
        -------
        mu : np.ndarray, shape (M,)
            Predictive mean.
        var : np.ndarray, shape (M,)
            Predictive variance (only if *return_var* is True).
        """
        X_star = np.atleast_2d(X_star)
        if X_star.ndim == 1:
            X_star = X_star[:, None]
        # If X_star was passed as (M, D) with M > D, try to detect and transpose
        if X_star.shape[0] != self.input_dim and X_star.shape[1] == self.input_dim:
            X_star = X_star.T

        M = X_star.shape[1]

        # No training data: return prior.
        if self._alpha is None:
            mu = np.full(M, self.mean)
            if return_var:
                var = self.kernel.kernelmatrix_diag(X_star)
                return mu, var
            return (mu,)

        # Kernel vectors between training and test data.
        k_star = np.column_stack(
            [self.kernel.kernelvector(self.X, X_star[:, j]) for j in range(M)]
        )  # shape (N, M)

        mu = self.mean + k_star.T @ self._alpha  # (M,)

        if not return_var:
            return (mu,)

        # Predictive variance.
        v = cho_solve((self._L, True), k_star, check_finite=False)  # (N, M)
        k_ss = self.kernel.kernelmatrix_diag(X_star)  # (M,)
        var = k_ss - np.sum(k_star * v, axis=0)
        np.maximum(var, 0.0, out=var)

        return mu, var

    def predict_single(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict mean and variance at a single test point.

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        mu : float
        var : float
        """
        x = np.asarray(x).ravel()
        if self._alpha is None:
            return self.mean, float(self.kernel.evaluate(x, x))

        k_star = self.kernel.kernelvector(self.X, x)  # (N,)
        mu = self.mean + float(k_star @ self._alpha)
        v = cho_solve((self._L, True), k_star, check_finite=False)
        var = float(self.kernel.evaluate(x, x) - k_star @ v)
        return mu, max(var, 0.0)

    def sample(self, x: np.ndarray) -> float:
        """
        Draw a single sample from the posterior predictive at *x*.

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        float
        """
        mu, var = self.predict_single(x)
        return float(np.random.normal(mu, math.sqrt(max(var, 0.0))))

    # ---------------------------------------------------------------
    # Log marginal likelihood
    # ---------------------------------------------------------------

    def log_marginal_likelihood(self) -> float:
        """
        Return the log marginal likelihood of the current fit.

        Returns
        -------
        float
            log p(y | X, theta).  Returns -inf if no data has been fit.
        """
        if self._log_marginal is None:
            return -math.inf
        return self._log_marginal

    # ---------------------------------------------------------------
    # Hyperparameter optimisation
    # ---------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        """
        Return all hyperparameters as a flat array.

        Layout: [kernel_params..., log_noise, mean].
        """
        return np.concatenate([
            self.kernel.get_params(),
            [self.log_noise, self.mean],
        ])

    def set_params(self, params: np.ndarray) -> None:
        """Set all hyperparameters from a flat array."""
        n_kern = self.kernel.n_params()
        self.kernel.set_params(params[:n_kern])
        self.log_noise = float(params[n_kern])
        self.mean = float(params[n_kern + 1])
        self._update_cache()

    def n_params(self) -> int:
        """Total number of hyperparameters."""
        return self.kernel.n_params() + 2  # +log_noise +mean

    def optimize_hyperparams(
        self,
        n_restarts: int = 5,
        maxiter: int = 500,
        bounds: Optional[np.ndarray] = None,
        optimize_mean: bool = True,
        optimize_noise: bool = True,
        method: str = "L-BFGS-B",
    ) -> float:
        """
        Optimise hyperparameters by maximising the log marginal likelihood.

        Parameters
        ----------
        n_restarts : int
            Number of random restarts for the optimiser.
        maxiter : int
            Maximum iterations per restart.
        bounds : np.ndarray or None, shape (n_params, 2)
            Parameter bounds [[lo, hi], ...].  If None, sensible defaults
            are used.
        optimize_mean : bool
            Whether to include the mean in the optimisation.
        optimize_noise : bool
            Whether to include the noise in the optimisation.
        method : str
            Scipy minimiser method.

        Returns
        -------
        float
            Best log marginal likelihood found.
        """
        if self.X is None or self.y is None or len(self.y) < 2:
            return self.log_marginal_likelihood()

        n_kern = self.kernel.n_params()
        n_total = n_kern + 2  # kernel + log_noise + mean

        # Build bounds.
        if bounds is None:
            bounds = self._default_bounds(optimize_mean, optimize_noise)
        else:
            bounds = np.asarray(bounds)

        # Indices of parameters to optimise.
        active = list(range(n_kern))
        if optimize_noise:
            active.append(n_kern)
        if optimize_mean:
            active.append(n_kern + 1)

        all_params = self.get_params().copy()
        active_bounds = [tuple(bounds[i]) for i in active]

        def neg_lml(p_active):
            params = all_params.copy()
            for idx, val in zip(active, p_active):
                params[idx] = val
            self.set_params(params)
            lml = self.log_marginal_likelihood()
            if not np.isfinite(lml):
                return 1e10
            return -lml

        best_lml = -np.inf
        best_params = all_params.copy()

        # Start from current parameters as first attempt.
        starting_points = [np.array([all_params[i] for i in active])]
        # Add random restarts.
        rng = np.random.default_rng()
        for _ in range(n_restarts - 1):
            p0 = np.array([
                rng.uniform(lo, hi)
                for lo, hi in active_bounds
            ])
            starting_points.append(p0)

        for p0 in starting_points:
            try:
                result = minimize(
                    neg_lml, p0,
                    method=method,
                    bounds=active_bounds,
                    options={"maxiter": maxiter, "disp": False},
                )
                if result.success or np.isfinite(result.fun):
                    lml = -result.fun
                    if lml > best_lml:
                        best_lml = lml
                        params = all_params.copy()
                        for idx, val in zip(active, result.x):
                            params[idx] = val
                        best_params = params.copy()
            except Exception:
                continue

        self.set_params(best_params)
        return best_lml

    def _default_bounds(self, optimize_mean: bool,
                        optimize_noise: bool) -> np.ndarray:
        """Build sensible default parameter bounds."""
        n_kern = self.kernel.n_params()
        bounds = np.zeros((n_kern + 2, 2))

        # Kernel parameters: [-4, 4] in log space.
        for i in range(n_kern):
            bounds[i] = [-4.0, 4.0]

        # Noise: [-6, 2] in log space (approx 2e-3 to 7).
        bounds[n_kern] = [-6.0, 2.0]

        # Mean: based on data range if available.
        if self.y is not None and len(self.y) > 0:
            y_range = float(np.ptp(self.y))
            y_mid = float(np.mean(self.y))
            bounds[n_kern + 1] = [y_mid - 2.0 * max(y_range, 1.0),
                                  y_mid + 2.0 * max(y_range, 1.0)]
        else:
            bounds[n_kern + 1] = [-10.0, 10.0]

        return bounds

    # ---------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------

    def maxy(self) -> float:
        """Return maximum observed target, or -inf if empty."""
        if self.y is None or len(self.y) == 0:
            return -math.inf
        return float(np.max(self.y))

    def dims(self) -> Tuple[int, int]:
        """Return (input_dim, n_observations)."""
        if self.X is None:
            return (0, 0)
        return self.X.shape

    def __repr__(self) -> str:
        return (f"GaussianProcess(kernel={self.kernel!r}, "
                f"log_noise={self.log_noise:.3f}, mean={self.mean:.3f}, "
                f"n_obs={self.n_obs})")


# ---------------------------------------------------------------------------
# MAPGPOptimizer — mirrors the Julia MAPGPOptimizer
# ---------------------------------------------------------------------------

@dataclass
class MAPGPOptimizer:
    """
    Optimise GP hyperparameters to the MAP estimate every *every* steps.

    Parameters
    ----------
    every : int
        Optimise every *every* model updates.
    n_restarts : int
        Number of random restarts for L-BFGS-B.
    maxiter : int
        Maximum iterations per restart.
    noise_bounds : tuple of float or None
        (lo, hi) bounds on log_noise.
    kern_bounds : np.ndarray or None
        Shape (n_kern_params, 2), bounds on kernel parameters.
    method : str
        Scipy optimizer method.
    optimize_mean : bool
        Whether to include mean in the optimization.
    optimize_noise : bool
        Whether to include noise in the optimization.
    """
    every: int = 10
    n_restarts: int = 5
    maxiter: int = 500
    noise_bounds: Optional[Tuple[float, float]] = (-4.0, 3.0)
    kern_bounds: Optional[np.ndarray] = None
    method: str = "L-BFGS-B"
    optimize_mean: bool = True
    optimize_noise: bool = True
    _counter: int = field(default=0, repr=False, init=False)

    def optimize_model(self, gp: GaussianProcess) -> None:
        """
        Conditionally optimise GP hyperparameters.

        Optimisation runs every *self.every* calls.
        """
        if self._counter % self.every == 0:
            bounds = self._build_bounds(gp)
            gp.optimize_hyperparams(
                n_restarts=self.n_restarts,
                maxiter=self.maxiter,
                bounds=bounds,
                optimize_mean=self.optimize_mean,
                optimize_noise=self.optimize_noise,
                method=self.method,
            )
        self._counter += 1

    def _build_bounds(self, gp: GaussianProcess) -> Optional[np.ndarray]:
        """Assemble bounds array from component bounds."""
        n_kern = gp.kernel.n_params()
        bounds = np.zeros((n_kern + 2, 2))

        # Kernel bounds.
        if self.kern_bounds is not None:
            kb = np.asarray(self.kern_bounds)
            if kb.shape == (2, n_kern):
                # Passed as [[lowers], [uppers]].
                bounds[:n_kern, 0] = kb[0]
                bounds[:n_kern, 1] = kb[1]
            elif kb.shape == (n_kern, 2):
                bounds[:n_kern] = kb
            else:
                # Fall back to defaults.
                bounds[:n_kern] = [[-4.0, 4.0]] * n_kern
        else:
            bounds[:n_kern] = [[-4.0, 4.0]] * n_kern

        # Noise bounds.
        if self.noise_bounds is not None:
            bounds[n_kern] = list(self.noise_bounds)
        else:
            bounds[n_kern] = [-6.0, 2.0]

        # Mean bounds.
        if gp.y is not None and len(gp.y) > 0:
            y_range = float(np.ptp(gp.y))
            y_mid = float(np.mean(gp.y))
            bounds[n_kern + 1] = [y_mid - 2.0 * max(y_range, 1.0),
                                  y_mid + 2.0 * max(y_range, 1.0)]
        else:
            bounds[n_kern + 1] = [-10.0, 10.0]

        return bounds


# ---------------------------------------------------------------------------
# NoModelOptimizer — never optimises
# ---------------------------------------------------------------------------

class NoModelOptimizer:
    """Never optimise the GP hyperparameters."""

    def optimize_model(self, gp: GaussianProcess) -> None:
        pass
