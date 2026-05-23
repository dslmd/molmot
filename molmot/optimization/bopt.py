"""
Bayesian optimiser (BOpt).

Port of the main BOpt struct and boptimize! from BayesianOptimization.jl.
Provides single-point and batch Bayesian optimisation with Sobol / LHS
initialisation, pluggable kernels, acquisition functions, and GP
hyperparameter optimisation strategies.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (Callable, Dict, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .kernels import Kernel, ARDMatern52Kernel, make_default_kernel
from .gp import GaussianProcess, MAPGPOptimizer, NoModelOptimizer
from .acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    MaxMean,
    ThompsonSampling,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Sense(IntEnum):
    """Optimisation direction."""
    MIN = -1
    MAX = 1


class Verbosity(IntEnum):
    """Verbosity levels."""
    SILENT = 0
    TIMINGS = 1
    PROGRESS = 2


# ---------------------------------------------------------------------------
# Initialisation iterators
# ---------------------------------------------------------------------------

class ScaledSobolIterator:
    """
    Quasi-random Sobol sequence scaled to [lowerbounds, upperbounds].

    Uses a simple Python implementation that does not require external
    Sobol libraries.  For high-dimensional problems consider replacing
    with scipy.stats.qmc.Sobol.

    Parameters
    ----------
    lowerbounds : np.ndarray
    upperbounds : np.ndarray
    n : int
        Number of points to generate.
    """
    def __init__(self, lowerbounds: np.ndarray, upperbounds: np.ndarray,
                 n: int):
        self.lowerbounds = np.asarray(lowerbounds, dtype=float)
        self.upperbounds = np.asarray(upperbounds, dtype=float)
        self.n = n
        self._dim = len(lowerbounds)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=self._dim, scramble=True)
            # Skip the first n points for better uniformity, then draw n.
            _ = sampler.fast_forward(self.n)
            points = sampler.random(self.n)  # shape (n, D) in [0, 1]^D
        except ImportError:
            # Fallback to LHS if scipy.stats.qmc is unavailable.
            points = _latin_hypercube_01(self._dim, self.n)

        for i in range(self.n):
            yield (self.lowerbounds
                   + points[i] * (self.upperbounds - self.lowerbounds))


class ScaledLHSIterator:
    """
    Latin Hypercube Sampling iterator scaled to [lowerbounds, upperbounds].

    Parameters
    ----------
    lowerbounds : np.ndarray
    upperbounds : np.ndarray
    n : int
        Number of points to generate.
    """
    def __init__(self, lowerbounds: np.ndarray, upperbounds: np.ndarray,
                 n: int):
        self.lowerbounds = np.asarray(lowerbounds, dtype=float)
        self.upperbounds = np.asarray(upperbounds, dtype=float)
        self.n = n
        self._dim = len(lowerbounds)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        points = _latin_hypercube_01(self._dim, self.n)
        for i in range(self.n):
            yield (self.lowerbounds
                   + points[i] * (self.upperbounds - self.lowerbounds))


def _latin_hypercube_01(dim: int, n: int) -> np.ndarray:
    """
    Generate an LHS sample in [0, 1]^dim.

    Returns
    -------
    points : np.ndarray, shape (n, dim)
    """
    result = np.empty((n, dim))
    for d in range(dim):
        perm = np.random.permutation(n)
        result[:, d] = (perm + np.random.rand(n)) / n
    return result


# ---------------------------------------------------------------------------
# Acquisition optimisation
# ---------------------------------------------------------------------------

def _optimize_acquisition(
    acq: AcquisitionFunction,
    model: GaussianProcess,
    lowerbounds: np.ndarray,
    upperbounds: np.ndarray,
    n_restarts: int = 10,
    method: str = "L-BFGS-B",
    maxeval: int = 2000,
) -> Tuple[float, np.ndarray]:
    """
    Maximise the acquisition function over the search domain.

    Uses multi-start L-BFGS-B (or a gradient-free method for Thompson
    Sampling) with LHS-initialised starting points.

    Parameters
    ----------
    acq : AcquisitionFunction
    model : GaussianProcess
    lowerbounds, upperbounds : np.ndarray, shape (D,)
    n_restarts : int
    method : str
    maxeval : int

    Returns
    -------
    best_val : float
        Best acquisition value found.
    best_x : np.ndarray, shape (D,)
        Corresponding input point.
    """
    D = len(lowerbounds)
    bounds = list(zip(lowerbounds, upperbounds))

    # Use gradient-free method for Thompson Sampling.
    if isinstance(acq, ThompsonSampling):
        method = "Nelder-Mead"

    def neg_acq(x):
        mu, var = model.predict_single(x)
        return -acq.evaluate(mu, var)

    best_val = -np.inf
    best_x = lowerbounds.copy()

    # Generate starting points with LHS.
    starts = _latin_hypercube_01(D, n_restarts)
    for i in range(n_restarts):
        x0 = lowerbounds + starts[i] * (upperbounds - lowerbounds)
        try:
            if method == "Nelder-Mead":
                result = minimize(neg_acq, x0, method="Nelder-Mead",
                                  options={"maxfev": maxeval, "adaptive": True})
            else:
                result = minimize(neg_acq, x0, method=method, bounds=bounds,
                                  options={"maxiter": maxeval, "maxfun": maxeval})
            # Clip to bounds.
            x_candidate = np.clip(result.x, lowerbounds, upperbounds)
            val = -neg_acq(x_candidate)
            if val > best_val:
                best_val = val
                best_x = x_candidate.copy()
        except Exception:
            continue

    return best_val, best_x


# ---------------------------------------------------------------------------
# BOpt
# ---------------------------------------------------------------------------

@dataclass
class BOpt:
    """
    Bayesian Optimisation driver.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float.  *x* is np.ndarray of shape (D,).
    lowerbounds : np.ndarray
        Lower bounds of the search space.
    upperbounds : np.ndarray
        Upper bounds of the search space.
    model : GaussianProcess or None
        GP surrogate model.  If None, a default ARD Matern 5/2 GP is
        created automatically.
    acquisition : AcquisitionFunction or None
        Acquisition function.  Defaults to ExpectedImprovement.
    model_optimizer : object
        GP hyperparameter optimiser.  Defaults to MAPGPOptimizer(every=20).
    sense : Sense
        MIN or MAX.  Default MAX.
    max_iterations : int
        Maximum number of function evaluations (including initialisation).
    max_duration : float
        Maximum wall-clock time in seconds.
    repetitions : int
        Number of repeated evaluations per candidate (for noisy functions).
    verbosity : Verbosity
        SILENT, TIMINGS, or PROGRESS.
    initializer_iterations : int or None
        Number of initialisation points.  Defaults to 5 * D.
    initializer : iterator or None
        Custom initialisation iterator.  Defaults to ScaledSobolIterator.
    batch_size : int
        Number of candidates to evaluate in parallel per iteration.
        Default 1 (sequential).
    acq_restarts : int
        Number of random restarts for acquisition optimisation.
    acq_method : str
        Scipy optimiser method for acquisition optimisation.
    """
    func: Callable
    lowerbounds: np.ndarray
    upperbounds: np.ndarray
    model: Optional[GaussianProcess] = None
    acquisition: Optional[AcquisitionFunction] = None
    model_optimizer: object = None
    sense: Sense = Sense.MAX
    max_iterations: int = 10_000
    max_duration: float = math.inf
    repetitions: int = 1
    verbosity: Verbosity = Verbosity.PROGRESS
    initializer_iterations: Optional[int] = None
    initializer: object = None
    batch_size: int = 1
    acq_restarts: int = 10
    acq_method: str = "L-BFGS-B"

    # Results.
    observed_optimum: float = field(default=-math.inf, init=False)
    observed_optimizer: Optional[np.ndarray] = field(default=None, init=False)
    model_optimum: float = field(default=-math.inf, init=False)
    _model_best_x: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Internal counters.
    _iteration: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _timings: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.lowerbounds = np.asarray(self.lowerbounds, dtype=float)
        self.upperbounds = np.asarray(self.upperbounds, dtype=float)
        D = len(self.lowerbounds)

        if len(self.lowerbounds) != len(self.upperbounds):
            raise ValueError("lowerbounds and upperbounds must have the "
                             "same length.")
        if not np.all(self.lowerbounds <= self.upperbounds):
            raise ValueError("lowerbounds must be pointwise <= upperbounds.")

        # Default model.
        if self.model is None:
            kernel = make_default_kernel(D)
            self.model = GaussianProcess(kernel=kernel, log_noise=-2.0, mean=0.0)

        # Default acquisition.
        if self.acquisition is None:
            self.acquisition = ExpectedImprovement()

        # Default model optimizer.
        if self.model_optimizer is None:
            n_kp = self.model.kernel.n_params()
            kern_bounds = np.array([[-3.0] * n_kp, [4.0] * n_kp])
            self.model_optimizer = MAPGPOptimizer(
                every=20,
                noise_bounds=(-4.0, 3.0),
                kern_bounds=kern_bounds,
            )

        # Default initialiser.
        if self.initializer_iterations is None:
            self.initializer_iterations = 5 * D
        if self.initializer is None:
            self.initializer = ScaledSobolIterator(
                self.lowerbounds, self.upperbounds,
                self.initializer_iterations,
            )

        if self.max_iterations < len(self.initializer):
            raise ValueError(
                f"max_iterations ({self.max_iterations}) must be >= "
                f"initializer length ({len(self.initializer)})."
            )

        # Initialise observed optimum.
        self.observed_optimum = -math.inf * int(self.sense)
        self.observed_optimizer = np.zeros(D)

    # ---------------------------------------------------------------
    # Termination
    # ---------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._iteration >= self.max_iterations:
            return True
        if time.time() - self._start_time >= self.max_duration:
            return True
        return False

    # ---------------------------------------------------------------
    # Evaluate objective
    # ---------------------------------------------------------------

    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective and track the optimum."""
        raw = self.func(x)
        y = int(self.sense) * raw  # Internally always maximise.
        if y > int(self.sense) * self.observed_optimum:
            self.observed_optimum = int(self.sense) * y
            self.observed_optimizer = x.copy()
        return y

    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def _initialise(self) -> None:
        """Evaluate the objective on the initialisation points."""
        xs = []
        ys = []
        for x in self.initializer:
            x = np.asarray(x, dtype=float)
            for _ in range(self.repetitions):
                y = self._evaluate(x)
                xs.append(x.copy())
                ys.append(y)
        n_init = len(ys) // self.repetitions
        self._iteration = n_init

        if len(ys) > 0:
            X = np.column_stack(xs)  # (D, N)
            y_arr = np.array(ys)
            t0 = time.time()
            self.model.fit(X, y_arr)
            self._timings["model_update"] = (
                self._timings.get("model_update", 0.0) + time.time() - t0
            )
            t0 = time.time()
            self.model_optimizer.optimize_model(self.model)
            self._timings["hyperparam_opt"] = (
                self._timings.get("hyperparam_opt", 0.0) + time.time() - t0
            )

    # ---------------------------------------------------------------
    # Batch acquisition
    # ---------------------------------------------------------------

    def _acquire_batch(self) -> List[np.ndarray]:
        """
        Select a batch of candidate points.

        For batch_size == 1, standard single-point acquisition.
        For batch_size > 1, uses the Kriging Believer heuristic:
        sequentially pick the next point, fantasise its value as the
        predictive mean, update a local copy of the GP, and repeat.
        """
        candidates = []

        if self.batch_size == 1:
            self.acquisition.set_params(self.model)
            _, x = _optimize_acquisition(
                self.acquisition, self.model,
                self.lowerbounds, self.upperbounds,
                n_restarts=self.acq_restarts,
                method=self.acq_method,
            )
            candidates.append(x)
        else:
            # Kriging Believer for batch selection.
            # Work on a temporary copy of the model data.
            X_orig = self.model.X.copy() if self.model.X is not None else None
            y_orig = self.model.y.copy() if self.model.y is not None else None

            for b in range(self.batch_size):
                self.acquisition.set_params(self.model)
                _, x = _optimize_acquisition(
                    self.acquisition, self.model,
                    self.lowerbounds, self.upperbounds,
                    n_restarts=self.acq_restarts,
                    method=self.acq_method,
                )
                candidates.append(x)

                if b < self.batch_size - 1:
                    # Fantasise: add predicted mean as a pseudo-observation.
                    mu, _ = self.model.predict_single(x)
                    self.model.update(x[:, None], np.array([mu]))

            # Restore model to original data.
            if X_orig is not None:
                self.model.fit(X_orig, y_orig)

        return candidates

    # ---------------------------------------------------------------
    # Main optimisation loop
    # ---------------------------------------------------------------

    def boptimize(self) -> Dict[str, object]:
        """
        Run the Bayesian optimisation loop.

        Returns
        -------
        dict with keys:
            observed_optimum : float
            observed_optimizer : np.ndarray
            model_optimum : float
            model_optimizer : np.ndarray
        """
        self._start_time = time.time()
        self._iteration = 0
        self._timings = {}

        # Initialisation phase.
        if len(self.initializer) > 0:
            self._initialise()

        # Main loop.
        while not self._is_done():
            if self.verbosity >= Verbosity.PROGRESS:
                logger.info(
                    "iteration %d | current optimum: %.6g",
                    self._iteration, self.observed_optimum,
                )

            # Acquire candidate(s).
            t0 = time.time()
            candidates = self._acquire_batch()
            self._timings["acquisition"] = (
                self._timings.get("acquisition", 0.0) + time.time() - t0
            )

            # Evaluate candidate(s).
            all_xs = []
            all_ys = []
            for x in candidates:
                self._iteration += 1
                for _ in range(self.repetitions):
                    y = self._evaluate(x)
                    all_xs.append(x.copy())
                    all_ys.append(y)

            # Update model.
            if len(all_ys) > 0:
                X_new = np.column_stack(all_xs)
                y_new = np.array(all_ys)
                t0 = time.time()
                self.model.update(X_new, y_new)
                self._timings["model_update"] = (
                    self._timings.get("model_update", 0.0) + time.time() - t0
                )
                t0 = time.time()
                self.model_optimizer.optimize_model(self.model)
                self._timings["hyperparam_opt"] = (
                    self._timings.get("hyperparam_opt", 0.0) + time.time() - t0
                )

        # Find model-based optimum via MaxMean.
        t0 = time.time()
        max_mean = MaxMean()
        _, model_opt_x = _optimize_acquisition(
            max_mean, self.model,
            self.lowerbounds, self.upperbounds,
            n_restarts=self.acq_restarts,
        )
        model_opt_mu, _ = self.model.predict_single(model_opt_x)
        self.model_optimum = int(self.sense) * model_opt_mu
        self._model_best_x = model_opt_x
        self._timings["acquisition"] = (
            self._timings.get("acquisition", 0.0) + time.time() - t0
        )

        if self.verbosity >= Verbosity.TIMINGS:
            logger.info("Timing breakdown: %s", self._timings)

        return {
            "observed_optimum": self.observed_optimum,
            "observed_optimizer": self.observed_optimizer,
            "model_optimum": self.model_optimum,
            "model_optimizer": self._model_best_x,
        }

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [
            "BOpt(",
            f"  sense={self.sense.name},",
            f"  model={self.model!r},",
            f"  acquisition={self.acquisition!r},",
            f"  bounds=[{self.lowerbounds}, {self.upperbounds}],",
        ]
        if self._iteration > 0:
            lines += [
                f"  observed_optimum={self.observed_optimum:.6g},",
                f"  observed_optimizer={self.observed_optimizer},",
                f"  model_optimum={self.model_optimum:.6g},",
                f"  iterations={self._iteration}/{self.max_iterations},",
            ]
        else:
            lines.append("  (not yet run)")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrapper — matches Julia's optimize()
# ---------------------------------------------------------------------------

def optimize(
    func: Callable,
    lowerbounds: np.ndarray,
    upperbounds: np.ndarray,
    *,
    sense: Sense = Sense.MAX,
    max_iterations: int = 1000,
    max_duration: float = math.inf,
    acquisition: Optional[AcquisitionFunction] = None,
    model: Optional[GaussianProcess] = None,
    model_optimizer: object = None,
    repetitions: int = 1,
    verbosity: Verbosity = Verbosity.PROGRESS,
    initializer_iterations: Optional[int] = None,
    initializer: object = None,
    batch_size: int = 1,
    acq_restarts: int = 10,
) -> Dict[str, object]:
    """
    Bayesian optimisation of *func* over [lowerbounds, upperbounds].

    This is a convenience wrapper around BOpt that uses sensible
    defaults for the model, acquisition function, and hyperparameter
    optimiser.

    Parameters
    ----------
    func : callable
        Objective f(x) -> float.
    lowerbounds, upperbounds : array-like, shape (D,)
    sense : Sense
        MIN or MAX.
    max_iterations : int
    max_duration : float
        Maximum wall-clock time in seconds.
    acquisition : AcquisitionFunction or None
    model : GaussianProcess or None
    model_optimizer : object or None
    repetitions : int
    verbosity : Verbosity
    initializer_iterations : int or None
    initializer : object or None
    batch_size : int
    acq_restarts : int

    Returns
    -------
    dict
        Keys: observed_optimum, observed_optimizer,
              model_optimum, model_optimizer.
    """
    opt = BOpt(
        func=func,
        lowerbounds=np.asarray(lowerbounds, dtype=float),
        upperbounds=np.asarray(upperbounds, dtype=float),
        model=model,
        acquisition=acquisition,
        model_optimizer=model_optimizer,
        sense=sense,
        max_iterations=max_iterations,
        max_duration=max_duration,
        repetitions=repetitions,
        verbosity=verbosity,
        initializer_iterations=initializer_iterations,
        initializer=initializer,
        batch_size=batch_size,
        acq_restarts=acq_restarts,
    )
    return opt.boptimize()
