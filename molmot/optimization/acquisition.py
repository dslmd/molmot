"""
Acquisition functions for Bayesian optimisation.

Port of the acquisition functions from BayesianOptimization.jl.
Provides Expected Improvement, Probability of Improvement, Upper
Confidence Bound, Thompson Sampling, and Max Variance strategies.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.special import erf
from scipy.stats import norm

from .gp import GaussianProcess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_cdf(mu: float, var: float) -> float:
    """Standard normal CDF: P(X <= mu) where X ~ N(0, var)."""
    if var <= 0.0:
        return 1.0 if mu >= 0.0 else 0.0
    return 0.5 * (1.0 + erf(mu / math.sqrt(2.0 * var)))


def _normal_pdf(mu: float, var: float) -> float:
    """Standard normal PDF evaluated at mu with variance var."""
    if var <= 0.0:
        return 0.0
    return (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-mu ** 2 / (2.0 * var))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""

    def set_params(self, model: GaussianProcess) -> None:
        """
        Update internal parameters from the current model state.

        Called before each acquisition optimisation step.
        """

    @abstractmethod
    def evaluate(self, mu: float, var: float) -> float:
        """
        Evaluate the acquisition function given predictive mean and variance.

        Parameters
        ----------
        mu : float
            Predictive mean at the candidate point.
        var : float
            Predictive variance at the candidate point.

        Returns
        -------
        float
            Acquisition value (higher is better).
        """

    def __call__(self, mu: float, var: float) -> float:
        return self.evaluate(mu, var)

    def evaluate_at(self, model: GaussianProcess, x: np.ndarray) -> float:
        """
        Evaluate the acquisition function at point *x* using *model*.

        Parameters
        ----------
        model : GaussianProcess
        x : np.ndarray, shape (D,)

        Returns
        -------
        float
        """
        mu, var = model.predict_single(x)
        return self.evaluate(mu, var)


# ---------------------------------------------------------------------------
# Expected Improvement
# ---------------------------------------------------------------------------

@dataclass
class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement acquisition function.

    EI(x) = (mu(x) - tau) * Phi(z) + sigma(x) * phi(z)

    where z = (mu(x) - tau) / sigma(x), Phi is the standard normal CDF,
    phi is the standard normal PDF, and tau is the best observed value.

    Attributes
    ----------
    tau : float
        Incumbent target (best observed value).  Updated automatically
        via *set_params*.
    """
    tau: float = -math.inf

    def set_params(self, model: GaussianProcess) -> None:
        self.tau = max(model.maxy(), self.tau)

    def evaluate(self, mu: float, var: float) -> float:
        if var <= 0.0:
            return max(mu - self.tau, 0.0)
        sigma = math.sqrt(var)
        z = (mu - self.tau) / sigma
        return (mu - self.tau) * norm.cdf(z) + sigma * norm.pdf(z)


# ---------------------------------------------------------------------------
# Probability of Improvement
# ---------------------------------------------------------------------------

@dataclass
class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition function.

    PI(x) = Phi[(mu(x) - tau) / sigma(x)]

    Attributes
    ----------
    tau : float
        Incumbent target.
    """
    tau: float = -math.inf

    def set_params(self, model: GaussianProcess) -> None:
        self.tau = max(model.maxy(), self.tau)

    def evaluate(self, mu: float, var: float) -> float:
        if var <= 0.0:
            return float(mu > self.tau)
        return _normal_cdf(mu - self.tau, var)


# ---------------------------------------------------------------------------
# Upper Confidence Bound
# ---------------------------------------------------------------------------

class BrochuBetaScaling:
    """
    Scale beta_t for UCB as in Brochu et al. (2010).

    beta_t = sqrt(2 * log(t^(D/2 + 2) * pi^2 / (3 * delta)))

    Parameters
    ----------
    delta : float
        Confidence parameter (default 0.1).
    """
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def __call__(self, n_obs: int, input_dim: int) -> float:
        if n_obs <= 0:
            n_obs = 1
        D = input_dim
        arg = n_obs ** (D / 2.0 + 2.0) * math.pi ** 2 / (3.0 * self.delta)
        if arg <= 0:
            return 1.0
        return math.sqrt(2.0 * math.log(arg))


class NoBetaScaling:
    """No scaling: return fixed beta_t."""
    def __call__(self, n_obs: int, input_dim: int) -> float:
        return 1.0  # beta_t is kept as-is in UpperConfidenceBound


@dataclass
class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.

    UCB(x) = mu(x) + beta_t * sigma(x)

    Attributes
    ----------
    beta_t : float
        Exploration-exploitation trade-off parameter.
    scaling : callable
        Scaling strategy for beta_t.  One of BrochuBetaScaling or
        NoBetaScaling.
    """
    beta_t: float = 1.0
    scaling: object = None

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = BrochuBetaScaling(0.1)

    def set_params(self, model: GaussianProcess) -> None:
        D, n_obs = model.dims()
        self.beta_t = self.scaling(n_obs, D)

    def evaluate(self, mu: float, var: float) -> float:
        return mu + self.beta_t * math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# Thompson Sampling (Simple)
# ---------------------------------------------------------------------------

class ThompsonSampling(AcquisitionFunction):
    """
    Thompson Sampling acquisition function (simple variant).

    Draws an independent sample from the predictive posterior at each
    candidate point.  Should be used with a gradient-free optimiser
    for the acquisition step.
    """

    def evaluate(self, mu: float, var: float) -> float:
        sigma = math.sqrt(max(var, 0.0))
        return float(np.random.normal(mu, sigma))

    def evaluate_at(self, model: GaussianProcess, x: np.ndarray) -> float:
        return model.sample(x)


# ---------------------------------------------------------------------------
# Max Variance (pure exploration)
# ---------------------------------------------------------------------------

class MaxVariance(AcquisitionFunction):
    """
    Maximum Variance acquisition function (pure exploration).

    Selects the point with highest predictive uncertainty.
    """

    def evaluate(self, mu: float, var: float) -> float:
        return var


# ---------------------------------------------------------------------------
# Max Mean (exploitation only, used for final model-based optimum)
# ---------------------------------------------------------------------------

class MaxMean(AcquisitionFunction):
    """
    Maximum (posterior) Mean acquisition function.

    Used internally to find the model-based optimum after optimisation.
    """

    def evaluate(self, mu: float, var: float) -> float:
        return mu


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

@dataclass
class MutualInformation(AcquisitionFunction):
    """
    Mutual Information acquisition function.

    MI(x) = mu(x) + sqrt(alpha) * (sqrt(var(x) + gamma_hat) - sqrt(gamma_hat))

    See Contal, Perchet, Vayatis (2014).

    Attributes
    ----------
    sqrt_alpha : float
    gamma_hat : float
        Running estimate of cumulative predictive variance.
    """
    sqrt_alpha: float = 1.0
    gamma_hat: float = 0.0

    def set_params(self, model: GaussianProcess) -> None:
        if model.n_obs == 0:
            self.gamma_hat = 0.0
        else:
            # Add the predictive variance of the last observation.
            last_x = model.X[:, -1]
            _, var = model.predict_single(last_x)
            self.gamma_hat += var

    def evaluate(self, mu: float, var: float) -> float:
        return (mu + self.sqrt_alpha
                * (math.sqrt(var + self.gamma_hat) - math.sqrt(self.gamma_hat)))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_acquisition(name: str, **kwargs) -> AcquisitionFunction:
    """
    Create an acquisition function by name.

    Parameters
    ----------
    name : str
        One of 'ei', 'pi', 'ucb', 'thompson', 'max_variance', 'mi'.

    Returns
    -------
    AcquisitionFunction
    """
    name = name.lower().replace(" ", "_")
    registry = {
        "ei": ExpectedImprovement,
        "expected_improvement": ExpectedImprovement,
        "pi": ProbabilityOfImprovement,
        "probability_of_improvement": ProbabilityOfImprovement,
        "ucb": UpperConfidenceBound,
        "upper_confidence_bound": UpperConfidenceBound,
        "thompson": ThompsonSampling,
        "thompson_sampling": ThompsonSampling,
        "max_variance": MaxVariance,
        "mi": MutualInformation,
        "mutual_information": MutualInformation,
    }
    if name not in registry:
        raise ValueError(f"Unknown acquisition function: {name!r}. "
                         f"Choose from {list(registry.keys())}.")
    return registry[name](**kwargs)
