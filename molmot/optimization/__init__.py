"""
Bayesian optimisation subpackage.

Port of BayesianOptimization.jl by jbrea.
Provides Gaussian process regression, kernel functions, acquisition
functions, and a Bayesian optimiser with batch support and Sobol/LHS
initialisation.
"""

from .kernels import (
    Kernel,
    SEKernel,
    Matern52Kernel,
    ARDMatern52Kernel,
    make_default_kernel,
)
from .gp import (
    GaussianProcess,
    MAPGPOptimizer,
    NoModelOptimizer,
)
from .acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    BrochuBetaScaling,
    NoBetaScaling,
    ThompsonSampling,
    MaxVariance,
    MaxMean,
    MutualInformation,
    make_acquisition,
)
from .bopt import (
    BOpt,
    Sense,
    Verbosity,
    ScaledSobolIterator,
    ScaledLHSIterator,
    optimize,
)

__all__ = [
    # Kernels
    "Kernel",
    "SEKernel",
    "Matern52Kernel",
    "ARDMatern52Kernel",
    "make_default_kernel",
    # Gaussian Process
    "GaussianProcess",
    "MAPGPOptimizer",
    "NoModelOptimizer",
    # Acquisition functions
    "AcquisitionFunction",
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
    "BrochuBetaScaling",
    "NoBetaScaling",
    "ThompsonSampling",
    "MaxVariance",
    "MaxMean",
    "MutualInformation",
    "make_acquisition",
    # Bayesian Optimiser
    "BOpt",
    "Sense",
    "Verbosity",
    "ScaledSobolIterator",
    "ScaledLHSIterator",
    "optimize",
]
