"""
Optimization module for hyperparameter tuning in evograd.

This module provides Bayesian optimization capabilities for automatically
tuning NEAT hyperparameters using Optuna.
"""

from evograd.optimization.search_space import SearchSpace
from evograd.optimization.bayesian_optimizer import BayesianOptimizer

__all__ = [
    'SearchSpace',
    'BayesianOptimizer',
]