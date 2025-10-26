"""
Activations Package

This package provides activation functions for NEAT neural networks.

Exported:
    activations: Dictionary mapping activation function names to functions
    Individual activation functions: identity_activation, clamped_activation,
                                     relu_activation, sigmoid_activation, tanh_activation
    LegendreActivation: Learnable activation using Legendre polynomial basis
"""

from neat.activations.basic_activations import (
    activations,
    identity_activation,
    clamped_activation,
    relu_activation,
    sigmoid_activation,
    tanh_activation
)
from neat.activations.legendre_activation import LegendreActivation

__all__ = [
    'activations',
    'identity_activation',
    'clamped_activation',
    'relu_activation',
    'sigmoid_activation',
    'tanh_activation',
    'LegendreActivation'
]
