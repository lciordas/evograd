"""
Activations Package

This package provides activation functions for NEAT neural networks.

Exported:
    activations: Dictionary mapping activation function names to functions
    Individual activation functions: identity_activation, clamped_activation,
                                     relu_activation, sigmoid_activation, tanh_activation,
                                     sin_activation, square_activation, cubed_activation,
                                     log_activation, inverse_activation, exponential_activation,
                                     abs_activation
    LegendreActivation: Learnable activation using Legendre polynomial basis
"""

from evograd.activations.basic_activations import (
    activations,
    activation_codes,
    identity_activation,
    clamped_activation,
    relu_activation,
    sigmoid_activation,
    tanh_activation,
    sin_activation,
    square_activation,
    cubed_activation,
    log_activation,
    inverse_activation,
    exponential_activation,
    abs_activation
)
from evograd.activations.legendre_activation import LegendreActivation

__all__ = [
    'activations',
    'activation_codes',
    'identity_activation',
    'clamped_activation',
    'relu_activation',
    'sigmoid_activation',
    'tanh_activation',
    'sin_activation',
    'square_activation',
    'cubed_activation',
    'log_activation',
    'inverse_activation',
    'exponential_activation',
    'abs_activation',
    'LegendreActivation'
]
