"""
Activations Package

This package provides activation functions for NEAT neural networks.

Exported:
    activations: Dictionary mapping activation function names to functions
    Individual activation functions: identity_activation, clamped_activation,
                                     relu_activation, sigmoid_activation, tanh_activation
"""

from activations.basic_activations import (
    activations,
    identity_activation,
    clamped_activation,
    relu_activation,
    sigmoid_activation,
    tanh_activation
)

__all__ = [
    'activations',
    'identity_activation',
    'clamped_activation',
    'relu_activation',
    'sigmoid_activation',
    'tanh_activation'
]
