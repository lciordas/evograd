import autograd.numpy as np

def identity_activation(z):
    return z

def clamped_activation(z):
    return np.maximum(-1.0, np.minimum(1.0, z))

def relu_activation(z):
    return np.maximum(0.0, z)

def sigmoid_activation(z):
    # Use soft clamping via tanh for smoother gradients (autograd-friendly)
    # This avoids hard boundaries that cause gradient issues
    z_scaled = 5.0 * z
    z_clamped = 60.0 * np.tanh(z_scaled / 60.0)
    return 1.0 / (1.0 + np.exp(-z_clamped))

def tanh_activation(z):
    # Use soft clamping for smoother gradients (autograd-friendly)
    # Nested tanh provides smooth transition instead of hard clipping
    z_scaled = 2.5 * z
    z_clamped = 60.0 * np.tanh(z_scaled / 60.0)
    return np.tanh(z_clamped)

activations = {
    "identity": identity_activation,
    "clamped" : clamped_activation,
    "relu"    : relu_activation,
    "sigmoid" : sigmoid_activation,
    "tanh"    : tanh_activation
    }
