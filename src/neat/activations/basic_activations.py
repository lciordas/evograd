import autograd.numpy as np  # type: ignore

def identity_activation(z):
    return z

def clamped_activation(z):
    return np.maximum(-1.0, np.minimum(1.0, z))

def relu_activation(z):
    return np.maximum(0.0, z)

def sigmoid_activation(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh_activation(z):
    return np.tanh(z)

def sin_activation(z):
    return np.sin(z)

def square_activation(z):
    return z ** 2

def cubed_activation(z):
    return z ** 3

def log_activation(z):
    # Returns '-inf' if z=0.
    return np.log(z)

def inverse_activation(z):
    try:
        return 1/z
    except ZeroDivisionError:
        return np.inf

def exponential_activation(z):
    return np.exp(z)

def abs_activation(z):
    return np.abs(z)

activations = {
    "identity"   : identity_activation,
    "clamped"    : clamped_activation,
    "relu"       : relu_activation,
    "sigmoid"    : sigmoid_activation,
    "tanh"       : tanh_activation,
    "sin"        : sin_activation,
    "square"     : square_activation,
    "cubed"      : cubed_activation,
    "log"        : log_activation,
    "inverse"    : inverse_activation,
    "exponential": exponential_activation,
    "abs"        : abs_activation
    }
