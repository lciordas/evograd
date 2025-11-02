import autograd.numpy as np  # type: ignore

def identity_activation(z):
    return z

def clamped_activation(z):
    return np.clip(z, -1.0, 1.0)

def relu_activation(z):
    return np.maximum(0.0, z)

def sigmoid_activation(z):
    K = 10
    Z = K * z
    Z = np.clip(Z, -100, 100)   # to prevent under/overflow when calculating exp
    return 1.0 / (1.0 + np.exp(-Z))

def tanh_activation(z):
    return np.tanh(z)

def sin_activation(z):
    return np.sin(z)

def square_activation(z):
    # Clip input to avoid overflow (±1e154 squared stays within float64 range)
    z_clipped = np.clip(z, -1e154, 1e154)
    return z_clipped ** 2

def cubed_activation(z):
    # Clip input to avoid overflow (±1e102 cubed stays within float64 range)
    z_clipped = np.clip(z, -1e102, 1e102)
    return z_clipped ** 3

def log_activation(z):
    # Clip input to avoid log of non-positive values
    # Returns log(1e-7) ≈ -16.1 for z <= 0
    z_safe = np.maximum(z, 1e-7)
    return np.log(z_safe)

def inverse_activation(z):
    # Avoid division by zero or very small values
    # Returns 1e7 for z=0, and caps magnitude at 1e7 for |z| < 1e-7
    z_safe = np.where(z == 0, 1e-7, z)
    z_safe = np.where(np.abs(z_safe) < 1e-7, np.sign(z_safe) * 1e-7, z_safe)
    return 1.0 / z_safe

def exponential_activation(z):
    # Clip input to avoid overflow (exp(100) ≈ 2.7e43)
    # and underflow (exp(-100) ≈ 3.7e-44)
    z_clipped = np.clip(z, -100, 100)
    return np.exp(z_clipped)

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

# 3-letter identifiers for each activation function
activation_codes = {
    "identity"   : "IDN",
    "clamped"    : "CLP",
    "relu"       : "RLU",
    "sigmoid"    : "SIG",
    "tanh"       : "TNH",
    "sin"        : "SIN",
    "square"     : "SQR",
    "cubed"      : "CUB",
    "log"        : "LOG",
    "inverse"    : "INV",
    "exponential": "EXP",
    "abs"        : "ABS",
    "legendre"   : "LEG"
    }
