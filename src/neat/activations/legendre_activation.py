"""
Legendre Polynomial Activation Function Module.

This module implements a learnable activation function parameterized 
as a linear combination of Legendre polynomials. The function computes:

    f(z) = sum(c_i * P_i(tanh(z))) for i=0 to degree

where P_i are Legendre polynomials and c_i are learnable coefficients.

Classes:
    LegendreActivation: Learnable activation using Legendre polynomial basis
"""

import autograd.numpy as np  # type: ignore

# ================================================
# Legendre Polynomial Definitions (P0 through P10)
# ================================================
# These polynomials are defined on the interval [-1, 1] and form
# an orthogonal basis. They are used as the basis functions for 
# the learnable activation function.

def P0(x):
    """Legendre polynomial P0(x) = 1"""
    return np.ones_like(x)

def P1(x):
    """Legendre polynomial P1(x) = x"""
    return x

def P2(x):
    """Legendre polynomial P2(x) = (1/2)(3x^2 - 1)"""
    return 0.5 * (3 * x**2 - 1)

def P3(x):
    """Legendre polynomial P3(x) = (1/2)(5x^3 - 3x)"""
    return 0.5 * (5 * x**3 - 3 * x)

def P4(x):
    """Legendre polynomial P4(x) = (1/8)(35x^4 - 30x^2 + 3)"""
    return 0.125 * (35 * x**4 - 30 * x**2 + 3)

def P5(x):
    """Legendre polynomial P5(x) = (1/8)(63x^5 - 70x^3 + 15x)"""
    return 0.125 * (63 * x**5 - 70 * x**3 + 15 * x)

def P6(x):
    """Legendre polynomial P6(x) = (1/16)(231x^6 - 315x^4 + 105x^2 - 5)"""
    return (1/16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)

def P7(x):
    """Legendre polynomial P7(x) = (1/16)(429x^7 - 693x^5 + 315x^3 - 35x)"""
    return (1/16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)

def P8(x):
    """Legendre polynomial P8(x) = (1/128)(6435x^8 - 12012x^6 + 6930x^4 - 1260x^2 + 35)"""
    return (1/128) * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)

def P9(x):
    """Legendre polynomial P9(x) = (1/128)(12155x^9 - 25740x^7 + 18018x^5 - 4620x^3 + 315x)"""
    return (1/128) * (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x)

def P10(x):
    """Legendre polynomial P10(x) = (1/256)(46189x^10 - 109395x^8 + 90090x^6 - 30030x^4 + 3465x^2 - 63)"""
    return (1/256) * (46189 * x**10 - 109395 * x**8 + 90090 * x**6 - 30030 * x**4 + 3465 * x**2 - 63)

# ========================
# LegendreActivation Class
# ========================

class LegendreActivation:
    """
    Learnable activation function using Legendre polynomial basis.

    This class implements an activation function as a linear combination
    of Legendre polynomials:

        f(z) = sum(c_i * P_i(tanh(z))) for i=0 to degree

    The 'tanh' mapping ensures inputs are in [-1, 1], the natural domain 
    for Legendre polynomials. This makes the activation well-behaved for 
    all input ranges while maintaining autograd compatibility.

    Public Methods:
        __call__(z, c): Apply activation to input z

    Public Attributes:
        degree: Maximum polynomial degree
    """

    # Available Legendre polynomials
    _POLYNOMIALS = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]
    _MAX_DEGREE  = len(_POLYNOMIALS) - 1

    def __init__(self, degree: int):
        """
        Initialize Legendre activation function.

        Parameters:
            degree: Polynomial degree. The activation will use polynomials
                    P_0 through P_degree, requiring degree+1 coefficients.

        Raises:
            ValueError: If degree > _MAX_DEGREE
        """
        if degree > self._MAX_DEGREE:
            raise ValueError(f"Degree {degree} exceeds maximum {self._MAX_DEGREE}")

        self.degree       = degree
        self._polynomials = self._POLYNOMIALS[:degree + 1]

    def __call__(self, z, c):
        """
        Apply Legendre activation function.

        Computes: f(z) = sum(c_i * P_i(tanh(z))) for i=0 to degree

        The 'tanh' mapping ensures all inputs are mapped to [-1, 1] 
        before polynomial evaluation, making the activation well-behaved 
        for unbounded inputs while preserving autograd compatibility.

        Parameters:
            z: Input values (scalar, array, or multi-dimensional tensor)
            c: Polynomial coefficients array of shape (degree+1,)
                          coefficients[i] multiplies P_i

        Returns:
            Activated values with the same shape as z

        Raises:
            ValueError: If len(coefficients) != degree+1

        Example:
            >>> activation = LegendreActivation(degree=2)
            >>> c = np.array([1.0, 0.5, -0.2])  # c0, c1, c2
            >>> z = np.array([0.0, 1.0, 2.0])
            >>> output = activation(z, c)
        """

        # Note: do not validate here that 'z' and 'c' are nd.arrays because they
        # might not be. For example, during autograd tracing nd.arrays are wrapped 
        # in ArrayBox objects.

        # Validate coefficient dimensions
        if c.size != self.degree + 1:
            raise ValueError(f"Expected {self.degree + 1} coefficients")

        # Map inputs to [-1, 1] using tanh (Legendre polynomial domain)
        # This ensures numerical stability for all input ranges
        x = np.tanh(z)

        # Compute activation: sum(c_i * P_i(x))
        # Start with zeros matching input shape
        result = np.zeros_like(x)

        # Add contribution from each polynomial
        for coeff, poly in zip(c, self._polynomials):
            result = result + coeff * poly(x)

        return result