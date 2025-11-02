"""
Unit tests for basic activation functions.

Tests all 12 activation functions in src/neat/activations/basic_activations.py
"""

import pytest
import numpy as np
from evograd.activations.basic_activations import (
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
    abs_activation,
    activations,
)


# Fixtures
@pytest.fixture
def sample_1d_array():
    """Standard 1D array for testing."""
    return np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.fixture
def sample_2d_array():
    """Standard 2D array for testing."""
    return np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])


class TestActivationsDictionary:
    """Test that all activations are accessible via the activations dictionary."""

    def test_all_twelve_functions_in_dictionary(self):
        """Test that all 12 activation functions are in the dictionary."""
        expected_names = [
            'identity', 'clamped', 'relu', 'sigmoid', 'tanh', 'sin',
            'square', 'cubed', 'log', 'inverse', 'exponential', 'abs'
        ]
        for name in expected_names:
            assert name in activations, f"{name} not found in activations dictionary"

    def test_dictionary_has_exactly_twelve_entries(self):
        """Test that dictionary has exactly 12 entries."""
        assert len(activations) == 12

    def test_dictionary_functions_callable(self):
        """Test that all functions in dictionary are callable."""
        for name, func in activations.items():
            assert callable(func), f"{name} is not callable"

    def test_identity_via_dictionary(self):
        """Test identity function via dictionary access."""
        assert activations['identity'](5.0) == 5.0

    def test_relu_via_dictionary(self):
        """Test relu function via dictionary access."""
        assert activations['relu'](-1.0) == 0.0
        assert activations['relu'](1.0) == 1.0


class TestIdentityActivation:
    """Test identity_activation function."""

    def test_scalar_zero(self):
        """Test identity with zero."""
        assert identity_activation(0.0) == 0.0

    def test_scalar_positive(self):
        """Test identity with positive value."""
        assert identity_activation(5.0) == 5.0

    def test_scalar_negative(self):
        """Test identity with negative value."""
        assert identity_activation(-5.0) == -5.0

    def test_scalar_large_positive(self):
        """Test identity with large positive value."""
        assert identity_activation(1000.0) == 1000.0

    def test_scalar_large_negative(self):
        """Test identity with large negative value."""
        assert identity_activation(-1000.0) == -1000.0

    def test_1d_array(self, sample_1d_array):
        """Test identity with 1D array."""
        result = identity_activation(sample_1d_array)
        np.testing.assert_array_equal(result, sample_1d_array)

    def test_2d_array(self, sample_2d_array):
        """Test identity with 2D array."""
        result = identity_activation(sample_2d_array)
        np.testing.assert_array_equal(result, sample_2d_array)


class TestClampedActivation:
    """Test clamped_activation function."""

    def test_value_within_range(self):
        """Test that values within [-1, 1] are unchanged."""
        assert clamped_activation(0.0) == 0.0
        assert clamped_activation(0.5) == 0.5
        assert clamped_activation(-0.5) == -0.5

    def test_value_at_upper_bound(self):
        """Test value at upper bound."""
        assert clamped_activation(1.0) == 1.0

    def test_value_at_lower_bound(self):
        """Test value at lower bound."""
        assert clamped_activation(-1.0) == -1.0

    def test_value_above_upper_bound(self):
        """Test that values > 1 are clamped to 1."""
        assert clamped_activation(2.0) == 1.0
        assert clamped_activation(100.0) == 1.0

    def test_value_below_lower_bound(self):
        """Test that values < -1 are clamped to -1."""
        assert clamped_activation(-2.0) == -1.0
        assert clamped_activation(-100.0) == -1.0

    def test_1d_array(self):
        """Test clamped with 1D array."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        result = clamped_activation(arr)
        np.testing.assert_array_equal(result, expected)

    def test_2d_array(self):
        """Test clamped with 2D array."""
        arr = np.array([[-2.0, 0.0, 2.0], [0.5, -1.5, 1.5]])
        expected = np.array([[-1.0, 0.0, 1.0], [0.5, -1.0, 1.0]])
        result = clamped_activation(arr)
        np.testing.assert_array_equal(result, expected)


class TestReluActivation:
    """Test relu_activation function."""

    def test_zero(self):
        """Test relu with zero."""
        assert relu_activation(0.0) == 0.0

    def test_positive_value(self):
        """Test that positive values pass through."""
        assert relu_activation(1.0) == 1.0
        assert relu_activation(5.0) == 5.0
        assert relu_activation(100.0) == 100.0

    def test_negative_value(self):
        """Test that negative values become 0."""
        assert relu_activation(-1.0) == 0.0
        assert relu_activation(-5.0) == 0.0
        assert relu_activation(-100.0) == 0.0

    def test_1d_array(self, sample_1d_array):
        """Test relu with 1D array."""
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        result = relu_activation(sample_1d_array)
        np.testing.assert_array_equal(result, expected)

    def test_2d_array(self, sample_2d_array):
        """Test relu with 2D array."""
        expected = np.array([[0.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
        result = relu_activation(sample_2d_array)
        np.testing.assert_array_equal(result, expected)


class TestSigmoidActivation:
    """Test sigmoid_activation function."""

    def test_zero(self):
        """Test sigmoid at zero (should be 0.5)."""
        result = sigmoid_activation(0.0)
        assert abs(result - 0.5) < 1e-9

    def test_large_positive(self):
        """Test sigmoid saturates to 1 for large positive values."""
        result = sigmoid_activation(100.0)
        assert result > 0.9999

    def test_large_negative(self):
        """Test sigmoid saturates to 0 for large negative values."""
        result = sigmoid_activation(-100.0)
        assert result < 0.0001

    def test_output_range(self):
        """Test that sigmoid output is in (0, 1)."""
        for z in [-10, -1, 0, 1, 10]:
            result = sigmoid_activation(z)
            assert 0 < result < 1

    def test_1d_array(self, sample_1d_array):
        """Test sigmoid with 1D array."""
        result = sigmoid_activation(sample_1d_array)
        assert result.shape == sample_1d_array.shape
        assert np.all((result > 0) & (result < 1))

    def test_2d_array(self, sample_2d_array):
        """Test sigmoid with 2D array."""
        result = sigmoid_activation(sample_2d_array)
        assert result.shape == sample_2d_array.shape
        assert np.all((result > 0) & (result < 1))

    def test_known_value(self):
        """Test sigmoid at a known value."""
        # sigmoid(1) ≈ 0.7310585786
        result = sigmoid_activation(1.0)
        assert abs(result - 0.7310585786) < 1e-9


class TestTanhActivation:
    """Test tanh_activation function."""

    def test_zero(self):
        """Test tanh at zero (should be 0)."""
        result = tanh_activation(0.0)
        assert abs(result) < 1e-9

    def test_large_positive(self):
        """Test tanh saturates to 1 for large positive values."""
        result = tanh_activation(100.0)
        assert abs(result - 1.0) < 1e-9

    def test_large_negative(self):
        """Test tanh saturates to -1 for large negative values."""
        result = tanh_activation(-100.0)
        assert abs(result - (-1.0)) < 1e-9

    def test_output_range(self):
        """Test that tanh output is in (-1, 1)."""
        for z in [-10, -1, 0, 1, 10]:
            result = tanh_activation(z)
            assert -1 <= result <= 1

    def test_1d_array(self, sample_1d_array):
        """Test tanh with 1D array."""
        result = tanh_activation(sample_1d_array)
        assert result.shape == sample_1d_array.shape
        assert np.all((result >= -1) & (result <= 1))

    def test_2d_array(self, sample_2d_array):
        """Test tanh with 2D array."""
        result = tanh_activation(sample_2d_array)
        assert result.shape == sample_2d_array.shape
        assert np.all((result >= -1) & (result <= 1))

    def test_known_value(self):
        """Test tanh at a known value."""
        # tanh(1) ≈ 0.7615941559
        result = tanh_activation(1.0)
        assert abs(result - 0.7615941559) < 1e-9


class TestSinActivation:
    """Test sin_activation function."""

    def test_zero(self):
        """Test sin at zero (should be 0)."""
        result = sin_activation(0.0)
        assert abs(result) < 1e-9

    def test_pi_over_2(self):
        """Test sin at π/2 (should be 1)."""
        result = sin_activation(np.pi / 2)
        assert abs(result - 1.0) < 1e-9

    def test_pi(self):
        """Test sin at π (should be ~0)."""
        result = sin_activation(np.pi)
        assert abs(result) < 1e-9

    def test_3pi_over_2(self):
        """Test sin at 3π/2 (should be -1)."""
        result = sin_activation(3 * np.pi / 2)
        assert abs(result - (-1.0)) < 1e-9

    def test_output_range(self):
        """Test that sin output is in [-1, 1]."""
        test_values = np.linspace(-10, 10, 100)
        result = sin_activation(test_values)
        assert np.all((result >= -1) & (result <= 1))

    def test_1d_array(self, sample_1d_array):
        """Test sin with 1D array."""
        result = sin_activation(sample_1d_array)
        assert result.shape == sample_1d_array.shape

    def test_2d_array(self, sample_2d_array):
        """Test sin with 2D array."""
        result = sin_activation(sample_2d_array)
        assert result.shape == sample_2d_array.shape


class TestSquareActivation:
    """Test square_activation function."""

    def test_zero(self):
        """Test square of zero."""
        assert square_activation(0.0) == 0.0

    def test_positive_value(self):
        """Test square of positive values."""
        assert square_activation(2.0) == 4.0
        assert square_activation(3.0) == 9.0

    def test_negative_value(self):
        """Test square of negative values (should be positive)."""
        assert square_activation(-2.0) == 4.0
        assert square_activation(-3.0) == 9.0

    def test_one(self):
        """Test square of 1 and -1."""
        assert square_activation(1.0) == 1.0
        assert square_activation(-1.0) == 1.0

    def test_always_non_negative(self):
        """Test that square output is always non-negative."""
        test_values = np.linspace(-10, 10, 100)
        result = square_activation(test_values)
        assert np.all(result >= 0)

    def test_1d_array(self, sample_1d_array):
        """Test square with 1D array."""
        expected = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
        result = square_activation(sample_1d_array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array(self, sample_2d_array):
        """Test square with 2D array."""
        expected = np.array([[1.0, 0.0, 1.0], [4.0, 9.0, 16.0]])
        result = square_activation(sample_2d_array)
        np.testing.assert_array_almost_equal(result, expected)


class TestCubedActivation:
    """Test cubed_activation function."""

    def test_zero(self):
        """Test cube of zero."""
        assert cubed_activation(0.0) == 0.0

    def test_positive_value(self):
        """Test cube of positive values."""
        assert cubed_activation(2.0) == 8.0
        assert cubed_activation(3.0) == 27.0

    def test_negative_value(self):
        """Test cube of negative values (should preserve sign)."""
        assert cubed_activation(-2.0) == -8.0
        assert cubed_activation(-3.0) == -27.0

    def test_one(self):
        """Test cube of 1 and -1."""
        assert cubed_activation(1.0) == 1.0
        assert cubed_activation(-1.0) == -1.0

    def test_preserves_sign(self):
        """Test that cube preserves sign."""
        assert cubed_activation(5.0) > 0
        assert cubed_activation(-5.0) < 0

    def test_1d_array(self, sample_1d_array):
        """Test cube with 1D array."""
        expected = np.array([-8.0, -1.0, 0.0, 1.0, 8.0])
        result = cubed_activation(sample_1d_array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array(self, sample_2d_array):
        """Test cube with 2D array."""
        expected = np.array([[-1.0, 0.0, 1.0], [8.0, 27.0, 64.0]])
        result = cubed_activation(sample_2d_array)
        np.testing.assert_array_almost_equal(result, expected)


class TestLogActivation:
    """Test log_activation function."""

    def test_one(self):
        """Test log(1) = 0."""
        result = log_activation(1.0)
        assert abs(result) < 1e-9

    def test_e(self):
        """Test log(e) = 1."""
        result = log_activation(np.e)
        assert abs(result - 1.0) < 1e-9

    def test_positive_value(self):
        """Test log with positive values."""
        result = log_activation(2.0)
        assert abs(result - np.log(2.0)) < 1e-9

    def test_zero_returns_negative_inf(self):
        """Test that log(0) returns -inf."""
        result = log_activation(0.0)
        assert result == float('-inf')

    def test_negative_returns_nan(self):
        """Test that log(negative) returns NaN."""
        result = log_activation(-1.0)
        assert np.isnan(result)

    def test_1d_array_positive(self):
        """Test log with 1D array of positive values."""
        arr = np.array([1.0, 2.0, np.e, 10.0])
        result = log_activation(arr)
        expected = np.log(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_array_with_zero(self):
        """Test log with array containing zero."""
        arr = np.array([1.0, 0.0, 2.0])
        result = log_activation(arr)
        assert result[1] == float('-inf')

    def test_1d_array_with_negative(self):
        """Test log with array containing negative value."""
        arr = np.array([1.0, -1.0, 2.0])
        result = log_activation(arr)
        assert np.isnan(result[1])

    def test_2d_array(self):
        """Test log with 2D array."""
        arr = np.array([[1.0, 2.0], [np.e, 10.0]])
        result = log_activation(arr)
        expected = np.log(arr)
        np.testing.assert_array_almost_equal(result, expected)


class TestInverseActivation:
    """Test inverse_activation function."""

    def test_one(self):
        """Test inverse(1) = 1."""
        assert inverse_activation(1.0) == 1.0

    def test_negative_one(self):
        """Test inverse(-1) = -1."""
        assert inverse_activation(-1.0) == -1.0

    def test_positive_value(self):
        """Test inverse with positive values."""
        assert inverse_activation(2.0) == 0.5
        assert inverse_activation(4.0) == 0.25

    def test_negative_value(self):
        """Test inverse with negative values."""
        assert inverse_activation(-2.0) == -0.5
        assert inverse_activation(-4.0) == -0.25

    def test_zero_returns_inf(self):
        """Test that inverse(0) returns inf."""
        result = inverse_activation(0.0)
        assert result == float('inf')

    def test_1d_array_nonzero(self):
        """Test inverse with 1D array of non-zero values."""
        arr = np.array([1.0, 2.0, 4.0, -2.0])
        expected = np.array([1.0, 0.5, 0.25, -0.5])
        result = inverse_activation(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_array_with_zero(self):
        """Test inverse with array containing zero."""
        arr = np.array([1.0, 0.0, 2.0])
        result = inverse_activation(arr)
        # Middle element should be inf
        assert result[1] == float('inf')
        assert result[0] == 1.0
        assert result[2] == 0.5

    def test_2d_array(self):
        """Test inverse with 2D array."""
        arr = np.array([[1.0, 2.0], [4.0, -2.0]])
        expected = np.array([[1.0, 0.5], [0.25, -0.5]])
        result = inverse_activation(arr)
        np.testing.assert_array_almost_equal(result, expected)


class TestExponentialActivation:
    """Test exponential_activation function."""

    def test_zero(self):
        """Test exp(0) = 1."""
        result = exponential_activation(0.0)
        assert abs(result - 1.0) < 1e-9

    def test_one(self):
        """Test exp(1) = e."""
        result = exponential_activation(1.0)
        assert abs(result - np.e) < 1e-9

    def test_negative_value(self):
        """Test exp with negative values."""
        result = exponential_activation(-1.0)
        assert abs(result - (1.0 / np.e)) < 1e-9

    def test_positive_value(self):
        """Test exp with positive values."""
        result = exponential_activation(2.0)
        assert abs(result - np.exp(2.0)) < 1e-9

    def test_large_positive_overflow(self):
        """Test that exp with very large values can overflow to inf."""
        result = exponential_activation(1000.0)
        assert np.isinf(result)

    def test_large_negative(self):
        """Test exp with very large negative values approaches 0."""
        result = exponential_activation(-100.0)
        assert result < 1e-40

    def test_1d_array(self, sample_1d_array):
        """Test exp with 1D array."""
        result = exponential_activation(sample_1d_array)
        expected = np.exp(sample_1d_array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array(self, sample_2d_array):
        """Test exp with 2D array."""
        result = exponential_activation(sample_2d_array)
        expected = np.exp(sample_2d_array)
        np.testing.assert_array_almost_equal(result, expected)


class TestAbsActivation:
    """Test abs_activation function."""

    def test_zero(self):
        """Test abs(0) = 0."""
        assert abs_activation(0.0) == 0.0

    def test_positive_value(self):
        """Test abs of positive values (unchanged)."""
        assert abs_activation(1.0) == 1.0
        assert abs_activation(5.0) == 5.0

    def test_negative_value(self):
        """Test abs of negative values (becomes positive)."""
        assert abs_activation(-1.0) == 1.0
        assert abs_activation(-5.0) == 5.0

    def test_always_non_negative(self):
        """Test that abs output is always non-negative."""
        test_values = np.linspace(-10, 10, 100)
        result = abs_activation(test_values)
        assert np.all(result >= 0)

    def test_symmetry(self):
        """Test that abs(x) = abs(-x)."""
        assert abs_activation(5.0) == abs_activation(-5.0)

    def test_1d_array(self, sample_1d_array):
        """Test abs with 1D array."""
        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        result = abs_activation(sample_1d_array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array(self, sample_2d_array):
        """Test abs with 2D array."""
        expected = np.array([[1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
        result = abs_activation(sample_2d_array)
        np.testing.assert_array_almost_equal(result, expected)
