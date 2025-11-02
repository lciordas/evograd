"""
Unit tests for LegendreActivation class.

Tests cover initialization, __call__ method, from_coeffs class method,
polynomial correctness, numerical stability, and edge cases.
"""

import pytest
import numpy as np
from evograd.activations.legendre_activation import (
    LegendreActivation,
    P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10,
)


# Fixtures
@pytest.fixture
def degree_0_activation():
    """LegendreActivation with degree 0 (only P0)."""
    return LegendreActivation(degree=0)


@pytest.fixture
def degree_1_activation():
    """LegendreActivation with degree 1 (P0, P1)."""
    return LegendreActivation(degree=1)


@pytest.fixture
def degree_5_activation():
    """LegendreActivation with degree 5 (P0-P5)."""
    return LegendreActivation(degree=5)


@pytest.fixture
def degree_10_activation():
    """LegendreActivation with degree 10 (P0-P10)."""
    return LegendreActivation(degree=10)


@pytest.fixture
def sample_coeffs_degree_2():
    """Sample coefficients for degree 2."""
    return np.array([1.0, 0.5, -0.2])


@pytest.fixture
def sample_coeffs_degree_5():
    """Sample coefficients for degree 5."""
    return np.array([1.0, 0.5, -0.2, 0.1, -0.05, 0.02])


class TestLegendreActivationInit:
    """Test LegendreActivation initialization."""

    def test_degree_0(self):
        """Test initialization with degree 0."""
        activation = LegendreActivation(degree=0)
        assert activation.degree == 0
        assert len(activation._polynomials) == 1

    def test_degree_1(self):
        """Test initialization with degree 1."""
        activation = LegendreActivation(degree=1)
        assert activation.degree == 1
        assert len(activation._polynomials) == 2

    def test_degree_5(self):
        """Test initialization with degree 5."""
        activation = LegendreActivation(degree=5)
        assert activation.degree == 5
        assert len(activation._polynomials) == 6

    def test_degree_10(self):
        """Test initialization with degree 10 (maximum)."""
        activation = LegendreActivation(degree=10)
        assert activation.degree == 10
        assert len(activation._polynomials) == 11

    def test_degree_greater_than_max_raises_error(self):
        """Test that degree > 10 raises ValueError."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            LegendreActivation(degree=11)

    def test_degree_much_greater_than_max_raises_error(self):
        """Test that degree >> 10 raises ValueError."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            LegendreActivation(degree=100)

    def test_negative_degree_does_not_raise_on_init(self):
        """Test that negative degree doesn't raise during init (but would fail on call)."""
        # Negative degree doesn't raise ValueError in __init__ because
        # the check is only for degree > _MAX_DEGREE
        activation = LegendreActivation(degree=-1)
        assert activation.degree == -1

    def test_max_degree_class_attribute(self):
        """Test that _MAX_DEGREE is set correctly."""
        assert LegendreActivation._MAX_DEGREE == 10

    def test_polynomials_class_attribute(self):
        """Test that _POLYNOMIALS has 11 entries."""
        assert len(LegendreActivation._POLYNOMIALS) == 11


class TestLegendreActivationCall:
    """Test LegendreActivation __call__ method."""

    def test_scalar_input_degree_0(self, degree_0_activation):
        """Test scalar input with degree 0."""
        z = 1.0
        c = np.array([2.0])  # Only c0
        result = degree_0_activation(z, c)

        # f(z) = c0 * P0(tanh(z)) = 2.0 * 1.0 = 2.0
        expected = 2.0
        assert abs(result - expected) < 1e-9

    def test_scalar_input_degree_1(self, degree_1_activation):
        """Test scalar input with degree 1."""
        z = 0.0
        c = np.array([1.0, 2.0])  # c0, c1
        result = degree_1_activation(z, c)

        # f(0) = c0 * P0(tanh(0)) + c1 * P1(tanh(0))
        #      = 1.0 * 1 + 2.0 * 0 = 1.0
        expected = 1.0
        assert abs(result - expected) < 1e-9

    def test_scalar_input_degree_5(self, degree_5_activation, sample_coeffs_degree_5):
        """Test scalar input with degree 5."""
        z = 0.0
        result = degree_5_activation(z, sample_coeffs_degree_5)

        # At z=0, tanh(0)=0, so only even polynomials contribute
        # P0(0)=1, P1(0)=0, P2(0)=-0.5, P3(0)=0, P4(0)=0.375, P5(0)=0
        # f(0) = 1.0*1 + 0.5*0 + (-0.2)*(-0.5) + 0.1*0 + (-0.05)*0.375 + 0.02*0
        #      = 1.0 + 0.1 - 0.01875 = 1.08125
        x = np.tanh(z)
        expected = (sample_coeffs_degree_5[0] * P0(x) +
                    sample_coeffs_degree_5[1] * P1(x) +
                    sample_coeffs_degree_5[2] * P2(x) +
                    sample_coeffs_degree_5[3] * P3(x) +
                    sample_coeffs_degree_5[4] * P4(x) +
                    sample_coeffs_degree_5[5] * P5(x))
        assert abs(result - expected) < 1e-9

    def test_1d_array_input(self, degree_1_activation):
        """Test 1D array input."""
        z = np.array([-1.0, 0.0, 1.0])
        c = np.array([1.0, 1.0])  # c0=1, c1=1
        result = degree_1_activation(z, c)

        # Compute expected values
        x = np.tanh(z)
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert result.shape == z.shape
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array_input(self, degree_1_activation):
        """Test 2D array input."""
        z = np.array([[0.0, 1.0], [2.0, 3.0]])
        c = np.array([1.0, 0.5])
        result = degree_1_activation(z, c)

        # Compute expected values
        x = np.tanh(z)
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert result.shape == z.shape
        np.testing.assert_array_almost_equal(result, expected)

    def test_3d_array_input(self, degree_1_activation):
        """Test 3D array input."""
        z = np.random.randn(2, 3, 4)
        c = np.array([1.0, 0.5])
        result = degree_1_activation(z, c)

        # Compute expected values
        x = np.tanh(z)
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert result.shape == z.shape
        np.testing.assert_array_almost_equal(result, expected)

    def test_wrong_coefficient_count_raises_error(self, degree_5_activation):
        """Test that wrong number of coefficients raises ValueError."""
        z = 0.0
        c = np.array([1.0, 0.5])  # Only 2 coeffs, but need 6 for degree 5

        with pytest.raises(ValueError, match="Expected 6 coefficients"):
            degree_5_activation(z, c)

    def test_too_many_coefficients_raises_error(self, degree_1_activation):
        """Test that too many coefficients raises ValueError."""
        z = 0.0
        c = np.array([1.0, 0.5, 0.3])  # 3 coeffs, but need 2 for degree 1

        with pytest.raises(ValueError, match="Expected 2 coefficients"):
            degree_1_activation(z, c)

    def test_large_positive_input(self, degree_1_activation):
        """Test that very large inputs are bounded by tanh."""
        z = 1000.0
        c = np.array([1.0, 1.0])
        result = degree_1_activation(z, c)

        # tanh(1000) ≈ 1, so f(1000) ≈ c0*1 + c1*1 = 2.0
        x = np.tanh(z)  # Should be very close to 1
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert abs(result - expected) < 1e-9
        assert abs(x - 1.0) < 1e-9  # Verify tanh bounds it

    def test_large_negative_input(self, degree_1_activation):
        """Test that very large negative inputs are bounded by tanh."""
        z = -1000.0
        c = np.array([1.0, 1.0])
        result = degree_1_activation(z, c)

        # tanh(-1000) ≈ -1, so f(-1000) ≈ c0*1 + c1*(-1) = 0.0
        x = np.tanh(z)  # Should be very close to -1
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert abs(result - expected) < 1e-9
        assert abs(x - (-1.0)) < 1e-9  # Verify tanh bounds it

    def test_zero_coefficients(self, degree_5_activation):
        """Test activation with all zero coefficients."""
        z = np.array([0.0, 1.0, 2.0])
        c = np.zeros(6)
        result = degree_5_activation(z, c)

        # All coefficients are zero, so result should be all zeros
        expected = np.zeros_like(z)
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_nonzero_coefficient(self, degree_5_activation):
        """Test activation with single non-zero coefficient."""
        z = np.array([0.0, 1.0])
        c = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Only c2 is non-zero
        result = degree_5_activation(z, c)

        # f(z) = 1.0 * P2(tanh(z))
        x = np.tanh(z)
        expected = P2(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_coefficients(self, degree_1_activation):
        """Test activation with negative coefficients."""
        z = 1.0
        c = np.array([-1.0, -2.0])
        result = degree_1_activation(z, c)

        x = np.tanh(z)
        expected = c[0] * P0(x) + c[1] * P1(x)

        assert abs(result - expected) < 1e-9

    def test_mixed_sign_coefficients(self, degree_5_activation):
        """Test activation with mixed positive/negative coefficients."""
        z = 0.5
        c = np.array([1.0, -0.5, 0.3, -0.2, 0.1, -0.05])
        result = degree_5_activation(z, c)

        x = np.tanh(z)
        expected = sum(c[i] * LegendreActivation._POLYNOMIALS[i](x) for i in range(6))

        assert abs(result - expected) < 1e-9


class TestLegendreActivationFromCoeffs:
    """Test LegendreActivation.from_coeffs class method."""

    def test_creates_callable(self):
        """Test that from_coeffs returns a callable."""
        coeffs = np.array([1.0, 0.5])
        activation_fn = LegendreActivation.from_coeffs(coeffs)
        assert callable(activation_fn)

    def test_callable_accepts_only_z(self):
        """Test that returned function only requires z parameter."""
        coeffs = np.array([1.0, 0.5])
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = 0.0
        result = activation_fn(z)  # Should work with just z

        # Verify it computed correctly
        x = np.tanh(z)
        expected = coeffs[0] * P0(x) + coeffs[1] * P1(x)
        assert abs(result - expected) < 1e-9

    def test_coefficients_captured_in_closure(self):
        """Test that coefficients are properly captured in closure."""
        coeffs = np.array([2.0, 3.0])
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = 1.0
        result = activation_fn(z)

        # Manually compute what it should be
        x = np.tanh(z)
        expected = coeffs[0] * P0(x) + coeffs[1] * P1(x)

        assert abs(result - expected) < 1e-9

    def test_array_input_works(self):
        """Test that from_coeffs works with array inputs."""
        coeffs = np.array([1.0, 0.5])
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = np.array([0.0, 1.0, 2.0])
        result = activation_fn(z)

        x = np.tanh(z)
        expected = coeffs[0] * P0(x) + coeffs[1] * P1(x)

        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_0_from_coeffs(self):
        """Test from_coeffs with degree 0."""
        coeffs = np.array([5.0])
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = 2.0
        result = activation_fn(z)

        # f(z) = 5.0 * P0(tanh(z)) = 5.0 * 1 = 5.0
        expected = 5.0
        assert abs(result - expected) < 1e-9

    def test_degree_10_from_coeffs(self):
        """Test from_coeffs with degree 10 (maximum)."""
        coeffs = np.ones(11)  # 11 coefficients for degree 10
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = 0.0
        result = activation_fn(z)

        # Compute manually
        x = np.tanh(z)
        expected = sum(LegendreActivation._POLYNOMIALS[i](x) for i in range(11))

        assert abs(result - expected) < 1e-9

    def test_invalid_degree_raises_error(self):
        """Test that degree > 10 raises ValueError."""
        coeffs = np.ones(12)  # 12 coefficients = degree 11

        with pytest.raises(ValueError, match="exceeds maximum"):
            LegendreActivation.from_coeffs(coeffs)

    def test_list_coefficients_converted_to_array(self):
        """Test that list coefficients are converted to array."""
        coeffs = [1.0, 0.5]  # List, not array
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        z = 0.0
        result = activation_fn(z)

        # Should work the same as array
        x = np.tanh(z)
        expected = 1.0 * P0(x) + 0.5 * P1(x)
        assert abs(result - expected) < 1e-9

    def test_from_coeffs_vs_call_equivalence(self):
        """Test that from_coeffs produces same results as direct call."""
        coeffs = np.array([1.0, -0.5, 0.3])

        # Method 1: from_coeffs
        activation_fn = LegendreActivation.from_coeffs(coeffs)

        # Method 2: direct instantiation and call
        activation = LegendreActivation(degree=2)

        z = np.array([0.0, 1.0, 2.0, -1.0])

        result1 = activation_fn(z)
        result2 = activation(z, coeffs)

        np.testing.assert_array_almost_equal(result1, result2)


class TestLegendrePolynomials:
    """Test individual Legendre polynomial functions."""

    def test_P0_always_one(self):
        """Test that P0(x) = 1 for all x."""
        x_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = P0(x_values)
        expected = np.ones_like(x_values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_P1_equals_x(self):
        """Test that P1(x) = x."""
        x_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = P1(x_values)
        np.testing.assert_array_almost_equal(result, x_values)

    def test_P2_at_standard_points(self):
        """Test P2(x) = (1/2)(3x^2 - 1) at standard points."""
        # P2(-1) = 0.5(3*1 - 1) = 1
        # P2(0) = 0.5(0 - 1) = -0.5
        # P2(1) = 0.5(3*1 - 1) = 1
        assert abs(P2(-1.0) - 1.0) < 1e-9
        assert abs(P2(0.0) - (-0.5)) < 1e-9
        assert abs(P2(1.0) - 1.0) < 1e-9

    def test_P3_at_standard_points(self):
        """Test P3(x) = (1/2)(5x^3 - 3x) at standard points."""
        # P3(-1) = 0.5(-5 + 3) = -1
        # P3(0) = 0
        # P3(1) = 0.5(5 - 3) = 1
        assert abs(P3(-1.0) - (-1.0)) < 1e-9
        assert abs(P3(0.0) - 0.0) < 1e-9
        assert abs(P3(1.0) - 1.0) < 1e-9

    def test_P4_at_zero(self):
        """Test P4(0)."""
        # P4(0) = (1/8)(0 - 0 + 3) = 3/8
        assert abs(P4(0.0) - 0.375) < 1e-9

    def test_P5_at_zero(self):
        """Test P5(0) = 0 (odd function)."""
        assert abs(P5(0.0)) < 1e-9

    def test_P6_at_zero(self):
        """Test P6(0)."""
        # P6(0) = (1/16)(0 - 0 + 0 - 5) = -5/16
        assert abs(P6(0.0) - (-5/16)) < 1e-9

    def test_P7_at_zero(self):
        """Test P7(0) = 0 (odd function)."""
        assert abs(P7(0.0)) < 1e-9

    def test_P8_at_zero(self):
        """Test P8(0)."""
        # P8(0) = (1/128)(0 - 0 + 0 - 0 + 35) = 35/128
        assert abs(P8(0.0) - (35/128)) < 1e-9

    def test_P9_at_zero(self):
        """Test P9(0) = 0 (odd function)."""
        assert abs(P9(0.0)) < 1e-9

    def test_P10_at_zero(self):
        """Test P10(0)."""
        # P10(0) = (1/256)(0 - 0 + 0 - 0 + 0 - 63) = -63/256
        assert abs(P10(0.0) - (-63/256)) < 1e-9

    def test_odd_polynomials_antisymmetric(self):
        """Test that odd polynomials satisfy P(x) = -P(-x)."""
        x = 0.5
        # P1, P3, P5, P7, P9 are odd
        assert abs(P1(x) + P1(-x)) < 1e-9
        assert abs(P3(x) + P3(-x)) < 1e-9
        assert abs(P5(x) + P5(-x)) < 1e-9
        assert abs(P7(x) + P7(-x)) < 1e-9
        assert abs(P9(x) + P9(-x)) < 1e-9

    def test_even_polynomials_symmetric(self):
        """Test that even polynomials satisfy P(x) = P(-x)."""
        x = 0.5
        # P0, P2, P4, P6, P8, P10 are even
        assert abs(P0(x) - P0(-x)) < 1e-9
        assert abs(P2(x) - P2(-x)) < 1e-9
        assert abs(P4(x) - P4(-x)) < 1e-9
        assert abs(P6(x) - P6(-x)) < 1e-9
        assert abs(P8(x) - P8(-x)) < 1e-9
        assert abs(P10(x) - P10(-x)) < 1e-9

    def test_all_polynomials_at_one(self):
        """Test that all Legendre polynomials equal 1 at x=1."""
        for i, poly in enumerate([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]):
            result = poly(1.0)
            assert abs(result - 1.0) < 1e-9, f"P{i}(1) should be 1"

    def test_all_polynomials_at_negative_one(self):
        """Test that P_n(-1) = (-1)^n."""
        expected_values = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
        for i, (poly, expected) in enumerate(zip([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10], expected_values)):
            result = poly(-1.0)
            assert abs(result - expected) < 1e-9, f"P{i}(-1) should be {expected}"

    def test_polynomials_accept_arrays(self):
        """Test that all polynomials accept array inputs."""
        x = np.array([-1.0, 0.0, 1.0])
        for poly in [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]:
            result = poly(x)
            assert result.shape == x.shape


class TestLegendreNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_large_input_bounded(self, degree_1_activation):
        """Test that very large inputs don't cause overflow."""
        z = 1e10
        c = np.array([1.0, 1.0])
        result = degree_1_activation(z, c)

        # Should not be inf or nan
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_very_small_input_bounded(self, degree_1_activation):
        """Test that very small negative inputs don't cause overflow."""
        z = -1e10
        c = np.array([1.0, 1.0])
        result = degree_1_activation(z, c)

        # Should not be inf or nan
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_all_zero_coefficients_stable(self, degree_10_activation):
        """Test that all zero coefficients produce zero output."""
        z = np.linspace(-10, 10, 100)
        c = np.zeros(11)
        result = degree_10_activation(z, c)

        expected = np.zeros_like(z)
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_coefficients_stable(self, degree_5_activation):
        """Test activation with very large coefficients."""
        z = np.array([0.0, 1.0])
        c = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
        result = degree_5_activation(z, c)

        # Should not overflow to inf
        assert not np.any(np.isnan(result))
        # May be very large but shouldn't be inf for bounded inputs after tanh
        # (actually might be inf due to large coefficients, but shouldn't be nan)

    def test_very_small_coefficients(self, degree_5_activation):
        """Test activation with very small coefficients."""
        z = np.array([0.0, 1.0, 2.0])
        c = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
        result = degree_5_activation(z, c)

        # Result should be very small but not exactly zero
        assert np.all(np.abs(result) < 1e-8)

    def test_mixed_extreme_coefficients(self, degree_1_activation):
        """Test with extreme positive and negative coefficients."""
        z = 0.5
        c = np.array([1e6, -1e6])
        result = degree_1_activation(z, c)

        # Should produce finite result
        assert not np.isnan(result)


class TestLegendreIntegration:
    """Test integration scenarios and consistency."""

    def test_call_and_from_coeffs_consistency(self):
        """Test that __call__ and from_coeffs produce identical results."""
        degree = 3
        coeffs = np.array([1.0, -0.5, 0.3, -0.1])
        z = np.linspace(-5, 5, 50)

        # Method 1: Direct call
        activation = LegendreActivation(degree)
        result1 = activation(z, coeffs)

        # Method 2: from_coeffs
        activation_fn = LegendreActivation.from_coeffs(coeffs)
        result2 = activation_fn(z)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_multiple_calls_same_activation(self, degree_5_activation):
        """Test that multiple calls to same activation are consistent."""
        z = np.array([1.0, 2.0, 3.0])
        c = np.array([1.0, 0.5, -0.2, 0.1, -0.05, 0.02])

        result1 = degree_5_activation(z, c)
        result2 = degree_5_activation(z, c)
        result3 = degree_5_activation(z, c)

        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def test_different_degrees_different_results(self):
        """Test that different degrees produce different results (generally)."""
        z = 1.0
        c1 = np.array([1.0, 0.5])
        c2 = np.array([1.0, 0.5, 0.3])

        activation1 = LegendreActivation(degree=1)
        activation2 = LegendreActivation(degree=2)

        result1 = activation1(z, c1)
        result2 = activation2(z, c2)

        # Should be different (c2[2] adds contribution from P2)
        assert abs(result1 - result2) > 1e-6

    def test_zero_input_with_various_degrees(self):
        """Test that zero input produces consistent results across degrees."""
        z = 0.0

        # At z=0, tanh(0)=0, so odd polynomials contribute 0
        # P0(0)=1, P2(0)=-0.5, P4(0)=0.375, P6(0)=-5/16, P8(0)=35/128, P10(0)=-63/256

        for degree in range(11):
            activation = LegendreActivation(degree)
            c = np.ones(degree + 1)
            result = activation(z, c)

            # Manually compute expected
            x = 0.0  # tanh(0) = 0
            expected = sum(LegendreActivation._POLYNOMIALS[i](x) for i in range(degree + 1))

            assert abs(result - expected) < 1e-9
