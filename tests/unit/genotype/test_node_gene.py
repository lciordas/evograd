"""
Unit tests for NodeGene class.

Tests cover initialization, mutation, activation functions, string representations,
and edge cases for INPUT, HIDDEN, and OUTPUT nodes.
"""

import pytest
import random
import numpy as np
from unittest.mock import Mock, patch

from evograd.genotype.node_gene import NodeGene, NodeType
from evograd.run.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Config with standard parameters for node genes."""
    config = Mock(spec=Config)
    # Bias parameters
    config.bias_init_mean = 0.0
    config.bias_init_stdev = 1.0
    config.min_bias = -5.0
    config.max_bias = 5.0
    config.bias_perturb_prob = 0.7
    config.bias_perturb_strength = 0.5
    config.bias_replace_prob = 0.1

    # Gain parameters
    config.gain_init_mean = 1.0
    config.gain_init_stdev = 0.5
    config.min_gain = 0.1
    config.max_gain = 2.0
    config.gain_perturb_prob = 0.7
    config.gain_perturb_strength = 0.3
    config.gain_replace_prob = 0.1

    # Activation parameters
    config.activation_initial = 'sigmoid'
    config.activation_mutate_prob = 0.2
    config.activation_options = ['sigmoid', 'tanh', 'relu', 'identity']

    # Legendre parameters
    config.num_legendre_coeffs = 4
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 1.0

    return config


@pytest.fixture
def extreme_config():
    """Config with tight bounds for boundary testing."""
    config = Mock(spec=Config)
    config.bias_init_mean = 0.0
    config.bias_init_stdev = 0.5
    config.min_bias = -1.0
    config.max_bias = 1.0
    config.bias_perturb_prob = 0.5
    config.bias_perturb_strength = 5.0  # Larger than bounds
    config.bias_replace_prob = 0.3

    config.gain_init_mean = 1.0
    config.gain_init_stdev = 0.5
    config.min_gain = 0.5
    config.max_gain = 1.5
    config.gain_perturb_prob = 0.5
    config.gain_perturb_strength = 10.0  # Larger than bounds
    config.gain_replace_prob = 0.3

    config.activation_initial = 'relu'
    config.activation_mutate_prob = 0.8
    config.activation_options = ['relu', 'sigmoid']

    config.num_legendre_coeffs = 3
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 1.0

    return config


@pytest.fixture
def no_mutation_config():
    """Config with zero mutation probabilities."""
    config = Mock(spec=Config)
    config.bias_init_mean = 0.0
    config.bias_init_stdev = 1.0
    config.min_bias = -10.0
    config.max_bias = 10.0
    config.bias_perturb_prob = 0.0
    config.bias_perturb_strength = 0.5
    config.bias_replace_prob = 0.0

    config.gain_init_mean = 1.0
    config.gain_init_stdev = 0.5
    config.min_gain = 0.1
    config.max_gain = 5.0
    config.gain_perturb_prob = 0.0
    config.gain_perturb_strength = 0.3
    config.gain_replace_prob = 0.0

    config.activation_initial = 'tanh'
    config.activation_mutate_prob = 0.0
    config.activation_options = ['tanh', 'relu', 'sigmoid']

    config.num_legendre_coeffs = 4
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 1.0

    return config


@pytest.fixture
def legendre_config():
    """Config specifically for legendre activation testing."""
    config = Mock(spec=Config)
    config.bias_init_mean = 0.0
    config.bias_init_stdev = 1.0
    config.min_bias = -5.0
    config.max_bias = 5.0
    config.bias_perturb_prob = 0.5
    config.bias_perturb_strength = 0.5
    config.bias_replace_prob = 0.1

    config.gain_init_mean = 1.0
    config.gain_init_stdev = 0.5
    config.min_gain = 0.1
    config.max_gain = 2.0
    config.gain_perturb_prob = 0.5
    config.gain_perturb_strength = 0.3
    config.gain_replace_prob = 0.1

    config.activation_initial = 'legendre'
    config.activation_mutate_prob = 0.3
    config.activation_options = ['legendre', 'sigmoid', 'tanh']

    config.num_legendre_coeffs = 5
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 0.5

    return config


# ============================================================================
# Test: Initialization - INPUT Nodes
# ============================================================================

class TestNodeGeneInitInput:
    """Test NodeGene initialization for INPUT nodes."""

    def test_input_node_basic(self, basic_config):
        """Test basic INPUT node initialization."""
        node = NodeGene(0, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        assert node.id == 0
        assert node.type == NodeType.INPUT
        assert node.bias == 0.0
        assert node.gain == 1.0
        assert node.activation is None
        assert node.activation_name is None
        assert node.activation_coeffs is None

    def test_input_node_activation_always_none(self, basic_config):
        """Test that INPUT nodes have None activation even if specified."""
        # Even if we specify activation_name, INPUT nodes should ignore it
        node = NodeGene(0, NodeType.INPUT, basic_config,
                       bias=0.0, gain=1.0, activation_name='sigmoid')

        assert node.activation is None
        assert node.activation_name is None
        assert node.activation_coeffs is None

    def test_input_node_with_different_bias_gain(self, basic_config):
        """Test INPUT node with non-standard bias and gain."""
        node = NodeGene(5, NodeType.INPUT, basic_config, bias=2.5, gain=0.5)

        assert node.id == 5
        assert node.bias == 2.5
        assert node.gain == 0.5
        assert node.type == NodeType.INPUT

    def test_input_node_negative_id(self, basic_config):
        """Test INPUT node with negative ID."""
        node = NodeGene(-1, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        assert node.id == -1
        assert node.type == NodeType.INPUT


# ============================================================================
# Test: Initialization - HIDDEN Nodes
# ============================================================================

class TestNodeGeneInitHidden:
    """Test NodeGene initialization for HIDDEN nodes."""

    def test_hidden_node_with_explicit_params(self, basic_config):
        """Test HIDDEN node with all explicit parameters."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.5, gain=1.5, activation_name='relu')

        assert node.id == 10
        assert node.type == NodeType.HIDDEN
        assert node.bias == 0.5
        assert node.gain == 1.5
        assert node.activation_name == 'relu'
        assert node.activation is not None
        assert node.activation_coeffs is None

    def test_hidden_node_default_initialization(self, basic_config):
        """Test HIDDEN node with default (None) parameters."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, basic_config)

        assert node.id == 10
        assert node.type == NodeType.HIDDEN
        # Bias and gain should be initialized from normal distribution
        assert isinstance(node.bias, float)
        assert isinstance(node.gain, float)
        # Should use config.activation
        assert node.activation_name == 'sigmoid'
        assert node.activation is not None

    def test_hidden_node_bias_clipping(self, extreme_config):
        """Test that initialized bias is clipped to bounds."""
        # Set seed to get predictable values
        random.seed(42)
        np.random.seed(42)

        # Create multiple nodes to test clipping
        for _ in range(10):
            node = NodeGene(10, NodeType.HIDDEN, extreme_config)
            assert extreme_config.min_bias <= node.bias <= extreme_config.max_bias

    def test_hidden_node_gain_clipping(self, extreme_config):
        """Test that initialized gain is clipped to bounds."""
        random.seed(42)
        np.random.seed(42)

        for _ in range(10):
            node = NodeGene(10, NodeType.HIDDEN, extreme_config)
            assert extreme_config.min_gain <= node.gain <= extreme_config.max_gain

    def test_hidden_node_different_activations(self, basic_config):
        """Test HIDDEN node with different activation functions."""
        activations = ['sigmoid', 'tanh', 'relu', 'identity']

        for act_name in activations:
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name=act_name)
            assert node.activation_name == act_name
            assert node.activation is not None

    def test_hidden_node_legendre_with_coeffs(self, legendre_config):
        """Test HIDDEN node with legendre activation and explicit coefficients."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                       bias=0.0, gain=1.0, activation_name='legendre',
                       activation_coeffs=coeffs)

        assert node.activation_name == 'legendre'
        assert node.activation is None  # Legendre stores None
        assert np.array_equal(node.activation_coeffs, coeffs)

    def test_hidden_node_legendre_without_coeffs(self, legendre_config):
        """Test HIDDEN node with legendre activation, coefficients initialized."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                       bias=0.0, gain=1.0, activation_name='legendre')

        assert node.activation_name == 'legendre'
        assert node.activation is None
        assert node.activation_coeffs is not None
        assert len(node.activation_coeffs) == legendre_config.num_legendre_coeffs

    def test_hidden_node_random_activation(self, basic_config):
        """Test HIDDEN node with 'random' activation selection."""
        random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.0, gain=1.0, activation_name='random')

        # Should have selected one of the available activations
        assert node.activation_name is not None
        assert node.activation_name != 'random'

    def test_hidden_node_random_fixed_activation(self, legendre_config):
        """Test HIDDEN node with 'random-fixed' activation (excludes legendre)."""
        random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                       bias=0.0, gain=1.0, activation_name='random-fixed')

        # Should not select legendre
        assert node.activation_name is not None
        assert node.activation_name != 'legendre'
        assert node.activation_name != 'random-fixed'


# ============================================================================
# Test: Initialization - OUTPUT Nodes
# ============================================================================

class TestNodeGeneInitOutput:
    """Test NodeGene initialization for OUTPUT nodes."""

    def test_output_node_with_explicit_params(self, basic_config):
        """Test OUTPUT node with all explicit parameters."""
        node = NodeGene(2, NodeType.OUTPUT, basic_config,
                       bias=0.5, gain=1.5, activation_name='tanh')

        assert node.id == 2
        assert node.type == NodeType.OUTPUT
        assert node.bias == 0.5
        assert node.gain == 1.5
        assert node.activation_name == 'tanh'
        assert node.activation is not None

    def test_output_node_default_initialization(self, basic_config):
        """Test OUTPUT node with default (None) parameters."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(2, NodeType.OUTPUT, basic_config)

        assert node.id == 2
        assert node.type == NodeType.OUTPUT
        assert isinstance(node.bias, float)
        assert isinstance(node.gain, float)
        assert node.activation_name == 'sigmoid'
        assert node.activation is not None

    def test_output_node_legendre(self, legendre_config):
        """Test OUTPUT node with legendre activation."""
        coeffs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        node = NodeGene(2, NodeType.OUTPUT, legendre_config,
                       bias=0.0, gain=1.0, activation_name='legendre',
                       activation_coeffs=coeffs)

        assert node.type == NodeType.OUTPUT
        assert node.activation_name == 'legendre'
        assert node.activation is None
        assert np.array_equal(node.activation_coeffs, coeffs)


# ============================================================================
# Test: Mutation - Bias
# ============================================================================

class TestNodeGeneMutateBias:
    """Test bias mutation behavior."""

    def test_no_bias_mutation_when_probabilities_zero(self, no_mutation_config):
        """Test that bias doesn't change when mutation probabilities are 0."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, no_mutation_config,
                       bias=1.5, gain=1.0, activation_name='tanh')
        original_bias = node.bias

        for _ in range(10):
            node.mutate()

        assert node.bias == original_bias

    def test_bias_perturbation(self, basic_config):
        """Test that bias perturbation changes the value."""
        random.seed(42)
        np.random.seed(42)

        biases = []
        for i in range(50):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name='relu')
            node.mutate()
            biases.append(node.bias)

        # With perturb_prob=0.7, most should have changed
        changed_count = sum(1 for b in biases if b != 0.0)
        assert changed_count > 25  # At least 50%

    def test_bias_clipping_at_min(self, extreme_config):
        """Test that bias perturbation clips at min_bias."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=-1.0, gain=1.0, activation_name='relu')

        for _ in range(50):
            node.mutate()
            assert node.bias >= extreme_config.min_bias

    def test_bias_clipping_at_max(self, extreme_config):
        """Test that bias perturbation clips at max_bias."""
        random.seed(99)
        np.random.seed(99)

        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=1.0, gain=1.0, activation_name='relu')

        for _ in range(50):
            node.mutate()
            assert node.bias <= extreme_config.max_bias

    def test_bias_replacement(self, basic_config):
        """Test that bias replacement produces values within bounds."""
        random.seed(123)
        np.random.seed(123)

        biases = []
        for i in range(100):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=50.0, gain=1.0, activation_name='relu')
            node.mutate()
            biases.append(node.bias)

        # Changed biases should be within bounds
        changed_biases = [b for b in biases if abs(b - 50.0) > 0.01]
        assert len(changed_biases) > 50
        for bias in changed_biases:
            assert basic_config.min_bias <= bias <= basic_config.max_bias


# ============================================================================
# Test: Mutation - Gain
# ============================================================================

class TestNodeGeneMutateGain:
    """Test gain mutation behavior."""

    def test_no_gain_mutation_when_probabilities_zero(self, no_mutation_config):
        """Test that gain doesn't change when mutation probabilities are 0."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, no_mutation_config,
                       bias=0.0, gain=1.5, activation_name='tanh')
        original_gain = node.gain

        for _ in range(10):
            node.mutate()

        assert node.gain == original_gain

    def test_gain_perturbation(self, basic_config):
        """Test that gain perturbation changes the value."""
        random.seed(42)
        np.random.seed(42)

        gains = []
        for i in range(50):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name='relu')
            node.mutate()
            gains.append(node.gain)

        # With perturb_prob=0.7, most should have changed
        changed_count = sum(1 for g in gains if g != 1.0)
        assert changed_count > 25  # At least 50%

    def test_gain_clipping_at_min(self, extreme_config):
        """Test that gain perturbation clips at min_gain."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=0.0, gain=0.5, activation_name='relu')

        for _ in range(50):
            node.mutate()
            assert node.gain >= extreme_config.min_gain

    def test_gain_clipping_at_max(self, extreme_config):
        """Test that gain perturbation clips at max_gain."""
        random.seed(99)
        np.random.seed(99)

        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=0.0, gain=1.5, activation_name='relu')

        for _ in range(50):
            node.mutate()
            assert node.gain <= extreme_config.max_gain

    def test_gain_replacement(self, basic_config):
        """Test that gain replacement produces values within bounds."""
        random.seed(123)
        np.random.seed(123)

        gains = []
        for i in range(100):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=10.0, activation_name='relu')
            node.mutate()
            gains.append(node.gain)

        # Changed gains should be within bounds
        changed_gains = [g for g in gains if abs(g - 10.0) > 0.01]
        assert len(changed_gains) > 50
        for gain in changed_gains:
            assert basic_config.min_gain <= gain <= basic_config.max_gain


# ============================================================================
# Test: Mutation - Activation
# ============================================================================

class TestNodeGeneMutateActivation:
    """Test activation mutation behavior."""

    def test_input_node_activation_never_mutates(self, basic_config):
        """Test that INPUT nodes never mutate their activation."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(0, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        for _ in range(20):
            node.mutate()

        # Should remain None
        assert node.activation is None
        assert node.activation_name is None

    def test_no_activation_mutation_when_probability_zero(self, no_mutation_config):
        """Test that activation doesn't change when mutation probability is 0."""
        random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, no_mutation_config,
                       bias=0.0, gain=1.0, activation_name='tanh')
        original_activation = node.activation_name

        for _ in range(10):
            node.mutate()

        assert node.activation_name == original_activation

    def test_activation_mutation_changes_function(self, extreme_config):
        """Test that activation mutation changes the activation function."""
        random.seed(42)
        np.random.seed(42)

        # extreme_config has high mutation prob (0.8) and only 2 options
        original_activation = 'relu'
        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=0.0, gain=1.0, activation_name=original_activation)

        mutations = 0
        for _ in range(50):
            node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                          bias=0.0, gain=1.0, activation_name=original_activation)
            original = node.activation_name
            node.mutate()
            if node.activation_name != original:
                mutations += 1

        # With prob=0.8, most should mutate
        assert mutations > 30

    def test_activation_mutation_to_legendre(self, legendre_config):
        """Test mutation from fixed activation to legendre."""
        random.seed(999)
        np.random.seed(999)

        found_legendre = False
        for i in range(100):
            node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                          bias=0.0, gain=1.0, activation_name='sigmoid')
            node.mutate()

            if node.activation_name == 'legendre':
                found_legendre = True
                # Should have coefficients and no activation function
                assert node.activation is None
                assert node.activation_coeffs is not None
                assert len(node.activation_coeffs) == legendre_config.num_legendre_coeffs
                break

        assert found_legendre, "Should have mutated to legendre at least once"

    def test_activation_mutation_from_legendre(self, legendre_config):
        """Test mutation from legendre to fixed activation."""
        random.seed(888)
        np.random.seed(888)

        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        found_fixed = False
        for i in range(100):
            node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                          bias=0.0, gain=1.0, activation_name='legendre',
                          activation_coeffs=coeffs.copy())
            node.mutate()

            if node.activation_name != 'legendre':
                found_fixed = True
                # Should have activation function and no coefficients
                assert node.activation is not None
                assert node.activation_coeffs is None
                break

        assert found_fixed, "Should have mutated from legendre at least once"

    def test_activation_mutation_removes_current_option(self):
        """Test that activation mutation always changes to different activation."""
        # Config with only 2 options ensures we can detect the change
        config = Mock(spec=Config)
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.min_bias = -5.0
        config.max_bias = 5.0
        config.bias_perturb_prob = 0.0
        config.bias_perturb_strength = 0.5
        config.bias_replace_prob = 0.0

        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.5
        config.min_gain = 0.1
        config.max_gain = 2.0
        config.gain_perturb_prob = 0.0
        config.gain_perturb_strength = 0.3
        config.gain_replace_prob = 0.0

        config.activation_initial = 'relu'
        config.activation_mutate_prob = 1.0  # Always mutate
        config.activation_options = ['relu', 'sigmoid']

        random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, config,
                       bias=0.0, gain=1.0, activation_name='relu')
        node.mutate()

        # Should have changed to sigmoid
        assert node.activation_name == 'sigmoid'


# ============================================================================
# Test: Mutation - Combined
# ============================================================================

class TestNodeGeneMutateCombined:
    """Test combined mutation behavior."""

    def test_multiple_mutations_in_single_call(self):
        """Test that bias, gain, and activation can all mutate in one call."""
        config = Mock(spec=Config)
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.min_bias = -5.0
        config.max_bias = 5.0
        config.bias_perturb_prob = 1.0  # Always
        config.bias_perturb_strength = 0.5
        config.bias_replace_prob = 0.0

        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.5
        config.min_gain = 0.1
        config.max_gain = 2.0
        config.gain_perturb_prob = 1.0  # Always
        config.gain_perturb_strength = 0.3
        config.gain_replace_prob = 0.0

        config.activation_initial = 'relu'
        config.activation_mutate_prob = 1.0  # Always
        config.activation_options = ['relu', 'sigmoid']

        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, config,
                       bias=0.0, gain=1.0, activation_name='relu')
        original_bias = node.bias
        original_gain = node.gain
        original_activation = node.activation_name

        node.mutate()

        # All should have changed
        assert node.bias != original_bias
        assert node.gain != original_gain
        assert node.activation_name != original_activation

    def test_mutation_does_not_change_id_or_type(self, basic_config):
        """Test that mutation only affects mutable parameters."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.5, gain=1.5, activation_name='relu')

        original_id = node.id
        original_type = node.type

        for _ in range(20):
            node.mutate()

        assert node.id == original_id
        assert node.type == original_type


# ============================================================================
# Test: activation_function Property
# ============================================================================

class TestNodeGeneActivationFunctionProperty:
    """Test the activation_function property."""

    def test_input_node_activation_function_is_none(self, basic_config):
        """Test that INPUT nodes return None for activation_function."""
        node = NodeGene(0, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        assert node.activation_function is None

    def test_fixed_activation_function(self, basic_config):
        """Test that fixed activations return the stored function."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.0, gain=1.0, activation_name='sigmoid')

        func = node.activation_function
        assert func is not None
        assert callable(func)
        # Test that it works
        result = func(0.0)
        assert isinstance(result, (int, float))

    def test_legendre_activation_function_created_dynamically(self, legendre_config):
        """Test that legendre activation creates function dynamically."""
        coeffs = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                       bias=0.0, gain=1.0, activation_name='legendre',
                       activation_coeffs=coeffs)

        func = node.activation_function
        assert func is not None
        assert callable(func)
        # Test that it works
        result = func(0.5)
        assert isinstance(result, (int, float, np.floating))

    def test_different_activation_functions_work(self, basic_config):
        """Test that various activation functions are callable."""
        activations_to_test = ['sigmoid', 'tanh', 'relu', 'identity']

        for act_name in activations_to_test:
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name=act_name)

            func = node.activation_function
            assert func is not None
            assert callable(func)

            # Test with different inputs
            for x in [-1.0, 0.0, 1.0]:
                result = func(x)
                assert isinstance(result, (int, float, np.floating))


# ============================================================================
# Test: String Representations
# ============================================================================

class TestNodeGeneStringMethods:
    """Test __repr__ and __str__ methods."""

    def test_repr_input_node(self, basic_config):
        """Test __repr__ for INPUT node."""
        node = NodeGene(0, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        repr_str = repr(node)

        assert "NodeGene" in repr_str
        assert "node_id=+00" in repr_str or "node_id=+0" in repr_str or "node_id=0" in repr_str
        assert "NodeType.INPUT" in repr_str
        assert "bias=0.0" in repr_str or "bias=0" in repr_str
        assert "gain=1.0" in repr_str or "gain=1" in repr_str

    def test_repr_hidden_node(self, basic_config):
        """Test __repr__ for HIDDEN node."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.5, gain=1.5, activation_name='relu')

        repr_str = repr(node)

        assert "NodeGene" in repr_str
        assert "NodeType.HIDDEN" in repr_str

    def test_repr_output_node(self, basic_config):
        """Test __repr__ for OUTPUT node."""
        node = NodeGene(2, NodeType.OUTPUT, basic_config,
                       bias=-0.5, gain=0.8, activation_name='tanh')

        repr_str = repr(node)

        assert "NodeGene" in repr_str
        assert "NodeType.OUTPUT" in repr_str

    def test_str_input_node(self, basic_config):
        """Test __str__ for INPUT node shows simple format."""
        node = NodeGene(0, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)

        str_repr = str(node)

        assert "[I0]" in str_repr or "[I" in str_repr

    def test_str_hidden_node(self, basic_config):
        """Test __str__ for HIDDEN node shows bias and gain."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.5, gain=1.5, activation_name='relu')

        str_repr = str(node)

        assert "H" in str_repr
        assert "b=" in str_repr
        assert "g=" in str_repr

    def test_str_output_node(self, basic_config):
        """Test __str__ for OUTPUT node shows bias and gain."""
        node = NodeGene(2, NodeType.OUTPUT, basic_config,
                       bias=0.5, gain=1.5, activation_name='tanh')

        str_repr = str(node)

        assert "O" in str_repr
        assert "b=" in str_repr
        assert "g=" in str_repr

    def test_str_negative_bias(self, basic_config):
        """Test __str__ with negative bias."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=-2.5, gain=1.0, activation_name='relu')

        str_repr = str(node)

        assert "-2.5" in str_repr or "-2.50" in str_repr


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestNodeGeneEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_bias_and_gain(self, basic_config):
        """Test node with zero bias and gain."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.0, gain=0.0, activation_name='relu')

        assert node.bias == 0.0
        assert node.gain == 0.0

    def test_negative_bias_and_gain(self, basic_config):
        """Test node with negative bias and gain."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=-5.0, gain=-2.0, activation_name='relu')

        assert node.bias == -5.0
        assert node.gain == -2.0

    def test_very_large_node_id(self, basic_config):
        """Test node with very large ID."""
        node = NodeGene(999999, NodeType.HIDDEN, basic_config,
                       bias=0.0, gain=1.0, activation_name='relu')

        assert node.id == 999999

    def test_config_object_reference(self, basic_config):
        """Test that config is stored as reference, not copied."""
        node = NodeGene(10, NodeType.HIDDEN, basic_config,
                       bias=0.0, gain=1.0, activation_name='relu')

        assert node._config is basic_config

    def test_multiple_nodes_same_config(self, basic_config):
        """Test that multiple nodes can share the same config object."""
        node1 = NodeGene(1, NodeType.INPUT, basic_config, bias=0.0, gain=1.0)
        node2 = NodeGene(2, NodeType.OUTPUT, basic_config,
                        bias=0.5, gain=1.5, activation_name='tanh')

        assert node1._config is node2._config

    def test_mutation_reproducibility_with_seed(self, basic_config):
        """Test that mutations are reproducible with same random seed."""
        biases1 = []
        biases2 = []

        # First run
        random.seed(12345)
        np.random.seed(12345)
        for i in range(10):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name='relu')
            node.mutate()
            biases1.append(node.bias)

        # Second run with same seed
        random.seed(12345)
        np.random.seed(12345)
        for i in range(10):
            node = NodeGene(10, NodeType.HIDDEN, basic_config,
                          bias=0.0, gain=1.0, activation_name='relu')
            node.mutate()
            biases2.append(node.bias)

        assert biases1 == biases2

    def test_bias_outside_bounds_not_clipped_on_init(self, extreme_config):
        """Test that constructor doesn't clip explicit bias values."""
        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=100.0, gain=1.0, activation_name='relu')

        assert node.bias == 100.0  # Not clipped during initialization

    def test_gain_outside_bounds_not_clipped_on_init(self, extreme_config):
        """Test that constructor doesn't clip explicit gain values."""
        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=0.0, gain=100.0, activation_name='relu')

        assert node.gain == 100.0  # Not clipped during initialization

    def test_mutation_with_bias_outside_bounds(self, extreme_config):
        """Test mutation starting from bias outside bounds."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, extreme_config,
                       bias=50.0, gain=1.0, activation_name='relu')

        node.mutate()

        # After mutation with replacement, should be within bounds (if replaced)
        # Or could stay at 50.0 if no mutation occurred
        # We can't assert specific value, but can check no crash

    def test_legendre_coefficients_shape(self, legendre_config):
        """Test that legendre coefficients have correct shape."""
        random.seed(42)
        np.random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, legendre_config,
                       bias=0.0, gain=1.0, activation_name='legendre')

        assert node.activation_coeffs.shape == (legendre_config.num_legendre_coeffs,)

    def test_activation_mutation_with_single_option(self):
        """Test activation mutation when only one option available."""
        config = Mock(spec=Config)
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.min_bias = -5.0
        config.max_bias = 5.0
        config.bias_perturb_prob = 0.0
        config.bias_perturb_strength = 0.5
        config.bias_replace_prob = 0.0

        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.5
        config.min_gain = 0.1
        config.max_gain = 2.0
        config.gain_perturb_prob = 0.0
        config.gain_perturb_strength = 0.3
        config.gain_replace_prob = 0.0

        config.activation_initial = 'relu'
        config.activation_mutate_prob = 1.0
        config.activation_options = ['relu']  # Only one option

        random.seed(42)

        node = NodeGene(10, NodeType.HIDDEN, config,
                       bias=0.0, gain=1.0, activation_name='relu')
        original_activation = node.activation_name

        node.mutate()

        # Should not crash, but no change possible
        # After removing current activation, list is empty
        # random.choice on empty list would error, but let's see what happens
        # This might be a bug in the original code
