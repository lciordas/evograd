"""
Unit tests for NetworkFast class.

Tests cover initialization, forward_pass with batch processing, buffer management,
numerical edge cases, and comparison with NetworkStandard.
"""

import pytest
import numpy as np
from evograd.phenotype.network_fast import NetworkFast
from evograd.phenotype.network_standard import NetworkStandard
from evograd.genotype import Genome, InnovationTracker
from evograd.run.config import Config
from unittest.mock import Mock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_innovation_tracker():
    """Automatically reset InnovationTracker before each test."""
    config = Mock(spec=Config)
    config.num_inputs = 2
    config.num_outputs = 1
    InnovationTracker.initialize(config)
    yield
    InnovationTracker.initialize(config)


@pytest.fixture
def minimal_genome_dict():
    """Minimal genome: only inputs and outputs, no connections."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output'},
        ],
        'connections': [],
        'activation': 'relu'
    }


@pytest.fixture
def linear_genome_dict():
    """Linear chain: input → hidden → output."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output'},
            {'id': 2, 'type': 'hidden'},
        ],
        'connections': [
            {'from': 0, 'to': 2, 'weight': 2.0},
            {'from': 2, 'to': 1, 'weight': 0.5},
        ],
        'activation': 'relu'
    }


@pytest.fixture
def diamond_genome_dict():
    """Diamond: input → {hidden1, hidden2} → output."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output'},
            {'id': 2, 'type': 'hidden'},
            {'id': 3, 'type': 'hidden'},
        ],
        'connections': [
            {'from': 0, 'to': 2, 'weight': 1.0},
            {'from': 0, 'to': 3, 'weight': 1.0},
            {'from': 2, 'to': 1, 'weight': 0.5},
            {'from': 3, 'to': 1, 'weight': 0.5},
        ],
        'activation': 'relu'
    }


@pytest.fixture
def no_hidden_genome_dict():
    """Direct connections: input → output."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output'},
        ],
        'connections': [
            {'from': 0, 'to': 1, 'weight': 1.0},
        ],
        'activation': 'relu'
    }


@pytest.fixture
def all_disabled_genome_dict():
    """All connections disabled."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output'},
        ],
        'connections': [
            {'from': 0, 'to': 1, 'weight': 1.0, 'enabled': False},
        ],
        'activation': 'relu'
    }


# ============================================================================
# Test Classes
# ============================================================================

class TestNetworkFastInit:
    """Test NetworkFast initialization."""

    def test_init_minimal_network(self, minimal_genome_dict):
        """Test initialization with minimal genome."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = NetworkFast(genome)

        assert network._num_nodes == 3
        assert network.weights.shape == (3, 3)
        assert len(network.biases) == 3
        assert len(network.gains) == 3

    def test_init_weight_matrix_shape(self, linear_genome_dict):
        """Test that weight matrix has correct shape."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        num_nodes = len(genome.node_genes)
        assert network.weights.shape == (num_nodes, num_nodes)

    def test_init_weight_matrix_values(self, no_hidden_genome_dict):
        """Test that weight matrix has correct values."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        # Should have weight=1.0 from node 0 to node 1
        idx_0 = network._node_id_to_idx[0]
        idx_1 = network._node_id_to_idx[1]
        assert network.weights[idx_0, idx_1] == 1.0

    def test_init_disabled_connections_ignored(self, all_disabled_genome_dict):
        """Test that disabled connections have zero weight in matrix."""
        genome = Genome.from_dict(all_disabled_genome_dict)
        network = NetworkFast(genome)

        # All weights should be zero since connection is disabled
        assert np.all(network.weights == 0.0)

    def test_init_bias_vector(self, linear_genome_dict):
        """Test that bias vector is created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network.biases, np.ndarray)
        assert len(network.biases) == 3

    def test_init_gain_vector(self, linear_genome_dict):
        """Test that gain vector is created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network.gains, np.ndarray)
        assert len(network.gains) == 3

    def test_init_node_id_to_idx_mapping(self, linear_genome_dict):
        """Test that node ID to index mapping is created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network._node_id_to_idx, dict)
        assert len(network._node_id_to_idx) == 3
        assert all(isinstance(k, int) and isinstance(v, int)
                   for k, v in network._node_id_to_idx.items())

    def test_init_activation_functions_list(self, linear_genome_dict):
        """Test that activation functions list is created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network._activations, list)
        assert len(network._activations) == 3
        assert all(callable(f) or f is None for f in network._activations)

    def test_init_input_indices(self, linear_genome_dict):
        """Test that input indices are created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network._input_indices, np.ndarray)
        assert len(network._input_indices) == 1  # One input

    def test_init_output_indices(self, linear_genome_dict):
        """Test that output indices are created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network._output_indices, np.ndarray)
        assert len(network._output_indices) == 1  # One output

    def test_init_incoming_connections(self, linear_genome_dict):
        """Test that incoming connections optimization is created."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert isinstance(network._incoming_sources, list)
        assert isinstance(network._incoming_weights, list)
        assert len(network._incoming_sources) == 3
        assert len(network._incoming_weights) == 3


class TestNetworkFastForwardPassBasic:
    """Test basic forward_pass functionality."""

    def test_forward_pass_simple_passthrough(self, no_hidden_genome_dict):
        """Test forward pass with simple input → output."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1.0]])
        result = network.forward_pass(inputs)

        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_forward_pass_linear_chain(self, linear_genome_dict):
        """Test forward pass with linear chain."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        # Input=2.0 → Hidden: relu(2.0*2.0)=4.0 → Output: relu(0.5*4.0)=2.0
        inputs = np.array([[2.0]])
        result = network.forward_pass(inputs)

        assert result.shape == (1, 1)
        assert result[0, 0] == 2.0

    def test_forward_pass_diamond_topology(self, diamond_genome_dict):
        """Test forward pass with diamond topology."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = NetworkFast(genome)

        # Input=1.0 → both hiddens get 1.0*1.0=1.0 → Output gets 0.5*1.0+0.5*1.0=1.0
        inputs = np.array([[1.0]])
        result = network.forward_pass(inputs)

        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_forward_pass_comparison_with_networkstandard(self, linear_genome_dict):
        """Test that NetworkFast matches NetworkStandard output."""
        genome = Genome.from_dict(linear_genome_dict)
        network_fast = NetworkFast(genome)
        network_std = NetworkStandard(genome)

        inputs_fast = np.array([[3.0]])
        inputs_std = (3.0,)

        result_fast = network_fast.forward_pass(inputs_fast)
        result_std = network_std.forward_pass(inputs_std)

        assert np.allclose(result_fast[0], result_std)

    def test_forward_pass_no_connections(self, minimal_genome_dict):
        """Test forward pass with no connections (bias only)."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1.0, 2.0]])
        result = network.forward_pass(inputs)

        # Output should be relu(0*sum + 0) = 0
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_forward_pass_disabled_connections(self, all_disabled_genome_dict):
        """Test that disabled connections don't affect output."""
        genome = Genome.from_dict(all_disabled_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[5.0]])
        result = network.forward_pass(inputs)

        # Disabled connection, so output = relu(0) = 0
        assert result[0, 0] == 0.0

    def test_forward_pass_multiple_calls_consistent(self, linear_genome_dict):
        """Test that multiple forward passes give consistent results."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[3.0]])
        result1 = network.forward_pass(inputs)
        result2 = network.forward_pass(inputs)
        result3 = network.forward_pass(inputs)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)


class TestNetworkFastForwardPassInputHandling:
    """Test input validation and batch processing."""

    def test_forward_pass_1d_input_auto_reshape(self, no_hidden_genome_dict):
        """Test that 1D input is auto-reshaped to 2D."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs_1d = np.array([1.0])
        result = network.forward_pass(inputs_1d)

        assert result.shape == (1, 1)

    def test_forward_pass_2d_batched_input(self, no_hidden_genome_dict):
        """Test 2D batched input."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs_2d = np.array([[1.0], [2.0], [3.0]])
        result = network.forward_pass(inputs_2d)

        assert result.shape == (3, 1)

    def test_forward_pass_large_batch(self, no_hidden_genome_dict):
        """Test large batch processing."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        batch_size = 100
        inputs = np.random.randn(batch_size, 1)
        result = network.forward_pass(inputs)

        assert result.shape == (batch_size, 1)

    def test_forward_pass_wrong_input_count_raises(self, linear_genome_dict):
        """Test that wrong input count raises ValueError."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        # Network expects 1 input, give 2
        inputs = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Expected 1 inputs, got 2"):
            network.forward_pass(inputs)

    def test_forward_pass_wrong_dimensions_raises(self, no_hidden_genome_dict):
        """Test that wrong dimensions raise ValueError."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        # 3D input
        inputs_3d = np.array([[[1.0]]])

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            network.forward_pass(inputs_3d)

    def test_forward_pass_list_input_converts(self, no_hidden_genome_dict):
        """Test that list input is converted to array."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs_list = [1.0]
        result = network.forward_pass(inputs_list)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)


class TestNetworkFastForwardPassBufferManagement:
    """Test buffer allocation and reuse."""

    def test_buffer_allocated_on_first_call(self, no_hidden_genome_dict):
        """Test that buffer is allocated on first forward_pass."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        assert network._node_values is None

        inputs = np.array([[1.0]])
        network.forward_pass(inputs)

        assert network._node_values is not None

    def test_buffer_reused_same_batch_size(self, no_hidden_genome_dict):
        """Test that buffer is reused when batch size stays same."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1.0]])
        network.forward_pass(inputs)
        buffer_id_1 = id(network._node_values)

        network.forward_pass(inputs)
        buffer_id_2 = id(network._node_values)

        assert buffer_id_1 == buffer_id_2  # Same buffer object

    def test_buffer_reallocated_different_batch_size(self, no_hidden_genome_dict):
        """Test that buffer is reallocated when batch size changes."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs_batch1 = np.array([[1.0]])
        network.forward_pass(inputs_batch1)
        assert network._last_batch_size == 1

        inputs_batch10 = np.array([[1.0]] * 10)
        network.forward_pass(inputs_batch10)
        assert network._last_batch_size == 10

    def test_buffer_batch_size_change_sequence(self, no_hidden_genome_dict):
        """Test sequence of different batch sizes."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        for batch_size in [1, 5, 10, 1, 100]:
            inputs = np.random.randn(batch_size, 1)
            result = network.forward_pass(inputs)
            assert result.shape == (batch_size, 1)


class TestNetworkFastForwardPassNumerical:
    """Test numerical edge cases."""

    def test_forward_pass_zero_inputs(self, no_hidden_genome_dict):
        """Test forward pass with zero inputs."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[0.0]])
        result = network.forward_pass(inputs)

        assert result[0, 0] == 0.0

    def test_forward_pass_negative_inputs(self, no_hidden_genome_dict):
        """Test forward pass with negative inputs (relu clips)."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[-5.0]])
        result = network.forward_pass(inputs)

        # relu(-5) = 0
        assert result[0, 0] == 0.0

    def test_forward_pass_large_inputs(self, no_hidden_genome_dict):
        """Test forward pass with large inputs."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1e10]])
        result = network.forward_pass(inputs)

        assert not np.isnan(result[0, 0])
        assert not np.isinf(result[0, 0])

    def test_forward_pass_small_inputs(self, no_hidden_genome_dict):
        """Test forward pass with small inputs."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1e-10]])
        result = network.forward_pass(inputs)

        assert not np.isnan(result[0, 0])

    def test_forward_pass_mixed_values(self, diamond_genome_dict):
        """Test forward pass with mixed positive/negative values."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[-1.0], [0.0], [1.0], [100.0]])
        result = network.forward_pass(inputs)

        assert result.shape == (4, 1)
        assert not np.any(np.isnan(result))


class TestNetworkFastForwardPassConnections:
    """Test connection pattern optimizations."""

    def test_forward_pass_no_incoming_connections(self, minimal_genome_dict):
        """Test node with no incoming connections."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1.0, 2.0]])
        result = network.forward_pass(inputs)

        # Output has no connections, so weighted_sum=0, relu(0)=0
        assert result[0, 0] == 0.0

    def test_forward_pass_single_incoming_connection(self, no_hidden_genome_dict):
        """Test node with single incoming connection (optimized path)."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[2.0]])
        result = network.forward_pass(inputs)

        # Single connection with weight=1.0
        assert result[0, 0] == 2.0

    def test_forward_pass_multiple_incoming_connections(self, diamond_genome_dict):
        """Test node with multiple incoming connections."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[2.0]])
        result = network.forward_pass(inputs)

        # Output gets input from two hidden nodes
        assert isinstance(result[0, 0], (float, np.floating))


class TestNetworkFastStringRepresentations:
    """Test string methods."""

    def test_str_representation(self, linear_genome_dict):
        """Test __str__ output."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        str_repr = str(network)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        assert "Node" in str_repr or "bias" in str_repr

    def test_repr_representation(self, linear_genome_dict):
        """Test __repr__ output."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        repr_str = repr(network)
        assert "NetworkFast" in repr_str
        assert "nodes=" in repr_str
        assert "connections=" in repr_str


class TestNetworkFastIntegration:
    """Integration and inherited functionality tests."""

    def test_works_with_real_genome(self, linear_genome_dict):
        """Test that NetworkFast works with real Genome objects."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        inputs = np.array([[1.0]])
        result = network.forward_pass(inputs)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1

    def test_properties_match_network_structure(self, diamond_genome_dict):
        """Test that inherited properties match network structure."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = NetworkFast(genome)

        assert network.number_nodes == 4
        assert network.number_nodes_hidden == 2
        assert network.number_connections == 4
        assert network.number_connections_enabled == 4

    def test_inherits_visualize(self, linear_genome_dict):
        """Test that NetworkFast inherits visualize from base class."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkFast(genome)

        assert hasattr(network, 'visualize')
        assert callable(network.visualize)
