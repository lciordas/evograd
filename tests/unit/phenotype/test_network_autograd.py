"""
Unit tests for neat.phenotype.network_autograd module.

This module contains comprehensive tests for the NetworkAutograd class,
which provides an autograd-compatible neural network implementation.
"""

import pytest
import autograd.numpy as np
from autograd import grad
from unittest.mock import Mock

from evograd.run.config import Config
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.network_autograd import NetworkAutograd
from evograd.phenotype.network_standard import NetworkStandard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_innovation_tracker():
    """Reset InnovationTracker before each test to ensure clean state."""
    config = Mock(spec=Config)
    config.num_inputs = 2
    config.num_outputs = 1
    InnovationTracker.initialize(config)
    yield
    InnovationTracker.initialize(config)


@pytest.fixture
def simple_genome_dict():
    """
    Minimal genome: 2 inputs -> 1 output (direct connections).
    Node numbering: inputs [0,2), outputs [2,3)
    """
    return {
        'activation': 'identity',
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output'},
        ],
        'connections': [
            {'from': 0, 'to': 2, 'weight': 1.0, 'enabled': True},
            {'from': 1, 'to': 2, 'weight': 1.0, 'enabled': True},
        ]
    }


@pytest.fixture
def linear_genome_dict():
    """
    Linear chain: input -> hidden -> output.
    Node numbering: inputs [0,1), outputs [1,2), hidden [2,3)
    """
    return {
        'activation': 'identity',
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output'},
            {'id': 2, 'type': 'hidden'},
        ],
        'connections': [
            {'from': 0, 'to': 2, 'weight': 2.0, 'enabled': True},
            {'from': 2, 'to': 1, 'weight': 0.5, 'enabled': True},
        ]
    }


@pytest.fixture
def relu_genome_dict():
    """
    Genome with ReLU activation: input -> hidden(relu) -> output(identity).
    """
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output', 'activation': 'identity'},
            {'id': 2, 'type': 'hidden', 'activation': 'relu'},
        ],
        'connections': [
            {'from': 0, 'to': 2, 'weight': 1.0, 'enabled': True},
            {'from': 2, 'to': 1, 'weight': 2.0, 'enabled': True},
        ]
    }


@pytest.fixture
def legendre_genome_dict():
    """
    Genome with Legendre activation for testing learnable activations.
    """
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'output', 'activation': 'legendre',
             'activation_coeffs': [1.0, 0.5, 0.0]},  # degree 2
        ],
        'connections': [
            {'from': 0, 'to': 1, 'weight': 1.0, 'enabled': True},
        ]
    }


# ============================================================================
# Test NetworkAutograd Initialization
# ============================================================================

class TestNetworkAutogradInit:
    """Test NetworkAutograd.__init__ method."""

    def test_init_creates_weight_matrix(self, simple_genome_dict):
        """Test that initialization creates correct weight matrix."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        assert network.weights.shape == (3, 3)  # 3 nodes
        assert isinstance(network.weights, np.ndarray)

    def test_init_creates_bias_and_gain_vectors(self, simple_genome_dict):
        """Test that initialization creates bias and gain vectors."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        assert network.biases.shape == (3,)  # 3 nodes
        assert network.gains.shape == (3,)   # 3 nodes
        assert isinstance(network.biases, np.ndarray)
        assert isinstance(network.gains, np.ndarray)

    def test_init_stores_activation_functions(self, relu_genome_dict):
        """Test that initialization stores activation functions."""
        genome = Genome.from_dict(relu_genome_dict)
        network = NetworkAutograd(genome)

        assert len(network.activations) == 3
        # Non-input nodes should have activation functions
        for idx in network._hidden_output_indices:
            assert network.activations[idx] is not None

    def test_init_handles_legendre_activation(self, legendre_genome_dict):
        """Test that initialization handles Legendre activation correctly."""
        genome = Genome.from_dict(legendre_genome_dict)
        network = NetworkAutograd(genome)

        # Output node (id=1) should have LegendreActivation
        output_idx = network._node_id_to_idx[1]
        from evograd.activations.legendre_activation import LegendreActivation
        assert isinstance(network.activations[output_idx], LegendreActivation)
        assert output_idx in network.activation_coeffs
        assert len(network.activation_coeffs[output_idx]) == 3

    def test_init_creates_node_mappings(self, simple_genome_dict):
        """Test that initialization creates node ID to index mappings."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        assert len(network._node_id_to_idx) == 3
        assert len(network._idx_to_node_id) == 3
        assert all(network._idx_to_node_id[network._node_id_to_idx[node_id]] == node_id
                   for node_id in genome.node_genes.keys())

    def test_init_computes_incoming_connections(self, linear_genome_dict):
        """Test that initialization pre-computes incoming connections."""
        genome = Genome.from_dict(linear_genome_dict)
        network = NetworkAutograd(genome)

        assert len(network._incoming_connections) == 3
        # Output node should have hidden node as incoming
        output_idx = network._node_id_to_idx[1]
        hidden_idx = network._node_id_to_idx[2]
        assert hidden_idx in network._incoming_connections[output_idx]


# ============================================================================
# Test NetworkAutograd get_parameters
# ============================================================================

class TestNetworkAutogradGetParameters:
    """Test NetworkAutograd.get_parameters method."""

    def test_get_parameters_returns_tuple(self, simple_genome_dict):
        """Test that get_parameters returns a tuple of 4 elements."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        result = network.get_parameters()

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_get_parameters_returns_copies(self, simple_genome_dict):
        """Test that get_parameters returns copies, not references."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        w1, b1, g1, c1 = network.get_parameters()

        # Modify returned parameters
        w1[0, 0] = 999.0
        b1[0] = 999.0
        g1[0] = 999.0

        # Verify network parameters unchanged
        w2, b2, g2, c2 = network.get_parameters()
        assert w2[0, 0] != 999.0
        assert b2[0] != 999.0
        assert g2[0] != 999.0

    def test_get_parameters_legendre_coeffs_copy(self, legendre_genome_dict):
        """Test that activation coefficients are copied."""
        genome = Genome.from_dict(legendre_genome_dict)
        network = NetworkAutograd(genome)

        _, _, _, coeffs = network.get_parameters()

        # Modify returned coefficients
        output_idx = network._node_id_to_idx[1]
        coeffs[output_idx][0] = 999.0

        # Verify network coefficients unchanged
        _, _, _, coeffs2 = network.get_parameters()
        assert coeffs2[output_idx][0] != 999.0


# ============================================================================
# Test NetworkAutograd set_parameters
# ============================================================================

class TestNetworkAutogradSetParameters:
    """Test NetworkAutograd.set_parameters method."""

    def test_set_parameters_updates_weights(self, simple_genome_dict):
        """Test that set_parameters updates weights."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        w, b, g, c = network.get_parameters()
        w_new = w * 2.0

        network.set_parameters(w_new, b, g, c)

        w_result, _, _, _ = network.get_parameters()
        assert np.allclose(w_result, w_new)

    def test_set_parameters_clips_to_bounds(self, simple_genome_dict):
        """Test that set_parameters clips values to config bounds."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)
        config = genome._config

        w, b, g, c = network.get_parameters()

        # Set extreme values
        w_extreme = np.full_like(w, 9999.0)
        b_extreme = np.full_like(b, -9999.0)
        g_extreme = np.full_like(g, 9999.0)

        network.set_parameters(w_extreme, b_extreme, g_extreme, c)

        w_result, b_result, g_result, _ = network.get_parameters()

        # Should be clipped to config bounds
        assert np.all(w_result <= config.max_weight)
        assert np.all(b_result >= config.min_bias)
        assert np.all(g_result <= config.max_gain)

    def test_set_parameters_updates_coefficients(self, legendre_genome_dict):
        """Test that set_parameters updates activation coefficients."""
        genome = Genome.from_dict(legendre_genome_dict)
        network = NetworkAutograd(genome)

        w, b, g, c = network.get_parameters()
        output_idx = network._node_id_to_idx[1]

        # Modify coefficients
        c_new = c.copy()
        c_new[output_idx] = np.array([5.0, 5.0, 5.0])

        network.set_parameters(w, b, g, c_new)

        _, _, _, c_result = network.get_parameters()
        assert np.allclose(c_result[output_idx], [5.0, 5.0, 5.0])


# ============================================================================
# Test NetworkAutograd forward_pass
# ============================================================================

class TestNetworkAutogradForwardPass:
    """Test NetworkAutograd.forward_pass method."""

    def test_forward_pass_basic_computation(self, simple_genome_dict):
        """Test basic forward pass computation."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0, 2.0]])
        outputs = network.forward_pass(inputs)

        # identity(1.0 * (1.0 * 1.0 + 1.0 * 2.0) + 0.0) = 3.0
        assert outputs.shape == (1, 1)
        assert np.allclose(outputs[0, 0], 3.0)

    def test_forward_pass_batch_processing(self, simple_genome_dict):
        """Test forward pass with batch inputs."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        outputs = network.forward_pass(inputs)

        assert outputs.shape == (3, 1)
        assert np.allclose(outputs[:, 0], [3.0, 7.0, 11.0])

    def test_forward_pass_1d_input_auto_reshape(self, simple_genome_dict):
        """Test that 1D input is automatically reshaped to 2D."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([1.0, 2.0])  # 1D
        outputs = network.forward_pass(inputs)

        assert outputs.shape == (1, 1)
        assert np.allclose(outputs[0, 0], 3.0)

    def test_forward_pass_list_input(self, simple_genome_dict):
        """Test that list input is converted to numpy array."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = [1.0, 2.0]  # Python list
        outputs = network.forward_pass(inputs)

        assert isinstance(outputs, np.ndarray)
        assert outputs.shape == (1, 1)

    def test_forward_pass_invalid_input_dimension(self, simple_genome_dict):
        """Test that 3D input raises ValueError."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[[1.0, 2.0]]])  # 3D

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            network.forward_pass(inputs)

    def test_forward_pass_wrong_input_size(self, simple_genome_dict):
        """Test that wrong input size raises ValueError."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0, 2.0, 3.0]])  # 3 inputs, expects 2

        with pytest.raises(ValueError, match="Expected 2 inputs, got 3"):
            network.forward_pass(inputs)

    def test_forward_pass_with_explicit_parameters(self, simple_genome_dict):
        """Test forward pass with explicitly passed parameters."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        w, b, g, c = network.get_parameters()
        w_modified = w * 2.0  # Double all weights

        inputs = np.array([[1.0, 2.0]])

        # Use modified weights
        outputs = network.forward_pass(inputs, weights=w_modified)

        # With doubled weights: identity(1.0 * (2.0 * 1.0 + 2.0 * 2.0) + 0.0) = 6.0
        assert np.allclose(outputs[0, 0], 6.0)

    def test_forward_pass_relu_activation(self, relu_genome_dict):
        """Test forward pass with ReLU activation."""
        genome = Genome.from_dict(relu_genome_dict)
        network = NetworkAutograd(genome)

        # Test positive input (ReLU passes through)
        outputs_pos = network.forward_pass(np.array([[2.0]]))
        # hidden: relu(1.0 * 2.0 + 0.0) = 2.0
        # output: identity(2.0 * 2.0 + 0.0) = 4.0
        assert np.allclose(outputs_pos[0, 0], 4.0)

        # Test negative input (ReLU blocks)
        outputs_neg = network.forward_pass(np.array([[-2.0]]))
        # hidden: relu(1.0 * -2.0 + 0.0) = 0.0
        # output: identity(2.0 * 0.0 + 0.0) = 0.0
        assert np.allclose(outputs_neg[0, 0], 0.0)

    def test_forward_pass_legendre_activation(self, legendre_genome_dict):
        """Test forward pass with Legendre activation."""
        genome = Genome.from_dict(legendre_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0]])
        outputs = network.forward_pass(inputs)

        # Should use Legendre polynomial with given coefficients
        assert outputs.shape == (1, 1)
        assert isinstance(outputs[0, 0], (float, np.floating))


# ============================================================================
# Test NetworkAutograd save_parameters_to_genome
# ============================================================================

class TestNetworkAutogradSaveParameters:
    """Test NetworkAutograd.save_parameters_to_genome method."""

    def test_save_parameters_updates_genome_weights(self, simple_genome_dict):
        """Test that save_parameters updates genome connection weights."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        # Modify network weights
        w, b, g, c = network.get_parameters()
        w_modified = w * 3.0
        network.set_parameters(w_modified, b, g, c)

        # Save back to genome
        network.save_parameters_to_genome()

        # Check genome connections have updated weights
        for conn_gene in genome.conn_genes.values():
            if conn_gene.enabled:
                from_idx = network._node_id_to_idx[conn_gene.node_in]
                to_idx = network._node_id_to_idx[conn_gene.node_out]
                expected_weight = w_modified[from_idx, to_idx]
                assert np.isclose(conn_gene.weight, expected_weight)

    def test_save_parameters_updates_genome_node_params(self, simple_genome_dict):
        """Test that save_parameters updates genome node biases and gains."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        # Modify network parameters
        w, b, g, c = network.get_parameters()
        b_modified = b + 5.0
        g_modified = g + 2.0
        network.set_parameters(w, b_modified, g_modified, c)

        # Save back to genome
        network.save_parameters_to_genome()

        # Check genome nodes have updated parameters
        for node_id in genome.node_genes.keys():
            idx = network._node_id_to_idx[node_id]
            node_gene = genome.node_genes[node_id]
            assert np.isclose(node_gene.bias, b_modified[idx])
            assert np.isclose(node_gene.gain, g_modified[idx])

    def test_save_parameters_updates_activation_coeffs(self, legendre_genome_dict):
        """Test that save_parameters updates Legendre coefficients."""
        genome = Genome.from_dict(legendre_genome_dict)
        network = NetworkAutograd(genome)

        # Modify coefficients
        w, b, g, c = network.get_parameters()
        output_idx = network._node_id_to_idx[1]
        c[output_idx] = np.array([10.0, 20.0, 30.0])
        network.set_parameters(w, b, g, c)

        # Save back to genome
        network.save_parameters_to_genome()

        # Check genome node has updated coefficients
        output_gene = genome.node_genes[1]
        assert np.allclose(output_gene.activation_coeffs, [10.0, 20.0, 30.0])

    def test_save_parameters_converts_to_python_types(self, simple_genome_dict):
        """Test that save_parameters converts numpy types to Python float."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        network.save_parameters_to_genome()

        # Check all weights, biases, gains are Python float (not numpy types)
        for conn_gene in genome.conn_genes.values():
            assert isinstance(conn_gene.weight, float)
            assert not isinstance(conn_gene.weight, np.floating)

        for node_gene in genome.node_genes.values():
            assert isinstance(node_gene.bias, float)
            assert isinstance(node_gene.gain, float)


# ============================================================================
# Test NetworkAutograd Autograd Compatibility
# ============================================================================

class TestNetworkAutogradAutogradCompatibility:
    """Test NetworkAutograd compatibility with autograd for gradient computation."""

    def test_gradient_computation_weights(self, simple_genome_dict):
        """Test that gradients can be computed with respect to weights."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        w, b, g, c = network.get_parameters()
        inputs = np.array([[1.0, 2.0]])
        target = np.array([[5.0]])

        # Define loss function
        def loss_fn(weights):
            outputs = network.forward_pass(inputs, weights=weights, biases=b, gains=g, coeffs=c)
            return np.mean((outputs - target)**2)

        # Compute gradient
        grad_fn = grad(loss_fn)
        gradient = grad_fn(w)

        # Gradient should be computable and have same shape as weights
        assert gradient.shape == w.shape
        assert not np.all(gradient == 0.0)  # Should have non-zero gradients

    def test_gradient_computation_biases(self, simple_genome_dict):
        """Test that gradients can be computed with respect to biases."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        w, b, g, c = network.get_parameters()
        inputs = np.array([[1.0, 2.0]])
        target = np.array([[5.0]])

        def loss_fn(biases):
            outputs = network.forward_pass(inputs, weights=w, biases=biases, gains=g, coeffs=c)
            return np.mean((outputs - target)**2)

        grad_fn = grad(loss_fn)
        gradient = grad_fn(b)

        assert gradient.shape == b.shape
        assert not np.all(gradient == 0.0)

    def test_gradient_descent_step(self, simple_genome_dict):
        """Test that gradient descent can improve predictions."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0, 2.0]])
        target = np.array([[10.0]])

        w, b, g, c = network.get_parameters()

        # Compute initial loss
        outputs_before = network.forward_pass(inputs, w, b, g, c)
        loss_before = np.mean((outputs_before - target)**2)

        # Gradient descent step on weights
        def loss_fn(weights):
            outputs = network.forward_pass(inputs, weights=weights, biases=b, gains=g, coeffs=c)
            return np.mean((outputs - target)**2)

        grad_fn = grad(loss_fn)
        gradient = grad_fn(w)

        # Update weights
        learning_rate = 0.1
        w_new = w - learning_rate * gradient
        network.set_parameters(w_new, b, g, c)

        # Compute loss after update
        outputs_after = network.forward_pass(inputs)
        loss_after = np.mean((outputs_after - target)**2)

        # Loss should decrease
        assert loss_after < loss_before


# ============================================================================
# Test NetworkAutograd Comparison with NetworkStandard
# ============================================================================

class TestNetworkAutogradComparison:
    """Test NetworkAutograd produces same results as NetworkStandard."""

    def test_comparison_simple_network(self, simple_genome_dict):
        """Test that NetworkAutograd matches NetworkStandard on simple network."""
        genome = Genome.from_dict(simple_genome_dict)

        net_autograd = NetworkAutograd(genome)
        net_standard = NetworkStandard(genome)

        inputs_autograd = np.array([[1.0, 2.0]])
        inputs_standard = (1.0, 2.0)

        output_autograd = net_autograd.forward_pass(inputs_autograd)
        output_standard = net_standard.forward_pass(inputs_standard)

        assert np.allclose(output_autograd[0], output_standard)

    def test_comparison_batch_processing(self, linear_genome_dict):
        """Test batch processing matches sequential standard processing."""
        genome = Genome.from_dict(linear_genome_dict)

        net_autograd = NetworkAutograd(genome)
        net_standard = NetworkStandard(genome)

        batch = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        outputs_autograd = net_autograd.forward_pass(batch)

        # Process each input individually with NetworkStandard
        outputs_standard = []
        for i in range(batch.shape[0]):
            output = net_standard.forward_pass((batch[i, 0],))
            outputs_standard.append(output[0])

        assert np.allclose(outputs_autograd[:, 0], outputs_standard)


# ============================================================================
# Test NetworkAutograd String Methods
# ============================================================================

class TestNetworkAutogradStringMethods:
    """Test NetworkAutograd.__str__ and __repr__ methods."""

    def test_str_contains_node_info(self, simple_genome_dict):
        """Test that __str__ contains node information."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        result = str(network)

        assert 'Node' in result
        assert 'bias=' in result
        assert 'gain=' in result

    def test_repr_contains_network_info(self, simple_genome_dict):
        """Test that __repr__ contains network structure info."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        result = repr(network)

        assert 'NetworkAutograd' in result
        assert 'nodes=' in result
        assert 'connections=' in result


# ============================================================================
# Test NetworkAutograd Edge Cases
# ============================================================================

class TestNetworkAutogradEdgeCases:
    """Test edge cases and special scenarios."""

    def test_no_hidden_nodes_direct_connections(self, simple_genome_dict):
        """Test network with no hidden nodes (direct input-output)."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0, 2.0]])
        outputs = network.forward_pass(inputs)

        assert outputs.shape == (1, 1)
        assert isinstance(outputs[0, 0], (float, np.floating))

    def test_single_input_sample(self, simple_genome_dict):
        """Test processing single input sample."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        inputs = np.array([[1.0, 2.0]])
        outputs = network.forward_pass(inputs)

        assert outputs.shape == (1, 1)

    def test_large_batch_processing(self, simple_genome_dict):
        """Test processing large batch of inputs."""
        genome = Genome.from_dict(simple_genome_dict)
        network = NetworkAutograd(genome)

        batch_size = 100
        inputs = np.random.randn(batch_size, 2)
        outputs = network.forward_pass(inputs)

        assert outputs.shape == (batch_size, 1)
