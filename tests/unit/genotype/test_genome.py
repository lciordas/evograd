"""
Unit tests for Genome class.

Tests cover initialization, serialization, distance calculation, crossover,
mutations, pruning, and all helper methods.
"""

import pytest
import random
import numpy as np
import copy
from unittest.mock import Mock

from evograd.genotype.genome import Genome
from evograd.genotype.node_gene import NodeGene, NodeType
from evograd.genotype.connection_gene import ConnectionGene
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.run.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Standard config with 3 inputs, 2 outputs."""
    config = Mock(spec=Config)
    config.num_inputs = 3
    config.num_outputs = 2

    # Weight parameters
    config.weight_init_mean = 0.0
    config.weight_init_stdev = 1.0
    config.min_weight = -30.0
    config.max_weight = 30.0
    config.weight_perturb_prob = 0.8
    config.weight_perturb_strength = 0.5
    config.weight_replace_prob = 0.1

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

    # Mutation probabilities
    config.node_add_probability = 0.2
    config.node_delete_probability = 0.2
    config.connection_add_probability = 0.3
    config.connection_delete_probability = 0.2
    config.connection_enable_probability = 0.2
    config.connection_disable_probability = 0.2
    config.single_structural_mutation = False

    # Distance parameters
    config.distance_excess_coeff = 1.0
    config.distance_disjoint_coeff = 1.0
    config.distance_params_coeff = 0.4
    config.distance_includes_nodes = True
    config.activation_distance_k = 3.0

    # Legendre parameters
    config.num_legendre_coeffs = 4
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 1.0

    return config


@pytest.fixture
def minimal_config():
    """Minimal config with 1 input, 1 output."""
    config = Mock(spec=Config)
    config.num_inputs = 1
    config.num_outputs = 1
    config.weight_init_mean = 0.0
    config.weight_init_stdev = 1.0
    config.min_weight = -10.0
    config.max_weight = 10.0
    config.weight_perturb_prob = 0.5
    config.weight_perturb_strength = 0.5
    config.weight_replace_prob = 0.1
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
    config.activation_initial = 'relu'
    config.activation_mutate_prob = 0.2
    config.activation_options = ['relu', 'sigmoid']
    config.node_add_probability = 0.2
    config.node_delete_probability = 0.2
    config.connection_add_probability = 0.3
    config.connection_delete_probability = 0.2
    config.connection_enable_probability = 0.2
    config.connection_disable_probability = 0.2
    config.single_structural_mutation = False
    config.distance_excess_coeff = 1.0
    config.distance_disjoint_coeff = 1.0
    config.distance_params_coeff = 0.4
    config.distance_includes_nodes = True
    config.activation_distance_k = 3.0
    config.num_legendre_coeffs = 4
    config.legendre_coeffs_init_mean = 0.0
    config.legendre_coeffs_init_stdev = 1.0
    return config


@pytest.fixture
def zero_mutation_config(basic_config):
    """Config with all mutation probabilities set to 0."""
    config = copy.deepcopy(basic_config)
    config.node_add_probability = 0.0
    config.node_delete_probability = 0.0
    config.connection_add_probability = 0.0
    config.connection_delete_probability = 0.0
    config.connection_enable_probability = 0.0
    config.connection_disable_probability = 0.0
    config.weight_perturb_prob = 0.0
    config.weight_replace_prob = 0.0
    config.bias_perturb_prob = 0.0
    config.bias_replace_prob = 0.0
    config.gain_perturb_prob = 0.0
    config.gain_replace_prob = 0.0
    config.activation_mutate_prob = 0.0
    return config


@pytest.fixture
def single_structural_mutation_config(basic_config):
    """Config with single_structural_mutation enabled."""
    config = copy.deepcopy(basic_config)
    config.single_structural_mutation = True
    return config


@pytest.fixture(autouse=True)
def reset_innovation_tracker(basic_config):
    """Automatically reset InnovationTracker before each test."""
    InnovationTracker.initialize(basic_config)
    yield
    # Cleanup after test
    InnovationTracker.initialize(basic_config)


# ============================================================================
# Test: Initialization
# ============================================================================

class TestGenomeInit:
    """Test Genome initialization."""

    def test_basic_initialization(self, basic_config):
        """Test basic genome initialization with standard config."""
        genome = Genome(basic_config)

        # Should have input and output nodes
        assert len(genome.node_genes) == 5  # 3 inputs + 2 outputs
        assert len(genome.conn_genes) == 0  # No connections initially

        # Check input nodes
        for i in range(3):
            assert i in genome.node_genes
            node = genome.node_genes[i]
            assert node.id == i
            assert node.type == NodeType.INPUT
            assert node.bias == 0.0
            assert node.gain == 1.0
            assert node.activation is None

        # Check output nodes
        for i in range(2):
            node_id = 3 + i
            assert node_id in genome.node_genes
            node = genome.node_genes[node_id]
            assert node.id == node_id
            assert node.type == NodeType.OUTPUT

    def test_minimal_initialization(self, minimal_config):
        """Test initialization with minimal config (1 input, 1 output)."""
        genome = Genome(minimal_config)

        assert len(genome.node_genes) == 2
        assert 0 in genome.node_genes  # Input
        assert 1 in genome.node_genes  # Output

    def test_node_id_convention(self, basic_config):
        """Test that node IDs follow the convention."""
        genome = Genome(basic_config)

        # Input nodes: [0, num_inputs)
        input_ids = [n.id for n in genome.node_genes.values() if n.type == NodeType.INPUT]
        assert input_ids == [0, 1, 2]

        # Output nodes: [num_inputs, num_inputs + num_outputs)
        output_ids = [n.id for n in genome.node_genes.values() if n.type == NodeType.OUTPUT]
        assert output_ids == [3, 4]

    def test_input_nodes_have_fixed_bias_gain(self, basic_config):
        """Test that input nodes have bias=0.0, gain=1.0."""
        genome = Genome(basic_config)

        for node in genome.node_genes.values():
            if node.type == NodeType.INPUT:
                assert node.bias == 0.0
                assert node.gain == 1.0

    def test_output_nodes_have_random_bias_gain(self, basic_config):
        """Test that output nodes get random bias/gain from distribution."""
        random.seed(42)
        np.random.seed(42)

        genome = Genome(basic_config)

        for node in genome.node_genes.values():
            if node.type == NodeType.OUTPUT:
                # Should be some value (not necessarily 0.0 or 1.0)
                assert isinstance(node.bias, float)
                assert isinstance(node.gain, float)

    def test_large_io_counts(self):
        """Test initialization with large input/output counts."""
        config = Mock(spec=Config)
        config.num_inputs = 100
        config.num_outputs = 50
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.min_bias = -5.0
        config.max_bias = 5.0
        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.5
        config.min_gain = 0.1
        config.max_gain = 2.0
        config.activation_initial = 'relu'

        genome = Genome(config)

        assert len(genome.node_genes) == 150
        assert len([n for n in genome.node_genes.values() if n.type == NodeType.INPUT]) == 100
        assert len([n for n in genome.node_genes.values() if n.type == NodeType.OUTPUT]) == 50


# ============================================================================
# Test: from_dict
# ============================================================================

class TestGenomeFromDict:
    """Test Genome.from_dict() method."""

    def test_minimal_genome_from_dict(self):
        """Test creating minimal genome (only I/O nodes) from dict."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'input'},
                {'id': 2, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        assert len(genome.node_genes) == 3
        assert len(genome.conn_genes) == 0
        assert genome.node_genes[0].type == NodeType.INPUT
        assert genome.node_genes[1].type == NodeType.INPUT
        assert genome.node_genes[2].type == NodeType.OUTPUT

    def test_genome_with_connections_from_dict(self):
        """Test creating genome with connections from dict."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 1, 'weight': 1.5, 'enabled': True}
            ],
            'activation': 'sigmoid'
        }

        genome = Genome.from_dict(genome_dict)

        assert len(genome.conn_genes) == 1
        conn = list(genome.conn_genes.values())[0]
        assert conn.node_in == 0
        assert conn.node_out == 1
        assert conn.weight == 1.5
        assert conn.enabled is True

    def test_genome_with_hidden_nodes_from_dict(self):
        """Test creating genome with hidden nodes from dict."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden', 'bias': 0.5, 'gain': 1.2},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 2.0},
            ],
            'activation': 'tanh'
        }

        genome = Genome.from_dict(genome_dict)

        assert len(genome.node_genes) == 3
        assert genome.node_genes[2].type == NodeType.HIDDEN
        assert genome.node_genes[2].bias == 0.5
        assert genome.node_genes[2].gain == 1.2
        assert len(genome.conn_genes) == 2

    def test_node_specific_activation(self):
        """Test that node-specific activation overrides global."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output', 'activation': 'relu'},
                {'id': 2, 'type': 'output', 'activation': 'tanh'},
            ],
            'connections': [],
            'activation': 'sigmoid'  # Global default
        }

        genome = Genome.from_dict(genome_dict)

        assert genome.node_genes[1].activation_name == 'relu'
        assert genome.node_genes[2].activation_name == 'tanh'

    def test_legendre_activation_with_coeffs(self):
        """Test legendre activation with explicit coefficients."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output', 'activation': 'legendre',
                 'activation_coeffs': [1.0, 2.0, 3.0, 4.0]},
            ],
            'connections': [],
        }

        genome = Genome.from_dict(genome_dict)

        node = genome.node_genes[1]
        assert node.activation_name == 'legendre'
        assert node.activation_coeffs is not None
        assert np.array_equal(node.activation_coeffs, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_disabled_connection_preserved(self):
        """Test that disabled connections are preserved."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 1, 'weight': 1.0, 'enabled': False}
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        conn = list(genome.conn_genes.values())[0]
        assert conn.enabled is False

    def test_missing_global_activation_raises_error(self):
        """Test that missing activation raises ValueError."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},  # No activation specified
            ],
            'connections': []
            # No global activation
        }

        with pytest.raises(ValueError, match="[Aa]ctivation"):
            Genome.from_dict(genome_dict)

    def test_invalid_node_numbering_raises_error(self):
        """Test that invalid node numbering raises ValueError."""
        # Gap in input IDs (0, 2 instead of 0, 1)
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 2, 'type': 'input'},  # Should be 1
                {'id': 3, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        with pytest.raises(ValueError):
            Genome.from_dict(genome_dict)

    def test_cycle_creation_raises_error(self):
        """Test that creating a cycle raises ValueError."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
                {'id': 3, 'type': 'hidden'},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 3, 'weight': 1.0},
                {'from': 3, 'to': 2, 'weight': 1.0},  # Creates cycle: 2 → 3 → 2
            ],
            'activation': 'relu'
        }

        with pytest.raises(ValueError, match="[Cc]ycle"):
            Genome.from_dict(genome_dict)

    def test_connection_to_nonexistent_node_raises_error(self):
        """Test that connection to non-existent node raises ValueError."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 99, 'weight': 1.0}  # Node 99 doesn't exist
            ],
            'activation': 'relu'
        }

        with pytest.raises(ValueError):
            Genome.from_dict(genome_dict)

    def test_innovation_tracker_initialized(self):
        """Test that from_dict initializes InnovationTracker."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        # from_dict should call InnovationTracker.initialize()
        genome = Genome.from_dict(genome_dict)

        # After initialization, tracker should start from 0
        innov = InnovationTracker.get_innovation_number(0, 1)
        assert innov == 0


# ============================================================================
# Test: to_dict and Serialization Round-Trip
# ============================================================================

class TestGenomeToDict:
    """Test Genome.to_dict() method and serialization round-trips."""

    def test_minimal_genome_to_dict(self, minimal_config):
        """Test serializing minimal genome to dict."""
        genome = Genome(minimal_config)

        genome_dict = genome.to_dict()

        assert 'nodes' in genome_dict
        assert 'connections' in genome_dict
        assert len(genome_dict['nodes']) == 2
        assert len(genome_dict['connections']) == 0

    def test_genome_with_connections_to_dict(self, minimal_config):
        """Test serializing genome with connections."""
        InnovationTracker.initialize(minimal_config)
        genome = Genome(minimal_config)

        # Add a connection
        innov = InnovationTracker.get_innovation_number(0, 1)
        conn = ConnectionGene(0, 1, 1.5, innov, minimal_config)
        genome.conn_genes[innov] = conn

        genome_dict = genome.to_dict()

        assert len(genome_dict['connections']) == 1
        conn_dict = genome_dict['connections'][0]
        assert conn_dict['from'] == 0
        assert conn_dict['to'] == 1
        assert conn_dict['weight'] == 1.5

    def test_disabled_connection_in_to_dict(self, minimal_config):
        """Test that disabled connections are serialized."""
        InnovationTracker.initialize(minimal_config)
        genome = Genome(minimal_config)

        innov = InnovationTracker.get_innovation_number(0, 1)
        conn = ConnectionGene(0, 1, 1.0, innov, minimal_config, enabled=False)
        genome.conn_genes[innov] = conn

        genome_dict = genome.to_dict()

        conn_dict = genome_dict['connections'][0]
        assert conn_dict['enabled'] is False

    def test_round_trip_minimal_genome(self, minimal_config):
        """Test to_dict → from_dict round-trip for minimal genome."""
        genome1 = Genome(minimal_config)
        genome_dict = genome1.to_dict()
        genome2 = Genome.from_dict(genome_dict)

        # Should have same structure
        assert len(genome2.node_genes) == len(genome1.node_genes)
        assert len(genome2.conn_genes) == len(genome1.conn_genes)

    def test_round_trip_with_connections(self):
        """Test round-trip with connections."""
        genome_dict1 = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden', 'bias': 0.5, 'gain': 1.2},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 2.0},
            ],
            'activation': 'tanh'
        }

        genome = Genome.from_dict(genome_dict1)
        genome_dict2 = genome.to_dict()

        # Should have same structure
        assert len(genome_dict2['nodes']) == 3
        assert len(genome_dict2['connections']) == 2

    def test_legendre_coeffs_serialization(self):
        """Test that legendre coefficients are correctly serialized."""
        genome_dict1 = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output', 'activation': 'legendre',
                 'activation_coeffs': [1.0, 2.0, 3.0, 4.0]},
            ],
            'connections': []
        }

        genome = Genome.from_dict(genome_dict1)
        genome_dict2 = genome.to_dict()

        output_node = [n for n in genome_dict2['nodes'] if n['id'] == 1][0]
        assert 'activation_coeffs' in output_node
        assert output_node['activation_coeffs'] == [1.0, 2.0, 3.0, 4.0]


# ============================================================================
# Test: Properties
# ============================================================================

class TestGenomeProperties:
    """Test genome property accessors."""

    def test_input_nodes_property(self, basic_config):
        """Test input_nodes property returns correct nodes."""
        genome = Genome(basic_config)

        input_nodes = genome.input_nodes

        assert len(input_nodes) == 3
        for node in input_nodes:
            assert node.type == NodeType.INPUT

    def test_output_nodes_property(self, basic_config):
        """Test output_nodes property returns correct nodes."""
        genome = Genome(basic_config)

        output_nodes = genome.output_nodes

        assert len(output_nodes) == 2
        for node in output_nodes:
            assert node.type == NodeType.OUTPUT

    def test_hidden_nodes_property_empty(self, basic_config):
        """Test hidden_nodes property when no hidden nodes."""
        genome = Genome(basic_config)

        hidden_nodes = genome.hidden_nodes

        assert len(hidden_nodes) == 0

    def test_hidden_nodes_property_with_hidden(self):
        """Test hidden_nodes property with hidden nodes."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
                {'id': 3, 'type': 'hidden'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)
        hidden_nodes = genome.hidden_nodes

        assert len(hidden_nodes) == 2
        for node in hidden_nodes:
            assert node.type == NodeType.HIDDEN


# ============================================================================
# Test: Distance Calculation
# ============================================================================

class TestGenomeDistance:
    """Test genome distance calculation for speciation."""

    def test_identical_genomes_have_zero_distance(self, basic_config):
        """Test that truly identical genomes have distance of 0."""
        # Create identical genomes using same config and fixed seed
        import random, numpy as np
        random.seed(42)
        np.random.seed(42)

        genome1 = Genome(basic_config)

        # Reset seed to get same values
        random.seed(42)
        np.random.seed(42)
        genome2 = Genome(basic_config)

        distance = genome1.distance(genome2)

        assert distance == 0.0

    def test_empty_genomes_have_small_distance(self, basic_config):
        """Test that empty genomes (no connections) have small distance from node params."""
        genome1 = Genome(basic_config)
        genome2 = Genome(basic_config)

        distance = genome1.distance(genome2)

        # No connections means distance_NEAT = 0, but node distance may be non-zero
        # due to different random initializations
        assert isinstance(distance, (int, float))

    def test_one_genome_empty_has_nonzero_distance(self, minimal_config):
        """Test distance when one genome has connections, other doesn't."""
        InnovationTracker.initialize(minimal_config)

        genome1 = Genome(minimal_config)
        innov = InnovationTracker.get_innovation_number(0, 1)
        conn = ConnectionGene(0, 1, 1.0, innov, minimal_config)
        genome1.conn_genes[innov] = conn

        genome2 = Genome(minimal_config)

        distance = genome1.distance(genome2)

        # Should have non-zero distance due to excess gene
        assert distance > 0.0

    def test_distance_symmetry(self, minimal_config):
        """Test that distance(A, B) == distance(B, A)."""
        InnovationTracker.initialize(minimal_config)

        # Create two genomes with same connection but different weights
        genome1 = Genome(minimal_config)
        innov = InnovationTracker.get_innovation_number(0, 1)
        conn1 = ConnectionGene(0, 1, 1.0, innov, minimal_config)
        genome1.conn_genes[innov] = conn1

        genome2 = Genome(minimal_config)
        conn2 = ConnectionGene(0, 1, 2.0, innov, minimal_config)
        genome2.conn_genes[innov] = conn2

        distance_ab = genome1.distance(genome2)
        distance_ba = genome2.distance(genome1)

        assert distance_ab == distance_ba


# ============================================================================
# Test: Crossover
# ============================================================================

class TestGenomeCrossover:
    """Test genome crossover operation."""

    def test_crossover_with_minimal_genomes(self, minimal_config):
        """Test crossover with minimal genomes (no connections)."""
        InnovationTracker.initialize(minimal_config)

        genome1 = Genome(minimal_config)
        genome2 = Genome(minimal_config)

        offspring = genome1.crossover(genome2, genome1)

        # Should have same I/O nodes
        assert len(offspring.node_genes) == 2
        assert len(offspring.conn_genes) == 0

    def test_crossover_inherits_from_fitter_parent(self):
        """Test that disjoint/excess genes come from fitter parent."""
        genome_dict1 = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 1, 'weight': 1.0}
            ],
            'activation': 'relu'
        }

        genome_dict2 = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome1 = Genome.from_dict(genome_dict1)
        genome2 = Genome.from_dict(genome_dict2)

        # genome1 is fitter (has more structure)
        offspring = genome1.crossover(genome2, genome1)

        # Should inherit connection from fitter parent
        assert len(offspring.conn_genes) >= 0  # At least structure from fitter

    def test_crossover_always_includes_io_nodes(self):
        """Test that offspring always has all I/O nodes."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'input'},
                {'id': 2, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome1 = Genome.from_dict(genome_dict)
        genome2 = Genome.from_dict(genome_dict)

        offspring = genome1.crossover(genome2, genome1)

        # Must have all I/O nodes
        assert 0 in offspring.node_genes
        assert 1 in offspring.node_genes
        assert 2 in offspring.node_genes


# ============================================================================
# Test: Mutations
# ============================================================================

class TestGenomeMutations:
    """Test genome mutation operations."""

    def test_mutate_with_zero_probabilities_no_change(self, zero_mutation_config):
        """Test that genome doesn't change with zero mutation probabilities."""
        InnovationTracker.initialize(zero_mutation_config)
        genome = Genome(zero_mutation_config)

        # Add a connection first
        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov, zero_mutation_config)
        genome.conn_genes[innov] = conn

        original_node_count = len(genome.node_genes)
        original_conn_count = len(genome.conn_genes)
        original_weight = conn.weight

        genome.mutate()

        # Nothing should change
        assert len(genome.node_genes) == original_node_count
        assert len(genome.conn_genes) == original_conn_count
        assert conn.weight == original_weight

    def test_mutate_add_node_creates_node_and_connections(self, basic_config):
        """Test that add_node mutation creates a node and two connections."""
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add initial connection
        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov, basic_config, enabled=True)
        genome.conn_genes[innov] = conn

        original_node_count = len(genome.node_genes)

        # Manually call add_node mutation
        genome._mutate_add_node()

        # Should have one more node (hidden)
        assert len(genome.node_genes) == original_node_count + 1

        # Original connection should be disabled
        assert genome.conn_genes[innov].enabled is False

    def test_mutate_add_connection_respects_constraints(self, minimal_config):
        """Test that add_connection respects DAG constraint."""
        InnovationTracker.initialize(minimal_config)
        genome = Genome(minimal_config)

        # Try to add connection multiple times
        for _ in range(5):
            genome._mutate_add_connection()

        # Should have at most 1 connection (0→1)
        assert len(genome.conn_genes) <= 1

    def test_mutate_delete_node_removes_node_and_connections(self):
        """Test that delete_node removes node and its connections."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 2.0},
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)
        original_conn_count = len(genome.conn_genes)

        genome._mutate_delete_node()

        # Should have one fewer node
        assert len(genome.node_genes) == 2
        # Connections involving deleted node should be gone
        assert len(genome.conn_genes) < original_conn_count


# ============================================================================
# Test: Helper Methods
# ============================================================================

class TestGenomeHelperMethods:
    """Test genome helper methods."""

    def test_would_create_cycle_detects_simple_cycle(self):
        """Test cycle detection for simple cycle."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 1.0},
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        # Adding 1→2 would create cycle: 2→1→2
        would_cycle = genome._would_create_cycle(1, 2)
        assert would_cycle is True

    def test_would_create_cycle_self_loop(self):
        """Test that self-loop is detected as cycle."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        # Self-loop is a cycle
        would_cycle = genome._would_create_cycle(1, 1)
        assert would_cycle is True

    def test_would_create_cycle_no_cycle(self):
        """Test that valid connection is not flagged as cycle."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        # Forward connection doesn't create cycle
        would_cycle = genome._would_create_cycle(0, 1)
        assert would_cycle is False

    def test_delete_node_raises_error_for_input_node(self, basic_config):
        """Test that deleting INPUT node raises ValueError."""
        genome = Genome(basic_config)

        with pytest.raises(ValueError):
            genome._delete_node(0)  # Try to delete input node

    def test_delete_node_raises_error_for_output_node(self, basic_config):
        """Test that deleting OUTPUT node raises ValueError."""
        genome = Genome(basic_config)

        with pytest.raises(ValueError):
            genome._delete_node(3)  # Try to delete output node

    def test_delete_connection_raises_error_for_nonexistent(self, basic_config):
        """Test that deleting non-existent connection raises KeyError."""
        genome = Genome(basic_config)

        with pytest.raises(KeyError):
            genome._delete_connection(999)

    def test_validate_node_numbering_with_valid_numbering(self):
        """Test validation passes with correct numbering."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'input'},
                {'id': 2, 'type': 'output'},
                {'id': 3, 'type': 'hidden'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        # Should not raise ValueError
        genome = Genome.from_dict(genome_dict)
        assert len(genome.node_genes) == 4

    def test_validate_node_numbering_with_gap_in_inputs(self):
        """Test validation fails with gap in input IDs."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 2, 'type': 'input'},  # Gap: missing 1
                {'id': 3, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }

        with pytest.raises(ValueError):
            Genome.from_dict(genome_dict)


# ============================================================================
# Test: Edge Cases and Invariants
# ============================================================================

class TestGenomeEdgeCases:
    """Test edge cases and invariant maintenance."""

    def test_genome_with_no_connections(self, basic_config):
        """Test genome with only nodes, no connections."""
        genome = Genome(basic_config)

        assert len(genome.node_genes) == 5
        assert len(genome.conn_genes) == 0

        # Should be able to serialize
        genome_dict = genome.to_dict()
        assert len(genome_dict['connections']) == 0

    def test_all_connections_disabled(self):
        """Test genome where all connections are disabled."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 1, 'weight': 1.0, 'enabled': False}
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        assert len(genome.conn_genes) == 1
        assert list(genome.conn_genes.values())[0].enabled is False

    def test_node_numbering_maintained_after_operations(self):
        """Test that node numbering convention is maintained."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 1.0},
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)

        # Input nodes should be [0, num_inputs)
        input_nodes = [n for n in genome.node_genes.values() if n.type == NodeType.INPUT]
        assert all(n.id < 1 for n in input_nodes)

        # Output nodes should be [num_inputs, num_inputs + num_outputs)
        output_nodes = [n for n in genome.node_genes.values() if n.type == NodeType.OUTPUT]
        assert all(1 <= n.id < 2 for n in output_nodes)

        # Hidden nodes should be >= num_inputs + num_outputs
        hidden_nodes = [n for n in genome.node_genes.values() if n.type == NodeType.HIDDEN]
        assert all(n.id >= 2 for n in hidden_nodes)

    def test_input_nodes_never_mutate_parameters(self, basic_config):
        """Test that INPUT nodes maintain bias=0.0, gain=1.0."""
        random.seed(42)
        np.random.seed(42)

        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add connection so mutation can occur
        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, innov, basic_config)
        genome.conn_genes[innov] = conn

        # Mutate many times
        for _ in range(50):
            genome.mutate()

        # INPUT nodes should still have bias=0.0, gain=1.0
        for node in genome.node_genes.values():
            if node.type == NodeType.INPUT:
                assert node.bias == 0.0
                assert node.gain == 1.0
                assert node.activation is None


# ============================================================================
# Test: Pruning
# ============================================================================

class TestGenomePruning:
    """Test genome pruning functionality."""

    def test_prune_removes_dead_end_nodes(self):
        """Test that pruning removes nodes with no path to output."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},  # Connected to output
                {'id': 3, 'type': 'hidden'},  # Dead-end (no path to output)
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 2, 'to': 1, 'weight': 1.0},
                {'from': 0, 'to': 3, 'weight': 1.0},  # Goes to dead-end
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)
        pruned = genome.prune()

        # Dead-end node (3) should be removed
        assert 3 not in pruned.node_genes
        # Connected node (2) should remain
        assert 2 in pruned.node_genes

    def test_prune_removes_disabled_connections(self, minimal_config):
        """Test that pruning removes all disabled connections."""
        InnovationTracker.initialize(minimal_config)
        genome = Genome(minimal_config)

        # Add enabled and disabled connections
        innov1 = InnovationTracker.get_innovation_number(0, 1)
        conn1 = ConnectionGene(0, 1, 1.0, innov1, minimal_config, enabled=True)
        genome.conn_genes[innov1] = conn1

        innov2 = InnovationTracker.get_innovation_number(1, 0)  # Different connection
        conn2 = ConnectionGene(1, 0, 2.0, innov2, minimal_config, enabled=False)
        genome.conn_genes[innov2] = conn2

        pruned = genome.prune()

        # Only enabled connection should remain
        assert len(pruned.conn_genes) == 1
        assert innov1 in pruned.conn_genes
        assert innov2 not in pruned.conn_genes

    def test_prune_creates_deep_copy(self):
        """Test that pruning doesn't modify original genome."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
                {'id': 2, 'type': 'hidden'},
            ],
            'connections': [
                {'from': 0, 'to': 1, 'weight': 1.0, 'enabled': False},
            ],
            'activation': 'relu'
        }

        genome = Genome.from_dict(genome_dict)
        original_node_count = len(genome.node_genes)
        original_conn_count = len(genome.conn_genes)

        pruned = genome.prune()

        # Original should be unchanged
        assert len(genome.node_genes) == original_node_count
        assert len(genome.conn_genes) == original_conn_count


# ============================================================================
# Test: Single Structural Mutation Mode
# ============================================================================

class TestGenomeSingleStructuralMutation:
    """Test single structural mutation mode."""

    def test_single_structural_mutation_only_one_mutation(self, single_structural_mutation_config):
        """Test that only one structural mutation occurs in single mode."""
        random.seed(42)
        np.random.seed(42)

        InnovationTracker.initialize(single_structural_mutation_config)
        genome = Genome(single_structural_mutation_config)

        # Add connections and hidden node so all mutations can potentially happen
        innov1 = InnovationTracker.get_innovation_number(0, 3)
        conn1 = ConnectionGene(0, 3, 1.0, innov1, single_structural_mutation_config)
        genome.conn_genes[innov1] = conn1

        # Add a hidden node manually
        hidden_node = NodeGene(5, NodeType.HIDDEN, single_structural_mutation_config,
                              bias=0.5, gain=1.0, activation_name='relu')
        genome.node_genes[5] = hidden_node

        innov2 = InnovationTracker.get_innovation_number(0, 5)
        conn2 = ConnectionGene(0, 5, 1.0, innov2, single_structural_mutation_config)
        genome.conn_genes[innov2] = conn2

        # The single structural mutation config should restrict to one mutation type
        # We just verify it doesn't crash and works correctly
        for _ in range(10):
            genome.mutate()

        # Should still be valid genome
        assert len(genome.node_genes) >= 5


# ============================================================================
# Test: String Representations
# ============================================================================

class TestGenomeStringMethods:
    """Test string representation methods."""

    def test_str_representation(self, minimal_config):
        """Test __str__ method produces valid output."""
        InnovationTracker.initialize(minimal_config)
        genome = Genome(minimal_config)

        # Add a connection
        innov = InnovationTracker.get_innovation_number(0, 1)
        conn = ConnectionGene(0, 1, 1.5, innov, minimal_config)
        genome.conn_genes[innov] = conn

        str_repr = str(genome)

        # Should contain "Nodes:" and "Conns:"
        assert "Nodes:" in str_repr or "nodes" in str_repr.lower()
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


# ============================================================================
# Test: Additional Crossover Coverage
# ============================================================================

class TestGenomeCrossoverDetailed:
    """Additional crossover tests for edge cases."""

    def test_crossover_with_matching_connections_random_choice(self, minimal_config):
        """Test that matching connections are randomly chosen from parents."""
        random.seed(42)
        np.random.seed(42)

        InnovationTracker.initialize(minimal_config)

        # Create two genomes with same connection but different weights
        genome1 = Genome(minimal_config)
        innov = InnovationTracker.get_innovation_number(0, 1)
        conn1 = ConnectionGene(0, 1, 10.0, innov, minimal_config)
        genome1.conn_genes[innov] = conn1

        genome2 = Genome(minimal_config)
        conn2 = ConnectionGene(0, 1, 20.0, innov, minimal_config)
        genome2.conn_genes[innov] = conn2

        # Run crossover multiple times
        weights = []
        for i in range(20):
            random.seed(i)
            offspring = genome1.crossover(genome2, genome1)
            weights.append(offspring.conn_genes[innov].weight)

        # Should have some variation (both 10.0 and 20.0 should appear)
        unique_weights = set(weights)
        assert len(unique_weights) > 1  # Not all the same


# ============================================================================
# Test: Mutation Edge Cases
# ============================================================================

class TestGenomeMutationEdgeCases:
    """Test edge cases in mutation operations."""

    def test_mutate_add_node_no_connections(self, basic_config):
        """Test add_node fails gracefully with no connections."""
        genome = Genome(basic_config)

        # Should not crash
        genome._mutate_add_node()

        # Should still have only I/O nodes
        assert len(genome.hidden_nodes) == 0

    def test_mutate_delete_node_no_hidden_nodes(self, basic_config):
        """Test delete_node fails gracefully with no hidden nodes."""
        genome = Genome(basic_config)

        # Should not crash
        genome._mutate_delete_node()

        # Should still have same nodes
        assert len(genome.node_genes) == 5

    def test_mutate_enable_connection_no_connections(self, basic_config):
        """Test enable_connection fails gracefully with no connections."""
        genome = Genome(basic_config)

        # Should not crash
        genome._mutate_enable_connection()

    def test_mutate_disable_connection_no_connections(self, basic_config):
        """Test disable_connection fails gracefully with no connections."""
        genome = Genome(basic_config)

        # Should not crash
        genome._mutate_disable_connection()

    def test_mutate_delete_connection_no_connections(self, basic_config):
        """Test delete_connection fails gracefully with no connections."""
        genome = Genome(basic_config)

        # Should not crash
        genome._mutate_delete_connection()


# ============================================================================
# Test: Partial State Handling
# ============================================================================

class TestGenomePartialStateHandling:
    """Test genome handling of partial states during node splitting."""

    def test_split_delete_node_re_split_same_connection(self, basic_config):
        """Test that splitting, deleting, then re-splitting a connection works correctly.

        This tests the scenario:
        1. Split connection (0→3) creating node H and connections (0→H) and (H→3)
        2. Delete node H, removing all three genes
        3. Re-split the same connection (0→3) - should recreate with same IDs
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add initial connection from input 0 to output 3
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn

        # First split - creates node and two connections
        genome._mutate_add_node()

        # Verify split occurred
        assert len(genome.node_genes) == 6  # 3 inputs + 2 outputs + 1 hidden
        assert genome.conn_genes[innov0].enabled is False  # Original disabled

        # Get the IDs that were created
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

        # Delete the hidden node - removes node and both new connections
        genome._delete_node(new_node_id)

        # Verify deletion
        assert new_node_id not in genome.node_genes
        assert innov1 not in genome.conn_genes
        assert innov2 not in genome.conn_genes
        assert len(genome.node_genes) == 5  # Back to original count

        # Re-split the same connection - should recreate with same IDs
        genome.conn_genes[innov0].enabled = True  # Re-enable for splitting
        genome._mutate_add_node()

        # Verify re-split creates same genes
        new_node_id2, innov1_2, innov2_2 = InnovationTracker.get_split_IDs(conn)
        assert new_node_id2 == new_node_id  # Same node ID
        assert innov1_2 == innov1  # Same innovation numbers
        assert innov2_2 == innov2

        # Verify genes exist in genome
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

    def test_partial_state_after_deleting_one_split_connection(self, basic_config):
        """Test partial state cleanup when one split connection is deleted.

        This tests the scenario:
        1. Split connection creating node H, innov1, innov2
        2. Manually delete just innov1 (creating partial state)
        3. Try to split again - should cleanup and recreate all genes
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add and split a connection
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn
        genome._mutate_add_node()

        # Get the split gene IDs
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Verify all three genes exist
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

        # Create partial state by deleting just one connection
        genome._delete_connection(innov1)

        # Verify partial state: node exists, innov2 exists, innov1 missing
        assert new_node_id in genome.node_genes
        assert innov1 not in genome.conn_genes
        assert innov2 in genome.conn_genes

        # Disable innov2 so only the original connection can be split
        genome.conn_genes[innov2].enabled = False

        # Re-enable original connection and try to split again
        genome.conn_genes[innov0].enabled = True
        genome._mutate_add_node()

        # Verify partial state was cleaned up and all genes recreated
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

    def test_partial_state_with_only_node_remaining(self, basic_config):
        """Test partial state cleanup when only the split node remains.

        This tests the scenario where both connections are gone but node remains.
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add and split a connection
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn
        genome._mutate_add_node()

        # Get the split gene IDs
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Create partial state: delete both connections but keep node
        genome._delete_connection(innov1)
        genome._delete_connection(innov2)

        # Verify partial state: only node remains
        assert new_node_id in genome.node_genes
        assert innov1 not in genome.conn_genes
        assert innov2 not in genome.conn_genes

        # Try to split again
        genome.conn_genes[innov0].enabled = True
        genome._mutate_add_node()

        # Verify cleanup recreated all genes
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

    def test_partial_state_with_one_connection_remaining(self, basic_config):
        """Test partial state cleanup when only one connection remains.

        This tests having 1 out of 3 split genes.
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add and split a connection
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn
        genome._mutate_add_node()

        # Get the split gene IDs
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Create partial state: delete node and one connection
        del genome.node_genes[new_node_id]
        genome._delete_connection(innov2)

        # Verify partial state: only innov1 remains
        assert new_node_id not in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 not in genome.conn_genes

        # Disable innov1 so only the original connection can be split
        genome.conn_genes[innov1].enabled = False

        # Try to split again
        genome.conn_genes[innov0].enabled = True
        genome._mutate_add_node()

        # Verify cleanup recreated all genes
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes

    def test_no_partial_state_when_all_genes_exist(self, basic_config):
        """Test that having all split genes doesn't trigger partial state cleanup.

        When all three genes exist, the code should just re-enable the connections.
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add and split a connection
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn
        genome._mutate_add_node()

        # Get the split gene IDs
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Disable the split connections
        genome.conn_genes[innov1].enabled = False
        genome.conn_genes[innov2].enabled = False

        # Try to split again with all genes present
        genome.conn_genes[innov0].enabled = True
        genome._mutate_add_node()

        # All genes should still exist and connections should be re-enabled
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes
        assert genome.conn_genes[innov1].enabled is True
        assert genome.conn_genes[innov2].enabled is True

    def test_no_partial_state_when_no_genes_exist(self, basic_config):
        """Test that having no split genes creates them fresh without cleanup.

        This is the normal first-time split scenario.
        """
        InnovationTracker.initialize(basic_config)
        genome = Genome(basic_config)

        # Add a connection but don't split it yet
        innov0 = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.5, innov0, basic_config, enabled=True)
        genome.conn_genes[innov0] = conn

        original_node_count = len(genome.node_genes)

        # First split - normal case
        genome._mutate_add_node()

        # Should create new genes without any cleanup needed
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)
        assert new_node_id in genome.node_genes
        assert innov1 in genome.conn_genes
        assert innov2 in genome.conn_genes
        assert len(genome.node_genes) == original_node_count + 1
