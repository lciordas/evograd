"""
Unit tests for NetworkBase class.

Tests cover initialization, properties, topological sort, visualization,
abstract class behavior, and integration scenarios.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from evograd.phenotype.network_base import NetworkBase
from evograd.genotype import Genome, NodeGene, ConnectionGene, NodeType


# ============================================================================
# Test Fixtures
# ============================================================================

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
            {'from': 0, 'to': 2, 'weight': 1.0},
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
def complex_genome_dict():
    """Complex genome with disabled connections."""
    return {
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output'},
            {'id': 3, 'type': 'hidden'},
            {'id': 4, 'type': 'hidden'},
        ],
        'connections': [
            {'from': 0, 'to': 3, 'weight': 1.0, 'enabled': True},
            {'from': 1, 'to': 3, 'weight': 1.0, 'enabled': False},
            {'from': 3, 'to': 4, 'weight': 0.5, 'enabled': True},
            {'from': 4, 'to': 2, 'weight': 0.5, 'enabled': True},
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


# Helper: Create a concrete subclass for testing
class ConcreteNetwork(NetworkBase):
    """Concrete implementation of NetworkBase for testing."""
    def forward_pass(self, inputs):
        """Dummy implementation."""
        return inputs


# ============================================================================
# Test Classes
# ============================================================================

class TestNetworkBaseInit:
    """Test NetworkBase initialization."""

    def test_init_with_minimal_genome(self, minimal_genome_dict):
        """Test initialization with minimal genome."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network._genome is genome
        assert len(network._input_ids) == 2
        assert len(network._output_ids) == 1
        assert len(network._sorted_nodes) == 3

    def test_input_ids_extracted_correctly(self, linear_genome_dict):
        """Test that input node IDs are extracted correctly."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        assert network._input_ids == [0]
        assert all(genome.node_genes[nid].type == NodeType.INPUT for nid in network._input_ids)

    def test_output_ids_extracted_correctly(self, diamond_genome_dict):
        """Test that output node IDs are extracted correctly."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        assert network._output_ids == [1]
        assert all(genome.node_genes[nid].type == NodeType.OUTPUT for nid in network._output_ids)

    def test_sorted_nodes_computed(self, linear_genome_dict):
        """Test that topological sort is computed during init."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        assert isinstance(network._sorted_nodes, list)
        assert len(network._sorted_nodes) == 3
        # Node 0 should come before node 2, which should come before node 1
        assert network._sorted_nodes.index(0) < network._sorted_nodes.index(2)
        assert network._sorted_nodes.index(2) < network._sorted_nodes.index(1)

    def test_init_with_complex_genome(self, complex_genome_dict):
        """Test initialization with complex genome."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        assert len(network._input_ids) == 2
        assert len(network._output_ids) == 1
        assert len(network._sorted_nodes) == 5

    def test_init_stores_genome_reference(self, minimal_genome_dict):
        """Test that initialization stores reference to genome."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network._genome is genome


class TestNetworkBaseProperties:
    """Test NetworkBase properties."""

    def test_number_nodes_minimal(self, minimal_genome_dict):
        """Test number_nodes with minimal genome."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes == 3

    def test_number_nodes_complex(self, complex_genome_dict):
        """Test number_nodes with complex genome."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes == 5

    def test_number_nodes_matches_genome(self, diamond_genome_dict):
        """Test that number_nodes matches genome node count."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes == len(genome.node_genes)

    def test_number_nodes_hidden_zero(self, minimal_genome_dict):
        """Test number_nodes_hidden returns 0 when no hidden nodes."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes_hidden == 0

    def test_number_nodes_hidden_one(self, linear_genome_dict):
        """Test number_nodes_hidden with one hidden node."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes_hidden == 1

    def test_number_nodes_hidden_multiple(self, diamond_genome_dict):
        """Test number_nodes_hidden with multiple hidden nodes."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_nodes_hidden == 2

    def test_number_nodes_hidden_calculation(self, complex_genome_dict):
        """Test number_nodes_hidden calculation."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        # 5 total - 2 inputs - 1 output = 2 hidden
        assert network.number_nodes_hidden == 2

    def test_number_connections_zero(self, minimal_genome_dict):
        """Test number_connections with no connections."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_connections == 0

    def test_number_connections_one(self, no_hidden_genome_dict):
        """Test number_connections with one connection."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_connections == 1

    def test_number_connections_includes_disabled(self, complex_genome_dict):
        """Test that number_connections includes disabled connections."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        # Total is 4 (includes both enabled and disabled)
        assert network.number_connections == 4

    def test_number_connections_enabled_zero(self, minimal_genome_dict):
        """Test number_connections_enabled with no connections."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_connections_enabled == 0

    def test_number_connections_enabled_all_disabled(self, all_disabled_genome_dict):
        """Test number_connections_enabled when all are disabled."""
        genome = Genome.from_dict(all_disabled_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_connections_enabled == 0

    def test_number_connections_enabled_partial(self, complex_genome_dict):
        """Test number_connections_enabled with mix of enabled/disabled."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        # 3 enabled, 1 disabled
        assert network.number_connections_enabled == 3

    def test_number_connections_enabled_all_enabled(self, linear_genome_dict):
        """Test number_connections_enabled when all are enabled."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        assert network.number_connections_enabled == 2

    def test_properties_consistency(self, complex_genome_dict):
        """Test consistency between properties."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        # number_nodes_hidden should equal total - inputs - outputs
        assert network.number_nodes_hidden == (
            network.number_nodes - len(network._input_ids) - len(network._output_ids)
        )

        # number_connections_enabled should be <= number_connections
        assert network.number_connections_enabled <= network.number_connections


class TestNetworkBaseTopologicalSort:
    """Test NetworkBase topological sort."""

    def test_linear_chain(self, linear_genome_dict):
        """Test topological sort with linear chain."""
        genome = Genome.from_dict(linear_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Should have all 3 nodes
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

        # Node 0 must come before 2, and 2 before 1
        assert result.index(0) < result.index(2)
        assert result.index(2) < result.index(1)

    def test_diamond_topology(self, diamond_genome_dict):
        """Test topological sort with diamond topology."""
        genome = Genome.from_dict(diamond_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Should have all 4 nodes
        assert len(result) == 4
        assert set(result) == {0, 1, 2, 3}

        # Node 0 must come before both 2 and 3
        assert result.index(0) < result.index(2)
        assert result.index(0) < result.index(3)

        # Both 2 and 3 must come before 1
        assert result.index(2) < result.index(1)
        assert result.index(3) < result.index(1)

    def test_no_connections(self, minimal_genome_dict):
        """Test topological sort with no connections."""
        genome = Genome.from_dict(minimal_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Should include all nodes
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_disabled_connections_ignored(self, complex_genome_dict):
        """Test that disabled connections are ignored in sort."""
        genome = Genome.from_dict(complex_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Should have all 5 nodes
        assert len(result) == 5

        # Disabled connection from 1->3 should not create dependency
        # So node 1 can appear anywhere (no constraint from disabled edge)

    def test_all_nodes_included(self, diamond_genome_dict):
        """Test that all nodes are included in result."""
        genome = Genome.from_dict(diamond_genome_dict)
        result = NetworkBase._topological_sort(genome)

        all_node_ids = set(genome.node_genes.keys())
        result_set = set(result)

        assert result_set == all_node_ids

    def test_dependencies_satisfied(self, linear_genome_dict):
        """Test that dependencies are satisfied (no node before its inputs)."""
        genome = Genome.from_dict(linear_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Build position map
        position = {node_id: i for i, node_id in enumerate(result)}

        # Check all enabled connections
        for conn in genome.conn_genes.values():
            if conn.enabled:
                # Input node must come before output node
                assert position[conn.node_in] < position[conn.node_out]

    def test_multiple_independent_nodes(self):
        """Test with multiple independent input/output pairs."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'input'},
                {'id': 2, 'type': 'output'},
                {'id': 3, 'type': 'output'},
            ],
            'connections': [
                {'from': 0, 'to': 2, 'weight': 1.0},
                {'from': 1, 'to': 3, 'weight': 1.0},
            ],
            'activation': 'relu'
        }
        genome = Genome.from_dict(genome_dict)
        result = NetworkBase._topological_sort(genome)

        assert len(result) == 4
        assert result.index(0) < result.index(2)
        assert result.index(1) < result.index(3)

    def test_complex_topology_correctness(self, complex_genome_dict):
        """Test correctness with complex topology."""
        genome = Genome.from_dict(complex_genome_dict)
        result = NetworkBase._topological_sort(genome)

        # Build position map
        position = {node_id: i for i, node_id in enumerate(result)}

        # Verify all enabled edges satisfy dependency order
        for conn in genome.conn_genes.values():
            if conn.enabled:
                assert position[conn.node_in] < position[conn.node_out], \
                    f"Node {conn.node_in} should come before {conn.node_out}"

    def test_single_node_network(self):
        """Test with single node (just output, no input)."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }
        genome = Genome.from_dict(genome_dict)
        result = NetworkBase._topological_sort(genome)

        assert result == [0]

    def test_two_unconnected_nodes(self):
        """Test with two unconnected nodes."""
        genome_dict = {
            'nodes': [
                {'id': 0, 'type': 'input'},
                {'id': 1, 'type': 'output'},
            ],
            'connections': [],
            'activation': 'relu'
        }
        genome = Genome.from_dict(genome_dict)
        result = NetworkBase._topological_sort(genome)

        assert len(result) == 2
        assert set(result) == {0, 1}


class TestNetworkBaseVisualize:
    """Test NetworkBase visualization."""

    @patch('graphviz.Digraph')
    def test_visualize_returns_digraph(self, mock_digraph, minimal_genome_dict):
        """Test that visualize returns a Digraph object."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        result = network.visualize(view=False)

        # Should return the Digraph instance
        assert result is mock_digraph.return_value

    @patch('graphviz.Digraph')
    def test_visualize_view_false(self, mock_digraph, minimal_genome_dict):
        """Test visualize with view=False doesn't call view()."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # view() should not be called
        mock_instance.view.assert_not_called()

    @patch('graphviz.Digraph')
    def test_visualize_view_true(self, mock_digraph, minimal_genome_dict):
        """Test visualize with view=True calls view()."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=True)

        # view() should be called once with cleanup=True
        mock_instance.view.assert_called_once_with(cleanup=True)

    @patch('graphviz.Digraph')
    def test_visualize_includes_all_nodes(self, mock_digraph, diamond_genome_dict):
        """Test that visualization includes all nodes."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # Should call node() for each node (4 nodes)
        # Called via subgraph context managers, so check call count
        assert mock_instance.subgraph.return_value.__enter__.return_value.node.call_count >= 4

    @patch('graphviz.Digraph')
    def test_visualize_includes_all_connections(self, mock_digraph, diamond_genome_dict):
        """Test that visualization includes all connections."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # Should call edge() for each connection (4 connections)
        assert mock_instance.edge.call_count == 4

    @patch('graphviz.Digraph')
    def test_visualize_enabled_disabled_differently(self, mock_digraph, complex_genome_dict):
        """Test that enabled and disabled connections shown differently."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # Check that edge() was called with different colors
        edge_calls = mock_instance.edge.call_args_list
        colors = [call[1].get('color') for call in edge_calls]

        # Should have both 'black' (enabled) and 'lightgray' (disabled)
        assert 'black' in colors
        assert 'lightgray' in colors

    @patch('graphviz.Digraph')
    def test_visualize_with_no_hidden_nodes(self, mock_digraph, no_hidden_genome_dict):
        """Test visualization with no hidden nodes."""
        genome = Genome.from_dict(no_hidden_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value

        # Should not raise any errors
        network.visualize(view=False)

        # Should still create input and output clusters
        assert mock_instance.subgraph.call_count >= 2

    @patch('graphviz.Digraph')
    def test_visualize_sets_rankdir(self, mock_digraph, minimal_genome_dict):
        """Test that visualize sets left-to-right layout."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # Should set rankdir='LR'
        mock_instance.attr.assert_any_call(rankdir='LR')


class TestNetworkBaseAbstract:
    """Test NetworkBase abstract class behavior."""

    def test_cannot_instantiate_directly(self):
        """Test that NetworkBase cannot be instantiated directly."""
        mock_genome = Mock()

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NetworkBase(mock_genome)

    def test_forward_pass_is_abstract(self):
        """Test that forward_pass is an abstract method."""
        # Check that the method is marked as abstract
        assert hasattr(NetworkBase.forward_pass, '__isabstractmethod__')
        assert NetworkBase.forward_pass.__isabstractmethod__ is True

    def test_subclass_without_forward_pass_fails(self):
        """Test that subclass without forward_pass cannot be instantiated."""
        class IncompleteNetwork(NetworkBase):
            pass  # Does not implement forward_pass

        mock_genome = Mock()
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteNetwork(mock_genome)

    def test_subclass_with_forward_pass_succeeds(self):
        """Test that subclass with forward_pass can be instantiated."""
        class CompleteNetwork(NetworkBase):
            def forward_pass(self, inputs):
                return inputs

        mock_genome = Mock()
        mock_genome.input_nodes = []
        mock_genome.output_nodes = []
        mock_genome.node_genes = {}
        mock_genome.conn_genes = {}

        # Should not raise
        network = CompleteNetwork(mock_genome)
        assert isinstance(network, NetworkBase)

    def test_concrete_network_forward_pass_callable(self, minimal_genome_dict):
        """Test that ConcreteNetwork's forward_pass is callable."""
        genome = Genome.from_dict(minimal_genome_dict)
        network = ConcreteNetwork(genome)

        # Should be callable
        result = network.forward_pass([1.0, 2.0])
        assert result == [1.0, 2.0]


class TestNetworkBaseIntegration:
    """Test NetworkBase integration scenarios."""

    def test_works_with_real_genome(self, linear_genome_dict):
        """Test that NetworkBase works with real Genome objects."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        # All properties should work
        assert network.number_nodes > 0
        assert network.number_connections > 0
        assert len(network._sorted_nodes) > 0

    def test_properties_match_genome_structure(self, diamond_genome_dict):
        """Test that properties accurately reflect genome structure."""
        genome = Genome.from_dict(diamond_genome_dict)
        network = ConcreteNetwork(genome)

        # Verify counts match
        assert network.number_nodes == len(genome.node_genes)
        assert network.number_connections == len(genome.conn_genes)

        # Verify IDs match
        input_ids = [g.id for g in genome.input_nodes]
        output_ids = [g.id for g in genome.output_nodes]
        assert network._input_ids == input_ids
        assert network._output_ids == output_ids

    def test_topological_sort_matches_structure(self, linear_genome_dict):
        """Test that topological sort respects genome connections."""
        genome = Genome.from_dict(linear_genome_dict)
        network = ConcreteNetwork(genome)

        # Build position map
        position = {node_id: i for i, node_id in enumerate(network._sorted_nodes)}

        # Verify all connections respect topological order
        for conn in genome.conn_genes.values():
            if conn.enabled:
                assert position[conn.node_in] < position[conn.node_out]

    @patch('graphviz.Digraph')
    def test_visualization_matches_genome(self, mock_digraph, complex_genome_dict):
        """Test that visualization reflects genome structure."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        mock_instance = mock_digraph.return_value
        network.visualize(view=False)

        # Should have edges for all connections (enabled and disabled)
        assert mock_instance.edge.call_count == len(genome.conn_genes)

    def test_multiple_networks_from_same_genome(self, minimal_genome_dict):
        """Test creating multiple networks from same genome."""
        genome = Genome.from_dict(minimal_genome_dict)
        network1 = ConcreteNetwork(genome)
        network2 = ConcreteNetwork(genome)

        # Both should reference same genome
        assert network1._genome is genome
        assert network2._genome is genome

        # Both should have same properties
        assert network1.number_nodes == network2.number_nodes
        assert network1._sorted_nodes == network2._sorted_nodes

    def test_consistency_across_properties(self, complex_genome_dict):
        """Test that all properties are internally consistent."""
        genome = Genome.from_dict(complex_genome_dict)
        network = ConcreteNetwork(genome)

        # Hidden nodes calculation
        hidden_count = network.number_nodes - len(network._input_ids) - len(network._output_ids)
        assert network.number_nodes_hidden == hidden_count

        # All nodes should be in sorted list
        assert len(network._sorted_nodes) == network.number_nodes

        # Enabled connections should be subset of total
        assert network.number_connections_enabled <= network.number_connections
