"""
Unit tests for InnovationTracker class.

Tests cover initialization, innovation number assignment, connection splits,
state management, and edge cases.
"""

import pytest
from unittest.mock import Mock

from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.genotype.connection_gene import ConnectionGene
from evograd.run.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Config with standard input/output counts."""
    config = Mock(spec=Config)
    config.num_inputs = 3
    config.num_outputs = 2
    # For ConnectionGene creation in split tests
    config.min_weight = -10.0
    config.max_weight = 10.0
    return config


@pytest.fixture
def zero_io_config():
    """Edge case config with zero inputs and outputs."""
    config = Mock(spec=Config)
    config.num_inputs = 0
    config.num_outputs = 0
    config.min_weight = -10.0
    config.max_weight = 10.0
    return config


@pytest.fixture
def large_io_config():
    """Config with large input/output counts."""
    config = Mock(spec=Config)
    config.num_inputs = 100
    config.num_outputs = 50
    config.min_weight = -10.0
    config.max_weight = 10.0
    return config


@pytest.fixture
def single_io_config():
    """Config with single input and output."""
    config = Mock(spec=Config)
    config.num_inputs = 1
    config.num_outputs = 1
    config.min_weight = -10.0
    config.max_weight = 10.0
    return config


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInnovationTrackerInitialize:
    """Test InnovationTracker initialization."""

    def test_initialize_with_standard_config(self, basic_config):
        """Test basic initialization with standard config."""
        InnovationTracker.initialize(basic_config)

        # Get first innovation number - should be 0
        innov = InnovationTracker.get_innovation_number(0, 1)
        assert innov == 0

        # Get first node ID - should be num_inputs + num_outputs = 5
        conn = ConnectionGene(0, 1, 1.0, 0, basic_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)
        assert node_id == 5  # 3 inputs + 2 outputs

    def test_initialize_clears_previous_state(self, basic_config):
        """Test that re-initialization clears all previous state."""
        InnovationTracker.initialize(basic_config)

        # Create some innovations
        innov1 = InnovationTracker.get_innovation_number(0, 1)
        innov2 = InnovationTracker.get_innovation_number(1, 2)

        # Create a split
        conn = ConnectionGene(0, 1, 1.0, innov1, basic_config)
        node_id1, _, _ = InnovationTracker.get_split_IDs(conn)

        # Re-initialize
        InnovationTracker.initialize(basic_config)

        # Should start from 0 again
        new_innov = InnovationTracker.get_innovation_number(0, 1)
        assert new_innov == 0

        # Node ID should restart
        conn2 = ConnectionGene(0, 1, 1.0, new_innov, basic_config)
        new_node_id, _, _ = InnovationTracker.get_split_IDs(conn2)
        assert new_node_id == 5  # Same as before, state reset

    def test_initialize_with_zero_inputs_outputs(self, zero_io_config):
        """Test initialization with zero inputs and outputs."""
        InnovationTracker.initialize(zero_io_config)

        # First innovation should be 0
        innov = InnovationTracker.get_innovation_number(0, 1)
        assert innov == 0

        # First node ID should be 0 (0 + 0)
        conn = ConnectionGene(0, 1, 1.0, 0, zero_io_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)
        assert node_id == 0

    def test_initialize_with_large_io_counts(self, large_io_config):
        """Test initialization with large input/output counts."""
        InnovationTracker.initialize(large_io_config)

        # First node ID should be 150 (100 + 50)
        conn = ConnectionGene(0, 1, 1.0, 0, large_io_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)
        assert node_id == 150

    def test_initialize_multiple_times(self, basic_config, single_io_config):
        """Test multiple initializations with different configs."""
        # Initialize with basic_config
        InnovationTracker.initialize(basic_config)
        innov1 = InnovationTracker.get_innovation_number(0, 1)

        # Initialize with different config
        InnovationTracker.initialize(single_io_config)

        # Should restart from 0
        innov2 = InnovationTracker.get_innovation_number(0, 1)
        assert innov2 == 0

        # Node ID should use new config
        conn = ConnectionGene(0, 1, 1.0, 0, single_io_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)
        assert node_id == 2  # 1 input + 1 output


# ============================================================================
# Test: get_innovation_number
# ============================================================================

class TestInnovationTrackerGetInnovationNumber:
    """Test innovation number assignment."""

    def test_sequential_assignment(self, basic_config):
        """Test that innovation numbers are assigned sequentially."""
        InnovationTracker.initialize(basic_config)

        innov0 = InnovationTracker.get_innovation_number(0, 1)
        innov1 = InnovationTracker.get_innovation_number(1, 2)
        innov2 = InnovationTracker.get_innovation_number(2, 3)

        assert innov0 == 0
        assert innov1 == 1
        assert innov2 == 2

    def test_idempotency_same_connection_returns_same_innovation(self, basic_config):
        """Test that same connection always returns same innovation number."""
        InnovationTracker.initialize(basic_config)

        innov1 = InnovationTracker.get_innovation_number(0, 1)
        innov2 = InnovationTracker.get_innovation_number(0, 1)
        innov3 = InnovationTracker.get_innovation_number(0, 1)

        assert innov1 == innov2 == innov3

    def test_different_connections_get_different_innovations(self, basic_config):
        """Test that different connections get different innovation numbers."""
        InnovationTracker.initialize(basic_config)

        innov1 = InnovationTracker.get_innovation_number(0, 1)
        innov2 = InnovationTracker.get_innovation_number(0, 2)
        innov3 = InnovationTracker.get_innovation_number(1, 2)

        assert innov1 != innov2
        assert innov1 != innov3
        assert innov2 != innov3

    def test_order_matters_for_first_assignment(self, basic_config):
        """Test that order of creation affects which innovation number is assigned."""
        InnovationTracker.initialize(basic_config)

        # First call to (0, 1) gets 0
        innov_a = InnovationTracker.get_innovation_number(0, 1)
        assert innov_a == 0

        # First call to (1, 2) gets 1
        innov_b = InnovationTracker.get_innovation_number(1, 2)
        assert innov_b == 1

        # Different order
        InnovationTracker.initialize(basic_config)

        # First call to (1, 2) gets 0
        innov_c = InnovationTracker.get_innovation_number(1, 2)
        assert innov_c == 0

        # First call to (0, 1) gets 1
        innov_d = InnovationTracker.get_innovation_number(0, 1)
        assert innov_d == 1

    def test_with_zero_node_ids(self, basic_config):
        """Test with node IDs of 0."""
        InnovationTracker.initialize(basic_config)

        innov = InnovationTracker.get_innovation_number(0, 0)
        assert innov == 0

    def test_with_negative_node_ids(self, basic_config):
        """Test with negative node IDs."""
        InnovationTracker.initialize(basic_config)

        innov1 = InnovationTracker.get_innovation_number(-1, 5)
        innov2 = InnovationTracker.get_innovation_number(3, -2)

        assert innov1 == 0
        assert innov2 == 1

    def test_with_large_node_ids(self, basic_config):
        """Test with very large node IDs."""
        InnovationTracker.initialize(basic_config)

        innov1 = InnovationTracker.get_innovation_number(999999, 888888)
        innov2 = InnovationTracker.get_innovation_number(777777, 999999)

        assert innov1 == 0
        assert innov2 == 1

    def test_reverse_connection_gets_different_innovation(self, basic_config):
        """Test that (A, B) and (B, A) are treated as different connections."""
        InnovationTracker.initialize(basic_config)

        innov_forward = InnovationTracker.get_innovation_number(0, 1)
        innov_reverse = InnovationTracker.get_innovation_number(1, 0)

        assert innov_forward != innov_reverse

    def test_many_innovations(self, basic_config):
        """Test creating many innovation numbers."""
        InnovationTracker.initialize(basic_config)

        innovations = []
        for i in range(100):
            innov = InnovationTracker.get_innovation_number(i, i + 1)
            innovations.append(innov)

        # Should be sequential from 0 to 99
        assert innovations == list(range(100))


# ============================================================================
# Test: get_split_IDs
# ============================================================================

class TestInnovationTrackerGetSplitIDs:
    """Test connection split ID assignment."""

    def test_new_split_creates_new_ids(self, basic_config):
        """Test that splitting a connection creates new node ID and innovations."""
        InnovationTracker.initialize(basic_config)

        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, innov, basic_config)

        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Node ID should be first hidden node (num_inputs + num_outputs)
        assert node_id == 5  # 3 + 2

        # Innovations should be assigned for the two new connections
        assert isinstance(innov1, int)
        assert isinstance(innov2, int)
        assert innov1 != innov2

    def test_node_ids_start_at_correct_value(self, basic_config):
        """Test that node IDs start at num_inputs + num_outputs."""
        InnovationTracker.initialize(basic_config)

        conn = ConnectionGene(0, 1, 1.0, 0, basic_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)

        assert node_id == 5  # 3 inputs + 2 outputs

    def test_sequential_node_id_assignment(self, basic_config):
        """Test that node IDs are assigned sequentially."""
        InnovationTracker.initialize(basic_config)

        # Create three different connections and split them
        conn1 = ConnectionGene(0, 1, 1.0, 0, basic_config)
        conn2 = ConnectionGene(1, 2, 1.0, 1, basic_config)
        conn3 = ConnectionGene(2, 3, 1.0, 2, basic_config)

        node_id1, _, _ = InnovationTracker.get_split_IDs(conn1)
        node_id2, _, _ = InnovationTracker.get_split_IDs(conn2)
        node_id3, _, _ = InnovationTracker.get_split_IDs(conn3)

        assert node_id1 == 5
        assert node_id2 == 6
        assert node_id3 == 7

    def test_consistency_same_split_returns_same_ids(self, basic_config):
        """Test that splitting the same connection multiple times returns same IDs."""
        InnovationTracker.initialize(basic_config)

        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, innov, basic_config)

        # Split the same connection multiple times
        result1 = InnovationTracker.get_split_IDs(conn)
        result2 = InnovationTracker.get_split_IDs(conn)
        result3 = InnovationTracker.get_split_IDs(conn)

        assert result1 == result2 == result3

    def test_split_creates_new_connection_innovations(self, basic_config):
        """Test that split uses get_innovation_number for new connections."""
        InnovationTracker.initialize(basic_config)

        # Original connection innovation
        orig_innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, orig_innov, basic_config)

        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # innov1 should be for connection (0, node_id)
        # innov2 should be for connection (node_id, 3)
        # They should be new innovation numbers
        assert innov1 != orig_innov
        assert innov2 != orig_innov
        assert innov1 != innov2

    def test_different_connections_split_get_different_node_ids(self, basic_config):
        """Test that different connections get different node IDs when split."""
        InnovationTracker.initialize(basic_config)

        conn1 = ConnectionGene(0, 1, 1.0, 0, basic_config)
        conn2 = ConnectionGene(1, 2, 1.0, 1, basic_config)

        node_id1, _, _ = InnovationTracker.get_split_IDs(conn1)
        node_id2, _, _ = InnovationTracker.get_split_IDs(conn2)

        assert node_id1 != node_id2

    def test_split_with_zero_io_config(self, zero_io_config):
        """Test split with zero inputs/outputs."""
        InnovationTracker.initialize(zero_io_config)

        conn = ConnectionGene(5, 10, 1.0, 0, zero_io_config)
        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # First node ID should be 0
        assert node_id == 0

    def test_split_with_large_io_config(self, large_io_config):
        """Test split with large input/output counts."""
        InnovationTracker.initialize(large_io_config)

        conn = ConnectionGene(0, 1, 1.0, 0, large_io_config)
        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # First node ID should be 150 (100 + 50)
        assert node_id == 150


# ============================================================================
# Test: Integration
# ============================================================================

class TestInnovationTrackerIntegration:
    """Test integration scenarios and interactions between methods."""

    def test_split_innovations_appear_in_innovation_numbers(self, basic_config):
        """Test that innovations from splits are retrievable via get_innovation_number."""
        InnovationTracker.initialize(basic_config)

        orig_innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, orig_innov, basic_config)

        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Calling get_innovation_number for the split connections should return same values
        innov1_check = InnovationTracker.get_innovation_number(0, node_id)
        innov2_check = InnovationTracker.get_innovation_number(node_id, 3)

        assert innov1 == innov1_check
        assert innov2 == innov2_check

    def test_crossover_scenario_two_genomes_same_connection(self, basic_config):
        """Test that two genomes creating the same connection get same innovation."""
        InnovationTracker.initialize(basic_config)

        # Genome 1 creates connection (0, 5)
        genome1_innov = InnovationTracker.get_innovation_number(0, 5)

        # Genome 2 also creates connection (0, 5)
        genome2_innov = InnovationTracker.get_innovation_number(0, 5)

        # Should be identical for crossover alignment
        assert genome1_innov == genome2_innov

    def test_historical_marking_multiple_genomes_split_same_connection(self, basic_config):
        """Test that multiple genomes splitting same connection get same node ID."""
        InnovationTracker.initialize(basic_config)

        # Create original connection
        orig_innov = InnovationTracker.get_innovation_number(0, 3)

        # Genome A splits it
        conn_a = ConnectionGene(0, 3, 1.0, orig_innov, basic_config)
        node_a, innov1_a, innov2_a = InnovationTracker.get_split_IDs(conn_a)

        # Genome B also splits the same connection (same innovation number)
        conn_b = ConnectionGene(0, 3, 1.0, orig_innov, basic_config)
        node_b, innov1_b, innov2_b = InnovationTracker.get_split_IDs(conn_b)

        # All IDs should be identical
        assert node_a == node_b
        assert innov1_a == innov1_b
        assert innov2_a == innov2_b

    def test_mixed_operations_connections_and_splits(self, basic_config):
        """Test interleaved connection and split operations."""
        InnovationTracker.initialize(basic_config)

        # Create some connections
        innov1 = InnovationTracker.get_innovation_number(0, 3)
        innov2 = InnovationTracker.get_innovation_number(1, 4)

        # Split first connection
        conn1 = ConnectionGene(0, 3, 1.0, innov1, basic_config)
        node1, split_innov1, split_innov2 = InnovationTracker.get_split_IDs(conn1)

        # Create another connection
        innov3 = InnovationTracker.get_innovation_number(2, 4)

        # Split second connection
        conn2 = ConnectionGene(1, 4, 1.0, innov2, basic_config)
        node2, split_innov3, split_innov4 = InnovationTracker.get_split_IDs(conn2)

        # Verify sequential assignment
        assert innov1 == 0
        assert innov2 == 1
        assert node1 == 5  # First hidden node
        assert node2 == 6  # Second hidden node
        # split innovations should be after the first two
        assert split_innov1 > innov2
        assert split_innov2 > innov2

    def test_complex_evolution_scenario(self, basic_config):
        """Test a realistic evolution scenario with multiple operations."""
        InnovationTracker.initialize(basic_config)

        # Initial fully connected network: 3 inputs, 2 outputs
        # Create 6 initial connections (3 * 2)
        initial_innovations = []
        for i in range(3):
            for o in range(2):
                innov = InnovationTracker.get_innovation_number(i, 3 + o)
                initial_innovations.append(innov)

        # Should be 0-5
        assert initial_innovations == [0, 1, 2, 3, 4, 5]

        # Add a node by splitting connection 2 (innovation 2)
        conn_to_split = ConnectionGene(1, 3, 1.0, 2, basic_config)
        node1, innov_a, innov_b = InnovationTracker.get_split_IDs(conn_to_split)

        assert node1 == 5  # First hidden node
        assert innov_a == 6  # Next innovations
        assert innov_b == 7

        # Add another connection from hidden node
        innov_new = InnovationTracker.get_innovation_number(node1, 4)
        assert innov_new == 8


# ============================================================================
# Test: State Management
# ============================================================================

class TestInnovationTrackerStateManagement:
    """Test state persistence and management."""

    def test_state_persists_across_calls(self, basic_config):
        """Test that state is maintained across multiple calls."""
        InnovationTracker.initialize(basic_config)

        innov1 = InnovationTracker.get_innovation_number(0, 1)
        innov2 = InnovationTracker.get_innovation_number(0, 2)

        # Call again - should return cached values
        innov1_again = InnovationTracker.get_innovation_number(0, 1)
        innov2_again = InnovationTracker.get_innovation_number(0, 2)

        assert innov1 == innov1_again
        assert innov2 == innov2_again

    def test_reset_clears_all_state(self, basic_config):
        """Test that initialize() completely resets state."""
        InnovationTracker.initialize(basic_config)

        # Create some state
        innov1 = InnovationTracker.get_innovation_number(0, 1)
        conn = ConnectionGene(0, 1, 1.0, innov1, basic_config)
        node1, _, _ = InnovationTracker.get_split_IDs(conn)

        # Reset
        InnovationTracker.initialize(basic_config)

        # Same operations should give same results (starting fresh)
        innov2 = InnovationTracker.get_innovation_number(0, 1)
        conn2 = ConnectionGene(0, 1, 1.0, innov2, basic_config)
        node2, _, _ = InnovationTracker.get_split_IDs(conn2)

        assert innov2 == 0  # Restarted from 0
        assert node2 == 5  # Restarted from num_inputs + num_outputs

    def test_counters_increment_correctly(self, basic_config):
        """Test that internal counters increment properly."""
        InnovationTracker.initialize(basic_config)

        innovations = []
        for i in range(10):
            innov = InnovationTracker.get_innovation_number(i, i + 100)
            innovations.append(innov)

        # Should be strictly increasing
        for i in range(len(innovations) - 1):
            assert innovations[i + 1] == innovations[i] + 1

    def test_dictionary_grows_with_unique_connections(self, basic_config):
        """Test that dictionaries grow with unique operations."""
        InnovationTracker.initialize(basic_config)

        # Create many unique connections
        for i in range(50):
            InnovationTracker.get_innovation_number(i, i + 1)

        # Each unique connection should be stored
        # We can verify by getting them again and checking they're all different
        innovations = []
        for i in range(50):
            innov = InnovationTracker.get_innovation_number(i, i + 1)
            innovations.append(innov)

        # All should be unique
        assert len(set(innovations)) == 50


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestInnovationTrackerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_calling_before_initialize_raises_error(self):
        """Test that calling methods before initialize raises TypeError."""
        # Explicitly set counters to None and clear dicts to simulate uninitialized state
        InnovationTracker._next_innovation_number = None
        InnovationTracker._next_node_id = None
        InnovationTracker._innovation_numbers = {}
        InnovationTracker._split_IDs = {}

        with pytest.raises(TypeError):
            # This should raise TypeError when trying to call next(None)
            InnovationTracker.get_innovation_number(999999, 888888)

    def test_self_loop_connection(self, basic_config):
        """Test with same node_in and node_out (self-loop)."""
        InnovationTracker.initialize(basic_config)

        innov = InnovationTracker.get_innovation_number(5, 5)
        assert innov == 0

        # Should be retrievable
        innov_again = InnovationTracker.get_innovation_number(5, 5)
        assert innov_again == innov

    def test_node_id_no_overlap_with_io_nodes(self, basic_config):
        """Test that hidden node IDs don't overlap with input/output IDs."""
        InnovationTracker.initialize(basic_config)

        # Input nodes: 0, 1, 2
        # Output nodes: 3, 4
        # First hidden should be 5

        conn = ConnectionGene(0, 3, 1.0, 0, basic_config)
        node_id, _, _ = InnovationTracker.get_split_IDs(conn)

        assert node_id >= basic_config.num_inputs + basic_config.num_outputs

    def test_very_large_innovation_numbers(self, basic_config):
        """Test with very large innovation numbers."""
        InnovationTracker.initialize(basic_config)

        # Create many innovations
        for i in range(1000):
            innov = InnovationTracker.get_innovation_number(i, i + 10000)

        # 1000th innovation should be 999
        final_innov = InnovationTracker.get_innovation_number(999, 10999)
        assert final_innov == 999

    def test_split_connection_with_high_innovation_number(self, basic_config):
        """Test splitting a connection that has a high innovation number."""
        InnovationTracker.initialize(basic_config)

        # Create many innovations first
        for i in range(100):
            InnovationTracker.get_innovation_number(i, i + 1000)

        # Now split a connection with high innovation
        high_innov = 50
        conn = ConnectionGene(50, 1050, 1.0, high_innov, basic_config)
        node_id, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Should still work correctly
        assert isinstance(node_id, int)
        assert isinstance(innov1, int)
        assert isinstance(innov2, int)

    def test_split_already_split_connection(self, basic_config):
        """Test the behavior of splitting the same connection twice."""
        InnovationTracker.initialize(basic_config)

        innov = InnovationTracker.get_innovation_number(0, 3)
        conn = ConnectionGene(0, 3, 1.0, innov, basic_config)

        # First split
        node1, innov1, innov2 = InnovationTracker.get_split_IDs(conn)

        # Second split of same connection
        node2, innov3, innov4 = InnovationTracker.get_split_IDs(conn)

        # Should return same results
        assert node1 == node2
        assert innov1 == innov3
        assert innov2 == innov4


# ============================================================================
# Test: Consistency
# ============================================================================

class TestInnovationTrackerConsistency:
    """Test consistency guarantees and invariants."""

    def test_determinism_same_operations_same_results(self, basic_config):
        """Test that same sequence of operations produces same results."""
        # Run 1
        InnovationTracker.initialize(basic_config)
        innovations1 = []
        for i in range(10):
            innov = InnovationTracker.get_innovation_number(i, i + 10)
            innovations1.append(innov)

        # Run 2
        InnovationTracker.initialize(basic_config)
        innovations2 = []
        for i in range(10):
            innov = InnovationTracker.get_innovation_number(i, i + 10)
            innovations2.append(innov)

        assert innovations1 == innovations2

    def test_innovation_numbers_monotonically_increasing(self, basic_config):
        """Test that new innovation numbers always increase."""
        InnovationTracker.initialize(basic_config)

        previous = -1
        for i in range(100):
            innov = InnovationTracker.get_innovation_number(i, i + 1000)
            assert innov > previous
            previous = innov

    def test_node_ids_monotonically_increasing(self, basic_config):
        """Test that new node IDs always increase."""
        InnovationTracker.initialize(basic_config)

        previous = basic_config.num_inputs + basic_config.num_outputs - 1
        for i in range(50):
            conn = ConnectionGene(i, i + 100, 1.0, i, basic_config)
            node_id, _, _ = InnovationTracker.get_split_IDs(conn)
            assert node_id > previous
            previous = node_id

    def test_dictionary_keys_immutable(self, basic_config):
        """Test that dictionary keys (tuples) are used correctly."""
        InnovationTracker.initialize(basic_config)

        # Create innovation
        innov1 = InnovationTracker.get_innovation_number(0, 5)

        # Retrieve with same tuple
        innov2 = InnovationTracker.get_innovation_number(0, 5)

        assert innov1 == innov2

    def test_split_uses_innovation_as_key(self, basic_config):
        """Test that split uses connection innovation as key, not endpoints."""
        InnovationTracker.initialize(basic_config)

        # Two connections with same endpoints but different innovations
        # (shouldn't happen in practice, but tests the key mechanism)
        innov1 = InnovationTracker.get_innovation_number(0, 5)
        innov2 = InnovationTracker.get_innovation_number(0, 5)  # Same, so same innov

        # Both should be the same
        assert innov1 == innov2

        # Split using this connection
        conn = ConnectionGene(0, 5, 1.0, innov1, basic_config)
        result1 = InnovationTracker.get_split_IDs(conn)
        result2 = InnovationTracker.get_split_IDs(conn)

        # Same innovation = same split results
        assert result1 == result2

    def test_no_innovation_collision(self, basic_config):
        """Test that different connections never get same innovation."""
        InnovationTracker.initialize(basic_config)

        innovations = set()
        for i in range(100):
            for j in range(100):
                if i != j:  # Avoid duplicate keys
                    innov = InnovationTracker.get_innovation_number(i, j)
                    assert innov not in innovations
                    innovations.add(innov)
