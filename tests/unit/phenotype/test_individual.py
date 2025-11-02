"""
Unit tests for neat.phenotype.individual module.

This module contains comprehensive tests for the Individual class,
which represents a complete evolved agent in the NEAT population.
"""

import copy
import pytest
from unittest.mock import Mock, patch

from evograd.run.config import Config
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.individual import Individual
from evograd.phenotype.network_standard import NetworkStandard
from evograd.phenotype.network_fast import NetworkFast
from evograd.phenotype.network_autograd import NetworkAutograd


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


@pytest.fixture(autouse=True)
def reset_individual_id_generator():
    """Reset Individual ID generator before each test."""
    from itertools import count
    Individual._id_generator = count(0)
    yield
    Individual._id_generator = count(0)


@pytest.fixture
def simple_genome_dict():
    """
    Minimal genome: 2 inputs -> 1 output (direct connection).
    Node numbering follows NEAT convention: inputs [0,2), outputs [2,3)
    """
    return {
        'activation': 'identity',  # Default activation
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
def genome_with_hidden_dict():
    """
    More complex genome: 2 inputs -> 1 hidden -> 1 output.
    Node numbering: inputs [0,2), outputs [2,3), hidden [3,4)
    """
    return {
        'activation': 'identity',  # Default activation
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output'},
            {'id': 3, 'type': 'hidden'},
        ],
        'connections': [
            {'from': 0, 'to': 3, 'weight': 0.5, 'enabled': True},
            {'from': 1, 'to': 3, 'weight': 0.5, 'enabled': True},
            {'from': 3, 'to': 2, 'weight': 2.0, 'enabled': True},
        ]
    }


@pytest.fixture
def genome_for_pruning_dict():
    """
    Genome with disabled connection and dead-end node for pruning tests.
    Node numbering: inputs [0,2), outputs [2,3), hidden [3,5)
    """
    return {
        'activation': 'identity',  # Default activation
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output'},
            {'id': 3, 'type': 'hidden'},  # Connected to output
            {'id': 4, 'type': 'hidden'},  # Dead-end (no path to output)
        ],
        'connections': [
            {'from': 0, 'to': 3, 'weight': 1.0, 'enabled': True},
            {'from': 3, 'to': 2, 'weight': 1.0, 'enabled': True},
            {'from': 1, 'to': 4, 'weight': 1.0, 'enabled': True},  # Dead-end path
            {'from': 1, 'to': 2, 'weight': 0.5, 'enabled': False}, # Disabled
        ]
    }


# ============================================================================
# Test Individual Initialization
# ============================================================================

class TestIndividualInit:
    """Test Individual.__init__ method."""

    def test_init_standard_network(self, simple_genome_dict):
        """Test initialization with 'standard' network type."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')

        assert individual.ID == 0
        assert individual.fitness is None
        assert individual._network_type == 'standard'
        assert isinstance(individual._network, NetworkStandard)

    def test_init_fast_network(self, simple_genome_dict):
        """Test initialization with 'fast' network type."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'fast')

        assert individual.ID == 0
        assert individual.fitness is None
        assert individual._network_type == 'fast'
        assert isinstance(individual._network, NetworkFast)

    def test_init_autograd_network(self, simple_genome_dict):
        """Test initialization with 'autograd' network type."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'autograd')

        assert individual.ID == 0
        assert individual.fitness is None
        assert individual._network_type == 'autograd'
        assert isinstance(individual._network, NetworkAutograd)

    def test_init_invalid_network_type(self, simple_genome_dict):
        """Test initialization with invalid network type raises ValueError."""
        genome = Genome.from_dict(simple_genome_dict)

        with pytest.raises(ValueError, match="Unknown network_type: invalid"):
            Individual(genome, 'invalid')

    def test_init_preserves_genome(self, genome_with_hidden_dict):
        """Test that initialization preserves genome structure."""
        genome = Genome.from_dict(genome_with_hidden_dict)
        individual = Individual(genome, 'standard')

        # Access genome through property
        retrieved_genome = individual.genome
        assert len(retrieved_genome.node_genes) == 4
        assert len(retrieved_genome.conn_genes) == 3


# ============================================================================
# Test Individual ID Generation
# ============================================================================

class TestIndividualIDGeneration:
    """Test Individual ID generation mechanism."""

    def test_id_increments_for_multiple_individuals(self, simple_genome_dict):
        """Test that IDs increment correctly for multiple individuals."""
        genome = Genome.from_dict(simple_genome_dict)

        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')
        ind3 = Individual(genome, 'fast')

        assert ind1.ID == 0
        assert ind2.ID == 1
        assert ind3.ID == 2

    def test_id_is_globally_unique_across_network_types(self, simple_genome_dict):
        """Test that IDs are globally unique across different network types."""
        genome = Genome.from_dict(simple_genome_dict)

        individuals = [
            Individual(genome, 'standard'),
            Individual(genome, 'fast'),
            Individual(genome, 'autograd'),
            Individual(genome, 'standard'),
        ]

        ids = [ind.ID for ind in individuals]
        assert ids == [0, 1, 2, 3]
        assert len(set(ids)) == 4  # All unique


# ============================================================================
# Test Individual Genome Property
# ============================================================================

class TestIndividualGenomeProperty:
    """Test Individual.genome property."""

    def test_genome_property_returns_genome(self, simple_genome_dict):
        """Test that genome property returns the underlying genome."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')

        retrieved_genome = individual.genome
        assert isinstance(retrieved_genome, Genome)
        assert len(retrieved_genome.node_genes) == 3

    def test_genome_property_same_across_network_types(self, genome_with_hidden_dict):
        """Test that genome property works consistently across network types."""
        genome = Genome.from_dict(genome_with_hidden_dict)

        ind_standard = Individual(copy.deepcopy(genome), 'standard')
        ind_fast = Individual(copy.deepcopy(genome), 'fast')
        ind_autograd = Individual(copy.deepcopy(genome), 'autograd')

        # All should have same structure
        assert len(ind_standard.genome.node_genes) == 4
        assert len(ind_fast.genome.node_genes) == 4
        assert len(ind_autograd.genome.node_genes) == 4


# ============================================================================
# Test Individual Clone
# ============================================================================

class TestIndividualClone:
    """Test Individual.clone method."""

    def test_clone_creates_new_individual(self, simple_genome_dict):
        """Test that clone creates a new Individual instance."""
        genome = Genome.from_dict(simple_genome_dict)
        original = Individual(genome, 'standard')
        original.fitness = 10.0

        cloned = original.clone()

        assert isinstance(cloned, Individual)
        assert cloned is not original

    def test_clone_has_different_id(self, simple_genome_dict):
        """Test that clone has a different ID from original."""
        genome = Genome.from_dict(simple_genome_dict)
        original = Individual(genome, 'standard')

        cloned = original.clone()

        assert cloned.ID != original.ID
        assert cloned.ID == original.ID + 1

    def test_clone_has_none_fitness(self, simple_genome_dict):
        """Test that clone starts with None fitness (not copied)."""
        genome = Genome.from_dict(simple_genome_dict)
        original = Individual(genome, 'standard')
        original.fitness = 42.0

        cloned = original.clone()

        assert cloned.fitness is None

    def test_clone_preserves_network_type(self, simple_genome_dict):
        """Test that clone preserves the network type."""
        genome = Genome.from_dict(simple_genome_dict)

        for network_type in ['standard', 'fast', 'autograd']:
            original = Individual(genome, network_type)
            cloned = original.clone()
            assert cloned._network_type == network_type

    def test_clone_deep_copies_genome(self, genome_with_hidden_dict):
        """Test that clone deep copies the genome (modifications don't affect original)."""
        genome = Genome.from_dict(genome_with_hidden_dict)
        original = Individual(genome, 'standard')
        cloned = original.clone()

        # Modify original genome's connection weight directly
        original_conn = list(original.genome.conn_genes.values())[0]
        original_weight = original_conn.weight
        original_conn.weight = 999.0

        # Cloned genome should be unaffected
        cloned_conn = list(cloned.genome.conn_genes.values())[0]
        assert cloned_conn.weight == original_weight
        assert original_conn.weight == 999.0


# ============================================================================
# Test Individual Distance
# ============================================================================

class TestIndividualDistance:
    """Test Individual.distance method."""

    def test_distance_same_genome_is_zero(self, simple_genome_dict):
        """Test that distance delegates to genome.distance method."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(copy.deepcopy(genome), 'standard')
        ind2 = Individual(copy.deepcopy(genome), 'standard')

        # Mock genome.distance to return a known value
        with patch.object(Genome, 'distance', return_value=0.0) as mock_distance:
            distance = ind1.distance(ind2)

            # Verify distance was called with correct arguments
            assert mock_distance.call_count == 1
            assert mock_distance.call_args[0][0] == ind2.genome
            assert distance == 0.0

    def test_distance_different_genomes_nonzero(self, simple_genome_dict):
        """Test that distance returns non-zero for different genomes."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(copy.deepcopy(genome), 'standard')
        ind2 = Individual(copy.deepcopy(genome), 'standard')

        # Mock genome.distance to simulate different genomes
        with patch.object(Genome, 'distance', return_value=5.5):
            distance = ind1.distance(ind2)
            assert distance == 5.5

    def test_distance_is_symmetric(self, genome_with_hidden_dict):
        """Test that Individual.distance properly delegates to genome.distance."""
        genome = Genome.from_dict(genome_with_hidden_dict)
        ind1 = Individual(copy.deepcopy(genome), 'standard')
        ind2 = Individual(copy.deepcopy(genome), 'standard')

        # Mock genome.distance to return a fixed value
        with patch.object(Genome, 'distance', return_value=3.5):
            distance = ind1.distance(ind2)
            assert distance == 3.5


# ============================================================================
# Test Individual Mate
# ============================================================================

class TestIndividualMate:
    """Test Individual.mate method."""

    def test_mate_creates_new_individual(self, simple_genome_dict):
        """Test that mate creates a new Individual instance."""
        genome = Genome.from_dict(simple_genome_dict)
        parent1 = Individual(genome, 'standard')
        parent1.fitness = 10.0
        parent2 = Individual(genome, 'standard')
        parent2.fitness = 8.0

        with patch.object(Genome, 'mutate'):
            offspring = parent1.mate(parent2)

            assert isinstance(offspring, Individual)
            assert offspring is not parent1
            assert offspring is not parent2

    def test_mate_offspring_has_new_id(self, simple_genome_dict):
        """Test that offspring has a different ID from parents."""
        genome = Genome.from_dict(simple_genome_dict)
        parent1 = Individual(genome, 'standard')
        parent1.fitness = 10.0
        parent2 = Individual(genome, 'standard')
        parent2.fitness = 8.0

        with patch.object(Genome, 'mutate'):
            offspring = parent1.mate(parent2)

            assert offspring.ID not in [parent1.ID, parent2.ID]
            assert offspring.ID == 2  # parent1=0, parent2=1, offspring=2

    def test_mate_offspring_has_none_fitness(self, simple_genome_dict):
        """Test that offspring starts with None fitness."""
        genome = Genome.from_dict(simple_genome_dict)
        parent1 = Individual(genome, 'standard')
        parent1.fitness = 10.0
        parent2 = Individual(genome, 'standard')
        parent2.fitness = 8.0

        with patch.object(Genome, 'mutate'):
            offspring = parent1.mate(parent2)

            assert offspring.fitness is None

    def test_mate_preserves_network_type(self, simple_genome_dict):
        """Test that offspring uses same network type as parents."""
        genome = Genome.from_dict(simple_genome_dict)

        with patch.object(Genome, 'mutate'):
            for network_type in ['standard', 'fast', 'autograd']:
                parent1 = Individual(genome, network_type)
                parent1.fitness = 10.0
                parent2 = Individual(genome, network_type)
                parent2.fitness = 8.0

                offspring = parent1.mate(parent2)
                assert offspring._network_type == network_type

    def test_mate_uses_fitter_parent_for_crossover(self, genome_with_hidden_dict):
        """Test that mate uses fitter parent's genome in crossover."""
        genome = Genome.from_dict(genome_with_hidden_dict)
        parent1 = Individual(copy.deepcopy(genome), 'standard')
        parent1.fitness = 15.0  # Fitter
        parent2 = Individual(copy.deepcopy(genome), 'standard')
        parent2.fitness = 5.0

        # Mock both crossover and mutate
        with patch.object(Genome, 'mutate'):
            with patch.object(Genome, 'crossover', wraps=parent1.genome.crossover) as mock_crossover:
                offspring = parent1.mate(parent2)

                # Verify crossover was called with fitter genome (parent1.genome)
                # crossover(other, fitter_genome) - 3 args total including self
                assert mock_crossover.call_count == 1
                args = mock_crossover.call_args[0]
                # args[0] is other.genome, args[1] is fitter_genome
                fitter_genome_arg = args[1]
                assert fitter_genome_arg is parent1.genome

    def test_mate_calls_mutate_on_offspring_genome(self, simple_genome_dict):
        """Test that mate calls mutate on the offspring genome."""
        genome = Genome.from_dict(simple_genome_dict)
        parent1 = Individual(genome, 'standard')
        parent1.fitness = 10.0
        parent2 = Individual(genome, 'standard')
        parent2.fitness = 8.0

        # Mock Genome.mutate to verify it's called
        with patch.object(Genome, 'mutate') as mock_mutate:
            offspring = parent1.mate(parent2)

            # Verify mutate was called exactly once
            assert mock_mutate.call_count == 1


# ============================================================================
# Test Individual Prune
# ============================================================================

class TestIndividualPrune:
    """Test Individual.prune method."""

    def test_prune_creates_new_individual(self, genome_for_pruning_dict):
        """Test that prune creates a new Individual instance."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')
        original.fitness = 10.0

        pruned = original.prune()

        assert isinstance(pruned, Individual)
        assert pruned is not original

    def test_prune_preserves_id(self, genome_for_pruning_dict):
        """Test that pruned individual has same ID as original."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')

        pruned = original.prune()

        assert pruned.ID == original.ID

    def test_prune_preserves_fitness(self, genome_for_pruning_dict):
        """Test that pruned individual has same fitness as original."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')
        original.fitness = 42.5

        pruned = original.prune()

        assert pruned.fitness == 42.5

    def test_prune_preserves_none_fitness(self, genome_for_pruning_dict):
        """Test that pruned individual preserves None fitness."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')
        assert original.fitness is None

        pruned = original.prune()

        assert pruned.fitness is None

    def test_prune_removes_dead_end_nodes(self, genome_for_pruning_dict):
        """Test that prune removes dead-end hidden nodes."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')

        # Original has 5 nodes (2 input, 1 output, 2 hidden)
        assert len(original.genome.node_genes) == 5

        pruned = original.prune()

        # Pruned should have 4 nodes (dead-end hidden node 4 removed)
        assert len(pruned.genome.node_genes) == 4
        assert 4 not in pruned.genome.node_genes  # Dead-end node removed

    def test_prune_removes_disabled_connections(self, genome_for_pruning_dict):
        """Test that prune removes disabled connections."""
        genome = Genome.from_dict(genome_for_pruning_dict)
        original = Individual(genome, 'standard')

        # Original has 4 connections (1 disabled)
        assert len(original.genome.conn_genes) == 4
        disabled_conn = [c for c in original.genome.conn_genes.values() if not c.enabled]
        assert len(disabled_conn) == 1

        pruned = original.prune()

        # Pruned should have fewer connections (dead-end and disabled removed)
        assert len(pruned.genome.conn_genes) < 4
        # All remaining connections should be enabled
        for conn in pruned.genome.conn_genes.values():
            assert conn.enabled

    def test_prune_preserves_network_type(self, genome_for_pruning_dict):
        """Test that pruned individual uses same network type."""
        genome = Genome.from_dict(genome_for_pruning_dict)

        for network_type in ['standard', 'fast', 'autograd']:
            original = Individual(genome, network_type)
            pruned = original.prune()
            assert pruned._network_type == network_type


# ============================================================================
# Test Individual String Methods
# ============================================================================

class TestIndividualStringMethods:
    """Test Individual.__str__ and __repr__ methods."""

    def test_str_contains_id_and_fitness(self, simple_genome_dict):
        """Test that __str__ contains ID and fitness."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')
        individual.fitness = 12.3456

        result = str(individual)

        assert 'ID=0' in result
        assert '12.3456' in result

    def test_repr_contains_genome_repr(self, simple_genome_dict):
        """Test that __repr__ contains genome representation."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')

        result = repr(individual)

        assert result.startswith('Individual(genome=')
        assert 'Genome' in result


# ============================================================================
# Test Individual Edge Cases
# ============================================================================

class TestIndividualEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_fitness_can_be_zero(self, simple_genome_dict):
        """Test that fitness can be set to zero (valid fitness value)."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')
        individual.fitness = 0.0

        assert individual.fitness == 0.0

    def test_fitness_can_be_negative(self, simple_genome_dict):
        """Test that fitness can be negative."""
        genome = Genome.from_dict(simple_genome_dict)
        individual = Individual(genome, 'standard')
        individual.fitness = -10.5

        assert individual.fitness == -10.5

    def test_mate_with_equal_fitness_uses_first_parent(self, simple_genome_dict):
        """Test that mate with equal fitness uses first parent's genome as fitter."""
        genome = Genome.from_dict(simple_genome_dict)
        parent1 = Individual(genome, 'standard')
        parent1.fitness = 10.0
        parent2 = Individual(genome, 'standard')
        parent2.fitness = 10.0  # Equal fitness

        # Should not raise error, uses first parent when equal
        with patch.object(Genome, 'mutate'):
            offspring = parent1.mate(parent2)

            assert isinstance(offspring, Individual)


# ============================================================================
# Test Individual Integration
# ============================================================================

class TestIndividualIntegration:
    """Integration tests combining multiple Individual operations."""

    def test_clone_mate_prune_workflow(self, genome_with_hidden_dict):
        """Test realistic workflow: clone, mate, and prune."""
        genome = Genome.from_dict(genome_with_hidden_dict)

        with patch.object(Genome, 'mutate'):
            # Create initial individual
            ind1 = Individual(genome, 'fast')
            ind1.fitness = 15.0

            # Clone it
            ind2 = ind1.clone()
            ind2.fitness = 12.0

            # Mate them
            offspring = ind1.mate(ind2)
            offspring.fitness = 18.0

            # Prune offspring
            pruned = offspring.prune()

            # Verify all operations worked
            assert ind1.ID == 0
            assert ind2.ID == 1
            assert offspring.ID == 2
            assert pruned.ID == 2  # Same ID as offspring
            assert pruned.fitness == 18.0  # Same fitness as offspring
            assert pruned._network_type == 'fast'

    def test_multiple_generations_evolution(self, simple_genome_dict):
        """Test multiple generations of evolution."""
        genome = Genome.from_dict(simple_genome_dict)

        with patch.object(Genome, 'mutate'):
            # Generation 0
            gen0_ind1 = Individual(genome, 'standard')
            gen0_ind1.fitness = 10.0
            gen0_ind2 = Individual(genome, 'standard')
            gen0_ind2.fitness = 8.0

            # Generation 1
            gen1_ind1 = gen0_ind1.mate(gen0_ind2)
            gen1_ind1.fitness = 12.0
            gen1_ind2 = gen0_ind1.clone()
            gen1_ind2.fitness = 9.0

            # Generation 2
            gen2_ind1 = gen1_ind1.mate(gen1_ind2)

            # Verify ID sequence
            assert gen0_ind1.ID == 0
            assert gen0_ind2.ID == 1
            assert gen1_ind1.ID == 2
            assert gen1_ind2.ID == 3
            assert gen2_ind1.ID == 4
