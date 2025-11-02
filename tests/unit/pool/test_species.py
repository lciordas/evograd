"""
Unit tests for neat.pool.species module.

This module contains comprehensive tests for the Species class,
which represents a cluster of genetically similar individuals in NEAT.
"""

import pytest
import autograd.numpy as np
from unittest.mock import Mock, patch
from copy import deepcopy

from evograd.run.config import Config
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.individual import Individual
from evograd.pool.species import Species


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
    Minimal genome: 2 inputs -> 1 output.
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
def mock_config():
    """Create a mock Config object with common species parameters."""
    config = Mock(spec=Config)
    config.elitism = 2
    config.survival_threshold = 0.5
    config.max_stagnation_period = 15
    config.min_weight = -5.0
    config.max_weight = 5.0
    config.min_bias = -5.0
    config.max_bias = 5.0
    config.min_gain = 0.1
    config.max_gain = 5.0
    return config


@pytest.fixture
def sample_individual(simple_genome_dict):
    """Create a sample individual for testing."""
    genome = Genome.from_dict(simple_genome_dict)
    individual = Individual(genome, 'standard')
    individual.fitness = 10.0
    return individual


# ============================================================================
# Test Species Initialization
# ============================================================================

class TestSpeciesInit:
    """Test Species.__init__ method."""

    def test_init_sets_basic_attributes(self, sample_individual, mock_config):
        """Test that initialization sets basic attributes correctly."""
        species = Species(1, sample_individual, mock_config)

        assert species.id == 1
        assert species.representative is sample_individual
        assert species._config is mock_config

    def test_init_creates_members_dict_with_representative(self, sample_individual, mock_config):
        """Test that initialization adds representative to members."""
        species = Species(1, sample_individual, mock_config)

        assert len(species.members) == 1
        assert sample_individual.ID in species.members
        assert species.members[sample_individual.ID] is sample_individual

    def test_init_sets_age_and_last_improved_to_zero(self, sample_individual, mock_config):
        """Test that age and last_improved start at 0."""
        species = Species(1, sample_individual, mock_config)

        assert species.age == 0
        assert species.last_improved == 0

    def test_init_sets_fitness_to_none(self, sample_individual, mock_config):
        """Test that fitness starts as None."""
        species = Species(1, sample_individual, mock_config)

        assert species.fitness is None

    def test_init_sets_max_fitness_to_negative_infinity(self, sample_individual, mock_config):
        """Test that max_fitness starts at negative infinity."""
        species = Species(1, sample_individual, mock_config)

        assert species.max_fitness == -np.inf

    def test_init_creates_empty_fitness_history(self, sample_individual, mock_config):
        """Test that fitness_history starts as empty list."""
        species = Species(1, sample_individual, mock_config)

        assert species.fitness_history == []
        assert isinstance(species.fitness_history, list)


# ============================================================================
# Test Species init_for_next_generation
# ============================================================================

class TestSpeciesInitForNextGeneration:
    """Test Species.init_for_next_generation method."""

    def test_init_for_next_generation_updates_representative(self, simple_genome_dict, mock_config):
        """Test that method updates representative."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')

        species = Species(1, ind1, mock_config)
        species.init_for_next_generation(ind2)

        assert species.representative is ind2

    def test_init_for_next_generation_resets_members(self, simple_genome_dict, mock_config):
        """Test that method resets members to only new representative."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')
        ind3 = Individual(genome, 'standard')

        species = Species(1, ind1, mock_config)
        # Add more members
        species.members[ind2.ID] = ind2

        species.init_for_next_generation(ind3)

        assert len(species.members) == 1
        assert ind3.ID in species.members
        assert ind1.ID not in species.members
        assert ind2.ID not in species.members

    def test_init_for_next_generation_increments_age(self, sample_individual, mock_config):
        """Test that method increments species age."""
        species = Species(1, sample_individual, mock_config)
        initial_age = species.age

        species.init_for_next_generation(sample_individual)

        assert species.age == initial_age + 1

    def test_init_for_next_generation_resets_fitness_to_none(self, sample_individual, mock_config):
        """Test that method resets fitness to None."""
        species = Species(1, sample_individual, mock_config)
        species.fitness = 42.0

        species.init_for_next_generation(sample_individual)

        assert species.fitness is None

    def test_init_for_next_generation_preserves_max_fitness(self, sample_individual, mock_config):
        """Test that method preserves max_fitness."""
        species = Species(1, sample_individual, mock_config)
        species.max_fitness = 100.0

        species.init_for_next_generation(sample_individual)

        assert species.max_fitness == 100.0

    def test_init_for_next_generation_preserves_fitness_history(self, sample_individual, mock_config):
        """Test that method preserves fitness_history."""
        species = Species(1, sample_individual, mock_config)
        species.fitness_history = [10.0, 15.0, 20.0]

        species.init_for_next_generation(sample_individual)

        assert species.fitness_history == [10.0, 15.0, 20.0]


# ============================================================================
# Test Species update_fitness
# ============================================================================

class TestSpeciesUpdateFitness:
    """Test Species.update_fitness method."""

    def test_update_fitness_sets_fitness(self, sample_individual, mock_config):
        """Test that update_fitness sets the fitness value."""
        species = Species(1, sample_individual, mock_config)

        species.update_fitness(25.0)

        assert species.fitness == 25.0

    def test_update_fitness_updates_max_fitness_when_higher(self, sample_individual, mock_config):
        """Test that update_fitness updates max_fitness when new fitness is higher."""
        species = Species(1, sample_individual, mock_config)
        species.max_fitness = 10.0

        species.update_fitness(20.0)

        assert species.max_fitness == 20.0

    def test_update_fitness_preserves_max_fitness_when_lower(self, sample_individual, mock_config):
        """Test that update_fitness preserves max_fitness when new fitness is lower."""
        species = Species(1, sample_individual, mock_config)
        species.max_fitness = 50.0

        species.update_fitness(20.0)

        assert species.max_fitness == 50.0

    def test_update_fitness_updates_last_improved_when_new_max(self, sample_individual, mock_config):
        """Test that last_improved is updated when reaching new max fitness."""
        species = Species(1, sample_individual, mock_config)
        species.age = 5
        species.max_fitness = 10.0

        species.update_fitness(20.0)

        assert species.last_improved == 5

    def test_update_fitness_does_not_update_last_improved_when_not_max(self, sample_individual, mock_config):
        """Test that last_improved is not updated when not reaching new max."""
        species = Species(1, sample_individual, mock_config)
        species.age = 10
        species.last_improved = 3
        species.max_fitness = 50.0

        species.update_fitness(40.0)

        assert species.last_improved == 3  # Unchanged

    def test_update_fitness_appends_to_history(self, sample_individual, mock_config):
        """Test that update_fitness appends to fitness_history."""
        species = Species(1, sample_individual, mock_config)

        species.update_fitness(10.0)
        species.update_fitness(15.0)
        species.update_fitness(12.0)

        assert species.fitness_history == [10.0, 15.0, 12.0]


# ============================================================================
# Test Species is_stagnant
# ============================================================================

class TestSpeciesIsStagnant:
    """Test Species.is_stagnant method."""

    def test_is_stagnant_returns_false_when_recently_improved(self, sample_individual, mock_config):
        """Test that is_stagnant returns False when species improved recently."""
        mock_config.max_stagnation_period = 15
        species = Species(1, sample_individual, mock_config)
        species.age = 10
        species.last_improved = 5  # Improved 5 generations ago

        assert not species.is_stagnant()

    def test_is_stagnant_returns_true_when_stagnant(self, sample_individual, mock_config):
        """Test that is_stagnant returns True when species hasn't improved."""
        mock_config.max_stagnation_period = 15
        species = Species(1, sample_individual, mock_config)
        species.age = 20
        species.last_improved = 0  # Improved 20 generations ago

        assert species.is_stagnant()

    def test_is_stagnant_boundary_condition_exactly_at_threshold(self, sample_individual, mock_config):
        """Test is_stagnant at exact threshold boundary."""
        mock_config.max_stagnation_period = 15
        species = Species(1, sample_individual, mock_config)
        species.age = 15
        species.last_improved = 0  # Exactly 15 generations ago

        # age - last_improved = 15, threshold is 15
        # 15 > 15 is False, so not stagnant
        assert not species.is_stagnant()

    def test_is_stagnant_boundary_condition_just_over_threshold(self, sample_individual, mock_config):
        """Test is_stagnant just over threshold boundary."""
        mock_config.max_stagnation_period = 15
        species = Species(1, sample_individual, mock_config)
        species.age = 16
        species.last_improved = 0  # 16 generations ago

        # age - last_improved = 16, threshold is 15
        # 16 > 15 is True, so stagnant
        assert species.is_stagnant()


# ============================================================================
# Test Species distance_to
# ============================================================================

class TestSpeciesDistanceTo:
    """Test Species.distance_to method."""

    def test_distance_to_delegates_to_representative(self, simple_genome_dict, mock_config):
        """Test that distance_to delegates to representative.distance."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')

        species = Species(1, ind1, mock_config)

        # Mock the representative's distance method
        with patch.object(Individual, 'distance', return_value=5.5) as mock_distance:
            result = species.distance_to(ind2)

            mock_distance.assert_called_once_with(ind2)
            assert result == 5.5

    def test_distance_to_returns_numeric_value(self, simple_genome_dict, mock_config):
        """Test that distance_to returns a numeric value."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')

        species = Species(1, ind1, mock_config)

        with patch.object(Individual, 'distance', return_value=3.2):
            result = species.distance_to(ind2)

            assert isinstance(result, (int, float))


# ============================================================================
# Test Species spawn
# ============================================================================

class TestSpeciesSpawn:
    """Test Species.spawn method."""

    def test_spawn_returns_empty_list_when_zero_offspring(self, sample_individual, mock_config):
        """Test that spawn returns empty list when num_offspring is 0."""
        species = Species(1, sample_individual, mock_config)

        offspring = species.spawn(0)

        assert offspring == []

    def test_spawn_returns_empty_list_when_no_members(self, sample_individual, mock_config):
        """Test that spawn returns empty list when species has no members."""
        species = Species(1, sample_individual, mock_config)
        species.members = {}  # Empty members

        offspring = species.spawn(5)

        assert offspring == []

    def test_spawn_applies_elitism(self, simple_genome_dict, mock_config):
        """Test that spawn preserves elite individuals."""
        mock_config.elitism = 2
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 5.0  # Representative needs fitness too
        species = Species(1, representative, mock_config)

        # Add more members with different fitness
        for i in range(5):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i * 10)
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate', side_effect=lambda other: Individual(genome, 'standard')):
            offspring = species.spawn(10)

            # First 2 should be elite (highest fitness) - but they are deep copied
            assert len(offspring) == 10
            # Elite offspring should have highest fitness values (40.0 and 30.0)
            assert offspring[0].fitness == 40.0
            assert offspring[1].fitness == 30.0

    def test_spawn_creates_parent_pool_based_on_survival_threshold(self, simple_genome_dict, mock_config):
        """Test that spawn creates parent pool from top performers."""
        mock_config.elitism = 0
        mock_config.survival_threshold = 0.5  # Top 50%

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 5.0
        species = Species(1, representative, mock_config)

        # Add 10 members
        for i in range(10):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i)
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate') as mock_mate:
            mock_mate.return_value = Individual(genome, 'standard')
            species.spawn(5)

            # mate should have been called (for offspring beyond elitism)
            assert mock_mate.call_count > 0

    def test_spawn_ensures_minimum_two_parents(self, simple_genome_dict, mock_config):
        """Test that spawn ensures at least 2 parents for mating."""
        mock_config.elitism = 0
        mock_config.survival_threshold = 0.1  # Very low, but should still have 2 parents

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 1.5
        species = Species(1, representative, mock_config)

        # Add only 3 members
        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i)
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate') as mock_mate:
            mock_mate.return_value = Individual(genome, 'standard')
            species.spawn(5)

            # Should have created offspring (requires at least 2 parents)
            assert mock_mate.call_count > 0

    def test_spawn_sorts_members_by_fitness(self, simple_genome_dict, mock_config):
        """Test that spawn sorts members by fitness (highest first)."""
        mock_config.elitism = 3
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 7.0
        species = Species(1, representative, mock_config)

        # Add members with specific fitness values
        fitnesses = [5.0, 10.0, 3.0, 15.0, 8.0]
        for fitness in fitnesses:
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = fitness
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate', side_effect=lambda other: Individual(genome, 'standard')):
            offspring = species.spawn(5)

            # Elite should be in descending fitness order
            assert offspring[0].fitness == 15.0
            assert offspring[1].fitness == 10.0
            assert offspring[2].fitness == 8.0

    def test_spawn_uses_id_as_tiebreaker(self, simple_genome_dict, mock_config):
        """Test that spawn uses ID as tiebreaker for equal fitness."""
        mock_config.elitism = 3
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 10.0
        species = Species(1, representative, mock_config)

        # Add members with same fitness but different IDs
        for i in range(5):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = 10.0  # Same fitness
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate', side_effect=lambda other: Individual(genome, 'standard')):
            offspring = species.spawn(5)

            # All elite should have same fitness
            assert all(ind.fitness == 10.0 for ind in offspring[:3])
            # All offspring should be individuals (elitism creates deep copies with new IDs)
            assert all(isinstance(ind, Individual) for ind in offspring)

    def test_spawn_calls_mate_for_non_elite_offspring(self, simple_genome_dict, mock_config):
        """Test that spawn calls mate to create non-elite offspring."""
        mock_config.elitism = 1
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 1.5
        species = Species(1, representative, mock_config)

        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i)
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate') as mock_mate:
            mock_mate.return_value = Individual(genome, 'standard')
            offspring = species.spawn(5)

            # Should spawn 5 offspring: 1 elite + 4 through mating
            assert len(offspring) == 5
            assert mock_mate.call_count == 4

    def test_spawn_returns_correct_number_of_offspring(self, simple_genome_dict, mock_config):
        """Test that spawn returns exactly the requested number of offspring."""
        mock_config.elitism = 1
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 2.5
        species = Species(1, representative, mock_config)

        for i in range(5):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i)
            species.members[ind.ID] = ind

        with patch.object(Individual, 'mate', side_effect=lambda other: Individual(genome, 'standard')):
            offspring = species.spawn(10)

            assert len(offspring) == 10

    def test_spawn_handles_elitism_exceeding_num_offspring(self, simple_genome_dict, mock_config):
        """Test that spawn handles case where elitism > num_offspring."""
        mock_config.elitism = 10  # More than requested offspring
        mock_config.survival_threshold = 1.0

        genome = Genome.from_dict(simple_genome_dict)
        representative = Individual(genome, 'standard')
        representative.fitness = 2.5
        species = Species(1, representative, mock_config)

        for i in range(5):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i)
            species.members[ind.ID] = ind

        offspring = species.spawn(3)

        # Should only return 3 offspring (all elite, no mating needed)
        assert len(offspring) == 3


# ============================================================================
# Test Species Edge Cases
# ============================================================================

class TestSpeciesEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_member_species_can_spawn(self, sample_individual, mock_config):
        """Test that species with single member can still spawn."""
        mock_config.elitism = 0
        mock_config.survival_threshold = 1.0

        species = Species(1, sample_individual, mock_config)

        with patch.object(Individual, 'mate', side_effect=lambda other: Individual(
            Genome.from_dict({
                'activation': 'identity',
                'nodes': [
                    {'id': 0, 'type': 'input'},
                    {'id': 1, 'type': 'input'},
                    {'id': 2, 'type': 'output'},
                ],
                'connections': [
                    {'from': 0, 'to': 2, 'weight': 1.0, 'enabled': True},
                ]
            }), 'standard'
        )):
            offspring = species.spawn(3)

            # Single member will mate with itself
            assert len(offspring) == 3

    def test_fitness_history_tracks_multiple_generations(self, sample_individual, mock_config):
        """Test that fitness_history correctly tracks over generations."""
        species = Species(1, sample_individual, mock_config)

        for gen in range(5):
            species.update_fitness(float(gen * 10))
            species.init_for_next_generation(sample_individual)

        # History should have all 5 fitness values
        assert len(species.fitness_history) == 5
        assert species.fitness_history == [0.0, 10.0, 20.0, 30.0, 40.0]

    def test_max_fitness_tracks_best_across_generations(self, sample_individual, mock_config):
        """Test that max_fitness correctly tracks best fitness ever."""
        species = Species(1, sample_individual, mock_config)

        # Fitness goes up then down
        species.update_fitness(10.0)
        species.update_fitness(20.0)  # Peak
        species.update_fitness(15.0)  # Drop
        species.update_fitness(12.0)  # Further drop

        assert species.max_fitness == 20.0
        assert species.fitness == 12.0  # Current fitness is lower

    def test_species_with_all_none_fitness_cannot_spawn(self, simple_genome_dict, mock_config):
        """Test that species with None fitness values will fail when sorting."""
        genome = Genome.from_dict(simple_genome_dict)
        species = Species(1, Individual(genome, 'standard'), mock_config)

        # Add members without fitness
        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = None  # Not evaluated
            species.members[ind.ID] = ind

        # Should raise error when trying to sort by None fitness
        with pytest.raises(TypeError):
            species.spawn(5)
