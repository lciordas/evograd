"""
Unit tests for neat.pool.population module.

This module contains comprehensive tests for the Population class,
which is the top-level evolutionary coordinator in NEAT.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

from evograd.run.config import Config
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.individual import Individual
from evograd.pool.population import Population
from evograd.pool.species_manager import SpeciesManager


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
def mock_config():
    """Create a mock Config object with common parameters."""
    config = Mock(spec=Config)
    config.num_inputs = 2
    config.num_outputs = 1
    config.population_size = 10
    config.initial_cxn_policy = "none"
    config.compatibility_threshold = 3.0
    config.species_elitism = 2
    config.min_species_size = 2
    config.elitism = 1
    config.survival_threshold = 0.5
    config.max_stagnation_period = 15
    config.min_weight = -5.0
    config.max_weight = 5.0
    config.min_bias = -5.0
    config.max_bias = 5.0
    config.min_gain = 0.1
    config.max_gain = 5.0
    config.weight_init_mean = 0.0
    config.weight_init_stdev = 1.0
    config.bias_init_mean = 0.0
    config.bias_init_stdev = 1.0
    config.gain_init_mean = 1.0
    config.gain_init_stdev = 0.1
    config.initial_cxn_fraction = 0.5
    config.activation_initial = 'sigmoid'  # Default activation (used by NodeGene)
    # Distance calculation coefficients
    config.distance_excess_coeff = 1.0
    config.distance_disjoint_coeff = 1.0
    config.distance_params_coeff = 0.4
    return config


# ============================================================================
# Test Population Initialization
# ============================================================================

class TestPopulationInit:
    """Test Population.__init__ method."""

    def test_init_creates_individuals(self, mock_config):
        """Test that initialization creates the correct number of individuals."""
        mock_config.population_size = 5

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        assert len(population.individuals) == 5
        assert all(isinstance(ind, Individual) for ind in population.individuals)

    def test_init_stores_config(self, mock_config):
        """Test that initialization stores config reference."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        assert population._config is mock_config

    def test_init_stores_network_type(self, mock_config):
        """Test that initialization stores network type."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'fast')

        assert population._network_type == 'fast'

    def test_init_creates_species_manager(self, mock_config):
        """Test that initialization creates SpeciesManager."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        assert isinstance(population._species_manager, SpeciesManager)

    def test_init_calls_speciate(self, mock_config):
        """Test that initialization calls speciate on species manager."""
        with patch.object(SpeciesManager, 'speciate') as mock_speciate:
            population = Population(mock_config, 'standard')

            mock_speciate.assert_called_once()

    def test_init_with_none_policy_no_connections(self, mock_config):
        """Test that 'none' policy creates individuals without connections."""
        mock_config.initial_cxn_policy = "none"

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # All individuals should have no connections
        for ind in population.individuals:
            assert len(ind.genome.conn_genes) == 0

    def test_init_with_one_input_policy(self, mock_config):
        """Test that 'one-input' policy creates connections from one input."""
        mock_config.initial_cxn_policy = "one-input"
        mock_config.num_inputs = 3
        mock_config.num_outputs = 2

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Each individual should have connections from 1 input to all outputs
        for ind in population.individuals:
            # Should have num_outputs connections (one input to all outputs)
            assert len(ind.genome.conn_genes) == mock_config.num_outputs

    def test_init_with_full_policy(self, mock_config):
        """Test that 'full' policy creates fully connected networks."""
        mock_config.initial_cxn_policy = "full"
        mock_config.num_inputs = 2
        mock_config.num_outputs = 3

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Each individual should have num_inputs * num_outputs connections
        expected_connections = mock_config.num_inputs * mock_config.num_outputs
        for ind in population.individuals:
            assert len(ind.genome.conn_genes) == expected_connections

    def test_init_with_partial_policy(self, mock_config):
        """Test that 'partial' policy creates partial connections."""
        mock_config.initial_cxn_policy = "partial"
        mock_config.initial_cxn_fraction = 0.5
        mock_config.num_inputs = 4
        mock_config.num_outputs = 4

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Each individual should have ~50% of possible connections
        max_connections = mock_config.num_inputs * mock_config.num_outputs
        expected_connections = int(max_connections * mock_config.initial_cxn_fraction)

        for ind in population.individuals:
            assert len(ind.genome.conn_genes) == expected_connections

    def test_init_with_invalid_policy_raises_error(self, mock_config):
        """Test that invalid connection policy raises RuntimeError."""
        mock_config.initial_cxn_policy = "invalid_policy"

        with pytest.raises(RuntimeError, match="bad initial connection policy"):
            with patch.object(SpeciesManager, 'speciate'):
                population = Population(mock_config, 'standard')


# ============================================================================
# Test Population get_fittest_individual
# ============================================================================

class TestPopulationGetFittestIndividual:
    """Test Population.get_fittest_individual method."""

    def test_get_fittest_individual_returns_highest_fitness(self, mock_config):
        """Test that method returns individual with highest fitness."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness values
        for i, ind in enumerate(population.individuals):
            ind.fitness = float(i * 10)

        fittest = population.get_fittest_individual()

        assert fittest is population.individuals[-1]
        assert fittest.fitness == 90.0

    def test_get_fittest_individual_empty_population_returns_none(self, mock_config):
        """Test that method returns None for empty population."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        population.individuals = []

        fittest = population.get_fittest_individual()

        assert fittest is None

    def test_get_fittest_individual_none_fitness_returns_none(self, mock_config):
        """Test that method returns None when fitness is not evaluated."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Leave fitness as None (not evaluated)
        for ind in population.individuals:
            assert ind.fitness is None

        fittest = population.get_fittest_individual()

        assert fittest is None

    def test_get_fittest_individual_with_negative_fitness(self, mock_config):
        """Test that method works with negative fitness values."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set negative fitness values
        for i, ind in enumerate(population.individuals):
            ind.fitness = float(i - 5)  # -5, -4, -3, -2, -1, 0, 1, 2, 3, 4

        fittest = population.get_fittest_individual()

        assert fittest is population.individuals[-1]
        assert fittest.fitness == 4.0

    def test_get_fittest_individual_with_equal_fitness(self, mock_config):
        """Test that method returns one individual when all have equal fitness."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set equal fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        fittest = population.get_fittest_individual()

        assert fittest is not None
        assert fittest.fitness == 10.0


# ============================================================================
# Test Population spawn_next_generation
# ============================================================================

class TestPopulationSpawnNextGeneration:
    """Test Population.spawn_next_generation method."""

    def test_spawn_next_generation_calls_remove_stagnating_species(self, mock_config):
        """Test that spawn calls remove_stagnating_species."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness for all individuals
        for ind in population.individuals:
            ind.fitness = 10.0

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())) as mock_remove:
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}):
                with patch.object(SpeciesManager, 'speciate'):
                    # Mock species manager to have at least one species
                    from evograd.pool.species import Species
                    mock_species = Mock(spec=Species)
                    mock_species.spawn = Mock(return_value=population.individuals[:10])
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

            mock_remove.assert_called_once_with(population)

    def test_spawn_next_generation_calls_calculate_offspring_allocations(self, mock_config):
        """Test that spawn calls calculate_offspring_allocations."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}) as mock_calc:
                with patch.object(SpeciesManager, 'speciate'):
                    from evograd.pool.species import Species
                    mock_species = Mock(spec=Species)
                    mock_species.spawn = Mock(return_value=population.individuals[:10])
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

            mock_calc.assert_called_once()

    def test_spawn_next_generation_calls_species_spawn(self, mock_config):
        """Test that spawn calls spawn on each species."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        from evograd.pool.species import Species
        mock_species = Mock(spec=Species)
        mock_species.spawn = Mock(return_value=population.individuals[:10])

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}):
                with patch.object(SpeciesManager, 'speciate'):
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

            mock_species.spawn.assert_called_once_with(10)

    def test_spawn_next_generation_updates_individuals(self, mock_config):
        """Test that spawn updates population.individuals."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        original_individuals = population.individuals.copy()

        # Create new individuals for next generation
        new_individuals = []
        for i in range(10):
            genome = Mock()
            genome.conn_genes = {}
            ind = Mock(spec=Individual)
            ind.genome = genome
            ind.fitness = 15.0
            new_individuals.append(ind)

        from evograd.pool.species import Species
        mock_species = Mock(spec=Species)
        mock_species.spawn = Mock(return_value=new_individuals)

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}):
                with patch.object(SpeciesManager, 'speciate'):
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

        # Population should now contain new individuals
        assert population.individuals == new_individuals
        assert population.individuals != original_individuals

    def test_spawn_next_generation_calls_speciate(self, mock_config):
        """Test that spawn calls speciate after creating new generation."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        from evograd.pool.species import Species
        mock_species = Mock(spec=Species)
        mock_species.spawn = Mock(return_value=population.individuals[:10])

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}):
                with patch.object(SpeciesManager, 'speciate') as mock_speciate:
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

            # Should be called once in spawn_next_generation (already called once in __init__)
            assert mock_speciate.call_count == 1

    def test_spawn_next_generation_handles_multiple_species(self, mock_config):
        """Test that spawn handles multiple species correctly."""
        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Set fitness
        for ind in population.individuals:
            ind.fitness = 10.0

        # Create offspring for two species
        offspring1 = [Mock(spec=Individual) for _ in range(6)]
        offspring2 = [Mock(spec=Individual) for _ in range(4)]

        from evograd.pool.species import Species
        mock_species1 = Mock(spec=Species)
        mock_species1.spawn = Mock(return_value=offspring1)
        mock_species2 = Mock(spec=Species)
        mock_species2.spawn = Mock(return_value=offspring2)

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 6, 2: 4}):
                with patch.object(SpeciesManager, 'speciate'):
                    population._species_manager.species = {1: mock_species1, 2: mock_species2}

                    population.spawn_next_generation()

        # Population should contain all offspring from both species
        assert len(population.individuals) == 10
        assert all(ind in population.individuals for ind in offspring1)
        assert all(ind in population.individuals for ind in offspring2)


# ============================================================================
# Test Population String Method
# ============================================================================

class TestPopulationStr:
    """Test Population.__str__ method."""

    def test_str_contains_individual_strings(self, mock_config):
        """Test that __str__ contains string representation of individuals."""
        mock_config.population_size = 3

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        result = str(population)

        # Should contain newlines separating individuals
        assert '\n' in result or len(population.individuals) == 1


# ============================================================================
# Test Population Integration
# ============================================================================

class TestPopulationIntegration:
    """Integration tests for Population."""

    def test_full_generation_cycle(self, mock_config):
        """Test a complete generation cycle."""
        mock_config.population_size = 10

        with patch.object(SpeciesManager, 'speciate'):
            population = Population(mock_config, 'standard')

        # Evaluate fitness
        for i, ind in enumerate(population.individuals):
            ind.fitness = float(i)

        # Get fittest before evolution
        fittest_before = population.get_fittest_individual()
        assert fittest_before.fitness == 9.0

        # Evolve
        from evograd.pool.species import Species
        mock_species = Mock(spec=Species)
        # Return same individuals for simplicity
        mock_species.spawn = Mock(return_value=population.individuals)

        with patch.object(SpeciesManager, 'remove_stagnating_species', return_value=(set(), set())):
            with patch.object(SpeciesManager, 'calculate_offspring_allocations', return_value={1: 10}):
                with patch.object(SpeciesManager, 'speciate'):
                    population._species_manager.species = {1: mock_species}

                    population.spawn_next_generation()

        # Should have new generation
        assert len(population.individuals) == 10

    def test_population_with_different_network_types(self, mock_config):
        """Test that population works with different network types."""
        for network_type in ['standard', 'fast', 'autograd']:
            with patch.object(SpeciesManager, 'speciate'):
                population = Population(mock_config, network_type)

            assert population._network_type == network_type
            assert len(population.individuals) == mock_config.population_size
