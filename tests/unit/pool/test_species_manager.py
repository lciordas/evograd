"""
Unit tests for neat.pool.species_manager module.

This module contains comprehensive tests for the SpeciesManager class,
which manages all species and the speciation process in NEAT.
"""

import pytest
from unittest.mock import Mock, patch
from copy import deepcopy

from evograd.run.config import Config
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.individual import Individual
from evograd.pool.species import Species
from evograd.pool.species_manager import SpeciesManager
from evograd.pool.population import Population


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
    """Minimal genome: 2 inputs -> 1 output."""
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
    """Create a mock Config object with common parameters."""
    config = Mock(spec=Config)
    config.compatibility_threshold = 3.0
    config.species_elitism = 2
    config.min_species_size = 2
    config.population_size = 10
    config.elitism = 1
    config.survival_threshold = 0.5
    config.max_stagnation_period = 15
    config.min_weight = -5.0
    config.max_weight = 5.0
    config.min_bias = -5.0
    config.max_bias = 5.0
    config.min_gain = 0.1
    config.max_gain = 5.0
    # Distance calculation coefficients
    config.distance_excess_coeff = 1.0
    config.distance_disjoint_coeff = 1.0
    config.distance_params_coeff = 0.4
    return config


@pytest.fixture
def mock_population(simple_genome_dict):
    """Create a mock Population with individuals."""
    genome = Genome.from_dict(simple_genome_dict)
    individuals = []
    for i in range(5):
        ind = Individual(deepcopy(genome), 'standard')
        ind.fitness = float(i * 10)
        individuals.append(ind)

    population = Mock(spec=Population)
    population.individuals = individuals
    return population


# ============================================================================
# Test SpeciesManager.DistanceCache
# ============================================================================

class TestDistanceCache:
    """Test SpeciesManager.DistanceCache nested class."""

    def test_distance_cache_caches_distance(self, simple_genome_dict):
        """Test that DistanceCache caches computed distances."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')

        cache = SpeciesManager.DistanceCache()

        # Mock the distance method to return a fixed value and track calls
        with patch.object(Individual, 'distance', return_value=5.0) as mock_distance:
            # First call should compute
            dist1 = cache(ind1, ind2)
            assert mock_distance.call_count == 1

            # Second call should use cache
            dist2 = cache(ind1, ind2)
            assert mock_distance.call_count == 1  # No additional call
            assert dist1 == dist2 == 5.0

    def test_distance_cache_symmetric(self, simple_genome_dict):
        """Test that cache works for both (a,b) and (b,a)."""
        genome = Genome.from_dict(simple_genome_dict)
        ind1 = Individual(genome, 'standard')
        ind2 = Individual(genome, 'standard')

        cache = SpeciesManager.DistanceCache()

        with patch.object(Individual, 'distance', return_value=5.0) as mock_distance:
            # Compute (ind1, ind2)
            dist1 = cache(ind1, ind2)
            assert mock_distance.call_count == 1

            # Query (ind2, ind1) should use cache
            dist2 = cache(ind2, ind1)
            assert mock_distance.call_count == 1  # Still only 1 call
            assert dist1 == dist2


# ============================================================================
# Test SpeciesManager Initialization
# ============================================================================

class TestSpeciesManagerInit:
    """Test SpeciesManager.__init__ method."""

    def test_init_creates_empty_species_dict(self, mock_config):
        """Test that initialization creates empty species dictionary."""
        manager = SpeciesManager(mock_config)

        assert manager.species == {}
        assert isinstance(manager.species, dict)

    def test_init_creates_empty_individual_to_species_dict(self, mock_config):
        """Test that initialization creates empty individual_to_species mapping."""
        manager = SpeciesManager(mock_config)

        assert manager.individual_to_species == {}
        assert isinstance(manager.individual_to_species, dict)

    def test_init_stores_config(self, mock_config):
        """Test that initialization stores config reference."""
        manager = SpeciesManager(mock_config)

        assert manager._config is mock_config

    def test_init_creates_id_generator(self, mock_config):
        """Test that initialization creates ID generator."""
        manager = SpeciesManager(mock_config)

        # ID generator should start at 1
        assert next(manager._id_generator) == 1
        assert next(manager._id_generator) == 2


# ============================================================================
# Test SpeciesManager speciate
# ============================================================================

class TestSpeciesManagerSpeciate:
    """Test SpeciesManager.speciate method."""

    def test_speciate_first_generation_creates_species(self, mock_config, mock_population):
        """Test that first speciation creates species for all individuals."""
        manager = SpeciesManager(mock_config)

        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(mock_population)

        # Should have created at least one species
        assert len(manager.species) >= 1
        # All individuals should be assigned
        total_members = sum(len(spec.members) for spec in manager.species.values())
        assert total_members == len(mock_population.individuals)

    def test_speciate_all_individuals_assigned(self, mock_config, mock_population):
        """Test that all individuals are assigned to species."""
        manager = SpeciesManager(mock_config)

        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(mock_population)

        # Check individual_to_species mapping
        assert len(manager.individual_to_species) == len(mock_population.individuals)
        for ind in mock_population.individuals:
            assert ind.ID in manager.individual_to_species

    def test_speciate_updates_representatives(self, mock_config, simple_genome_dict):
        """Test that speciation updates species representatives."""
        genome = Genome.from_dict(simple_genome_dict)

        # Create first generation
        pop1 = Mock(spec=Population)
        pop1.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(3)]
        for ind in pop1.individuals:
            ind.fitness = 10.0

        manager = SpeciesManager(mock_config)

        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(pop1)

        initial_species_count = len(manager.species)
        first_spec_id = list(manager.species.keys())[0]

        # Create second generation (new individuals)
        pop2 = Mock(spec=Population)
        pop2.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(3)]
        for ind in pop2.individuals:
            ind.fitness = 15.0

        # Mock distance to ensure they're compatible
        with patch.object(Individual, 'distance', return_value=1.0):
            manager.speciate(pop2)

        # Should still have species (not extinct)
        assert len(manager.species) >= 1
        # Representative should be from new generation
        if first_spec_id in manager.species:
            rep_id = manager.species[first_spec_id].representative.ID
            assert any(ind.ID == rep_id for ind in pop2.individuals)

    def test_speciate_creates_new_species_when_incompatible(self, mock_config, simple_genome_dict):
        """Test that new species are created for incompatible individuals."""
        genome = Genome.from_dict(simple_genome_dict)

        # Create population
        population = Mock(spec=Population)
        population.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(3)]
        for ind in population.individuals:
            ind.fitness = 10.0

        manager = SpeciesManager(mock_config)

        # Mock distance to force multiple species
        # First individual creates species, others are incompatible
        distances = [100.0] * 10  # All distances above threshold
        with patch.object(Individual, 'distance', side_effect=distances):
            manager.speciate(population)

        # Should have created multiple species (each individual incompatible)
        assert len(manager.species) > 1

    def test_speciate_removes_extinct_species(self, mock_config, simple_genome_dict):
        """Test that species with no members are removed."""
        genome = Genome.from_dict(simple_genome_dict)

        # First generation
        pop1 = Mock(spec=Population)
        pop1.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(2)]
        for ind in pop1.individuals:
            ind.fitness = 10.0

        manager = SpeciesManager(mock_config)

        # Create two species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(pop1)

        initial_species_count = len(manager.species)
        assert initial_species_count >= 2

        # Second generation with only one individual
        pop2 = Mock(spec=Population)
        pop2.individuals = [Individual(deepcopy(genome), 'standard')]
        pop2.individuals[0].fitness = 10.0

        # All belong to same species
        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(pop2)

        # Should have fewer species (some went extinct)
        assert len(manager.species) < initial_species_count


# ============================================================================
# Test SpeciesManager update_fitness
# ============================================================================

class TestSpeciesManagerUpdateFitness:
    """Test SpeciesManager.update_fitness method."""

    def test_update_fitness_calculates_average(self, mock_config, simple_genome_dict):
        """Test that update_fitness calculates average fitness for each species."""
        genome = Genome.from_dict(simple_genome_dict)

        # Create population with known fitness values
        population = Mock(spec=Population)
        population.individuals = []
        for fitness in [10.0, 20.0, 30.0]:
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = fitness
            population.individuals.append(ind)

        manager = SpeciesManager(mock_config)

        # Force all into one species
        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(population)

        manager.update_fitness()

        # Species fitness should be average of member fitnesses
        species = list(manager.species.values())[0]
        assert species.fitness == 20.0  # (10 + 20 + 30) / 3

    def test_update_fitness_updates_all_species(self, mock_config, simple_genome_dict):
        """Test that update_fitness updates all species."""
        genome = Genome.from_dict(simple_genome_dict)

        population = Mock(spec=Population)
        population.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(4)]
        for i, ind in enumerate(population.individuals):
            ind.fitness = float(i * 10)

        manager = SpeciesManager(mock_config)

        # Create multiple species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        # All species should have fitness calculated
        for species in manager.species.values():
            assert species.fitness is not None


# ============================================================================
# Test SpeciesManager remove_stagnating_species
# ============================================================================

class TestSpeciesManagerRemoveStagnatingSpecies:
    """Test SpeciesManager.remove_stagnating_species method."""

    def test_remove_stagnating_species_returns_empty_when_no_individuals(self, mock_config):
        """Test that method returns empty sets when population is empty."""
        population = Mock(spec=Population)
        population.individuals = []
        population.get_fittest_individual = Mock(return_value=None)

        manager = SpeciesManager(mock_config)

        stagnant_specs, stagnant_inds = manager.remove_stagnating_species(population)

        assert stagnant_specs == set()
        assert stagnant_inds == set()

    def test_remove_stagnating_species_protects_fittest_species(self, mock_config, simple_genome_dict):
        """Test that species with fittest individual is protected."""
        genome = Genome.from_dict(simple_genome_dict)

        # Create stagnant species with fittest individual
        population = Mock(spec=Population)
        fittest_ind = Individual(deepcopy(genome), 'standard')
        fittest_ind.fitness = 100.0

        other_ind = Individual(deepcopy(genome), 'standard')
        other_ind.fitness = 50.0

        population.individuals = [fittest_ind, other_ind]
        population.get_fittest_individual = Mock(return_value=fittest_ind)

        manager = SpeciesManager(mock_config)

        # Create two species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        # Make both species stagnant
        for spec in manager.species.values():
            spec.age = 100
            spec.last_improved = 0

        stagnant_specs, _ = manager.remove_stagnating_species(population)

        # Species with fittest should be protected
        fittest_species = manager.individual_to_species[fittest_ind.ID]
        assert fittest_species.id not in stagnant_specs

    def test_remove_stagnating_species_protects_elite_species(self, mock_config, simple_genome_dict):
        """Test that top N species are protected from removal."""
        mock_config.species_elitism = 2
        genome = Genome.from_dict(simple_genome_dict)

        # Create 3 species
        population = Mock(spec=Population)
        population.individuals = []
        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float((i + 1) * 10)
            population.individuals.append(ind)

        population.get_fittest_individual = Mock(return_value=population.individuals[-1])

        manager = SpeciesManager(mock_config)

        # Create three species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        # Make all stagnant
        for spec in manager.species.values():
            spec.age = 100
            spec.last_improved = 0

        stagnant_specs, _ = manager.remove_stagnating_species(population)

        # Top 2 species should be protected
        sorted_species = sorted(manager.species.values(), key=lambda s: s.fitness, reverse=True)
        protected_ids = {sorted_species[0].id, sorted_species[1].id}

        assert not protected_ids.intersection(stagnant_specs)

    def test_remove_stagnating_species_removes_stagnant(self, mock_config, simple_genome_dict):
        """Test that stagnant species are actually removed."""
        mock_config.species_elitism = 0
        genome = Genome.from_dict(simple_genome_dict)

        # Create 2 species
        population = Mock(spec=Population)
        ind1 = Individual(deepcopy(genome), 'standard')
        ind1.fitness = 10.0
        ind2 = Individual(deepcopy(genome), 'standard')
        ind2.fitness = 20.0

        population.individuals = [ind1, ind2]
        population.get_fittest_individual = Mock(return_value=ind2)

        manager = SpeciesManager(mock_config)

        # Create two species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        initial_count = len(manager.species)

        # Make first species stagnant (not the fittest)
        for spec in manager.species.values():
            if ind1.ID in spec.members:
                spec.age = 100
                spec.last_improved = 0

        stagnant_specs, stagnant_inds = manager.remove_stagnating_species(population)

        # Should have removed stagnant species
        assert len(stagnant_specs) > 0
        assert len(manager.species) < initial_count


# ============================================================================
# Test SpeciesManager calculate_offspring_allocations
# ============================================================================

class TestSpeciesManagerCalculateOffspringAllocations:
    """Test SpeciesManager.calculate_offspring_allocations method."""

    def test_calculate_offspring_allocations_proportional_to_fitness(self, mock_config, simple_genome_dict):
        """Test that offspring are allocated proportionally to species fitness."""
        mock_config.population_size = 100
        mock_config.min_species_size = 1
        mock_config.elitism = 0

        genome = Genome.from_dict(simple_genome_dict)

        # Create 2 species with different fitness
        population = Mock(spec=Population)
        ind1 = Individual(deepcopy(genome), 'standard')
        ind1.fitness = 30.0  # 30% of total
        ind2 = Individual(deepcopy(genome), 'standard')
        ind2.fitness = 70.0  # 70% of total

        population.individuals = [ind1, ind2]

        manager = SpeciesManager(mock_config)

        # Create two species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        allocations = manager.calculate_offspring_allocations()

        # Higher fitness species should get more offspring
        spec1_id = manager.individual_to_species[ind1.ID].id
        spec2_id = manager.individual_to_species[ind2.ID].id

        assert allocations[spec2_id] > allocations[spec1_id]

    def test_calculate_offspring_allocations_respects_minimum(self, mock_config, simple_genome_dict):
        """Test that all species get at least min_species_size offspring."""
        mock_config.population_size = 20
        mock_config.min_species_size = 5
        mock_config.elitism = 0

        genome = Genome.from_dict(simple_genome_dict)

        # Create species with very different fitness
        population = Mock(spec=Population)
        population.individuals = []
        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i * 50) if i > 0 else 1.0  # One very low fitness
            population.individuals.append(ind)

        manager = SpeciesManager(mock_config)

        # Create three species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        allocations = manager.calculate_offspring_allocations()

        # All species should get at least min_species_size
        for num_offspring in allocations.values():
            assert num_offspring >= mock_config.min_species_size

    def test_calculate_offspring_allocations_handles_zero_fitness(self, mock_config, simple_genome_dict):
        """Test that allocation works when all species have zero fitness."""
        mock_config.population_size = 10
        mock_config.min_species_size = 1
        mock_config.elitism = 0

        genome = Genome.from_dict(simple_genome_dict)

        # Create species with zero fitness
        population = Mock(spec=Population)
        population.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(2)]
        for ind in population.individuals:
            ind.fitness = 0.0

        manager = SpeciesManager(mock_config)

        # Create two species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        allocations = manager.calculate_offspring_allocations()

        # Should divide equally when all fitness is zero
        assert len(allocations) == 2
        # Total should be close to population size
        total = sum(allocations.values())
        assert abs(total - mock_config.population_size) <= len(allocations)

    def test_calculate_offspring_allocations_adjusts_for_rounding(self, mock_config, simple_genome_dict):
        """Test that allocations are adjusted to match population size."""
        mock_config.population_size = 100
        mock_config.min_species_size = 1
        mock_config.elitism = 0

        genome = Genome.from_dict(simple_genome_dict)

        # Create 3 species
        population = Mock(spec=Population)
        population.individuals = []
        for i in range(3):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = 33.3  # Causes rounding issues
            population.individuals.append(ind)

        manager = SpeciesManager(mock_config)

        # Create three species
        with patch.object(Individual, 'distance', return_value=100.0):
            manager.speciate(population)

        manager.update_fitness()

        allocations = manager.calculate_offspring_allocations()

        # Total should be close to population size (within species count due to rounding)
        total = sum(allocations.values())
        assert abs(total - mock_config.population_size) <= len(manager.species)


# ============================================================================
# Test SpeciesManager Integration
# ============================================================================

class TestSpeciesManagerIntegration:
    """Integration tests for complete speciation workflow."""

    def test_full_speciation_workflow(self, mock_config, simple_genome_dict):
        """Test complete workflow: speciate, update fitness, allocate offspring."""
        genome = Genome.from_dict(simple_genome_dict)

        # Create population
        population = Mock(spec=Population)
        population.individuals = []
        for i in range(10):
            ind = Individual(deepcopy(genome), 'standard')
            ind.fitness = float(i * 10)
            population.individuals.append(ind)

        population.get_fittest_individual = Mock(return_value=population.individuals[-1])

        manager = SpeciesManager(mock_config)

        # Speciate
        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(population)
        assert len(manager.species) > 0

        # Update fitness
        manager.update_fitness()
        for spec in manager.species.values():
            assert spec.fitness is not None

        # Allocate offspring
        allocations = manager.calculate_offspring_allocations()
        assert sum(allocations.values()) > 0

    def test_multi_generation_species_tracking(self, mock_config, simple_genome_dict):
        """Test that species are tracked correctly across generations."""
        genome = Genome.from_dict(simple_genome_dict)

        manager = SpeciesManager(mock_config)

        # Generation 1
        pop1 = Mock(spec=Population)
        pop1.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(3)]
        for ind in pop1.individuals:
            ind.fitness = 10.0

        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(pop1)

        gen1_species_count = len(manager.species)
        gen1_species_ids = set(manager.species.keys())

        # Generation 2 - new individuals but compatible
        pop2 = Mock(spec=Population)
        pop2.individuals = [Individual(deepcopy(genome), 'standard') for _ in range(3)]
        for ind in pop2.individuals:
            ind.fitness = 15.0

        with patch.object(Individual, 'distance', return_value=0.5):
            manager.speciate(pop2)

        gen2_species_ids = set(manager.species.keys())

        # Species IDs should be preserved (same species across generations)
        assert gen1_species_ids == gen2_species_ids
