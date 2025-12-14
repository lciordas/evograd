"""
Unit tests for evograd.run.trial module.

This module contains comprehensive tests for the Trial class,
which is the abstract base class for NEAT trials.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from abc import ABC

from evograd.run.config import Config
from evograd.run.trial import Trial
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.phenotype.individual import Individual
from evograd.pool.population import Population


# ============================================================================
# Concrete Trial Implementation for Testing
# ============================================================================

class ConcreteTrial(Trial):
    """Concrete implementation of Trial for testing purposes."""

    def __init__(self, config, network_type='standard', suppress_output=False):
        super().__init__(config, network_type, suppress_output)
        self.reset_called = False
        self.evaluate_fitness_calls = []
        self.generation_report_calls = []
        self.final_report_called = False

    def _reset(self):
        super()._reset()
        self.reset_called = True

    def _evaluate_fitness(self, individual):
        self.evaluate_fitness_calls.append(individual)
        # Return a simple fitness value
        return 10.0

    def _generation_report(self):
        self.generation_report_calls.append(self._generation_counter)

    def _final_report(self):
        self.final_report_called = True


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
    config.max_number_generations = 5
    config.fitness_termination_check = False
    config.fitness_criterion = "max"
    config.fitness_threshold = 100.0
    config.population_size = 10
    return config


# ============================================================================
# Test Trial Initialization
# ============================================================================

class TestTrialInit:
    """Test Trial.__init__ method."""

    def test_init_stores_config(self, mock_config):
        """Test that initialization stores config reference."""
        trial = ConcreteTrial(mock_config)

        assert trial._config is mock_config

    def test_init_stores_network_type(self, mock_config):
        """Test that initialization stores network type."""
        trial = ConcreteTrial(mock_config, network_type='fast')

        assert trial._network_type == 'fast'

    def test_init_stores_suppress_output(self, mock_config):
        """Test that initialization stores suppress_output flag."""
        trial = ConcreteTrial(mock_config, suppress_output=True)

        assert trial._suppress_output is True

    def test_init_sets_generation_counter_to_zero(self, mock_config):
        """Test that generation counter starts at 0."""
        trial = ConcreteTrial(mock_config)

        assert trial._generation_counter == 0

    def test_init_sets_population_to_none(self, mock_config):
        """Test that population starts as None."""
        trial = ConcreteTrial(mock_config)

        assert trial._population is None

    def test_init_sets_failed_to_true(self, mock_config):
        """Test that failed flag starts as True."""
        trial = ConcreteTrial(mock_config)

        assert trial.failed is True

    def test_cannot_instantiate_abstract_trial(self, mock_config):
        """Test that Trial abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Trial(mock_config, 'standard')


# ============================================================================
# Test Trial _reset
# ============================================================================

class TestTrialReset:
    """Test Trial._reset method."""

    def test_reset_initializes_innovation_tracker(self, mock_config):
        """Test that _reset initializes InnovationTracker."""
        trial = ConcreteTrial(mock_config)

        with patch.object(InnovationTracker, 'initialize') as mock_init:
            trial._reset()

            mock_init.assert_called_once_with(mock_config)

    def test_reset_sets_generation_counter_to_zero(self, mock_config):
        """Test that _reset resets generation counter."""
        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 10

        trial._reset()

        assert trial._generation_counter == 0

    def test_reset_sets_failed_to_true(self, mock_config):
        """Test that _reset sets failed flag to True."""
        trial = ConcreteTrial(mock_config)
        trial.failed = False

        trial._reset()

        assert trial.failed is True


# ============================================================================
# Test Trial _evaluate_fitness_all
# ============================================================================

class TestTrialEvaluateFitnessAll:
    """Test Trial._evaluate_fitness_all method."""

    def test_evaluate_fitness_all_serial(self, mock_config):
        """Test serial fitness evaluation (num_jobs=1)."""
        trial = ConcreteTrial(mock_config)

        # Create mock population
        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual) for _ in range(3)]
        mock_population.individuals = mock_individuals
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()

        trial._population = mock_population

        trial._evaluate_fitness_all(num_jobs=1)

        # Should evaluate all individuals
        assert len(trial.evaluate_fitness_calls) == 3
        for ind in mock_individuals:
            assert ind.fitness == 10.0

        # Should update species fitness
        mock_population._species_manager.update_fitness.assert_called_once()

    def test_evaluate_fitness_all_parallel(self, mock_config):
        """Test parallel fitness evaluation (num_jobs>1)."""
        trial = ConcreteTrial(mock_config)

        # Create mock population
        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual) for _ in range(3)]
        mock_population.individuals = mock_individuals
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()

        trial._population = mock_population

        # Mock Parallel to avoid actual parallelization
        with patch('evograd.run.trial.Parallel') as mock_parallel:
            mock_parallel.return_value = MagicMock(return_value=[10.0, 20.0, 30.0])

            trial._evaluate_fitness_all(num_jobs=2)

        # Should assign fitness values
        assert mock_individuals[0].fitness == 10.0
        assert mock_individuals[1].fitness == 20.0
        assert mock_individuals[2].fitness == 30.0

        # Should update species fitness
        mock_population._species_manager.update_fitness.assert_called_once()


# ============================================================================
# Test Trial _terminate
# ============================================================================

class TestTrialTerminate:
    """Test Trial._terminate method."""

    def test_terminate_when_max_generations_reached(self, mock_config):
        """Test that terminate returns True when max generations reached."""
        mock_config.max_number_generations = 5
        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 5

        result = trial._terminate()

        assert result is True

    def test_terminate_when_max_generations_not_reached(self, mock_config):
        """Test that terminate returns False before max generations."""
        mock_config.max_number_generations = 5
        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 3

        result = trial._terminate()

        assert result is False

    def test_terminate_with_fitness_check_max_success(self, mock_config):
        """Test termination with fitness check using max criterion (success)."""
        mock_config.fitness_termination_check = True
        mock_config.fitness_criterion = "max"
        mock_config.fitness_threshold = 50.0
        mock_config.max_number_generations = 10

        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 3

        # Create population with fitness values
        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual) for _ in range(3)]
        for i, ind in enumerate(mock_individuals):
            ind.fitness = float(i * 20)  # 0, 20, 40, 60
        mock_individuals.append(Mock(spec=Individual))
        mock_individuals[-1].fitness = 60.0  # Max is 60, above threshold

        mock_population.individuals = mock_individuals
        trial._population = mock_population

        result = trial._terminate()

        assert result is True
        assert trial.failed is False

    def test_terminate_with_fitness_check_max_failure(self, mock_config):
        """Test termination with fitness check using max criterion (failure)."""
        mock_config.fitness_termination_check = True
        mock_config.fitness_criterion = "max"
        mock_config.fitness_threshold = 100.0
        mock_config.max_number_generations = 10

        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 10  # At max generations

        # Create population with fitness values below threshold
        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual) for _ in range(3)]
        for i, ind in enumerate(mock_individuals):
            ind.fitness = float(i * 20)  # Max is 40, below threshold

        mock_population.individuals = mock_individuals
        trial._population = mock_population

        result = trial._terminate()

        assert result is True
        assert trial.failed is True

    def test_terminate_with_fitness_check_mean_success(self, mock_config):
        """Test termination with fitness check using mean criterion."""
        mock_config.fitness_termination_check = True
        mock_config.fitness_criterion = "mean"
        mock_config.fitness_threshold = 30.0
        mock_config.max_number_generations = 10

        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 3

        # Create population with mean fitness = 40
        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual) for _ in range(3)]
        mock_individuals[0].fitness = 30.0
        mock_individuals[1].fitness = 40.0
        mock_individuals[2].fitness = 50.0  # Mean = 40.0

        mock_population.individuals = mock_individuals
        trial._population = mock_population

        result = trial._terminate()

        assert result is True
        assert trial.failed is False

    def test_terminate_with_invalid_fitness_criterion(self, mock_config):
        """Test that invalid fitness criterion raises RuntimeError."""
        mock_config.fitness_termination_check = True
        mock_config.fitness_criterion = "invalid"
        mock_config.max_number_generations = 10

        trial = ConcreteTrial(mock_config)
        trial._generation_counter = 3

        mock_population = Mock(spec=Population)
        mock_individuals = [Mock(spec=Individual)]
        mock_individuals[0].fitness = 10.0
        mock_population.individuals = mock_individuals
        trial._population = mock_population

        with pytest.raises(RuntimeError, match="bad 'fitness_criterion'"):
            trial._terminate()


# ============================================================================
# Test Trial run
# ============================================================================

class TestTrialRun:
    """Test Trial.run method."""

    def test_run_calls_reset(self, mock_config):
        """Test that run calls _reset."""
        trial = ConcreteTrial(mock_config)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        assert trial.reset_called is True

    def test_run_creates_population(self, mock_config):
        """Test that run creates initial population."""
        trial = ConcreteTrial(mock_config)

        with patch.object(Population, '__init__', return_value=None) as mock_pop_init:
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        mock_pop_init.assert_called_once_with(mock_config, 'standard')

    def test_run_evaluates_initial_fitness(self, mock_config):
        """Test that run evaluates fitness for initial population."""
        trial = ConcreteTrial(mock_config)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all') as mock_eval:
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run(num_jobs=2)

        # Should be called at least once for initial population
        assert mock_eval.call_count >= 1
        mock_eval.assert_any_call(2)

    def test_run_calls_generation_report_when_not_suppressed(self, mock_config):
        """Test that run calls generation report when output not suppressed."""
        trial = ConcreteTrial(mock_config, suppress_output=False)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        # Should call generation report for initial population
        assert len(trial.generation_report_calls) >= 1

    def test_run_suppresses_generation_report_when_requested(self, mock_config):
        """Test that run suppresses generation report when requested."""
        trial = ConcreteTrial(mock_config, suppress_output=True)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        # Should not call generation report
        assert len(trial.generation_report_calls) == 0

    def test_run_evolution_loop(self, mock_config):
        """Test that run executes evolution loop correctly."""
        mock_config.max_number_generations = 3
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_population.individuals = [Mock(spec=Individual) for _ in range(3)]
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        mock_population.spawn_next_generation = Mock()

        # Create a function that returns the mock population
        def mock_population_init(self, config, network_type):
            # Copy attributes from our mock to the actual object
            self.individuals = mock_population.individuals
            self._species_manager = mock_population._species_manager
            self.spawn_next_generation = mock_population.spawn_next_generation

        with patch.object(Population, '__init__', mock_population_init):
            with patch.object(trial, '_evaluate_fitness_all'):
                # Terminate after 3 iterations
                with patch.object(trial, '_terminate', side_effect=[False, False, False, True]):
                    trial.run()

        # Should spawn 3 generations
        assert mock_population.spawn_next_generation.call_count == 3
        # Generation counter should be 3
        assert trial._generation_counter == 3

    def test_run_calls_final_report_when_not_suppressed(self, mock_config):
        """Test that run calls final report when output not suppressed."""
        trial = ConcreteTrial(mock_config, suppress_output=False)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        assert trial.final_report_called is True

    def test_run_suppresses_final_report_when_requested(self, mock_config):
        """Test that run suppresses final report when requested."""
        trial = ConcreteTrial(mock_config, suppress_output=True)

        with patch.object(Population, '__init__', return_value=None):
            with patch.object(trial, '_evaluate_fitness_all'):
                with patch.object(trial, '_terminate', side_effect=[True]):
                    trial.run()

        assert trial.final_report_called is False

    def test_run_increments_generation_counter(self, mock_config):
        """Test that run increments generation counter in evolution loop."""
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_population.individuals = [Mock(spec=Individual) for _ in range(3)]
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        mock_population.spawn_next_generation = Mock()

        def mock_population_init(self, config, network_type):
            self.individuals = mock_population.individuals
            self._species_manager = mock_population._species_manager
            self.spawn_next_generation = mock_population.spawn_next_generation

        with patch.object(Population, '__init__', mock_population_init):
            with patch.object(trial, '_evaluate_fitness_all'):
                # Run for 5 generations
                with patch.object(trial, '_terminate', side_effect=[False]*5 + [True]):
                    trial.run()

        assert trial._generation_counter == 5


# ============================================================================
# Test Trial Integration
# ============================================================================

class TestTrialIntegration:
    """Integration tests for Trial."""

    def test_complete_trial_workflow(self, mock_config):
        """Test complete trial workflow from start to finish."""
        mock_config.max_number_generations = 2
        trial = ConcreteTrial(mock_config, suppress_output=False)

        # Create mock population that will be used throughout
        mock_population = Mock(spec=Population)
        mock_population.individuals = [Mock(spec=Individual) for _ in range(3)]
        for ind in mock_population.individuals:
            ind.fitness = None
        mock_population.spawn_next_generation = Mock()
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()

        def mock_population_init(self, config, network_type):
            self.individuals = mock_population.individuals
            self._species_manager = mock_population._species_manager
            self.spawn_next_generation = mock_population.spawn_next_generation

        # Patch Population creation to return our mock
        with patch.object(Population, '__init__', mock_population_init):
            with patch.object(trial, '_terminate', side_effect=[False, False, True]):
                trial.run()

        # Should have reset
        assert trial.reset_called is True
        # Should have run 2 generations
        assert trial._generation_counter == 2
        # Should have called generation report 3 times (initial + 2 generations)
        assert len(trial.generation_report_calls) == 3
        # Should have called final report
        assert trial.final_report_called is True

    def test_trial_with_different_network_types(self, mock_config):
        """Test that trial works with different network types."""
        for network_type in ['standard', 'fast', 'autograd']:
            trial = ConcreteTrial(mock_config, network_type=network_type)

            with patch.object(Population, '__init__', return_value=None) as mock_pop:
                with patch.object(trial, '_evaluate_fitness_all'):
                    with patch.object(trial, '_terminate', side_effect=[True]):
                        trial.run()

            # Should create population with correct network type
            mock_pop.assert_called_with(mock_config, network_type)
