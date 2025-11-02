"""
Unit tests for Experiment abstract base class.
"""

import pytest
from unittest.mock import Mock, patch, call
from abc import ABC

from evograd.run.experiment import Experiment
from evograd.run.trial import Trial
from evograd.run.config import Config
from evograd.pool.population import Population
from evograd.phenotype.individual import Individual
from evograd.phenotype.network_standard import NetworkStandard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock Config object."""
    config = Mock(spec=Config)
    config.max_number_generations = 10
    config.fitness_termination_check = False
    return config


@pytest.fixture
def mock_trial_class():
    """Create a mock Trial class."""
    mock_class = Mock(spec=Trial)
    return mock_class


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================

class ConcreteTrial(Trial):
    """Concrete implementation of Trial for testing purposes."""

    def __init__(self, config, network_type='standard', suppress_output=False):
        super().__init__(config, network_type, suppress_output)
        self.reset_called = False

    def _reset(self):
        super()._reset()
        self.reset_called = True

    def _evaluate_fitness(self, individual):
        return 10.0

    def _generation_report(self):
        pass

    def _final_report(self):
        pass


class ConcreteExperiment(Experiment):
    """Concrete implementation of Experiment for testing purposes."""

    def __init__(self, trial_class, num_trials, config, *args, **kwargs):
        super().__init__(trial_class, num_trials, config, *args, **kwargs)
        self.reset_called = False
        self.prepare_trial_calls = []
        self.extract_trial_results_calls = []
        self.analyze_trial_results_calls = []
        self.final_report_called = False

    def _reset(self):
        super()._reset()
        self.reset_called = True

    def _prepare_trial(self, trial, trial_number):
        self.prepare_trial_calls.append((trial, trial_number))
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial, trial_number):
        self.extract_trial_results_calls.append((trial, trial_number))
        results = super()._extract_trial_results(trial, trial_number)
        return results

    def _analyze_trial_results(self, results):
        self.analyze_trial_results_calls.append(results)
        super()._analyze_trial_results(results)

    def _final_report(self):
        self.final_report_called = True


# ============================================================================
# Test Experiment Initialization
# ============================================================================

class TestExperimentInit:
    """Test Experiment initialization."""

    def test_cannot_instantiate_abstract_class(self, mock_config):
        """Test that Experiment cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Experiment(ConcreteTrial, 5, mock_config)

    def test_init_stores_trial_class(self, mock_config):
        """Test that __init__ stores trial class."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        assert experiment._trial_class == ConcreteTrial

    def test_init_stores_num_trials(self, mock_config):
        """Test that __init__ stores number of trials."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        assert experiment._num_trials == 5

    def test_init_stores_config(self, mock_config):
        """Test that __init__ stores config."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        assert experiment._config == mock_config

    def test_init_stores_trial_args(self, mock_config):
        """Test that __init__ stores positional arguments for trial class."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config, 'arg1', 'arg2')
        assert experiment._trial_args == ('arg1', 'arg2')

    def test_init_stores_trial_kwargs(self, mock_config):
        """Test that __init__ stores keyword arguments for trial class."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config, kwarg1='val1', kwarg2='val2')
        assert experiment._trial_kwargs == {'kwarg1': 'val1', 'kwarg2': 'val2'}

    def test_init_initializes_counters(self, mock_config):
        """Test that __init__ initializes counters."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        assert experiment._trial_counter == 0
        assert experiment._success_counter == 0

    def test_init_initializes_result_lists(self, mock_config):
        """Test that __init__ initializes result lists."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        assert experiment._number_generations == []
        assert experiment._max_fitness == []
        assert experiment._number_neurons == []
        assert experiment._number_connections == []


# ============================================================================
# Test Experiment Reset
# ============================================================================

class TestExperimentReset:
    """Test Experiment reset."""

    def test_reset_resets_trial_counter(self, mock_config):
        """Test that _reset resets trial counter."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        experiment._trial_counter = 10
        experiment._reset()
        assert experiment._trial_counter == 0

    def test_reset_resets_success_counter(self, mock_config):
        """Test that _reset resets success counter."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        experiment._success_counter = 10
        experiment._reset()
        assert experiment._success_counter == 0

    def test_reset_clears_result_lists(self, mock_config):
        """Test that _reset clears result lists."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        experiment._number_generations = [1, 2, 3]
        experiment._max_fitness = [10.0, 20.0, 30.0]
        experiment._number_neurons = [5, 6, 7]
        experiment._number_connections = [10, 11, 12]
        experiment._reset()
        assert experiment._number_generations == []
        assert experiment._max_fitness == []
        assert experiment._number_neurons == []
        assert experiment._number_connections == []


# ============================================================================
# Test Experiment Run Trial
# ============================================================================

class TestExperimentRunTrial:
    """Test Experiment _run_trial method."""

    def test_run_trial_creates_trial_instance(self, mock_config):
        """Test that _run_trial creates trial instance."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)

        # Mock the trial's run method and population
        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            # Access self which is the trial instance
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            result = experiment._run_trial(1, num_jobs=1)

        # Should have called _prepare_trial
        assert len(experiment.prepare_trial_calls) == 1
        assert isinstance(experiment.prepare_trial_calls[0][0], ConcreteTrial)
        assert experiment.prepare_trial_calls[0][1] == 1

    def test_run_trial_calls_prepare_trial(self, mock_config):
        """Test that _run_trial calls _prepare_trial."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment._run_trial(2, num_jobs=1)

        assert len(experiment.prepare_trial_calls) == 1
        assert experiment.prepare_trial_calls[0][1] == 2

    def test_run_trial_calls_trial_run(self, mock_config):
        """Test that _run_trial calls trial.run()."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run) as mock_run:
            experiment._run_trial(1, num_jobs=2)

        mock_run.assert_called_once_with(2)

    def test_run_trial_calls_extract_trial_results(self, mock_config):
        """Test that _run_trial calls _extract_trial_results."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            result = experiment._run_trial(1, num_jobs=1)

        assert len(experiment.extract_trial_results_calls) == 1
        assert experiment.extract_trial_results_calls[0][1] == 1

    def test_run_trial_passes_args_to_trial_class(self, mock_config):
        """Test that _run_trial passes args to trial class constructor."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config, network_type='fast')

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment._run_trial(1, num_jobs=1)

        # Check that trial was created with network_type='fast'
        trial_instance = experiment.prepare_trial_calls[0][0]
        assert trial_instance._network_type == 'fast'


# ============================================================================
# Test Experiment Extract Trial Results
# ============================================================================

class TestExperimentExtractTrialResults:
    """Test Experiment _extract_trial_results method."""

    def test_extract_trial_results_returns_dict(self, mock_config):
        """Test that _extract_trial_results returns a dictionary."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert isinstance(result, dict)

    def test_extract_trial_results_includes_trial_number(self, mock_config):
        """Test that _extract_trial_results includes trial number."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["trial_number"] == 3

    def test_extract_trial_results_includes_number_generations(self, mock_config):
        """Test that _extract_trial_results includes number of generations."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["number_generations"] == 7

    def test_extract_trial_results_includes_max_fitness(self, mock_config):
        """Test that _extract_trial_results includes max fitness."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["max_fitness"] == 50.0

    def test_extract_trial_results_includes_number_neurons(self, mock_config):
        """Test that _extract_trial_results includes number of neurons."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["number_neurons"] == 5

    def test_extract_trial_results_includes_number_connections(self, mock_config):
        """Test that _extract_trial_results includes number of connections."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["number_connections"] == 10

    def test_extract_trial_results_includes_success_flag(self, mock_config):
        """Test that _extract_trial_results includes success flag."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = False

        result = experiment._extract_trial_results(trial, 3)

        assert result["success"] is True

    def test_extract_trial_results_success_false_when_failed(self, mock_config):
        """Test that _extract_trial_results sets success=False when trial failed."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        trial = ConcreteTrial(mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)
        trial._population = mock_population
        trial._generation_counter = 7
        trial.failed = True  # Failed trial

        result = experiment._extract_trial_results(trial, 3)

        assert result["success"] is False


# ============================================================================
# Test Experiment Analyze Trial Results
# ============================================================================

class TestExperimentAnalyzeTrialResults:
    """Test Experiment _analyze_trial_results method."""

    def test_analyze_trial_results_increments_success_counter(self, mock_config):
        """Test that _analyze_trial_results increments success counter for successful trials."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        results = {
            "trial_number": 1,
            "number_generations": 5,
            "max_fitness": 50.0,
            "number_neurons": 5,
            "number_connections": 10,
            "success": True
        }

        experiment._analyze_trial_results(results)

        assert experiment._success_counter == 1

    def test_analyze_trial_results_does_not_increment_for_failure(self, mock_config):
        """Test that _analyze_trial_results doesn't increment counter for failed trials."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        results = {
            "trial_number": 1,
            "number_generations": 5,
            "max_fitness": 50.0,
            "number_neurons": 5,
            "number_connections": 10,
            "success": False
        }

        experiment._analyze_trial_results(results)

        assert experiment._success_counter == 0

    def test_analyze_trial_results_appends_statistics(self, mock_config):
        """Test that _analyze_trial_results appends statistics for successful trials."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        results = {
            "trial_number": 1,
            "number_generations": 5,
            "max_fitness": 50.0,
            "number_neurons": 7,
            "number_connections": 12,
            "success": True
        }

        experiment._analyze_trial_results(results)

        assert experiment._number_generations == [5]
        assert experiment._max_fitness == [50.0]
        assert experiment._number_neurons == [7]
        assert experiment._number_connections == [12]

    def test_analyze_trial_results_does_not_append_for_failure(self, mock_config):
        """Test that _analyze_trial_results doesn't append statistics for failed trials."""
        experiment = ConcreteExperiment(ConcreteTrial, 5, mock_config)
        results = {
            "trial_number": 1,
            "number_generations": 5,
            "max_fitness": 50.0,
            "number_neurons": 7,
            "number_connections": 12,
            "success": False
        }

        experiment._analyze_trial_results(results)

        assert experiment._number_generations == []
        assert experiment._max_fitness == []
        assert experiment._number_neurons == []
        assert experiment._number_connections == []


# ============================================================================
# Test Experiment Run
# ============================================================================

class TestExperimentRun:
    """Test Experiment run method."""

    def test_run_calls_reset(self, mock_config):
        """Test that run calls _reset."""
        experiment = ConcreteExperiment(ConcreteTrial, 2, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment.run(num_jobs_trials=1, num_jobs_fitness=1)

        assert experiment.reset_called is True

    def test_run_serial_execution(self, mock_config):
        """Test that run executes trials serially when num_jobs_trials=1."""
        experiment = ConcreteExperiment(ConcreteTrial, 3, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment.run(num_jobs_trials=1, num_jobs_fitness=1)

        # Should have run 3 trials
        assert experiment._trial_counter == 3
        assert len(experiment.prepare_trial_calls) == 3

    def test_run_parallel_execution(self, mock_config):
        """Test that run executes trials in parallel when num_jobs_trials>1."""
        experiment = ConcreteExperiment(ConcreteTrial, 3, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            # In parallel mode, we can't rely on prepare_trial_calls being populated yet
            # So we need to set up the trial differently
            return None

        # Mock the _run_trial method to return fake results
        def mock_run_trial(trial_number, num_jobs):
            return {
                "trial_number": trial_number,
                "number_generations": 5,
                "max_fitness": 50.0,
                "number_neurons": 5,
                "number_connections": 10,
                "success": True
            }

        with patch.object(experiment, '_run_trial', side_effect=mock_run_trial):
            experiment.run(num_jobs_trials=2, num_jobs_fitness=1)

        # Should have run 3 trials
        assert experiment._trial_counter == 3

    def test_run_calls_analyze_for_each_result(self, mock_config):
        """Test that run calls _analyze_trial_results for each result."""
        experiment = ConcreteExperiment(ConcreteTrial, 2, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment.run(num_jobs_trials=1, num_jobs_fitness=1)

        # Should have called analyze_trial_results twice
        assert len(experiment.analyze_trial_results_calls) == 2

    def test_run_calls_final_report(self, mock_config):
        """Test that run calls _final_report."""
        experiment = ConcreteExperiment(ConcreteTrial, 2, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment.run(num_jobs_trials=1, num_jobs_fitness=1)

        assert experiment.final_report_called is True

    def test_run_passes_num_jobs_fitness_to_trials(self, mock_config):
        """Test that run passes num_jobs_fitness to each trial."""
        experiment = ConcreteExperiment(ConcreteTrial, 2, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run) as mock_run:
            experiment.run(num_jobs_trials=1, num_jobs_fitness=4)

        # Each trial should be called with num_jobs=4
        assert mock_run.call_count == 2
        for call_args in mock_run.call_args_list:
            assert call_args[0][0] == 4


# ============================================================================
# Test Experiment Integration
# ============================================================================

class TestExperimentIntegration:
    """Integration tests for Experiment."""

    def test_complete_experiment_workflow(self, mock_config):
        """Test complete experiment workflow from start to finish."""
        experiment = ConcreteExperiment(ConcreteTrial, 3, mock_config)

        mock_population = Mock(spec=Population)
        mock_individual = Mock(spec=Individual)
        mock_individual.fitness = 50.0
        mock_network = Mock(spec=NetworkStandard)
        mock_network.number_nodes = 5
        mock_network.number_connections_enabled = 10
        mock_individual._network = mock_network
        mock_population.get_fittest_individual = Mock(return_value=mock_individual)

        def mock_trial_run(num_jobs):
            trial_instance = experiment.prepare_trial_calls[-1][0]
            trial_instance._population = mock_population
            trial_instance._generation_counter = 5
            trial_instance.failed = False

        with patch.object(ConcreteTrial, 'run', side_effect=mock_trial_run):
            experiment.run(num_jobs_trials=1, num_jobs_fitness=1)

        # Should have reset
        assert experiment.reset_called is True
        # Should have run 3 trials
        assert experiment._trial_counter == 3
        # Should have prepared 3 trials
        assert len(experiment.prepare_trial_calls) == 3
        # Should have extracted results 3 times
        assert len(experiment.extract_trial_results_calls) == 3
        # Should have analyzed results 3 times
        assert len(experiment.analyze_trial_results_calls) == 3
        # Should have called final report
        assert experiment.final_report_called is True
        # Should have 3 successes
        assert experiment._success_counter == 3
