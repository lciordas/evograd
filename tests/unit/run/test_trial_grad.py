"""
Unit tests for TrialGrad class.
"""

import pytest
import autograd.numpy as np   # type: ignore
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

from evograd.run.trial_grad import TrialGrad
from evograd.run.config import Config
from evograd.pool.population import Population
from evograd.phenotype.individual import Individual
from evograd.phenotype.network_autograd import NetworkAutograd


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock Config object with gradient descent settings."""
    config = Mock(spec=Config)
    config.max_number_generations = 10
    config.fitness_termination_check = False
    config.num_inputs = 2
    config.num_outputs = 1

    # Gradient descent settings
    config.enable_gradient = False
    config.gradient_steps = 10
    config.learning_rate = 0.01
    config.gradient_frequency = 1
    config.gradient_selection = 'top_k'
    config.gradient_top_k = 5
    config.gradient_top_percent = 0.1
    config.lamarckian_evolution = False

    # Freeze settings
    config.freeze_weights = False
    config.freeze_biases = False
    config.freeze_gains = False
    config.freeze_activation_coeffs = False

    return config


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================

class ConcreteTrialGrad(TrialGrad):
    """Concrete implementation of TrialGrad for testing purposes."""

    def __init__(self, config, network_type='autograd', suppress_output=False):
        super().__init__(config, network_type, suppress_output)
        self.reset_called = False
        self.loss_function_calls = []
        self.loss_to_fitness_calls = []
        self.get_training_data_calls = 0

    def _reset(self):
        super()._reset()
        self.reset_called = True

    def _loss_function(self, outputs, targets):
        self.loss_function_calls.append((outputs, targets))
        # Simple MSE
        return float(np.mean((outputs - targets) ** 2))

    def _loss_to_fitness(self, loss):
        self.loss_to_fitness_calls.append(loss)
        # Inverse with offset
        return 1.0 / (0.1 + loss)

    def _get_training_data(self):
        self.get_training_data_calls += 1
        # Return simple training data (2 inputs, 1 output, 5 samples)
        inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        targets = np.array([[0.0], [1.0], [1.0], [0.0], [0.5]])
        return inputs, targets

    def _generation_report(self):
        pass

    def _final_report(self):
        pass


# ============================================================================
# Test TrialGrad Initialization
# ============================================================================

class TestTrialGradInit:
    """Test TrialGrad initialization."""

    def test_cannot_instantiate_abstract_class(self, mock_config):
        """Test that TrialGrad cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrialGrad(mock_config, 'autograd')

    def test_init_with_autograd_network_type(self, mock_config):
        """Test initialization with autograd network type."""
        trial = ConcreteTrialGrad(mock_config, network_type='autograd')
        assert trial._network_type == 'autograd'

    def test_init_gradient_disabled_allows_non_autograd(self, mock_config):
        """Test that non-autograd network types are allowed when gradient disabled."""
        mock_config.enable_gradient = False
        trial = ConcreteTrialGrad(mock_config, network_type='standard')
        assert trial._network_type == 'standard'

    def test_init_gradient_enabled_requires_autograd(self, mock_config):
        """Test that gradient enabled requires autograd network type."""
        mock_config.enable_gradient = True
        with pytest.raises(ValueError, match="Gradient descent requires network_type='autograd'"):
            ConcreteTrialGrad(mock_config, network_type='standard')

    def test_init_initializes_gradient_data(self, mock_config):
        """Test that __init__ initializes gradient data dictionary."""
        trial = ConcreteTrialGrad(mock_config)
        assert trial._gradient_data == {}


# ============================================================================
# Test TrialGrad Reset
# ============================================================================

class TestTrialGradReset:
    """Test TrialGrad reset."""

    def test_reset_clears_gradient_data(self, mock_config):
        """Test that _reset clears gradient data."""
        trial = ConcreteTrialGrad(mock_config)
        trial._gradient_data = {1: {'fitness': 10.0}, 2: {'fitness': 20.0}}
        trial._reset()
        assert trial._gradient_data == {}


# ============================================================================
# Test TrialGrad Evaluate Fitness
# ============================================================================

class TestTrialGradEvaluateFitness:
    """Test TrialGrad _evaluate_fitness method."""

    def test_evaluate_fitness_calls_get_training_data(self, mock_config):
        """Test that _evaluate_fitness calls _get_training_data."""
        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)
        mock_network.forward_pass = Mock(return_value=np.array([[0.5]]))
        mock_individual._network = mock_network

        fitness = trial._evaluate_fitness(mock_individual)

        assert trial.get_training_data_calls == 1

    def test_evaluate_fitness_calls_loss_function(self, mock_config):
        """Test that _evaluate_fitness calls _loss_function."""
        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)
        outputs = np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])
        mock_network.forward_pass = Mock(return_value=outputs)
        mock_individual._network = mock_network

        fitness = trial._evaluate_fitness(mock_individual)

        assert len(trial.loss_function_calls) == 1

    def test_evaluate_fitness_calls_loss_to_fitness(self, mock_config):
        """Test that _evaluate_fitness calls _loss_to_fitness."""
        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)
        outputs = np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])
        mock_network.forward_pass = Mock(return_value=outputs)
        mock_individual._network = mock_network

        fitness = trial._evaluate_fitness(mock_individual)

        assert len(trial.loss_to_fitness_calls) == 1

    def test_evaluate_fitness_returns_positive_value(self, mock_config):
        """Test that _evaluate_fitness returns a positive fitness value."""
        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)
        outputs = np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])
        mock_network.forward_pass = Mock(return_value=outputs)
        mock_individual._network = mock_network

        fitness = trial._evaluate_fitness(mock_individual)

        assert fitness > 0


# ============================================================================
# Test TrialGrad Select GD Individuals
# ============================================================================

class TestTrialGradSelectGDIndividuals:
    """Test TrialGrad _select_GD_individuals method."""

    def test_select_all_individuals(self, mock_config):
        """Test selection with 'all' strategy."""
        mock_config.gradient_selection = 'all'
        trial = ConcreteTrialGrad(mock_config)

        # Create mock population with individuals
        mock_population = Mock(spec=Population)
        individuals = [Mock(spec=Individual) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.ID = i
            ind.fitness = float(i * 10)
        mock_population.individuals = individuals
        trial._population = mock_population

        selected = trial._select_GD_individuals()

        assert len(selected) == 10
        assert selected == individuals

    def test_select_top_k_individuals(self, mock_config):
        """Test selection with 'top_k' strategy."""
        mock_config.gradient_selection = 'top_k'
        mock_config.gradient_top_k = 3
        trial = ConcreteTrialGrad(mock_config)

        mock_population = Mock(spec=Population)
        individuals = [Mock(spec=Individual) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.ID = i
            ind.fitness = float(i * 10)
        mock_population.individuals = individuals
        trial._population = mock_population

        selected = trial._select_GD_individuals()

        assert len(selected) == 3
        # Should select top 3 by fitness
        assert selected[0].fitness == 90.0
        assert selected[1].fitness == 80.0
        assert selected[2].fitness == 70.0

    def test_select_top_percent_individuals(self, mock_config):
        """Test selection with 'top_percent' strategy."""
        mock_config.gradient_selection = 'top_percent'
        mock_config.gradient_top_percent = 0.2  # 20%
        trial = ConcreteTrialGrad(mock_config)

        mock_population = Mock(spec=Population)
        individuals = [Mock(spec=Individual) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.ID = i
            ind.fitness = float(i * 10)
        mock_population.individuals = individuals
        trial._population = mock_population

        selected = trial._select_GD_individuals()

        # 20% of 10 = 2
        assert len(selected) == 2
        assert selected[0].fitness == 90.0
        assert selected[1].fitness == 80.0

    def test_select_top_percent_minimum_one(self, mock_config):
        """Test that top_percent selects at least one individual."""
        mock_config.gradient_selection = 'top_percent'
        mock_config.gradient_top_percent = 0.01  # 1% of 10 = 0.1, should round to 1
        trial = ConcreteTrialGrad(mock_config)

        mock_population = Mock(spec=Population)
        individuals = [Mock(spec=Individual) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.ID = i
            ind.fitness = float(i * 10)
        mock_population.individuals = individuals
        trial._population = mock_population

        selected = trial._select_GD_individuals()

        assert len(selected) >= 1

    def test_select_unknown_strategy_raises_error(self, mock_config):
        """Test that unknown gradient_selection raises ValueError."""
        mock_config.gradient_selection = 'unknown_strategy'
        trial = ConcreteTrialGrad(mock_config)

        mock_population = Mock(spec=Population)
        mock_population.individuals = []
        trial._population = mock_population

        with pytest.raises(ValueError, match="Unknown gradient_selection"):
            trial._select_GD_individuals()


# ============================================================================
# Test TrialGrad Flatten/Unflatten Activation Coeffs
# ============================================================================

class TestTrialGradActivationCoeffs:
    """Test TrialGrad activation coefficient flattening/unflattening."""

    def test_flatten_empty_coeffs(self, mock_config):
        """Test flattening empty coefficient dictionary."""
        trial = ConcreteTrialGrad(mock_config)

        flat_array, metadata = trial._flatten_activation_coeffs({})

        assert len(flat_array) == 0
        assert metadata['node_indices'] == []
        assert metadata['shapes'] == []
        assert metadata['start_indices'] == []

    def test_flatten_single_node_coeffs(self, mock_config):
        """Test flattening coefficients for single node."""
        trial = ConcreteTrialGrad(mock_config)

        coeffs_dict = {5: np.array([1.0, 2.0, 3.0])}
        flat_array, metadata = trial._flatten_activation_coeffs(coeffs_dict)

        assert len(flat_array) == 3
        assert np.allclose(flat_array, [1.0, 2.0, 3.0])
        assert metadata['node_indices'] == [5]
        assert metadata['shapes'] == [(3,)]

    def test_flatten_multiple_node_coeffs(self, mock_config):
        """Test flattening coefficients for multiple nodes."""
        trial = ConcreteTrialGrad(mock_config)

        coeffs_dict = {
            3: np.array([1.0, 2.0]),
            7: np.array([3.0, 4.0, 5.0]),
            5: np.array([6.0])
        }
        flat_array, metadata = trial._flatten_activation_coeffs(coeffs_dict)

        # Should be sorted by node index: 3, 5, 7
        assert len(flat_array) == 6
        assert np.allclose(flat_array, [1.0, 2.0, 6.0, 3.0, 4.0, 5.0])
        assert metadata['node_indices'] == [3, 5, 7]

    def test_unflatten_empty_coeffs(self, mock_config):
        """Test unflattening empty coefficient array."""
        trial = ConcreteTrialGrad(mock_config)

        metadata = {'node_indices': [], 'shapes': [], 'start_indices': []}
        coeffs_dict = trial._unflatten_activation_coeffs(np.array([]), metadata)

        assert coeffs_dict == {}

    def test_unflatten_single_node_coeffs(self, mock_config):
        """Test unflattening coefficients for single node."""
        trial = ConcreteTrialGrad(mock_config)

        flat_array = np.array([1.0, 2.0, 3.0])
        metadata = {
            'node_indices': [5],
            'shapes': [(3,)],
            'start_indices': [0]
        }
        coeffs_dict = trial._unflatten_activation_coeffs(flat_array, metadata)

        assert 5 in coeffs_dict
        assert np.allclose(coeffs_dict[5], [1.0, 2.0, 3.0])

    def test_unflatten_multiple_node_coeffs(self, mock_config):
        """Test unflattening coefficients for multiple nodes."""
        trial = ConcreteTrialGrad(mock_config)

        flat_array = np.array([1.0, 2.0, 6.0, 3.0, 4.0, 5.0])
        metadata = {
            'node_indices': [3, 5, 7],
            'shapes': [(2,), (1,), (3,)],
            'start_indices': [0, 2, 3]
        }
        coeffs_dict = trial._unflatten_activation_coeffs(flat_array, metadata)

        assert 3 in coeffs_dict
        assert 5 in coeffs_dict
        assert 7 in coeffs_dict
        assert np.allclose(coeffs_dict[3], [1.0, 2.0])
        assert np.allclose(coeffs_dict[5], [6.0])
        assert np.allclose(coeffs_dict[7], [3.0, 4.0, 5.0])

    def test_flatten_unflatten_roundtrip(self, mock_config):
        """Test that flatten->unflatten preserves coefficient dictionary."""
        trial = ConcreteTrialGrad(mock_config)

        original_coeffs = {
            3: np.array([1.0, 2.0]),
            7: np.array([3.0, 4.0, 5.0]),
            5: np.array([6.0])
        }

        flat_array, metadata = trial._flatten_activation_coeffs(original_coeffs)
        reconstructed_coeffs = trial._unflatten_activation_coeffs(flat_array, metadata)

        assert set(reconstructed_coeffs.keys()) == set(original_coeffs.keys())
        for node_idx in original_coeffs:
            assert np.allclose(reconstructed_coeffs[node_idx], original_coeffs[node_idx])


# ============================================================================
# Test TrialGrad GD Optimize
# ============================================================================

class TestTrialGradGDOptimize:
    """Test TrialGrad _GD_optimize method."""

    def test_gd_optimize_all_frozen_returns_unchanged(self, mock_config):
        """Test that _GD_optimize returns unchanged parameters when all frozen."""
        mock_config.freeze_weights = True
        mock_config.freeze_biases = True
        mock_config.freeze_gains = True
        mock_config.freeze_activation_coeffs = True

        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)

        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        biases = np.array([0.5, 0.6])
        gains = np.array([1.0, 1.0])
        coeffs = {}

        mock_network.get_parameters = Mock(return_value=(weights, biases, gains, coeffs))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_individual._network = mock_network

        results = trial._GD_optimize(mock_individual)

        # Should return original parameters unchanged
        assert np.allclose(results['optimized_weights'], weights)
        assert np.allclose(results['optimized_biases'], biases)
        assert np.allclose(results['optimized_gains'], gains)
        assert results['loss_improvement'] == 0.0
        assert results['fitness_improvement'] == 0.0

    def test_gd_optimize_returns_dict_with_required_keys(self, mock_config):
        """Test that _GD_optimize returns dict with all required keys."""
        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)

        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        biases = np.array([0.5, 0.6])
        gains = np.array([1.0, 1.0])
        coeffs = {}

        mock_network.get_parameters = Mock(return_value=(weights, biases, gains, coeffs))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_individual._network = mock_network

        with patch('neat.run.trial_grad.adam', return_value=np.concatenate([weights.flatten(), biases.flatten(), gains.flatten()])):
            results = trial._GD_optimize(mock_individual)

        required_keys = [
            'initial_loss', 'final_loss', 'loss_improvement',
            'initial_fitness', 'final_fitness', 'fitness_improvement',
            'optimized_weights', 'optimized_biases', 'optimized_gains', 'optimized_coeffs'
        ]
        for key in required_keys:
            assert key in results

    def test_gd_optimize_calls_adam_optimizer(self, mock_config):
        """Test that _GD_optimize calls adam optimizer."""
        mock_config.gradient_steps = 5
        mock_config.learning_rate = 0.01

        trial = ConcreteTrialGrad(mock_config)

        mock_individual = Mock(spec=Individual)
        mock_network = Mock(spec=NetworkAutograd)

        weights = np.array([[1.0, 2.0]])
        biases = np.array([0.5])
        gains = np.array([1.0])
        coeffs = {}

        mock_network.get_parameters = Mock(return_value=(weights, biases, gains, coeffs))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_individual._network = mock_network

        with patch('neat.run.trial_grad.adam') as mock_adam:
            mock_adam.return_value = np.concatenate([weights.flatten(), biases.flatten(), gains.flatten()])
            trial._GD_optimize(mock_individual)

            mock_adam.assert_called_once()
            # Check that num_iters and step_size were passed correctly
            call_kwargs = mock_adam.call_args[1]
            assert call_kwargs['num_iters'] == 5
            assert call_kwargs['step_size'] == 0.01


# ============================================================================
# Test TrialGrad Evaluate Fitness All
# ============================================================================

class TestTrialGradEvaluateFitnessAll:
    """Test TrialGrad _evaluate_fitness_all method with gradient descent."""

    def test_evaluate_fitness_all_gradient_disabled(self, mock_config):
        """Test that _evaluate_fitness_all skips gradient descent when disabled."""
        mock_config.enable_gradient = False
        trial = ConcreteTrialGrad(mock_config)

        # Mock population
        mock_population = Mock(spec=Population)
        mock_population.individuals = []
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        # Mock parent's _evaluate_fitness_all
        with patch('neat.run.trial.Trial._evaluate_fitness_all') as mock_parent:
            trial._evaluate_fitness_all(num_jobs=1)

            # Should call parent method
            mock_parent.assert_called_once_with(1)

            # Should not store any gradient data
            assert trial._gradient_data == {}

    def test_evaluate_fitness_all_gradient_enabled_but_wrong_frequency(self, mock_config):
        """Test that gradient descent is skipped when generation counter doesn't match frequency."""
        mock_config.enable_gradient = True
        mock_config.gradient_frequency = 5
        trial = ConcreteTrialGrad(mock_config)
        trial._generation_counter = 3  # Not divisible by 5

        mock_population = Mock(spec=Population)
        mock_population.individuals = []
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        with patch('neat.run.trial.Trial._evaluate_fitness_all'):
            trial._evaluate_fitness_all(num_jobs=1)

            # Should not apply gradient descent
            assert trial._gradient_data == {}

    def test_evaluate_fitness_all_gradient_enabled_and_correct_frequency(self, mock_config):
        """Test that gradient descent is applied when enabled and at correct frequency."""
        mock_config.enable_gradient = True
        mock_config.gradient_frequency = 2
        mock_config.gradient_selection = 'top_k'
        mock_config.gradient_top_k = 2
        mock_config.lamarckian_evolution = False

        trial = ConcreteTrialGrad(mock_config)
        trial._generation_counter = 4  # Divisible by 2

        # Create mock individuals with networks
        mock_individuals = []
        for i in range(3):
            mock_ind = Mock(spec=Individual)
            mock_ind.ID = i
            mock_ind.fitness = float(i * 10)

            mock_network = Mock(spec=NetworkAutograd)
            mock_network.get_parameters = Mock(return_value=(
                np.array([[1.0]]), np.array([0.5]), np.array([1.0]), {}
            ))
            mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
            mock_network.set_parameters = Mock()
            mock_network.save_parameters_to_genome = Mock()
            mock_ind._network = mock_network

            mock_individuals.append(mock_ind)

        mock_population = Mock(spec=Population)
        mock_population.individuals = mock_individuals
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        with patch('neat.run.trial.Trial._evaluate_fitness_all'):
            with patch('neat.run.trial_grad.adam', return_value=np.array([1.0, 0.5, 1.0])):
                trial._evaluate_fitness_all(num_jobs=1)

        # Should have applied gradient descent to top 2 individuals
        assert len(trial._gradient_data) == 2

    def test_evaluate_fitness_all_lamarckian_saves_to_genome(self, mock_config):
        """Test that Lamarckian evolution saves optimized parameters to genome."""
        mock_config.enable_gradient = True
        mock_config.gradient_frequency = 1
        mock_config.gradient_selection = 'all'
        mock_config.lamarckian_evolution = True

        trial = ConcreteTrialGrad(mock_config)
        trial._generation_counter = 1

        mock_ind = Mock(spec=Individual)
        mock_ind.ID = 0
        mock_ind.fitness = 10.0

        mock_network = Mock(spec=NetworkAutograd)
        mock_network.get_parameters = Mock(return_value=(
            np.array([[1.0]]), np.array([0.5]), np.array([1.0]), {}
        ))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_network.set_parameters = Mock()
        mock_network.save_parameters_to_genome = Mock()
        mock_ind._network = mock_network

        mock_population = Mock(spec=Population)
        mock_population.individuals = [mock_ind]
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        with patch('neat.run.trial.Trial._evaluate_fitness_all'):
            with patch('neat.run.trial_grad.adam', return_value=np.array([1.0, 0.5, 1.0])):
                trial._evaluate_fitness_all(num_jobs=1)

        # Should have called save_parameters_to_genome
        mock_network.save_parameters_to_genome.assert_called_once()

    def test_evaluate_fitness_all_baldwin_does_not_save_to_genome(self, mock_config):
        """Test that Baldwin effect (non-Lamarckian) does not save to genome."""
        mock_config.enable_gradient = True
        mock_config.gradient_frequency = 1
        mock_config.gradient_selection = 'all'
        mock_config.lamarckian_evolution = False

        trial = ConcreteTrialGrad(mock_config)
        trial._generation_counter = 1

        mock_ind = Mock(spec=Individual)
        mock_ind.ID = 0
        mock_ind.fitness = 10.0

        mock_network = Mock(spec=NetworkAutograd)
        mock_network.get_parameters = Mock(return_value=(
            np.array([[1.0]]), np.array([0.5]), np.array([1.0]), {}
        ))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_network.set_parameters = Mock()
        mock_network.save_parameters_to_genome = Mock()
        mock_ind._network = mock_network

        mock_population = Mock(spec=Population)
        mock_population.individuals = [mock_ind]
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        with patch('neat.run.trial.Trial._evaluate_fitness_all'):
            with patch('neat.run.trial_grad.adam', return_value=np.array([1.0, 0.5, 1.0])):
                trial._evaluate_fitness_all(num_jobs=1)

        # Should NOT have called save_parameters_to_genome
        mock_network.save_parameters_to_genome.assert_not_called()


# ============================================================================
# Test TrialGrad Report GD Statistics
# ============================================================================

class TestTrialGradReportGDStatistics:
    """Test TrialGrad _report_GD_statistics method."""

    def test_report_gd_statistics_empty_data(self, mock_config):
        """Test reporting with no gradient data."""
        trial = ConcreteTrialGrad(mock_config)
        trial._gradient_data = {}

        report = trial._report_GD_statistics()

        assert report == ""

    def test_report_gd_statistics_with_data(self, mock_config):
        """Test reporting with gradient data."""
        trial = ConcreteTrialGrad(mock_config)
        trial._gradient_data = {
            1: {'fitness_improvement': 5.0},
            2: {'fitness_improvement': 10.0},
            3: {'fitness_improvement': 15.0}
        }

        report = trial._report_GD_statistics()

        assert "10.000000" in report  # Average is (5+10+15)/3 = 10
        assert "fitness improvement" in report.lower()


# ============================================================================
# Test TrialGrad Integration
# ============================================================================

class TestTrialGradIntegration:
    """Integration tests for TrialGrad."""

    def test_complete_gradient_training_workflow(self, mock_config):
        """Test complete gradient training workflow."""
        mock_config.enable_gradient = True
        mock_config.gradient_frequency = 1
        mock_config.gradient_selection = 'top_k'
        mock_config.gradient_top_k = 1
        mock_config.lamarckian_evolution = False
        mock_config.gradient_steps = 5
        mock_config.learning_rate = 0.01

        trial = ConcreteTrialGrad(mock_config)
        trial._generation_counter = 1

        # Create mock individual
        mock_ind = Mock(spec=Individual)
        mock_ind.ID = 0
        mock_ind.fitness = 10.0

        mock_network = Mock(spec=NetworkAutograd)
        mock_network.get_parameters = Mock(return_value=(
            np.array([[1.0, 2.0]]), np.array([0.5]), np.array([1.0]), {}
        ))
        mock_network.forward_pass = Mock(return_value=np.array([[0.5], [0.6], [0.7], [0.8], [0.9]]))
        mock_network.set_parameters = Mock()
        mock_ind._network = mock_network

        mock_population = Mock(spec=Population)
        mock_population.individuals = [mock_ind]
        mock_population._species_manager = Mock()
        mock_population._species_manager.update_fitness = Mock()
        trial._population = mock_population

        with patch('neat.run.trial.Trial._evaluate_fitness_all'):
            with patch('neat.run.trial_grad.adam', return_value=np.array([1.5, 2.5, 0.6, 1.1])):
                trial._evaluate_fitness_all(num_jobs=1)

        # Should have updated individual's fitness
        assert mock_ind.fitness != 10.0  # Fitness should change

        # Should have updated network parameters
        mock_network.set_parameters.assert_called_once()

        # Should have gradient data stored
        assert 0 in trial._gradient_data
        assert 'fitness_before_gd' in trial._gradient_data[0]
        assert 'fitness_after_gd' in trial._gradient_data[0]
        assert 'fitness_improvement' in trial._gradient_data[0]
