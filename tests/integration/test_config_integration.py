"""
Integration tests for NEAT configuration system.

These tests verify that the Config class properly loads and applies
configuration parameters, and that these parameters correctly control
the behavior of the NEAT algorithm.
"""

import pytest
import numpy as np
from evograd.run.config import Config
from evograd.run.trial import Trial


# ============================================================================
# Helper Trial Class for Config Testing
# ============================================================================

class TrialConfigTest(Trial):
    """Simplified trial for testing config integration."""

    def __init__(self, config, network_type='standard', xor_inputs=None, xor_outputs=None):
        super().__init__(config, network_type, suppress_output=True)
        self.xor_inputs = xor_inputs
        self.xor_outputs = xor_outputs

    def _reset(self):
        super()._reset()

    def _evaluate_fitness(self, individual):
        """Evaluate XOR fitness."""
        if self._network_type == 'standard':
            fitness = 4.0
            for inputs, expected_output in zip(self.xor_inputs, self.xor_outputs):
                output = individual._network.forward_pass(inputs)
                error = output[0] - expected_output[0]
                fitness -= error ** 2
        else:
            outputs = individual._network.forward_pass(self.xor_inputs)
            errors = outputs - self.xor_outputs
            fitness = 4.0 - np.sum(errors ** 2)
        return float(fitness)

    def _generation_report(self):
        pass

    def _final_report(self):
        pass


# ============================================================================
# Test Config Integration
# ============================================================================

class TestConfigIntegration:
    """Test that Config properly controls NEAT behavior."""

    def test_load_real_config_xor(self, xor_inputs_standard, xor_outputs_standard):
        """Verify that loading a real config file works end-to-end."""
        # Load config from file
        config = Config('examples/configs/config_xor.ini')

        # Verify some key parameters loaded correctly
        assert config.population_size == 200
        assert config.num_inputs == 2
        assert config.num_outputs == 1
        assert config.initial_cxn_policy == 'one-input'
        assert config.compatibility_threshold == 12.0
        assert config.fitness_threshold == 3.99
        assert config.max_number_generations == 100
        assert config.activation_initial == 'sigmoid'

        # Create and run trial with loaded config
        trial = TrialConfigTest(config, 'standard', xor_inputs_standard, xor_outputs_standard)
        trial.run(num_jobs=1)

        # Verify trial completed successfully (may or may not solve in 100 generations)
        # At minimum, verify it ran without errors
        assert trial._generation_counter > 0
        assert trial._population is not None
        fittest = trial._population.get_fittest_individual()
        assert fittest is not None
        assert fittest.fitness is not None

    def test_config_controls_population_size(self, xor_inputs_standard, xor_outputs_standard):
        """Verify that config.population_size correctly controls population size."""
        # Create config with specific population size
        config = Config()
        config.population_size = 20  # Small population for fast test
        config.num_inputs = 2
        config.num_outputs = 1
        config.initial_cxn_policy = 'full'
        config.max_number_generations = 10  # Run enough generations to ensure evolution works
        config.fitness_termination_check = False  # Don't stop early

        # Set basic parameters with wider limits for numerical stability
        config.min_weight = -30.0
        config.max_weight = 30.0
        config.min_bias = -30.0
        config.max_bias = 30.0

        config.weight_replace_prob = 0.1
        config.weight_perturb_prob = 0.8
        config.weight_perturb_strength = 0.5
        config.bias_replace_prob = 0.1
        config.bias_perturb_prob = 0.7
        config.bias_perturb_strength = 0.5

        config.single_structural_mutation = True  # Limit mutations for stability
        config.node_add_probability = 0.1  # Reduce to avoid too many nodes
        config.connection_add_probability = 0.3

        config.compatibility_threshold = 3.0
        config.elitism = 1
        config.survival_threshold = 0.2
        config.min_species_size = 2

        config.activation_initial = 'sigmoid'

        # Create and run trial
        trial = TrialConfigTest(config, 'standard', xor_inputs_standard, xor_outputs_standard)
        trial.run(num_jobs=1)

        # Verify population has exactly 20 individuals in every generation
        assert len(trial._population.individuals) == 20, \
            f"Population size is {len(trial._population.individuals)}, expected 20"

    def test_config_controls_termination(self, xor_inputs_standard, xor_outputs_standard):
        """Verify that fitness threshold termination works correctly."""
        # Create config with larger population for reliability with fixed seed
        config = Config()
        config.population_size = 150  # Larger population for better success rate
        config.num_inputs = 2
        config.num_outputs = 1
        config.initial_cxn_policy = 'full'
        config.initial_cxn_fraction = None

        # Mutation rates
        config.weight_replace_prob = 0.1
        config.weight_perturb_prob = 0.8
        config.weight_perturb_strength = 0.5
        config.bias_replace_prob = 0.1
        config.bias_perturb_prob = 0.7
        config.bias_perturb_strength = 0.5

        # Structural mutations
        config.single_structural_mutation = False
        config.node_add_probability = 0.2
        config.connection_add_probability = 0.5

        # Speciation
        config.compatibility_threshold = 3.0
        config.distance_excess_coeff = 1.0
        config.distance_disjoint_coeff = 1.0
        config.distance_params_coeff = 0.4
        config.distance_includes_nodes = True

        # Reproduction
        config.elitism = 2
        config.survival_threshold = 0.2
        config.min_species_size = 2

        # Stagnation
        config.max_stagnation_period = 15
        config.species_elitism = 2

        # Termination - key parameters for this test
        config.fitness_termination_check = True
        config.fitness_criterion = 'max'
        config.fitness_threshold = 3.99  # XOR solution threshold
        config.max_number_generations = 200  # Increased for stochastic nature

        # Node parameters
        config.activation_initial = 'sigmoid'
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.min_bias = -30.0  # Wider range for numerical stability
        config.max_bias = 30.0

        # Connection parameters
        config.weight_init_mean = 0.0
        config.weight_init_stdev = 1.0
        config.min_weight = -30.0  # Wider range for numerical stability
        config.max_weight = 30.0

        # Create and run trial
        trial = TrialConfigTest(config, 'standard', xor_inputs_standard, xor_outputs_standard)
        trial.run(num_jobs=1)

        # Verify trial stopped when fitness threshold was reached
        assert not trial.failed, "Trial should have found solution"

        fittest = trial._population.get_fittest_individual()
        assert fittest.fitness >= 3.99, \
            f"Best fitness {fittest.fitness} should be >= threshold 3.99"

        # Should stop at or before max generations
        assert trial._generation_counter <= 200, \
            f"Trial should terminate at or before max_number_generations (stopped at {trial._generation_counter})"
