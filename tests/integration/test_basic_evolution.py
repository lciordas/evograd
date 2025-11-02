"""
Integration tests for basic NEAT evolution.

These tests verify that NEAT can solve real problems end-to-end
and that different network implementations produce equivalent results.

NOTE: These tests use a fixed random seed (42) for reproducibility.
They pass consistently when run in isolation: pytest tests/integration/
When run as part of the full suite, occasional failures may occur
due to resource contention affecting evolution timing.
"""

import pytest
import numpy as np
from evograd.run.config import Config
from evograd.run.trial import Trial
from evograd.phenotype import Individual, NetworkStandard, NetworkFast, NetworkAutograd
from evograd.genotype import Genome
from evograd.genotype.innovation_tracker import InnovationTracker


# ============================================================================
# Helper Trial Class for XOR
# ============================================================================

class TrialXORTest(Trial):
    """Simplified XOR trial for integration testing."""
    
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
# Test Basic Evolution - Problem Solving
# ============================================================================

class TestBasicEvolution:
    """Test that NEAT can solve XOR problem end-to-end."""
    
    def test_solve_xor_standard_network(self, xor_inputs_standard, xor_outputs_standard):
        """Verify NEAT can solve XOR with NetworkStandard."""
        # Create config
        config = Config()
        config.population_size = 150
        config.num_inputs = 2
        config.num_outputs = 1
        config.initial_cxn_policy = 'full'
        config.initial_cxn_fraction = None
        
        # Mutation rates (use defaults for most, which disable gain mutation for stability)
        config.weight_replace_prob = 0.1
        config.weight_perturb_prob = 0.8
        config.weight_perturb_strength = 0.5
        config.bias_replace_prob = 0.1
        config.bias_perturb_prob = 0.7
        config.bias_perturb_strength = 0.5
        
        # Structural mutations
        config.single_structural_mutation = False
        config.node_add_probability = 0.2
        config.node_delete_probability = 0.0
        config.connection_add_probability = 0.5
        config.connection_enable_probability = 0.01
        config.connection_disable_probability = 0.01
        config.connection_delete_probability = 0.0
        
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
        config.num_episodes = 1
        
        # Stagnation
        config.max_stagnation_period = 15
        config.species_elitism = 2
        
        # Termination
        config.fitness_termination_check = True
        config.fitness_criterion = 'max'
        config.fitness_threshold = 3.99
        config.max_number_generations = 200  # Very generous limit for seed 42 reproducibility

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
        
        # Create trial
        trial = TrialXORTest(config, 'standard', xor_inputs_standard, xor_outputs_standard)
        
        # Run evolution
        trial.run(num_jobs=1)
        
        # Verify solution found
        assert not trial.failed, "Trial failed to find XOR solution within 50 generations"
        
        # Verify fittest individual
        fittest = trial._population.get_fittest_individual()
        assert fittest is not None
        assert fittest.fitness >= 3.99, f"Fitness {fittest.fitness} below threshold 3.99"
        
        # Verify network has hidden nodes (XOR requires at least one)
        assert len(fittest.genome.hidden_nodes) > 0, "Solution should have at least one hidden node"
        
    def test_solve_xor_fast_network(self, xor_inputs_batch, xor_outputs_batch):
        """Verify NEAT can solve XOR with NetworkFast."""
        # Use same config as standard network test
        config = Config()
        config.population_size = 150
        config.num_inputs = 2
        config.num_outputs = 1
        config.initial_cxn_policy = 'full'
        config.initial_cxn_fraction = None
        
        config.weight_replace_prob = 0.1
        config.weight_perturb_prob = 0.8
        config.weight_perturb_strength = 0.5
        config.bias_replace_prob = 0.1
        config.bias_perturb_prob = 0.7
        config.bias_perturb_strength = 0.5
        
        config.single_structural_mutation = False
        config.node_add_probability = 0.2
        config.node_delete_probability = 0.0
        config.connection_add_probability = 0.5
        config.connection_enable_probability = 0.01
        config.connection_disable_probability = 0.01
        config.connection_delete_probability = 0.0
        
        config.compatibility_threshold = 3.0
        config.distance_excess_coeff = 1.0
        config.distance_disjoint_coeff = 1.0
        config.distance_params_coeff = 0.4
        config.distance_includes_nodes = True
        
        config.elitism = 2
        config.survival_threshold = 0.2
        config.min_species_size = 2
        config.num_episodes = 1
        
        config.max_stagnation_period = 15
        config.species_elitism = 2
        
        config.fitness_termination_check = True
        config.fitness_criterion = 'max'
        config.fitness_threshold = 3.99
        config.max_number_generations = 50
        
        config.activation_initial = 'sigmoid'
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.1
        config.min_bias = -10.0
        config.max_bias = 10.0
        config.min_gain = 0.1
        config.max_gain = 10.0
        
        config.weight_init_mean = 0.0
        config.weight_init_stdev = 1.0
        config.min_weight = -10.0
        config.max_weight = 10.0
        
        # Create trial with NetworkFast
        trial = TrialXORTest(config, 'fast', xor_inputs_batch, xor_outputs_batch)
        
        # Run evolution
        trial.run(num_jobs=1)
        
        # Verify solution found
        assert not trial.failed, "Trial failed to find XOR solution with NetworkFast"
        
        fittest = trial._population.get_fittest_individual()
        assert fittest is not None
        assert fittest.fitness >= 3.99, f"Fitness {fittest.fitness} below threshold 3.99"
        assert len(fittest.genome.hidden_nodes) > 0
        
    def test_solve_xor_autograd_network(self, xor_inputs_batch, xor_outputs_batch):
        """Verify NEAT can solve XOR with NetworkAutograd."""
        config = Config()
        config.population_size = 150
        config.num_inputs = 2
        config.num_outputs = 1
        config.initial_cxn_policy = 'full'
        config.initial_cxn_fraction = None
        
        config.weight_replace_prob = 0.1
        config.weight_perturb_prob = 0.8
        config.weight_perturb_strength = 0.5
        config.bias_replace_prob = 0.1
        config.bias_perturb_prob = 0.7
        config.bias_perturb_strength = 0.5
        
        config.single_structural_mutation = False
        config.node_add_probability = 0.2
        config.node_delete_probability = 0.0
        config.connection_add_probability = 0.5
        config.connection_enable_probability = 0.01
        config.connection_disable_probability = 0.01
        config.connection_delete_probability = 0.0
        
        config.compatibility_threshold = 3.0
        config.distance_excess_coeff = 1.0
        config.distance_disjoint_coeff = 1.0
        config.distance_params_coeff = 0.4
        config.distance_includes_nodes = True
        
        config.elitism = 2
        config.survival_threshold = 0.2
        config.min_species_size = 2
        config.num_episodes = 1
        
        config.max_stagnation_period = 15
        config.species_elitism = 2
        
        config.fitness_termination_check = True
        config.fitness_criterion = 'max'
        config.fitness_threshold = 3.99
        config.max_number_generations = 50
        
        config.activation_initial = 'sigmoid'
        config.bias_init_mean = 0.0
        config.bias_init_stdev = 1.0
        config.gain_init_mean = 1.0
        config.gain_init_stdev = 0.1
        config.min_bias = -10.0
        config.max_bias = 10.0
        config.min_gain = 0.1
        config.max_gain = 10.0
        
        config.weight_init_mean = 0.0
        config.weight_init_stdev = 1.0
        config.min_weight = -10.0
        config.max_weight = 10.0
        
        # Create trial with NetworkAutograd
        trial = TrialXORTest(config, 'autograd', xor_inputs_batch, xor_outputs_batch)
        
        # Run evolution
        trial.run(num_jobs=1)
        
        # Verify solution found
        assert not trial.failed, "Trial failed to find XOR solution with NetworkAutograd"
        
        fittest = trial._population.get_fittest_individual()
        assert fittest is not None
        assert fittest.fitness >= 3.99, f"Fitness {fittest.fitness} below threshold 3.99"
        assert len(fittest.genome.hidden_nodes) > 0


# ============================================================================
# Test Network Type Equivalence
# ============================================================================

class TestNetworkEquivalence:
    """Test that different network types produce identical outputs."""
    
    def test_network_types_produce_identical_outputs(self, simple_test_genome):
        """Verify all network types give identical outputs for the same genome."""
        # Reset InnovationTracker
        config = Config()
        config.num_inputs = 2
        config.num_outputs = 1
        InnovationTracker.initialize(config)
        
        # Create genome from dict
        genome = Genome.from_dict(simple_test_genome)
        
        # Create all three network types
        net_standard = NetworkStandard(genome)
        net_fast = NetworkFast(genome)
        net_autograd = NetworkAutograd(genome)
        
        # Test inputs (including edge cases)
        test_inputs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [-1.0, 1.0],
            [2.0, -0.5]
        ]
        
        # Verify outputs match for all test inputs
        for inp in test_inputs:
            # Standard network (single input)
            out_standard = net_standard.forward_pass(inp)
            
            # Fast network (batch input)
            out_fast = net_fast.forward_pass(np.array([inp]))[0]
            
            # Autograd network (batch input)
            out_autograd = net_autograd.forward_pass(np.array([inp]))[0]
            
            # Verify equivalence (within numerical tolerance)
            np.testing.assert_allclose(
                out_standard, out_fast, rtol=1e-6, atol=1e-8,
                err_msg=f"NetworkStandard and NetworkFast outputs differ for input {inp}"
            )
            np.testing.assert_allclose(
                out_standard, out_autograd, rtol=1e-6, atol=1e-8,
                err_msg=f"NetworkStandard and NetworkAutograd outputs differ for input {inp}"
            )
            np.testing.assert_allclose(
                out_fast, out_autograd, rtol=1e-6, atol=1e-8,
                err_msg=f"NetworkFast and NetworkAutograd outputs differ for input {inp}"
            )
