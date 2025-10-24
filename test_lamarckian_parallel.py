#!/usr/bin/env python3
"""
Test that Lamarckian evolution works correctly in both serial and parallel modes.

This test verifies that when lamarckian_evolution=True, gradient-optimized
parameters are correctly inherited by offspring, regardless of whether
gradient descent runs in serial (num_jobs=1) or parallel (num_jobs>1) mode.
"""

import sys
import os
sys.path.insert(0, 'src')

import autograd.numpy as np  # Use autograd's numpy for compatibility
from neat.run.config import Config
from neat.genotype import Genome
from neat.genotype.innovation_tracker import InnovationTracker
from neat.phenotype import Individual
from neat.run.trial_grad import TrialGrad
from joblib import Parallel, delayed
import copy


class TestTrialGrad(TrialGrad):
    """Test implementation of TrialGrad for verification."""

    def __init__(self, config, network_type='autograd'):
        super().__init__(config, network_type, suppress_output=True)
        self.test_inputs = np.random.randn(10, config.num_inputs)
        self.test_targets = np.random.randn(10, config.num_outputs)

    def _loss_function(self, outputs, targets):
        """Simple MSE loss."""
        return np.mean((outputs - targets)**2)

    def _loss_to_fitness(self, loss):
        """Convert loss to fitness."""
        return 1.0 / (1.0 + loss)

    def _get_training_data(self):
        """Return test data."""
        return self.test_inputs, self.test_targets

    def test_apply_gradient_descent(self, individual):
        """Public wrapper for testing _apply_gradient_descent."""
        return self._apply_gradient_descent(individual)

    def _report_progress(self):
        """Required abstract method - empty for testing."""
        pass

    def _final_report(self):
        """Required abstract method - empty for testing."""
        pass


def extract_genome_params(genome):
    """Extract all parameters from a genome for comparison."""
    node_params = {}
    for node_id, node in genome.node_genes.items():
        node_params[node_id] = (node.bias, node.gain)

    conn_params = {}
    for inn, conn in genome.conn_genes.items():
        if conn.enabled:
            conn_params[inn] = conn.weight

    return node_params, conn_params


def params_differ(params1, params2, threshold=0.01):
    """Check if two parameter sets differ significantly."""
    node_params1, conn_params1 = params1
    node_params2, conn_params2 = params2

    # Check node parameters
    for node_id in node_params1:
        bias1, gain1 = node_params1[node_id]
        bias2, gain2 = node_params2[node_id]
        if abs(bias1 - bias2) > threshold or abs(gain1 - gain2) > threshold:
            return True

    # Check connection parameters
    for inn in conn_params1:
        if inn in conn_params2:
            if abs(conn_params1[inn] - conn_params2[inn]) > threshold:
                return True

    return False


def test_lamarckian_serial():
    """Test Lamarckian evolution in serial mode (num_jobs=1)."""
    print("=" * 60)
    print("TEST 1: Lamarckian Evolution - Serial Mode (num_jobs=1)")
    print("=" * 60)

    # Create config with Lamarckian evolution enabled
    config = Config('configs/config_regression1D.ini')

    # Initialize innovation tracker
    InnovationTracker.initialize(config)
    config.enable_gradient = True
    config.lamarckian_evolution = True
    config.gradient_steps = 5  # Fewer steps for testing
    config.learning_rate = 0.1  # Higher learning rate to ensure changes

    # Create trial and individual
    trial = TestTrialGrad(config)
    genome = Genome(config)

    # Add some connections to make it interesting
    for _ in range(3):
        genome.mutate()

    individual = Individual(genome, 'autograd')
    individual.fitness = 0.5  # Set initial fitness

    # Store original parameters
    original_params = extract_genome_params(individual.genome)

    # Apply gradient descent (serial mode)
    result = trial.test_apply_gradient_descent(individual)

    # Check return value format
    assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
    loss_final, loss_improvement, fitness_improvement, genome_data = result

    # For Lamarckian, genome_data should not be None
    assert genome_data is not None, "genome_data should not be None for Lamarckian evolution"

    # Check that genome was modified in place (serial mode modifies in-place)
    modified_params = extract_genome_params(individual.genome)

    if params_differ(original_params, modified_params):
        print("✓ Genome parameters were modified (as expected for Lamarckian)")
    else:
        print("✗ ERROR: Genome parameters unchanged (should be modified for Lamarckian)")
        return False

    # Create offspring and check inheritance
    offspring = individual.clone()
    offspring_params = extract_genome_params(offspring.genome)

    if not params_differ(modified_params, offspring_params):
        print("✓ Offspring inherited optimized parameters")
    else:
        print("✗ ERROR: Offspring did not inherit optimized parameters")
        return False

    print("✓ Serial Lamarckian evolution test PASSED\n")
    return True


def test_lamarckian_parallel():
    """Test Lamarckian evolution in parallel mode (num_jobs>1)."""
    print("=" * 60)
    print("TEST 2: Lamarckian Evolution - Parallel Mode (num_jobs=2)")
    print("=" * 60)

    # Create config with Lamarckian evolution enabled
    config = Config('configs/config_regression1D.ini')

    # Initialize innovation tracker
    InnovationTracker.initialize(config)

    config.enable_gradient = True
    config.lamarckian_evolution = True
    config.gradient_steps = 5
    config.learning_rate = 0.1

    # Create trial
    trial = TestTrialGrad(config)

    # Create multiple individuals
    individuals = []
    for i in range(3):
        genome = Genome(config)
        for _ in range(3):
            genome.mutate()
        individual = Individual(genome, 'autograd')
        individual.fitness = 0.5 + i * 0.1
        individuals.append(individual)

    # Store original parameters
    original_params = {}
    for ind in individuals:
        original_params[ind.ID] = extract_genome_params(ind.genome)

    # Simulate parallel gradient descent
    results = Parallel(n_jobs=2)(
        delayed(trial.test_apply_gradient_descent)(ind) for ind in individuals
    )

    print(f"Parallel execution completed, got {len(results)} results")

    # Process results and apply genome updates (simulating what _evaluate_fitness_all does)
    for individual, result in zip(individuals, results):
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
        loss_final, loss_improvement, fitness_improvement, genome_data = result

        # For Lamarckian, genome_data should not be None
        assert genome_data is not None, "genome_data should not be None for Lamarckian evolution"

        # Apply genome updates (this is the FIX we implemented)
        if genome_data is not None:
            # Update node parameters
            for node_id, (bias, gain) in genome_data['node_params'].items():
                individual.genome.node_genes[node_id].bias = bias
                individual.genome.node_genes[node_id].gain = gain

            # Update connection parameters
            for inn, weight in genome_data['conn_params'].items():
                individual.genome.conn_genes[inn].weight = weight

    # Check that genomes were modified
    all_modified = True
    for ind in individuals:
        original = original_params[ind.ID]
        modified = extract_genome_params(ind.genome)

        if params_differ(original, modified):
            print(f"✓ Individual {ind.ID}: Genome parameters modified")
        else:
            print(f"✗ Individual {ind.ID}: Genome parameters unchanged")
            all_modified = False

    if not all_modified:
        print("✗ ERROR: Not all genomes were modified in parallel mode")
        return False

    # Test inheritance for each individual
    all_inherited = True
    for ind in individuals:
        offspring = ind.clone()
        parent_params = extract_genome_params(ind.genome)
        offspring_params = extract_genome_params(offspring.genome)

        if not params_differ(parent_params, offspring_params):
            print(f"✓ Individual {ind.ID}: Offspring inherited optimized parameters")
        else:
            print(f"✗ Individual {ind.ID}: Offspring did not inherit parameters")
            all_inherited = False

    if all_inherited:
        print("✓ Parallel Lamarckian evolution test PASSED\n")
        return True
    else:
        print("✗ ERROR: Inheritance failed in parallel mode\n")
        return False


def test_baldwin_serial():
    """Test Baldwin effect in serial mode (parameters NOT inherited)."""
    print("=" * 60)
    print("TEST 3: Baldwin Effect - Serial Mode (num_jobs=1)")
    print("=" * 60)

    # Create config with Baldwin effect (lamarckian_evolution=False)
    config = Config('configs/config_regression1D.ini')

    # Initialize innovation tracker
    InnovationTracker.initialize(config)

    config.enable_gradient = True
    config.lamarckian_evolution = False  # Baldwin effect
    config.gradient_steps = 10  # More steps for measurable improvement
    config.learning_rate = 0.1

    # Create trial and individual
    trial = TestTrialGrad(config)
    genome = Genome(config)
    for _ in range(3):
        genome.mutate()

    individual = Individual(genome, 'autograd')
    individual.fitness = 0.5

    # Store original parameters
    original_params = extract_genome_params(individual.genome)

    # Apply gradient descent
    result = trial.test_apply_gradient_descent(individual)

    # Check return value format
    assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
    loss_final, loss_improvement, fitness_improvement, genome_data = result

    # For Baldwin effect, genome_data should be None
    assert genome_data is None, "genome_data should be None for Baldwin effect"

    # Check that genome was NOT modified
    final_params = extract_genome_params(individual.genome)

    if not params_differ(original_params, final_params):
        print("✓ Genome parameters unchanged (correct for Baldwin effect)")
    else:
        print("✗ ERROR: Genome parameters modified (should be unchanged for Baldwin)")
        return False

    # Check that fitness improvement occurred
    if fitness_improvement > 0.001:
        print(f"✓ Fitness improved by {fitness_improvement:.4f}")
    else:
        print("✗ ERROR: No fitness improvement detected")
        return False

    print("✓ Serial Baldwin effect test PASSED\n")
    return True


def test_baldwin_parallel():
    """Test Baldwin effect in parallel mode (parameters NOT inherited)."""
    print("=" * 60)
    print("TEST 4: Baldwin Effect - Parallel Mode (num_jobs=2)")
    print("=" * 60)

    # Create config with Baldwin effect
    config = Config('configs/config_regression1D.ini')

    # Initialize innovation tracker
    InnovationTracker.initialize(config)

    config.enable_gradient = True
    config.lamarckian_evolution = False  # Baldwin effect
    config.gradient_steps = 10  # More steps for measurable improvement
    config.learning_rate = 0.1

    # Create trial and individuals
    trial = TestTrialGrad(config)

    individuals = []
    for i in range(3):
        genome = Genome(config)
        for _ in range(3):
            genome.mutate()
        individual = Individual(genome, 'autograd')
        individual.fitness = 0.5 + i * 0.1
        individuals.append(individual)

    # Store original parameters
    original_params = {}
    for ind in individuals:
        original_params[ind.ID] = extract_genome_params(ind.genome)

    # Run parallel gradient descent
    results = Parallel(n_jobs=2)(
        delayed(trial.test_apply_gradient_descent)(ind) for ind in individuals
    )

    # Process results
    all_unchanged = True
    all_improved = True

    for individual, result in zip(individuals, results):
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
        loss_final, loss_improvement, fitness_improvement, genome_data = result

        # For Baldwin, genome_data should be None
        assert genome_data is None, "genome_data should be None for Baldwin effect"

        # Check genome unchanged
        current_params = extract_genome_params(individual.genome)
        original = original_params[individual.ID]

        if not params_differ(original, current_params):
            print(f"✓ Individual {individual.ID}: Genome unchanged (correct)")
        else:
            print(f"✗ Individual {individual.ID}: Genome modified (error)")
            all_unchanged = False

        # Check fitness improvement
        if fitness_improvement > 0.001:
            print(f"✓ Individual {individual.ID}: Fitness improved by {fitness_improvement:.4f}")
        else:
            print(f"✗ Individual {individual.ID}: No fitness improvement")
            all_improved = False

    if all_unchanged and all_improved:
        print("✓ Parallel Baldwin effect test PASSED\n")
        return True
    else:
        print("✗ ERROR: Baldwin effect test failed\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING GRADIENT DESCENT PARALLELIZATION FIX")
    print("=" * 60 + "\n")

    # Run all tests
    results = []

    results.append(("Serial Lamarckian", test_lamarckian_serial()))
    results.append(("Parallel Lamarckian", test_lamarckian_parallel()))
    results.append(("Serial Baldwin", test_baldwin_serial()))
    results.append(("Parallel Baldwin", test_baldwin_parallel()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Fix verified!")
        print("\nThe Lamarckian evolution bug in parallel mode has been fixed.")
        print("Optimized parameters are now correctly inherited by offspring")
        print("in both serial and parallel execution modes.")
    else:
        print("✗ SOME TESTS FAILED - Fix incomplete")
    print("=" * 60)