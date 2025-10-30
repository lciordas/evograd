"""
Shared fixtures for integration tests.
"""

import pytest
import numpy as np
from neat.run.config import Config
from neat.genotype import Genome


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility and reset global state."""
    import random
    from itertools import count
    from neat.phenotype.individual import Individual
    from neat.genotype.innovation_tracker import InnovationTracker

    # Set random seeds FIRST (before creating any objects that use random)
    # Using seed 42 for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Reset Individual ID generator for reproducibility
    # This ensures IDs start from 0 in each test
    Individual._id_generator = count(0)

    # Reset InnovationTracker global state
    # This is normally done in Trial._reset(), but we do it here too
    # to ensure clean state even if previous tests didn't clean up properly
    InnovationTracker._next_innovation_number = None
    InnovationTracker._next_node_id = None
    InnovationTracker._innovation_numbers = {}
    InnovationTracker._split_IDs = {}

    yield

    # Reset seeds after test (optional, but good practice)
    np.random.seed(None)
    random.seed(None)


@pytest.fixture
def xor_inputs_standard():
    """XOR inputs for standard network (list format)."""
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


@pytest.fixture
def xor_outputs_standard():
    """XOR expected outputs for standard network (list format)."""
    return [[0.0], [1.0], [1.0], [0.0]]


@pytest.fixture
def xor_inputs_batch():
    """XOR inputs for batch networks (numpy array format)."""
    return np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])


@pytest.fixture
def xor_outputs_batch():
    """XOR expected outputs for batch networks (numpy array format)."""
    return np.array([[0.0], [1.0], [1.0], [0.0]])


@pytest.fixture
def simple_test_genome():
    """Create a simple genome for testing network equivalence."""
    return {
        'activation': 'sigmoid',
        'nodes': [
            {'id': 0, 'type': 'input'},
            {'id': 1, 'type': 'input'},
            {'id': 2, 'type': 'output', 'bias': 0.5, 'gain': 1.0},
            {'id': 3, 'type': 'hidden', 'bias': -0.3, 'gain': 1.2}
        ],
        'connections': [
            {'from': 0, 'to': 3, 'weight': 1.5, 'enabled': True},
            {'from': 1, 'to': 3, 'weight': -1.2, 'enabled': True},
            {'from': 3, 'to': 2, 'weight': 2.0, 'enabled': True},
            {'from': 0, 'to': 2, 'weight': 0.5, 'enabled': True},
        ]
    }


def evaluate_xor_fitness_standard(network, xor_inputs, xor_outputs):
    """
    Evaluate XOR fitness for standard network (serial processing).
    
    Parameters:
        network: NetworkStandard instance
        xor_inputs: List of input pairs
        xor_outputs: List of expected outputs
        
    Returns:
        Fitness score (max 4.0 for perfect solution)
    """
    fitness = 4.0
    for inputs, expected_output in zip(xor_inputs, xor_outputs):
        output = network.forward_pass(inputs)
        error = output[0] - expected_output[0]
        fitness -= error ** 2
    return fitness


def evaluate_xor_fitness_batch(network, xor_inputs, xor_outputs):
    """
    Evaluate XOR fitness for batch networks (parallel processing).
    
    Parameters:
        network: NetworkFast or NetworkAutograd instance
        xor_inputs: NumPy array of input pairs (4, 2)
        xor_outputs: NumPy array of expected outputs (4, 1)
        
    Returns:
        Fitness score (max 4.0 for perfect solution)
    """
    outputs = network.forward_pass(xor_inputs)
    errors = outputs - xor_outputs
    fitness = 4.0 - np.sum(errors ** 2)
    return float(fitness)
