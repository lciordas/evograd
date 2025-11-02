"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    from evograd.run.config import Config
    # Return a minimal config for testing
    config_dict = {
        'POPULATION_INIT': {
            'population_size': 100,
            'num_inputs': 2,
            'num_outputs': 1,
        },
        'SPECIATION': {
            'compatibility_threshold': 3.0,
        }
    }
    return config_dict


@pytest.fixture
def sample_genome():
    """Create a sample genome for testing."""
    from evograd.genotype.genome import Genome
    from evograd.genotype.innovation_tracker import InnovationTracker

    tracker = InnovationTracker()
    genome = Genome(tracker=tracker)
    return genome