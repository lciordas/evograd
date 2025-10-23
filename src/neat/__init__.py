"""
NEAT (NeuroEvolution of Augmenting Topologies) - A Python implementation.

This package provides a complete implementation of the NEAT algorithm for evolving
artificial neural networks through genetic algorithms. It supports various network
types, including standard, fast (vectorized), and autograd-compatible implementations.

Main components:
- genotype: Genetic encoding (genomes, genes, innovation tracking)
- phenotype: Neural network expression (various network implementations)
- pool: Population and speciation management
- run: Trial execution, configuration, and experiment framework
- activations: Activation functions for neural networks

Example:
    >>> from neat import Config, Trial
    >>> config = Config("config.ini")
    >>> class MyTrial(Trial):
    ...     def _evaluate_fitness(self, individual):
    ...         # Implement fitness evaluation
    ...         pass
    >>> trial = MyTrial(config)
    >>> trial.run()
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for convenient access
from neat.run.config import Config
from neat.run.trial import Trial
from neat.run.trial_grad import TrialGrad
from neat.run.experiment import Experiment
from neat.genotype.genome import Genome
from neat.genotype.node_gene import NodeGene
from neat.genotype.connection_gene import ConnectionGene
from neat.phenotype.individual import Individual
from neat.pool.population import Population

__all__ = [
    "Config",
    "Trial",
    "TrialGrad",
    "Experiment",
    "Genome",
    "NodeGene",
    "ConnectionGene",
    "Individual",
    "Population",
]