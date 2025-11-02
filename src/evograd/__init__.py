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
    >>> from evograd import Config, Trial
    >>> config = Config("config.ini")
    >>> class MyTrial(Trial):
    ...     def _evaluate_fitness(self, individual):
    ...         # Implement fitness evaluation
    ...         pass
    >>> trial = MyTrial(config)
    >>> trial.run()
"""

__version__ = "0.1.0"
__author__ = "Lucas Ciordas"
__email__ = "22lciordas@gmail.com"

# Import main classes for convenient access
from evograd.run.config import Config
from evograd.run.trial import Trial
from evograd.run.trial_grad import TrialGrad
from evograd.run.experiment import Experiment
from evograd.genotype.genome import Genome
from evograd.genotype.node_gene import NodeGene
from evograd.genotype.connection_gene import ConnectionGene
from evograd.phenotype.individual import Individual
from evograd.pool.population import Population

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