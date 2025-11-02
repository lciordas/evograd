"""
NEAT Phenotype Package

This package implements the phenotype representation for the NEAT (NeuroEvolution
of Augmenting Topologies) algorithm. It provides classes for expressing genomes as
executable neural networks and evolved individuals.

The phenotype layer transforms the genetic representation (genotype) into functioning
neural networks that can process inputs and produce outputs. This separation between
genotype and phenotype allows the genetic algorithm to evolve network structure and
parameters while maintaining an executable form for fitness evaluation.

Modules:
    network_base:     Abstract base class for network implementations
    network_standard: Standard object-oriented network implementation
    network_autograd: Autograd-compatible vectorized network implementation
    network_fast:     High-performance vectorized network implementation
    individual:       Evolved agent combining genome, network, and fitness

Exported Classes:
    Connection:      A weighted connection between two neurons
    Individual:      A complete evolved agent with genome, network, and fitness
    NetworkBase:     Abstract base class for network implementations
    NetworkStandard: Object-oriented feedforward neural network (single-sample processing)
    NetworkAutograd: Autograd-compatible feedforward neural network (batch processing)
    NetworkFast:     High-performance feedforward neural network (batch processing)
    Neuron:          A computational node applying activation functions
"""

from evograd.phenotype.individual       import Individual
from evograd.phenotype.network_base     import NetworkBase
from evograd.phenotype.network_standard import Connection, Neuron, NetworkStandard
from evograd.phenotype.network_autograd import NetworkAutograd
from evograd.phenotype.network_fast     import NetworkFast

__all__ = ['Connection',
           'Neuron',
           'Individual',
           'NetworkBase',
           'NetworkStandard',
           'NetworkAutograd',
           'NetworkFast']
