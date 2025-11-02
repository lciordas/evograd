"""
NEAT Genotype Package

This package implements the genotype representation for the NEAT (NeuroEvolution of
Augmenting Topologies) algorithm. It provides classes for encoding neural network
structures and parameters at the genetic level.

The NEAT genotype consists of two types of genes:
- Node genes:       Encode individual neurons with their parameters (bias, gain, activation)
- Connection genes: Encode weighted connections between neurons with innovation numbers

Modules:
    node_gene:          NodeType enumeration and NodeGene class
    connection_gene:    ConnectionGene class
    genome:             Genome class
    innovation_tracker: InnovationTracker class

Exported Classes:
    NodeType:          Enumeration for node types (INPUT, HIDDEN, OUTPUT)
    NodeGene:          Gene encoding a single network node
    ConnectionGene:    Gene encoding a weighted connection between nodes
    Genome:            Complete genome representing a neural network
    InnovationTracker: Global tracker for innovation numbers and node IDs
"""

from evograd.genotype.connection_gene    import ConnectionGene
from evograd.genotype.genome             import Genome
from evograd.genotype.innovation_tracker import InnovationTracker
from evograd.genotype.node_gene          import NodeType, NodeGene

__all__ = ['ConnectionGene',
           'Genome',
           'InnovationTracker',
           'NodeGene',
           'NodeType']
