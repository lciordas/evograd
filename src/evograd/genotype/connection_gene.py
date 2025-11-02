"""
NEAT Connection Gene Module

This module implements the ConnectionGene class for the
NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

Classes:
    ConnectionGene: Gene encoding a weighted connection between nodes
"""

import numpy as np
import random
from evograd.run.config import Config

class ConnectionGene:
    """
    A gene describing a weighted connection between two nodes in a Neural Network.

    Each connection gene represents a directed edge in the neural network graph,
    connecting a source node to a destination node with an associated weight.
    Connection genes are uniquely identified by their innovation number, which
    serves as a historical marker enabling proper gene alignment during crossover.

    Connections can be enabled or disabled, allowing NEAT to preserve structural
    information while temporarily deactivating pathways. Disabled connections may
    be re-enabled through mutation.

    Public Attributes:
        node_in:    ID of the source node
        node_out:   ID of the destination node
        weight:     Weight of the connection
        enabled:    Whether this connection is active in the network
        innovation: Global innovation number uniquely identifying this connection

    Public Methods:
        mutate(): Stochastically mutate the connection weight
    """

    def __init__(self,
                 node_in   : int,
                 node_out  : int,
                 weight    : float,
                 innovation: int,
                 config    : Config,
                 enabled   : bool = True):
        """
        Initialize a connection gene.

        Parameters:
            node_in:    ID of the source node
            node_out:   ID of the destination node
            weight:     Weight of the connection
            innovation: Number uniquely and globally identifying this connection
            config:     Stores configuration parameters
            enabled:    Whether this connection is active in the network
        """
        self.node_in   : int    = node_in
        self.node_out  : int    = node_out
        self.weight    : float  = weight
        self.enabled   : bool   = enabled
        self.innovation: int    = innovation
        self._config   : Config = config

    def mutate(self) -> None:
        """
        Stochastically mutate the (gene describing the) connection.

        Both whether a mutation occurs and its nature & magnitude are stochastic.
        For a connection gene, mutating means changing the 'weight' parameter.
        Mutating a parameter can be accomplished in two ways:
         + modifying the current value additively by a small amount
         + replacing the current value by a new one
        """
        perturb_prob = self._config.weight_perturb_prob   # prob of perturbing the 'weight'
        replace_prob = self._config.weight_replace_prob   # prob of replacing  the 'weight'

        r = random.random()
        if r < perturb_prob:
            chg_weight  = random.gauss(0, self._config.weight_perturb_strength)
            new_weight  = self.weight + chg_weight
            new_weight  = np.maximum(self._config.min_weight, np.minimum(self._config.max_weight, new_weight))  # Clip it
            self.weight = new_weight

        elif r < perturb_prob + replace_prob:
            self.weight = random.uniform(self._config.min_weight, self._config.max_weight)

    def __repr__(self):
        return (f"ConnectionGene(node_in={self.node_in:03d}, node_out={self.node_out:03d},"
                f"weight={self.weight:+.6f}, enabled={self.enabled}, innovation={self.innovation:03d})")

    def __str__(self):
        s  = f"[{self.innovation:03d},{'E' if self.enabled else 'D'},"
        s += f"{self.node_in:02d}=>{self.node_out:02d},{self.weight:+.02f}]"
        return s
