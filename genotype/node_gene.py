"""
NEAT Node Gene Module.

This module implements the NodeGene class and NodeType enumeration 
for the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

Classes:
    NodeType: Enumeration for node types (INPUT, HIDDEN, OUTPUT)
    NodeGene: Gene encoding a single network node with parameters
"""

import numpy as np
import random
from enum   import Enum
from typing import Callable

from activations import activations
from run.config import Config

class NodeType(Enum):
    """
    Nodes come in three types: input, hidden, output.
    """
    INPUT  = "I"
    HIDDEN = "H"
    OUTPUT = "O"

class NodeGene:
    """
    A gene describing a node in a Neural Network.

    Each node gene encodes the properties of a single node in the neural network,
    including its type (input, hidden, or output), bias, gain, and activation function.
    Node genes are identified by a unique node ID which remains consistent across
    structural mutations and crossover operations.

    The node computes its output as: activation(gain * weighted_input + bias)

    Public Attributes:
        id:   Unique identifier for this node
        type: Type of node (INPUT, HIDDEN, or OUTPUT)
        bias: Bias value added to the node's weighted input
        gain: Multiplier applied to the node's weighted input

    Public Methods:
        mutate(): Stochastically mutate the bias and gain parameters
    """

    def __init__(self,
                 node_id  : int,
                 node_type: NodeType,
                 config   : Config,
                 bias     : float | None = None,
                 gain     : float | None = None):
        """
        Initialize a node gene.
        If the 'bias' and 'gain' parameters are not specified, they will be
        initialized with random values, according to the configuration file.

        Parameters:
            node_id:   Unique identifier for this node
            node_type: Type of node (INPUT, HIDDEN, or OUTPUT)
            config:    Stores configuration parameters
            bias:      Bias value added to the node's weighted input
            gain:      Multiplier applied to the node's weighted input
        """
        if bias is None:
            bias = np.random.normal(config.bias_init_mean, config.bias_init_stdev)
            bias = np.minimum(np.maximum(bias, config.min_bias), config.max_bias)

        if gain is None:
            gain = np.random.normal(config.gain_init_mean, config.gain_init_stdev)
            gain = np.minimum(np.maximum(gain, config.min_gain), config.max_gain)

        self.id              : int                      = node_id
        self.type            : NodeType                 = node_type
        self.bias            : float                    = bias
        self.gain            : float                    = gain
        self._activation     : Callable[[float], float] = activations[config.activation]
        self._activation_name: str                      = config.activation
        self._config         : Config                   = config

    def mutate(self) -> None:
        """
        Stochastically mutate the (gene describing the) node.

        Both whether a mutation occurs and its nature & magnitude are stochastic.
        For a node gene, mutating means changing the 'bias' and 'gain' parameters.
        Mutating a parameter can be accomplished in two ways:
         + modifying the current value additively by a small amount
         + replacing the current value by a new one
        """

        # Attempt to mutate the 'bias'
        perturb_prob = self._config.bias_perturb_prob   # prob of perturbing the 'bias'
        replace_prob = self._config.bias_replace_prob   # prob of replacing  the 'bias'

        r_bias = random.random()
        if r_bias < perturb_prob:
            chg_bias  = random.gauss(0, self._config.bias_perturb_strength)
            new_bias  = self.bias + chg_bias
            new_bias  = np.maximum(self._config.min_bias, np.minimum(self._config.max_bias, new_bias))  # Clip it
            self.bias = new_bias

        elif r_bias < perturb_prob + replace_prob:
            self.bias = random.uniform(self._config.min_bias, self._config.max_bias)

        # Attempt to mutate the 'gain'
        perturb_prob = self._config.gain_perturb_prob   # prob of perturbing the 'gain'
        replace_prob = self._config.gain_replace_prob   # prob of replacing  the 'gain'

        r_perturb = random.random()
        if r_perturb < perturb_prob:
            chg_gain  = random.gauss(0, self._config.gain_perturb_strength)
            new_gain  = self.gain + chg_gain
            new_gain  = np.maximum(self._config.min_gain, np.minimum(self._config.max_gain, new_gain))  # Clip it
            self.gain = new_gain

        elif r_perturb < perturb_prob + replace_prob:
            self.gain = random.uniform(self._config.min_gain, self._config.max_gain)

    def __repr__(self):
        return (f"NodeGene(node_id={self.id:+03d}, node_type=NodeType.{self.type.name:6s},"
                f"bias={self.bias}, gain={self.gain}, activation={repr(self._activation)})")

    def __str__(self):
        if self.type == NodeType.INPUT:
            return f"[{self.type.value}{self.id}]"
        else:
            return f"[{self.type.value}{self.id},b={self.bias:.2f},g={self.gain:.2f}]"