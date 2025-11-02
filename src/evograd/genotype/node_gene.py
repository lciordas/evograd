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

from evograd.activations import activations, activation_codes, LegendreActivation
from evograd.run.config  import Config

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
        id:                Unique identifier for this node
        type:              Type of node (INPUT, HIDDEN, or OUTPUT)
        bias:              Bias value added to the node's weighted input
        gain:              Multiplier applied to the node's weighted input
        activation_name:   Name of the activation function (e.g., 'tanh', 'relu')
        activation:        The activation function itself (callable)
        activation_coeffs: Coefficients for learnable activation functions (only for 'legendre')

    Public Properties:
        activation_function: Returns a callable activation function (for 'legendre',
                             creates a callable with fixed coefficients; otherwise
                             returns the stored activation function)

    Public Methods:
        mutate(): Stochastically mutate the bias and gain parameters
    """

    def __init__(self,
                 node_id          : int,
                 node_type        : NodeType,
                 config           : Config,
                 bias             : float      | None = None,
                 gain             : float      | None = None,
                 activation_name  : str        | None = None,
                 activation_coeffs: np.ndarray | None = None):
        """
        Initialize a node gene.
        If the 'bias' and 'gain' parameters are not specified, they will be
        initialized with random values, according to the configuration file.
        If 'activation_name' is not specified, it will use the default from
        the configuration file.
        If 'activation_coeffs' is not specified and the activation function
        is 'legendre', they will be initialized with random values according
        to the configuration file.

        Parameters:
            node_id:           Unique identifier for this node
            node_type:         Type of node (INPUT, HIDDEN, or OUTPUT)
            config:            Stores configuration parameters
            bias:              Bias value added to the node's weighted input
            gain:              Multiplier applied to the node's weighted input
            activation_name:   Name of activation function (e.g., 'tanh', 'relu', 'legendre')
                               If None, uses value from 'config.activation_initial'
                               Special values: "random" (randomly select any activation),
                                               "random-fixed" (randomly select non-learnable activation)
            activation_coeffs: Coefficients for learnable activation functions
                               Only used when activation_name is 'legendre'
        """
        self._config: Config   = config
        self.id     : int      = node_id
        self.type   : NodeType = node_type

        if bias is None:
            bias = np.random.normal(config.bias_init_mean, config.bias_init_stdev)
            bias = np.minimum(np.maximum(bias, config.min_bias), config.max_bias)
        self.bias: float = bias

        if gain is None:
            gain = np.random.normal(config.gain_init_mean, config.gain_init_stdev)
            gain = np.minimum(np.maximum(gain, config.min_gain), config.max_gain)
        self.gain: float = gain

        if node_type == NodeType.INPUT:
            self.activation_name   = None
            self.activation_coeffs = None
            self.activation        = None
        else:
            if activation_name is None:
                activation_name = config.activation_initial

            # Handle "random" and "random-fixed" activation selection
            if activation_name == "random":
                all_activations = list(activations.keys()) + ["legendre"]
                activation_name = random.choice(all_activations)
            elif activation_name == "random-fixed":
                activation_name = random.choice(list(activations.keys()))

            self.activation_name: str = activation_name

            if activation_name == "legendre" and activation_coeffs is None:
                activation_coeffs = np.random.normal(config.legendre_coeffs_init_mean,
                                                     config.legendre_coeffs_init_stdev,
                                                     config.num_legendre_coeffs)
            self.activation_coeffs: np.ndarray | None = activation_coeffs

            # For legendre activation, we don't store a function reference here
            # because it's a parameterized activation that will be instantiated elsewhere
            self.activation: Callable[[float], float] | None = \
                None if activation_name == 'legendre' else activations[activation_name]

    @property
    def activation_function(self) -> Callable[[float], float] | None:
        """
        Get the activation function for this node.

        Returns:
           The activation function for the node (None for INPUT nodes).
        """
        if self.activation_name == "legendre":
            return LegendreActivation.from_coeffs(self.activation_coeffs)
        else:
            return self.activation

    def mutate(self) -> None:
        """
        Stochastically mutate the (gene describing the) node.

        Both whether a mutation occurs and its nature & magnitude are stochastic.
        For a node gene, mutating means changing the 'bias', 'gain', and 'activation'
        parameters. Mutating bias/gain can be accomplished in two ways:
         + modifying the current value additively by a small amount
         + replacing the current value by a new one

        NOTE: mutating activation function selection is now supported. However,
              mutating Legendre polynomial coefficients directly is not implemented -
              those can only change via gradient descent.
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

        # Attempt to mutate the activation function (only for non-input nodes)
        if self.type == NodeType.INPUT:
            return

        mutate_prob  = self._config.activation_mutate_prob
        r_activation = random.random()
        if r_activation < mutate_prob:

            # Get available activation options from config
            # Remove current activation to ensure we select a NEW activation
            available_activations = self._config.activation_options.copy()
            if self.activation_name in available_activations:
                available_activations.remove(self.activation_name)

            # Select random activation function
            if available_activations:
                new_activation = random.choice(available_activations)
                self.activation_name = new_activation

                if new_activation == 'legendre':
                    self.activation_coeffs = np.random.normal(self._config.legendre_coeffs_init_mean,
                                                              self._config.legendre_coeffs_init_stdev,
                                                              self._config.num_legendre_coeffs)
                    self.activation = None   # will be instantiated elsewhere
                else:
                    self.activation_coeffs = None
                    self.activation = activations[new_activation]

    def __repr__(self):
        return (f"NodeGene(node_id={self.id:+03d}, node_type=NodeType.{self.type.name:6s},"
                f"bias={self.bias}, gain={self.gain}, activation={repr(self.activation)})")

    def __str__(self):
        if self.type == NodeType.INPUT:
            return f"[{self.type.value}{self.id}]"
        else:
            # Get the 3-letter activation code
            act_code = activation_codes.get(self.activation_name, "???")
            return f"[{self.type.value}{self.id},{act_code},b={self.bias:.2f},g={self.gain:.2f}]"