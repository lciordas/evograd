"""
NEAT Autograd-Compatible Network Module

This module implements an autograd-compatible neural network for NEAT that supports
gradient-based learning. Unlike NetworkFast, with which it shares a common architecture,
this implementation follows a functional programming style, avoiding in-place array updates,
to preserve autograd's computation graph, enabling gradient-based optimization of network
parameters.

Key features:
- Functional programming style (no in-place updates) preserves autograd computation graph
- Sparse topology optimization for efficient forward passes
- Support for both evolutionary (NEAT) and gradient-based (SGD) training
- Explicit parameter passing to forward_pass() enables gradient computation via autograd

This functional approach comes with a performance penalty. If you don't need gradients,
use NetworkFast instead.

Classes:
    NetworkAutograd: Autograd-compatible feedforward neural network using numpy arrays
"""

import autograd.numpy as np  # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neat.genotype import Genome
from neat.activations                     import activations
from neat.activations.legendre_activation import LegendreActivation
from neat.phenotype.network_base          import NetworkBase

class NetworkAutograd(NetworkBase):
    """
    Autograd-compatible batch-processing neural network for NEAT.

    Supports both standard activation functions and learnable activation functions
    (e.g., LegendreActivation with learnable polynomial coefficients).

    Public Methods:
        forward_pass(inputs, weights=None, biases=None, gains=None, activation_coeffs=None):
            Process batch through network with optional explicit parameters
            Input:  (batch_size, num_inputs) or (num_inputs,) auto-reshaped to (1, num_inputs)
            Output: (batch_size, num_outputs)

        get_parameters():
            Get current network parameters as (weights, biases, gains, activation_coeffs) tuple.
            activation_coeffs is a dict mapping node_idx to coefficient arrays (only for nodes
            with learnable activations).

        set_parameters(weights, biases, gains, coeffs=None):
            Update network parameters (always clipped to config bounds)

        save_parameters_to_genome():
            Save current network parameters back into the genome (Lamarckian evolution),
            including activation coefficients for learnable activation nodes.

    Public Properties (inherited from NetworkBase):
        number_nodes:               Total number of nodes in the network
        number_nodes_hidden:        Number of hidden nodes in the network
        number_connections:         Total number of connections in the network
        number_connections_enabled: Number of enabled connections in the network
    """

    def __init__(self, genome: 'Genome'):
        """
        Initialize vectorized network from genome.

        Builds weight matrix, bias vectors, and gain vectors from the genome's
        node and connection genes. Uses dense numpy arrays for maximum performance
        with matrix operations.

        Parameters:
            genome: The Genome encoding the network structure
        """
        # Initialize base class (sets _input_ids, _output_ids, _sorted_nodes)
        super().__init__(genome)

        # Get number of nodes
        num_nodes = len(genome.node_genes)
        self._num_nodes = num_nodes

        # Create "node ID => array index" mapping
        # (genome node IDs may not be contiguous, but array indices must be)
        sorted_node_ids = sorted(genome.node_genes.keys())
        self._node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        self._idx_to_node_id = {idx: node_id for node_id, idx in self._node_id_to_idx.items()}

        # Build weight matrix (adjacency matrix representation)
        # weights[i, j] = weight of connection from node i to node j
        # 0.0 means no connection or disabled connection
        weight_data = []
        for conn_gene in genome.conn_genes.values():
            if conn_gene.enabled:
                from_idx = self._node_id_to_idx[conn_gene.node_in]
                to_idx   = self._node_id_to_idx[conn_gene.node_out]
                weight_data.append((from_idx, to_idx, conn_gene.weight))

        weights = np.zeros((num_nodes, num_nodes))
        for from_idx, to_idx, weight_val in weight_data:
            weights[from_idx, to_idx] = weight_val
        self.weights = weights

        # Extract node parameters (bias, gain, activation) from genome
        bias_list = []
        gain_list = []
        activation_names = []

        for idx in range(num_nodes):
            node_id   = self._idx_to_node_id[idx]
            node_gene = genome.node_genes[node_id]
            bias_list.append(node_gene.bias)
            gain_list.append(node_gene.gain)
            activation_names.append(node_gene.activation_name)

        # Convert to numpy arrays
        self.biases = np.array(bias_list)
        self.gains  = np.array(gain_list)

        # Store activation functions by index for runtime use
        # For standard activations: store the function reference
        # For Legendre activations: instantiate LegendreActivation objects
        self.activations = []
        self.activation_coeffs = {}  # node_idx => coefficients (only for learnable activations)

        for idx in range(num_nodes):
            node_id         = self._idx_to_node_id[idx]
            node_gene       = genome.node_genes[node_id]
            activation_name = activation_names[idx]

            if activation_name == 'legendre':
                degree = len(node_gene.activation_coeffs) - 1 
                self.activations.append(LegendreActivation(degree))
                self.activation_coeffs[idx] = node_gene.activation_coeffs.copy()
            else:
                self.activations.append(activations.get(activation_name, None))

        # Convert input/output IDs to indices
        self._input_indices  = [self._node_id_to_idx[node_id] for node_id in self._input_ids]
        self._output_indices = [self._node_id_to_idx[node_id] for node_id in self._output_ids]

        # Convert sorted nodes from IDs to indices
        self._sorted_indices = [self._node_id_to_idx[node_id] for node_id in self._sorted_nodes]

        # Pre-compute indices for non-input nodes (performance optimization)
        # Avoids membership check in hot loop during forward pass
        input_indices_set = set(self._input_indices)
        self._hidden_output_indices = [idx for idx in self._sorted_indices if idx not in input_indices_set]

        # Pre-compute network topology (sparse computation optimization)
        # node index => list of indices for all nodes that can send data to it
        self._incoming_connections = [[] for _ in range(num_nodes)]
        for from_idx in range(num_nodes):
            for to_idx in range(num_nodes):
                if self.weights[from_idx, to_idx] != 0.0:  # Only store actual connections
                    self._incoming_connections[to_idx].append(from_idx)

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Get current network parameters as copies.

        Returns copies of all parameters to prevent external modifications from
        affecting the network's internal state. This ensures full encapsulation
        and prevents accidental bugs.

        Returns:
            tuple: (weights, biases, gains, activation_coeffs)
                - weights: (num_nodes, num_nodes) weight matrix (copy)
                - biases:  (num_nodes,) bias vector (copy)
                - gains:   (num_nodes,) gain vector (copy)
                - coeffs:   dict mapping node_idx to coefficient arrays (copies)
        """
        coeffs_copy = {idx: cs.copy() for idx, cs in self.activation_coeffs.items()}
        return self.weights.copy(), self.biases.copy(), self.gains.copy(), coeffs_copy

    def set_parameters(self,
                       weights: np.ndarray,
                       biases:  np.ndarray,
                       gains:   np.ndarray,
                       coeffs:  dict[int, np.ndarray] | None = None) -> None:
        """
        Set network parameters.

        Parameters are clipped to the bounds specified in the config to ensure
        they remain within valid ranges.

        This overwrites all existing parameters (possibly loaded from genome).
        This does not overwrite the parameter values stored in the genome, however 
        see 'save_parameters_to_genome()'.

        NOTE: This assumes network topology (connection structure) does not change.
        Only parameter VALUES are updated. Topology is fixed after initialization.

        Parameters:
            weights: Weight matrix (num_nodes, num_nodes)
            biases:  Bias vector (num_nodes,)
            gains:   Gain vector (num_nodes,)
            coeffs:  dict mapping node_idx => coefficient arrays for learnable activations
        """
        config = self._genome._config

        # Note: 'np.clip' creates a copy of the array it processes
        self.weights = np.clip(weights, config.min_weight, config.max_weight)
        self.biases  = np.clip(biases,  config.min_bias,   config.max_bias)
        self.gains   = np.clip(gains,   config.min_gain,   config.max_gain)

        # Learnable activation coefficients are not subject to clipping.
        # We copy them to be consistent with how other parameters are set.
        if coeffs is not None:
            self.activation_coeffs = {idx: c.copy() for idx, c in coeffs.items()}

    def forward_pass(self,
                     inputs:  np.ndarray | list,
                     weights: np.ndarray | None = None,
                     biases:  np.ndarray | None = None,
                     gains:   np.ndarray | None = None,
                     coeffs:  dict[int, np.ndarray] | None = None) -> np.ndarray:
        """
        Perform forward pass through the network using batched matrix operations.

        Supports explicit parameter passing for gradient computation with autograd.
        When parameters are None, uses current network parameters (self.weights, etc.).

        Why explicit parameter passing is needed:
        Autograd computes gradients by tracing the computation graph from inputs to outputs.
        If the network reads parameters from self.weights/biases/gains (instance attributes),
        autograd cannot trace the dependency between parameters and outputs. By passing
        parameters explicitly as function arguments, autograd can track how changes to the
        parameters affect the output, enabling gradient-based optimization (e.g., SGD).

        Example usage for gradient computation:

            # Get current parameters
            weights, biases, gains, activation_coeffs = network.get_parameters()

            # Define loss function that uses explicit parameters
            def loss_fn(w, b, g, a_coeffs):
                outputs = network.forward_pass(inputs, weights=w, biases=b, gains=g,
                                               activation_coeffs=a_coeffs)
                return np.mean((outputs - targets)**2)

            # Compute gradients with respect to parameters
            grad_w = autograd.grad(loss_fn, argnum=0)(weights, biases, gains, activation_coeffs)
            grad_b = autograd.grad(loss_fn, argnum=1)(weights, biases, gains, activation_coeffs)
            grad_g = autograd.grad(loss_fn, argnum=2)(weights, biases, gains, activation_coeffs)
            # For activation coefficients, need to compute gradients for each node separately
            # since activation_coeffs is a dict

        Parameters:
            inputs:  Batched input values as numpy array or list
                     Shape: (batch_size, num_inputs)
                     For single sample, use shape (1, num_inputs)
            weights: Optional weight matrix (num_nodes, num_nodes). Uses self.weights if None.
            biases:  Optional bias vector (num_nodes,). Uses self.biases if None.
            gains:   Optional gain vector (num_nodes,). Uses self.gains if None.
            activation_coeffs: Optional dict mapping node_idx to coefficient arrays for learnable
                               activations. Uses self._activation_coeffs if None.

        Returns:
            Batched output values as numpy array
            Shape: (batch_size, num_outputs)
        """
        weights = weights if weights is not None else self.weights
        biases  = biases  if biases  is not None else self.biases
        gains   = gains   if gains   is not None else self.gains
        coeffs  = coeffs  if coeffs  is not None else self.activation_coeffs

        # Prepare and validate inputs.
        # We need a 2D array (a batch of 1D input vectors), where the size
        # of each input vector must match the number of input neurons.
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)   # convert to batch of size 1
        elif inputs.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {inputs.ndim}D")

        batch_size = inputs.shape[0]            # number of inputs in a batch
        input_size = inputs.shape[1]            # size of an input vector

        num_inputs = len(self._input_indices)   # number of input neurons
        if input_size != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {input_size}")

        # Store node values using a Python list of 1D arrays.
        # Each list element stores a (batch_size,) array for that node's value across the batch.
        #
        # This is done to avoid in-place updates that would break the computation graph needed 
        # by autograd, Why use a list instead of a single 2D array?
        #
        # - WRONG approach: node_values = np.zeros((num_nodes, batch_size))
        #                   node_values[i] = new_value  # In-place update breaks autograd!
        #
        # - CORRECT approach: node_values_list = [array1, array2, ...]
        #                     node_values_list[i] = new_value  # Replaces reference, preserves graph
        #
        # In-place array updates (array[i] = value) modify existing array objects, which breaks
        # autograd's computation graph. By storing arrays in a Python list and replacing list
        # elements (list[i] = new_array), we create new array objects while preserving the
        # computation graph that autograd uses to compute gradients.
        node_values_list = [np.zeros(batch_size) for _ in range(self._num_nodes)]

        # Input nodes: set their value directly from input array
        for i, input_idx in enumerate(self._input_indices):
            node_values_list[input_idx] = inputs[:, i]

        # Hidden & output nodes: propagate through them in topological order
        for node_idx in self._hidden_output_indices:

            # Get nodes whose output is sent to the current node
            source_indices = self._incoming_connections[node_idx]

            # Compute weighted sum of inputs using functional style for autograd compatibility
            # Use 'weighted_sum = weighted_sum + ...' (creates new array) instead of
            # 'weighted_sum += ...' (in-place modification that breaks autograd tracking)
            weighted_sum = np.zeros(batch_size)
            for source_idx in source_indices:
                weight = weights[source_idx, node_idx]
                weighted_sum = weighted_sum + node_values_list[source_idx] * weight

            # Apply gain, bias, and activation function
            activation = self.activations[node_idx]
            z = gains[node_idx] * weighted_sum + biases[node_idx]
            if isinstance(activation, LegendreActivation):
                activation_output = activation(z, coeffs[node_idx])
            else:
                activation_output = activation(z)

            # Store result by replacing list element (preserves autograd computation graph)
            node_values_list[node_idx] = activation_output

        # Extract output values from output nodes: stack into (batch_size, num_outputs) array
        outputs = np.column_stack([node_values_list[idx] for idx in self._output_indices])
        return outputs

    def save_parameters_to_genome(self):
        """
        Save current network parameters back into the genome (Lamarckian evolution).

        This method updates the genome's node and connection genes with the current
        network parameters, allowing gradient-descent-optimized parameters to be
        inherited by offspring through crossover and passed to future generations.

        This implements Lamarckian evolution: acquired characteristics (parameters
        learned through gradient descent) can be inherited, unlike purely Darwinian
        evolution where only random mutations and selection occur.

        Note:
            Only updates parameter values (weights, biases, gains, activation coefficients).
            Network topology (connections, nodes) is never modified by this method.
        """
        # Update node genes (bias, gain, and activation coefficients)
        for idx in range(self._num_nodes):
            node_id   = self._idx_to_node_id[idx]
            node_gene = self._genome.node_genes[node_id]

            # Convert numpy types to Python native float for cleaner storage
            # self.biases[idx] and self.gains[idx] are numpy.float64 (or numpy.float32)
            # Converting to Python float ensures consistent serialization (e.g., pickle)
            # and avoids numpy-specific type dependencies in the genome
            node_gene.bias = float(self.biases[idx])
            node_gene.gain = float(self.gains[idx])

            # Update activation coefficients if this node has learnable activation
            if idx in self.activation_coeffs:
                node_gene.activation_coeffs = self.activation_coeffs[idx].copy()

        # Update connection genes (weights)
        for conn_gene in self._genome.conn_genes.values():
            if conn_gene.enabled:
                from_idx = self._node_id_to_idx[conn_gene.node_in]
                to_idx   = self._node_id_to_idx[conn_gene.node_out]
                conn_gene.weight = float(self.weights[from_idx, to_idx])

    def __str__(self):
        """String representation showing network structure."""
        node_info = []
        for node_id in self._sorted_nodes:
            gene = self._genome.node_genes[node_id]
            node_info.append(f"  Node {node_id} ({gene.type.name}): bias={gene.bias:.2f}, gain={gene.gain:.2f}")

        conn_info = []
        for conn in self._genome.conn_genes.values():
            if conn.enabled:
                conn_info.append(f"  [{conn.innovation:03d}] {conn.node_in:02d}=>{conn.node_out:02d}, w={conn.weight:+.2f}")

        return "\n".join(node_info) + "\n\n" + "\n".join(conn_info)

    def __repr__(self):
        """Short representation for debugging."""
        return (f"NetworkAutograd(nodes={self._num_nodes}, "
                f"hidden={self.number_nodes_hidden}, "
                f"connections={self.number_connections_enabled}/{self.number_connections})")
