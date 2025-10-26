"""
NEAT Fast Network Module

This module implements a high-performance neural network for NEAT optimized for
batch inference. This implementation (NetworkFast) shares a common architecture
with NetworkAutograd, however it does not support gradient computation. Due to
this difference, NetworkFast can perform further optimizations: for example use
in-place array operations, which are not allowed in the functional programming
style required when using 'autograd'.

Classes:
    NetworkFast: High-performance feedforward neural network using NumPy arrays
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neat.genotype import Genome
from neat.phenotype.network_base import NetworkBase

class NetworkFast(NetworkBase):
    """
    High-performance batch-processing implementation of a NEAT neural network.

    Optimized for batch inference using in-place array operations and vectorized
    computation. Automatically handles both batched and non-batched inputs.

    Public Methods:
        forward_pass(inputs): Process batch through network
                             Input:  (batch_size, num_inputs) or (num_inputs,) auto-reshaped to (1, num_inputs)
                             Output: (batch_size, num_outputs)

    Public Properties (inherited from NetworkBase):
        number_nodes:               Total number of nodes in the network
        number_nodes_hidden:        Number of hidden nodes in the network
        number_connections:         Total number of connections in the network
        number_connections_enabled: Number of enabled connections in the network
    """

    def __init__(self, genome: 'Genome'):
        """
        Initialize optimized network from genome.

        Builds weight matrix, bias vectors, gain vectors, and pre-allocates
        reusable arrays for maximum performance during forward pass.

        Parameters:
            genome: The Genome encoding the network structure
        """
        # Initialize base class (sets _input_ids, _output_ids, _sorted_nodes)
        super().__init__(genome)

        # Get number of nodes
        num_nodes = len(genome.node_genes)
        self._num_nodes = num_nodes

        # Create "node ID => array index" mapping
        sorted_node_ids = sorted(genome.node_genes.keys())
        self._node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        self._idx_to_node_id = {idx: node_id for node_id, idx in self._node_id_to_idx.items()}

        # Build weight matrix (adjacency matrix representation)
        # weights[i, j] = weight of connection from node i to node j
        weights = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for conn in genome.conn_genes.values():
            if conn.enabled:
                from_idx = self._node_id_to_idx[conn.node_in]
                to_idx   = self._node_id_to_idx[conn.node_out]
                weights[from_idx, to_idx] = conn.weight
        self.weights = weights

        # Extract node parameters (bias, gain, activation)
        biases = np.zeros(num_nodes, dtype=np.float64)
        gains  = np.zeros(num_nodes, dtype=np.float64)
        activation_names = []

        for idx in range(num_nodes):
            node_id     = self._idx_to_node_id[idx]
            node_gene   = genome.node_genes[node_id]
            biases[idx] = node_gene.bias
            gains[idx]  = node_gene.gain
            activation_names.append(node_gene._activation_name)

        self.biases = biases
        self.gains  = gains
        self._activation_names = activation_names

        # Store activation functions by index for runtime use
        # Note: These functions are already vectorized (they use NumPy operations)
        # and work natively on arrays, so no np.vectorize wrapper is needed
        from neat.activations import activations
        self._activations = [activations[name] for name in activation_names]

        # Convert input/output IDs to indices
        self._input_indices  = np.array([self._node_id_to_idx[node_id] for node_id in self._input_ids], dtype=np.int32)
        self._output_indices = np.array([self._node_id_to_idx[node_id] for node_id in self._output_ids], dtype=np.int32)

        # Convert sorted nodes from IDs to indices
        self._sorted_indices = [self._node_id_to_idx[node_id] for node_id in self._sorted_nodes]

        # Pre-compute indices for non-input nodes
        input_indices_set = set(self._input_indices)
        self._hidden_output_indices = [idx for idx in self._sorted_indices if idx not in input_indices_set]

        # Pre-compute incoming connections as structured arrays for fast vectorized access
        # For each node, store array of source indices and corresponding weights
        self._incoming_sources = []  # List of arrays: incoming_sources[node_idx] = array of source indices
        self._incoming_weights = []  # List of arrays: incoming_weights[node_idx] = array of weights

        for node_idx in range(num_nodes):
            sources = []
            weights_list = []
            for from_idx in range(num_nodes):
                weight = self.weights[from_idx, node_idx]
                if weight != 0.0:
                    sources.append(from_idx)
                    weights_list.append(weight)

            # Convert to numpy arrays for fast indexing
            self._incoming_sources.append(np.array(sources, dtype=np.int32) if sources else np.array([], dtype=np.int32))
            self._incoming_weights.append(np.array(weights_list, dtype=np.float64) if weights_list else np.array([], dtype=np.float64))

        # Pre-allocate node values buffer (will be resized in forward_pass if needed)
        # This avoids allocating new arrays on every forward pass
        self._node_values = None
        self._last_batch_size = 0

        # Pre-allocate weighted sum buffer for reuse
        self._weighted_sum_buffer = None

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the network using optimized batch operations.

        This method supports both batched (2D) and non-batched (1D) inputs.
        Non-batched inputs are automatically converted to batch size 1.

        Parameters:
            inputs: Input values as numpy array or list
                    Shape: (batch_size, num_inputs) or (num_inputs,)
                    For single sample, use shape (num_inputs,) or (1, num_inputs)

        Returns:
            Output values as numpy array
            Shape: (batch_size, num_outputs)
        """
        # Convert to numpy array if needed
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs, dtype=np.float64)

        # Ensure inputs are 2D (batched). If 1D, convert to batch of size 1
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        elif inputs.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {inputs.ndim}D")

        batch_size = inputs.shape[0]

        # Validate input size
        expected_inputs = len(self._input_indices)
        actual_inputs   = inputs.shape[1]
        if actual_inputs != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {actual_inputs}")

        # Allocate or reuse node values buffer
        if self._node_values is None or self._last_batch_size != batch_size:
            self._node_values = np.zeros((batch_size, self._num_nodes), dtype=np.float64)
            self._weighted_sum_buffer = np.zeros(batch_size, dtype=np.float64)
            self._last_batch_size = batch_size
        else:
            # Reuse existing buffer - zero it out in-place
            self._node_values.fill(0.0)

        # Set input node values using advanced indexing
        # This is a single vectorized operation instead of a loop
        self._node_values[:, self._input_indices] = inputs

        # Propagate through hidden and output nodes in topological order
        for node_idx in self._hidden_output_indices:
            sources = self._incoming_sources[node_idx]
            weights = self._incoming_weights[node_idx]

            if len(sources) == 0:
                # No incoming connections - weighted sum is zero
                # Buffer already contains zeros, so we skip
                self._weighted_sum_buffer.fill(0.0)
            elif len(sources) == 1:
                # Single connection - direct multiplication (optimized path)
                # Use out parameter for in-place operation
                np.multiply(self._node_values[:, sources[0]], weights[0], out=self._weighted_sum_buffer)
            else:
                # Multiple connections - vectorized dot product
                # node_values[:, sources] creates view of shape (batch_size, num_sources)
                # weights has shape (num_sources,)
                # Result is (batch_size,)
                np.dot(self._node_values[:, sources], weights, out=self._weighted_sum_buffer)

            # Apply gain and bias in-place
            # weighted_sum = gain * weighted_sum + bias
            self._weighted_sum_buffer *= self.gains[node_idx]
            self._weighted_sum_buffer += self.biases[node_idx]

            # Apply activation function (already vectorized, works on arrays natively)
            activation_func = self._activations[node_idx]
            self._node_values[:, node_idx] = activation_func(self._weighted_sum_buffer)

        # Extract output values using advanced indexing
        outputs = self._node_values[:, self._output_indices]

        return outputs

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
        return (f"NetworkFast(nodes={self._num_nodes}, "
                f"hidden={self.number_nodes_hidden}, "
                f"connections={self.number_connections_enabled}/{self.number_connections})")
