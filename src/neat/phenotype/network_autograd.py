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
from neat.phenotype.network_base import NetworkBase

class NetworkAutograd(NetworkBase):
    """
    Autograd-compatible batch-processing neural network for NEAT.

    Public Methods:
        forward_pass(inputs, weights=None, biases=None, gains=None):
            Process batch through network with optional explicit parameters
            Input:  (batch_size, num_inputs) or (num_inputs,) auto-reshaped to (1, num_inputs)
            Output: (batch_size, num_outputs)

        get_parameters():
            Get current network parameters as (weights, biases, gains) tuple

        set_parameters(weights, biases, gains, enforce_bounds=True):
            Update network parameters with optional bounds enforcement

        save_parameters_to_genome(enforce_bounds=True):
            Save current network parameters back into the genome (Lamarckian evolution)

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
            activation_names.append(node_gene._activation_name)

        # Convert to numpy arrays
        self.biases = np.array(bias_list)
        self.gains  = np.array(gain_list)
        self._activation_names = activation_names

        # Store activation functions by index for runtime use
        from neat.activations import activations
        self._activations = [activations[name] for name in activation_names]

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

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current network parameters.

        Returns:
            tuple: (weights, biases, gains) as numpy arrays
                - weights: (num_nodes, num_nodes) weight matrix
                - biases:  (num_nodes,) bias vector
                - gains:   (num_nodes,) gain vector
        """
        return self.weights, self.biases, self.gains

    def set_parameters(self, weights, biases, gains, enforce_bounds=True):
        """
        Set network parameters.

        NOTE: This assumes network topology (connection structure) does not change.
        Only parameter VALUES are updated. Topology is fixed after initialization.

        Parameters:
            weights: Weight matrix (num_nodes, num_nodes)
            biases: Bias vector (num_nodes,)
            gains:  Gain vector (num_nodes,)
            enforce_bounds: If True, clip parameters to config bounds (default: True)
        """
        if enforce_bounds:
            config  = self._genome._config
            weights = np.clip(weights, config.min_weight, config.max_weight)
            biases  = np.clip(biases , config.min_bias  , config.max_bias)
            gains   = np.clip(gains  , config.min_gain  , config.max_gain)

        self.weights = weights
        self.biases  = biases
        self.gains   = gains

    def forward_pass(self, inputs: np.ndarray | list, weights=None, biases=None, gains=None) -> np.ndarray:
        """
        Perform forward pass through the network using batched matrix operations.

        This network is optimized for batch processing. Single samples should be
        converted to a batch of size 1 by reshaping: input.reshape(1, -1)

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
            weights, biases, gains = network.get_parameters()

            # Define loss function that uses explicit parameters
            def loss_fn(w, b, g):
                outputs = network.forward_pass(inputs, weights=w, biases=b, gains=g)
                return np.mean((outputs - targets)**2)

            # Compute gradients with respect to parameters
            grad_w = autograd.grad(loss_fn, argnum=0)(weights, biases, gains)
            grad_b = autograd.grad(loss_fn, argnum=1)(weights, biases, gains)
            grad_g = autograd.grad(loss_fn, argnum=2)(weights, biases, gains)

        Parameters:
            inputs: Batched input values as numpy array or list
                    Shape: (batch_size, num_inputs)
                    For single sample, use shape (1, num_inputs)
            weights: Optional weight matrix (num_nodes, num_nodes). Uses self.weights if None.
            biases:  Optional bias vector (num_nodes,). Uses self.biases if None.
            gains:   Optional gain vector (num_nodes,). Uses self.gains if None.

        Returns:
            Batched output values as numpy array
            Shape: (batch_size, num_outputs)
        """
        weights = weights if weights is not None else self.weights
        biases  = biases  if biases  is not None else self.biases
        gains   = gains   if gains   is not None else self.gains

        # Convert to numpy array if needed (autograd-compatible)
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

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

        # Initialize node values using a Python list of arrays (autograd-compatible)
        # Each list element stores a (batch_size,) array for that node's value across the batch
        #
        # Why use a list instead of a single 2D array?
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

        # Set input node values directly from input array
        for i, input_idx in enumerate(self._input_indices):
            node_values_list[input_idx] = inputs[:, i]

        # Propagate through hidden and output nodes in topological order
        for node_idx in self._hidden_output_indices:

            # Nodes whose output is sent to the current node
            incoming_indices = self._incoming_connections[node_idx]

            # Compute weighted sum of inputs (use functional style: 
            #   weighted_sum = weighted_sum + ... 
            # creates new arrays (we do want to avoid in-place updates)
            weighted_sum = np.zeros(batch_size)
            for source_idx in incoming_indices:
                weight = weights[source_idx, node_idx] 
                weighted_sum = weighted_sum + node_values_list[source_idx] * weight

            # Apply gain, bias, and activation function
            activation_func = self._activations[node_idx]
            activated_value = activation_func(gains[node_idx] * weighted_sum + biases[node_idx])

            # Store result by replacing list element (preserves autograd computation graph)
            node_values_list[node_idx] = activated_value

        # Extract output values: stack into (batch_size, num_outputs) array
        outputs = np.column_stack([node_values_list[idx] for idx in self._output_indices])

        return outputs

    def load_parameters_from_genome(self, enforce_bounds=True):
        """
        Load network parameters from the genome (reverse of save_parameters_to_genome).

        This method restores the network's parameters to match the genome's current state.
        Useful when gradient descent has modified network parameters temporarily for
        fitness evaluation (Baldwin effect), and you want to revert to the genome's
        parameters before reproduction/cloning.

        Parameters:
            enforce_bounds: If True, clip parameters to config bounds after loading (default: True)

        Note:
            Only updates parameter values (weights, biases, gains). Network topology
            (connections, nodes) remains unchanged.
        """
        # Build weight matrix from genome
        weights = np.zeros((self._num_nodes, self._num_nodes))
        for conn_gene in self._genome.conn_genes.values():
            if conn_gene.enabled:
                from_idx = self._node_id_to_idx[conn_gene.node_in]
                to_idx   = self._node_id_to_idx[conn_gene.node_out]
                weights[from_idx, to_idx] = conn_gene.weight

        # Build bias and gain arrays from genome
        bias_list = []
        gain_list = []
        for idx in range(self._num_nodes):
            node_id   = self._idx_to_node_id[idx]
            node_gene = self._genome.node_genes[node_id]
            bias_list.append(node_gene.bias)
            gain_list.append(node_gene.gain)

        biases = np.array(bias_list)
        gains  = np.array(gain_list)

        # Optionally enforce parameter bounds
        if enforce_bounds:
            config  = self._genome._config
            weights = np.clip(weights, config.min_weight, config.max_weight)
            biases  = np.clip(biases , config.min_bias  , config.max_bias)
            gains   = np.clip(gains  , config.min_gain  , config.max_gain)

        # Update network parameters
        self.weights = weights
        self.biases  = biases
        self.gains   = gains

    def save_parameters_to_genome(self, enforce_bounds=True):
        """
        Save current network parameters back into the genome (Lamarckian evolution).

        This method updates the genome's node and connection genes with the current
        network parameters, allowing gradient-descent-optimized parameters to be
        inherited by offspring through crossover and passed to future generations.

        This implements Lamarckian evolution: acquired characteristics (parameters
        learned through gradient descent) can be inherited, unlike purely Darwinian
        evolution where only random mutations and selection occur.

        Parameters:
            enforce_bounds: If True, clip parameters to config bounds before saving (default: True)

        Note:
            Only updates parameter values (weights, biases, gains). Network topology
            (connections, nodes) is never modified by this method.
        """
        weights = self.weights
        biases  = self.biases
        gains   = self.gains

        # Optionally enforce parameter bounds
        if enforce_bounds:
            config  = self._genome._config
            weights = np.clip(weights, config.min_weight, config.max_weight)
            biases  = np.clip(biases , config.min_bias  , config.max_bias)
            gains   = np.clip(gains  , config.min_gain  , config.max_gain)

        # Update node genes (biases and gains)
        for idx in range(self._num_nodes):
            node_id   = self._idx_to_node_id[idx]
            node_gene = self._genome.node_genes[node_id]

            # Convert numpy types to Python native types for cleaner storage
            node_gene.bias = float(biases[idx])
            node_gene.gain = float(gains[idx])

        # Update connection genes (weights)
        # Iterate through all connections in the genome
        for conn_gene in self._genome.conn_genes.values():
            if conn_gene.enabled:
                from_idx = self._node_id_to_idx[conn_gene.node_in]
                to_idx   = self._node_id_to_idx[conn_gene.node_out]

                # Update weight from the weight matrix
                conn_gene.weight = float(weights[from_idx, to_idx])

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
