"""
NEAT Network Base Module

This module defines the abstract base class for the NEAT neural network implementation.
It provides a common interface and shared functionality for different network backends
(object-oriented, numpy vectorized, autograd compatible etc.).

Classes:
    NetworkBase: Abstract base class defining the network interface
"""

from abc         import ABC, abstractmethod
from collections import deque, defaultdict
from typing      import Any, TYPE_CHECKING
import graphviz  # type: ignore

if TYPE_CHECKING:
    from neat.genotype import Genome

class NetworkBase(ABC):
    """
    Abstract base class for NEAT neural network implementations.

    This class defines the common interface that all network implementations
    must follow, regardless of their internal representation (object-oriented,
    numpy arrays, PyTorch tensors, etc.).

    The base class provides:
        - Common initialization
        - Topological sort algorithm
        - Standard network introspection properties
        - Network visualization

    Public Properties (available to all subclasses):
        number_nodes:               Total number of nodes in the network
        number_nodes_hidden:        Number of hidden nodes in the network
        number_connections:         Total number of connections in the network
        number_connections_enabled: Number of enabled connections in the network

    Public Methods (must be implemented by subclasses):
        forward_pass(inputs): Process inputs through the network and return outputs
    """

    def __init__(self, genome: 'Genome'):
        """
        Initialize common network attributes from genome.

        Parameters:
            genome: The Genome encoding the network structure
        """
        self._genome       = genome
        self._input_ids    = [gene.id for gene in genome.input_nodes]
        self._output_ids   = [gene.id for gene in genome.output_nodes]
        self._sorted_nodes = self._topological_sort(genome)

    @property
    def number_nodes(self) -> int:
        """Total number of nodes in the network."""
        return len(self._genome.node_genes)

    @property
    def number_nodes_hidden(self) -> int:
        """Number of hidden nodes in the network."""
        return len(self._genome.node_genes) - len(self._input_ids) - len(self._output_ids)

    @property
    def number_connections(self) -> int:
        """Total number of connections in the network."""
        return len(self._genome.conn_genes)

    @property
    def number_connections_enabled(self) -> int:
        """Number of enabled connections in the network."""
        return sum(1 for conn in self._genome.conn_genes.values() if conn.enabled)

    @abstractmethod
    def forward_pass(self, inputs: Any) -> Any:
        """
        Perform a complete forward pass through the network.

        Parameters:
            inputs: Network inputs (implementation-specific type)

        Returns:
            Network outputs (implementation-specific type)
        """
        pass

    @staticmethod
    def _topological_sort(genome: 'Genome') -> list[int]:
        """
        Perform topological sort using Kahn's algorithm.

        Sorts the network nodes in topological order, ensuring that all
        dependencies (incoming connections) are processed before each node.
        Assumes the network is a DAG (no cycles).

        Parameters:
            genome: The Genome containing node and connection genes

        Returns:
            List of node IDs in topological order
        """
        # Get all node IDs from the genome
        node_ids = list(genome.node_genes.keys())

        # Build adjacency list for efficiency
        adjacency = defaultdict(list)
        in_degree = {node_id: 0 for node_id in node_ids}

        # Build graph from enabled connections only
        for conn in genome.conn_genes.values():
            if conn.enabled:
                adjacency[conn.node_in].append(conn.node_out)
                in_degree[conn.node_out] += 1

        # Start with nodes that have no incoming edges
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Process all outgoing edges
            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def visualize(self, view: bool = True) -> graphviz.Digraph:
        """
        Visualize the network using Graphviz.

        This method works for all network implementations (NetworkStandard, NetworkJAX, etc.)
        by accessing the genome directly rather than implementation-specific structures.

        Parameters:
            view: If True, automatically open the visualization after rendering

        Returns:
            graphviz.Digraph object representing the network
        """
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')  # Left to right layout
        dot.attr('graph', labelloc='t')

        # Define node colors and shapes
        node_attrs = {
            'INPUT':  {'fillcolor': 'lightgrey', 'color': 'black', 'style': 'filled', 'shape': 'circle', 'penwidth': '0.5', 'fontsize': '5', 'width': '0.5', 'height': '0.5', 'fixedsize': 'true'},
            'HIDDEN': {'fillcolor': 'lightblue', 'color': 'black', 'style': 'filled', 'shape': 'circle', 'penwidth': '0.5', 'fontsize': '5', 'width': '0.5', 'height': '0.5', 'fixedsize': 'true'},
            'OUTPUT': {'fillcolor': 'white'    , 'color': 'black', 'style': 'filled', 'shape': 'circle', 'penwidth': '0.5', 'fontsize': '5', 'width': '0.5', 'height': '0.5', 'fixedsize': 'true'}
        }

        # Create subgraphs for better layout
        with dot.subgraph(name='cluster_input') as input_cluster:
            input_cluster.attr(rank='source', label='Inputs', style='invisible')
            for node_id in sorted(self._input_ids):
                node_gene = self._genome.node_genes[node_id]
                attrs = node_attrs[node_gene.type.name].copy()
                attrs['label'] = f"id={node_id}\\nbias={node_gene.bias:.2f}\ngain={node_gene.gain:.2f}"
                input_cluster.node(str(node_id), **attrs)

        # Add hidden nodes if any
        hidden_nodes = [n for n in self._genome.node_genes.keys() if n not in self._input_ids and n not in self._output_ids]
        if hidden_nodes:
            with dot.subgraph(name='cluster_hidden') as hidden_cluster:
                hidden_cluster.attr(rank='same', label='Hidden', style='invisible')
                for node_id in sorted(hidden_nodes):
                    node_gene = self._genome.node_genes[node_id]
                    attrs = node_attrs[node_gene.type.name].copy()
                    attrs['label'] = f"id={node_id}\\nbias={node_gene.bias:.2f}\ngain={node_gene.gain:.2f}"
                    hidden_cluster.node(str(node_id), **attrs)

        with dot.subgraph(name='cluster_output') as output_cluster:
            output_cluster.attr(rank='sink', label='Outputs', style='invisible')
            for node_id in sorted(self._output_ids):
                node_gene = self._genome.node_genes[node_id]
                attrs = node_attrs[node_gene.type.name].copy()
                attrs['label'] = f"id={node_id}\\nbias={node_gene.bias:.2f}\ngain={node_gene.gain:.2f}"
                output_cluster.node(str(node_id), **attrs)

        # Add edges with weights (both enabled and disabled)
        for conn in self._genome.conn_genes.values():
            edge_attrs = {
                'label': f"i={conn.innovation},w={conn.weight:.2f}",
                'fontsize' : '5',
                'penwidth' : '0.5',
                'arrowsize': '0.5',
                'labelfloat': 'false'
            }

            # Use light gray color for disabled connections, black for enabled
            if conn.enabled:
                edge_attrs['color'] = 'black'
            else:
                edge_attrs['color'] = 'lightgray'

            dot.edge(str(conn.node_in), str(conn.node_out), **edge_attrs)

        if view:
            dot.view(cleanup=True)

        return dot
