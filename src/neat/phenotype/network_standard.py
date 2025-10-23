"""
NEAT Standard Network Module

This module implements the phenotype representation for the NEAT algorithm. 
It provides classes for expressing a genome as an executable neural network.
This is a 'standard' implementation of the Neural Network, using an Object
Oriented approach to representing Nodes, Connections and the Network.

Classes:
    Connection:      A weighted connection between two neurons
    Neuron:          A computational node applying activation functions
    NetworkStandard: A feedforward neural network from a genome
"""

from typing import Callable, Optional, TYPE_CHECKING

from neat.genotype.node_gene import NodeType  # Needed at runtime

if TYPE_CHECKING:
    from neat.genotype import ConnectionGene, Genome, NodeGene

from neat.phenotype.network_base import NetworkBase

class Connection:
    """
    A weighted connection between two neurons in a neural network.

    This class represents the phenotype manifestation of a ConnectionGene, implementing
    an connection that propagates signals between neurons during network execution. 
    Each connection wraps a ConnectionGene and provides read-only access to its properties.

    The connection transmits signals from a source neuron to a destination neuron,
    applying a weight multiplier to the signal. Connections can be enabled or disabled,
    allowing the network to preserve structural information while temporarily deactivating
    signal pathways.

    Public Properties:
        nodeID_in:  ID of the source neuron
        nodeID_out: ID of the destination neuron
        enabled:    Whether this connection is active in the network
        weight:     Weight multiplier applied to the transmitted signal
        innovation: Global innovation number identifying this connection

    The connection is identified by its innovation number, which matches the innovation
    number of its underlying ConnectionGene and serves as a unique identifier across the
    entire population.
    """

    def __init__(self, gene: "ConnectionGene"):
        """
        Parameters:
            gene: the gene encoding the Connection
        """
        self._gene: "ConnectionGene" = gene

    @property
    def nodeID_in(self) -> int:
        """The ID of the node/neuron representing the connection start."""
        return self._gene.node_in

    @property
    def nodeID_out(self) -> int:
        """The ID of the node/neuron representing the connection end."""
        return self._gene.node_out

    @property
    def enabled(self) -> bool:
        """Whether the connection is enabled."""
        return self._gene.enabled

    @property
    def weight(self) -> float:
        """The weight associated with this connection."""
        return self._gene.weight

    @property
    def innovation(self):
        """The globally unique ID associated with this connection."""
        return self._gene.innovation

    def __repr__(self):
        return (f"Connection(gene={self._gene})")

class Neuron:
    """
    A computational node (neuron) in a neural network.

    This class represents the phenotype manifestation of a NodeGene, implementing
    a neuron that processes signals during network execution. Each neuron wraps a 
    NodeGene and provides read-only access to its properties while maintaining
    internal state for output values during forward passes.

    Input neurons simply pass through their input unchanged. 
    Hidden and output neurons compute their output as: 
        activation(gain * weighted_input + bias)

    Public Attributes:
        output: The computed output value (None until calculated)

    Public Properties:
        id:         Globally unique ID for this neuron
        type:       Neuron type (INPUT, HIDDEN, or OUTPUT)
        bias:       Bias value added to weighted input
        gain:       Gain multiplier applied to weighted input
        activation: Activation function applied to (gain * weighted_input + bias)

    Public Methods:
        calculate_output(input_data): Compute and store the neuron's output value

    The neuron is identified by its ID, which matches the node ID of its underlying
    NodeGene and remains consistent across mutations and crossover operations.
    """

    def __init__(self, gene: "NodeGene"):
        """
        Parameters:
            gene: the gene encoding the Node/Neuron
        """
        self._gene : "NodeGene" = gene

        # the output of the calculation performed by the Neuron as:
        #     activation(gain * weighted_input + bias)
        self.output: Optional[float] = None

    @property
    def id(self) -> int:
        """The globally unique ID associated with this Node/Neuron."""
        return self._gene.id
    
    @property
    def type(self) -> NodeType:
        """The Node/Neuron type: INPUT, HIDDEN, OUTPUT"""
        return self._gene.type

    @property
    def bias(self) -> float:
        """The Neuron bias, used to calculate: output = activation(gain * weighted_input + bias)"""
        return self._gene.bias
    
    @property
    def gain(self) -> float:
        """The Neuron gain, used to calculate: output = activation(gain * weighted_input + bias)"""
        return self._gene.gain
    
    @property
    def activation(self) -> Callable[[float], float]:
        """The Neuron activation function, used to calculate: output = activation(gain * weighted_input + bias)"""
        return self._gene._activation

    def calculate_output(self, input_data: float) -> None:
        """
        Calculate the output of this node/neuron.
        The result is saved internally in 'self.output'.

        Parameters:
            input_data: the weighted input from each connection, using the connection weight
        """
        # an input node always outputs its input, un-modified
        if self.type == NodeType.INPUT:
            self.output = input_data
        else:
            if self.activation is None:
                raise ValueError(f"Neuron {self.id} of type {self.type.name} requires an activation function")
            self.output = self.activation(self.gain * input_data + self.bias)
            
    def __str__(self):
        return f"Neuron({self.id:+03d}, NodeType.{self.type.name:6s}, {self.bias}, {repr(self.activation)})"

    def __repr__(self):
        return (f"Neuron(gene={self._gene})")

class NetworkStandard(NetworkBase):
    """
    Object-oriented implementation of a NEAT neural network.

    This class implements NetworkBase using an object-oriented approach with
    explicit Neuron and Connection objects. It represents the traditional
    phenotype manifestation of a Genome, providing an executable neural network
    that can process inputs and produce outputs.

    Unlike array-based implementations (such as NetworkFast), this NetworkStandard uses:
    - Individual Neuron objects for each node (maintaining mutable state)
    - Individual Connection objects for each connection
    - Explicit iteration through neurons in topological order

    This object-based approach is well-suited for:
    - Single-input evaluation (one input at a time)
    - Network visualization and inspection
    - Educational purposes and debugging

    If you want a faster network (especially if input data is available in batches) use NetworkFast.
    If you need to calculate gradients, use NetworkAutograd.

    The network maintains a directed acyclic graph (DAG) structure where neurons
    are organized in topological order to enable efficient feedforward computation.
    During a forward pass, signals propagate from input neurons through hidden
    neurons to output neurons, with each connection applying its weight to the
    transmitted signal.

    Public Methods:
        forward_pass(inputs): Process inputs through the network and return outputs

    Public Properties (inherited from NetworkBase):
        number_nodes:               Total number of nodes in the network
        number_nodes_hidden:        Number of hidden nodes in the network
        number_connections:         Total number of connections in the network
        number_connections_enabled: Number of enabled connections in the network
    """

    def __init__(self, genome: "Genome"):
        """
        Parameters:
            genome: the Genome encoding the network
        """
        # Initialize common base class attributes
        super().__init__(genome)

        # Create nodes/neurons objects from node genes
        self._neurons: dict[int, Neuron] = {}
        for gene in genome.node_genes.values():
            self._neurons[gene.id] = Neuron(gene)

        # Create network connections objects from connection genes
        self._connections: dict[int, Connection] = {}
        for gene in genome.conn_genes.values():
            self._connections[gene.innovation] = Connection(gene)

        # For each neuron, build list of incoming connections (both enabled and disabled)
        self._incoming_connections: dict[int, list[Connection]] = {}   # neuron ID => [Connection instance]
        for conn in self._connections.values():
            self._incoming_connections.setdefault(conn.nodeID_out, []).append(conn)

    def forward_pass(self, inputs: tuple[float]) -> list[float]:
        """
        Perform a complete forward pass through the network.

        Parameters:
            the network inputs (as many as input neurons)

        Returns:
            the results of passing the inputs through the network (as many as output neurons)
        """
        # The number of inputs must match the number of input neurons
        if len(inputs) != len(self._input_ids):
            raise ValueError(f"Expected {len(self._input_ids)} inputs, got {len(inputs)}")
            
        # Reset the output of all neurons
        for neuron in self._neurons.values():
            neuron.output = None
            
        # Set input values
        for i, input_id in enumerate(self._input_ids):
            self._neurons[input_id].calculate_output(inputs[i])

        # Propagate values through the network, in topological order
        for node_id in self._sorted_nodes:
            if node_id not in self._input_ids:    # skip input nodes, already set
                neuron     = self._neurons[node_id]
                conns_in   = self._incoming_connections.get(node_id, [])
                input_data = sum(c.weight * self._neurons[c.nodeID_in].output for c in conns_in if c.enabled)
                neuron.calculate_output(input_data)
        
        # Get output values
        return [self._neurons[ID].output for ID in self._output_ids]

    def __str__(self):
        neurons_str = "\n".join([f"  {neuron}" for neuron_id, neuron in self._neurons.items()])
        connections_str = "\n".join([f"  {conn}" for conn in self._connections])
        return f"{neurons_str},\n\n{connections_str}"
