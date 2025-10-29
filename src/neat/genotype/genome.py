"""
NEAT Genome Module

This module implements the Genome class for the NEAT
(NeuroEvolution of Augmenting Topologies) algorithm.

Classes:
    Genome: Complete genome representing a neural network structure
"""

import copy
import numpy as np
import random
from neat.run.config                  import Config
from neat.genotype.connection_gene    import ConnectionGene
from neat.genotype.innovation_tracker import InnovationTracker
from neat.genotype.node_gene          import NodeType, NodeGene

class Genome:
    """
    A NEAT genome representing a neural network as a collection of node and connection genes.

    In the NEAT (NeuroEvolution of Augmenting Topologies) algorithm, a genome encodes
    the structure and parameters of a neural network at the genotype level. It consists of:
    - Node genes: describe network nodes (input, hidden, output) with their parameters
    - Connection genes: describe weighted connections between nodes, each with a unique
      innovation number for tracking historical markings during crossover

    A minimal genome contains only input and output nodes with no connections. Via mutation 
    operations, genomes can grow by adding nodes and connections, forming increasingly complex 
    network topologies while maintaining a DAG structure.

    Node numbering convention:
        - Input nodes:  [0, num_inputs)
        - Output nodes: [num_inputs, num_inputs + num_outputs)
        - Hidden nodes: [num_inputs + num_outputs, ...)
    
    Attributes:
        node_genes: Dictionary mapping node IDs to NodeGene objects
        conn_genes: Dictionary mapping innovation numbers to ConnectionGene objects

    Public Properties:
        input_nodes:  List of all input node genes
        output_nodes: List of all output node genes
        hidden_nodes: List of all hidden node genes

    Public Methods:
        distance(other):                 Calculate genetic distance to another genome
        prune():                         Create a pruned copy of this genome by removing dead-end nodes and disabled connections.
        crossover(other, fitter_parent): Create offspring by crossing this genome with another
        mutate():                        Apply all possible mutation operations stochastically
        to_dict():                       Convert genome to dictionary representation

    Class Methods:
        from_dict(genome_dict): Create a genome from a dictionary description

    Static Methods:
        show_aligned(genome1, genome2): Print two genomes with aligned genes for comparison
    """

    def __init__(self, config: Config):
        """
        Initialize a minimal Genome.

        A minimal genome is defined as a genome that describes the smallest possible network:
        a network consisting of only input and output nodes (whose number never changes and is
        retrieved from the configuration file) and having no connections. The number of input
        and output nodes is retrieved from the Config object.

        Parameters:
            config: Stores configuration parameters
        """
        self._config = config

        self.node_genes: dict[int, NodeGene]       = {}  # node ID => node gene
        self.conn_genes: dict[int, ConnectionGene] = {}  # innovation number => connection gene

        # A minimal network has only input and output nodes (no hidden nodes).
        # A minimal network does not have any connections.

        # Initialize input nodes
        # By convention, input nodes are numbered: [0, NUMBER INPUT NODES - 1)
        for i in range(self._config.num_inputs):
            node_id = i
            input_node = NodeGene(node_id, NodeType.INPUT, self._config, bias=0.0, gain=1.0)
            self.node_genes[node_id] = input_node

        # Initialize output nodes
        # By convention, output nodes are numbered: [NUMBER INPUT NODES, NUMBER INPUT NODES + NUMBER OUTPUT NODES - 1)
        for i in range(self._config.num_outputs):
            node_id = self._config.num_inputs + i
            output_node = NodeGene(node_id, NodeType.OUTPUT, self._config)
            self.node_genes[node_id] = output_node

    @classmethod
    def from_dict(cls, genome_dict: dict) -> 'Genome':
        """
        Create a Genome from a dictionary description.

        This method allows programmatic creation of genomes with specific structures.
        The dictionary specifies nodes and connections, and the method validates that
        the structure follows NEAT conventions and is acyclic.

        Dictionary format:
            {
                "activation": "sigmoid",  # Optional global activation for all nodes
                "nodes": [
                    {"id": 0, "type": "input"},
                    {"id": 1, "type": "input"},
                    {"id": 2, "type": "output", "bias": 0.0, "gain": 1.0},
                    {"id": 3, "type": "hidden", "bias": 0.5, "gain": 1.0, "activation": "relu"}
                ],
                "connections": [
                    {"from": 0, "to": 3, "weight":  0.5, "enabled": true},
                    {"from": 1, "to": 3, "weight": -0.3, "enabled": true},
                    {"from": 3, "to": 2, "weight":  1.5, "enabled": true}
                ]
            }

        Activation function resolution:
        - Each node (hidden/output) first looks for its own "activation" field
        - If not found, uses the global "activation" field
        - If neither exists, raises ValueError
        - Node-specific activations override the global activation

        Note: For learnable activations (e.g., "legendre"), nodes may include an
        optional "activation_coeffs" field with a list of coefficient values.

        Node numbering convention (validated by this method):
            - Input nodes:  [0, num_inputs)
            - Output nodes: [num_inputs, num_inputs + num_outputs)
            - Hidden nodes: [num_inputs + num_outputs, ...)

        Note:
            This method automatically initializes the InnovationTracker with the
            inferred num_inputs and num_outputs. This resets any previous innovation
            tracking state.

        Parameters:
            genome_dict: Dictionary describing the genome structure

        Returns:
            A new Genome object with the specified structure

        Raises:
            ValueError: If the structure is invalid (wrong node numbering, cycles, etc.)
            KeyError: If required fields are missing from the dictionary
        """
        # Parse nodes data
        nodes_data   = genome_dict["nodes"]
        input_nodes  = [n for n in nodes_data if n["type"] == "input"]
        output_nodes = [n for n in nodes_data if n["type"] == "output"]
        hidden_nodes = [n for n in nodes_data if n["type"] == "hidden"]
        num_inputs   = len(input_nodes)
        num_outputs  = len(output_nodes)

        # Validate node numbering convention
        cls._validate_node_numbering(input_nodes, output_nodes, hidden_nodes, num_inputs, num_outputs)

        # Get optional network-level default activation
        network_activation = genome_dict.get("activation", None)

        # Create minimal Config object with necessary fields
        config = Config(config_file=None)
        config.num_inputs         = num_inputs
        config.num_outputs        = num_outputs
        config.activation_initial = genome_dict.get("activation")
        config.initial_cxn_policy = "none"

        # Initialize InnovationTracker for this genome
        # This ensures innovation numbers are assigned correctly for connections
        InnovationTracker.initialize(config)

        # Create empty genome
        genome = cls.__new__(cls)
        genome._config    = config
        genome.node_genes = {}
        genome.conn_genes = {}

        # Add input nodes
        for node_data in input_nodes:
            ID   = node_data["id"]
            node = NodeGene(ID, NodeType.INPUT, config, bias=0.0, gain=1.0)
            genome.node_genes[ID] = node

        # Add hidden nodes
        for node_data in hidden_nodes:
            ID     = node_data["id"]
            bias   = node_data.get("bias", 0.0)
            gain   = node_data.get("gain", 1.0)
            coeffs = np.array(node_data["activation_coeffs"]) if "activation_coeffs" in node_data else None

            if "activation" in node_data:
                actname = node_data["activation"]
            elif "activation" in genome_dict:
                actname = genome_dict["activation"]
            else:
                raise ValueError(f"No activation function specified for hidden node {ID}.")

            node = NodeGene(ID, NodeType.HIDDEN, config, bias, gain, actname, coeffs)
            genome.node_genes[ID] = node

        # Add output nodes
        for node_data in output_nodes:
            ID     = node_data["id"]
            bias   = node_data.get("bias", 0.0)
            gain   = node_data.get("gain", 1.0)
            coeffs = np.array(node_data["activation_coeffs"]) if "activation_coeffs" in node_data else None

            if "activation" in node_data:
                actname = node_data["activation"]
            elif "activation" in genome_dict:
                actname = genome_dict["activation"]
            else:
                raise ValueError(f"No activation function specified for output node {ID}.")

            node = NodeGene(ID, NodeType.OUTPUT, config, bias, gain, actname, coeffs)
            genome.node_genes[ID] = node

        # Add connections and validate network is acyclic
        connections_data = genome_dict.get("connections", [])
        for conn_data in connections_data:
            node_in  = conn_data["from"]
            node_out = conn_data["to"]
            weight   = conn_data["weight"]
            enabled  = conn_data.get("enabled", True)

            # Validate that nodes exist
            if node_in not in genome.node_genes:
                raise ValueError(f"Connection references non-existent source node: {node_in}")
            if node_out not in genome.node_genes:
                raise ValueError(f"Connection references non-existent destination node: {node_out}")

            # Validate that connection wouldn't create a cycle
            if genome._would_create_cycle(node_in, node_out):
                raise ValueError(f"Connection from {node_in} to {node_out} would create a cycle")

            # Get innovation number for this connection
            innovation = InnovationTracker.get_innovation_number(node_in, node_out)

            # Create and add connection
            conn = ConnectionGene(node_in, node_out, weight, innovation, config, enabled=enabled)
            genome.conn_genes[innovation] = conn

        return genome

    def to_dict(self) -> dict:
        """
        Convert the genome to a dictionary representation.

        This is the inverse operation of from_dict(), producing
        a dictionary that can be used to reconstruct the genome.

        Activation function serialization:
        - Global "activation" field contains the config activation (may be missing)
        - Nodes include "activation" field only if:
          - Their activation differs from the global activation, OR
          - No global activation is defined
        - This preserves heterogeneous networks (e.g., from "random" activation)

        Note: For learnable activations (e.g., "legendre"), nodes
        include an "activation_coeffs" field with a list of values
        of the coefficients.

        Returns:
            Dictionary with the following structure:
            {
                "activation": "sigmoid",  # Global activation (optional)
                "nodes": [
                    {"id": 0, "type": "input"},
                    {"id": 1, "type": "output", "bias": 0.0, "gain": 1.0},
                    {"id": 2, "type": "hidden", "bias": 0.5, "gain": 1.0, "activation": "relu"}
                ],
                "connections": [
                    {"from": 0, "to": 2, "weight":  0.5, "enabled": true},
                    {"from": 2, "to": 1, "weight":  1.5, "enabled": true}
                ]
            }
        """
        # Build nodes list
        nodes = []

        # Add input nodes (sorted by ID)
        for node in sorted(self.input_nodes, key=lambda n: n.id):
            nodes.append({
                "id"  : node.id,
                "type": "input"
            })

        # Add output nodes (sorted by ID)
        for node in sorted(self.output_nodes, key=lambda n: n.id):
            node_dict = {
                "id"        : node.id,
                "type"      : "output",
                "bias"      : node.bias,
                "gain"      : node.gain,
                "activation": node.activation_name
            }
            # Include node-specific activation if different from global or if no global activation
            if node.activation_name != self._config.activation_initial or self._config.activation_initial is None:
                node_dict["activation"] = node.activation_name
            if node.activation_coeffs is not None:
                node_dict["activation_coeffs"] = node.activation_coeffs.tolist()
            nodes.append(node_dict)

        # Add hidden nodes (sorted by ID)
        for node in sorted(self.hidden_nodes, key=lambda n: n.id):
            node_dict = {
                "id"        : node.id,
                "type"      : "hidden",
                "bias"      : node.bias,
                "gain"      : node.gain,
                "activation": node.activation_name
            }
            # Include node-specific activation if different from global or if no global activation
            if node.activation_name != self._config.activation_initial or self._config.activation_initial is None:
                node_dict["activation"] = node.activation_name
            if node.activation_coeffs is not None:
                node_dict["activation_coeffs"] = node.activation_coeffs.tolist()
            nodes.append(node_dict)

        # Build connections list (sorted by innovation number)
        connections = []
        for conn in sorted(self.conn_genes.values(), key=lambda c: c.innovation):
            connections.append({
                "from"   : conn.node_in,
                "to"     : conn.node_out,
                "weight" : conn.weight,
                "enabled": conn.enabled
            })

        # Build final dictionary
        result = {
            "nodes"      : nodes,
            "connections": connections
        }
        if self._config.activation_initial is not None:
            result["activation"] = self._config.activation_initial

        return result

    @staticmethod
    def _validate_node_numbering(input_nodes : list, 
                                 output_nodes: list, 
                                 hidden_nodes: list,
                                 num_inputs  : int, 
                                 num_outputs : int) -> None:
        """
        Validate that nodes follow the NEAT numbering convention.

        Node numbering convention:
            - Input nodes:  [0, num_inputs)
            - Output nodes: [num_inputs, num_inputs + num_outputs)
            - Hidden nodes: [num_inputs + num_outputs, ...)

        Parameters:
            input_nodes:  List of input node dictionaries
            output_nodes: List of output node dictionaries
            hidden_nodes: List of hidden node dictionaries
            num_inputs:   Number of input nodes
            num_outputs:  Number of output nodes

        Raises:
            ValueError: If node numbering doesn't follow the convention
        """
        # Check input nodes are numbered [0, num_inputs)
        input_ids = sorted([n["id"] for n in input_nodes])
        expected_input_ids = list(range(num_inputs))
        if input_ids != expected_input_ids:
            raise ValueError(f"Input nodes must be numbered {expected_input_ids}, got {input_ids}")

        # Check output nodes are numbered [num_inputs, num_inputs + num_outputs)
        output_ids = sorted([n["id"] for n in output_nodes])
        expected_output_ids = list(range(num_inputs, num_inputs + num_outputs))
        if output_ids != expected_output_ids:
            raise ValueError(f"Output nodes must be numbered {expected_output_ids}, got {output_ids}")

        # Check hidden nodes are numbered >= num_inputs + num_outputs
        hidden_ids = [n["id"] for n in hidden_nodes]
        min_hidden_id = num_inputs + num_outputs
        for hid in hidden_ids:
            if hid < min_hidden_id:
                raise ValueError(f"Hidden node {hid} has ID below minimum {min_hidden_id}")

        # Check for duplicate node IDs
        all_ids = input_ids + output_ids + hidden_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate node IDs found in node list")

    @property        
    def input_nodes(self) -> list[NodeGene]:
        return [node for node in self.node_genes.values() if node.type == NodeType.INPUT]
        
    @property
    def output_nodes(self) -> list[NodeGene]:
        return [node for node in self.node_genes.values() if node.type == NodeType.OUTPUT]
        
    @property
    def hidden_nodes(self) -> list[NodeGene]:
        return [node for node in self.node_genes.values() if node.type == NodeType.HIDDEN]

    def distance(self, other: 'Genome') -> float:
        """
        Calculate genetic distance between this genome and another.

        The genetic distance is the sum two components:
        + a term calculated using the original NEAT formula, based on connection genes
        + a term quantifying the parameter difference between the matching nodes in the
          two networks
        This second term is not part of the 'classical' NEAT approach. Its contribution
        to the final result is controlled by a weight (specified on the configuration
        file), which can be set to 0 to recover the original distance.

        Parameters:
            other: the genome relative to which we are calculating the distance

        Returns:
            the genetic distance between this genome and 'other'
        """
        # distance based on connections
        distance = self._distance_NEAT(other)

        # also include the distance between homologus nodes in the calculation
        if self._config.distance_includes_nodes:
            distance += self._distance_nodes(other)
        return distance
    
    def _distance_NEAT(self, other: 'Genome') -> float:
        """
        Calculate genetic distance between this genome and another using the original NEAT formula.
        
        The original NEAT formula only looks at connections.
           distance = (c1 * E / N) + (c2 * D / N) + c3 * W̄

        Where:
        - E = number of excess connection genes
        - D = number of disjoint connection genes
        - N = number of connection genes in larger genome
        - W̄ = average weight difference of matching connection genes
        - c1, c2, c3 = weight of various terms (from configuration file)

        Parameters:
            other: the genome relative to which we are calculating the distance

        Returns:
            the original NEAT distance between this genome and 'other'
        """

        # Get innovation numbers from both genomes
        innovs1 = set(self.conn_genes.keys())
        innovs2 = set(other.conn_genes.keys())
        if not innovs1 and not innovs2:
            return 0.0

        # Find matching, disjoint, and excess genes
        matching_innovs     =  innovs1 & innovs2
        non_matching_innovs = (innovs1 | innovs2) - matching_innovs

        max_innov1 = max(innovs1) if innovs1 else -1
        max_innov2 = max(innovs2) if innovs2 else -1

        # Excess   genes: beyond the smaller genome's max innovation number
        # Disjoint genes: within the overlapping range but not matching
        num_excess   = 0
        num_disjoint = 0
        for innov in non_matching_innovs:
            if innov > min(max_innov1, max_innov2):
                num_excess += 1
            else:
                num_disjoint += 1

        # Average connection weight difference for matching connection genes
        avg_weight_diff = 0.0
        if matching_innovs:
            weight_diff = sum(abs(self.conn_genes[i].weight - other.conn_genes[i].weight) for i in matching_innovs)
            avg_weight_diff = weight_diff / len(matching_innovs)
        
        # Calculate distance
        N = max(len(self.conn_genes), len(other.conn_genes))
        distance = (self._config.distance_excess_coeff   * num_excess   / N +
                    self._config.distance_disjoint_coeff * num_disjoint / N +
                    self._config.distance_params_coeff   * avg_weight_diff)
        return distance

    def _distance_nodes(self, other: 'Genome') -> float:
        """
        Calculate the node based component of the genetic distance between this genome and another.

        This component is not part of the original NEAT formula.
        It calculates a component of genetic distance between two networks
        by quantifying the parameter difference between the matching nodes.
        The parameters involved in the calculation are 'bias', 'gain',
        'activation_coeffs' (if using legendre activation), and activation function type.

        Parameters:
            other: the genome relative to which we are calculating the distance

        Returns:
            the nodes based component of the genetic distance between this genome and 'other'
        """

        # Identify the matching nodes in the two networks
        node_ids1    = set(self.node_genes.keys())
        node_ids2    = set(other.node_genes.keys())
        matching_ids = node_ids1 & node_ids2

        # Sum up the difference in 'bias', 'gain', 'activation_coeffs'
        # and activation type for all matching nodes
        params_diff = 0.0
        num_params  = 0
        for node_id in matching_ids:
            node1 = self.node_genes [node_id]
            node2 = other.node_genes[node_id]

            # Distance due to parameter difference
            params_diff += abs(node1.bias - node2.bias)
            params_diff += abs(node1.gain - node2.gain)
            num_params  += 2

            # Distance due to activation function difference
            if node1.activation_name != node2.activation_name:

                # Different activation types: add 1.0
                params_diff += 1.0
                num_params  += 1

            elif node1.activation_name == 'legendre':

                # Same activation type (both legendre): compare coefficients.
                # Calculate mean absolute difference and map to [0, 1] via tanh(k * mean_diff)
                # where k is a configurable scaling factor that controls sensitivity
                mean_coeff_diff = np.mean(np.abs(node1.activation_coeffs - node2.activation_coeffs))
                params_diff    += np.tanh(self._config.activation_distance_k * mean_coeff_diff)
                num_params     += 1

        # normalize the difference
        if num_params > 0:
            params_diff /= num_params
        if matching_ids:
            params_diff /= len(matching_ids)

        return self._config.distance_params_coeff * params_diff

    def prune(self) -> 'Genome':
        """
        Create a pruned copy of this genome by removing dead-end nodes and disabled connections.

        This method creates a deep copy of the current genome and then:
        1. Removes all hidden nodes that cannot reach any output node via enabled connections
        2. Removes all disabled connections

        Returns:
            A new Genome object with dead-end nodes and disabled connections removed
        """
        # Create a deep copy of this genome
        pruned_genome = copy.deepcopy(self)

        # Remove all dead-end hidden nodes
        dead_end_nodes = pruned_genome._get_dead_end_nodes()
        for node_id in dead_end_nodes:
            pruned_genome._delete_node(node_id)

        # Remove all disabled connections
        disabled_innovations = [i for i, conn in pruned_genome.conn_genes.items() if not conn.enabled]
        for innov in disabled_innovations:
            pruned_genome._delete_connection(innov)

        return pruned_genome

    def crossover(self, other: 'Genome', fitter_parent: 'Genome') -> 'Genome':
        """
        Perform NEAT crossover between this genome and another to create offspring.

        NEAT crossover rules:
        - Matching genes: randomly inherit from either parent
        - Disjoint/excess genes: inherit from fitter parent only

        Parameters:
            other:         the other parent genome
            fitter_parent: which parent is fitter (must be 'self' or 'other')

        Returns:
            New offspring genome
        """

        # Create empty (no node or connection genes) offspring genome
        offspring = Genome.__new__(Genome)
        offspring._config    = self._config
        offspring.node_genes = {}
        offspring.conn_genes = {}

        # Start by deciding which connections are part of the new network.
        # Once this is decided, the ends of these connections give us the
        # set of nodes which are part of the new network.

        # ----------------

        # Get connection innovation numbers from both parents
        innovs_self  = set(self.conn_genes.keys())
        innovs_other = set(other.conn_genes.keys())

        # Categorize connection genes
        matching_innovs   = innovs_self  & innovs_other   # conn genes shared by both genomes
        only_self_innovs  = innovs_self  - innovs_other   # conn genes present only in 'self'
        only_other_innovs = innovs_other - innovs_self    # conn genes present only in 'other'

        # Matching connections: inherit connection gene randomly from either parent
        for innov in matching_innovs:
            conn_gene = (self.conn_genes if random.random() < 0.5 else other.conn_genes)[innov]
            conn_gene = copy.copy(conn_gene)

            # Handle 'enabled' status:
            # - if parents disagree on enabled status, 75% chance of being enabled
            # - if parents agree, inherit that status
            conn_self  = self.conn_genes [innov]
            conn_other = other.conn_genes[innov]
            if conn_self.enabled != conn_other.enabled:
                conn_gene.enabled = random.random() < 0.75

            offspring.conn_genes[innov] = conn_gene

        # Disjoint & excess connections: inherit connection genes from the fitter parent
        extra_innovs = only_self_innovs if fitter_parent is self else only_other_innovs
        for innov in extra_innovs:
            offspring.conn_genes[innov] = copy.copy(fitter_parent.conn_genes[innov])

        # ----------------

        # Collect the IDs of all nodes needed by the offspring's connections.
        node_ids = set()
        for conn_gene in offspring.conn_genes.values():
            node_ids.add(conn_gene.node_in)
            node_ids.add(conn_gene.node_out)

        # Ensure all input and output nodes are included (even if they have no connections)
        # Here we use again the convention according to which:
        #  + input nodes are numbered: [0, NUMBER INPUT NODES - 1)
        #  + output nodes are numbered: [NUMBER INPUT NODES, NUMBER INPUT NODES + NUMBER OUTPUT NODES - 1)
        for i in range(self._config.num_inputs + self._config.num_outputs):
            node_ids.add(i)

        # Inherit node genes:
        # - matching nodes:     inherit randomly from either parent
        # - non-matching nodes: inherit from whichever parent has it
        for nid in node_ids:
            if nid in self.node_genes and nid in other.node_genes:
                node_gene = self.node_genes[nid] if random.random() < 0.5 else other.node_genes[nid]
            elif nid in self.node_genes:
                node_gene = self.node_genes[nid]
            elif nid in other.node_genes:
                node_gene = other.node_genes[nid]
            else:
                raise RuntimeError(f"node ID {nid} cannot be found in either parent")
            offspring.node_genes[nid] = copy.deepcopy(node_gene)

        # ----------------

        return offspring

    def mutate(self) -> None:
        """
        Apply to the current genome all possible mutation operations.

        The list of possible mutations is:
          + add a node
          + delete a node
          + add a connection
          + delete a connection
          + enable  existing connection
          + disable existing connection
          + mutate node parameters
          + mutate connection parameters
        Each mutation occurs randomly with a given probability.
        """

        # Apply structural mutations (mutations that change the network graph)
        # Case #1: only one structural mutation is allowed at a time
        if self._config.single_structural_mutation:
            normalizer = self._config.node_add_probability          + \
                         self._config.node_delete_probability       + \
                         self._config.connection_add_probability    + \
                         self._config.connection_delete_probability + \
                         self._config.connection_enable_probability + \
                         self._config.connection_disable_probability
            
            if normalizer > 0:
                r = random.random()
                if r < (self._config.node_add_probability / normalizer):
                    self._mutate_add_node()

                elif r < (self._config.node_add_probability +
                          self._config.node_delete_probability) / normalizer:
                    self._mutate_delete_node()

                elif r < (self._config.node_add_probability       +
                          self._config.node_delete_probability    +
                          self._config.connection_add_probability) / normalizer:
                    self._mutate_add_connection()

                elif r < (self._config.node_add_probability          +
                          self._config.node_delete_probability       +
                          self._config.connection_add_probability    +
                          self._config.connection_delete_probability) / normalizer:
                    self._mutate_delete_connection()

                elif r < (self._config.node_add_probability          +
                          self._config.node_delete_probability       +
                          self._config.connection_add_probability    +
                          self._config.connection_delete_probability +
                          self._config.connection_enable_probability) / normalizer:
                    self._mutate_enable_connection()

                elif r < (self._config.node_add_probability             +
                          self._config.node_delete_probability          +
                          self._config.connection_add_probability       +
                          self._config.connection_delete_probability    +
                          self._config.connection_enable_probability    +
                          self._config.connection_disable_probability) / normalizer:
                    self._mutate_disable_connection()

        # Case #2: multiple structural mutations are allowed at a time
        else:
            do_add_node           = random.random() < self._config.node_add_probability
            do_delete_node        = random.random() < self._config.node_delete_probability
            do_add_connection     = random.random() < self._config.connection_add_probability
            do_delete_connection  = random.random() < self._config.connection_delete_probability
            do_enable_connection  = random.random() < self._config.connection_enable_probability
            do_disable_connection = random.random() < self._config.connection_disable_probability

            if do_add_node:
                self._mutate_add_node()
            if do_delete_node:
                self._mutate_delete_node()
            if do_add_connection:
                self._mutate_add_connection()
            if do_delete_connection:
                self._mutate_delete_connection()
            if do_enable_connection:
                self._mutate_enable_connection()
            if do_disable_connection:
                self._mutate_disable_connection()

        # Mutate connection parameters
        for conn in self.conn_genes.values():
            conn.mutate()

        # Mutate node parameters
        for node in self.node_genes.values():
            if node.type != NodeType.INPUT: # input nodes are not mutated, they always pass in the inputs unchanged
                node.mutate()

    def _mutate_add_node(self) -> None:
        """
        Split an existing connection by adding a new node.
        The connection to split is selected at random from all 'enabled' connections.
        Since we are at the genotype level, 'splitting a connection' means modifying
        the genome such that it describes the new network structure.
        """
        # A newly initialized genome has no connection genes.
        if not self.conn_genes:
            return

        # Pick a random connection gene (for an enabled connection)
        enabled_conn_genes = [gene for gene in self.conn_genes.values() if gene.enabled]
        if not enabled_conn_genes:
            return
        split_conn_gene = random.choice(enabled_conn_genes)

        # The connection being split must be disabled.
        split_conn_gene.enabled = False

        # From the global registry, get the ID for the new node and the
        # innovation numbers (connection IDs) for the two new connections
        new_node_id, innov1, innov2 = InnovationTracker.get_split_IDs(split_conn_gene)

        # There are two possibilities:
        # - the selected connection has been split before and this genome already 
        #   contains the genes for the new node and and for the two new connections 
        # - this genome does not contain the genes for the new node and the two new
        #   connections (either the connection has never been split before, in any
        #   genome, or it has been split, however the new structure has not been 
        #   inherited by this genome)
        # NOTE: We currently don't allow for a connection for be split multiple 
        #       times (generating more than two paths connecting its endpoints).
        has_new_genes = new_node_id in self.node_genes and \
                        innov1      in self.conn_genes and \
                        innov2      in self.conn_genes

        lacks_new_genes = new_node_id not in self.node_genes and \
                          innov1      not in self.conn_genes and \
                          innov2      not in self.conn_genes

        assert (has_new_genes or lacks_new_genes)

        # Add new genes to this genome.
        if lacks_new_genes:

            # Create the gene describing the new node (it is a hidden node)
            new_node = NodeGene(new_node_id, NodeType.HIDDEN, self._config)
            self.node_genes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN, self._config)

            # Create the gene describing the first new connection: input -> new node (weight = 1.0)
            conn1 = ConnectionGene(split_conn_gene.node_in, new_node_id, 1.0, innov1, self._config)
            self.conn_genes[innov1] = conn1

            # Create the gene describing the second new connection: new node -> output (weight = old weight)
            conn2 = ConnectionGene(new_node_id, split_conn_gene.node_out, split_conn_gene.weight, innov2, self._config)
            self.conn_genes[innov2] = conn2

        # The genes resulting from the split are already part of the genome; since we
        # disabled the split gene, the resulting two new connections should be enabled
        else:
            self.conn_genes[innov1].enabled = True
            self.conn_genes[innov2].enabled = True

    def _mutate_delete_node(self) -> None:
        """
        Randomly delete a hidden node and all its connections.
        """
        # Get all hidden nodes
        hidden_nodes = self.hidden_nodes

        # Pick a random hidden node and delete it
        if hidden_nodes:
            node_to_delete = random.choice(hidden_nodes)
            self._delete_node(node_to_delete.id)

    def _mutate_add_connection(self):
        """
        Add a new connection between two existing nodes.

        Since we are at the genotype level, 'adding a connection' means 
        modifying the genome to describe the new network structure.

        The nodes representing the two ends of the new connection
        are selected at random, however we cannot add a connection:
         + starting at an OUTPUT node
         + ending   at an INPUT  node
         + between two nodes already connected by a direct connection
         + which would create a cycle in the DAG network graph
        
        Note that the method does NOT add a new connection if it fails to do
        so due to the constraints listed above more a maximum number of times.                
        """

        # Get the node pairs already connected by a direct connection.       
        connected_nodes = {(conn.node_in, conn.node_out) for conn in self.conn_genes.values()}

        # To prevent an infinite loop, this method only attempts 
        # to create a new connection a maximum number of times.
        NUM_ATTEMPTS = 20
        for _ in range(NUM_ATTEMPTS):

            # Select at random the two ends of the new connection
            node_IDs = list(self.node_genes.keys())
            node_in  = random.choice(node_IDs)
            node_out = random.choice(node_IDs)
            
            # Carry out quick checks first
            if self.node_genes[node_in].type == NodeType.OUTPUT:
                continue
            if self.node_genes[node_out].type == NodeType.INPUT:
                continue
            if (node_in, node_out) in connected_nodes:
                continue
                
            # Carry out expensive check last
            if self._would_create_cycle(node_in, node_out):
                continue
            
            # Success - add connection gene to the genome and return
            innovation_num = InnovationTracker.get_innovation_number(node_in, node_out)
            weight         = random.uniform(self._config.min_weight, self._config.max_weight)
            new_connection = ConnectionGene(node_in, node_out, weight, innovation_num, self._config)     
            self.conn_genes[innovation_num] = new_connection
            break 
    
    def _mutate_enable_connection(self) -> None:
        """
        Randomly enable a currently disabled connection.
        """
        # Get all disabled connections
        disabled_conns = [c for c in self.conn_genes.values() if not c.enabled]
        
        # Pick a random disabled connection and enable it
        if disabled_conns:            
            connection_to_enable = random.choice(disabled_conns)
            connection_to_enable.enabled = True
    
    def _mutate_disable_connection(self) -> None:
        """
        Randomly disable a currently enabled connection.
        """
        # Get all enabled connections
        enabled_conns = [c for c in self.conn_genes.values() if c.enabled]

        # Pick a random enabled connection and disable it
        if enabled_conns:
            connection_to_disable = random.choice(enabled_conns)
            connection_to_disable.enabled = False

    def _mutate_delete_connection(self) -> None:
        """
        Randomly delete a connection (either enabled or disabled).
        """
        # Get all connections
        conns = self.conn_genes

        # Pick a random connection and delete it
        if conns:
            connection_to_delete = random.choice(list(self.conn_genes.values()))
            self._delete_connection(connection_to_delete.innovation)

    def _delete_node(self, node_id: int) -> None:
        """
        Delete a node from the genome and remove all connections starting or ending at this node.

        Only hidden nodes can be deleted. 
        Attempting to delete an input or output node will raise a ValueError.

        Parameters:
            node_id: ID of the node to delete

        Raises:
            ValueError: If the node is not a hidden node
            KeyError:   If the node ID does not exist in the genome
        """
        # Check if the node exists
        if node_id not in self.node_genes:
            raise KeyError(f"Node with ID {node_id} does not exist in the genome")
        node = self.node_genes[node_id]

        # Check if the node is a hidden node
        if node.type != NodeType.HIDDEN:
            raise ValueError(f"Cannot delete node {node_id}: only hidden nodes can be deleted (node type is {node.type.name})")

        # Remove all connections that involve this node
        connections_to_remove = [innov for innov, conn in self.conn_genes.items()
                                 if conn.node_in == node_id or conn.node_out == node_id]
        for innov in connections_to_remove:
            self._delete_connection(innov)

        # Remove the node itself
        del self.node_genes[node_id]

    def _delete_connection(self, innovation_number: int) -> None:
        """
        Delete a connection from the genome.

        Parameters:
            innovation_number: Innovation number of the connection to delete

        Raises:
            KeyError: If the innovation number does not exist in the genome
        """
        if innovation_number not in self.conn_genes:
            raise KeyError(f"Connection with innovation number {innovation_number} does not exist in the genome")

        del self.conn_genes[innovation_number]

    def _get_dead_end_nodes(self) -> list[int]:
        """
        Find all hidden nodes from which no output node can be reached via enabled connections.

        A hidden node is considered a "dead-end" if there is no path from that node to any
        output node using only enabled connections. Such nodes do not contribute to the
        network's output and could potentially be pruned.

        This method performs a single backward search starting from all output nodes to find
        all nodes that can reach an output. Hidden nodes not in this set are dead-ends.

        Returns:
            List of node IDs for hidden nodes that cannot reach any output node
        """
        # Build reverse adjacency list from enabled connections only
        # reverse_adjacency[node_out] = list of nodes that have enabled connections TO node_out
        reverse_adjacency = {}
        for conn in self.conn_genes.values():
            if conn.enabled:
                if conn.node_out not in reverse_adjacency:
                    reverse_adjacency[conn.node_out] = []
                reverse_adjacency[conn.node_out].append(conn.node_in)

        # Perform backward DFS from all output nodes to find all reachable nodes
        reachable = set()
        stack = [node.id for node in self.output_nodes]

        while stack:
            current = stack.pop()

            if current in reachable:
                continue
            reachable.add(current)

            # Add all nodes that can reach 'current' (traverse backwards)
            if current in reverse_adjacency:
                for predecessor in reverse_adjacency[current]:
                    if predecessor not in reachable:
                        stack.append(predecessor)

        # Find hidden nodes that are NOT reachable from any output
        dead_end_nodes = [node.id for node in self.hidden_nodes if node.id not in reachable]

        return dead_end_nodes

    def _would_create_cycle(self, from_node: int, to_node: int) -> bool:
        """
        Check if adding a connection from_node -> to_node would create a cycle.
        Uses DFS to check if there's already a path from 'to_node' back to 'from_node'.
        Considers ALL connections (both enabled and disabled) to maintain DAG structure.

        Parameters:
            from_node: proposed start of the new connection
            to_node:   proposed end   of the new connections

        Returns:
            whether adding the new connection would create a cycle in the network
        """
        # Avoid trivial connections.
        if from_node == to_node:
            return True

        # If we can reach 'from_node' starting at 'to_node', then adding a 
        # connection 'from_node' -> 'to_node' would create a network cycle
        visited = set()
        stack = [to_node]
        
        while stack:

            current = stack.pop()            
            if current == from_node:
                return True   # found path 'to_node' -> 'from_node', would create cycle            
            if current in visited:
                continue                
            visited.add(current)
            
            # Add all nodes that can be reached from 'current' in
            # one step (via both enabled and disabled connections)
            for conn_gene in self.conn_genes.values():
                if conn_gene.node_in == current:     # connection starting in 'current'
                    stack.append(conn_gene.node_out)   
        
        return False 

    def __str__(self):
        node_genes_str  = ''.join(str(node) for node in self.input_nodes)
        node_genes_str += ''.join(str(node) for node in self.hidden_nodes)
        node_genes_str += ''.join(str(node) for node in self.output_nodes)
        conn_genes_str  = ''.join(str(conn) for conn in self.conn_genes.values())
        return f"Nodes: {node_genes_str}\nConns: {conn_genes_str}"

    @staticmethod
    def show_aligned(genome1: 'Genome', genome2: 'Genome') -> None:
        """
        Print two genomes aligning the node and connection genes.
        """

        # Align nodes by ID
        node_ids_all = sorted(set(genome1.node_genes.keys()) | set(genome2.node_genes.keys()))
        node_str1 = ""
        node_str2 = ""        
        padding   = ' ' * 13
        for node_id in node_ids_all:
            node_str1 += str(genome1.node_genes[node_id]) if node_id in genome1.node_genes else padding
            node_str2 += str(genome2.node_genes[node_id]) if node_id in genome2.node_genes else padding

        # Print aligned nodes
        print(f"Nodes:\n{node_str1}\n{node_str2}\n")
        
        # Align connections by innovation number
        innovs_all = sorted(set(genome1.conn_genes.keys()) | set(genome2.conn_genes.keys()))
        conn_str1 = ""
        conn_str2 = ""
        padding   = ' ' * 18
        for inov in innovs_all:
            conn_str1 += str(genome1.conn_genes[inov]) if inov in genome1.conn_genes else padding
            conn_str2 += str(genome2.conn_genes[inov]) if inov in genome2.conn_genes else padding
        
        # Print aligned connections
        print(f"Connections:\n{conn_str1}\n{conn_str2}\n")


