"""
NEAT Innovation Tracker Module

This module implements the InnovationTracker class for the
NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

Classes:
    InnovationTracker: Global tracker for innovation numbers and node IDs
"""

from itertools import count
from typing    import TYPE_CHECKING

from evograd.run.config import Config
if TYPE_CHECKING:
    from evograd.genotype.connection_gene import ConnectionGene

class InnovationTracker:
    """
    Tracks structural changes globally across all genomes.
    Ensures the same structural change gets the same innovation
    number (for connections) and ID (for nodes).

    Must call 'InnovationTracker.initialize()' before using it.
    """
    
    # Counters (initialized via 'reset()')
    _next_innovation_number = None
    _next_node_id           = None

    # For each connection ever created, map its endpoints to its innovation number
    _innovation_numbers = {}            # (node_in, node_out) -> innovation number
        
    # When a connection is split, tracks what node was created and
    # what innovation numbers were assigned to the new connections.
    _split_IDs = {}        # (split_connection_innov_number,) -> (new_node_id, innov1, innov2)
    
    @classmethod
    def initialize(cls, config: Config):
        """
        Reset the tracker.

        Parameters:
            config: Stores configuration parameters
        """
        cls._next_innovation_number = count(0)
        cls._next_node_id           = count(config.num_inputs + config.num_outputs)
        cls._innovation_numbers     = {}
        cls._split_IDs              = {}

    @classmethod
    def get_innovation_number(cls, node_in: int, node_out: int) -> int:
        """
        Get innovation number for a connection, identified by its endpoints.
        Returns existing innovation number if this connection was created 
        before, otherwise assigns a new innovation number.

        Parameters:
            node_in: node ID for the 'from' end of the connection
            node_in: node ID for the 'to'   end of the connection

        Returns:
            connection ID (a.k.a. innovation number)    
        """
        key = (node_in, node_out)
        
        # This is a new connection
        if key not in cls._innovation_numbers:
            cls._innovation_numbers[key]  = next(cls._next_innovation_number)        

        return cls._innovation_numbers[key]
    
    @classmethod
    def get_split_IDs(cls, conn_to_split: 'ConnectionGene') -> tuple[int, int, int]:
        """
        Get node ID and innovation numbers for splitting a connection.
        If this exact connection has been split before, returns the same 
        values, otherwise creates new ones.

        Parameters
            conn_to_split: the connection being split

        Returns 
            3-tuple: (new_node_id, innovation1, innovation2)
            innovation1 is for the connection from the 'from' node of 'conn_to_split' to the new node
            innovation2 is for the connection from the the new node to the 'to' node of 'conn_to_split' 
        """
        key = (conn_to_split.innovation,)

        # This connection hasn't been split before
        if key not in cls._split_IDs:
        
            # Generate the ID for the new node
            new_node_id = next(cls._next_node_id)
            
            # First new connection: original_in -> new_node
            innov1 = cls.get_innovation_number(conn_to_split.node_in, new_node_id)
            
            # Second new connection: new_node -> original_out
            innov2 = cls.get_innovation_number(new_node_id, conn_to_split.node_out)
            
            cls._split_IDs[key] = (new_node_id, innov1, innov2)
        
        return cls._split_IDs[key]
