"""
NEAT Individual Module

This module implements the Individual class, representing a complete evolved 
agent in the NEAT (NeuroEvolution of Augmenting Topologies) population.

Classes:
    Individual: A complete evolved agent with genome, network, and fitness
"""

import copy
from itertools import count
from typing    import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neat.genotype import Genome
from neat.phenotype.network_base     import NetworkBase
from neat.phenotype.network_autograd import NetworkAutograd
from neat.phenotype.network_fast     import NetworkFast
from neat.phenotype.network_standard import NetworkStandard

class Individual:
    """
    An individual organism in the NEAT population.

    This class may be absent in other NEAT implementations, which may simplify the
    design by reducing an agent to the neural network that calculates its output. 
    You can regard an individual as a thin wrapper around the network that powers
    it, to which it adds a unique ID and a fitness.

    In our implementation, the NEAT algorithm operates on Individual(s) and not on 
    networks, Individul(a) which are evaluated, compared, and reproduced to create 
    offspring through mutation and crossover.

    The Individual can be powered by different neural network implementations
    (NetworkStandard, NetworkAutograd, or NetworkFast), chosen via the 'network_type' 
    parameter during initialization.

    Public Attributes:
        ID:      Globally unique identifier for this individual
        fitness: Fitness score (None until evaluated)

    Public properties:
        genome: Access to this individual's genome.

    Public Methods:
        clone():         Create a genetic copy of this individual
        distance(other): Calculate genetic distance to another individual
        mate(other):     Reproduce with another individual via crossover and mutation
    """

    _id_generator = count(0)

    def __init__(self, genome: 'Genome', network_type: str):
        """
        Initialize the Individual given its genotype.

        Parameters:
            genome:       The Genome encoding the neural network that powers this Individual
            network_type: Type of network backend to use
                         'standard' - Object-oriented implementation (basic implementation, processes one input at a time)
                         'fast'     - High-performance vectorized numpy implementation (when data to process comes in batches)
                         'autograd' - Autograd-compatible vectorized implementation (when we need gradients)
        """
        self.ID      : int             = next(Individual._id_generator)  # unique ID
        self.fitness : Optional[float] = None                            # fitness used when reproducing
        self._network_type: str        = network_type                    # stored for cloning/mating

        # Create appropriate network type based on parameter
        if network_type == 'standard':
            self._network: NetworkBase = NetworkStandard(genome)
        elif network_type == 'autograd':
            self._network: NetworkBase = NetworkAutograd(genome)
        elif network_type == 'fast':
            self._network: NetworkBase = NetworkFast(genome)
        else:
            raise ValueError(f"Unknown network_type: {network_type}. Use 'standard', 'autograd', or 'fast'.")

    @property
    def genome(self) -> 'Genome':
        """
        Access the genome of this individual.
        """
        return self._network._genome

    def clone(self) -> 'Individual':
        """
        Create a new Individual from the same genome as the current one.
        """
        return Individual(copy.deepcopy(self.genome), self._network_type)

    def distance(self, other: 'Individual') -> float:
        """
        Calculate the genetic distance between this individual and another.

        Parameters:
            other: The other individual to compare against

        Returns:
            The genetic distance between the two individuals' genomes
        """
        return self.genome.distance(other.genome)

    def mate(self, other: 'Individual') -> 'Individual':
        """
        Create a new Individual by mating with another Individual.

        Matting involves the following steps:
         - create a new genome, via crossover between the genomes of the two mating Individuals
         - mutate the new genome
         - create a new Individual based on the new genome, this is the offspring

        Parameters:
            other: the Individual with whom this Individual is mating

        Returns:
            the offspring resulting from the mating process
        """
        fitter_genome = self.genome if self.fitness > other.fitness else other.genome
        genome_child  = self.genome.crossover(other.genome, fitter_genome)
        genome_child.mutate()
        return Individual(genome_child, self._network_type)

    def prune(self) -> 'Individual':
        """
        Create a pruned copy of this individual by removing dead-end nodes and disabled connections.

        This method creates a new Individual with a pruned genome that has:
        1. All dead-end hidden nodes removed (nodes that cannot reach any output via enabled connections)
        2. All disabled connections removed

        The pruned individual will have a simpler, more efficient network while maintaining
        the same functional behavior. The ID and fitness are copied from the original
        individual. The original individual is not modified.

        Returns:
            A new Individual with a pruned genome and network, same ID and fitness
        """
        pruned_genome     = self.genome.prune()
        pruned_individual = Individual(pruned_genome, self._network_type)

        # Copy ID and fitness from the original individual
        pruned_individual.ID      = self.ID
        pruned_individual.fitness = self.fitness

        return pruned_individual

    def __str__(self):
        return f"ID={self.ID}, fitness={self.fitness:.4f}\n{self.genome}"

    def __repr__(self):
        return f"Individual(genome={repr(self.genome)})"
