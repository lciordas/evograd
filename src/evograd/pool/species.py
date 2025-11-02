"""
NEAT Species Module

This module implements the Species class for the NEAT algorithm. 
A species represents a cluster of genetically similar individuals 
that compete primarily within their own niche.

Classes:
    Species: Represents a single species with members and fitness tracking
"""

import autograd.numpy as np  # type: ignore
import random
from copy   import deepcopy
from typing import TYPE_CHECKING

from evograd.run.config import Config
if TYPE_CHECKING:
    from evograd.phenotype import Individual

class Species:
    """
    A species representing a cluster of genetically similar individuals in NEAT.

    In NEAT, the population is divided into species based on genetic similarity,
    allowing different evolutionary niches to develop independently. This protects
    innovative structures from being eliminated by competition with more mature
    solutions, as individuals only compete for resources within their own species.

    Each species maintains a representative individual used for distance calculations
    during speciation. New individuals are assigned to the species whose representative
    is genetically closest (below the compatibility threshold). Species track their
    fitness over time and can be eliminated if they stagnate (fail to improve).

    The species manages reproduction by:
    - Preserving elite individuals through elitism
    - Creating a parent pool from top performers
    - Generating offspring through crossover and mutation
    - Allocating offspring proportional to species fitness

    Public Attributes:
        id:              Unique species identifier
        representative:  Individual used for distance calculations during speciation
        members:         The individuals that are part of this species
        age:             Number of generations this species has existed
        last_improved:   Last generation when max fitness improved
        fitness:         Average fitness of all members (None until calculated)
        max_fitness:     Best fitness ever achieved by this species
        fitness_history: List of average fitness values over generations

    Public Methods:
        init_for_next_generation(representative): Prepare species for a new generation
        update_fitness(fitness):                  Update species fitness and history
        is_stagnant():                            Check if species has stopped improving
        distance_to(individual):                  Calculate genetic distance to an individual
        spawn(num_offspring):                     Generate the next generation

    Life Cycle:
    1. Created when an individual doesn't fit into existing species
    2. Accumulates members during speciation based on genetic similarity
    3. Fitness is calculated as average of member fitnesses
    4. Spawns offspring proportional to relative fitness
    5. Representative is updated each generation
    6. Removed if stagnant or all members die out
    """

    def __init__(self, species_id: int, representative: 'Individual', config: Config):
        """
        Initialize a new species.

        Parameters:
            species_id:     unique species identifier
            representative: the Individual that represents this species in the speciation process
            config:         stores configuration parameters
        """
        self._config: Config = config

        # Unique species identifier
        self.id: int = species_id

        # Representative individual for distance calculations during speciation
        self.representative: 'Individual' = representative

        # All individuals in this species: individual ID => Individual
        # For now we only have one member: the representative.
        self.members: dict[int, 'Individual'] = {}
        self.members[representative.ID] = representative  

        self.age: int           = 0                 # How many generations this species has existed
        self.last_improved: int = 0                 # Last generation when the fitness improved

        self.fitness        : float | None = None     # Specie fitness (average fitness of all members)
        self.max_fitness    : float        = -np.inf  # Best fitness ever achieved by this species
        self.fitness_history: list[float]  = []       # Track fitness over generations

    def init_for_next_generation(self, representative: 'Individual') -> None:
        """
        Resets the species in preparation of being assigned the individuals in a new generatio.
        Parameters:
            representative: the new Individual that represents this species in the speciation process
        """
        self.representative =  representative
        self.members        = {representative.ID: representative}
        self.age           += 1
        self.fitness        = None

    def update_fitness(self, fitness) -> None:
        """
        Update the species fitness with a new given value.
        Treats NaN fitness as 0.0 (worst possible fitness for invalid networks).
        """
        import math
        # Treat NaN as 0 (invalid networks get worst fitness)
        if math.isnan(fitness):
            fitness = 0.0

        self.fitness = fitness
        if fitness > self.max_fitness:
            self.max_fitness   = fitness
            self.last_improved = self.age
        self.fitness_history.append(fitness)

    def is_stagnant(self) -> bool:
        """
        Check whether the species is stagnant.
        A species is stagnant if its fitness has not improved in a given number of generations.
        """
        return self.age - self.last_improved > self._config.max_stagnation_period

    def distance_to(self, individual: 'Individual') -> float:
        """
        Calculate the genetic distance between this species and a given individual.
        Uses the species representative individual for comparison.

        Parameters:
            individual: The individual whose distance to this species we want to calculate

        Returns:
            The genetic distance between the individual and the species representative
        """
        return self.representative.distance(individual)

    def spawn(self, num_offspring: int) -> list['Individual']:
        """
        Generate offspring for the next generation through elitism and reproduction.

        This method implements the reproduction process within a species, combining
        elitism (carrying over top performers) with sexual reproduction (crossover
        and mutation). The process ensures genetic diversity while preserving the
        best solutions found so far.

        The spawning process:
        1. Sort all members by fitness (highest first)
        2. Transfer elite individuals unchanged to preserve best solutions
        3. Create parent pool from top performers based on survival threshold
        4. Fill remaining offspring slots through crossover and mutation

        As a precondition for running this method, all member individuals must have their
        fitness already evaluated (fitness != None), so they can be sorted when evaluated
        for elitism and/or parenting.

        Configuration parameters used:
            - elitism:            Number of top individuals to carry over unchanged
            - survival_threshold: Fraction of species that can reproduce (0.0-1.0)

        Parameters:
            num_offspring: Number of individuals this species should produce

        Returns:
            List of offspring individuals for the next generation
        """

        # Trivial case
        if num_offspring == 0 or len(self.members) == 0:
            return []

        # Sort all member by fitness.
        # It is assumed that individual fitness has already been calculated
        # Use ID as tie-breaker to ensure deterministic ordering when fitnesses are equal
        sorted_members = sorted(self.members.values(), key=lambda ind: (ind.fitness, -ind.ID), reverse=True)

        # Apply elitism: the top individuals from the species
        # are transferred to the next generation unchanged.
        elite_number = min(self._config.elitism, num_offspring)
        offspring    = [deepcopy(individual) for individual in sorted_members[:elite_number]]

        # Select the parent pool - this is the top fraction of individuals in the species
        num_parents = max(2, int(len(sorted_members) * self._config.survival_threshold))
        num_parents = min(num_parents, len(sorted_members))  # can't exceed actual size
        parent_pool = sorted_members[:num_parents]

        # Spwan new offspring, until we get the requested number
        while len(offspring) < num_offspring:

            # Randomly select parents.
            # Note that if the parents are not distinct, crossover will produce a
            # genetically identical clone of the parent (but with a different ID).
            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)

            child = parent1.mate(parent2)
            offspring.append(child)

        return offspring
