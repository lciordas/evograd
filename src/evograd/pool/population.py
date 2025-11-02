"""
NEAT Population Module

This module implements the Population class, the top-level orchestrator for the NEAT
evolutionary algorithm. The population manages the complete lifecycle of evolution, 
from initialization through convergence.

Classes:
    Population: Top-level evolutionary coordinator managing individuals and generations
"""

import autograd.numpy as np  # type: ignore
import random
from typing import TYPE_CHECKING

from evograd.run.config import Config

if TYPE_CHECKING:
    from evograd.genotype import Genome, ConnectionGene, InnovationTracker
    from evograd.phenotype import Individual

from evograd.pool.species_manager import SpeciesManager

class Population:
    """
    A population of evolving individuals in the NEAT algorithm.

    The Population class represents the top-level container for the evolutionary
    process, managing a collection of individuals and coordinating their evolution
    through generations. It handles initialization, fitness evaluation, reproduction,
    and maintains the speciation structure that protects innovation.

    Public Attributes:
        individuals: List of all Individual objects in the current generation

    Public Methods:
        get_fittest_individual(): Return the individual with highest fitness
        spawn_next_generation():  Create the next generation through evolution
    """

    def __init__(self, config: Config, network_type: str):
        """
        Initialize the population with a given number of Individuals and split them into species.

        Parameters:
            config: Stores configuration parameters
            network_type: Type of network backend to use ('standard', 'fast', 'autograd')
        """
        # Import here to avoid circular import
        from evograd.genotype import Genome
        from evograd.phenotype import Individual

        self._config = config
        self._network_type = network_type

        # Step 1: create a number of identical individuals, each with a
        # network consisting only of unconnected input and output nodes.
        self.individuals = []
        for _ in range(self._config.population_size):
            genome     = Genome(self._config)
            individual = Individual(genome, network_type)
            self.individuals.append(individual)

        # Step 2: add connections to the neural networks of each Individual.
        # The maner in which this is done depends on the initialization policy.
        if self._config.initial_cxn_policy == "none":
            pass  # already unconnected
        elif self._config.initial_cxn_policy == "one-input":
            self._connect_one_input()
        elif self._config.initial_cxn_policy == "partial":
            self._connect_partial()
        elif self._config.initial_cxn_policy == "full":
            self._connect_full()
        else:
            raise RuntimeError("bad initial connection policy")

        # Split the initial population into species
        self._species_manager = SpeciesManager(config)
        self._species_manager.speciate(self)

    def _connect_one_input(self):
        """
        For each network, connect one random input node to all outputs nodes.
        """
        # Import here to avoid circular import
        from evograd.genotype import ConnectionGene, InnovationTracker

        # Add connections to each Individual's network.
        for individual in self.individuals:
            genome = individual.genome

            # Select one random input node
            input_node = random.choice(genome.input_nodes)

            # Connect it to all output nodes
            for output_node in genome.output_nodes:
                innovation = InnovationTracker.get_innovation_number(input_node.id, output_node.id)
                weight     = np.random.normal(self._config.weight_init_mean, self._config.weight_init_stdev)
                weight     = np.minimum(np.maximum(weight, self._config.min_weight), self._config.max_weight)
                connection = ConnectionGene(input_node.id, output_node.id, weight, innovation, self._config)
                genome.conn_genes[innovation] = connection

    def _connect_partial(self):
        """
        For each network, connect a fraction of all possible connections.
        The connections are chosen at random.
        """
        # Import here to avoid circular import
        from evograd.genotype import ConnectionGene, InnovationTracker

        # Add connections to each Individual's network.
        for individual in self.individuals:
            genome = individual.genome

            # Randomly select the input-output node pairs which are to be connected
            all_pairs  = [(inp.id, out.id) for inp in genome.input_nodes for out in genome.output_nodes]
            num_conns  = int(len(all_pairs) * self._config.initial_cxn_fraction)
            make_pairs = random.sample(all_pairs, num_conns)

            # Create the selected connections
            for input_id, output_id in make_pairs:
                innovation = InnovationTracker.get_innovation_number(input_id, output_id)
                weight     = np.random.normal(self._config.weight_init_mean, self._config.weight_init_stdev)
                weight     = np.minimum(np.maximum(weight, self._config.min_weight), self._config.max_weight)
                connection = ConnectionGene(input_id, output_id, weight, innovation, self._config)
                genome.conn_genes[innovation] = connection

    def _connect_full(self):
        """
        For each network, connect all inputs nodes to all output nodes.
        """
        # Import here to avoid circular import
        from evograd.genotype import ConnectionGene, InnovationTracker

        # Add connections to each Individual's network.
        for individual in self.individuals:
            genome = individual.genome

            # Connect all input nodes to all output nodes
            for input_node in genome.input_nodes:
                for output_node in genome.output_nodes:
                    innovation = InnovationTracker.get_innovation_number(input_node.id, output_node.id)
                    weight     = np.random.normal(self._config.weight_init_mean, self._config.weight_init_stdev)
                    weight     = np.minimum(np.maximum(weight, self._config.min_weight), self._config.max_weight)
                    connection = ConnectionGene(input_node.id, output_node.id, weight, innovation, self._config)
                    genome.conn_genes[innovation] = connection

    def get_fittest_individual(self) -> 'Individual | None':
        """
        Find and return the individual with the highest fitness in the population.
        Assumes that the fitness of each Individual has already been calculated
        and saved inside each of them.
        
        Returns:
            The individual with the highest fitness value, or None if population 
            is empty, or the fitness of individuals has not been calculated yet
        """
        if not self.individuals:
            return None

        # 'max' raises a TypeError if called on a list that contains 'None'
        # in our case, this happens if the individual fitness has not been
        # evaluated yet.
        try:
            return max(self.individuals, key=lambda ind: ind.fitness)
        except TypeError:
            return None

    def spawn_next_generation(self):
        """
        Create the next generation through evaluation, selection, and reproduction.

        This is the main generational step in the NEAT algorithm, coordinating all
        aspects of evolution: fitness evaluation, speciation, stagnation removal,
        offspring allocation, and reproduction. The method transforms the current
        population into a new generation while preserving species structure and
        promoting innovation.

        The generation process follows these steps:

        Step 1: Stagnation Removal
        - Identify species that haven't improved for too long
        - Remove stagnant species and all their members
        - Protect elite species and species containing the fittest individual

        Step 2: Offspring Allocation
        - Calculate how many offspring each surviving species should produce
        - Allocate offspring proportional to species fitness (explicit fitness sharing)
        - Enforce minimum species size and population size constraints

        Step 3: Reproduction
        - Each species spawns its allocated number of offspring
        - Elite individuals are preserved unchanged (elitism)
        - Remaining offspring created through crossover and mutation
        - Parent selection from top performers in each species

        Step 4: Speciation
        - Assign all offspring to species based on genetic similarity
        - Update species representatives from new population
        - Create new species for novel genomes
        - Remove extinct species (those with no members)
        """

        # Remove stagnating species and all individuals that belong to them
        _, _ = self._species_manager.remove_stagnating_species(self)

        # Calculate how many offspring are contributed by each surviving species
        allocations = self._species_manager.calculate_offspring_allocations()

        # Spawn the new generation, one species at a time
        offspring_all = []
        for spec_id, spec in self._species_manager.species.items():
            offspring_number = allocations[spec_id]
            offspring = spec.spawn(offspring_number)
            offspring_all.extend(offspring)
        self.individuals = offspring_all

        # Split the new population into species
        self._species_manager.speciate(self)

    def __str__(self):
        return '\n'.join(str(individual) for individual in self.individuals)
