"""
NEAT Species Manager Module

This module implements the SpeciesManager class for the NEAT algorithm. 
The manager coordinates the speciation process and manages the lifecycle 
of all species across generations.

Speciation in NEAT:
In traditional genetic algorithms, new structural innovations often have lower
initial fitness and are quickly eliminated. NEAT addresses this by organizing
the population into species - groups of genetically similar individuals that
compete primarily within their own niche. This allows novel structures time to
optimize before facing global competition.

Key Concepts:
- Species: A cluster of genetically similar individuals
- Representative: An individual used to define species membership
- Compatibility Threshold: Maximum genetic distance for same-species membership
- Explicit Fitness Sharing: Offspring allocation proportional to species fitness
- Stagnation: Species removed if they fail to improve over many generations

How Speciation Works:
1. Calculate genetic distance between individuals using NEAT distance formula
2. Assign individuals to species whose representative is closest (if within threshold)
3. Create new species for individuals that don't fit existing species
4. Each species spawns offspring proportional to its average fitness
5. Elite individuals from each species are preserved unchanged
6. Stagnant species are removed to free resources for innovation

The Speciation Process:
- Representatives are updated each generation from current population
- Species track fitness history and age
- Protected elite species cannot be removed even if stagnant
- Minimum species size ensures diversity preservation
- Distance cache optimizes expensive genetic distance calculations

Classes:
    SpeciesManager: Manages all species, handles speciation and reproduction coordination
"""

import math
from itertools  import count
from statistics import mean
from typing     import TYPE_CHECKING

if TYPE_CHECKING:
    from neat.phenotype import Individual
    from neat.pool.population import Population
from neat.run.config   import Config
from neat.pool.species import Species

class SpeciesManager:
    """
    Manages the collection of species and speciation process across generations.

    The SpeciesManager is responsible for organizing the entire population into
    species based on genetic similarity, tracking species across generations, and
    coordinating reproduction. It implements the core NEAT speciation mechanism that
    protects innovation by allowing individuals to compete primarily within their
    own species rather than across the entire population.

    The manager maintains mappings between individuals and species, handles species
    lifecycle (creation, updating, extinction), and implements policies for stagnation
    removal and offspring allocation.

    Public Attributes:
        species:               Dictionary mapping species IDs to Species instances
        individual_to_species: Dictionary mapping individual IDs to their Species

    Public Methods:
        speciate(population):                  Assign all individuals to species
        update_fitness():                      Calculate and update all species fitnesses
        remove_stagnating_species(population): Remove species that haven't improved
        calculate_offspring_allocations():     Determine offspring count per species
    """

    class DistanceCache:
        """
        Caches the genomic distance between individuals.
        """
        def __init__(self):
            self.distances = {}

        def __call__(self, individual1, individual2):
            id1  = individual1.ID
            id2  = individual2.ID
            dist = self.distances.get((id1, id2))
            if dist is None:
                dist = individual1.distance(individual2)
                self.distances[(id1, id2)] = dist
                self.distances[(id2, id1)] = dist
            return dist

    def __init__(self, config: Config):
        """
        Initialize the Species Manager.

        Parameters:
            condif: Stores configuration parameters.
        """
        self.species               = {}        # species ID    => Species instance
        self.individual_to_species = {}        # individual ID => Species instance
        self._id_generator         = count(1)  # generates species IDs
        self._config               = config    # stores config parameters

    def speciate(self, population: 'Population'):
        """
        Assign all individuals in a population to species based on genetic similarity.

        Speciation takes place each time a new generation of individuals has spawn.

        This is the core speciation algorithm in NEAT that partitions the population
        into distinct species. Each individual is assigned to the species whose
        representative is genetically closest, provided the distance is below the
        compatibility threshold. Individuals that don't fit into any existing species
        form new species.

        The algorithm proceeds in three phases:

        Phase 1: Update Representatives
        - For each existing species, find the individual in the new generation closest 
          to its current representative
        - This individual becomes the new representative for that species
        - Representatives are chosen from the current generation, not carried over

        Phase 2: Assign Individuals to Species
        - For each unassigned individual, calculate distance to all species representatives
        - If distance to any representative is below compatibility threshold, assign to closest
        - If no compatible species exists, create a new species with this individual as representative

        Phase 3: Cleanup
        - Update species member lists with assigned individuals
        - Update individual-to-species mapping for quick lookups
        - Remove species that have no members (extinction)

        Postconditions:
            - Every individual in the population is assigned to exactly one species
            - Each species has at least one member
            - Species representatives are from the current population

        Parameters:
            population: The Population object containing all individuals to speciate
        """
        dist_cache = SpeciesManager.DistanceCache()

        # Find the new representative for each species by calculating the
        # distance between each individual from the new population and
        # each existing species. The new representative is the individual
        # closest to the species (closest to the current species rep).
        # When speciating for the first time, there are no species yet,
        # and this step is a no-op.

        unspeciated = set(population.individuals)
        new_reps   : dict[int, Individual]       = {}  # species ID => representative individual
        new_members: dict[int, list["Individual"]] = {}  # species ID => all species members
        for spec_id, spec in self.species.items():     # this is empty when speciating for the first time
            candidates = []
            for individual in unspeciated:
                distance = dist_cache(individual, spec.representative)
                candidates.append((distance, individual))
            if not candidates:
                continue

            # The new representative is the individual closest to the current representative.
            _, new_rep = min(candidates, key=lambda x: x[0])
            new_reps   [spec_id] =  new_rep
            new_members[spec_id] = [new_rep]
            unspeciated.remove(new_rep)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            individual = unspeciated.pop()

            # For each individual, find the closest species based
            # on its distance to the species *new* representative
            candidate_species = []
            for spec_id, new_rep in new_reps.items():
                distance = dist_cache(individual, new_rep)
                if distance < self._config.compatibility_threshold:
                    candidate_species.append((distance, spec_id))

            # We found at least one species to which this individual
            # could be allocated - allocate it to the closest one
            if candidate_species:
                _, spec_id = min(candidate_species, key=lambda x: x[0])
                new_members[spec_id].append(individual)

            # No species is similar enough, assign to a new species,
            # with this individual as its species representative.
            else:
                spec_id = next(self._id_generator)
                new_reps   [spec_id] =  individual
                new_members[spec_id] = [individual]

        # Update all species.
        self.individual_to_species = {}       # reset
        for spec_id, rep in new_reps.items():

            if spec_id not in self.species:
                self.species[spec_id] = Species(spec_id, rep, self._config)
            else:
                self.species[spec_id].init_for_next_generation(rep)
            spec = self.species[spec_id]

            members = new_members[spec_id]
            for individual in members:
                self.individual_to_species[individual.ID] = spec
                spec.members[individual.ID] = individual

        # Remove extinct species
        extinct_species = set(self.species.keys()) - set(new_reps.keys())
        for spec_id in extinct_species:
            del self.species[spec_id]

        # Error check: all individuals in the population must have been allocated to a species
        assigned_count = sum(len(spec.members) for spec in self.species.values())
        assert assigned_count == len(population.individuals), "Lost individuals during speciation!"

    def update_fitness(self):
        """
        Calculates and updates the fitness for all species.
        The fitness of a species is the average fitness of this members.
        This method assumes that the fitness of all Individuals has already
        been calculated (otherwise it will raise a TypeError).
        """
        for spec in self.species.values():
            fitness = mean([individual.fitness for individual in spec.members.values()])
            spec.update_fitness(fitness)

    def remove_stagnating_species(self, population: 'Population') -> tuple[set[int], set[int]]:
        """
        Identify and remove species that have been stagnating for too long.

        A species is stagnant if it hasn't improved its max fitness for a
        given number of generations. However, a species is protected from
        being marked as stagnant if:
        - It contains the fittest individual in the entire population, OR
        - It is in the top 'self._config.species_elitism' species ranked by fitness

        Population:
            population: The population whose species we are processing

        Returns:
            Tuple of (stagnant_species_ids, removed_individual_ids)
        """

        # Find the fittest individual in the entire population and its species
        fittest_individual = population.get_fittest_individual()
        if fittest_individual is None:
            return set(), set()
        species_fittest_indiv = self.individual_to_species[fittest_individual.ID]

        # Rank all species by fitness
        sorted_species = sorted(self.species.values(), key=lambda s: s.fitness, reverse=True)

        # Protect from elimination the top N species,
        # plus the species with the fittest individual
        protected_spec_ids = {s.id for s in sorted_species[:self._config.species_elitism]}
        protected_spec_ids.add(species_fittest_indiv.id)

        # Identify stagnating species
        stagnating_spec_ids = set()
        for spec in self.species.values():
            if spec.id not in protected_spec_ids and spec.is_stagnant():
                stagnating_spec_ids.add(spec.id)

        # Identify all individuals that belong to stagnating species
        stagnating_indiv_ids = set()
        for spec_id in stagnating_spec_ids:
            spec = self.species[spec_id]
            stagnating_indiv_ids.update(spec.members.keys())

        # Clean-up: remove stagnating species and individuals
        for spec_id in stagnating_spec_ids:
            del self.species[spec_id]
        for indiv_id in stagnating_indiv_ids:
            del self.individual_to_species[indiv_id]
        population.individuals = [ind for ind in population.individuals
                                  if ind.ID not in stagnating_indiv_ids]

        return stagnating_spec_ids, stagnating_indiv_ids

    def calculate_offspring_allocations(self) -> dict[int, int]:
        """
        Calculate how many offspring each species should produce.

        Allocates offspring proportionally to species fitness, with the constraint
        that each species gets at least 'self._config.min_species_size' offspring.
    
        Returns:
            Dictionary mapping species_id to number of offspring to produce
        """
        # Constraint on how small a species size can get.
        min_species_sz  = max(self._config.min_species_size, self._config.elitism)

        # Calculate total fitness across all species
        # Treat NaN fitness as 0 (invalid networks get worst fitness)
        total_fitness = sum(0.0 if math.isnan(spec.fitness) else spec.fitness
                            for spec in self.species.values())

        # Allocate offspring proportionally to species fitness relative to total fitness
        allocations     = {}   # species ID => number of offspring
        total_allocated = 0
        for spec_id, spec in self.species.items():

            # Get fitness, treating NaN as 0
            spec_fitness = 0.0 if math.isnan(spec.fitness) else spec.fitness

            # Proportional allocation based on fitness if total fitness > 0
            # Equal allocation when all species have zero fitness (fitness cannot be negative)
            if total_fitness == 0:
                num_offspring = int(self._config.population_size / len(self.species))
            else:
                num_offspring = int(spec_fitness / total_fitness * self._config.population_size)

            # Enforce minimum species size
            num_offspring = max(num_offspring, min_species_sz)

            allocations[spec_id] = num_offspring
            total_allocated     += num_offspring

        # Adjust if we over/under-allocated due to rounding and minimums.
        # Attempt to adjust the species with highest fitness - it's OK if
        # we cannot adjust the population size perfectly.
        difference = self._config.population_size - total_allocated
        if difference != 0:
            sorted_species = sorted(self.species.values(),
                                    key=lambda s: 0.0 if math.isnan(s.fitness) else s.fitness,
                                    reverse=True)
            if sorted_species:
                spec_id = sorted_species[0].id
                allocations[spec_id] += difference
                allocations[spec_id]  = max(allocations[spec_id], min_species_sz)

        return allocations
