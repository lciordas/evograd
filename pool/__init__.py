"""
NEAT Pool Package

This package contains classes for managing populations and species in the NEAT
(NeuroEvolution of Augmenting Topologies) algorithm.

The pool package coordinates the evolutionary process at the population level,
organizing individuals into species based on genetic similarity and managing
reproduction across generations.

Modules:
    species:         Individual species representation and reproduction
    species_manager: Manages all species and the speciation process
    population:      Top-level population management and evolution

Exported Classes:
    Species:        A cluster of genetically similar individuals
    SpeciesManager: Manages species across generations
    Population:     Top-level evolutionary coordinator
"""

from pool.species         import Species
from pool.species_manager import SpeciesManager
from pool.population      import Population

__all__ = [
    'Species',
    'SpeciesManager',
    'Population',
]
