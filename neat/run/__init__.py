"""
NEAT Run Package

This package implements trial and experiment execution for the NEAT (NeuroEvolution of
Augmenting Topologies) algorithm.

A trial represents a complete evolutionary run, managing the population through
generations until a solution is found or maximum generations are reached.

An experiment represents a collection of multiple trials for gathering statistical data.

Modules:
    config:      Configuration management for NEAT parameters
    trial:       Abstract base class for NEAT trials
    experiment:  Abstract base class for NEAT experiments

Exported Classes:
    Config:      Configuration parameters for NEAT algorithm
    Trial:       Abstract base class for NEAT trials with joblib parallelization
    Experiment:  Abstract base class for NEAT experiments (multi-trial runs)
"""

from neat.run.config     import Config
from neat.run.trial      import Trial
from neat.run.experiment import Experiment

__all__ = ['Config','Trial','Experiment']
