"""
NEAT Trial Module

This module defines the abstract base class for NEAT trials with built-in
support for CPU-based parallelization using joblib.

A trial represents one independent run of the NEAT algorithm, evolving a
population through generations until a solution is found or the maximum
number of generations is reached.
"""

from abc        import ABC, abstractmethod
from joblib     import Parallel, delayed
from statistics import mean
from typing     import TYPE_CHECKING

from run.config                  import Config
from genotype.innovation_tracker import InnovationTracker
from pool                        import Population
if TYPE_CHECKING:
    from phenotype import Individual

class Trial(ABC):
    """
    Abstract base class for implementing a NEAT trial.

    A trial represents one independent run of the NEAT algorithm, evolving a
    population through generations until a solution is found or the maximum
    number of generations is reached.

    Subclasses must implement:
    - _reset(): Reset trial-specific state and call super()._reset()
    - _evaluate_fitness(individual): Evaluate fitness for a single individual
    - _report_progress(): Display progress after each generation
    - _final_report(): Display final results

    Subclasses can override:
    - _terminate(): Custom termination logic (default: max generations + fitness threshold)

    Network Types:
        'standard' - Object-oriented implementation (basic implementation, processes one input at a time)
        'fast'     - High-performance vectorized numpy implementation (when data to process comes in batches)
        'autograd' - Autograd-compatible vectorized implementation (when we need gradients)

    Public Methods:
        run(): Execute a complete NEAT trial

    Parallelization of fitness evaluation for individuals:
        num_jobs=1:  Serial evaluation (no parallelization)
        num_jobs>1:  Use specified number of parallel processes
        num_jobs=-1: Use all available CPU cores
    """

    def __init__(self, config: Config, network_type: str, suppress_output: bool = False):
        """
        Initialize the trial.

        Parameters:
            config:          Configuration parameters
            network_type:    Type of network backend to use
                             'standard'   - Object-oriented implementation (basic implementation, processes one input at a time)
                             'fast'       - High-performance vectorized numpy implementation (when data to process comes in batches)
                             'autograd'   - Autograd-compatible vectorized implementation (when we need gradients)
            suppress_output: If True, suppress progress and final reports
                             (useful when running multiple trials in experiments)
        """
        self._config            : Config     = config
        self._generation_counter: int        = 0
        self._population        : Population = None
        self._suppress_output   : bool       = suppress_output
        self._network_type      : str        = network_type
        self.failed             : bool       = True

    def run(self, num_jobs: int = 1):
        """
        Run the trial.

        Resets the trial state and runs the evolutionary
        algorithm until the terminate condition is met.

        Parameters:
            num_jobs: Number of parallel processes for fitness evaluation of individuals
                      1 = serial (no parallelization)
                     -1 = use all available CPU cores
                     >1 = use specified number of processes
        """
        # Reset the trial state before starting a new run
        self._reset()

        # Create the initial population
        self._population =  Population(self._config, self._network_type)

        # Evaluate the fitness of the initial population
        self._evaluate_fitness_all(num_jobs)

        # Display progress for the initial population
        if not self._suppress_output:
            self._report_progress()

        # Evolution loop
        while not self._terminate():
            self._generation_counter += 1

            # The members of the population mate and create offspring
            self._population.spawn_next_generation()

            # Evaluate the fitness of each individual in the new generation
            self._evaluate_fitness_all(num_jobs)

            # Display progress after each generation
            if not self._suppress_output:
                self._report_progress()

        # Produce final report
        if not self._suppress_output:
            self._final_report()

    @abstractmethod
    def _reset(self):
        """
        Reset the trial state before starting a new run.

        Subclasses should call super()._reset() and then
        initialize their problem-specific data.
        """
        InnovationTracker.initialize(self._config)
        self._generation_counter = 0
        self.failed = True

    @abstractmethod
    def _evaluate_fitness(self, individual: 'Individual') -> float:
        """
        Evaluate and return the fitness of an individual.

        This method should test the individual's neural network on the
        problem domain and compute a fitness score. Higher fitness values
        indicate better performance and higher probability of procreating.

        IMPORTANT: The fitness must be a positive number (or zero).

        Parameters:
            individual: The Individual (neural network) to evaluate

        Returns:
            float: Fitness score for the individual
        """
        pass

    def _evaluate_fitness_all(self, num_jobs: int):
        """
        Evaluate fitness for all individuals in the population.

        Uses serial or parallel evaluation based on num_jobs:
        - num_jobs=1: Sequential evaluation in single process
        - num_jobs>1 or -1: Parallel evaluation using joblib

        Parameters:
            num_jobs: Number of parallel processes for fitness evaluation

        Updates individual.fitness for all individuals and calls
        self._population._species_manager.update_fitness().
        """
        individuals = self._population.individuals
        serialize   = num_jobs == 1

        # Calculate individuals' fitness
        if serialize:
            for individual in individuals:
                individual.fitness = self._evaluate_fitness(individual)
        else:
            fitness_all = Parallel(num_jobs)(delayed(self._evaluate_fitness)(i) for i in individuals)
            for individual, fitness in zip(individuals, fitness_all):
                individual.fitness = fitness

        # Calculate species' fitness
        self._population._species_manager.update_fitness()

    @abstractmethod
    def _report_progress(self):
        """
        Report trial progress after each generation.

        This method is called after evaluating each generation and can be used to
        log statistics, visualize results, save checkpoints, or display progress
        information (e.g., generation number, best fitness, species count).

        This method is suppressed by setting 'self._suppress_output' to 'True',
        which we might do when running many trials, as part of an experiment.
        """
        pass

    @abstractmethod
    def _final_report(self):
        """
        Produce final report at the end of the trial.

        This method is suppressed by setting 'self._suppress_output' to 'True',
        which we might do when running many trials, as part of an experiment.
        """
        pass

    def _terminate(self) -> bool:
        """
        Determine whether the trial should terminate.

        This default implementation stops the trial after a maximum number
        of generations and (optionally) also stops it if a given measure of
        population fitness has reached a given threshold.

        Subclasses can override this method for custom termination logic.

        Returns:
            bool: True if the trial should stop, False otherwise
        """
        # Has this trial run for too long?
        terminate = self._generation_counter >= self._config.max_number_generations

        # Check whether the fitness has reached a target threshold
        if self._config.fitness_termination_check:
            individual_fitness = [indv.fitness for indv in self._population.individuals]
            overall_fitness = None

            if self._config.fitness_criterion == "max":
                overall_fitness = max(individual_fitness)
            elif self._config.fitness_criterion == "mean":
                overall_fitness = mean(individual_fitness)
            else:
                raise RuntimeError("bad 'fitness_criterion' in configuration file")

            # Compare a measure of population fitness (max, mean, ...) against a threshold
            success = overall_fitness >= self._config.fitness_threshold
            terminate = terminate or success

            if terminate:
                self.failed = not success

        return terminate
