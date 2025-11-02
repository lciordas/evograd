"""
NEAT Experiment Module

This module defines the abstract base class for NEAT experiments with built-in
support for CPU-based parallelization using joblib.

An experiment represents a collection of multiple independent trials (runs),
used to gather statistical data about the NEAT algorithm's performance.
"""

from abc    import ABC, abstractmethod
from joblib import Parallel, delayed
from sys    import stdout
from typing import Type

from evograd.run.config import Config
from evograd.run.trial  import Trial

class Experiment(ABC):
    """
    Abstract base class for implementing a NEAT experiment.

    An experiment represents a collection of multiple independent trials (runs),
    used to gather statistical data about the NEAT algorithm's performance on a
    specific problem across multiple runs.

    Each trial represents one complete execution of the NEAT algorithm, and the
    experiment aggregates results across all trials to provide statistical insights
    such as success rate, average network complexity, convergence speed, and fitness
    distributions.

    Subclasses must implement:
    - _reset(): Reset experiment-specific state and call super()._reset()
    - _prepare_trial(trial, trial_number): Configure each trial before execution
    - _extract_trial_results(trial, trial_number): Extract results after trial completes
    - _analyze_trial_results(results): Process and display individual trial results
    - _final_report(): Produce aggregated statistical report for entire experiment

    Subclasses inherit:
    - run(): Execute all trials (serial or parallel)
    - Automatic trial instantiation and execution
    - Result aggregation and parallelization management

    Public Methods:
        run(num_jobs_trials=1, num_jobs_fitness=1): Execute the complete experiment

    Parallelization:
        Supports two levels of parallelization:

        Trial-level parallelization (num_jobs_trials):
            1:  Serial trial execution (no parallelization)
           >1:  Use specified number of parallel processes for trials
           -1:  Use all available CPU cores for trials

        Fitness-level parallelization within each trial (num_jobs_fitness):
            1:  Serial fitness evaluation (recommended when num_jobs_trials > 1)
           >1:  Use specified number of parallel processes per trial
           -1:  Use all available CPU cores per trial
    """

    def __init__(self, trial_class: Type[Trial], num_trials: int, config: Config,
                 *args, **kwargs):
        """
        Parameters:
            trial_class: the class describing the trials in this experiment
            num_trials:  number of trials in this experiment
            config:      configuration parameters
            *args:       positional arguments to pass to trial class constructor
            **kwargs:    keyword arguments to pass to trial class constructor
        """
        self._num_trials : int         = num_trials
        self._trial_class: Type[Trial] = trial_class
        self._config     : Config      = config
        self._trial_args               = args
        self._trial_kwargs             = kwargs

        # progress counters
        self._trial_counter  : int = 0  # how many trials we've run so far
        self._success_counter: int = 0  # how many trials found an acceptable solution

        # for each trial, some stats
        self._number_generations: list[int]   = []  # length of trial, in generations
        self._max_fitness       : list[float] = []  # max fitness achieved in trial

        # for each trial, the size of the best network
        self._number_neurons    : list[int] = []   # number of neurons/nodes
        self._number_connections: list[int] = []   # number of *enabled* connections

    @abstractmethod
    def _reset(self):
        """
        Reset experiment state before starting a new run.
        """
        self._trial_counter      = 0
        self._success_counter    = 0
        self._number_generations = []
        self._max_fitness        = []
        self._number_neurons     = []
        self._number_connections = []

    def run(self, num_jobs_trials: int = 1, num_jobs_fitness: int = 1):
        """
        Run the experiment.

        Resets the experiment state and runs the necessary number of trials.
        Trials can be run serially or in parallel based on num_jobs parameter.

        Parameters:
            num_jobs_trials:  Number of parallel processes for running trials
                               1 = serial trial execution (default)
                              -1 = use all available CPU cores for trials
                              >1 = use specified number of processes for trials
            num_jobs_fitness: Number of parallel processes for fitness evaluation within each trial
                               1 = serial (default, recommended when num_jobs_trials > 1 to avoid nested parallelization)
                              -1 = use all available CPU cores
                              >1 = use specified number of processes
        """
        # Reset the state at the beginning of each new experiment
        self._reset()

        # Run all trials, gather results
        serialize = num_jobs_trials == 1

        if serialize:
            results = []
            while self._trial_counter < self._num_trials:
                self._trial_counter += 1
                r = self._run_trial(self._trial_counter, num_jobs_fitness)
                results.append(r)
        else:
            results = Parallel(num_jobs_trials)(
                delayed(self._run_trial)(n, num_jobs_fitness)
                for n in range(1, self._num_trials + 1)
            )
            self._trial_counter = self._num_trials

        # Analyze and display data for each trial, then
        # assemble all the data gathered in a final report
        for r in results:
            self._analyze_trial_results(r)
        self._final_report()

    def _run_trial(self, trial_number: int, num_jobs: int = 1) -> dict:
        """
        Prepare, run, analyze one trial.
        Returns the relevant data generated by the trial.

        Parameters:
            trial_number: The trial number (1-indexed)
            num_jobs:     Number of parallel processes for fitness evaluation within this trial
        """
        # create new trial instance
        trial = \
            self._trial_class(*self._trial_args, config=self._config, suppress_output=True, **self._trial_kwargs)

        # configure the trial we are about to run; this
        # usually means updating the configuration values
        self._prepare_trial(trial, trial_number)

        # Run the trial with the specified number of parallel jobs
        trial.run(num_jobs)

        # extract and return the relevant data for
        # the trial that just completed
        return self._extract_trial_results(trial, trial_number)

    @abstractmethod
    def _prepare_trial(self, trial: Trial, trial_number: int):
        """
        Configure the experiment in preparation for the next run.
        For example, this could mean changing configuration parameters.
        The default implementation prints a progress report.
        Derived implementations need not call this method.
        """
        s = f"Starting trial {trial_number:03d} of {self._num_trials}..."
        stdout.write(s + '\r')
        stdout.flush()

    @abstractmethod
    def _extract_trial_results(self, trial, trial_number: int) -> dict:
        """
        Extract relevant results at the end of a trial.
        This implementation extracts basic information, common to all experiments.
        Derived implementations MUST call this method.
        """
        # get the fittest individual and its neural network
        individual = trial._population.get_fittest_individual()
        network    = individual._network

        results = {"trial_number": trial_number}

        # for how many generations did the trial run
        results["number_generations"] = trial._generation_counter

        # the maximum fitness achieved
        results["max_fitness"] = individual.fitness

        # get the size of the fittest network
        results["number_neurons"]     = network.number_nodes
        results["number_connections"] = network.number_connections_enabled

        # whether an acceptable solution was found during this trial
        results["success"] = not trial.failed

        return results

    @abstractmethod
    def _analyze_trial_results(self, results: dict):
        """
        Analyze, and display the results of each trial.
        This implementation updates basic statistics, common to all experiments.
        Derived implementations MUST call this method.
        """
        if results["success"]:
            self._success_counter += 1 
            self._number_generations.append(results["number_generations"])
            self._max_fitness.append(results["max_fitness"])
            self._number_neurons.append(results["number_neurons"])
            self._number_connections.append(results["number_connections"])

    @abstractmethod
    def _final_report(self):
        """
        Produce the final report, aggregating the data obtained from each trial.
        """
        pass