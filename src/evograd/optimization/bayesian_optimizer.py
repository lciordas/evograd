"""
Bayesian optimizer for hyperparameter tuning.

This module provides an abstract base class - BayesianOptimizer - that uses the
"Optuna" library to optimize hyperparameters for NEAT + gradient descent trials. 

Subclasses must implement two abstract methods:
 - _extract_trial_results():       Extract experiment-specific results from completed trials
 - _compute_optimization_metric(): Compute the optimization metric from trial results
"""

from abc    import ABC, abstractmethod
from typing import Type, Optional, List, Dict, Any, Callable

import optuna                                   # type: ignore
from optuna import Study, Trial as OptunaTrial  # type: ignore
from optuna.visualization import (              # type: ignore
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import warnings

from evograd                           import Config
from evograd.run                       import Trial
from evograd.optimization.search_space import SearchSpace

class BayesianOptimizer(ABC):
    """
    Abstract base class for Bayesian hyperparameter optimization.

    This class provides a simplified wrapper around Optuna for optimizing
    hyperparameters used in NEAT + gradient descent trials. Subclasses must
    implement two abstract methods:

    1. _extract_trial_results(trial, trial_number) -> dict
       Extract experiment-specific results from completed trials.

    2. _compute_optimization_metric(results) -> float
       Compute the metric to optimize from the list of trial results.

    The optimizer always maximizes the computed metric.

    Example:

        >>> # Define search space
        >>> search_space = SearchSpace()
        >>> search_space.add_int('population_size', 50, 500, log=True)
        >>> search_space.add_float('compatibility_threshold', 1.0, 20.0)
        >>>
        >>> # Run optimization
        >>> optimizer = MyOptimizer(    # class derived from BayesianOptimizer
        ...     trial_class=Trial_XOR,
        ...     config_path='config_xor.ini',
        ...     search_space=search_space
        ... )
        >>> study = optimizer.optimize(n_trials=50)
        >>> best_config = optimizer.get_best_config()
    """

    def __init__(self,
                trial_class        : Type[Trial],
                config_path        : str,
                search_space       : SearchSpace,
                num_trials_per_eval: int = 1,
                num_workers_fitness: int = 1,
                study_name         : Optional[str] = None,
                storage            : Optional[str] = None,
                **trial_kwargs):
        """
        Initialize the Bayesian optimizer.

        Parameters:
            trial_class:         The Trial subclass whose output we want to optimize
            config_path:         Path to base configuration INI file
            search_space:        SearchSpace object defining parameters to optimize
            num_trials_per_eval: Number of evograd trials to run per hyperparameter set evaluation
            num_workers_fitness: Number of parallel processes for fitness evaluation of individuals
            study_name:          Name for the Optuna study (for persistence)
            storage:             Database URL for study persistence (e.g., 'sqlite:///optimization.db')
            **trial_kwargs:      Additional keyword arguments for trial constructor

        Note:
            When using parallel configuration evaluation (num_parallel_configs > 1 in optimize()),
            the total number of worker processes will be approximately:
               num_parallel_configs × (1 + num_workers_fitness)
            Be careful when increasing either parameter to avoid excessive resource usage.
        """
        self.trial_class         = trial_class
        self.base_config_path    = config_path
        self.search_space        = search_space
        self.num_trials_per_eval = num_trials_per_eval
        self.num_workers_fitness = num_workers_fitness
        self.trial_kwargs        = trial_kwargs

        # Create or load Optuna study with TPESampler
        self.study = optuna.create_study(study_name     = study_name,
                                         storage        = storage,
                                         direction      = 'maximize',
                                         sampler        = optuna.samplers.TPESampler(),
                                         load_if_exists = True)

    @abstractmethod
    def _extract_trial_results(self, trial: Trial) -> dict:
        """
        Extract relevant results at the end of a trial.

        This method should be implemented by subclasses to extract 
        experiment-specific information from completed trials. 

        Parameters:
            trial: The completed Trial object

        Returns:
            Dictionary containing extracted results
        """
        pass

    @abstractmethod
    def _compute_optimization_metric(self, results: List[dict]) -> float:
        """
        Compute the metric we want to optimize from the collected trial results.

        This method should be implemented by subclasses to define how to aggregate
        and compute the final optimization metric from all trial results.

        Parameters:
            results: List of trial results as returned by '_extract_trial_results()'

        Returns:
            The metric value to maximize (Optuna will maximize this value)
        """
        pass

    def _objective(self, trial: OptunaTrial) -> float:
        """
        Objective function for Optuna optimization.

        This function:
        1. Creates Config instance from hyperparameters recommended 
           by Optuna as the next step in optimization process
        2. Runs one or more NEAT trials with that config
        3. Extracts and returns the metric we want to maximize

        Parameters:
            trial: Optuna trial object

        Returns:
            Metric value to maximize
        """

        # Create Config instance from the hyperparameters recommended
        # by Optuna as the next step in the optimization process
        suggestions = self.search_space.suggest(trial)
        config = Config(self.base_config_path)
        for param_name, value in suggestions.items():   
            setattr(config, param_name, value)

        # Run all needed evograd trials
        results = []
        for _ in range(self.num_trials_per_eval):
            evograd_trial = self.trial_class(config=config,
                                             suppress_output=True,
                                             **self.trial_kwargs)
            evograd_trial.run(num_jobs=self.num_workers_fitness)

            # Extract results from this trial
            trial_results = self._extract_trial_results(evograd_trial)
            results.append(trial_results)

        # Return optimization metric computed from results
        return self._compute_optimization_metric(results)

    def optimize(self,
                 num_configs         : Optional[int] = None,
                 timeout             : Optional[float] = None,
                 num_parallel_configs: int = 1,
                 callbacks           : Optional[List[Callable]] = None) -> Study:
        """
        Run Bayesian optimization.

        Parameters:
            num_configs:          Total number of parameter configurations to evaluate
            timeout:              Time limit in seconds (alternative to 'num_configs')
            num_parallel_configs: Number of hyperparameter configurations to evaluate in parallel
            callbacks:            List of callback functions

        Returns:
            Optuna Study object with optimization results

        Note:
            Each parameter configuration evaluation runs 'num_trials_per_eval' evograd 
            trials sequentially, with individual fitness evaluations parallelized using 
            'num_workers_fitness workers' (both set in __init__).
            The total number of worker processes is approximately:
              num_parallel_configs × (1 + num_workers_fitness)
        """
        if num_configs is None and timeout is None:
            raise ValueError("Please specify at least one of 'num_configs' or 'timeout' for optimization.")

        self.study.optimize(self._objective,
                            n_trials          = num_configs,
                            timeout           = timeout,
                            n_jobs            = num_parallel_configs,
                            gc_after_trial    = True,
                            show_progress_bar = True,
                            callbacks         = callbacks)
        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        """
        return self.study.best_params

    def get_best_config(self) -> Config:
        """
        Get a Config object with the best parameters.
        """
        config = Config(self.base_config_path)
        for param_name, value in self.study.best_params.items():
            setattr(config, param_name, value)
        return config

    def save_best_config(self, path: str) -> None:
        """
        Save the best configuration to an INI file.
        Parameters:
            path: Path to save the configuration file
        """
        best_config = self.get_best_config()
        best_config.save(path)

    def get_best_value(self) -> float:
        """Get the best objective value found."""
        return self.study.best_value

    def get_best_trial(self) -> optuna.Trial:
        """Get the best trial object."""
        return self.study.best_trial

    def get_n_trials(self) -> int:
        """Get the number of completed trials."""
        return len(self.study.trials)

    # Visualization methods

    def plot_optimization_history(self, **kwargs):
        """
        Plot the optimization history.
        Returns:
            Plotly figure object
        """
        return plot_optimization_history(self.study, **kwargs)

    def plot_param_importances(self, **kwargs):
        """
        Plot hyperparameter importances.
        Returns:
            Plotly figure object
        """
        if len(self.study.trials) < 10:
            warnings.warn("Need at least 10 trials for parameter importance analysis")
            return None
        return plot_param_importances(self.study, **kwargs)

    def plot_parallel_coordinate(self, **kwargs):
        """
        Plot parallel coordinate visualization of trials.
        Returns:
            Plotly figure object
        """
        return plot_parallel_coordinate(self.study, **kwargs)

    def plot_slice(self, **kwargs):
        """
        Plot the parameter slice plot.
        Returns:
            Plotly figure object
        """
        return plot_slice(self.study, **kwargs)

    def print_summary(self) -> None:
        """Print a summary of the optimization results."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Number of finished trials: {len(self.study.trials)}")
        print(f"Best fitness (maximize): {self.study.best_value:.6f}")
        print("\nBest parameters:")
        for param_name, value in self.study.best_params.items():
            print(f"  {param_name}: {value}")
        print("="*60)