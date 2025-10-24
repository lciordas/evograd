"""
N-Dimensional Function Regression Implementation for NEAT

This module implements N-dimensional function approximation as a regression task for the NEAT algorithm.
The goal is to evolve neural networks that can approximate arbitrary N-dimensional mathematical functions
over specified bounds for each dimension.

The N-Dimensional Function Regression Problem:
    Given a continuous function f(x) where x ∈ ℝⁿ defined over bounds [x_min, x_max] for each dimension,
    evolve a neural network that approximates f(x) for all x in that region. The network is evaluated
    on multiple sample points within the bounds.

Fitness Function:
    Fitness = 1.0 / (0.1 + MSE)
    where: MSE = mean((network_output - target)²)

    Higher fitness indicates better approximation.
    The offset (0.1) prevents division by zero and helps with numerical stability.

Classes:
    Trial_RegressionND:      NEAT trial for N-dimensional function approximation with configurable network backend
    Experiment_RegressionND: Multi-trial experiment for N-dimensional function approximation with statistical analysis
"""

import autograd.numpy as np  # type: ignore
from pathlib    import Path
from statistics import mean
from typing     import Callable, List, Tuple, Optional

from neat.run.config import Config
from neat.phenotype  import Individual
from neat.run        import Experiment, Trial

class Trial_RegressionND(Trial):
    """
    NEAT trial for N-dimensional function approximation with configurable network backend.

    Evolves neural networks to approximate an N-dimensional function over specified bounds
    using multiple sample points. Supports standard, autograd, and fast network backends.

    The trial evaluates fitness based on mean squared error (MSE) between network
    outputs and target function values.
    """

    def __init__(self,
                 config         : Config,
                 network_type   : str,
                 function       : Callable[[np.ndarray], float],
                 bounds         : List[Tuple[float, float]],
                 num_points     : Optional[int] = None,
                 sampling       : str = 'auto',
                 suppress_output: bool = False):
        """
        Initialize the N-dimensional function approximation trial.

        Parameters:
            config: Configuration parameters
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
            function: The N-dimensional function being approximated
                     Takes an array of shape (n,) and returns a scalar
            bounds: List of (min, max) tuples for each dimension
            num_points: Total number of sample points (auto-scaled if None)
            sampling: Sampling strategy - 'grid', 'random', or 'auto'
                     'auto' uses grid for dims <= 3, random for higher
            suppress_output: If True, suppress progress and final reports
        """
        # Determine dimensionality and validate bounds
        self._n_dims = len(bounds)
        self._bounds = bounds

        # Update config to match problem dimensionality
        config.num_inputs = self._n_dims

        super().__init__(config, network_type, suppress_output)

        self._function = function
        self._sampling = sampling

        # Auto-determine sampling strategy if needed
        if sampling == 'auto':
            self._sampling = 'grid' if self._n_dims <= 3 else 'random'

        # Auto-scale number of points based on dimensionality
        if num_points is None:
            # Use fewer points for higher dimensions to avoid exponential growth
            base_points = 100
            if self._n_dims == 1:
                num_points = base_points
            elif self._n_dims == 2:
                num_points = base_points * 4  # 400 points
            elif self._n_dims == 3:
                num_points = base_points * 8  # 800 points
            else:
                # For high dims, use linear scaling
                num_points = base_points * self._n_dims * 2

        self._num_points = num_points

        # Generate sample points
        self._generate_samples()

    def _generate_samples(self):
        """Generate sample points for function approximation."""
        if self._sampling == 'grid' and self._n_dims <= 3:
            self._generate_grid_samples()
        else:
            self._generate_random_samples()

        # Compute target values
        self._compute_targets()

        # Reshape for batch processing if needed
        serial_mode = (self._network_type == 'standard')
        if serial_mode:
            # Keep Xs as list of arrays for serial processing
            self._Xs_list = [self._Xs[i] for i in range(len(self._Xs))]
        # For batch modes, Xs is already in correct shape (num_points, n_dims)

    def _generate_grid_samples(self):
        """Generate grid-based sample points (for low dimensions)."""
        # Calculate points per dimension
        points_per_dim = int(np.power(self._num_points, 1.0 / self._n_dims))
        actual_points = points_per_dim ** self._n_dims

        # Create grid for each dimension
        grids = []
        for i in range(self._n_dims):
            grids.append(np.linspace(self._bounds[i][0], self._bounds[i][1], points_per_dim))

        # Create meshgrid and flatten
        if self._n_dims == 1:
            self._Xs = grids[0].reshape(-1, 1)
        else:
            mesh = np.meshgrid(*grids)
            self._Xs = np.stack([m.ravel() for m in mesh], axis=-1)

        # Update actual number of points
        self._num_points = actual_points

    def _generate_random_samples(self):
        """Generate random sample points (for high dimensions or by choice)."""
        low = np.array([b[0] for b in self._bounds])
        high = np.array([b[1] for b in self._bounds])
        self._Xs = np.random.uniform(low=low, high=high, size=(self._num_points, self._n_dims))

    def _compute_targets(self):
        """Compute target values for all sample points."""
        # Apply function to each sample point
        self._Ys = np.array([self._function(x) for x in self._Xs])

    def _reset(self):
        return super()._reset()

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate individual fitness by measuring function approximation error.

        Implementation depends on network_type:
        - 'standard':         Loop through all sample points individually
        - 'autograd', 'fast': Process all points in single batch

        Parameters:
            individual: The individual to evaluate

        Returns:
            Fitness score (higher is better, based on inverse MSE)
        """
        # Serial mode => process each point individually
        if self._network_type == 'standard':
            Os = np.array([individual._network.forward_pass(x)[0] for x in self._Xs_list])

        # Batch mode => process entire dataset in one pass
        else:
            Os = individual._network.forward_pass(self._Xs).flatten()

        # Calculate mean squared error
        mse = np.mean((Os - self._Ys)**2)

        # Fitness is inverse of (MSE + offset) to avoid division by zero
        fitness = 1.0 / (0.1 + mse)

        return fitness

    def _generation_report(self):
        """
        Print a report describing the current generation.
        """
        display_report = (self._generation_counter % 20 == 1) or self._terminate()

        if display_report:
            fittest = self._population.get_fittest_individual()
            fittest = fittest.prune()

            s  = f"===============\n"
            s += f"GENERATION {self._generation_counter:04d}\n"
            s += f"Dimensions      = {self._n_dims}\n"
            s += f"Sample points   = {self._num_points} ({self._sampling} sampling)\n"
            s += f"population size = {len(self._population.individuals)}\n"
            s += f"number species  = {len(self._population._species_manager.species)}\n"
            s += f"maximum fitness = {fittest.fitness:.4f}\n"
            s += '\n'
            s += "Fittest Individual:\n"
            s += str(fittest)
            s += '\n'

            print(s)

    def _final_report(self):
        """
        Display results at the end of the trial.
        """
        individual_fittest = self._population.get_fittest_individual().prune()

        # Report final MSE
        if self._network_type == 'standard':
            Os = np.array([individual_fittest._network.forward_pass(x)[0] for x in self._Xs_list])
        else:
            Os = individual_fittest._network.forward_pass(self._Xs).flatten()

        final_mse = np.mean((Os - self._Ys)**2)
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final Fitness: {1.0 / (0.1 + final_mse):.4f}")

        # Visualize the network
        try:
            individual_fittest._network.visualize()
            print("Network visualization saved as 'Digraph.gv.pdf'")
        except Exception as e:
            print(f"Could not visualize network: {e}")

    def _terminate(self) -> bool:
        """
        Determine whether the trial should terminate.
        """
        # using the default implementation.
        return super()._terminate()

class Experiment_RegressionND(Experiment):
    """
    Multi-trial experiment for N-dimensional function approximation with statistical analysis.

    Runs multiple independent trials and aggregates results including success rate,
    average network complexity, and convergence statistics.
    """

    def __init__(self,
                 num_trials    : int,
                 config        : Config,
                 network_type  : str,
                 function      : Callable[[np.ndarray], float],
                 bounds        : List[Tuple[float, float]],
                 num_points    : Optional[int] = None,
                 sampling      : str = 'auto'):
        """
        Initialize N-dimensional function approximation experiment with multiple trials.

        Parameters:
            num_trials: Number of trials in this experiment
            config: Configuration parameters
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
            function: The N-dimensional function being approximated
            bounds: List of (min, max) tuples for each dimension
            num_points: Total number of sample points (auto-scaled if None)
            sampling: Sampling strategy - 'grid', 'random', or 'auto'
        """
        super().__init__(Trial_RegressionND, num_trials, config,
                         network_type=network_type, function=function,
                         bounds=bounds, num_points=num_points, sampling=sampling)

    def _reset(self):
        """
        Reset experiment state before starting a new run.
        """
        super()._reset()

    def _prepare_trial(self, trial: Trial_RegressionND, trial_number: int):
        """
        Configure the experiment in preparation for the next run.
        """
        # the default implementation prints a progress report.
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial: Trial_RegressionND, trial_number: int) -> dict:
        """
        Extract relevant results at the end of a trial.
        """
        results = super()._extract_trial_results(trial, trial_number)
        return results

    def _analyze_trial_results(self, results: dict):
        """
        Extract results of each trial, once complete.
        """
        # Call parent class method to populate statistics lists
        super()._analyze_trial_results(results)

        # print trial summary
        s  = f"Trial {results['trial_number']:03d}: "
        s += f"max fitness={results['max_fitness']:5.2f}, "
        s += f"neurons={results['number_neurons']:2}, "
        s += f"connections={results['number_connections']:3}, "
        s += f"generations={results['number_generations']:3} "
        s += "[SUCCESS]" if results['success'] else "[FAILED]"
        print(s)

    def _final_report(self):
        """
        Produce the final report, aggregating the data obtained from each trial.
        """
        # percentage of trials that found a solution
        success_rate = self._success_counter / self._trial_counter

        # print experiment summary
        s  = "\nSUMMARY:\n"
        s += f"Total trials:         = {self._trial_counter}\n"
        s += f"Success rate          = {100*success_rate:.0f}%\n"

        # Only compute statistics if there were successful trials
        if self._number_neurons:
            num_io_neurons = self._config.num_inputs + self._config.num_outputs

            s += f"Avg # hidden neurons  = {mean(self._number_neurons)-num_io_neurons:.2f}\n"
            s += f"Avg # enabled conns   = {mean(self._number_connections):.2f}\n"
            s += f"Avg # generations     = {mean(self._number_generations):.0f}\n"
            s += f"Avg max fitness       = {mean(self._max_fitness):.4f}\n"
        else:
            s += "No successful trials - cannot compute statistics\n"

        print(s)