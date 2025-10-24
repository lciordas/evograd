"""
N-Dimensional Function Regression Implementation for NEAT with Gradient Descent.

This module extends the N-dimensional function regression solution based on NEAT
to include gradient-based parameter optimization.

The hybrid approach combines:
- NEAT evolution for network topology search
- Gradient descent for fine-tuning connection weights

This can lead to faster convergence and better final accuracy compared
to pure evolutionary optimization.

Classes:
    Trial_RegressionNDGrad: N-dimensional function approximation with gradient descent
    Experiment_RegressionNDGrad: Multi-trial experiment with gradient support
"""

import autograd.numpy as np  # type: ignore
from pathlib import Path
from typing  import Callable, List, Tuple, Optional

from neat.run.config     import Config
from neat.run.trial_grad import TrialGrad
from neat.run            import Experiment

class Trial_RegressionNDGrad(TrialGrad):
    """
    NEAT trial + gradient-descent for N-dimensional function approximation.

    This trial combines NEAT's evolutionary topology search with
    gradient descent for parameter optimization. After each generation,
    the top-performing individuals receive gradient training to
    fine-tune their weights for better N-dimensional function approximation.

    The gradient descent minimizes MSE loss on the same sample points
    used for fitness evaluation, leading to more accurate approximations.
    """

    def __init__(self,
                 config         : Config,
                 function       : Callable[[np.ndarray], float],
                 bounds         : List[Tuple[float, float]],
                 num_points     : Optional[int] = None,
                 sampling       : str = 'auto',
                 suppress_output: bool = False):
        """
        Initialize the gradient-enhanced N-dimensional function approximation trial.

        Parameters:
            config: Configuration parameters (including gradient descent settings)
            function: The N-dimensional function being approximated
                     Takes an array of shape (n,) and returns a scalar
            bounds: List of (min, max) tuples for each dimension
            num_points: Total number of sample points (auto-scaled if None)
            sampling: Sampling strategy - 'grid', 'random', or 'auto'
                     'auto' uses grid for dims <= 3, random for higher
            suppress_output: If True, suppress progress reports

        Note:
            Gradient descent parameters are now configured via the [GRADIENT_DESCENT]
            section in the config file, not as constructor arguments.
        """
        # Determine dimensionality and validate bounds
        self._n_dims = len(bounds)
        self._bounds = bounds
        self._function = function
        self._sampling_strategy = sampling

        # Update config to match problem dimensionality
        config.num_inputs = self._n_dims

        super().__init__(
            config=config,
            network_type='autograd',  # must use 'autograd' network type for gradient support
            suppress_output=suppress_output
        )

        # Auto-determine sampling strategy if needed
        if sampling == 'auto':
            self._sampling_strategy = 'grid' if self._n_dims <= 3 else 'random'

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
        if self._sampling_strategy == 'grid' and self._n_dims <= 3:
            self._generate_grid_samples()
        else:
            self._generate_random_samples()

        # Compute target values
        self._Ys = np.array([self._function(x) for x in self._Xs]).reshape(-1, 1)

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

    def _reset(self):
        """Reset trial state."""
        return super()._reset()

    def _loss_function(self,
                       outputs: np.ndarray,
                       targets: np.ndarray) -> float:
        """
        Compute MSE loss for function approximation.

        Parameters:
            outputs: Network output values (batch_size, num_outputs)
            targets: Target output values (batch_size, num_outputs)

        Returns:
            Mean squared error loss
        """
        return np.mean((outputs - targets)**2)

    def _loss_to_fitness(self, loss: float) -> float:
        """
        Transform MSE loss to fitness using inverse with offset.

        Parameters:
            loss: MSE loss value

        Returns:
            Fitness score (higher is better)
        """
        return 1.0 / (0.1 + loss)

    def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provide training data for both fitness evaluation and gradient descent.

        Returns:
            Tuple of (inputs, targets) for training
        """
        return self._Xs, self._Ys

    def _generation_report(self):
        """
        Print a report describing the current generation.
        """
        display_report = (self._generation_counter % 1 == 0) or self._terminate()

        if display_report:
            fittest = self._population.get_fittest_individual()
            fittest = fittest.prune()

            s = f"===============\n"
            s += f"GENERATION {self._generation_counter:04d}\n"
            s += f"Dimensions      = {self._n_dims}\n"
            s += f"Sample points   = {self._num_points} ({self._sampling_strategy} sampling)\n"
            s += f"population size = {len(self._population.individuals)}\n"
            s += f"number species  = {len(self._population._species_manager.species)}\n"
            s += f"maximum fitness = {fittest.fitness:.4f}\n"

            # Add gradient statistics if enabled
            if self._config.enable_gradient and self._gradient_data:
                s += self._report_GD_statistics()

            s += '\n'
            s += "Fittest Individual:\n"
            s += str(fittest)
            s += '\n'

            print(s)

    def _final_report(self):
        """
        Display final results including gradient training summary.
        """
        individual_fittest = self._population.get_fittest_individual().prune()

        # Report gradient training summary
        if self._config.enable_gradient and self._gradient_data:
            # Extract loss and fitness improvements from gradient data dict
            loss_improvements = [data['loss_improvement'] for data in self._gradient_data.values()]
            fitness_improvements = [data['fitness_improvement'] for data in self._gradient_data.values()]

            print("\n" + "=" * 50)
            print("GRADIENT DESCENT SUMMARY")
            print("=" * 50)
            print(f"Problem dimensionality:        {self._n_dims}D")
            print(f"Total gradient steps applied:  {len(self._gradient_data)}")
            print(f"Average loss improvement:      {np.mean(loss_improvements):.6f}")
            print(f"Total loss improvement:        {np.sum(loss_improvements):.6f}")
            print(f"Average fitness improvement:   {np.mean(fitness_improvements):.6f}")
            print(f"Total fitness improvement:     {np.sum(fitness_improvements):.6f}")
            print()

        # Report final MSE
        Os = individual_fittest._network.forward_pass(self._Xs).reshape(-1, 1)
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
        """Use default termination criteria."""
        return super()._terminate()


class Experiment_RegressionNDGrad(Experiment):
    """
    Multi-trial experiment for gradient-enhanced N-dimensional function approximation.

    Runs multiple independent trials and aggregates results including
    both evolutionary and gradient descent performance metrics.
    """

    def __init__(self,
                 num_trials: int,
                 config    : Config,
                 function  : Callable[[np.ndarray], float],
                 bounds    : List[Tuple[float, float]],
                 num_points: Optional[int] = None,
                 sampling  : str = 'auto'):
        """
        Initialize gradient-enhanced N-dimensional function approximation experiment.

        Parameters:
            num_trials: Number of trials to run
            config: Configuration parameters (including gradient descent settings)
            function: The N-dimensional function to approximate
            bounds: List of (min, max) tuples for each dimension
            num_points: Total number of sample points (auto-scaled if None)
            sampling: Sampling strategy - 'grid', 'random', or 'auto'

        Note:
            Gradient descent parameters are now configured via the [GRADIENT_DESCENT]
            section in the config file, not as constructor arguments.
        """
        super().__init__(Trial_RegressionNDGrad,
                         num_trials,
                         config,
                         function=function,
                         bounds=bounds,
                         num_points=num_points,
                         sampling=sampling)

    def _reset(self):
        """Reset experiment state before starting a new run."""
        super()._reset()

    def _prepare_trial(self, trial: Trial_RegressionNDGrad, trial_number: int):
        """Configure the experiment in preparation for the next run."""
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial: Trial_RegressionNDGrad, trial_number: int) -> dict:
        """Extract relevant results at the end of a trial."""
        results = super()._extract_trial_results(trial, trial_number)
        return results

    def _analyze_trial_results(self, results: dict):
        """Extract results of each trial, once complete."""
        # Call parent class method to populate statistics lists
        super()._analyze_trial_results(results)

        # Print trial summary
        s  = f"Trial {results['trial_number']:03d}: "
        s += f"max fitness={results['max_fitness']:5.2f}, "
        s += f"neurons={results['number_neurons']:2}, "
        s += f"connections={results['number_connections']:3}, "
        s += f"generations={results['number_generations']:3} "
        s += "[SUCCESS]" if results['success'] else "[FAILED]"
        print(s)

    def _final_report(self):
        """
        Produce final report with gradient descent statistics.
        """
        # Call base class report
        super()._final_report()

        # Add gradient-specific statistics if enabled
        if self._config.enable_gradient:
            print("\nGRADIENT DESCENT IMPACT:")
            print(f"  Gradient descent was {'ENABLED' if self._config.enable_gradient else 'DISABLED'}")
            print(f"  Learning rate:      {self._config.learning_rate}")
            print(f"  Steps per update:   {self._config.gradient_steps}")
            print(f"  Update frequency:   Every {self._config.gradient_frequency} generations")
            print(f"  Selection strategy: {self._config.gradient_selection}")
            if self._config.gradient_selection == 'top_k':
                print(f"  Top K individuals:  {self._config.gradient_top_k}")
            print(f"  Lamarckian evolution: {'ENABLED' if self._config.lamarckian_evolution else 'DISABLED'}")