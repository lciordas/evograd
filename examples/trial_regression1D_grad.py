"""
1D Function Regression Implementation for NEAT with Gradient Descent.

This module extends the 1D function regression solution based on NEAT
to include gradient-based parameter optimization.

The hybrid approach combines:
- NEAT evolution for network topology search
- Gradient descent for fine-tuning connection weights

This can lead to faster convergence and better final accuracy compared
to pure evolutionary optimization.

Classes:
    Trial_Regression1DGrad: 1D function approximation with gradient descent
    Experiment_Regression1DGrad: Multi-trial experiment with gradient support
"""

import autograd.numpy as np  # type: ignore
import sys
from pathlib import Path
from typing  import Callable, Tuple
sys.path.append(str(Path(__file__).parent.parent))

from run.config     import Config
from run.trial_grad import TrialGrad
from run            import Experiment

class Trial_Regression1DGrad(TrialGrad):
    """
    NEAT trial + gradient-descent for 1D function approximation.

    This trial combines NEAT's evolutionary topology search with
    gradient descent for parameter optimization. After each generation,
    the top-performing individuals receive gradient training to
    fine-tune their weights for better 1D function approximation.

    The gradient descent minimizes MSE loss on the same sample points
    used for fitness evaluation, leading to more accurate approximations.
    """

    def __init__(self,
                 config: Config,
                 function       : Callable[[float], float],
                 x_min          : float,
                 x_max          : float,
                 suppress_output: bool = False):
        """
        Initialize the gradient-enhanced 1D function approximation trial.

        Parameters:
            config:          Configuration parameters (including gradient descent settings)
            function:        The 1D function being approximated
            x_min:           Beginning of approximation range
            x_max:           End of approximation range
            suppress_output: If True, suppress progress reports

        Note:
            Gradient descent parameters are now configured via the [GRADIENT_DESCENT]
            section in the config file, not as constructor arguments.
        """
        super().__init__(
            config=config,
            network_type='autograd',  # must use 'autograd' network type for gradient support
            suppress_output=suppress_output
        )

        # Generate sample points for function approximation
        NUM_POINTS = 100
        self._Xs = np.linspace(x_min, x_max, NUM_POINTS).reshape(-1, 1)
        self._Ys = np.array([function(x) for x in self._Xs]).reshape(-1, 1)

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

    def _report_progress(self):
        """
        Display progress including gradient descent statistics.
        """
        display_progress = (self._generation_counter % 20 == 1) or self._terminate()

        if display_progress:
            fittest = self._population.get_fittest_individual()
            fittest = fittest.prune()

            s = f"===============\n"
            s += f"GENERATION {self._generation_counter:04d}\n"
            s += f"population size = {len(self._population.individuals)}\n"
            s += f"number species  = {len(self._population._species_manager.species)}\n"
            s += f"maximum fitness = {fittest.fitness:.4f}\n"

            # Add gradient statistics if enabled
            if self._config.enable_gradient and self._gradient_improvements:
                s += self._report_gradient_statistics()

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
        if self._config.enable_gradient and self._gradient_improvements:
            print("\n" + "=" * 50)
            print("GRADIENT DESCENT SUMMARY")
            print("=" * 50)
            print(f"Total gradient steps applied: {len(self._gradient_improvements)}")
            print(f"Average loss improvement:      {np.mean(self._gradient_improvements):.6f}")
            print(f"Total loss improvement:        {np.sum(self._gradient_improvements):.6f}")
            print()

        # Visualize the network
        try:
            individual_fittest._network.visualize()
            print("Network visualization saved as 'Digraph.gv.pdf'")
        except Exception as e:
            print(f"Could not visualize network: {e}")

    def _terminate(self) -> bool:
        """Use default termination criteria."""
        return super()._terminate()


class Experiment_Regression1DGrad(Experiment):
    """
    Multi-trial experiment for gradient-enhanced 1D function approximation.

    Runs multiple independent trials and aggregates results including
    both evolutionary and gradient descent performance metrics.
    """

    def __init__(self,
                 num_trials: int,
                 config    : Config,
                 function  : Callable[[float], float],
                 x_min     : float,
                 x_max     : float):
        """
        Initialize gradient-enhanced 1D function approximation experiment.

        Parameters:
            num_trials: Number of trials to run
            config:     Configuration parameters (including gradient descent settings)
            function:   The 1D function to approximate
            x_min:      Beginning of approximation range
            x_max:      End of approximation range

        Note:
            Gradient descent parameters are now configured via the [GRADIENT_DESCENT]
            section in the config file, not as constructor arguments.
        """
        super().__init__(Trial_Regression1DGrad,
                         num_trials,
                         config,
                         function=function,
                         x_min=x_min,
                         x_max=x_max)

    def _reset(self):
        """Reset experiment state before starting a new run."""
        super()._reset()

    def _prepare_trial(self, trial: Trial_Regression1DGrad, trial_number: int):
        """Configure the experiment in preparation for the next run."""
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial: Trial_Regression1DGrad, trial_number: int) -> dict:
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
