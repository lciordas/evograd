"""
Function Regression Implementation for NEAT with Gradient Descent.

This module extends the function regression solution based on NEAT
to include gradient-based parameter optimization.

The hybrid approach combines:
- NEAT evolution for network topology search
- Gradient descent for fine-tuning connection weights

This can lead to faster convergence and better final accuracy compared
to pure evolutionary optimization.

Classes:
    Trial_ApproxFunction_Grad: Function approximation with gradient descent
    Experiment_ApproxFunction_Grad: Multi-trial experiment with gradient support
"""

import autograd.numpy as np  # type: ignore
import sys
from pathlib import Path
from typing  import Callable, Tuple
sys.path.append(str(Path(__file__).parent.parent))

from run.config     import Config
from run.trial_grad import TrialGrad
from run            import Experiment

class Trial_RegressionGrad(TrialGrad):
    """
    NEAT trial + gradient-descent for function approximation.
    
    This trial combines NEAT's evolutionary topology search with
    gradient descent for parameter optimization. After each generation,
    the top-performing individuals receive gradient training to
    fine-tune their weights for better function approximation.

    The gradient descent minimizes MSE loss on the same sample points
    used for fitness evaluation, leading to more accurate approximations.
    """

    def __init__(self,
                 config: Config,
                 function          : Callable[[float], float],
                 x_min             : float,
                 x_max             : float,
                 suppress_output   : bool = False,
                 enable_gradient   : bool = True,
                 gradient_steps    : int = 20,
                 learning_rate     : float = 0.1,
                 gradient_frequency: int = 5,
                 gradient_selection: str = 'top_k',
                 gradient_top_k    : int = 10):
        """
        Initialize the gradient-enhanced function approximation trial.

        Parameters:
            config:             Configuration parameters
            function:           The 1D function being approximated
            x_min:              Beginning of approximation range
            x_max:              End of approximation range
            suppress_output:    If True, suppress progress reports
            enable_gradient:    Whether to use gradient descent
            gradient_steps:     Number of SGD steps per application
            learning_rate:      Learning rate for gradient updates
            gradient_frequency: Apply gradients every N generations
            gradient_selection: How to select individuals for training
            gradient_top_k:     Number of top individuals to train
        """
        super().__init__(
            config=config,
            network_type='autograd', # must use 'autograd' network type for gradient support
            suppress_output=suppress_output,
            enable_gradient=enable_gradient,
            gradient_steps=gradient_steps,
            learning_rate=learning_rate,
            gradient_frequency=gradient_frequency,
            gradient_selection=gradient_selection,
            gradient_top_k=gradient_top_k
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
            if self._enable_gradient and self._gradient_improvements:
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
        if self._enable_gradient and self._gradient_improvements:
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


class Experiment_RegressionGrad(Experiment):
    """
    Multi-trial experiment for gradient-enhanced function approximation.

    Runs multiple independent trials and aggregates results including
    both evolutionary and gradient descent performance metrics.
    """

    def __init__(self,
                 num_trials        : int,
                 config            : Config,
                 function          : Callable[[float], float],
                 x_min             : float,
                 x_max             : float,
                 enable_gradient   : bool = True,
                 gradient_steps    : int = 20,
                 learning_rate     : float = 0.1,
                 gradient_frequency: int = 5,
                 gradient_selection: str = 'top_k',
                 gradient_top_k    : int = 10):
        """
        Initialize gradient-enhanced experiment.

        Parameters:
            num_trials:         Number of trials to run
            config:             Configuration parameters
            function:           The 1D function to approximate
            x_min:              Beginning of approximation range
            x_max:              End of approximation range
            enable_gradient:    Whether to use gradient descent
            gradient_steps:     Number of SGD steps per application
            learning_rate:      Learning rate for gradient updates
            gradient_frequency: Apply gradients every N generations
            gradient_selection: How to select individuals for training
            gradient_top_k:     Number of top individuals to train
        """
        # Store gradient parameters for trial creation
        self._gradient_params = {
            'enable_gradient'   : enable_gradient,
            'gradient_steps'    : gradient_steps,
            'learning_rate'     : learning_rate,
            'gradient_frequency': gradient_frequency,
            'gradient_selection': gradient_selection,
            'gradient_top_k'    : gradient_top_k
        }

        super().__init__(Trial_RegressionGrad,
                         num_trials, 
                         config,
                         function=function,
                         x_min=x_min,
                         x_max=x_max,
                       **self._gradient_params)

    def _final_report(self):
        """
        Produce final report with gradient descent statistics.
        """
        # Call base class report
        super()._final_report()

        # Add gradient-specific statistics if enabled
        if self._gradient_params['enable_gradient']:
            print("\nGRADIENT DESCENT IMPACT:")
            print(f"  Gradient descent was {'ENABLED' if self._gradient_params['enable_gradient'] else 'DISABLED'}")
            print(f"  Learning rate:      {self._gradient_params['learning_rate']}")
            print(f"  Steps per update:   {self._gradient_params['gradient_steps']}")
            print(f"  Update frequency:   Every {self._gradient_params['gradient_frequency']} generations")
            print(f"  Selection strategy: {self._gradient_params['gradient_selection']}")
            if self._gradient_params['gradient_selection'] == 'top_k':
                print(f"  Top K individuals:  {self._gradient_params['gradient_top_k']}")
