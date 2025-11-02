"""
1D Function Regression Implementation for NEAT

This module implements 1D function approximation as a regression task for the NEAT algorithm.
The goal is to evolve neural networks that can approximate arbitrary 1D mathematical functions
over a specified range.

The 1D Function Regression Problem:
    Given a continuous function f(x) defined over the range [x_min, x_max], evolve a neural
    network that approximates f(x) for all x in that range. The network is evaluated on multiple
    sample points within the range.

Fitness Function:
    Fitness = 1.0 / (0.1 + MSE)
    where: MSE = mean((network_output - target)²)

    Higher fitness indicates better approximation.
    The offset (0.1) prevents division by zero and helps with numerical stability.

Classes:
    Trial_Regression1D:      NEAT trial for 1D function approximation with configurable network backend
    Experiment_Regression1D: Multi-trial experiment for 1D function approximation with statistical analysis
"""

import autograd.numpy as np  # type: ignore
from pathlib    import Path
from statistics import mean
from typing     import Callable

from evograd.run.config import Config
from evograd.phenotype  import Individual
from evograd.run        import Experiment, Trial

class Trial_Regression1D(Trial):
    """
    NEAT trial for 1D function approximation with configurable network backend.

    Evolves neural networks to approximate a 1D function over a specified range
    using multiple sample points. Supports standard, autograd, and fast network
    backends.

    The trial evaluates fitness based on mean squared error (MSE) between network
    outputs and target function values.
    """

    def __init__(self,
                 config         : Config,
                 network_type   : str,
                 function       : Callable[[float], float],
                 x_min          : float,
                 x_max          : float,
                 suppress_output: bool = False):
        """
        Initialize the 1D function approximation trial.

        Parameters:
            config: Configuration parameters
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
            function: The 1D function being approximated
            x_min: The beginning of the range on which the function is approximated
            x_max: The end of the range on which the function is approximated
            suppress_output: If True, suppress progress and final reports
        """
        super().__init__(config, network_type, suppress_output)

        # Generate sample points for function approximation
        NUM_POINTS = 100
        self._Xs = np.linspace(x_min, x_max, NUM_POINTS)
        self._Ys = np.array([function(x) for x in self._Xs])

        # Repack data if needed
        serial_mode = (self._network_type == 'standard')
        if not serial_mode:
            self._Xs = self._Xs.reshape(-1, 1)

    def _reset(self):
        return super()._reset()

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate individual fitness by measuring function approximation error
        with a complexity penalty based on the number of nodes.

        Implementation depends on network_type:
        - 'standard':         Loop through all grid points individually (NUM_POINTS forward passes)
        - 'autograd', 'fast': Process all grid single batch (1 forward pass)

        Parameters:
            individual: The individual to evaluate

        Returns:
            Fitness score (higher is better, based on inverse MSE with complexity penalty)
        """

        # Serial mode => process each grid point individually
        if self._network_type == 'standard':
            Os = np.array([individual._network.forward_pass([x])[0] for x in self._Xs])

        # Batch mode => process entire grid in one pass
        else:
            Os = individual._network.forward_pass(self._Xs).flatten()

        # Calculate mean squared error
        mse = np.mean((Os - self._Ys)**2)

        # Add complexity penalties based on network structure
        complexity_penalty = \
            (self._config.num_nodes_penalty       * individual._network.number_nodes_hidden +
             self._config.num_connections_penalty * individual._network.number_connections)

        # Fitness is inverse of (MSE + complexity penalty + offset)
        fitness = 1.0 / (0.1 + mse + complexity_penalty)

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

class Experiment_Regression1D(Experiment):
    """
    Multi-trial experiment for 1D function approximation with statistical analysis.

    Runs multiple independent trials and aggregates results including success rate,
    average network complexity, and convergence statistics.
    """

    def __init__(self,
                 num_trials    : int,
                 config        : Config,
                 network_type  : str,
                 function      : Callable[[float], float],
                 x_min         : float,
                 x_max         : float):
        """
        Initialize 1D function approximation experiment with multiple trials.

        Parameters:
            num_trials: Number of trials in this experiment
            config: Configuration parameters
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
            function: The 1D function being approximated
            x_min: The beginning of the range on which the function is approximated
            x_max: The end of the range on which the function is approximated
        """
        super().__init__(Trial_Regression1D, num_trials, config,
                         network_type=network_type, function=function, x_min=x_min, x_max=x_max)

    def _reset(self):
        """
        Reset experiment state before starting a new run.
        """
        super()._reset()

    def _prepare_trial(self, trial: Trial_Regression1D, trial_number: int):
        """
        Configure the experiment in preparation for the next run.
        """
        # the default implementation prints a progress report.
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial: Trial_Regression1D, trial_number: int) -> dict:
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

        # how many times did the algorithm found the minimal network;
        # the minimal network has 1 hidden node => 3 nodes in total
        count_min_network = self._number_neurons.count(3)

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
            s += f"Found minimal network = {100 * count_min_network/self._trial_counter:.0f}%\n"
        else:
            s += "No successful trials - cannot compute statistics\n"

        print(s)
