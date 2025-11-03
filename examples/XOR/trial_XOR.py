"""
XOR Problem Implementation for NEAT

This module implements the classic XOR (exclusive OR) problem as a benchmark
for the NEAT algorithm. The XOR problem is a fundamental test case in neural
network research, demonstrating the necessity of hidden layers for solving
non-linearly separable problems.

The XOR Problem:
    XOR is a two-input, one-output boolean function where the output is True
    only when the inputs differ:
        Input (0, 0) → Output 0
        Input (0, 1) → Output 1
        Input (1, 0) → Output 1
        Input (1, 1) → Output 0

    This problem cannot be solved by a single-layer perceptron (linear classifier)
    and requires at least one hidden node, making it an ideal minimal test case 
    for topology-evolving algorithms like NEAT.

Fitness Function:
    Fitness = 4.0 - Σ(output - target)²

    Maximum fitness of 4.0 is achieved when all four XOR cases produce exact outputs.
    The algorithm typically succeeds when fitness exceeds 3.99.

Expected Solution:
    The minimal solution network has:
    -  2 input nodes
    -  1 hidden node
    -  1 output node
    - ~4 connections

    NEAT typically discovers this minimal topology through evolution.

Classes:
    Trial_XOR:      NEAT trial for solving XOR
    Experiment_XOR: Multi-trial experiment for XOR with statistical analysis

Usage:
    Single Trial:
        config = Config("config_xor.ini")
        trial = Trial_XOR(config, network_type='standard')
        trial.run(num_jobs=1)

    Experiment (Multiple Trials):
        config = Config("config_xor.ini")
        experiment = Experiment_XOR(num_trials=100, config=config, network_type='standard')
        experiment.run(num_jobs=-1)
"""

import autograd.numpy as np  # type: ignore
from pathlib    import Path
from statistics import mean

from evograd.run.config import Config
from evograd.phenotype  import Individual
from evograd.run        import Experiment, Trial

class Trial_XOR(Trial):
    """
    NEAT trial for solving the XOR (exclusive OR) problem with configurable network backend.

    This trial evolves neural networks to solve the XOR boolean function,
    a classic benchmark problem in neural network research. The XOR problem
    requires at least one hidden layer, making it a minimal test case for
    verifying that NEAT can evolve network topology.

    Supports different network backends controlled by the 'network_type' parameter:
    - 'standard': Uses NetworkStandard, processes XOR cases individually (single-sample)
    - 'autograd': Uses NetworkAutograd, batch processes all 4 cases together (~20-25x faster)
    - 'fast': Uses NetworkFast, high-performance batch processing (2-5x faster than autograd)

    Data Format:
    - 'standard': XOR data stored as Python lists for individual processing
    - 'autograd' and 'fast': XOR data stored as NumPy arrays for batch processing

    Problem Definition:
        Inputs: 2 binary values (0 or 1)
        Output: 1 binary value (XOR of inputs)
        Training cases: All 4 possible input combinations
        Target accuracy: < 0.01 error per case

    Fitness Calculation:
        - Start with maximum fitness of 4.0
        - Subtract squared error for each of the 4 XOR cases
        - Fitness ≥ 3.99 typically indicates success

    Success Criteria:
        Trial succeeds when fitness reaches the threshold or after
        maximum generations (both specified in the configuration file)

    Implemented Methods:
        _initialize_xor_data(): Initialize XOR training data based on network_type
        _evaluate_fitness(individual): Test network on all 4 XOR cases
        _report_progress(): Display generation statistics and XOR truth table
        _final_report(): Visualize the evolved network structure
    """

    def __init__(self, config: Config, network_type: str, suppress_output: bool = False):
        """
        Initialize the XOR trial.

        Parameters:
            config: Configuration parameters (population size, mutation rates, etc.)
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
            suppress_output: If True, suppress progress and final reports
                           (useful when running multiple trials in an experiment)
        """
        super().__init__(config, network_type, suppress_output)

        # Initialize XOR training data
        if self._network_type == 'standard':

            # when using the standard network, use a simple
            # list as data is processed one item as a time
            self.xor_inputs  = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            self.xor_outputs = [[0.0],      [1.0],      [1.0],      [0.0]]
  
        else:

            # using a network that supports batch processing => pack data as NumPy arrays 
            # shape: (4, 2) for inputs, (4, 1) for outputs
            self.xor_inputs  = np.array([[0.0, 0.0],
                                         [0.0, 1.0],
                                         [1.0, 0.0],
                                         [1.0, 1.0]])
            self.xor_outputs = np.array([[0.0],
                                         [1.0],
                                         [1.0],
                                         [0.0]])

    def _reset(self):
        """Reset trial state."""
        return super()._reset()

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate individual fitness by testing on XOR inputs.

        Implementation depends on network_type:
        - 'standard':         Loop through 4 XOR cases individually (4 forward passes)
        - 'autograd', 'fast': Process all 4 cases in single batch (1 forward pass)

        Parameters:
            individual: The individual to evaluate

        Returns:
            Fitness score (maximum 4.0 for perfect XOR solution)
        """

        # Serial network => must loop through each XOR case individually
        if self._network_type == 'standard':

            fitness = 4.0  # max possible fitness
            for inputs, expected_output in zip(self.xor_inputs, self.xor_outputs):        
                output   = individual._network.forward_pass(inputs)   # forward pass through network
                error    = output[0] - expected_output[0]             # calculate error
                fitness -= error ** 2                                 # errors cause the fitness to decrease

        # Batch mode => single batched forward pass for all 4 XOR cases
        else:
            outputs = individual._network.forward_pass(self.xor_inputs)  # forward pass through network
            errors  = outputs - self.xor_outputs                         # calculate errors
            fitness = 4.0 - np.sum(errors ** 2)                          # errors cause the fitness to decrease

        return fitness

    def _get_xor_outputs(self, network):
        """
        Get network outputs for all 4 XOR test cases.
        This helper method works for both standard and vectorized modes.

        Parameters:
            network: The network to evaluate

        Returns:
            List of (input, output, target, error) tuples for display
        """

        batch_mode = isinstance(self.xor_inputs, np.ndarray)
        results    = []

        for i in range(len(self.xor_inputs)):

            if batch_mode:
                input  = self.xor_inputs[i]
                output = network.forward_pass(input.reshape(1, -1))[0, 0]
                target = self.xor_outputs[i, 0]

            else:
                input  = self.xor_inputs[i]
                output = network.forward_pass(input)[0]
                target = self.xor_outputs[i][0]

            error = abs(output - target)
            results.append((input, output, target, error))

        return results

    def _generation_report(self):
        """
        Print a report describing the current generation.

        This method is suppressed by setting 'self._suppress_output' to 'True',
        which we might do when running many trials, as part of an experiment.
        """
        fittest = self._population.get_fittest_individual()
        fittest = fittest.prune()

        s  = f"===============\n"
        s += f"GENERATION {self._generation_counter:04d}\n"
        s += f"population size = {len(self._population.individuals)}\n"
        s += f"number species  = {len(self._population._species_manager.species)}\n"
        s += f"maximum fitness = {fittest.fitness:.4f}\n"
        s += '\n'
        s += str(fittest)
        s += '\n\n'

        s += "input         output   target  error\n"
        s += "------------------------------------\n"

        # Get outputs for all XOR cases
        results = self._get_xor_outputs(fittest._network)
        for input_vals, output, target, error in results:
            if isinstance(input_vals, np.ndarray):
                input_vals = input_vals.tolist()
            s += f"{input_vals} -> {output:.4f}    {target}   {error:.4f}\n"

        print(s)

    def _final_report(self):
        """
        Display results at the end of the trial.

        This method is suppressed by setting 'self._suppress_output' to 'True',
        which we might do when running many trials, as part of an experiment.
        """
        individual_fittest = self._population.get_fittest_individual().prune()

        # Visualize the network
        try:
            individual_fittest._network.visualize()
            print("Network visualization saved as 'Digraph.gv.pdf'")
        except Exception as e:
            print(f"Could not visualize network: {e}")

class Experiment_XOR(Experiment):

    def __init__(self, num_trials: int, config: Config, network_type: str):
        """
        Initialize XOR experiment with multiple trials.

        Parameters:
            num_trials: Number of trials in this experiment
            config: Configuration parameters
            network_type: Type of network backend to use
                         'standard' - uses NetworkStandard
                         'autograd' - uses NetworkAutograd
                         'fast'     - uses NetworkFast
        """
        super().__init__(Trial_XOR, num_trials, config, network_type=network_type)

    def _reset(self): 
        """
        Reset experiment state before starting a new run.
        """
        super()._reset()

    def _prepare_trial(self, trial: Trial_XOR, trial_number: int):
        """
        Configure the experiment in preparation for the next run.
        """
        # the default implementation prints a progress report.
        super()._prepare_trial(trial, trial_number)

    def _extract_trial_results(self, trial: Trial_XOR, trial_number: int) -> dict:
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
        s += f"max fitness={results['max_fitness']:.2f}, "
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
        # the minimal network has 1 hidden node => 4 nodes in total
        count_min_network = self._number_neurons.count(4)

        # print experiment summary
        s  = "\nSUMMARY:\n"
        s += f"Total trials:         = {self._trial_counter}\n"
        s += f"Success rate          = {100*success_rate:.0f}%\n"

        # Only compute statistics if there were successful trials
        if self._number_neurons:
            num_io_neurons = self._config.num_inputs + self._config.num_outputs

            s += f"Avg # hidden neurons  = {(mean(self._number_neurons)-num_io_neurons):.2f}\n"
            s += f"Avg # enabled conns   = {mean(self._number_connections):.2f}\n"
            s += f"Avg # generations     = {mean(self._number_generations):.0f}\n"
            s += f"Avg max fitness       = {mean(self._max_fitness):.2f}\n"
            s += f"Found minimal network = {100 * count_min_network/self._trial_counter:.0f}%\n"
        else:
            s += "No successful trials - cannot compute statistics\n"
        print(s)