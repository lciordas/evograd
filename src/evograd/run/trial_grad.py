"""
NEAT Trial with Gradient Descent Module

This module defines an abstract base class for NEAT trials that combine
evolutionary topology search with gradient-based parameter optimization.

TrialGrad extends the base Trial class to add optional gradient descent
training phases, allowing for hybrid optimization strategies where NEAT
evolves network topology while gradient descent fine-tunes weights.

Classes:
    TrialGrad: Abstract base class combining NEAT evolution with gradient descent
"""

import autograd.numpy as np   # type: ignore
from abc                      import abstractmethod
from joblib                   import Parallel, delayed
from autograd                 import grad   # type: ignore
from autograd.misc.optimizers import adam   # type: ignore
from typing                   import TYPE_CHECKING

if TYPE_CHECKING:
    from evograd.phenotype import Individual
from evograd.run.trial  import Trial
from evograd.run.config import Config

class TrialGrad(Trial):
    """
    Abstract base class for NEAT trials with gradient descent support.

    This class extends the standard NEAT Trial to add gradient-based
    optimization capabilities. It allows for hybrid optimization, where
    evolutionary algorithms handle topology search while gradient descent
    optimizes connection weights, biases, and gains.

    Subclasses must implement (in addition to Trial requirements):
    - _loss_function(outputs, targets): Define how to compute loss given network outputs
    - _loss_to_fitness(loss): Define how to transform loss to fitness
    - _get_training_data(): Provide training data for both fitness and gradient descent

    Gradient Training Configuration (via Config.GRADIENT_DESCENT section):
        enable_gradient:      Whether to use gradient descent (default: False)
        gradient_steps:       Number of Adam optimizer iterations per application (default: 10)
        learning_rate:        Learning rate for Adam optimizer (default: 0.01)
        gradient_frequency:   Apply gradients every N generations (default: 1)
        gradient_selection:   Which individuals to train ('all', 'top_k', 'top_percent')
        gradient_top_k:       Number of top individuals to train (default: 5)
        gradient_top_percent: Percentage of top individuals to train (default: 0.1)
        lamarckian_evolution: Save optimized parameters to genome for inheritance (default: False)

    Public Methods (inherited from Trial):
        run(): Execute a complete NEAT trial with optional gradient descent
    """

    def __init__(self,
                 config         : Config,
                 network_type   : str,
                 suppress_output: bool = False):
        """
        Initialize the gradient-enabled trial.

        Parameters:
            config:          Configuration parameters (including gradient descent settings)
            network_type:    Type of network backend ('autograd' required for gradients)
            suppress_output: If True, suppress progress and final reports
        """
        super().__init__(config, network_type, suppress_output)

        # Validate network type for gradient support
        if self._config.enable_gradient and network_type != 'autograd':
            raise ValueError("Gradient descent requires network_type='autograd'")

        # Track gradient descent data per individual
        # Gets reset each generatios
        self._gradient_data = {}  # Dict[int, dict] - key: individual.ID

    @abstractmethod
    def _loss_function(self,
                       outputs: np.ndarray,
                       targets: np.ndarray) -> float:
        """
        Compute the loss given network outputs and target values.

        This is the problem-specific loss function that subclasses must implement.
        Common choices include:
        - Mean Squared Error (MSE) for regression
        - Cross-entropy for classification
        - Custom losses for specific tasks

        Parameters:
            outputs: Network output values (batch_size, num_outputs)
            targets: Target output values  (batch_size, num_outputs)

        Returns:
            Loss value (scalar) to be minimized via gradient descent
        """
        pass

    @abstractmethod
    def _loss_to_fitness(self, loss: float) -> float:
        """
        Transform loss value to fitness value.

        Since NEAT maximizes fitness but gradient descent minimizes loss,
        this method defines the transformation between these two metrics.

        IMPORTANT: Must return a positive value (or zero), as required by NEAT.

        Examples:
            # Inverse with offset (common for MSE)
            def _loss_to_fitness(self, loss):
                return 1.0 / (0.1 + loss)

            # Exponential decay
            def _loss_to_fitness(self, loss):
                return np.exp(-loss)

            # Linear transformation (if loss has known bounds)
            def _loss_to_fitness(self, loss):
                return max(0.0, 100.0 - loss)
        
        Parameters:
            loss: Loss value (output from _loss_function)

        Returns:
            Fitness value (positive number, higher is better)
        """
        pass

    @abstractmethod
    def _get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Provide training/evaluation data for fitness evaluation and gradient descent.

        This data is used for both:
        - Fitness evaluation: Computing how well individuals perform
        - Gradient descent:   Computing gradients and updating network parameters

        Returns:
            Tuple of (inputs, targets) arrays:
            - inputs:  (batch_size, num_inputs)
            - targets: (batch_size, num_outputs)
        """
        pass

    def _reset(self):
        """Reset trial state."""
        super()._reset()
        self._gradient_data = {}

    def _evaluate_fitness(self, individual: "Individual") -> float:
        """
        Evaluate fitness of an individual.

        Parameters:
            individual: The individual to evaluate

        Returns:
            Fitness score (positive number, higher is better)
        """
        # Get training data
        inputs, targets = self._get_training_data()

        # Forward pass through network
        outputs = individual._network.forward_pass(inputs)

        # Compute loss and transform to fitness
        loss = self._loss_function(outputs, targets)
        return self._loss_to_fitness(loss)

    def _evaluate_fitness_all(self, num_jobs: int):
        """
        Evaluate fitness for all individuals with optional gradient descent.

        Uses a two-pass approach:
        1. First pass:  Standard NEAT fitness evaluation for all individuals
        2. Second pass: Apply gradient descent to selected individuals to
                        fine-tune their neural-network and increase fitness

        IMPORTANT: This method modifies individuals that undergo gradient descent:
        - Always modified: fitness and network parameters (weights, biases, gains)
        - Lamarckian only: genome is also modified to persist optimized parameters

        The key distinction between modes:
        - Baldwin effect: Individuals benefit from optimization (better fitness/params)
                         during selection, but offspring inherit unoptimized genomes
        - Lamarckian:    Individuals benefit from optimization AND pass optimized
                         parameters to offspring via modified genomes

        Parameters:
            num_jobs: Number of parallel processes for evaluation
        """
        # Pass #1: Compute standard NEAT fitness for all individuals
        super()._evaluate_fitness_all(num_jobs)

        # Pass #2: Fine-tune selected individuals via gradient descent
        do_gradient_descent = (self._config.enable_gradient and
                               self._generation_counter % self._config.gradient_frequency == 0)
        selected_individuals = self._select_GD_individuals() if do_gradient_descent else []
        do_gradient_descent  = do_gradient_descent and selected_individuals

        if do_gradient_descent:

            # Forget previous generation data
            self._gradient_data = {}

            # Store pre-GD fitness for all selected individuals
            fitness_before_gd = {ind.ID: ind.fitness for ind in selected_individuals}

            # -----

            # Run gradient descent optimization. This calculates optimized NN parameters 
            # (and the corresponding increased fitness) for each individual, and returns 
            # these results without modifying the individual in any way. 
            optimization_results = []
            serialize = (num_jobs == 1)
            if serialize:
                for individual in selected_individuals:
                    opt_results = self._GD_optimize(individual)
                    optimization_results.append(opt_results)
            else:
                optimization_results = \
                    Parallel(num_jobs)(delayed(self._GD_optimize)(i) for i in selected_individuals)

            # -----

            # Update individuals:
            # 1. Fitness is replaced by gd-optimized fitness, which is used to select members for reproduction.
            # 2. Network parameters are replaced with optimized values (to match the optimized fitness).
            # 3. If Lamarckian evolution, then optimized network parameters are also saved to genome (to be inherited),
            #    otherwise (Baldwin effect) optimized parameters stay in network but NOT in genome (not inherited).
            for individual, opt_results in zip(selected_individuals, optimization_results):

                # Update individual fitness with optimized value
                individual.fitness = opt_results['final_fitness']

                # Update network parameters with optimized values.
                # Evaluating fitness using the updated network
                # generates the optimized fitness (consistency).
                network = individual._network
                network.set_parameters(
                    opt_results['optimized_weights'],
                    opt_results['optimized_biases'],
                    opt_results['optimized_gains'],
                    opt_results['optimized_coeffs']
                )

                # For Lamarckian evolution: save the optimized parameters to genome for inheritance
                if self._config.lamarckian_evolution:
                    network.save_parameters_to_genome()

            # Update species fitness after gradient improvements
            self._population._species_manager.update_fitness()

            # -----

            # Store gradient optimization results for reporting / analytics purposes
            for individual, opt_results in zip(selected_individuals, optimization_results):

                self._gradient_data[individual.ID] = {
                    'fitness_before_gd'  : fitness_before_gd[individual.ID],
                    'fitness_after_gd'   : opt_results['final_fitness'],
                    'fitness_improvement': opt_results['fitness_improvement'],
                    'loss_before_gd'     : opt_results['initial_loss'],
                    'loss_after_gd'      : opt_results['final_loss'],
                    'loss_improvement'   : opt_results['loss_improvement'],
                    'generation'         : self._generation_counter
                }

    def _GD_optimize(self, individual: "Individual") -> dict:
        """
        Apply gradient descent to optimize an individual's network parameters.

        This is a pure function that computes optimized parameters without
        modifying the individual, its network, or genome in any way.

        Respects the freeze settings from configuration to selectively prevent
        updates to certain parameter types during optimization.

        Parameters:
            individual: The individual to optimize via gradient descent

        Returns:
            dict: Optimization results containing:
                - 'initial_loss':        loss before gradient descent
                - 'final_loss':          loss after gradient descent
                - 'loss_improvement':    initial_loss - final_loss
                - 'initial_fitness':     fitness before gradient descent
                - 'final_fitness':       fitness after gradient descent
                - 'fitness_improvement': final_fitness - initial_fitness
                - 'optimized_weights':   optimized weight matrix (or original if frozen)
                - 'optimized_biases':    optimized bias vector (or original if frozen)
                - 'optimized_gains':     optimized gain vector (or original if frozen)
                - 'optimized_coeffs':    optimized activation coefficients dict (or original if frozen)
        """
        # Get training data
        inputs, targets = self._get_training_data()

        # Get current network parameters (including activation coefficients)
        network = individual._network
        weights, biases, gains, activation_coeffs = network.get_parameters()

        # Store original parameters for frozen ones
        original_weights = weights.copy()
        original_biases  = biases.copy()
        original_gains   = gains.copy()
        original_coeffs  = {k: v.copy() for k, v in activation_coeffs.items()} if activation_coeffs else {}

        # Check if all parameters are frozen (skip optimization if so)
        all_frozen = (self._config.freeze_weights and
                      self._config.freeze_biases  and
                      self._config.freeze_gains   and
                     (self._config.freeze_activation_coeffs or not activation_coeffs))

        if all_frozen:
            outputs = network.forward_pass(inputs, weights, biases, gains, activation_coeffs)
            initial_loss = self._loss_function(outputs, targets)
            initial_fitness = self._loss_to_fitness(initial_loss)
            return {
                'initial_loss':        initial_loss,
                'final_loss':          initial_loss,
                'loss_improvement':    0.0,
                'initial_fitness':     initial_fitness,
                'final_fitness':       initial_fitness,
                'fitness_improvement': 0.0,
                'optimized_weights':   original_weights,
                'optimized_biases':    original_biases,
                'optimized_gains':     original_gains,
                'optimized_coeffs':    original_coeffs
            }

        # Build flattened parameter array with only non-frozen parameters
        params_to_optimize = []
        param_info = []  # Track which parameters are included and their shapes

        w_size = np.prod(weights.shape)
        b_size = np.prod(biases.shape)
        g_size = np.prod(gains.shape)

        # Add non-frozen parameters to optimization list
        if not self._config.freeze_weights:
            params_to_optimize.append(weights.flatten())
            param_info.append(('weights', weights.shape, w_size))

        if not self._config.freeze_biases:
            params_to_optimize.append(biases.flatten())
            param_info.append(('biases', biases.shape, b_size))

        if not self._config.freeze_gains:
            params_to_optimize.append(gains.flatten())
            param_info.append(('gains', gains.shape, g_size))

        flat_coeffs, coeffs_metadata = self._flatten_activation_coeffs(activation_coeffs)
        if not self._config.freeze_activation_coeffs and len(flat_coeffs) > 0:
            params_to_optimize.append(flat_coeffs)
            param_info.append(('coeffs', None, len(flat_coeffs)))

        flat_params_to_optimize = np.concatenate(params_to_optimize)

        # Define loss function that reconstructs full parameter set
        def objective(opt_params, _iter=None):

            # Reconstruct full parameter set from optimized subset
            w = original_weights if self._config.freeze_weights else weights
            b = original_biases  if self._config.freeze_biases  else biases
            g = original_gains   if self._config.freeze_gains   else gains
            c = original_coeffs  if self._config.freeze_activation_coeffs else activation_coeffs

            # Update non-frozen parameters from opt_params
            offset = 0
            for param_name, shape, size in param_info:
                if param_name == 'weights':
                    w = opt_params[offset:offset + size].reshape(shape)
                elif param_name == 'biases':
                    b = opt_params[offset:offset + size].reshape(shape)
                elif param_name == 'gains':
                    g = opt_params[offset:offset + size].reshape(shape)
                elif param_name == 'coeffs':
                    c_flat = opt_params[offset:offset + size]
                    c = self._unflatten_activation_coeffs(c_flat, coeffs_metadata)
                offset += size

            # Compute loss with full parameter set
            outputs = network.forward_pass(inputs, w, b, g, c)
            return self._loss_function(outputs, targets)

        # Track initial loss and fitness (using original full parameter set)
        flat_params = np.concatenate([weights.flatten(),
                                      biases.flatten(),
                                      gains.flatten(),
                                      flat_coeffs])
        initial_loss = objective(flat_params_to_optimize) if len(flat_params_to_optimize) > 0 else objective(flat_params)
        initial_fitness = self._loss_to_fitness(initial_loss)

        # Run optimizer
        grad_fn = grad(objective)
        optimized_params = adam(grad_fn, flat_params_to_optimize,
                                num_iters=self._config.gradient_steps,
                                step_size=self._config.learning_rate)

        # Calculate final loss and fitness
        final_loss = objective(optimized_params) if len(optimized_params) > 0 else initial_loss
        loss_improvement = initial_loss - final_loss
        final_fitness = self._loss_to_fitness(final_loss)
        fitness_improvement = final_fitness - initial_fitness

        # Reconstruct full optimized parameters
        opt_weights = original_weights if self._config.freeze_weights else weights
        opt_biases  = original_biases  if self._config.freeze_biases  else biases
        opt_gains   = original_gains   if self._config.freeze_gains   else gains
        opt_coeffs  = original_coeffs  if self._config.freeze_activation_coeffs else activation_coeffs

        # Extract optimized values for non-frozen parameters
        if len(optimized_params) > 0:
            offset = 0
            for param_name, shape, size in param_info:
                if param_name == 'weights':
                    opt_weights = optimized_params[offset:offset + size].reshape(shape)
                elif param_name == 'biases':
                    opt_biases = optimized_params[offset:offset + size].reshape(shape)
                elif param_name == 'gains':
                    opt_gains = optimized_params[offset:offset + size].reshape(shape)
                elif param_name == 'coeffs':
                    opt_coeffs_flat = optimized_params[offset:offset + size]
                    opt_coeffs = self._unflatten_activation_coeffs(opt_coeffs_flat, coeffs_metadata)
                offset += size

        # Return results
        return {
            'initial_loss':        initial_loss,
            'final_loss':          final_loss,
            'loss_improvement':    loss_improvement,
            'initial_fitness':     initial_fitness,
            'final_fitness':       final_fitness,
            'fitness_improvement': fitness_improvement,
            'optimized_weights':   opt_weights,
            'optimized_biases':    opt_biases,
            'optimized_gains':     opt_gains,
            'optimized_coeffs':    opt_coeffs
        }

    def _select_GD_individuals(self) -> list["Individual"]:
        """
        Select which individuals should receive gradient descent training.

        Returns:
            List of individuals to train with gradient descent
        """
        individuals = self._population.individuals

        if self._config.gradient_selection == 'all':
            return individuals

        elif self._config.gradient_selection == 'top_k':
            sorted_individuals = sorted(individuals,
                                       key=lambda x: (x.fitness, -x.ID),
                                       reverse=True)
            return sorted_individuals[:self._config.gradient_top_k]

        elif self._config.gradient_selection == 'top_percent':
            sorted_individuals = sorted(individuals,
                                       key=lambda x: (x.fitness, -x.ID),
                                       reverse=True)
            num_to_select = max(1, int(len(individuals) * self._config.gradient_top_percent))
            return sorted_individuals[:num_to_select]

        else:
            raise ValueError(f"Unknown gradient_selection: {self._config.gradient_selection}")

    def _flatten_activation_coeffs(self, coeffs_dict: dict[int, np.ndarray]) -> tuple[np.ndarray, dict]:
        """
        Flatten activation coefficients dictionary into a 1D array for Adam optimizer.

        Parameters:
            coeffs_dict: Dictionary mapping node_idx => coefficient arrays

        Returns:
            tuple: (flat_array, metadata)
                - flat_array: 1D concatenated array of all coefficients
                - metadata: dict with keys 'node_indices', 'shapes', 'start_indices'
        """
        # No learnable activations
        if not coeffs_dict:
            return np.array([]), {'node_indices': [], 'shapes': [], 'start_indices': []}

        # Sort by node index for deterministic ordering
        sorted_items = sorted(coeffs_dict.items())
        node_indices = [idx          for idx, _    in sorted_items]
        shapes       = [coeffs.shape for _, coeffs in sorted_items]

        # Concatenate all coefficient arrays (already 1D, no need to flatten)
        coeffs_list = [coeffs for _, coeffs in sorted_items]
        flat_array  = np.concatenate(coeffs_list) if coeffs_list else np.array([])

        # Calculate start indices for unflattening
        start_indices = [0]
        for shape in shapes[:-1]:
            start_indices.append(start_indices[-1] + np.prod(shape))

        metadata = {
            'node_indices': node_indices,
            'shapes': shapes,
            'start_indices': start_indices
        }

        return flat_array, metadata

    def _unflatten_activation_coeffs(self, flat_array: np.ndarray, metadata: dict) -> dict[int, np.ndarray]:
        """
        Reconstruct activation coefficients dictionary from flattened array.

        Parameters:
            flat_array: 1D array containing all coefficients
            metadata: Dictionary with 'node_indices', 'shapes', 'start_indices'

        Returns:
            Dictionary mapping node_idx => coefficient arrays
        """
        if len(metadata['node_indices']) == 0:
            return {}

        coeffs_dict = {}
        node_indices = metadata['node_indices']
        shapes = metadata['shapes']
        start_indices = metadata['start_indices']

        for i, node_idx in enumerate(node_indices):
            start = start_indices[i]
            size  = np.prod(shapes[i])
            end   = start + size
            coeffs_dict[node_idx] = flat_array[start:end].reshape(shapes[i])

        return coeffs_dict

    def _report_GD_statistics(self):
        """
        Report statistics about gradient descent performance.

        This method can be called from '_generation_report()' 
        to display gradient training statistics.
        """
        if self._gradient_data:
            fitness_improvements = [data['fitness_improvement'] for data in self._gradient_data.values()]
            avg_fitness_improvement = np.mean(fitness_improvements)
            s = f"Avg fitness improvement due to gradient descent: {avg_fitness_improvement:.6f}\n"
            return s
        return ""
