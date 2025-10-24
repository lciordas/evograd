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
    from neat.phenotype import Individual
from neat.run.trial  import Trial
from neat.run.config import Config

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

    Internal Attributes:
        _gradient_data: Dict mapping individual ID to gradient descent statistics
                       Structure: {individual_id: {'fitness_before_gd', 'fitness_after_gd',
                                   'fitness_improvement', 'loss_before_gd', 'loss_after_gd',
                                   'loss_improvement', 'generation'}}
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
        2. Second pass: Apply gradient descent to selected individuals and re-evaluate

        Parameters:
            num_jobs: Number of parallel processes for evaluation
        """
        # Pass #1: Compute standard NEAT fitness for all individuals
        super()._evaluate_fitness_all(num_jobs)

        # Pass #2: Apply gradient descent to selected individuals
        do_gradient_descent = (self._config.enable_gradient and
                               self._generation_counter % self._config.gradient_frequency == 0)
        selected_individuals = self._select_individuals_for_gradient() if do_gradient_descent else []
        do_gradient_descent  = do_gradient_descent and selected_individuals

        if do_gradient_descent:
            # Clear gradient data from previous generations to avoid stale reporting
            # Only keep data for individuals being optimized in THIS generation
            self._gradient_data = {}

            # Store pre-GD fitness for all selected individuals
            fitness_before_gd = {ind.ID: ind.fitness for ind in selected_individuals}

            serialize = (num_jobs == 1)
            if serialize:
                for individual in selected_individuals:

                    # Apply gradient descent to one individual
                    result = self._apply_gradient_descent(individual)
                    # Unpack result (genome_data is 4th element if present)
                    loss_final = result[0]
                    loss_improvement = result[1]
                    fitness_improvement = result[2]
                    # genome_data = result[3] if len(result) > 3 else None  # Not needed in serial mode

                    fitness_final      = self._loss_to_fitness(loss_final)
                    individual.fitness = fitness_final

                    # Store gradient data for this individual
                    self._gradient_data[individual.ID] = {
                        'fitness_before_gd'  : fitness_before_gd[individual.ID],
                        'fitness_after_gd'   : fitness_final,
                        'fitness_improvement': fitness_improvement,
                        'loss_before_gd'     : loss_final + loss_improvement,
                        'loss_after_gd'      : loss_final,
                        'loss_improvement'   : loss_improvement,
                        'generation'         : self._generation_counter
                    }
            else:

                # Apply gradient descent to all individuals in parallel
                results = \
                    Parallel(num_jobs)(delayed(self._apply_gradient_descent)(i) for i in selected_individuals)

                for individual, result in zip(selected_individuals, results):
                    # Unpack result
                    loss_final = result[0]
                    loss_improvement = result[1]
                    fitness_improvement = result[2]
                    genome_data = result[3] if len(result) > 3 else None

                    fitness_final      = self._loss_to_fitness(loss_final)
                    individual.fitness = fitness_final

                    # Apply genome updates if Lamarckian evolution (genome_data is not None)
                    if genome_data is not None:
                        # Update node parameters in main process genome
                        for node_id, (bias, gain) in genome_data['node_params'].items():
                            individual.genome.node_genes[node_id].bias = bias
                            individual.genome.node_genes[node_id].gain = gain

                        # Update connection parameters in main process genome
                        for innovation_num, weight in genome_data['conn_params'].items():
                            individual.genome.conn_genes[innovation_num].weight = weight

                        # CRITICAL FIX: Sync network parameters with updated genome
                        # In parallel mode, the network's internal arrays are still holding old parameters
                        # We need to reload them from the now-updated genome
                        individual._network.load_parameters_from_genome(enforce_bounds=True)
                    else:
                        # Baldwin effect: Also need to sync network with genome
                        # In parallel mode, gradient descent happened in worker process
                        # The main process network may have stale parameters
                        individual._network.load_parameters_from_genome(enforce_bounds=True)

                    # Store gradient data for this individual
                    self._gradient_data[individual.ID] = {
                        'fitness_before_gd'  : fitness_before_gd[individual.ID],
                        'fitness_after_gd'   : fitness_final,
                        'fitness_improvement': fitness_improvement,
                        'loss_before_gd'     : loss_final + loss_improvement,
                        'loss_after_gd'      : loss_final,
                        'loss_improvement'   : loss_improvement,
                        'generation'         : self._generation_counter
                    }

            # Update species fitness after gradient improvements
            self._population._species_manager.update_fitness()

    def _apply_gradient_descent(self, individual: "Individual") -> tuple[float, float, float]:
        """
        Apply gradient descent to optimize an individual's network parameters.

        Parameters:
            individual: The individual to optimize

        Returns:
            tuple: (final_loss, loss_improvement, fitness_improvement)
                - final_loss: Loss after gradient descent
                - loss_improvement: initial_loss - final_loss
                - fitness_improvement: final_fitness - initial_fitness
        """
        # Get network reference
        network = individual._network

        # Get training data
        inputs, targets = self._get_training_data()

        # Get current network parameters
        weights, biases, gains = network.get_parameters()

        # Flatten parameters for Adam optimizer
        flat_params = np.concatenate([weights.flatten(), biases.flatten(), gains.flatten()])
        shapes      = (weights.shape, biases.shape, gains.shape)
        w_size      = np.prod(shapes[0])
        b_size      = np.prod(shapes[1])

        # Loss function parameterized by flattened parameters
        # Note: iter parameter is required by adam optimizer but unused here
        def objective(flat_params, _iter=None):
            # Unflatten parameters
            w = flat_params[:w_size].reshape(shapes[0])
            b = flat_params[w_size:w_size + b_size].reshape(shapes[1])
            g = flat_params[w_size + b_size:].reshape(shapes[2])

            # Compute loss
            outputs = network.forward_pass(inputs, w, b, g)
            return self._loss_function(outputs, targets)

        # Track initial loss and fitness
        initial_loss = objective(flat_params)
        initial_fitness = self._loss_to_fitness(initial_loss)

        # Create gradient function and apply Adam optimizer
        grad_fn = grad(objective)
        optimized_params = adam(grad_fn, flat_params,
                                num_iters=self._config.gradient_steps,
                                step_size=self._config.learning_rate)

        # Unflatten optimized parameters
        weights = optimized_params[:w_size].reshape(shapes[0])
        biases  = optimized_params[w_size:w_size + b_size].reshape(shapes[1])
        gains   = optimized_params[w_size + b_size:].reshape(shapes[2])

        # Update network with optimized parameters
        network.set_parameters(weights, biases, gains, enforce_bounds=True)

        # Calculate final loss and fitness (using optimized parameters)
        final_loss = objective(optimized_params)
        final_fitness = self._loss_to_fitness(final_loss)

        # Calculate improvements
        loss_improvement = initial_loss - final_loss
        fitness_improvement = final_fitness - initial_fitness

        # Handle parameter inheritance based on evolution mode
        if self._config.lamarckian_evolution:
            # Lamarckian: Save optimized parameters to genome for inheritance
            network.save_parameters_to_genome(enforce_bounds=True)

            # Extract genome parameters to return to main process (for parallel execution)
            # This is necessary because in parallel mode, workers operate on copies
            # and these updates need to be transferred back to the main process
            genome_data = {
                'node_params': {},  # node_id -> (bias, gain)
                'conn_params': {}   # innovation_num -> weight
            }

            # Extract node parameters
            for node_id, node_gene in individual.genome.node_genes.items():
                genome_data['node_params'][node_id] = (node_gene.bias, node_gene.gain)

            # Extract connection parameters
            for innovation_num, conn_gene in individual.genome.conn_genes.items():
                if conn_gene.enabled:
                    genome_data['conn_params'][innovation_num] = conn_gene.weight

            return final_loss, loss_improvement, fitness_improvement, genome_data
        else:
            # Baldwin effect: Restore network to genome parameters
            # (fitness improvement used for selection, but learned parameters not inherited)
            network.load_parameters_from_genome(enforce_bounds=True)

            # For Baldwin effect, no genome updates needed
            return final_loss, loss_improvement, fitness_improvement, None

    def _select_individuals_for_gradient(self) -> list["Individual"]:
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

    def _report_gradient_statistics(self):
        """
        Report statistics about gradient descent performance.

        This method can be called from _generation_report() to
        display gradient training statistics.
        """
        if self._gradient_data:
            fitness_improvements = [data['fitness_improvement'] for data in self._gradient_data.values()]
            avg_fitness_improvement = np.mean(fitness_improvements)
            s = f"Avg fitness improvement due to gradient descent: {avg_fitness_improvement:.6f}\n"
            return s
        return ""

    def run(self, num_jobs: int = 1):
        """
        Run the trial with gradient descent support.

        Extends the base Trial.run() to add final champion optimization
        when using Baldwin effect. After evolution completes, if Baldwin
        effect was used, the champion gets one final gradient descent
        optimization and the parameters are saved.

        Parameters:
            num_jobs: Number of parallel processes for fitness evaluation
                      1 = serial (no parallelization)
                     -1 = use all available CPU cores
                      n = use n processes
        """
        # Run the standard NEAT evolution
        super().run(num_jobs)

        # After evolution completes, optimize the final champion if using Baldwin effect
        self._optimize_final_champion()

    def _optimize_final_champion(self):
        """
        Apply gradient descent to the final champion and save optimized parameters.

        This is called after evolution completes. When using Baldwin effect during
        evolution, the champion's fitness represents its learning potential but the
        network has unoptimized parameters. This method applies gradient descent
        one final time and saves the optimized parameters to both the network and
        genome, ensuring the champion performs as well as its fitness suggests.

        This gives us the best of both worlds:
        - Baldwin effect during evolution (no Lamarckian inheritance)
        - Optimized champion after evolution completes
        """
        if not self._config.enable_gradient:
            return  # No gradient descent enabled, nothing to do

        if self._config.lamarckian_evolution:
            return  # Lamarckian already optimized, nothing more to do

        champion = self._population.get_fittest_individual()
        if champion is None or champion.fitness is None:
            return  # No valid champion

        # Temporarily enable Lamarckian mode for the final optimization
        # This ensures the optimized parameters are kept
        original_lamarckian = self._config.lamarckian_evolution
        self._config.lamarckian_evolution = True

        # Apply gradient descent to optimize the champion
        result = self._apply_gradient_descent(champion)
        loss_final = result[0]
        loss_improvement = result[1]
        fitness_improvement = result[2]

        # Restore original Lamarckian setting
        self._config.lamarckian_evolution = original_lamarckian

        # Update champion's fitness with the optimized value
        fitness_final = self._loss_to_fitness(loss_final)
        champion.fitness = fitness_final

        if not self._suppress_output:
            print(f"\nFinal champion optimization (Baldwin → Optimized):")
            print(f"  Champion ID: {champion.ID}")
            print(f"  Fitness improvement: {fitness_improvement:.6f}")
            print(f"  Final fitness: {fitness_final:.4f}")
            print(f"  Note: Optimized parameters saved to champion's genome")