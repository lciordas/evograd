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

from abc      import abstractmethod
from joblib   import Parallel, delayed
from autograd import grad    # type: ignore
import autograd.numpy as np  # type: ignore
from typing   import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotype import Individual
from run.trial  import Trial
from run.config import Config

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

    Gradient Training Configuration:
        enable_gradient:      Whether to use gradient descent (default: False)
        gradient_steps:       Number of gradient descent steps per generation
        learning_rate:        Learning rate for gradient updates
        gradient_frequency:   Apply gradients every N generations (1 = every generation)
        gradient_selection:   Which individuals to train ('all', 'top_k', 'top_percent')
        gradient_top_k:       Number of top individuals to train (if selection='top_k')
        gradient_top_percent: Percentage of top individuals (if selection='top_percent')

    Public Methods (inherited from Trial):
        run(): Execute a complete NEAT trial with optional gradient descent
    """

    def __init__(self,
                 config              : Config,
                 network_type        : str,
                 suppress_output     : bool  = False,
                 enable_gradient     : bool  = False,
                 gradient_steps      : int   = 10,
                 learning_rate       : float = 0.01,
                 gradient_frequency  : int   = 1,
                 gradient_selection  : str   = 'top_k',
                 gradient_top_k      : int   = 5,
                 gradient_top_percent: float = 0.1):
        """
        Initialize the gradient-enabled trial.

        Parameters:
            config:               Configuration parameters
            network_type:         Type of network backend ('autograd' required for gradients)
            suppress_output:      If True, suppress progress and final reports
            enable_gradient:      Whether to apply gradient descent
            gradient_steps:       Number of gradient descent steps per application
            learning_rate:        Learning rate for gradient updates
            gradient_frequency:   Apply gradients every N generations
            gradient_selection:   How to select individuals ('all', 'top_k', 'top_percent')
            gradient_top_k:       Number of top individuals to train
            gradient_top_percent: Percentage of top individuals to train
        """
        super().__init__(config, network_type, suppress_output)

        # Validate network type for gradient support
        if enable_gradient and network_type != 'autograd':
            raise ValueError("Gradient descent requires network_type='autograd'")

        # Store gradient training parameters
        self._enable_gradient      = enable_gradient
        self._gradient_steps       = gradient_steps
        self._learning_rate        = learning_rate
        self._gradient_frequency   = gradient_frequency
        self._gradient_selection   = gradient_selection
        self._gradient_top_k       = gradient_top_k
        self._gradient_top_percent = gradient_top_percent

        # Track fitness improvements from gradients
        self._gradient_improvements = []

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
        self._gradient_improvements = []

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
        do_gradient_descent = (self._enable_gradient and
                               self._generation_counter > 0 and
                               self._generation_counter % self._gradient_frequency == 0)
        selected_individuals = self._select_individuals_for_gradient() if do_gradient_descent else []
        do_gradient_descent  = do_gradient_descent and selected_individuals

        if do_gradient_descent:
            serialize = (num_jobs == 1)
            if serialize:
                for individual in selected_individuals:
                    loss, improvement  = self._apply_gradient_descent(individual)
                    individual.fitness = self._loss_to_fitness(loss)
                    self._gradient_improvements.append(improvement)
            else:
                results = \
                    Parallel(num_jobs)(delayed(self._apply_gradient_descent)(i) for i in selected_individuals)
                for individual, (loss, improvement) in zip(selected_individuals, results):
                    individual.fitness = self._loss_to_fitness(loss)
                    self._gradient_improvements.append(improvement)

            # Update species fitness after gradient improvements
            self._population._species_manager.update_fitness()

    def _apply_gradient_descent(self, individual: "Individual") -> tuple[float, float]:
        """
        Apply gradient descent to optimize an individual's network parameters.

        Parameters:
            individual: The individual to optimize

        Returns: 
            tuple: (final loss, improvement := initial_loss - final_loss)
        """
        # Get network reference
        network = individual._network

        # Get training data
        inputs, targets = self._get_training_data()

        # Create loss function parameterized by weights, biases, gains
        # This allows autograd to compute gradients with respect to these parameters
        def loss_fn(weights, biases, gains):
            outputs = network.forward_pass(inputs, weights, biases, gains)
            return self._loss_function(outputs, targets)

        # Get current network parameters
        weights, biases, gains = network.get_parameters()

        # Track initial loss for statistics
        initial_loss = loss_fn(weights, biases, gains)

        # Create gradient function using autograd
        grad_fn = grad(loss_fn, argnum=[0, 1, 2])

        # Perform gradient descent steps
        for _ in range(self._gradient_steps):

            # Compute gradients
            grad_w, grad_b, grad_g = grad_fn(weights, biases, gains)

            # Update parameters (basic SGD, can be extended to other optimizers)
            weights = weights - self._learning_rate * grad_w
            biases  = biases  - self._learning_rate * grad_b
            gains   = gains   - self._learning_rate * grad_g

        # Update network with optimized parameters (with bounds enforcement)
        network.set_parameters(weights, biases, gains, enforce_bounds=True)

        # Calculate and return final loss and improvement
        final_loss  = loss_fn(weights, biases, gains)
        improvement = initial_loss - final_loss
        return final_loss, improvement

    def _select_individuals_for_gradient(self) -> list[Individual]:
        """
        Select which individuals should receive gradient descent training.

        Returns:
            List of individuals to train with gradient descent
        """
        individuals = self._population.individuals

        if self._gradient_selection == 'all':
            return individuals

        elif self._gradient_selection == 'top_k':
            sorted_individuals = sorted(individuals,
                                       key=lambda x: x.fitness,
                                       reverse=True)
            return sorted_individuals[:self._gradient_top_k]

        elif self._gradient_selection == 'top_percent':
            sorted_individuals = sorted(individuals,
                                       key=lambda x: x.fitness,
                                       reverse=True)
            num_to_select = max(1, int(len(individuals) * self._gradient_top_percent))
            return sorted_individuals[:num_to_select]

        else:
            raise ValueError(f"Unknown gradient_selection: {self._gradient_selection}")

    def _report_gradient_statistics(self):
        """
        Report statistics about gradient descent performance.

        This method can be called from _report_progress() to display
        gradient training statistics.
        """
        if self._gradient_improvements:
            avg_improvement = np.mean(self._gradient_improvements[-10:])  # Last 10
            total_improvement = np.sum(self._gradient_improvements)

            s  = f"  Gradient Descent Statistics:\n"
            s += f"    Recent avg improvement: {avg_improvement:.6f}\n"
            s += f"    Total improvement:      {total_improvement:.6f}\n"
            s += f"    Gradient steps applied: {len(self._gradient_improvements)}\n"

            return s
        return ""