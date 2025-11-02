# EvoGrad

EvoGrad: A Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a genetic algorithm for evolving artificial neural networks.

## Features

- **Multiple Network Backends**: Choose from standard, fast (vectorized), or autograd-compatible implementations
- **Hybrid Optimization**: Combine NEAT with gradient descent for enhanced performance
- **Parallel Evaluation**: Built-in support for parallel fitness evaluation using joblib
- **Comprehensive Configuration**: Extensive configuration system with 60+ parameters
- **Innovation Tracking**: Maintains structural innovations across generations
- **Speciation**: Automatic species management to protect innovation

## Installation

### From PyPI (coming soon)

```bash
pip install evograd
```

### From Source

```bash
git clone https://github.com/yourusername/evograd.git
cd evograd
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/evograd.git
cd evograd
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from evograd import Config, Trial

class XORTrial(Trial):
    def _evaluate_fitness(self, individual):
        """Evaluate XOR performance."""
        xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        xor_outputs = [0, 1, 1, 0]

        error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            output = individual.forward(inputs)[0]
            error += (output - expected) ** 2

        return 4.0 - error  # Higher fitness is better

    def _reset(self):
        """Reset trial state."""
        self.best_fitness = -float('inf')

    def _report_progress(self):
        """Report generation progress."""
        if self.population.best_individual.fitness > self.best_fitness:
            self.best_fitness = self.population.best_individual.fitness
            print(f"Generation {self.generation}: Best fitness = {self.best_fitness:.4f}")

# Load configuration
config = Config("examples/configs/config_xor.ini")

# Run trial
trial = XORTrial(config, network_type='standard')
trial.run(num_jobs=4)  # Use 4 cores for parallel evaluation
```

### Running Multiple Experiments

```python
from evograd import Experiment

class XORExperiment(Experiment):
    def _prepare_trial(self, trial, trial_number):
        """Prepare each trial."""
        print(f"Starting trial {trial_number}")

    def _extract_trial_results(self, trial, trial_number):
        """Extract results from completed trial."""
        return {
            'best_fitness': trial.population.best_individual.fitness,
            'generations': trial.generation,
            'best_genome': trial.population.best_individual.genome
        }

    def _analyze_trial_results(self, results):
        """Analyze results across all trials."""
        avg_fitness = sum(r['best_fitness'] for r in results) / len(results)
        avg_generations = sum(r['generations'] for r in results) / len(results)
        print(f"Average fitness: {avg_fitness:.4f}")
        print(f"Average generations: {avg_generations:.1f}")

# Run 30 trials with parallelization
experiment = XORExperiment(
    XORTrial,
    num_trials=30,
    config=config,
    network_type='standard'
)
experiment.run(num_jobs_trials=4, num_jobs_fitness=2)
```

### Hybrid NEAT + Gradient Descent

```python
from evograd.run import TrialGrad

class RegressionTrialGrad(TrialGrad):
    def _evaluate_fitness(self, individual):
        """Evaluate regression performance."""
        # Your fitness evaluation
        pass

    def _compute_loss(self, individual):
        """Compute differentiable loss for gradient descent."""
        # Return autograd-compatible loss
        pass

# Requires autograd-compatible network
trial = RegressionTrialGrad(config, network_type='autograd')
trial.run()
```

## Configuration

Configuration files use INI format with the following sections:

- `[POPULATION_INIT]` - Population size, input/output dimensions, initial connectivity
- `[SPECIATION]` - Compatibility threshold, distance coefficients
- `[REPRODUCTION]` - Elitism, survival threshold, min species size
- `[STAGNATION]` - Max stagnation period, species elitism
- `[TERMINATION]` - Fitness criteria, max generations
- `[NODE]` - Node mutation parameters
- `[CONNECTION]` - Connection mutation parameters
- `[STRUCTURAL_MUTATIONS]` - Structural mutation probabilities
- `[GRADIENT_DESCENT]` - Optional gradient descent parameters

See `examples/configs/` directory for example configurations.

## Examples

The `examples/` directory contains:

- **XOR Problem** - Classic benchmark for evolving logic gates
- **Function Regression** - 1D and N-dimensional function approximation
- **CartPole** - Classic control problem from OpenAI Gym
- **Gymnasium Integration** - General framework for Gymnasium environments
- **Jupyter Notebooks** - Interactive tutorials and experiments

## Documentation

Full documentation is available at [Read the Docs](https://evograd.readthedocs.io/) (coming soon).

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=evograd --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this library in your research, please cite:

```bibtex
@software{evograd,
  author = {Your Name},
  title = {EvoGrad: A Python Implementation of NEAT with Gradient Descent Extensions},
  year = {2024},
  url = {https://github.com/yourusername/evograd}
}
```

Original NEAT paper:

```bibtex
@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002},
  publisher={MIT Press}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kenneth O. Stanley and Risto Miikkulainen for the original NEAT algorithm
- The Python community for excellent scientific computing libraries
- Contributors to this implementation