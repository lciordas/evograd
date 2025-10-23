# NEAT Documentation

Welcome to the documentation for the NEAT (NeuroEvolution of Augmenting Topologies) Python library.

## Quick Links

- [Getting Started](guides/getting_started.md)
- [API Reference](api/index.md)
- [Examples](examples/index.md)
- [Architecture Overview](architecture.md)

## What is NEAT?

NEAT is a genetic algorithm for evolving artificial neural networks. It was developed by Kenneth O. Stanley and Risto Miikkulainen in 2002. This implementation provides:

- Multiple network backends (standard, fast/vectorized, autograd-compatible)
- Hybrid optimization with gradient descent
- Parallel evaluation support
- Comprehensive configuration system
- Speciation and innovation tracking

## Installation

```bash
pip install neat-python
```

Or install from source:

```bash
git clone https://github.com/yourusername/neat-python.git
cd neat-python
pip install -e .
```

## Quick Start

```python
from neat import Config, Trial

class XORTrial(Trial):
    def _evaluate_fitness(self, individual):
        # Implement XOR logic evaluation
        pass

config = Config("config_xor.ini")
trial = XORTrial(config)
trial.run()
```