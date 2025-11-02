# EvoGrad Documentation

Welcome to the documentation for EvoGrad, a Python implementation of NEAT (NeuroEvolution of Augmenting Topologies) with gradient descent extensions.

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
pip install evograd
```

Or install from source:

```bash
git clone https://github.com/yourusername/evograd.git
cd evograd
pip install -e .
```

## Quick Start

```python
from evograd import Config, Trial

class XORTrial(Trial):
    def _evaluate_fitness(self, individual):
        # Implement XOR logic evaluation
        pass

config = Config("config_xor.ini")
trial = XORTrial(config)
trial.run()
```