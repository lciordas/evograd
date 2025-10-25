# Claude Code Guidelines

This document contains guidelines and conventions for working on this project with Claude Code.

## Project Overview

This is a Python implementation of **NEAT** (NeuroEvolution of Augmenting Topologies), an evolutionary algorithm for evolving neural network topologies and parameters.

### What is NEAT?

NEAT is a genetic algorithm for evolving artificial neural networks. Unlike traditional approaches that evolve only the connection weights of fixed-topology networks, NEAT evolves both the topology (structure) and the connection weights simultaneously.

Key features of NEAT:
- **Starts minimal**: Begins with simple networks (just inputs and outputs, no hidden nodes)
- **Grows complexity**: Adds nodes and connections through mutation
- **Historical markings**: Tracks structural innovations via innovation numbers
- **Speciation**: Protects topological innovations by grouping similar genomes
- **Crossover-friendly**: Innovation numbers enable meaningful crossover between different topologies

### This Implementation

This codebase provides a comprehensive NEAT implementation with several enhancements:

#### Core Architecture

- **Genotype/Phenotype separation**: Clean separation between genome representation and network execution
- **Multiple network backends**:
  - `NetworkStandard`: Object-oriented implementation (basic, processes one input at a time)
  - `NetworkFast`: High-performance vectorized numpy (for batch processing)
  - `NetworkAutograd`: Autograd-compatible functional implementation (for gradient-based optimization)

#### Key Components

- **Genome** (`genotype/genome.py`): Represents network structure as node genes and connection genes
- **Population** (`pool/population.py`): Top-level evolutionary coordinator managing individuals and generations
- **SpeciesManager** (`pool/species_manager.py`): Handles speciation to protect innovation
- **Trial** (`run/trial.py`): Abstract base class for running NEAT experiments
- **Config** (`run/config.py`): Configuration management via INI files

#### Advanced Features

1. **Hybrid Optimization** (`TrialGrad`): Combines NEAT topology evolution with gradient descent for parameter optimization
   - NEAT evolves network topology (structure)
   - Gradient descent (Adam optimizer) fine-tunes weights, biases, and gains
   - Configurable application frequency and individual selection

2. **Lamarckian Evolution**: Optional mode where gradient-learned parameters are written back to the genome and can be inherited by offspring (vs. Baldwin effect where learning only affects fitness)

3. **Parallel Fitness Evaluation**: CPU-based parallelization using joblib for efficient evaluation

4. **Pruning**: Remove dead-end nodes that don't contribute to network output

5. **Visualization**: Graphviz integration for network structure visualization

### Typical Workflow

1. Define a `Trial` subclass that implements problem-specific fitness evaluation
2. Create a configuration file (`.ini`) with NEAT parameters
3. Instantiate and run the trial
4. NEAT evolves a population of networks over generations
5. Retrieve the fittest individual when convergence is reached

### Network Types Explained

- **`standard`**: Use when processing single inputs, basic implementation
- **`fast`**: Use when data comes in batches, high-performance vectorized operations
- **`autograd`**: Use when you need gradients for hybrid NEAT+gradient-descent optimization

---

## Git Conventions

* NEVER mention Claude, Anthropic, or AI assistance in commit messages
* NEVER include AI-generated footers or co-authorship attributions
* Write commits as technical changes only - no AI attributions
* NEVER commit without being explicitly instructed to
* Format: `<type>: <description>` (e.g., `feat: add mutation operator`)
* Focus on what changed and why, not how it was created

---

## Project-Specific Guidelines

(This section will be populated as we establish project conventions)
