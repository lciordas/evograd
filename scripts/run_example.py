#!/usr/bin/env python3
"""
Utility script to run NEAT examples easily.

Usage:
    python scripts/run_example.py xor
    python scripts/run_example.py regression1d
    python scripts/run_example.py cartpole
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evograd import Config
from examples.trial_XOR import XORTrial, XORExperiment
from examples.trial_regression1D import RegressionTrial1D, RegressionExperiment1D
from examples.trial_cartpole import CartPoleTrial, CartPoleExperiment


EXAMPLES = {
    'xor': {
        'trial': XORTrial,
        'experiment': XORExperiment,
        'config': 'examples/configs/config_xor.ini',
        'description': 'XOR logic problem'
    },
    'regression1d': {
        'trial': RegressionTrial1D,
        'experiment': RegressionExperiment1D,
        'config': 'examples/configs/config_regression1D.ini',
        'description': '1D function regression'
    },
    'cartpole': {
        'trial': CartPoleTrial,
        'experiment': CartPoleExperiment,
        'config': 'examples/configs/config_gymnasium.ini',
        'description': 'CartPole control problem'
    }
}


def main():
    parser = argparse.ArgumentParser(description='Run NEAT examples')
    parser.add_argument('example', choices=EXAMPLES.keys(),
                        help='Example to run')
    parser.add_argument('--mode', choices=['trial', 'experiment'], default='trial',
                        help='Run single trial or full experiment')
    parser.add_argument('--network-type', choices=['standard', 'fast', 'autograd'],
                        default='standard', help='Network implementation to use')
    parser.add_argument('--num-trials', type=int, default=30,
                        help='Number of trials for experiment mode')
    parser.add_argument('--num-jobs', type=int, default=4,
                        help='Number of parallel jobs')

    args = parser.parse_args()

    example = EXAMPLES[args.example]
    print(f"Running {example['description']}...")
    print(f"Mode: {args.mode}")
    print(f"Network type: {args.network_type}")

    config = Config(example['config'])

    if args.mode == 'trial':
        trial = example['trial'](config, network_type=args.network_type)
        trial.run(num_jobs=args.num_jobs)
        print(f"\nBest fitness: {trial.population.best_individual.fitness:.4f}")
    else:
        experiment = example['experiment'](
            example['trial'],
            num_trials=args.num_trials,
            config=config,
            network_type=args.network_type
        )
        experiment.run(num_jobs_trials=args.num_jobs, num_jobs_fitness=1)


if __name__ == '__main__':
    main()