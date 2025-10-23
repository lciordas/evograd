#!/usr/bin/env python3
"""
Utility script to visualize NEAT neural networks.

Usage:
    python scripts/visualize_network.py --genome saved_genome.pkl
"""

import sys
import argparse
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neat.genotype import Genome
from neat.phenotype import Individual
import graphviz


def visualize_genome(genome, output_file='network', format='png', view=True):
    """
    Visualize a NEAT genome as a neural network graph.

    Args:
        genome: The genome to visualize
        output_file: Output filename (without extension)
        format: Output format (png, pdf, svg, etc.)
        view: Whether to automatically open the generated file
    """
    dot = graphviz.Digraph(format=format, engine='dot')
    dot.attr('node', shape='circle')

    # Add nodes
    for node_id, node in genome.nodes.items():
        if node.node_type == 'input':
            dot.node(str(node_id), label=f'In{node_id}', color='green', style='filled')
        elif node.node_type == 'output':
            dot.node(str(node_id), label=f'Out{node_id}', color='red', style='filled')
        else:
            dot.node(str(node_id), label=str(node_id), color='lightblue', style='filled')

    # Add connections
    for conn in genome.connections.values():
        if conn.enabled:
            weight = conn.weight
            color = 'blue' if weight > 0 else 'red'
            penwidth = str(min(abs(weight) * 2, 5))
            dot.edge(str(conn.in_node), str(conn.out_node),
                    label=f'{weight:.2f}', color=color, penwidth=penwidth)

    dot.render(output_file, view=view)
    print(f"Network visualization saved to {output_file}.{format}")


def main():
    parser = argparse.ArgumentParser(description='Visualize NEAT neural networks')
    parser.add_argument('--genome', type=str, required=True,
                        help='Path to pickled genome file')
    parser.add_argument('--output', type=str, default='network',
                        help='Output filename (without extension)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format')
    parser.add_argument('--no-view', action='store_true',
                        help='Do not automatically open the generated file')

    args = parser.parse_args()

    # Load genome
    with open(args.genome, 'rb') as f:
        genome = pickle.load(f)

    if not isinstance(genome, Genome):
        print("Error: Loaded object is not a Genome instance")
        sys.exit(1)

    # Visualize
    visualize_genome(genome, args.output, args.format, not args.no_view)


if __name__ == '__main__':
    main()