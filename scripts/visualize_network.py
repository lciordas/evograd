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

from evograd.genotype import Genome
from evograd.genotype.node_gene import NodeType
from evograd.phenotype import Individual
from evograd.activations import activation_codes
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
    for node_id, node in genome.node_genes.items():
        if node.type == NodeType.INPUT:
            dot.node(str(node_id), label=f'In{node_id}', color='green', style='filled')
        elif node.type == NodeType.OUTPUT:
            # Get activation code for output nodes
            act_code = activation_codes.get(node.activation_name, '???')
            label = f'Out{node_id}\\n{act_code}'
            dot.node(str(node_id), label=label, color='red', style='filled')
        else:
            # Get activation code for hidden nodes
            act_code = activation_codes.get(node.activation_name, '???')
            label = f'{node_id}\\n{act_code}'
            dot.node(str(node_id), label=label, color='lightblue', style='filled')

    # Add connections
    for conn in genome.conn_genes.values():
        if conn.enabled:
            weight = conn.weight
            color = 'blue' if weight > 0 else 'red'
            penwidth = str(min(abs(weight) * 2, 5))
            dot.edge(str(conn.node_in), str(conn.node_out),
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