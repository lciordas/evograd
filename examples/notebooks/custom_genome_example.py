"""
Example: Creating Custom Genomes from Dictionary Descriptions

This example demonstrates how to use the Genome.from_dict() class method
to programmatically create genomes with specific structures.
"""

from neat.genotype.genome import Genome

# Example 1: XOR-like network with two hidden nodes
# Note: InnovationTracker is automatically initialized by from_dict()
print("="*60)
print("Example 1: XOR-like network")
print("="*60)

xor_network = {
    "activation": "sigmoid",
    "nodes": [
        # Input nodes: must be numbered [0, num_inputs)
        {"id": 0, "type": "input"},
        {"id": 1, "type": "input"},

        # Output nodes: must be numbered [num_inputs, num_inputs + num_outputs)
        {"id": 2, "type": "output", "bias": 0.0, "gain": 1.0},

        # Hidden nodes: must be numbered >= num_inputs + num_outputs
        {"id": 3, "type": "hidden", "bias": 0.5, "gain": 1.0},
        {"id": 4, "type": "hidden", "bias": -0.5, "gain": 1.0}
    ],
    "connections": [
        {"from": 0, "to": 3, "weight": 0.5},
        {"from": 1, "to": 3, "weight": -0.3},
        {"from": 0, "to": 4, "weight": 0.8},
        {"from": 1, "to": 4, "weight": 0.2},
        {"from": 3, "to": 2, "weight": 1.5},
        {"from": 4, "to": 2, "weight": -1.2}
    ]
}

genome1 = Genome.from_dict(xor_network)
print(f"Created genome with:")
print(f"  - {len(genome1.input_nodes)} inputs")
print(f"  - {len(genome1.output_nodes)} outputs")
print(f"  - {len(genome1.hidden_nodes)} hidden nodes")
print(f"  - {len(genome1.conn_genes)} connections")
print(f"\n{genome1}\n")

# Example 2: Simple feedforward network (no hidden nodes)
print("="*60)
print("Example 2: Simple feedforward network")
print("="*60)

simple_network = {
    "activation": "relu",
    "nodes": [
        {"id": 0, "type": "input"},
        {"id": 1, "type": "input"},
        {"id": 2, "type": "input"},
        {"id": 3, "type": "output", "bias": 0.5, "gain": 1.0},
        {"id": 4, "type": "output", "bias": -0.5, "gain": 1.0}
    ],
    "connections": [
        {"from": 0, "to": 3, "weight": 0.7},
        {"from": 1, "to": 3, "weight": -0.3},
        {"from": 2, "to": 3, "weight": 0.9},
        {"from": 0, "to": 4, "weight": -0.5},
        {"from": 1, "to": 4, "weight": 0.8},
        {"from": 2, "to": 4, "weight": 0.2}
    ]
}

genome2 = Genome.from_dict(simple_network)
print(f"Created genome with:")
print(f"  - {len(genome2.input_nodes)} inputs")
print(f"  - {len(genome2.output_nodes)} outputs")
print(f"  - {len(genome2.hidden_nodes)} hidden nodes")
print(f"  - {len(genome2.conn_genes)} connections")
print(f"\n{genome2}\n")

# Example 3: Network with disabled connections
print("="*60)
print("Example 3: Network with disabled connections")
print("="*60)

network_with_disabled = {
    "activation": "tanh",
    "nodes": [
        {"id": 0, "type": "input"},
        {"id": 1, "type": "input"},
        {"id": 2, "type": "output"}
    ],
    "connections": [
        {"from": 0, "to": 2, "weight": 1.0, "enabled": True},
        {"from": 1, "to": 2, "weight": -0.5, "enabled": False}  # Disabled connection
    ]
}

genome3 = Genome.from_dict(network_with_disabled)
print(f"Created genome with:")
print(f"  - {len(genome3.conn_genes)} total connections")
print(f"  - {sum(1 for c in genome3.conn_genes.values() if c.enabled)} enabled")
print(f"  - {sum(1 for c in genome3.conn_genes.values() if not c.enabled)} disabled")
print(f"\n{genome3}\n")

print("="*60)
print("Key features of Genome.from_dict():")
print("="*60)
print("✓ Automatically initializes InnovationTracker (no manual setup needed)")
print("✓ Infers num_inputs and num_outputs from node types")
print("✓ Validates node numbering follows NEAT conventions")
print("✓ Validates network is acyclic (no loops)")
print("✓ Supports multiple activation function names")
print("✓ Requires exact node type strings: 'input', 'output', 'hidden'")
print("✓ Allows specifying custom bias and gain for nodes")
print("✓ Allows enabling/disabling connections")
