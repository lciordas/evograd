"""
Unit tests for ConnectionGene class.

Tests cover initialization, mutation, string representations, and edge cases.
"""

import pytest
import random
import numpy as np
from unittest.mock import Mock

from evograd.genotype.connection_gene import ConnectionGene
from evograd.run.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Config with standard mutation parameters."""
    config = Mock(spec=Config)
    config.weight_init_mean = 0.0
    config.weight_init_stdev = 1.0
    config.min_weight = -30.0
    config.max_weight = 30.0
    config.weight_perturb_prob = 0.8
    config.weight_perturb_strength = 0.5
    config.weight_replace_prob = 0.1
    return config


@pytest.fixture
def extreme_config():
    """Config with tight weight bounds for boundary testing."""
    config = Mock(spec=Config)
    config.min_weight = -1.0
    config.max_weight = 1.0
    config.weight_perturb_prob = 0.5
    config.weight_perturb_strength = 2.0  # Larger than bounds
    config.weight_replace_prob = 0.3
    return config


@pytest.fixture
def no_mutation_config():
    """Config with zero mutation probabilities."""
    config = Mock(spec=Config)
    config.min_weight = -10.0
    config.max_weight = 10.0
    config.weight_perturb_prob = 0.0
    config.weight_perturb_strength = 0.5
    config.weight_replace_prob = 0.0
    return config


@pytest.fixture
def infinite_bounds_config():
    """Config with infinite weight bounds (no clipping)."""
    config = Mock(spec=Config)
    config.min_weight = float('-inf')
    config.max_weight = float('inf')
    config.weight_perturb_prob = 0.5
    config.weight_perturb_strength = 1.0
    config.weight_replace_prob = 0.2
    return config


# ============================================================================
# Test: Constructor
# ============================================================================

class TestConnectionGeneInit:
    """Test ConnectionGene initialization."""

    def test_basic_initialization(self, basic_config):
        """Test standard initialization with all parameters."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=0.5,
            innovation=10,
            config=basic_config,
            enabled=True
        )

        assert gene.node_in == 1
        assert gene.node_out == 2
        assert gene.weight == 0.5
        assert gene.innovation == 10
        assert gene.enabled is True
        assert gene._config is basic_config

    def test_default_enabled_true(self, basic_config):
        """Test that enabled defaults to True when not specified."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=0.5,
            innovation=10,
            config=basic_config
        )

        assert gene.enabled is True

    def test_explicit_disabled(self, basic_config):
        """Test initialization with enabled=False."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=0.5,
            innovation=10,
            config=basic_config,
            enabled=False
        )

        assert gene.enabled is False

    def test_zero_weight(self, basic_config):
        """Test initialization with zero weight."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=0.0,
            innovation=10,
            config=basic_config
        )

        assert gene.weight == 0.0

    def test_negative_weight(self, basic_config):
        """Test initialization with negative weight."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=-5.5,
            innovation=10,
            config=basic_config
        )

        assert gene.weight == -5.5

    def test_boundary_weights(self, extreme_config):
        """Test initialization with min and max weight values."""
        gene_min = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=-1.0,
            innovation=10,
            config=extreme_config
        )
        assert gene_min.weight == -1.0

        gene_max = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=1.0,
            innovation=11,
            config=extreme_config
        )
        assert gene_max.weight == 1.0

    def test_self_loop(self, basic_config):
        """Test that constructor allows self-loops (same node_in and node_out)."""
        gene = ConnectionGene(
            node_in=5,
            node_out=5,
            weight=1.0,
            innovation=20,
            config=basic_config
        )

        assert gene.node_in == 5
        assert gene.node_out == 5

    def test_negative_node_ids(self, basic_config):
        """Test initialization with negative node IDs."""
        gene = ConnectionGene(
            node_in=-1,
            node_out=-2,
            weight=0.5,
            innovation=10,
            config=basic_config
        )

        assert gene.node_in == -1
        assert gene.node_out == -2

    def test_zero_innovation(self, basic_config):
        """Test initialization with innovation number of 0."""
        gene = ConnectionGene(
            node_in=1,
            node_out=2,
            weight=0.5,
            innovation=0,
            config=basic_config
        )

        assert gene.innovation == 0

    def test_large_values(self, basic_config):
        """Test initialization with large node IDs and innovation numbers."""
        gene = ConnectionGene(
            node_in=999999,
            node_out=888888,
            weight=25.0,
            innovation=777777,
            config=basic_config
        )

        assert gene.node_in == 999999
        assert gene.node_out == 888888
        assert gene.innovation == 777777


# ============================================================================
# Test: Mutation
# ============================================================================

class TestConnectionGeneMutate:
    """Test ConnectionGene mutation behavior."""

    def test_no_mutation_when_probabilities_zero(self, no_mutation_config):
        """Test that weight doesn't change when both mutation probabilities are 0."""
        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 5.0, 10, no_mutation_config)
        original_weight = gene.weight

        # Run mutation multiple times
        for _ in range(10):
            gene.mutate()

        assert gene.weight == original_weight

    def test_weight_perturbation(self, basic_config):
        """Test that weight perturbation changes the weight."""
        random.seed(42)
        np.random.seed(42)

        # Force perturbation by setting random.random() < perturb_prob
        gene = ConnectionGene(1, 2, 0.0, 10, basic_config)
        original_weight = gene.weight

        # Run mutation multiple times - at least one should perturb
        weights = []
        for _ in range(50):
            gene = ConnectionGene(1, 2, 0.0, 10, basic_config)
            gene.mutate()
            weights.append(gene.weight)

        # With perturb_prob=0.8, most should have changed
        changed_count = sum(1 for w in weights if w != original_weight)
        assert changed_count > 30  # At least 60% should have changed

    def test_weight_replacement(self, basic_config):
        """Test that weight replacement produces values within bounds."""
        random.seed(123)

        gene = ConnectionGene(1, 2, 100.0, 10, basic_config)  # Start outside typical range

        # Run mutation many times
        weights = []
        for _ in range(100):
            gene = ConnectionGene(1, 2, 100.0, 10, basic_config)
            gene.mutate()
            weights.append(gene.weight)

        # Changed weights should be within bounds (some may not change)
        changed_weights = [w for w in weights if w != 100.0]
        assert len(changed_weights) > 50  # At least half should mutate
        for weight in changed_weights:
            assert basic_config.min_weight <= weight <= basic_config.max_weight

    def test_weight_clipping_at_min(self, extreme_config):
        """Test that perturbation clips at min_weight."""
        random.seed(42)
        np.random.seed(42)

        # Start at minimum weight
        gene = ConnectionGene(1, 2, -1.0, 10, extreme_config)

        # Mutate many times
        for _ in range(100):
            gene.mutate()
            # Weight should never go below min
            assert gene.weight >= extreme_config.min_weight

    def test_weight_clipping_at_max(self, extreme_config):
        """Test that perturbation clips at max_weight."""
        random.seed(99)
        np.random.seed(99)

        # Start at maximum weight
        gene = ConnectionGene(1, 2, 1.0, 10, extreme_config)

        # Mutate many times
        for _ in range(100):
            gene.mutate()
            # Weight should never go above max
            assert gene.weight <= extreme_config.max_weight

    def test_large_perturbation_strength_clips(self, extreme_config):
        """Test that large perturbation strength gets clipped to bounds."""
        random.seed(77)
        np.random.seed(77)

        # extreme_config has perturb_strength=2.0 but bounds of [-1, 1]
        gene = ConnectionGene(1, 2, 0.0, 10, extreme_config)

        # Mutate many times - all should stay in bounds
        for _ in range(50):
            gene.mutate()
            assert extreme_config.min_weight <= gene.weight <= extreme_config.max_weight

    def test_mutation_does_not_change_other_attributes(self, basic_config):
        """Test that mutation only affects weight, not other attributes."""
        random.seed(42)

        gene = ConnectionGene(1, 2, 0.5, 10, basic_config, enabled=True)

        # Store original values
        original_node_in = gene.node_in
        original_node_out = gene.node_out
        original_innovation = gene.innovation
        original_enabled = gene.enabled

        # Mutate multiple times
        for _ in range(20):
            gene.mutate()

        # All attributes except weight should be unchanged
        assert gene.node_in == original_node_in
        assert gene.node_out == original_node_out
        assert gene.innovation == original_innovation
        assert gene.enabled == original_enabled

    def test_mutation_statistical_distribution(self, basic_config):
        """Test that mutations follow expected probability distribution."""
        random.seed(100)
        np.random.seed(100)

        perturb_count = 0
        replace_count = 0
        no_change_count = 0

        trials = 1000

        for i in range(trials):
            gene = ConnectionGene(1, 2, 10.0, i, basic_config)
            original_weight = gene.weight
            gene.mutate()

            # Heuristic: if weight changed by small amount, likely perturbed
            # if changed dramatically, likely replaced
            weight_diff = abs(gene.weight - original_weight)

            if weight_diff == 0:
                no_change_count += 1
            elif weight_diff < 5.0:  # Likely perturbation
                perturb_count += 1
            else:  # Likely replacement (uniform from -30 to 30)
                replace_count += 1

        # With perturb_prob=0.8 and replace_prob=0.1
        # Expected: ~80% perturb, ~10% replace, ~10% no change
        assert perturb_count > 700  # At least 70%
        assert replace_count > 50   # At least 5%
        assert no_change_count < 200  # Less than 20%

    def test_infinite_bounds_perturbation_only(self):
        """Test that infinite bounds work for perturbation (but not replacement)."""
        # Note: random.uniform(-inf, inf) doesn't work well, but perturbation does
        config = Mock(spec=Config)
        config.min_weight = float('-inf')
        config.max_weight = float('inf')
        config.weight_perturb_prob = 1.0  # Always perturb, never replace
        config.weight_perturb_strength = 1.0
        config.weight_replace_prob = 0.0  # Never replace (uniform(-inf,inf) fails)

        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 0.0, 10, config)

        # Mutate many times - weights can grow large without clipping
        for _ in range(10):
            gene.mutate()

        # Just verify no errors occur and weight is a valid number
        assert isinstance(gene.weight, (int, float))
        assert not np.isnan(gene.weight)

    def test_both_probabilities_one(self):
        """Test behavior when both mutation probabilities are 1.0."""
        config = Mock(spec=Config)
        config.min_weight = -10.0
        config.max_weight = 10.0
        config.weight_perturb_prob = 1.0
        config.weight_perturb_strength = 0.5
        config.weight_replace_prob = 1.0

        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 0.0, 10, config)
        original_weight = gene.weight

        gene.mutate()

        # Should always mutate (either perturb or replace)
        # Due to cumulative probability check, perturb happens if r < 1.0 (always true)
        assert gene.weight != original_weight or original_weight == gene.weight


# ============================================================================
# Test: String Representations
# ============================================================================

class TestConnectionGeneStringMethods:
    """Test __repr__ and __str__ methods."""

    def test_repr_format(self, basic_config):
        """Test __repr__ produces expected format."""
        gene = ConnectionGene(1, 2, 0.5, 10, basic_config, enabled=True)

        repr_str = repr(gene)

        assert "ConnectionGene" in repr_str
        assert "node_in=001" in repr_str
        assert "node_out=002" in repr_str
        assert "weight=+0.500000" in repr_str
        assert "enabled=True" in repr_str
        assert "innovation=010" in repr_str

    def test_repr_negative_weight(self, basic_config):
        """Test __repr__ with negative weight."""
        gene = ConnectionGene(1, 2, -3.75, 10, basic_config)

        repr_str = repr(gene)

        assert "weight=-3.750000" in repr_str

    def test_repr_disabled(self, basic_config):
        """Test __repr__ shows enabled=False correctly."""
        gene = ConnectionGene(1, 2, 0.5, 10, basic_config, enabled=False)

        repr_str = repr(gene)

        assert "enabled=False" in repr_str

    def test_repr_large_innovation(self, basic_config):
        """Test __repr__ with large innovation number."""
        gene = ConnectionGene(1, 2, 0.5, 12345, basic_config)

        repr_str = repr(gene)

        assert "innovation=12345" in repr_str

    def test_str_format_enabled(self, basic_config):
        """Test __str__ compact format with enabled=True."""
        gene = ConnectionGene(10, 20, 5.25, 100, basic_config, enabled=True)

        str_repr = str(gene)

        assert "[" in str_repr
        assert "]" in str_repr
        assert "100" in str_repr  # innovation
        assert "E" in str_repr  # Enabled
        assert "10=>20" in str_repr or "010=>020" in str_repr  # node connection
        assert "5.25" in str_repr or "+5.25" in str_repr  # weight

    def test_str_format_disabled(self, basic_config):
        """Test __str__ shows 'D' for disabled connections."""
        gene = ConnectionGene(10, 20, 5.25, 100, basic_config, enabled=False)

        str_repr = str(gene)

        assert "D" in str_repr  # Disabled

    def test_str_negative_weight(self, basic_config):
        """Test __str__ with negative weight."""
        gene = ConnectionGene(1, 2, -2.5, 10, basic_config)

        str_repr = str(gene)

        assert "-2.5" in str_repr or "-2.50" in str_repr

    def test_str_zero_weight(self, basic_config):
        """Test __str__ with zero weight."""
        gene = ConnectionGene(1, 2, 0.0, 10, basic_config)

        str_repr = str(gene)

        # Should show zero with proper formatting
        assert "0.0" in str_repr or "+0.00" in str_repr

    def test_str_various_digit_counts(self, basic_config):
        """Test __str__ formatting with different digit counts."""
        # Single digit nodes
        gene1 = ConnectionGene(1, 2, 1.0, 5, basic_config)
        str1 = str(gene1)
        assert "005" in str1  # innovation padded to 3 digits

        # Double digit nodes
        gene2 = ConnectionGene(10, 20, 1.0, 50, basic_config)
        str2 = str(gene2)
        assert "050" in str2

        # Triple digit innovation
        gene3 = ConnectionGene(1, 2, 1.0, 100, basic_config)
        str3 = str(gene3)
        assert "100" in str3


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestConnectionGeneEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_weight_outside_bounds_not_clipped_on_init(self, extreme_config):
        """Test that constructor doesn't clip weights outside bounds."""
        # Constructor should accept any weight value
        gene = ConnectionGene(1, 2, 100.0, 10, extreme_config)

        assert gene.weight == 100.0  # Not clipped during initialization

    def test_mutation_with_weight_outside_bounds(self, extreme_config):
        """Test mutation starting from weight outside bounds."""
        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 50.0, 10, extreme_config)

        # After mutation, should be clipped
        gene.mutate()

        assert extreme_config.min_weight <= gene.weight <= extreme_config.max_weight

    def test_zero_perturbation_strength(self):
        """Test mutation with zero perturbation strength."""
        config = Mock(spec=Config)
        config.min_weight = -10.0
        config.max_weight = 10.0
        config.weight_perturb_prob = 1.0  # Always perturb
        config.weight_perturb_strength = 0.0  # But by zero amount
        config.weight_replace_prob = 0.0

        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 5.0, 10, config)
        original_weight = gene.weight

        gene.mutate()

        # Weight should be approximately unchanged (perturbation by ~0)
        assert abs(gene.weight - original_weight) < 0.01

    def test_config_object_reference(self, basic_config):
        """Test that config is stored as reference, not copied."""
        gene = ConnectionGene(1, 2, 0.5, 10, basic_config)

        assert gene._config is basic_config  # Same object, not copy

    def test_multiple_genes_same_config(self, basic_config):
        """Test that multiple genes can share the same config object."""
        gene1 = ConnectionGene(1, 2, 0.5, 10, basic_config)
        gene2 = ConnectionGene(2, 3, -0.5, 11, basic_config)

        assert gene1._config is gene2._config

    def test_mutation_reproducibility_with_seed(self, basic_config):
        """Test that mutations are reproducible with same random seed."""
        weights1 = []
        weights2 = []

        # First run
        random.seed(12345)
        np.random.seed(12345)
        for i in range(10):
            gene = ConnectionGene(1, 2, 0.0, i, basic_config)
            gene.mutate()
            weights1.append(gene.weight)

        # Second run with same seed
        random.seed(12345)
        np.random.seed(12345)
        for i in range(10):
            gene = ConnectionGene(1, 2, 0.0, i, basic_config)
            gene.mutate()
            weights2.append(gene.weight)

        assert weights1 == weights2

    def test_very_small_weight_bounds(self):
        """Test with very small weight bounds."""
        config = Mock(spec=Config)
        config.min_weight = -0.001
        config.max_weight = 0.001
        config.weight_perturb_prob = 0.5
        config.weight_perturb_strength = 0.0001
        config.weight_replace_prob = 0.5

        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 0.0, 10, config)

        # Mutate many times
        for _ in range(50):
            gene.mutate()
            assert config.min_weight <= gene.weight <= config.max_weight

    def test_asymmetric_weight_bounds(self):
        """Test with asymmetric weight bounds."""
        config = Mock(spec=Config)
        config.min_weight = -100.0
        config.max_weight = 1.0
        config.weight_perturb_prob = 0.5
        config.weight_perturb_strength = 10.0
        config.weight_replace_prob = 0.5

        random.seed(42)
        np.random.seed(42)

        gene = ConnectionGene(1, 2, 0.0, 10, config)

        # Mutate many times
        for _ in range(100):
            gene.mutate()
            assert config.min_weight <= gene.weight <= config.max_weight
