"""
Unit tests for Config class.
"""

import pytest
import os
from evograd.run.config import Config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config_dir():
    """Return the directory containing test configuration files."""
    return os.path.join(os.path.dirname(__file__), 'test_configs')


# ============================================================================
# Test Config Initialization
# ============================================================================

class TestConfigInit:
    """Test Config initialization."""

    def test_init_without_file_creates_empty_config(self):
        """Test that Config() without file creates empty config with bounds."""
        config = Config()

        assert config.min_weight == float('-inf')
        assert config.max_weight == float('inf')
        assert config.min_bias == float('-inf')
        assert config.max_bias == float('inf')
        assert config.min_gain == float('-inf')
        assert config.max_gain == float('inf')

    def test_init_with_nonexistent_file_raises_error(self):
        """Test that Config with nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file .* not found"):
            Config('nonexistent_file.ini')

    def test_init_with_minimal_config(self, test_config_dir):
        """Test initialization with minimal configuration file."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)

        # Spot check some values
        assert config.population_size == 100
        assert config.num_inputs == 2
        assert config.num_outputs == 1
        assert config.initial_cxn_policy == 'full'


# ============================================================================
# Test Config Population Init Section
# ============================================================================

class TestConfigPopulationInit:
    """Test Config POPULATION_INIT section parsing."""

    def test_population_size(self, test_config_dir):
        """Test population_size parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.population_size == 100

    def test_num_inputs(self, test_config_dir):
        """Test num_inputs parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.num_inputs == 2

    def test_num_outputs(self, test_config_dir):
        """Test num_outputs parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.num_outputs == 1

    def test_initial_cxn_policy(self, test_config_dir):
        """Test initial_cxn_policy parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.initial_cxn_policy == 'full'

    def test_initial_cxn_fraction_none(self, test_config_dir):
        """Test initial_cxn_fraction parsing when set to None."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.initial_cxn_fraction is None

    def test_initial_cxn_fraction_value(self, test_config_dir):
        """Test initial_cxn_fraction parsing with numeric value."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)
        assert config.initial_cxn_fraction == 0.5


# ============================================================================
# Test Config Speciation Section
# ============================================================================

class TestConfigSpeciation:
    """Test Config SPECIATION section parsing."""

    def test_compatibility_threshold(self, test_config_dir):
        """Test compatibility_threshold parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.compatibility_threshold == 3.0

    def test_distance_coefficients(self, test_config_dir):
        """Test distance coefficient parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.distance_excess_coeff == 1.0
        assert config.distance_disjoint_coeff == 1.0
        assert config.distance_params_coeff == 0.4

    def test_distance_includes_nodes(self, test_config_dir):
        """Test distance_includes_nodes parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.distance_includes_nodes is True

    def test_activation_distance_k_default(self, test_config_dir):
        """Test activation_distance_k with default value."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.activation_distance_k == 3.0

    def test_activation_distance_k_custom(self, test_config_dir):
        """Test activation_distance_k with custom value."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)
        assert config.activation_distance_k == 5.0


# ============================================================================
# Test Config Reproduction Section
# ============================================================================

class TestConfigReproduction:
    """Test Config REPRODUCTION section parsing."""

    def test_elitism(self, test_config_dir):
        """Test elitism parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.elitism == 2

    def test_survival_threshold(self, test_config_dir):
        """Test survival_threshold parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.survival_threshold == 0.2

    def test_min_species_size(self, test_config_dir):
        """Test min_species_size parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.min_species_size == 2

    def test_num_episodes(self, test_config_dir):
        """Test num_episodes_average parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.num_episodes == 1


# ============================================================================
# Test Config Stagnation Section
# ============================================================================

class TestConfigStagnation:
    """Test Config STAGNATION section parsing."""

    def test_max_stagnation_period(self, test_config_dir):
        """Test max_stagnation_period parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.max_stagnation_period == 15

    def test_species_elitism(self, test_config_dir):
        """Test species_elitism parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.species_elitism == 2


# ============================================================================
# Test Config Termination Section
# ============================================================================

class TestConfigTermination:
    """Test Config TERMINATION section parsing."""

    def test_fitness_termination_check_false(self, test_config_dir):
        """Test fitness_termination_check when False."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.fitness_termination_check is False

    def test_fitness_termination_check_true(self, test_config_dir):
        """Test fitness_termination_check when True."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)
        assert config.fitness_termination_check is True

    def test_fitness_criterion(self, test_config_dir):
        """Test fitness_criterion parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.fitness_criterion == 'max'

    def test_fitness_threshold(self, test_config_dir):
        """Test fitness_threshold parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.fitness_threshold == 100.0

    def test_max_number_generations(self, test_config_dir):
        """Test max_number_generations parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.max_number_generations == 100


# ============================================================================
# Test Config Node Section
# ============================================================================

class TestConfigNode:
    """Test Config NODE section parsing."""

    def test_activation_initial(self, test_config_dir):
        """Test activation_initial parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.activation_initial == 'sigmoid'

    def test_bias_init_params(self, test_config_dir):
        """Test bias initialization parameters."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.bias_init_mean == 0.0
        assert config.bias_init_stdev == 1.0

    def test_gain_init_params(self, test_config_dir):
        """Test gain initialization parameters."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.gain_init_mean == 1.0
        assert config.gain_init_stdev == 0.1

    def test_bias_bounds(self, test_config_dir):
        """Test bias min/max bounds."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.min_bias == -10.0
        assert config.max_bias == 10.0

    def test_gain_bounds(self, test_config_dir):
        """Test gain min/max bounds."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.min_gain == 0.1
        assert config.max_gain == 10.0

    def test_bias_mutation_probs(self, test_config_dir):
        """Test bias mutation probabilities."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.bias_replace_prob == 0.1
        assert config.bias_perturb_prob == 0.5

    def test_gain_mutation_probs(self, test_config_dir):
        """Test gain mutation probabilities."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.gain_replace_prob == 0.1
        assert config.gain_perturb_prob == 0.5

    def test_bias_perturb_strength(self, test_config_dir):
        """Test bias perturbation strength."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.bias_perturb_strength == 0.5

    def test_gain_perturb_strength(self, test_config_dir):
        """Test gain perturbation strength."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.gain_perturb_strength == 0.5

    def test_activation_mutate_prob_default(self, test_config_dir):
        """Test activation_mutate_prob with default value."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.activation_mutate_prob == 0.0

    def test_activation_mutate_prob_custom(self, test_config_dir):
        """Test activation_mutate_prob with custom value."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)
        assert config.activation_mutate_prob == 0.1

    def test_activation_options_all(self, test_config_dir):
        """Test activation_options='all' includes all activations."""
        config_file = os.path.join(test_config_dir, 'activation_options.ini')
        config = Config(config_file)
        assert 'sigmoid' in config.activation_options
        assert 'tanh' in config.activation_options
        assert 'legendre' in config.activation_options

    def test_activation_options_custom_list(self, test_config_dir):
        """Test activation_options with custom comma-separated list."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)
        assert config.activation_options == ['sigmoid', 'tanh', 'relu', 'legendre']

    def test_activation_options_fixed(self, test_config_dir):
        """Test activation_options='fixed' excludes legendre."""
        # Need to create a test file with activation_options=fixed
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[POPULATION_INIT]
population_size = 100
num_inputs = 2
num_outputs = 1
initial_cxn_policy = full
initial_cxn_fraction = None

[SPECIATION]
compatibility_threshold = 3.0
distance_excess_coeff = 1.0
distance_disjoint_coeff = 1.0
distance_params_coeff = 0.4
distance_includes_nodes = True

[REPRODUCTION]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
num_episodes_average = 1

[STAGNATION]
max_stagnation_period = 15
species_elitism = 2

[TERMINATION]
fitness_termination_check = False
fitness_criterion = max
fitness_threshold = 100.0
max_number_generations = 100

[NODE]
activation_initial = sigmoid
bias_init_mean = 0.0
bias_init_stdev = 1.0
gain_init_mean = 1.0
gain_init_stdev = 0.1
min_bias = -10.0
max_bias = 10.0
min_gain = 0.1
max_gain = 10.0
bias_replace_prob = 0.1
gain_replace_prob = 0.1
bias_perturb_prob = 0.5
gain_perturb_prob = 0.5
bias_perturb_strength = 0.5
gain_perturb_strength = 0.5
activation_mutate_prob = 0.1
activation_options = fixed

[CONNECTION]
weight_init_mean = 0.0
weight_init_stdev = 1.0
min_weight = -10.0
max_weight = 10.0
weight_replace_prob = 0.1
weight_perturb_prob = 0.5
weight_perturb_strength = 0.5

[STRUCTURAL_MUTATIONS]
single_structural_mutation = False
node_add_probability = 0.2
node_delete_probability = 0.0
connection_add_probability = 0.5
connection_enable_probability = 0.01
connection_disable_probability = 0.01
connection_delete_probability = 0.0
""")
            temp_file = f.name

        try:
            config = Config(temp_file)
            assert 'sigmoid' in config.activation_options
            assert 'legendre' not in config.activation_options
        finally:
            os.unlink(temp_file)

    def test_activation_options_invalid_raises_error(self, test_config_dir):
        """Test that invalid activation in options raises ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[POPULATION_INIT]
population_size = 100
num_inputs = 2
num_outputs = 1
initial_cxn_policy = full
initial_cxn_fraction = None

[SPECIATION]
compatibility_threshold = 3.0
distance_excess_coeff = 1.0
distance_disjoint_coeff = 1.0
distance_params_coeff = 0.4
distance_includes_nodes = True

[REPRODUCTION]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
num_episodes_average = 1

[STAGNATION]
max_stagnation_period = 15
species_elitism = 2

[TERMINATION]
fitness_termination_check = False
fitness_criterion = max
fitness_threshold = 100.0
max_number_generations = 100

[NODE]
activation_initial = sigmoid
bias_init_mean = 0.0
bias_init_stdev = 1.0
gain_init_mean = 1.0
gain_init_stdev = 0.1
min_bias = -10.0
max_bias = 10.0
min_gain = 0.1
max_gain = 10.0
bias_replace_prob = 0.1
gain_replace_prob = 0.1
bias_perturb_prob = 0.5
gain_perturb_prob = 0.5
bias_perturb_strength = 0.5
gain_perturb_strength = 0.5
activation_mutate_prob = 0.1
activation_options = sigmoid,invalid_activation

[CONNECTION]
weight_init_mean = 0.0
weight_init_stdev = 1.0
min_weight = -10.0
max_weight = 10.0
weight_replace_prob = 0.1
weight_perturb_prob = 0.5
weight_perturb_strength = 0.5

[STRUCTURAL_MUTATIONS]
single_structural_mutation = False
node_add_probability = 0.2
node_delete_probability = 0.0
connection_add_probability = 0.5
connection_enable_probability = 0.01
connection_disable_probability = 0.01
connection_delete_probability = 0.0
""")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid activation function"):
                Config(temp_file)
        finally:
            os.unlink(temp_file)


# ============================================================================
# Test Config Connection Section
# ============================================================================

class TestConfigConnection:
    """Test Config CONNECTION section parsing."""

    def test_weight_init_params(self, test_config_dir):
        """Test weight initialization parameters."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.weight_init_mean == 0.0
        assert config.weight_init_stdev == 1.0

    def test_weight_bounds(self, test_config_dir):
        """Test weight min/max bounds."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.min_weight == -10.0
        assert config.max_weight == 10.0

    def test_weight_mutation_probs(self, test_config_dir):
        """Test weight mutation probabilities."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.weight_replace_prob == 0.1
        assert config.weight_perturb_prob == 0.5

    def test_weight_perturb_strength(self, test_config_dir):
        """Test weight perturbation strength."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.weight_perturb_strength == 0.5


# ============================================================================
# Test Config Structural Mutations Section
# ============================================================================

class TestConfigStructuralMutations:
    """Test Config STRUCTURAL_MUTATIONS section parsing."""

    def test_single_structural_mutation(self, test_config_dir):
        """Test single_structural_mutation parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.single_structural_mutation is False

    def test_node_mutation_probs(self, test_config_dir):
        """Test node add/delete probabilities."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.node_add_probability == 0.2
        assert config.node_delete_probability == 0.0

    def test_connection_add_probability(self, test_config_dir):
        """Test connection_add_probability parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.connection_add_probability == 0.5

    def test_connection_enable_disable_probs(self, test_config_dir):
        """Test connection enable/disable probabilities."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.connection_enable_probability == 0.01
        assert config.connection_disable_probability == 0.01

    def test_connection_delete_probability(self, test_config_dir):
        """Test connection_delete_probability parsing."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)
        assert config.connection_delete_probability == 0.0


# ============================================================================
# Test Config Gradient Descent Section
# ============================================================================

class TestConfigGradientDescent:
    """Test Config GRADIENT_DESCENT section parsing."""

    def test_gradient_descent_defaults(self, test_config_dir):
        """Test gradient descent with default values (section missing)."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)

        assert config.enable_gradient is False
        assert config.gradient_steps == 10
        assert config.learning_rate == 0.01
        assert config.gradient_frequency == 1
        assert config.gradient_selection == 'top_k'
        assert config.gradient_top_k == 5
        assert config.gradient_top_percent == 0.1
        assert config.lamarckian_evolution is False
        assert config.freeze_weights is False
        assert config.freeze_biases is False
        assert config.freeze_gains is False
        assert config.freeze_activation_coeffs is False

    def test_gradient_descent_enabled(self, test_config_dir):
        """Test gradient descent with custom values."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)

        assert config.enable_gradient is True
        assert config.gradient_steps == 20
        assert config.learning_rate == 0.001
        assert config.gradient_frequency == 2
        assert config.gradient_selection == 'top_percent'
        assert config.gradient_top_k == 10
        assert config.gradient_top_percent == 0.2
        assert config.lamarckian_evolution is True
        assert config.freeze_weights is False
        assert config.freeze_biases is True
        assert config.freeze_gains is False
        assert config.freeze_activation_coeffs is True


# ============================================================================
# Test Config Legendre Section
# ============================================================================

class TestConfigLegendre:
    """Test Config LEGENDRE section parsing."""

    def test_legendre_defaults(self, test_config_dir):
        """Test Legendre with default values (section missing)."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)

        assert config.num_legendre_coeffs == 10
        assert config.legendre_coeffs_init_mean == 0.0
        assert config.legendre_coeffs_init_stdev == 1.0

    def test_legendre_custom_values(self, test_config_dir):
        """Test Legendre with custom values."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)

        assert config.num_legendre_coeffs == 15
        assert config.legendre_coeffs_init_mean == 0.1
        assert config.legendre_coeffs_init_stdev == 0.5


# ============================================================================
# Test Config Integration
# ============================================================================

class TestConfigIntegration:
    """Integration tests for Config."""

    def test_complete_config_parsing(self, test_config_dir):
        """Test complete configuration parsing with all sections."""
        config_file = os.path.join(test_config_dir, 'with_gradient.ini')
        config = Config(config_file)

        # Verify a few values from each section
        assert config.population_size == 50
        assert config.compatibility_threshold == 2.5
        assert config.elitism == 1
        assert config.max_stagnation_period == 10
        assert config.fitness_termination_check is True
        assert config.activation_initial == 'legendre'
        assert config.weight_init_mean == 0.5
        assert config.single_structural_mutation is True
        assert config.enable_gradient is True
        assert config.num_legendre_coeffs == 15

    def test_minimal_config_parsing(self, test_config_dir):
        """Test minimal configuration without optional sections."""
        config_file = os.path.join(test_config_dir, 'minimal.ini')
        config = Config(config_file)

        # Verify required sections are parsed
        assert config.population_size == 100
        assert config.num_inputs == 2

        # Verify optional sections have defaults
        assert config.enable_gradient is False
        assert config.num_legendre_coeffs == 10
