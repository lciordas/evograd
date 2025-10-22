import configparser
import os

class Config:

    def __init__(self, config_file: str | None = None):
        """
        Initialize Config by parsing an INI file, or create an empty Config.

        Parameters:
            config_file: Path to the INI configuration file.
                         If None, creates an empty Config for manual attribute setting.
        """
        if config_file is None:
            # Empty config for testing/manual setup
            return

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")

        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Sentinel for missing default values
        _NO_DEFAULT = object()

        # Helper function to safely parse values
        def get_value(section, key, value_type, default=_NO_DEFAULT):
            try:
                raw_value = parser.get(section, key)
                if raw_value.lower() == 'none':
                    return None
                if value_type == int:
                    return parser.getint(section, key)
                elif value_type == float:
                    return parser.getfloat(section, key)
                elif value_type == bool:
                    return parser.getboolean(section, key)
                elif value_type == str:
                    return raw_value
            except (configparser.NoSectionError, configparser.NoOptionError):
                if default is not _NO_DEFAULT:
                    return default
                raise

        # [POPULATION INIT]

        # The number of individuals in each generation.
        self.population_size = get_value('POPULATION_INIT', 'population_size', int)

        # The number of input nodes, through which the network receives inputs.
        self.num_inputs = get_value('POPULATION_INIT', 'num_inputs', int)

        # The number of output nodes, to which the network delivers outputs.
        self.num_outputs = get_value('POPULATION_INIT', 'num_outputs', int)

        # Specifies the initial connectivity of newly-created networks.
        # Allowed values:
        #   "none"      - no connections are initially present
        #   "one-input" - one random input node is connected to all outputs nodes
        #   "partial"   - a fraction of all possible connections are instantiate randomly
        #   "full"      - connect all input nodes to all output nodes
        self.initial_cxn_policy = get_value('POPULATION_INIT', 'initial_cxn_policy', str)

        # The fraction of connections to instantiate (only applicable
        # if the initial connection policy is "partial").
        # Use "None" if not applicable.
        self.initial_cxn_fraction = get_value('POPULATION_INIT', 'initial_cxn_fraction', float)

        # The activation function assigned to each node.
        # For the list of all available choices, see the 'activation' module.
        self.activation = get_value('POPULATION_INIT', 'activation', str)

        # [SPECIATION]

        # Individuals whose genomic distance is less than this
        # threshold are considered to be in the same species.
        self.compatibility_threshold = get_value('SPECIATION', 'compatibility_threshold', float)

        # The coefficient for the excess gene counts'
        # contribution to the genomic distance.
        # Usually set equal to 'distance_disjoint_coeff'.
        self.distance_excess_coeff = get_value('SPECIATION', 'distance_excess_coeff', float)

        # The coefficient for the disjoint gene counts'
        # contribution to the genomic distance.
        # Usually set equal to 'distance_excess_coeff'.
        self.distance_disjoint_coeff = get_value('SPECIATION', 'distance_disjoint_coeff', float)

        # The coefficient for scalar parameter (connection weight,
        # node bias or gain) difference's contribution to the genomic
        # distance (for homologous nodes or connections).
        self.distance_params_coeff = get_value('SPECIATION', 'distance_params_coeff', float)

        # Whether to include in the genomic distance the contribution
        # coming from the difference in parameters of homologous nodes.
        self.distance_includes_nodes = get_value('SPECIATION', 'distance_includes_nodes', bool)

        # [REPRODUCTION]

        # The number of most-fit individuals in each species that
        # will be preserved as-is from one generation to the next.
        self.elitism = get_value('REPRODUCTION', 'elitism', int)

        # The fraction of individuals allowed to reproduce in each species.
        self.survival_threshold = get_value('REPRODUCTION', 'survival_threshold', float)

        # The minimum number of individual per species after reproduction.
        self.min_species_size = get_value('REPRODUCTION', 'min_species_size', int)

        # Number of episodes to average over for fitness evaluation.
        # N/A in this case, as we are solving a deterministic problem.
        self.num_episodes = get_value('REPRODUCTION', 'num_episodes_average', int)

        # [STAGNATION]

        # Species that have not shown improvement in more than this
        # number of generations will be considered stagnant and removed.
        self.max_stagnation_period = get_value('STAGNATION', 'max_stagnation_period', int)

        # The number of species that will be protected from stagnation.
        self.species_elitism = get_value('STAGNATION', 'species_elitism', int)

        # [TERMINATION]

        # Whether to use the fitness of the most recent
        # generation as a criterion for stopping the run.
        self.fitness_termination_check = get_value('TERMINATION', 'fitness_termination_check', bool)

        # The function used to compute the termination criterion.
        # Only applicable if 'fitness_termination_check' is 'True'.
        # Allowed values:
        #   "mean" calculate the mean fitness across the entire population
        #   "max"  get the fitness of the fittest individual in the population
        self.fitness_criterion = get_value('TERMINATION', 'fitness_criterion', str)

        # The fitness value which when met or exceeded causes the run to end.
        # Only applicable if 'fitness_termination_check' is 'True'.
        # The value to be compared with this threshold is calculated according
        # to 'fitness_criterion'.
        self.fitness_threshold = get_value('TERMINATION', 'fitness_threshold', float)

        # The number of generations after which to stop the run.
        # If 'fitness_termination_check' is 'True', the run may stop sooner.
        self.max_number_generations = get_value('TERMINATION', 'max_number_generations', int)

        # [NODE]

        # The mean and standard deviation of the normal distributions
        # used to initialize the 'bias' & 'gain' parameters for new nodes.
        self.bias_init_mean  = get_value('NODE', 'bias_init_mean' , float)
        self.bias_init_stdev = get_value('NODE', 'bias_init_stdev', float)
        self.gain_init_mean  = get_value('NODE', 'gain_init_mean' , float)
        self.gain_init_stdev = get_value('NODE', 'gain_init_stdev', float)

        # The minimum and maximum allowed 'bias' and 'gain' values.
        # Biases outside this range will be clamped to this range.
        self.min_bias = get_value('NODE', 'min_bias', float)
        self.max_bias = get_value('NODE', 'max_bias', float)
        self.min_gain = get_value('NODE', 'min_gain', float)
        self.max_gain = get_value('NODE', 'max_gain', float)

        # The probability that mutation will replace the 'bias' and 'gain' of
        # a node with a newly chosen random value (as if it were a new node).
        self.bias_replace_prob = get_value('NODE', 'bias_replace_prob', float)
        self.gain_replace_prob = get_value('NODE', 'gain_replace_prob', float)

        # The probability that mutation will change the 'bias' and 'gain'
        # of a node by adding a random value.
        self.bias_perturb_prob = get_value('NODE', 'bias_perturb_prob', float)
        self.gain_perturb_prob = get_value('NODE', 'gain_perturb_prob', float)

        # The standard deviation of the zero-centered normal distributions
        # from which a 'bias' and a 'gain' perturbation value is drawn.
        self.bias_perturb_strength = get_value('NODE', 'bias_perturb_strength', float)
        self.gain_perturb_strength = get_value('NODE', 'gain_perturb_strength', float)

        # [CONNECTION]

        # The mean and standard deviation of the normal distribution
        # used to initialize the 'weight' parameter for new connections.
        self.weight_init_mean  = get_value('CONNECTION', 'weight_init_mean' , float)
        self.weight_init_stdev = get_value('CONNECTION', 'weight_init_stdev', float)

        # The minimum and maximum allowed 'weight' values.
        # Biases outside this range will be clamped to this range.
        self.min_weight = get_value('CONNECTION', 'min_weight', float)
        self.max_weight = get_value('CONNECTION', 'max_weight', float)

        # The probability that mutation will replace the 'weight' of a connection
        # with a newly chosen random value (as if it were a new connection).
        self.weight_replace_prob = get_value('CONNECTION', 'weight_replace_prob', float)

        # The probability that mutation will change the 'weight'
        # of a connection by adding a random value.
        self.weight_perturb_prob = get_value('CONNECTION', 'weight_perturb_prob', float)

        # The standard deviation of the zero-centered normal distribution
        # from which a 'weight' perturbation value is drawn.
        self.weight_perturb_strength = get_value('CONNECTION', 'weight_perturb_strength', float)

        # [STRUCTURAL MUTATIONS]

        # If this is 'True', only one structural mutation (the addition or removal
        # of a node or connection) will be allowed per genome per generation.
        self.single_structural_mutation = get_value('STRUCTURAL_MUTATIONS', 'single_structural_mutation', bool)

        # The probability that mutation will add a new node (essentially replacing
        # an existing connection, the enabled status of which will be set to False).
        self.node_add_probability = get_value('STRUCTURAL_MUTATIONS', 'node_add_probability', float)

        # The probability that mutation will delete an existing node (and all connections to it).
        # This goes a bit against the original NEAT philosophy (start with a small network and
        # continuously grow it, so keep its value to 0 unless you have a good reason not to).
        self.node_delete_probability = get_value('STRUCTURAL_MUTATIONS', 'node_delete_probability', float)

        # The probability that mutation will add a connection between existing nodes
        self.connection_add_probability = get_value('STRUCTURAL_MUTATIONS', 'connection_add_probability', float)

        # The probability that a mutation will enabled a currently
        # disabled connection, or the other way around.
        self.connection_enable_probability  = get_value('STRUCTURAL_MUTATIONS', 'connection_enable_probability' , float)
        self.connection_disable_probability = get_value('STRUCTURAL_MUTATIONS', 'connection_disable_probability', float)

        # The probability that mutation will delete an existing connection (irrespective of
        # whether currently enabled or disabled).
        # This goes a bit against the original NEAT approach, which disables rather than delete
        # connections, so keep its value to 0 unless you have a good reason not to).
        self.connection_delete_probability = get_value('STRUCTURAL_MUTATIONS', 'connection_delete_probability', float)

        # [GRADIENT DESCENT] (optional section)

        # Whether to apply gradient descent optimization to individuals.
        self.enable_gradient = get_value('GRADIENT_DESCENT', 'enable_gradient', bool, default=False)

        # Number of gradient descent steps per application.
        self.gradient_steps = get_value('GRADIENT_DESCENT', 'gradient_steps', int, default=10)

        # Learning rate for gradient updates (step size for Adam optimizer).
        self.learning_rate = get_value('GRADIENT_DESCENT', 'learning_rate', float, default=0.01)

        # Apply gradients every N generations (1 = every generation).
        self.gradient_frequency = get_value('GRADIENT_DESCENT', 'gradient_frequency', int, default=1)

        # Which individuals to train with gradient descent.
        # Allowed values: 'all', 'top_k', 'top_percent'
        self.gradient_selection = get_value('GRADIENT_DESCENT', 'gradient_selection', str, default='top_k')

        # Number of top individuals to train (only applicable if gradient_selection='top_k').
        self.gradient_top_k = get_value('GRADIENT_DESCENT', 'gradient_top_k', int, default=5)

        # Percentage of top individuals to train (only applicable if gradient_selection='top_percent').
        self.gradient_top_percent = get_value('GRADIENT_DESCENT', 'gradient_top_percent', float, default=0.1)
    