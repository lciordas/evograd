"""
Gymnasium Environment Base Module for NEAT

This module provides the base class for training NEAT agents on Gymnasium (formerly OpenAI Gym)
reinforcement learning environments. It handles environment interaction, episode management,
and data collection, allowing derived classes to focus on fitness calculation.

Classes:
    Trial_Gymnasium: Abstract base class for Gymnasium environment trials
"""

from pathlib import Path
import autograd.numpy as np   # type: ignore
import gymnasium as gym       # type: ignore

from neat.run.config import Config
from neat.phenotype  import Individual
from neat.run        import Trial

class Trial_Gymnasium(Trial):
    """
    Abstract base class for NEAT trials on Gymnasium environments.

    Handles environment interaction and data collection. Derived classes must implement
    '_evaluate_fitness()' to compute fitness from collected observations and rewards.

    Automatically configures network input/output dimensions based on the environment's
    observation and action spaces.
    """

    def __init__(self,
                 config         : Config,
                 environment    : str,
                 suppress_output: bool = False):
        """
        Initialize the trial.

        Parameters:
            config:          Configuration parameters
            environment:     An instance of the Gymnasium environment we want to solve
            suppress_output: Whether to suppress output during training
        """

        # Using a 'standard' network, since when interacting with a gym environment we
        # get serialized data (individual [action, observation] tuples), not batches.
        super().__init__(config, network_type='standard', suppress_output=suppress_output)

        self.env = environment

        # The number of inputs and outputs depends on the environment we're solving.
        # We get them from the environment object and use them to update the config object.
        self._config.num_inputs = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self._config.num_outputs = self.env.action_space.n
        else:
            self._config.num_outputs = self.env.action_space.shape[0]

    def _reset(self):
        """
        Reset the trial state before starting a new run.
        """
        return super()._reset()

    def _run_environment(self, individual: Individual) -> tuple[list[float], list[float]]:
        """
        Runs multiple episodes of the given gym environment.
        Gathers and returns all data generated; it provides
        the input to '_evaluate_fitness()'.

        Returns:
            a tuple containing two lists
            the first  list has the observations returned after each step
            the second list has the reward after each step
        """
        observations = {}  # episode number => list of observations after each step
        rewards      = {}  # episode number => list of rewards after each step

        for n in range(self._config.num_episodes):

            # Reset environment before starting new episode
            observation, _ = self.env.reset()

            # Create containers to store rewards and observations for this episode
            observations[n+1] = [observation]
            rewards[n+1]      = []

            # Run episode
            done = False
            while not done:

                # Choose action: select the output node with highest activation
                output = individual._network.forward_pass(observation.tolist())
                action = np.argmax(output)

                # Take action in environment
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Save data
                observations[n+1].append(observation)
                rewards[n+1].append(reward)

        return observations, rewards

    def _generation_report(self):
        """
        Print a report describing the current generation.
        """
        fittest = self._population.get_fittest_individual()
        fittest_pruned = fittest.prune()

        s  = f"===============\n"
        s += f"GENERATION {self._generation_counter:04d}\n"
        s += f"Maximum fitness  = {fittest.fitness:.2f}\n"
        s += f"Population size  = {len(self._population.individuals)}\n"
        s += f"Number species   = {len(self._population._species_manager.species)}\n"
        s += f"Network topology = {fittest_pruned._network.number_nodes} neurons, "
        s += f"{fittest_pruned._network.number_connections_enabled} connections\n"
        s += '\n'
        s += str(fittest_pruned)
        s += '\n'
        print(s)

    def _final_report(self):
        """
        Display results at the end of the trial.
        """
        individual_fittest = self._population.get_fittest_individual().prune()

        print("="*21)
        print("FINAL BEST INDIVIDUAL")
        print("="*21)
        print(individual_fittest)
        print(f"\nFinal fitness: {individual_fittest.fitness:.2f}")
        print(f"Network: {individual_fittest._network.number_nodes} neurons, "
              f"{individual_fittest._network.number_connections_enabled} connections")

        # Visualize the network
        try:
            individual_fittest._network.visualize()
            print("Network visualization saved as 'Digraph.gv.pdf'")
        except Exception as e:
            print(f"Could not visualize network: {e}")

    def _terminate(self) -> bool:
        """
        Determine whether the trial should terminate.
        """
        return super()._terminate()
