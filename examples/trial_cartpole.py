"""
CartPole Problem Implementation for NEAT

This module implements the classic CartPole balancing task from Gymnasium as a NEAT trial.
The goal is to evolve neural networks that can balance a pole on a moving cart by applying
left or right forces.

The CartPole Problem:
    The agent controls a cart that moves along a frictionless track. A pole is attached
    to the cart via an un-actuated joint. The agent must balance the pole by moving the
    cart left or right.

    State Space (4 continuous values):
        - Cart position: [-2.4, 2.4]
        - Cart velocity: [-∞, ∞]
        - Pole angle: [-0.209, 0.209] radians (~12 degrees)
        - Pole angular velocity: [-∞, ∞]

    Action Space (2 discrete actions):
        0 - Push cart to the left
        1 - Push cart to the right

    Termination Conditions:
        - Pole angle exceeds ±12 degrees
        - Cart position exceeds ±2.4
        - Episode length exceeds 500 timesteps

Fitness Function:
    Fitness = avg(total_reward - position_penalty_coeff * |final_position|)

    - total_reward: Number of timesteps the pole remained balanced
    - position_penalty: Penalizes ending far from center (encourages stability)
    - Averaged over multiple episodes for robustness

Expected Solution:
    Successful agents achieve 500 timesteps per episode (maximum reward).
    NEAT typically evolves minimal networks with 1-2 hidden nodes.

Classes:
    Trial_CartPole: NEAT trial for the CartPole balancing task

Usage:
    Single Trial:
        config = Config("config_cartpole.ini")
        trial = Trial_CartPole(config)
        trial.run(num_jobs=-1)

    With Custom Configuration:
        config = Config("config_cartpole.ini")
        config.num_episodes = 5  # Test on 5 episodes
        trial = Trial_CartPole(config, suppress_output=False)
        trial.run(num_jobs=1)
"""

import gymnasium as gym    # type: ignore
from statistics import mean

from run.config import Config
from examples.trial_gymbase   import Trial_Gymnasium
from phenotype import Individual

class Trial_CartPole(Trial_Gymnasium):
    """
    NEAT trial for the CartPole-v1 balancing task.

    Evolves networks that balance a pole on a moving cart. Fitness combines
    episode duration (reward) with a penalty for ending far from center.
    """

    def __init__(self,
                 config         : Config,
                 suppress_output: bool = False):
        """
        Initialize the trial.

        Parameters:
            config:          Configuration parameters
            suppress_output: Whether to suppress output during training
        """
        environment = gym.make("CartPole-v1")
        super().__init__(config, environment, suppress_output)

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        The fitness of an Individual has two components:
        + the total reward returned by the environment - this 
          is the number of time steps the pole stayed vertical 
        + a penalty proportional to the absolute distance from
          the center at the end of the episode
        
        Note that we could use stronger rewards shaping by penalizing
        the agent at each step (proportional to the distance from the
        center at that time) - however that does not seem to change
        things in a significant way.

        The fitness is the average of this quantity over multiple episodes.                  
        """

        # Run the environment for multiple episodes and gather data
        observations, rewards = self._run_environment(individual)

        # Calculate the total reward for each episode
        total_rewards = {n:sum(rs) for n,rs in rewards.items()}

        # Extract final position for each episode.
        # NOTE: for the CartPole, an observation is a tuple consisting of:
        # (cart position, car velocity, pole angle, pole angular velocity)
        # The cart position is between -2.4 and +2.4.
        final_pos = {n:pos[-1][0] for n,pos in observations.items()}

        # Calculate for each episode the total reward adjusted 
        # by the deviation from the center
        rewards_adj = [total_rewards[n] - self._config.position_penalty_coeff * abs(final_pos[n]) for n in total_rewards]

        # Fitness is the average of adjusted total reward over multiple episodes
        fitness = max(0, mean(rewards_adj))
        return fitness
        