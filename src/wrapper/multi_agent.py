import gymnasium as gym
import numpy as np

class MultiAgentWrapper(gym.Env):
    """
    Wraps a multi-agent environment (DeliveryFleetEnv) into a single Gym environment
    for multi-agent PPO training. Observations are concatenated for all agents.
    """
    metadata = {"render_modes": []}