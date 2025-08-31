import numpy as np
from env import DeliveryFleetEnv

def greedy_policy(obs, env, agent_id):
    """Decide action based on a greedy policy."""
    agents_pos = env.agent_position[agent_id]