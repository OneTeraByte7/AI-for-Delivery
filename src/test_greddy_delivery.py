import numpy as np
from env import DeliveryFleetEnv

def greedy_policy(obs, env, agent_id):
    """Decide action based on a greedy policy."""
    agent_pos = env.agent_positions[agent_id]
    
    if env.agent_carrying[agent_id]:
        targets = env.deliveries
        
    else:
        targets = env.orders
        
    if not targets:
        return 0
    
    target = min(targets, key=lambda t: abs(t[0] - agent_pos[0]))