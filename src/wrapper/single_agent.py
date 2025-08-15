import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class SingleAgentWrapper(gym.env):
    """
    Wraps DeliveryFleetEnv to a single agent gym env that controls control_agent
    Other agents act randomly. Obseravtions = that agent's obs
    """
    
    metadata = {"render_modes":[]}
    
    def __init__(self, base_env_cls, env_kwargs=None, control_agent="agent_0"):
        super().__init__()
        env_kwargs = env_kwargs or {}
        self.base_env = base_env_cls(**env_kwargs)
        self.control_agent = control_agent
        self.actions_space = self.base_env.action_space(self.control_agent)
        self.obsevations_space = self.base_env.observations+spaces(self.control_agent)
        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.base_env.reset()
        return obs[self.control_agent], {}
    
    def step(self, actions):
        
        actions = {}
        for agent in self.base_env.agents:
            if agent == self.control_agent:
                actions[agent] = int(actions[agent])
            else:
                actions[agent] = np.random.randint(0, 5)
                
        obs, rewards, dones, infos = self.base_env.steo(actions)
        terminated = dones.get(self.control_agent, False)
        truncated = False
        info = infos.get(self.control_agent, {})
        return obs[self.control_agent], rewards[self.control_agent], terminated, truncated, info
    
    def render(self):
        self.base_env.render()
        
    def close(self):
        pass