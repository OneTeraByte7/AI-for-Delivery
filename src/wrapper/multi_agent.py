import gymnasium as gym
import numpy as np
#trains multiple agents
class MultiAgentWrapper(gym.Env):
    """
    Wraps a multi-agent environment (DeliveryFleetEnv) into a single Gym environment
    for multi-agent PPO training. Observations are concatenated for all agents.
    """
    metadata = {"render_modes": []}
    
    def __init__ (self, base_env_cls, env_kwargs=None):
        super().init__()
        env_kwargs = env_kwargs or {}
        self.base_env = base_env_cls(**env_kwargs)
        self.agents = self.base_env.agents
        
        
        sample_obs = np.concantenate(
            [np.array(self.base_env.reset()[agent], dtype=np.float32) for agent in self.agents]
        )
        
        self.observations_space = gym.spaces.Box(
            low =- np.inf * np.ones_like(sample_obs, dtype = np.float32),
            high = np.inf * np.ones_like(sample_obs, dtype = np.float32),
            dtype = np.float32,
        )
        
        max_actions = max([self.base_env.action_spaces[agent] for agent in self.agents])
        self.action_space = gym.spaces.MultiDiscrete([max_actions for _ in self.agents])
        
    def reset(self, *, seed=None, options=None, **kwargs):
        obs_dict = self.base_env.reset()
        obs = np.concatenate([np.array(obs_dict[agent], dtype=np.float32) for agent in self.agents])
        return obs, {}
    
    def step(self, action):
        """
        `action` is a list/array of actions for all agents in order of self.agents
        """
        actions = {agent: int(a) for agent, a in zip(self.agents, action)}
        obs_dict, rewards_dict, dones_dict, infos_dict = self.base_env.step(actions)

        obs = np.concatenate([np.array(obs_dict[agent], dtype=np.float32) for agent in self.agents])
        reward = sum(rewards_dict.values())  # global reward (can also use individual)
        terminated = all(dones_dict.values())
        truncated = False
        info = infos_dict

        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()
        
        
        #for better working