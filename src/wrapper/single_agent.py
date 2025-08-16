import gymnasium as gym
import numpy as np

class SingleAgentWrapper(gym.Env):
    """
    Wraps a multi-agent env (DeliveryFleetEnv) into a single-agent Gym environment.
    Only the `control_agent` is controlled; other agents act randomly.
    Compatible with Stable-Baselines3 (PPO).
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env_cls, env_kwargs=None, control_agent="agent_0"):
        super().__init__()
        env_kwargs = env_kwargs or {}
        self.base_env = base_env_cls(**env_kwargs)
        self.control_agent = control_agent

        # ----- Observation space: sample a real observation -----
        sample_obs = np.array(self.base_env.reset()[self.control_agent], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones_like(sample_obs, dtype=np.float32),
            high=np.inf * np.ones_like(sample_obs, dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Action space -----
        act_space = self.base_env.action_spaces[self.control_agent]
        if isinstance(act_space, int):
            self.action_space = gym.spaces.Discrete(act_space)
        else:
            self.action_space = act_space

    def reset(self, *, seed=None, options=None, **kwargs):
        # ignore seed/options if base_env does not support them
        obs = self.base_env.reset()
        return np.array(obs[self.control_agent], dtype=np.float32), {}

    def step(self, action):
        # Build full action dict
        actions = {}
        for agent in self.base_env.agents:
            if agent == self.control_agent:
                actions[agent] = int(action)
            else:
                # Sample a random valid action for other agents
                space = self.base_env.action_spaces[agent]
                if isinstance(space, int):
                    actions[agent] = np.random.randint(0, space)
                else:
                    actions[agent] = space.sample()

        obs, rewards, dones, infos = self.base_env.step(actions)

        terminated = dones.get(self.control_agent, False)
        truncated = False  # Can implement max-step truncation if needed

        return (
            np.array(obs[self.control_agent], dtype=np.float32),
            rewards[self.control_agent],
            terminated,
            truncated,
            infos.get(self.control_agent, {}),
        )

    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()
