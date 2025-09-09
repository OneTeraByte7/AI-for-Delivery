import gymnasium as gym
import numpy as np


class SingleAgentWrapper(gym.Env):
    """
    Wraps a multi-agent env (DeliveryFleetEnv) into a single-agent Gym environment.
    Only the `control_agent` is controlled; other agents act randomly.
    Compatible with Stable-Baselines3 (PPO).
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env_cls, env_kwargs=None, control_agent="agent_0", max_episode_steps=100):
        super().__init__()
        env_kwargs = env_kwargs or {}
        self.base_env = base_env_cls(**env_kwargs)
        self.control_agent = control_agent
        self._t = 0
        self.max_episode_steps = max_episode_steps

        # ----- Observation space -----
        obs, _ = self.base_env.reset()  # unpack properly
        sample_obs = np.array(obs[self.control_agent], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones_like(sample_obs, dtype=np.float32),
            high=np.inf * np.ones_like(sample_obs, dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Action space -----
        act_space = getattr(self.base_env, "action_spaces", {}).get(self.control_agent, None)
        if isinstance(act_space, int):
            self.action_space = gym.spaces.Discrete(act_space)
        elif act_space is not None:
            self.action_space = act_space
        else:
            try:
                self.action_space = self.base_env.action_space(self.control_agent)
            except Exception:
                self.action_space = gym.spaces.Discrete(5)

    def reset(self, *, seed=None, options=None, **kwargs):
        self._t = 0
        obs, info = self.base_env.reset(seed=seed, options=options, **kwargs)
        return np.array(obs[self.control_agent], dtype=np.float32), info.get(self.control_agent, {})

    def step(self, action):
        self._t += 1

        # Build full action dict (random for other agents)
        actions = {}
        for agent in self.base_env.agents:
            if agent == self.control_agent:
                actions[agent] = int(action)
            else:
                space = getattr(self.base_env, "action_spaces", {}).get(agent, None)
                if isinstance(space, int):
                    actions[agent] = np.random.randint(0, space)
                elif space is not None:
                    actions[agent] = space.sample()
                else:
                    actions[agent] = np.random.randint(0, 5)

        obs, rewards, terminateds, truncateds, infos = self.base_env.step(actions)

        terminated = terminateds.get(self.control_agent, False)
        truncated = truncateds.get(self.control_agent, False) or (self._t >= self.max_episode_steps)

        info = infos.get(self.control_agent, {})

        return (
            np.array(obs[self.control_agent], dtype=np.float32),
            float(rewards[self.control_agent]),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
