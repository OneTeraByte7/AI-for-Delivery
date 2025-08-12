# src/env.py

from pettingzoo.utils.env import ParallelEnv
import numpy as np

class DeliveryFleetEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=10, num_agents=3):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agent_positions = {}
        self.action_spaces = {agent: 5 for agent in self.agents}
        self.observation_spaces = {agent: (2,) for agent in self.agents}
        self.reset()

        
    def reset(self):
        # Assign random unique starting positions
        positions = set()
        for agent in self.agents:
            while True:
                pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
                if pos not in positions:
                    self.agent_positions[agent] = pos
                    positions.add(pos)
                    break
        observations = self._get_obs()
        return observations
    
    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            obs[agent] = np.array([x, y], dtype=np.int32)
        return obs
    
    def step(self, actions):
        rewards = {}
        dones = {}
        infos = {}
        
        new_positions = {}
        for agent, action in actions.items():
            x, y = self.agent_positions[agent]
            if action == 1 and y > 0:
                y -= 1
            elif action == 2 and y < self.grid_size - 1:
                y += 1
            elif action == 3 and x > 0:
                x -= 1
            elif action == 4 and x < self.grid_size - 1:
                x += 1
            # else action 0 (stay) or invalid -> no move
            new_positions[agent] = (x, y)
        
        # Collision check: block moves where multiple agents want the same spot
        pos_counts = {}
        for pos in new_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        for agent, pos in new_positions.items():
            if pos_counts[pos] > 1:
                # collision: revert to old pos and penalize
                new_positions[agent] = self.agent_positions[agent]
                rewards[agent] = -1
            else:
                rewards[agent] = 0
        
        self.agent_positions = new_positions
        
        dones = {agent: False for agent in self.agents}
        dones['__all__'] = False
        
        observations = self._get_obs()
        
        return observations, rewards, dones, infos
    
    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for agent, (x, y) in self.agent_positions.items():
            grid[y, x] = agent[-1]  # last char (agent id)
        print("\n".join(" ".join(row) for row in grid))
        print()
