import numpy as np
import matplotlib.pyplot as plt
import random
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# Actions
STAY, UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF = range(7)


class DeliveryFleetEnv(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=8, num_agents=3, max_orders=6, order_spawn_rate=3, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_orders = max_orders
        self.order_spawn_rate = order_spawn_rate
        self.max_steps = max_steps

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]

        # Action and observation spaces
        self.action_spaces = {agent: spaces.Discrete(7) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32)
            for agent in self.agents
        }

        # State
        self.t = 0
        self.agent_positions = {}
        self.agent_carrying = {}
        self.orders = []

    def reset(self, seed=None, options=None):
        self.t = 0
        self.agents = self.possible_agents[:]
        self.agent_positions = {agent: self._random_empty_cell() for agent in self.agents}
        self.agent_carrying = {agent: None for agent in self.agents}
        self.orders = []
        obs = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        self.t += 1
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"delivered": False} for agent in self.agents}

        # Spawn new orders
        if len(self.orders) < self.max_orders and self.t % self.order_spawn_rate == 0:
            self.orders.append(self._generate_order())

        # Apply actions
        for agent, action in actions.items():
            x, y = self.agent_positions[agent]

            if action == UP:
                y = max(0, y - 1)
            elif action == DOWN:
                y = min(self.grid_size - 1, y + 1)
            elif action == LEFT:
                x = max(0, x - 1)
            elif action == RIGHT:
                x = min(self.grid_size - 1, x + 1)
            elif action == PICKUP and self.agent_carrying[agent] is None:
                for order in self.orders:
                    if order["status"] == "waiting" and (x, y) == order["pickup"]:
                        order["status"] = "picked"
                        self.agent_carrying[agent] = order["id"]
                        rewards[agent] += 5
                        break
            elif action == DROPOFF and self.agent_carrying[agent] is not None:
                for order in self.orders:
                    if order["id"] == self.agent_carrying[agent] and (x, y) == order["dropoff"]:
                        order["status"] = "delivered"
                        self.agent_carrying[agent] = None
                        rewards[agent] += 20
                        infos[agent]["delivered"] = True
                        break

            self.agent_positions[agent] = (x, y)

        # End condition
        done = self.t >= self.max_steps
        if done:
            truncations = {agent: True for agent in self.agents}

        obs = {agent: self._get_obs(agent) for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    def _random_empty_cell(self):
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def _generate_order(self):
        return {
            "id": len(self.orders),
            "pickup": self._random_empty_cell(),
            "dropoff": self._random_empty_cell(),
            "status": "waiting",
        }

    def _get_obs(self, agent):
        grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Agents
        for pos in self.agent_positions.values():
            grid[0, pos[1], pos[0]] = 1.0

        # Orders
        for order in self.orders:
            if order["status"] == "waiting":
                px, py = order["pickup"]
                grid[1, py, px] = 1.0
            elif order["status"] == "picked":
                dx, dy = order["dropoff"]
                grid[2, dy, dx] = 1.0

        return grid

    def render(self, mode="human"):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8) + 255  # white background

        # Orders
        for order in self.orders:
            if order["status"] == "waiting":
                x, y = order["pickup"]
                grid[y, x] = [255, 0, 0]  # red pickup
            elif order["status"] == "picked":
                x, y = order["dropoff"]
                grid[y, x] = [0, 0, 255]  # blue dropoff
            elif order["status"] == "delivered":
                x, y = order["dropoff"]
                grid[y, x] = [180, 180, 180]  # gray delivered

        # Agents
        for _, pos in self.agent_positions.items():
            x, y = pos
            grid[y, x] = [0, 200, 0]  # green agent

        plt.imshow(grid, interpolation="nearest")
        plt.axis("off")
        plt.pause(0.1)
        plt.clf()
