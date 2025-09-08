# src/env.py
from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces

# --- Action definitions ---
UP, DOWN, LEFT, RIGHT, STAY, PICKUP, DROPOFF = range(7)


class DeliveryFleetEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=8, num_agents=3, max_orders=6, order_spawn_rate=3):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_orders = max_orders
        self.order_spawn_rate = order_spawn_rate
        self.step_count = 0

        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agent_positions = {}
        self.agent_carrying = {agent: None for agent in self.agents}  # order_id if carrying

        self.orders = []  # active orders
        self.next_order_id = 0

        # actions for pickup/dropoff (used by CoordinatedGreedy)
        self.pickup_action = PICKUP
        self.dropoff_action = DROPOFF

        # action & obs spaces
        self.action_spaces = {agent: spaces.Discrete(7) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array([0, 0, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([self.grid_size - 1] * 6, dtype=np.float32),
                shape=(6,),
                dtype=np.float32
            )
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.orders = []
        self.next_order_id = 0
        self.agent_carrying = {agent: None for agent in self.agents}

        positions = set()
        for agent in self.agents:
            while True:
                pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
                if pos not in positions:
                    self.agent_positions[agent] = pos
                    positions.add(pos)
                    break
        return self._get_obs()

    def _spawn_order(self):
        if len(self.orders) >= self.max_orders:
            return
        pickup = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        dropoff = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        while dropoff == pickup:
            dropoff = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        order = {
            "id": self.next_order_id,
            "pickup": pickup,
            "dropoff": dropoff,
            "status": "waiting"
        }
        self.orders.append(order)
        self.next_order_id += 1

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            ax, ay = self.agent_positions[agent]
            if self.orders:
                nearest_order = min(
                    self.orders,
                    key=lambda o: abs(o['pickup'][0] - ax) + abs(o['pickup'][1] - ay)
                )
                px, py = nearest_order['pickup']
                dx, dy = nearest_order['dropoff']
            else:
                px, py, dx, dy = -1, -1, -1, -1
            obs[agent] = np.array([ax, ay, px, py, dx, dy], dtype=np.float32)
        return obs

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: -0.01 for agent in self.agents}  # step penalty
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # spawn new order
        if self.step_count % self.order_spawn_rate == 0:
            self._spawn_order()

        new_positions = {}
        for agent, action in actions.items():
            x, y = self.agent_positions[agent]

            # pickup
            if action == PICKUP and self.agent_carrying[agent] is None:
                for order in self.orders:
                    if order['status'] == 'waiting' and (x, y) == order['pickup']:
                        order['status'] = 'picked'
                        self.agent_carrying[agent] = order['id']
                        rewards[agent] += 5
                        break

            # dropoff
            elif action == DROPOFF and self.agent_carrying[agent] is not None:
                for order in self.orders:
                    if order['id'] == self.agent_carrying[agent] and (x, y) == order['dropoff']:
                        order['status'] = 'delivered'
                        self.agent_carrying[agent] = None
                        rewards[agent] += 20
                        break

            # movement
            elif action == UP and y > 0:
                y -= 1
            elif action == DOWN and y < self.grid_size - 1:
                y += 1
            elif action == LEFT and x > 0:
                x -= 1
            elif action == RIGHT and x < self.grid_size - 1:
                x += 1
            # elif action == STAY: do nothing

            new_positions[agent] = (x, y)

        # collision penalty
        pos_counts = {}
        for pos in new_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for agent, pos in new_positions.items():
            if pos_counts[pos] > 1:
                new_positions[agent] = self.agent_positions[agent]  # revert
                rewards[agent] -= 1

        self.agent_positions = new_positions

        # remove delivered orders
        self.orders = [o for o in self.orders if o['status'] != 'delivered']

        observations = self._get_obs()
        dones['__all__'] = False
        return observations, rewards, dones, infos

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for order in self.orders:
            px, py = order['pickup']
            dx, dy = order['dropoff']
            if order['status'] == 'waiting':
                grid[py, px] = 'P'
            elif order['status'] == 'picked':
                grid[dy, dx] = 'D'
        for agent, (x, y) in self.agent_positions.items():
            grid[y, x] = agent[-1]
        print("\n".join(" ".join(row) for row in grid))
        print()
