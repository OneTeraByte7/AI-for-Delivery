from pettingzoo.utils.env import ParallelEnv
import numpy as np
import random

class DeliveryFleetEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=10, num_agents=3, max_orders=5, order_spawn_rate=5):
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

        self.action_spaces = {agent: 5 for agent in self.agents}
        self.observation_spaces = {agent: (6,) for agent in self.agents}  # x, y, px, py, dx, dy
        
        self.reset()
        
    def reset(self):
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
            # nearest order pickup (for simplicity)
            if self.orders:
                nearest_order = min(self.orders, key=lambda o: abs(o['pickup'][0]-ax) + abs(o['pickup'][1]-ay))
                px, py = nearest_order['pickup']
                dx, dy = nearest_order['dropoff']
            else:
                px, py, dx, dy = -1, -1, -1, -1
            obs[agent] = np.array([ax, ay, px, py, dx, dy], dtype=np.float32)
        return obs
    
    def step(self, actions):
        self.step_count += 1
        rewards = {agent: -0.1 for agent in self.agents}  # step penalty
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # possibly spawn a new order
        if self.step_count % self.order_spawn_rate == 0:
            self._spawn_order()
        
        # Move agents
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
            new_positions[agent] = (x, y)
        
        # Collision resolution
        pos_counts = {}
        for pos in new_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for agent, pos in new_positions.items():
            if pos_counts[pos] > 1:
                new_positions[agent] = self.agent_positions[agent]  # revert
                rewards[agent] -= 1  # collision penalty
        
        self.agent_positions = new_positions
        
        # Pickup & Dropoff handling
        for agent in self.agents:
            pos = self.agent_positions[agent]
            if self.agent_carrying[agent] is None:
                # try pickup
                for order in self.orders:
                    if order["status"] == "waiting" and pos == order["pickup"]:
                        order["status"] = "picked"
                        self.agent_carrying[agent] = order["id"]
                        rewards[agent] += 1  # pickup bonus
                        break
            else:
                # try dropoff
                for order in self.orders:
                    if order["id"] == self.agent_carrying[agent] and pos == order["dropoff"]:
                        order["status"] = "delivered"
                        rewards[agent] += 10  # delivery bonus
                        self.agent_carrying[agent] = None
                        break
        
        # Remove delivered orders
        self.orders = [o for o in self.orders if o["status"] != "delivered"]
        
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
