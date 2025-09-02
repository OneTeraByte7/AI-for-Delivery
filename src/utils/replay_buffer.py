from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Deque

class PerAgentReplayBuffer:
    """
    Stores per-agent transitions:
      (obs, action, reward, next_obs, done)
    Obs and next_obs are stored as np.float32 arrays.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.data: Dict[str, Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]]] = {}

    def _ensure_agent(self, agent_id: str):
        if agent_id not in self.data:
            self.data[agent_id] = deque(maxlen=self.capacity)
            
    def add_step(self,
                 obs: Dict[str, np.ndarray],
                 actions: Dict[str, int],
                 rewards: Dict[str, float],
                 next_obs: Dict[str, np.ndarray],
                 dones: Dict[str, bool]):
        for a in actions.keys():
            self._ensure_agent(a)
            o = np.asarray(obs[a], dtype=np.float32)
            n = np.asarray(next_obs[a], dtype=np.float32)
            r = float(rewards.get(a, 0.0))
            d = bool(dones.get(a, False))
            u = int(actions[a])
            self.data[a].append((o, u, r, n, d))