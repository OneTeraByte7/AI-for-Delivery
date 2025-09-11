# replay buffer for multi-agent RL

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
            
    def size(self) -> int:
        return sum(len(v) for v in self.data.values())

    def save_npz(self, path: str):
        # flatten to arrays per agent
        npz_dict = {}
        for a, buf in self.data.items():
            if not buf:
                continue
            O, A, R, N, D = zip(*buf)
            npz_dict[f"{a}/obs"] = np.stack(O, axis=0)
            npz_dict[f"{a}/actions"] = np.asarray(A, dtype=np.int64)
            npz_dict[f"{a}/rewards"] = np.asarray(R, dtype=np.float32)
            npz_dict[f"{a}/next_obs"] = np.stack(N, axis=0)
            npz_dict[f"{a}/dones"] = np.asarray(D, dtype=np.bool_)
        np.savez_compressed(path, **npz_dict)

    def clear(self):
        for a in list(self.data.keys()):
            self.data[a].clear()