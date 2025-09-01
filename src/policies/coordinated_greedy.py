from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import random

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

def _sign(x: int) -> int:
    return (x > 0) - (x < 0)

def _move_towards(src: Tuple[int, int], dist: Tuple[int, int]) -> int:
    (r0, c0), (r1, c1) = src, dist
    dr, dc = r1 - r0, c1 - c0
    if abs(dr) >= abs(dc):
        return DOWN if dr > 0 else (UP if dr < 0 else (RIGHT if dc > 0 else (LEFT if dc < 0 else STAY)))
    else:
        return RIGHT if dc > 0 else (LEFT if dc < 0 else (DOWN if dr > 0 else (UP if dr < 0 else STAY)))

def _closest(src: Tuple[int, int], targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not targets:
        return None
    return min(targets, key=lambda t: abs(t[0]-src[0]) + abs(t[1]-src[1]))

class CoordinatedGreedy:
    """
    Coordination layer:
      - Partition grid into vertical zones; each agent patrols its zone (lawnmower).
      - If we can detect pickups/deliveries, chase nearest relevant target within zone.
      - If agent is carrying (via env.agent_carrying if present), bias toward delivery.
    Works without env internals (degrades to zone sweep).
    """

    def __init__(self, env, seed: int = 0):
        random.seed(seed)
        self.env = env
        self.grid_size = getattr(env, "grid_size", 8)
        # Agents order must be stable
        self.agents: List[str] = list(env.agents)
        self.num_agents = len(self.agents)

        # Build zones: split columns among agents
        cols_per = max(1, self.grid_size // self.num_agents)
        self.zones: Dict[str, Tuple[int, int]] = {}
        for i, a in enumerate(self.agents):
            c0 = i * cols_per
            c1 = self.grid_size - 1 if i == self.num_agents - 1 else min(self.grid_size - 1, c0 + cols_per - 1)
            self.zones[a] = (c0, c1)

        # Precompute sweep paths per agent (row-wise boustrophedon inside zone)
        self.paths: Dict[str, List[Tuple[int, int]]] = {}
        for a, (c0, c1) in self.zones.items():
            path = []
            for r in range(self.grid_size):
                cols = range(c0, c1 + 1)
                cols = cols if r % 2 == 0 else reversed(list(cols))
                for c in cols:
                    path.append((r, c))
            self.paths[a] = path

        self.path_idx: Dict[str, int] = {a: 0 for a in self.agents}