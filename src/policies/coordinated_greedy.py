from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import random

UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

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
        
    def _in_zone(self, pos, zone: Tuple[int, int]) -> bool:
        if isinstance(pos, dict):
            r, c = pos.get("pos", (None, None))[:2]
        elif isinstance(pos, (tuple, list)):
            r, c = pos[:2]
        else:
            raise ValueError(f"Unexpected pos format in _in_zone: {pos}")

        c0, c1 = zone
        return c0 <= c <= c1


    def _extract_targets(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        def normalize(lst):
            out = []
            for x in lst or []:
                if isinstance(x, dict) and "pos" in x:
                    out.append(tuple(x["pos"][:2]))
                elif isinstance(x, (tuple, list)):
                    out.append(tuple(x[:2]))
            return out

        pickup_candidates = [
            getattr(self.env, "pickups", None),
            getattr(self.env, "orders", None),
            getattr(self.env, "pickup_locations", None),
        ]
        delivery_candidates = [
            getattr(self.env, "deliveries", None),
            getattr(self.env, "dropoffs", None),
            getattr(self.env, "delivery_locations", None),
        ]

        pickups, deliveries = [], []
        for cand in pickup_candidates:
            if cand:
                pickups = normalize(cand)
                if pickups: break
        for cand in delivery_candidates:
            if cand:
                deliveries = normalize(cand)
                if deliveries: break

        return pickups, deliveries


    def act(self, obs: Dict[str, object]) -> Dict[str, int]:
        """
        obs: per-agent observation dict from env.reset()/env.step()
        returns: per-agent discrete action
        """
        pickups, deliveries = self._extract_targets()
        positions = getattr(self.env, "agent_positions", {})
        carrying = getattr(self.env, "agent_carrying", {})

        actions: Dict[str, int] = {}

        for a in self.agents:
            pos = positions.get(a, None)
            if pos is None:
                # If env doesn't expose positions, just stay (or random)
                actions[a] = STAY
                continue

            zone = self.zones[a]

            # If carrying, head toward nearest delivery (if known)
            if carrying.get(a, None):
                if deliveries:
                    target = _closest(pos, deliveries)
                    actions[a] = _move_towards(pos, target)
                    continue
                # else: fall back to sweep

            # If not carrying, look for a pickup in-zone
            if pickups:
                # prioritize targets inside my zone
                in_zone = [p for p in pickups if self._in_zone(p, zone)]
                target = _closest(pos, in_zone) if in_zone else _closest(pos, pickups)
                actions[a] = _move_towards(pos, target)
                continue

            # Fallback: follow zone sweep path
            path = self.paths[a]
            idx = self.path_idx[a]
            goal = path[idx]
            if pos == goal:
                idx = (idx + 1) % len(path)
                self.path_idx[a] = idx
                goal = path[idx]
            actions[a] = _move_towards(pos, goal)

        return actions