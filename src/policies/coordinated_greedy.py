# src/policies/coordinated_greedy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import random

UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

def _move_towards(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    (r0, c0), (r1, c1) = src, dst
    dr, dc = r1 - r0, c1 - c0
    # prefer largest axis first (Manhattan)
    if abs(dr) >= abs(dc):
        if dr > 0: return DOWN
        if dr < 0: return UP
        if dc > 0: return RIGHT
        if dc < 0: return LEFT
        return STAY
    else:
        if dc > 0: return RIGHT
        if dc < 0: return LEFT
        if dr > 0: return DOWN
        if dr < 0: return UP
        return STAY

def _closest(src: Tuple[int, int], targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not targets:
        return None
    return min(targets, key=lambda t: abs(t[0]-src[0]) + abs(t[1]-src[1]))

class CoordinatedGreedy:
    """
    Robust coordinated greedy policy:
    - Extracts pickups & dropoffs from env.orders when available
    - Zones & sweep fallback
    - If agent at pickup => STAY to ensure pickup processed
    """
    def __init__(self, env, seed: int = 0):
        random.seed(seed)
        self.env = env
        self.grid_size = getattr(env, "grid_size", 8)
        self.agents: List[str] = list(env.agents)
        self.num_agents = len(self.agents)

        cols_per = max(1, self.grid_size // max(1, self.num_agents))
        self.zones: Dict[str, Tuple[int, int]] = {}
        for i, a in enumerate(self.agents):
            c0 = i * cols_per
            c1 = self.grid_size - 1 if i == self.num_agents - 1 else min(self.grid_size - 1, c0 + cols_per - 1)
            self.zones[a] = (c0, c1)

        # sweep paths
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

    def _normalize_pos(self, item) -> Optional[Tuple[int, int]]:
        # Accept (r,c), [r,c], dict with 'pos', or order dicts
        if item is None:
            return None
        if isinstance(item, (tuple, list)):
            return (int(item[0]), int(item[1]))
        if isinstance(item, dict):
            if "pos" in item:
                p = item["pos"]
                return (int(p[0]), int(p[1]))
            # order dicts might have 'pickup'/'dropoff'
            if "pickup" in item and "dropoff" in item:
                # caller will decide which to use
                return None
        return None

    def _extract_targets(self):
        """
        Returns two lists of (r,c): pickups, dropoffs
        Tries multiple env attributes, and also inspects env.orders list if present.
        """
        pickups: List[Tuple[int, int]] = []
        dropoffs: List[Tuple[int, int]] = []

        # 1) If env has orders list of dicts with 'pickup'/'dropoff'
        orders = getattr(self.env, "orders", None)
        if orders:
            try:
                for o in orders:
                    if isinstance(o, dict):
                        p = o.get("pickup", None)
                        d = o.get("dropoff", None) or o.get("dropoff", o.get("drop", None))
                        if p:
                            pickups.append((int(p[0]), int(p[1])))
                        if d:
                            dropoffs.append((int(d[0]), int(d[1])))
                    elif isinstance(o, (tuple, list)) and len(o) >= 2:
                        pickups.append((int(o[0][0]), int(o[0][1])))
                        dropoffs.append((int(o[1][0]), int(o[1][1])))
            except Exception:
                # fall through to other candidates if structure unexpected
                pickups = []
                dropoffs = []

        # 2) Other candidate attributes
        if not pickups:
            cand = getattr(self.env, "pickups", None) or getattr(self.env, "pickup_locations", None)
            if cand:
                for x in cand:
                    p = self._normalize_pos(x)
                    if p: pickups.append(p)

        if not dropoffs:
            cand = getattr(self.env, "deliveries", None) or getattr(self.env, "dropoffs", None) or getattr(self.env, "delivery_locations", None)
            if cand:
                for x in cand:
                    d = self._normalize_pos(x)
                    if d: dropoffs.append(d)

        return pickups, dropoffs

    def act(self, obs: Dict[str, object]) -> Dict[str, int]:
        pickups, dropoffs = self._extract_targets()
        positions = getattr(self.env, "agent_positions", {})
        carrying = getattr(self.env, "agent_carrying", {})

        actions: Dict[str, int] = {}

        for a in self.agents:
            pos_raw = positions.get(a, None)
            pos = None
            if pos_raw is not None:
                if isinstance(pos_raw, dict) and "pos" in pos_raw:
                    pos = (int(pos_raw["pos"][0]), int(pos_raw["pos"][1]))
                elif isinstance(pos_raw, (tuple, list)):
                    pos = (int(pos_raw[0]), int(pos_raw[1]))

            if pos is None:
                actions[a] = STAY
                continue

            # If carrying, go to nearest dropoff
            if carrying.get(a) is not None:
                if dropoffs:
                    tgt = _closest(pos, dropoffs)
                    # if already at dropoff, stay to ensure delivery processed
                    if tgt == pos:
                        actions[a] = STAY
                    else:
                        actions[a] = _move_towards(pos, tgt)
                    continue

            # If not carrying, go to nearest pickup (prefer in-zone)
            if pickups:
                # try in-zone first
                zone = self.zones[a]
                in_zone = [p for p in pickups if zone[0] <= p[1] <= zone[1]]
                tgt = _closest(pos, in_zone) if in_zone else _closest(pos, pickups)
                if tgt == pos:
                    actions[a] = STAY  # allow env to register pickup
                else:
                    actions[a] = _move_towards(pos, tgt)
                continue

            # fallback sweep
            path = self.paths[a]
            idx = self.path_idx[a]
            goal = path[idx]
            if pos == goal:
                idx = (idx + 1) % len(path)
                self.path_idx[a] = idx
                goal = path[idx]
            actions[a] = _move_towards(pos, goal)

        return actions
