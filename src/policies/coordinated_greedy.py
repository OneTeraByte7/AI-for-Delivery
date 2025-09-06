from typing import Dict, List, Tuple, Optional
import random

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
DEFAULT_PICKUP, DEFAULT_DROPOFF = STAY, STAY

def _move_towards(src: Tuple[int,int], dst: Tuple[int,int]) -> int:
    (r0,c0),(r1,c1)=src,dst
    dr,dc = r1-r0, c1-c0
    if abs(dr) >= abs(dc):
        if dr>0: return DOWN
        if dr<0: return UP
        if dc>0: return RIGHT
        if dc<0: return LEFT
        return STAY
    else:
        if dc>0: return RIGHT
        if dc<0: return LEFT
        if dr>0: return DOWN
        if dr<0: return UP
        return STAY

def _closest(src: Tuple[int,int], targets: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    if not targets: return None
    return min(targets, key=lambda t: abs(t[0]-src[0])+abs(t[1]-src[1]))

class CoordinatedGreedy:
    def __init__(self, env, seed=0):
        random.seed(seed)
        self.env = env
        self.grid_size = getattr(env,"grid_size",8)
        self.agents = list(env.agents)
        self.num_agents = len(self.agents)
        self.pickup_action = getattr(env,"pickup_action",DEFAULT_PICKUP)
        self.dropoff_action = getattr(env,"dropoff_action",DEFAULT_DROPOFF)

        cols_per = max(1,self.grid_size//max(1,self.num_agents))
        self.zones = {}
        for i,a in enumerate(self.agents):
            c0=i*cols_per
            c1=self.grid_size-1 if i==self.num_agents-1 else min(self.grid_size-1,c0+cols_per-1)
            self.zones[a]=(c0,c1)

        self.paths={}
        for a,(c0,c1) in self.zones.items():
            path=[]
            for r in range(self.grid_size):
                cols=range(c0,c1+1)
                cols=cols if r%2==0 else reversed(list(cols))
                for c in cols:
                    path.append((r,c))
            self.paths[a]=path
        self.path_idx={a:0 for a in self.agents}

    def _extract_targets(self):
        pickups=[]
        dropoffs=[]
        orders=getattr(self.env,"orders",None)
        if orders:
            for o in orders:
                if "pickup" in o and "dropoff" in o:
                    pickups.append(tuple(o["pickup"]))
                    dropoffs.append(tuple(o["dropoff"]))
        return pickups, dropoffs

    def act(self, obs: Dict[str, object]) -> Dict[str,int]:
        pickups, dropoffs = self._extract_targets()
        positions = getattr(self.env,"agent_positions",{})
        carrying = getattr(self.env,"agent_carrying",{})

        actions={}
        for a in self.agents:
            pos = positions.get(a)
            if pos is None:
                actions[a]=STAY
                continue
            if carrying.get(a):
                if dropoffs:
                    tgt=_closest(pos,dropoffs)
                    actions[a]=self.env.dropoff_action if tgt==pos else _move_towards(pos,tgt)
                    continue
            if pickups:
                zone=self.zones[a]
                in_zone=[p for p in pickups if zone[0]<=p[1]<=zone[1]]
                tgt=_closest(pos,in_zone) if in_zone else _closest(pos,pickups)
                actions[a]=self.env.pickup_action if tgt==pos else _move_towards(pos,tgt)
                continue
            # fallback sweep
            path=self.paths[a]
            idx=self.path_idx[a]
            goal=path[idx]
            if pos==goal:
                idx=(idx+1)%len(path)
                self.path_idx[a]=idx
                goal=path[idx]
            actions[a]=_move_towards(pos,goal)
        return actions
