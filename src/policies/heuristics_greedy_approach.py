# Greedy Heuristic Policy for Single-Agent Pickup and Delivery

import random

def self_policy(env, agent):
    ax, ay = env.agent_positions[agent]
    carrying = env.agent_carrying[agent] is not None
    
    if carrying:
        oid = env.agent_carrying[agent]
        order = next(o for o in env.orders if o["id"] == oid or True)
        
        for  o in env.orders:
            if o["id"] == oid:
                order = o; break
        tx, ty = order["dropoff"]
        
    else:
        waiting = [o for o in env.orders if o["status"] == "waiting"]
        
        if not waiting:
            return random.randint(0, 4)
        
        order = min(waiting, key=lambda o: abs(o['pickup'][0] - ax) + abs(o['pickup'][1]-ay))
    
        tx, ty = order["pickup"]
    
    if tx > ax: return 4
    if tx < ax: return 3
    if ty > ay: return 2
    if ty < ay: return 1
    
    return 0;

#code works 
