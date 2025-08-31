import numpy as np
from env import DeliveryFleetEnv

def greedy_policy(obs, env, agent_id):
    """Decide action based on a greedy policy."""
    agent_pos = env.agent_positions[agent_id]
    
    if env.agent_carrying[agent_id]:
        targets = env.deliveries
        
    else:
        targets = env.orders
        
    if not targets:
        return 0
    
    target = min(targets, key=lambda t: abs(t[0] - agent_pos[0]) + abs(t[1] - agent_pos[1]))
    dx, dy = target[0] - agent_pos[0], target[1] - agent_pos[1]
    
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4
    
    else:
        return 1 if dy > 0 else 1
    
def main():
    env = DeliveryFleetEnv(grid_size=8, num_agents=3, max_orders=5, order_spawn_rate=2)
    obs = env.reset()
    env.render()
    
    total_rewards = {i: 0 for i in range(env.num_agents)}
    deliveries = 0
    
    for step in range(20):
        actions={}
        for agent_id in range(env.num_agents):
            actions[agent_id] = greedy_policy(obs, env, agent_id)
            
        obs, rewards, done, info = env.step(actions)
        env.render()
        
        for i, r in rewards.items():
            total_rewards[i] += r
        deliveries = sum(env.agent_deliveries.values())
        
        print(f"\n--- Step {step+1} ---")
        print("Rewards:", rewards)
        print("Total deliveries so far:", deliveries)
        
        if all(done.values()):
            break
    print("\n Final Results:")
    print("Total Rewards:", total_rewards)
    print("Total Deliveries:", deliveries)
    
    
if __name__ == "__main__":
    main()