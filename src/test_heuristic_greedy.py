from env import DeliveryFleetEnv
from policies.heuristics_greedy_approach import self_policy
import time, random

def main():
    env = DeliveryFleetEnv(grid_size=8, num_agents=3, max_orders=5, order_spawn_rate = 3)
    env.reset()
    delivered = 0
    
    for t in range(40):
        actions = {}
        
        for a in env.agents:
            actions[a] = self_policy(env, a) if a == "agent_0" else random.randint(0,4)
        _, rewards, _, _ = env.step(actions)
        delivered += sum(1 for r in rewards.values() if r >= 10)
        print(f"\n-- step {t+1} -- delivered spike? {rewards}")
        env.render()
        time.sleep(0.2)
        
    print("Done")
    
if __name__ == "__main__":
    main()