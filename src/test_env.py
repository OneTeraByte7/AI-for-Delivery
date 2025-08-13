from env import DeliveryFleetEnv
import random
import time

def main():
    env = DeliveryFleetEnv(grid_size=6, num_agents=2, max_orders=3, order_spawn_rate=3)
    obs = env.reset()
    print("Initial Observations:", obs)
    print("Inital Orders:", env.orders)
    env.render()

    for step in range(10):   
        actions = {agent: random.randint(0, 4) for agent in env.agents}
        obs, rewards, dones, infos = env.step(actions)
        
        print(f"\n--- Step {step+1}---")
        env.render()
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Active Orders:", env.orders)
        
        time.sleep(0.3)

if __name__ == "__main__":
    main()
