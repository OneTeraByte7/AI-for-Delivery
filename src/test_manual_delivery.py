from env import DeliveryFleetEnv
import random

def main():
    env = DeliveryFleetEnv(grid_size=8, num_agents=3, max_orders=5, order_spawn_rate=2)
    obs = env.reset()
    print("Initial Grid:")
    env.render()
    print()

    total_deliveries = 0

    for step in range(20):
        actions = {agent: random.randint(0, 4) for agent in env.agents}
        obs, rewards, dones, infos = env.step(actions)

        # Count deliveries: positive reward indicates a successful delivery
        for agent, reward in rewards.items():
            if reward > 0:
                total_deliveries += 1

        print(f"--- Step {step+1} ---")
        env.render()
        print("Step rewards:", rewards)
        print("Total deliveries so far:", total_deliveries)
        print()

        if all(dones.values()):
            print("All agents done!")
            break

if __name__ == "__main__":
    main()
