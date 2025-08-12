from env import DeliveryFleetEnv

def main():
    env = DeliveryFleetEnv(grid_size=5, num_agents=2)
    obs = env.reset()
    print("Initial Observations:", obs)
    env.render()

    actions = {'agent_0': 4, 'agent_1': 1}  # agent_0 moves right, agent_1 moves up
    obs, rewards, dones, infos = env.step(actions)
    print("Rewards:", rewards)
    env.render()

if __name__ == "__main__":
    main()
