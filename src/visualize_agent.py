from stable_baselines3 import PPO
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv
import matplotlib.pyplot as plt

def main():
    env = SingleAgentWrapper(
        DeliveryFleetEnv,
        env_kwargs=dict(grid_size=8, num_agents=3, max_orders=6, order_spawn_rate=3),
        control_agent="agent_0",
        max_episode_steps=200,
    )

    model = PPO.load("models/ppo_agent_0.zip")  # or random agent for now

    obs, _ = env.reset()
    plt.ion()  # interactive mode
    plt.figure()

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
