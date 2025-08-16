import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

def make_env():
    return SingleAgentWrapper(
        DeliveryFleetEnv,
        env_kwargs=dict(
            grid_size=8,
            num_agents=3,
            max_orders=5,
            order_spawn_rate=3,
        ),
        control_agent="agent_0"
    )

def main():
    # Wrap environment for SB3
    venv = DummyVecEnv([make_env])

    # Create PPO model
    model = PPO("MlpPolicy", venv, verbose=1)

    # Train the model
    model.learn(total_timesteps=50_000)

    # Save model
    model.save("models/ppo_agent_0.zip")
    print("Saved PPO model to models/ppo_agent_0.zip")

if __name__ == "__main__":
    main()
