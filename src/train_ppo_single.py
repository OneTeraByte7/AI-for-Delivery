from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv


def make_env():
    return SingleAgentWrapper(DeliveryFleetEnv, env_kwargs=dict(
        grid_size=8,
        num_agents=3,
        max_orders=5,
        order_spawn_rate=3,
    ), control_agent="agent_0")
    
def main():
    venv = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("models/ppo_agent_0.zip")
    print("saved models to models folder!!")
    
    
if __name__ == "__main__":
    main()
    