from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrapper.multi_agent import MultiAgentWrapper
from env import DeliveryFleetEnv

curriculum = [
    dict(grid_size  =5, num_agents=2, max_orders=3, order_spawn_rate=2),
    dict(grid_size=6, num_agents=3, max_orders=4, order_spawn_rate=3),
    dict(grid_size=8, num_agents=4, max_orders=5, order_spawn_rate=4),
]

def make_enc(env_kwargs):
    return MultiAgentWrapper(DeliveryFleetEnv, env_kwargs=env_kwargs)


def main():
    total_timesteps_per_stage = 50_000
    for stage, env_kwargs in enumerate(curriculum, 1):
        print(f"\n--- Curriculum Stage {stage}: {env_kwargs} ---")
        venv = DummyVecEnv([lambda: make_enc(env_kwargs)])
        model = PPO("MlpPolicy", venv, verbose=1, tensorboard_log="./tensorboard_ma/")
        model.learn(total_timesteps=total_timesteps_per_stage)
        model.save(f"model/ppo_ma_stage_{stage}.zip")
        print(F"Saved model for stage {stage}")
        
if __name__ == "__main__":
    main()