import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv

# Folders for the use
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def make_env(max_episode_steps=400):
    # Wrap with Monitor so SB3 logs episode rewards/length
    def _fn():
        env = SingleAgentWrapper(
            DeliveryFleetEnv,
            env_kwargs=dict(
                grid_size=8,
                num_agents=3,
                max_orders=5,
                order_spawn_rate=3,
            ),
            control_agent="agent_0",
            max_episode_steps=max_episode_steps,
        )
        return Monitor(env)
    return _fn

def main():
    # Training vec env
    venv = DummyVecEnv([make_env(400)])

    # Separate eval env (same config, deterministic eval)
    eval_env = DummyVecEnv([make_env(400)])

    # Callbacks: evaluate every N steps, save best; plus periodic checkpoints
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="logs",
        eval_freq=5_000,              # evaluate every 5k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path="checkpoints", name_prefix="ppo_agent0")

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        tensorboard_log="logs/tb",     # run: tensorboard --logdir logs/tb
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
    )

    model.learn(total_timesteps=200_000, callback=[eval_cb, ckpt_cb])
    model.save("models/ppo_agent_0_final.zip")
    print("Saved final model to models/ppo_agent_0_final.zip")

if __name__ == "__main__":
    main()
