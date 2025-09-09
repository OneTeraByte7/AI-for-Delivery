import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv

def make_env(max_episode_steps=400):
    def _fn():
        env = SingleAgentWrapper(
            DeliveryFleetEnv,
            env_kwargs=dict(
                grid_size=8,
                num_agents=3,
                max_orders=6,
                order_spawn_rate=3,
            ),
            control_agent="agent_0",
            max_episode_steps=max_episode_steps,
        )
        return Monitor(env)
    return _fn

def main():
    venv = DummyVecEnv([make_env(400)])
    model = PPO.load("./models/ppo_agent_0.zip")

    obs = venv.reset()
    ep_rewards = 0.0
    steps = 0
    episodes = 3

    for ep in range(episodes):
        done = False
        ep_rewards = 0.0
        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            ep_rewards += float(reward)
            steps += 1

            # render underlying env each step (optional)
            venv.envs[0].env.render()
            time.sleep(0.03)

            # VecEnv returns array for dones; single env -> index 0
            done = bool(dones[0])
            if done:
                print(f"[Episode {ep+1}] Reward: {ep_rewards:.2f}, Steps: {steps}")
                obs = venv.reset()
                steps = 0

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
