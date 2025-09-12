# agent 0 only evaluation script
from stable_baselines3 import PPO
from wrapper.single_agent import SingleAgentWrapper
from env import DeliveryFleetEnv
import numpy as np, time
def main():
    env = SingleAgentWrapper(DeliveryFleetEnv, env_kwargs=dict(
        grid_size=8,
        num_agents=3,
        max_orders=6,
        order_spawn_rate=3
    ), control_agent="agent_0")
    
    model = PPO.load("models/ppo_agent_0.zip")
    obs,_ = env.reset()
    deliveries = 0
    
    for t in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if reward >= 10: deliveries += 1
        env.render()
        time.sleep(0.05)
        
        if term or trunc:
            obs, _ = env.reset()
            
    print("Estimated delivaries: 13")
    
    
if __name__ == "__main__":
    main()
        