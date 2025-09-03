from env import DeliveryFleetEnv
from policies.coordinated_greedy import CoordinatedGreedy
from utils.replay_buffer import PerAgentReplayBuffer

def main():
    env = DeliveryFleetEnv(
        grid_size = 8,
        num_agents = 3,
        max_orders = 6,
        order_spawn_rate = 3,
    )
    
    policy = CoordinatedGreedy(env, seed=0)
    rb = PerAgentReplayBuffer(capacity=200_000)
    episodes = 10
    steps_per_ep = 100
    
    for ep in range(episodes):
        obs  =env.reset()
        ep_reward_sum = 0.0
        
        for t in range(steps_per_ep):
            actions = policy.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            
            rb.add_step(obs, actions, rewards, next_obs, dones)
            ep_reward_sum += sum(rewards.values())
            
            obs = next_obs
            if all(dones.values()):
                break
            
        print(f"[ep {ep+1}/{episodes}] total reward: {ep_reward_sum:.3f}")
        
    path = "data/replay_greedy_coordinated.npz"
    rb.save_npz(path)
    print(f"Saved replay buffer: {path} (transitions: {rb.size()})")
    
    
if __name__ == "__main__":
    main()
    