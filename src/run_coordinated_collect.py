from env import DeliveryFleetEnv
from policies.coordinated_greedy import CoordinatedGreedy
from utils.replay_buffer import PerAgentReplayBuffer


def main():
    env = DeliveryFleetEnv(
        grid_size=8,
        num_agents=3,
        max_orders=6,
        order_spawn_rate=3,
    )
    
    policy = CoordinatedGreedy(env, seed=0)
    rb = PerAgentReplayBuffer(capacity=200_000)
    episodes = 10
    steps_per_ep = 100
    
    for ep in range(episodes):
        obs = env.reset()
        ep_reward_sum = 0.0
        delivered = 0
        spawned = 0
        
        for t in range(steps_per_ep):
            actions = policy.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            # track reward
            ep_reward_sum += sum(rewards.values())

            # track orders
            spawned = max(spawned, env.next_order_id)  # count how many ever spawned
            delivered = ep * 1000  # placeholder, fix below
            delivered = (
                env.next_order_id - len(env.orders)
            )  # delivered orders = spawned - active

            rb.add_step(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            if all(dones.values()):
                break
        
        delivery_rate = delivered / spawned if spawned > 0 else 0.0
        avg_agent_reward = ep_reward_sum / env._num_agents

        print(
            f"[ep {ep+1}/{episodes}] "
            f"total reward={ep_reward_sum:.2f}, "
            f"avg/agent={avg_agent_reward:.2f}, "
            f"orders spawned={spawned}, "
            f"delivered={delivered}, "
            f"success_rate={delivery_rate:.2%}"
        )
        
    path = "data/replay_greedy_coordinated.npz"
    rb.save_npz(path)
    print(f"Saved replay buffer: {path} (transitions: {rb.size()})")


if __name__ == "__main__":
    main()
