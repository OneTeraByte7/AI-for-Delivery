# src/inspect_replay.py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    path = "data/replay_greedy_coordinated.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Replay buffer not found: {path}")

    data = np.load(path, allow_pickle=True)
    print(f"✅ Loaded replay buffer from {path}")
    print("Available keys:", list(data.keys()))

    agents = sorted(set(k.split("/")[0] for k in data.keys()))
    print("\n🤖 Agents found:", agents)

    for agent in agents:
        obs = data[f"{agent}/obs"]
        actions = data[f"{agent}/actions"]
        rewards = data[f"{agent}/rewards"]
        dones = data[f"{agent}/dones"]

        print(f"\n📊 Stats for {agent}:")
        print(f"  Transitions: {len(obs)}")
        print(f"  Obs shape: {obs.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Dones shape: {dones.shape}")
        print(f"  Reward mean: {rewards.mean():.3f}, std: {rewards.std():.3f}, min: {rewards.min():.3f}, max: {rewards.max():.3f}")

        # --- Action distribution ---
        unique, counts = np.unique(actions, return_counts=True)
        action_dist = dict(zip(unique, counts))
        print("  Action distribution:")
        for a, c in action_dist.items():
            print(f"    {a}: {c}")

        # --- Episode returns ---
        ep_returns = []
        ep_return = 0.0
        for r, d in zip(rewards, dones):
            ep_return += r
            if d:
                ep_returns.append(ep_return)
                ep_return = 0.0
        if ep_return != 0:
            ep_returns.append(ep_return)  # last partial ep

        print(f"  Episodes found: {len(ep_returns)}")
        if ep_returns:
            print(f"  Avg return: {np.mean(ep_returns):.3f}, min: {np.min(ep_returns):.3f}, max: {np.max(ep_returns):.3f}")

        # --- Plots ---
        plt.figure(figsize=(10, 4))

        # Reward curve
        plt.subplot(1, 2, 1)
        plt.plot(ep_returns, marker="o")
        plt.title(f"{agent} - Episode Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")

        # Action distribution
        plt.subplot(1, 2, 2)
        plt.bar(action_dist.keys(), action_dist.values())
        plt.title(f"{agent} - Action Distribution")
        plt.xlabel("Action")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
