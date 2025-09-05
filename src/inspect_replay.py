import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "data/replay_greedy_coordinated.npz"
    data = np.load(path, allow_pickle=True)
    
    print(f"Loaded replay buffer from {path}")
    print("Keys:", list(data.keys()))
    
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    
    print(f"Total transitions: {len(obs)}")
    print(f"Observation shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Reward stats
    print("\nReward stats:")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Min: {np.min(rewards):.3f}")
    print(f"  Max: {np.max(rewards):.3f}")
    
    unique, counts = np.unique(actions, return_counts=True)
    print("\nAction distribution:")
    for u, c in zip(unique, counts):
        print(f"  Action {u}: {c} times")
    
    # Plot rewards
    plt.figure()
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.show()
    
if __name__ == "__main__":
    main()
    
    