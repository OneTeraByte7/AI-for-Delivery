import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "data/replay_greedy_coordinated.npz"
    data = np.load(path, allow_pickle=True)
    
    print(f"✅ Loaded replay buffer from {path}")
    keys = list(data.keys())
    print("Available keys:", keys)
    
    # Detect agents dynamically
    agents = sorted({k.split("/")[0] for k in keys})
    print(f"\n🤖 Agents found: {agents}")
    
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
        
        # Reward stats
        print(f"  Reward mean: {np.mean(rewards):.3f}, std: {np.std(rewards):.3f}, min: {np.min(rewards):.3f}, max: {np.max(rewards):.3f}")
        
        # Action distribution
        unique, counts = np.unique(actions, return_counts=True)
        print("  Action distribution:")
        for u, c in zip(unique, counts):
            print(f"    {u}: {c}")
        
        # Plot reward histogram
        plt.figure(figsize=(5,3))
        plt.hist(rewards, bins=20, alpha=0.7)
        plt.title(f"Reward Distribution - {agent}")
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
