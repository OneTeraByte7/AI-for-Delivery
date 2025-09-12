# 🚚 Multi-Agent Delivery Fleet Simulator

A reinforcement learning environment for **multi-agent delivery coordination**.  
Agents operate on a grid-world, picking up and delivering customer orders while competing or collaborating to maximize rewards.  

This project includes:
- A **PettingZoo parallel environment** (`DeliveryFleetEnv`)
- **Single-agent wrapper** for training with Stable-Baselines3
- **Greedy heuristic policy** for baseline evaluation
- **Visualization tools** to animate agents, pickups, and deliveries
- Scripts for **training, evaluation, and replay collection**

---

## ✨ Features

- **Multi-agent environment** built with [PettingZoo](https://www.pettingzoo.ml/)
- Configurable:
  - Grid size
  - Number of agents
  - Order spawn rate
  - Max orders per episode
  - Episode length
- Agents can:
  - Move (`UP, DOWN, LEFT, RIGHT`)
  - Stay put (`STAY`)
  - Pick up orders (`PICKUP`)
  - Drop off orders (`DROPOFF`)
- **Reward shaping**:
  - +5 for successful pickup
  - +20 for successful delivery
- **Visualization**:
  - Grid-based animation with `matplotlib`
  - Agents = 🟢 green  
  - Pickup locations = 🔴 red  
  - Dropoff locations = 🔵 blue  

---

## 📂 Project Structure
```bash
├── env.py # Multi-agent environment (PettingZoo ParallelEnv)
├── wrapper/
│ └── single_agent.py # Single-agent Gym wrapper
├── policies/
│ └── coordinated_greedy.py # Baseline greedy policy
├── run_coordinated_collect.py # Collect replay buffer using greedy policy
├── train_ppo_single.py # Train single-agent PPO (Stable-Baselines3)
├── eval_ppo_agent_single.py # Evaluate PPO agent
├── visualize_agent.py # Visualize trained/random agent
└── visualize_random.py # Visualize random agent (smoke test)
```

---


---

## ⚡ Installation

Clone and set up:

```bash
git clone https://github.com/your-username/delivery-fleet-rl.git
cd delivery-fleet-rl

# create virtualenv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```

If requirements.txt isn’t created yet, install the basics:
```bash
pip install numpy matplotlib gymnasium pettingzoo stable-baselines3[extra]
```

---

## ▶️ Usage
1. Smoke test with random agents
```bash
python src/visualize_random.py
```

You should see a live grid animation with agents moving randomly, pickups, and deliveries.

2. Collect replay buffer with greedy policy
```bash
mkdir -p data
python src/run_coordinated_collect.py
```

Output: data/replay_greedy_coordinated.npz

3. Train a PPO agent (single-agent wrapper)
```bash
mkdir -p models
python src/train_ppo_single.py
```

Saves model to models/ppo_agent_0_grid.zip

4. Evaluate trained PPO agent
```bash
python src/eval_ppo_agent_single.py
```

5. Visualize trained agent in action
```bash
python src/visualize_agent.py
```

---

## 📊 Environment Details

Observation space:
Shape (3, grid_size, grid_size)
Channels:
[0] Agent positions
[1] Pickup locations
[2] Dropoff locations

Action space:
STAY, UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF (7 discrete actions)

Rewards:
+5 for pickup
+20 for successful delivery
0 otherwise

---

## 🏗️ Future Work

✅ Basic visualization
✅ Greedy heuristic baseline
✅ SB3 PPO single-agent training

Planned:
⏳ True multi-agent RL training (via RLlib or MARLlib)
⏳ Richer reward signals (time penalties, collision penalties)
⏳ More advanced policies (communication, shared goals)
⏳ Dashboard with stats: deliveries completed, efficiency, agent trails

---

## 👀Visuals:

See in folder

---

## Copyright

All copyrights are reserved to Soham 

