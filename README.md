# 🐍 Snake AI — Deep Q-Network with Ablation Study

An autonomous Snake agent trained using Deep Q-Learning (DQN), built by extending a base implementation with custom environment design, reward shaping, and a target network. Includes a comparative ablation study across three versions.

---

## 🧠 What I Built On Top of the Base

Starting from a standard DQN Snake tutorial, I added:

- **Static obstacle environment** — 9 fixed obstacles increasing navigation difficulty
- **Distance-based reward shaping** — intermediate rewards (+0.3 closer, -0.5 farther) to accelerate learning
- **2-layer hidden network** — extended architecture (256→128) for better representational capacity
- **Target network** — separate frozen network synced every 100 steps for stable training
- **Ablation study** — systematic comparison of single DQN vs target network on identical environments

---

## 📊 Results

> **Note:** Base DQN runs on a clean environment (no obstacles). Single DQN and Target Network runs include 9 static obstacles, making direct score comparison misleading. The meaningful comparison is **Single DQN vs Target Network** on the same environment.

| Version | Environment | Mean Score | Peak Score | Episodes |
|---|---|---|---|---|
| Base DQN | No obstacles | 22.05 | 74 | 210 |
| Single DQN | With obstacles | 16.4 | 85 | 1000 |
| Target Network DQN | With obstacles | **17.3** | 62 | 1000 |

**Key finding:** Target network achieves higher mean score (17.3 vs 16.4) with smoother convergence. Single DQN shows higher peak scores but unstable training — consistent with overestimation bias expected without a target network.

### Base DQN (no obstacles)
![Base DQN](without obstacle and path based reward.png)

### Single DQN (with obstacles + reward shaping)
![Single DQN](single_dqn.png)

### Target Network DQN (with obstacles + reward shaping)
![Target Network](double_dqn.png)

---

## 🔬 Experiments & Iterations

Not everything worked. Here's the full iteration history including what was tried and removed:

| Experiment | Result | Decision |
|---|---|---|
| Past 15 head positions added to state | Increased state complexity without clear learning benefit — agent struggled with larger input | Removed |
| Golden apple (bonus high-reward item) | Negligible impact on learning speed or final performance | Removed |
| Distance-based reward (0.3 / -0.5) | Significantly faster convergence vs sparse reward alone | Kept |
| Target network sync every 100 steps | More frequent updates improved stability over longer sync intervals | Kept |
| 2-layer hidden network (256 to 128) | Better performance than single hidden layer | Kept |
| Epsilon decay: 0.99^n_games | Smooth exploration-exploitation tradeoff over training | Kept |

> The memory addition (past head positions) was the most instructive failure — it showed that naively expanding state space without meaningful signal hurts more than it helps.

---

## 🧠 Architecture

### Neural Network
```
Input (11) -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Output(3)
```

| Layer | Size | Description |
|---|---|---|
| Input | 11 | Danger (3), direction (4), food location (4) |
| Hidden 1 | 256 | Fully connected + ReLU |
| Hidden 2 | 128 | Fully connected + ReLU |
| Output | 3 | Q-values for [straight, right, left] |

### State Representation (11 inputs)
```python
[
  danger_straight, danger_right, danger_left,   # collision detection
  dir_left, dir_right, dir_up, dir_down,        # current direction
  food_left, food_right, food_up, food_down     # food relative position
]
```

---

## ⚙️ Algorithm Details

### Deep Q-Learning (DQN)
Uses the Bellman equation for Q-value updates:
```
Q(s, a) = reward + gamma * max(Q(s', a'))
```
where gamma = 0.9 (discount factor)

### Epsilon-Greedy Exploration
```python
epsilon = max(0.01, 0.99 ** n_games)
```
Starts fully random, decays exponentially — balances exploration vs exploitation.

### Experience Replay
- Replay buffer of **20,000** transitions
- Random batch sampling of **500** per training step
- Breaks correlation between consecutive steps for stable learning

### Target Network
```python
# Separate frozen network — synced every 100 steps
Q_target = reward + gamma * max(target_network(next_state))
```
Provides a stable learning signal. Without it, the network chases a moving target causing oscillation.

### Reward Structure
| Event | Reward |
|---|---|
| Food eaten | +20 |
| Moving closer to food | +0.3 |
| Moving away from food | -0.5 |
| Death (collision/timeout) | -24 |

---

## 🗂️ Project Structure

```
├── agent.py                   # DQN agent — memory, epsilon-greedy, training loop
├── model.py                   # Neural network + QTrainer with target network
├── game.py                    # Snake environment with obstacles (Pygame)
├── helper.py                  # Live training plot
├── results_base_dqn.png       # Training curve — base DQN, no obstacles
├── results_single_dqn.png     # Training curve — single DQN, with obstacles
├── results_target_network.png # Training curve — target network DQN
└── model/
    └── model.pth              # Saved best model weights
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pygame torch numpy matplotlib
```

### Run Training
```bash
python agent.py
```
Live training plot appears automatically. Best model auto-saves to `model/model.pth`.

---

## 🔑 Key Concepts

### Why Target Network?
Without a target network, the same network both predicts Q-values and generates targets — like chasing a moving goalpost. The target network is frozen for 100 steps, providing a stable learning signal and reducing oscillation.

### Why Experience Replay?
Consecutive game steps are highly correlated. Randomly sampling from a large replay buffer breaks this correlation, leading to more stable and efficient training.

### Why Reward Shaping?
Sparse rewards (only +20 on food) make early learning slow — the agent rarely finds food by chance. Distance-based intermediate rewards provide a denser learning signal, significantly accelerating convergence.

### Why 2-Layer Network?
A single hidden layer struggles to learn the non-linear relationships between danger signals, direction, and food location simultaneously. The second layer (256 to 128) allows hierarchical feature learning.

---

## 📚 References
- [Playing Atari with Deep Reinforcement Learning — DeepMind (2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning — Nature (2015)](https://www.nature.com/articles/nature14236)

---

## 🛠️ Built With
Python 3.x · PyTorch · Pygame · NumPy · Matplotlib
