# Session 27 – Deep Reinforcement Learning

## 📚 Table of Contents
1. [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
2. [Policy Gradient Deep RL](#policy-gradient-deep-rl)
3. [Actor-Critic Methods](#actor-critic-methods)
4. [Advanced Algorithms](#advanced-algorithms)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Deep Q-Networks (DQN)

## 📘 Motivation

**Problem**: Q-learning with function approximation unstable.

**Solution**: **Experience replay** + **Target network**

## 🧮 DQN Architecture

**Q-network**: Neural network Q(s,a;θ) approximates Q-values.

**Input**: State s  
**Output**: Q-value for each action

## 🧮 Experience Replay

**Replay buffer** D: Store transitions (s, a, r, s', done)

**Benefits**:
1. **Breaks correlation**: Samples i.i.d
2. **Reuses experience**: Sample efficiency
3. **Stabilizes training**: Reduces variance

**Algorithm**:
```
Store (s,a,r,s') in D
Sample minibatch from D
Update Q-network on minibatch
```

## 🧮 Target Network

**Problem**: Chasing moving target (Q-values change during training)

**Solution**: Separate target network Q̂(s,a;θ⁻)

**Update**:
```
y = r + γ max_{a'} Q̂(s',a';θ⁻)  # Use target network
L = (Q(s,a;θ) - y)²  # Update main network
```

**Synchronize**: θ⁻ ← θ every C steps (e.g., C=10,000)

## 🧪 DQN Implementation

```python
import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.action_dim = action_dim
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.q_network(state)
                return q_values.argmax().item()
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

---

# Policy Gradient Deep RL

## 🧮 Advantage Actor-Critic (A2C)

**Actor**: Policy network π_θ(a|s)  
**Critic**: Value network V_φ(s)

**Advantage**:
```
A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)
```

**Actor update**:
```
θ ← θ + α ∇_θ log π_θ(a|s) A(s,a)
```

**Critic update**:
```
φ ← φ - β ∇_φ (V_φ(s) - (r + γV_φ(s')))²
```

---

# Actor-Critic Methods

## 📊 A3C (Asynchronous Advantage Actor-Critic)

**Key innovation**: Multiple parallel agents

**Benefits**:
- Decorrelates experience (no replay buffer needed)
- Faster training (parallel exploration)

**Algorithm**:
```
Multiple workers in parallel:
  Each worker has copy of network
  Collect experience
  Compute gradients
  Asynchronously update global network
```

---

# Advanced Algorithms

## 🧮 PPO (Proximal Policy Optimization)

**Problem**: Large policy updates can be harmful.

**Solution**: Clip objective to limit update size.

**Clipped objective**:
```
L^CLIP(θ) = min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)

where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
```

**ε typically 0.2**: Limits ratio to [0.8, 1.2]

## 🧮 DDPG (Deep Deterministic Policy Gradient)

**For continuous actions**.

**Deterministic policy**: μ_θ(s) → a

**Q-function critic**: Q_φ(s,a)

**Actor update**:
```
∇_θ J ≈ E[∇_a Q_φ(s,a)|_{a=μ_θ(s)} ∇_θ μ_θ(s)]
```

**Uses**: Target networks + replay buffer (like DQN)

---

# 🔥 MCQs

### Q1. DQN uses:
**Options:**
- A) Only replay buffer
- B) Experience replay + target network ✓
- C) Only target network
- D) Neither

**Explanation**: Both are critical for DQN stability.

---

### Q2. Experience replay:
**Options:**
- A) Increases correlation
- B) Breaks correlation ✓
- C) Slows training
- D) Not useful

**Explanation**: Samples i.i.d from buffer, breaks temporal correlation.

---

### Q3. PPO clips:
**Options:**
- A) Rewards
- B) Policy ratio ✓
- C) Q-values
- D) States

**Explanation**: Clips π_new/π_old to prevent large updates.

---

### Q4. A3C uses:
**Options:**
- A) Single agent
- B) Multiple parallel agents ✓
- C) Replay buffer
- D) Target network

**Explanation**: Asynchronous parallel workers.

---

### Q5. DDPG is for:
**Options:**
- A) Discrete actions
- B) Continuous actions ✓
- C) Both
- D) Neither

**Explanation**: Deterministic policy for continuous action space.

---

# ⚠️ Common Mistakes

1. **Not using target network**: Chasing moving target problem
2. **Small replay buffer**: Need sufficient size (e.g., 100K-1M)
3. **Wrong update frequency**: Update target network periodically, not every step
4. **Ignoring gradient clipping**: Can prevent exploding gradients
5. **Too large PPO clip ratio**: ε=0.2 is standard
6. **Not normalizing states**: Neural networks need normalized inputs
7. **Wrong discount factor**: γ=0.99 typical for most tasks
8. **Insufficient exploration**: Decay ε over time in DQN

---

# ⭐ One-Line Exam Facts

1. **DQN**: Deep Q-Network with experience replay + target network
2. **Experience replay**: Store and sample transitions from buffer
3. **Target network**: Q̂(s,a;θ⁻) updated every C steps
4. **DQN loss**: (Q(s,a) - [r + γ max Q̂(s',a')])²
5. **A2C**: Actor-Critic with advantage A(s,a) = Q(s,a) - V(s)
6. **A3C**: Asynchronous parallel agents (no replay buffer)
7. **PPO**: Clip policy ratio to [1-ε, 1+ε]
8. **DDPG**: Deterministic policy for continuous actions
9. **Advantage**: Reduces variance in policy gradient
10. **Target network sync**: θ⁻ ← θ every C steps
11. **Replay buffer capacity**: Typically 100K-1M transitions
12. **TD error**: δ = r + γV(s') - V(s)
13. **Actor**: Outputs policy π_θ(a|s)
14. **Critic**: Estimates value V_φ(s) or Q_φ(s,a)
15. **PPO ε**: Typically 0.2 (clip to [0.8, 1.2])

---

**End of Session 27**
