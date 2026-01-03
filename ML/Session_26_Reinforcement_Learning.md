# Session 26 – Reinforcement Learning Fundamentals

## 📚 Table of Contents
1. [MDP Framework](#mdp-framework)
2. [Value Functions](#value-functions)
3. [Bellman Equations](#bellman-equations)
4. [Q-Learning](#q-learning)
5. [Policy Gradient Methods](#policy-gradient-methods)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# MDP Framework

## 📘 Markov Decision Process

**Components**:
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition function P(s'|s,a)
- **R**: Reward function R(s,a)
- **γ**: Discount factor ∈ [0,1]

**Markov Property**: Future depends only on current state, not history.

## 🧮 Goal

Find optimal policy π* that maximizes expected cumulative reward:
```
π* = argmax_π E[Σ_{t=0}^∞ γ^t r_t | π]
```

**Discount factor γ**:
- γ = 0: Only immediate reward matters
- γ → 1: Future rewards important
- γ < 1: Ensures convergence for infinite horizon

---

# Value Functions

## 🧮 State Value Function

**V^π(s)**: Expected return starting from state s, following policy π
```
V^π(s) = E_π[Σ_{t=0}^∞ γ^t r_t | s_0 = s]
```

## 🧮 Action Value Function

**Q^π(s,a)**: Expected return starting from s, taking action a, then following π
```
Q^π(s,a) = E_π[Σ_{t=0}^∞ γ^t r_t | s_0 = s, a_0 = a]
```

**Relationship**:
```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

## 🧮 Optimal Value Functions

```
V*(s) = max_π V^π(s)
Q*(s,a) = max_π Q^π(s,a)
```

**Optimal policy**: π*(s) = argmax_a Q*(s,a)

---

# Bellman Equations

## 🧮 Bellman Expectation Equation

**For V^π**:
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a) + γV^π(s')]
```

**For Q^π**:
```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a) + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

## 🧮 Bellman Optimality Equation

**For V***:
```
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]
```

**For Q***:
```
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a) + γ max_{a'} Q*(s',a')]
```

---

# Q-Learning

## 📘 Off-Policy TD Control

**Update rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
```

Where:
- α = learning rate
- r = observed reward
- s' = next state
- Target: r + γ max_{a'} Q(s',a')

## 🧮 Q-Learning Algorithm

```
Initialize Q(s,a) arbitrarily
for each episode:
    Initialize s
    for each step:
        Choose a using ε-greedy policy derived from Q
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
        s ← s'
    until s is terminal
```

**ε-greedy exploration**:
```
a = {random action with probability ε
    {argmax_a Q(s,a) with probability 1-ε
```

## 🧪 Python Implementation

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
    
    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        """Q-learning update."""
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, env, n_episodes=1000):
        """Train agent."""
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
```

## 📊 SARSA (On-Policy Alternative)

**Update rule**:
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

**Difference from Q-Learning**:
- Q-learning: Uses max_a' Q(s',a') (off-policy)
- SARSA: Uses actual next action a' (on-policy)

---

# Policy Gradient Methods

## 📘 Direct Policy Optimization

**Parameterize policy**: π_θ(a|s)

**Objective**: Maximize expected return
```
J(θ) = E_π[Σ γ^t r_t]
```

## 🧮 Policy Gradient Theorem

```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q^π(s,a)]
```

## 🧮 REINFORCE Algorithm

**Monte Carlo policy gradient**:
```
θ ← θ + α ∇_θ log π_θ(a_t|s_t) G_t
```

Where G_t = Σ_{k=t}^T γ^{k-t} r_k (return from time t)

**With baseline** (reduce variance):
```
θ ← θ + α ∇_θ log π_θ(a_t|s_t) [G_t - b(s_t)]
```

Common baseline: b(s) = V(s)

## 🧪 REINFORCE Implementation

```python
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state):
        """Sample action from policy."""
        probs = self.policy(torch.FloatTensor(state))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update(self, log_probs, rewards, gamma=0.99):
        """REINFORCE update."""
        returns = []
        G = 0
        
        # Compute returns
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        
        loss = torch.stack(loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

# 🔥 MCQs

### Q1. MDP Markov property:
**Options:**
- A) Depends on all history
- B) Depends only on current state ✓
- C) Random
- D) Not defined

**Explanation**: Future independent of past given present state.

---

### Q2. Q-Learning is:
**Options:**
- A) On-policy
- B) Off-policy ✓
- C) Model-based
- D) Supervised

**Explanation**: Uses max Q(s',a'), not actual next action.

---

### Q3. Discount factor γ → 1 means:
**Options:**
- A) Only immediate reward
- B) Future rewards important ✓
- C) No discounting
- D) Random

**Explanation**: Higher γ gives more weight to future rewards.

---

### Q4. Policy gradient optimizes:
**Options:**
- A) Value function
- B) Policy directly ✓
- C) Q-function
- D) Transition model

**Explanation**: Directly parameterizes and optimizes policy π_θ.

---

### Q5. ε-greedy balances:
**Options:**
- A) Speed vs accuracy
- B) Exploration vs exploitation ✓
- C) Bias vs variance
- D) Memory vs compute

**Explanation**: ε = exploration, 1-ε = exploitation.

---

# ⚠️ Common Mistakes

1. **Confusing Q-learning and SARSA**: Q-learning off-policy, SARSA on-policy
2. **Wrong learning rate**: Too high → instability, too low → slow
3. **Forgetting discount factor**: γ critical for long-term planning
4. **Not using baseline in REINFORCE**: High variance without it
5. **Insufficient exploration**: ε-greedy or other exploration needed
6. **Ignoring terminal states**: Q(terminal, ·) = 0
7. **Wrong Bellman equation**: Expectation vs optimality
8. **Not decaying ε**: Should decrease ε over time

---

# ⭐ One-Line Exam Facts

1. **MDP**: (S, A, P, R, γ) framework
2. **V^π(s)**: Expected return from state s following π
3. **Q^π(s,a)**: Expected return from (s,a) then following π
4. **Bellman expectation**: V^π(s) = Σ_a π(a|s)[R + γΣ_{s'} P(s'|s,a)V^π(s')]
5. **Bellman optimality**: V*(s) = max_a[R + γΣ_{s'} P(s'|s,a)V*(s')]
6. **Q-learning**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
7. **Off-policy**: Q-learning (learns optimal policy while following different policy)
8. **On-policy**: SARSA (learns policy being followed)
9. **Policy gradient**: ∇J(θ) = E[∇log π_θ(a|s) Q^π(s,a)]
10. **REINFORCE**: Monte Carlo policy gradient
11. **ε-greedy**: Exploration-exploitation tradeoff
12. **Discount γ**: Weight on future rewards
13. **TD error**: δ = r + γV(s') - V(s)
14. **Baseline**: Reduces variance in policy gradient
15. **Optimal policy**: π*(s) = argmax_a Q*(s,a)

---

**End of Session 26**
