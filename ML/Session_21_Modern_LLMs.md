# Session 21 – Modern Large Language Models

## 📚 Table of Contents
1. [Scaling Laws](#scaling-laws)
2. [LLM Architectures](#llm-architectures)
3. [Training Techniques](#training-techniques)
4. [Emergent Abilities](#emergent-abilities)
5. [In-Context Learning](#in-context-learning)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Scaling Laws

## 📘 Concept Overview

**Scaling laws**: Predictable relationship between model performance and size.

## 🧮 Mathematical Relationship

**Power law**:
```
Loss ∝ N^(-α)
```

Where:
- N = number of parameters
- α ≈ 0.076 for language modeling

**Three factors scale together**:
1. Model size (parameters)
2. Dataset size (tokens)
3. Compute budget (FLOPs)

**Chinchilla scaling**: For compute-optimal training, scale data proportionally with parameters.

**Example**: 70B parameter model → train on ~1.4T tokens.

---

# LLM Architectures

## 📊 Major Models

### GPT-3.5/4 (OpenAI)

**GPT-3**:
- 175B parameters
- 96 layers, 12,288 hidden size
- Context: 2048 → 4096 tokens

**GPT-4**:
- Multimodal (text + images)
- Improved reasoning
- Longer context (8K/32K)

### LLaMA (Meta)

**Open-source family**:
- 7B, 13B, 33B, 65B parameters
- Efficient architecture
- Rotary positional embeddings (RoPE)
- SwiGLU activation

### Claude (Anthropic)

**Constitutional AI**:
- Self-critique mechanism
- 100K token context
- Harmlessness + helpfulness training

### PaLM 2 (Google)

**Pathways Language Model**:
- Multilingual (100+ languages)
- Improved reasoning
- Efficient serving

## 🧮 Architecture Improvements

### Rotary Position Embeddings (RoPE)

```
RoPE(x, m) = [x₁ cos(mθ₁) - x₂ sin(mθ₁),
              x₁ sin(mθ₁) + x₂ cos(mθ₁), ...]
```

**Advantages**:
- Relative position encoding
- Better extrapolation to longer sequences

### SwiGLU Activation

```
SwiGLU(x) = Swish(xW) ⊙ (xV)
where Swish(x) = x·σ(x)
```

Replaces ReLU in FFN, improves performance.

---

# Training Techniques

## 📊 Instruction Tuning

**Goal**: Make models follow instructions.

**Dataset format**:
```
Instruction: "Translate to French: Hello"
Output: "Bonjour"
```

**Methods**:
- Supervised fine-tuning on instruction-output pairs
- Self-instruct (model generates its own training data)

## 🧮 RLHF (Reinforcement Learning from Human Feedback)

### Three-Stage Process

**Stage 1: Supervised Fine-Tuning (SFT)**
```
Train on high-quality human demonstrations
Loss = -Σ log P(y|x)
```

**Stage 2: Reward Model Training**
```
Human ranks multiple outputs for same input
Train model to predict preferences:
Loss = -E[log σ(r(x, y_win) - r(x, y_lose))]
```

**Stage 3: PPO Optimization**
```
Optimize policy to maximize reward:
L = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] - β·KL(π_θ || π_ref)
```

**KL penalty**: Prevents drifting too far from reference model.

## 📊 Constitutional AI

**Self-improvement process**:
```
1. Generate initial response
2. Critique based on principles (constitution)
3. Revise response
4. Repeat until satisfactory
```

**Principles example**:
- "Choose response that is helpful and harmless"
- "Avoid stereotypes and discrimination"

---

# Emergent Abilities

## 📘 Abilities at Scale

**Emergent abilities**: Capabilities that appear suddenly above certain scale.

### Chain-of-Thought Reasoning

**Standard prompting**:
```
Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?
A: 7
```

**Chain-of-thought**:
```
Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?
A: Roger started with 5 balls. He bought 2 more.
   5 + 2 = 7. So he has 7 tennis balls.
```

**Trigger**: "Let's think step by step"

### Few-Shot Learning

Learn from examples in prompt:
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese => [model predicts: fromage]
```

**In-context learning**: No parameter updates!

---

# In-Context Learning

## 🧮 Mechanism

**How it works** (hypothesis):
- Model recognizes pattern in prompt
- Applies learned pattern to new example
- Similar to meta-learning

**Factors affecting performance**:
1. **Example quality**: Better examples → better performance
2. **Order**: Performance varies with example order
3. **Number of examples**: More examples (up to context limit)
4. **Task familiarity**: Seen during pre-training

## 🧪 Prompt Engineering

### Techniques

**Zero-shot**:
```
Classify sentiment: "This movie was terrible"
Answer: Negative
```

**Few-shot**:
```
Classify sentiment:
"I loved it" → Positive
"Boring" → Negative  
"Amazing!" → Positive
"This movie was terrible" → [model predicts: Negative]
```

**Chain-of-thought**:
```
Q: There are 15 trees. We plant 8 more. Then 5 die. How many trees?
A: Initially 15 trees. Plant 8 more: 15+8=23.
   Then 5 die: 23-5=18. Answer: 18 trees.
```

---

# 🔥 MCQs

### Q1. Scaling laws show performance scales as:
**Options:**
- A) Linear with parameters
- B) Power law with parameters ✓
- C) Exponential
- D) No relationship

**Explanation**: Loss ∝ N^(-α), power law relationship.

---

### Q2. RLHF Stage 2 trains:
**Options:**
- A) Language model
- B) Reward model ✓
- C) Instruction model
- D) Classifier

**Explanation**: Reward model learns to predict human preferences.

---

### Q3. Chain-of-thought prompting:
**Options:**
- A) Requires fine-tuning
- B) Emerges at scale ✓
- C) Works for small models
- D) Not useful

**Explanation**: CoT reasoning emerges in large models (>100B params).

---

### Q4. Constitutional AI uses:
**Options:**
- A) Human feedback only
- B) Self-critique and revision ✓
- C) Supervised learning
- D) Adversarial training

**Explanation**: Model critiques/revises based on principles.

---

### Q5. In-context learning:
**Options:**
- A) Updates model weights
- B) Learns from prompt examples only ✓
- C) Requires training
- D) Doesn't work

**Explanation**: No weight updates, learns pattern from context.

---

# ⚠️ Common Mistakes

1. **Ignoring scaling laws**: Assuming bigger is always better (need proportional data)
2. **Over-relying on prompts**: Some tasks need fine-tuning
3. **Not understanding RLHF stages**: All three stages critical
4. **Expecting emergent abilities in small models**: Need scale
5. **Poor prompt engineering**: Example quality matters greatly
6. **Ignoring context limits**: Most models have 2K-32K token limits
7. **Not using chain-of-thought**: Improves reasoning significantly
8. **Forgetting KL penalty in RLHF**: Prevents mode collapse

---

# ⭐ One-Line Exam Facts

1. **Scaling laws**: Performance ∝ N^(-α) (power law)
2. **Chinchilla optimal**: Scale data proportionally with parameters
3. **RLHF**: SFT → Reward model → PPO optimization
4. **Constitutional AI**: Self-critique based on principles
5. **Emergent abilities**: Chain-of-thought, few-shot at scale
6. **In-context learning**: Learn from examples in prompt (no weight updates)
7. **GPT-4**: Multimodal, 8K/32K context
8. **LLaMA**: Open-source, 7B-65B parameters
9. **Claude**: 100K context, Constitutional AI
10. **RoPE**: Rotary position embeddings, better extrapolation
11. **SwiGLU**: Improved activation function for LLMs
12. **Few-shot**: Examples in prompt guide inference
13. **Chain-of-thought**: "Let's think step by step"
14. **PPO**: Policy optimization in RLHF (stage 3)
15. **KL divergence**: Regularization in RLHF training

---

**End of Session 21**
