# Sessions 21-30 – Modern AI & Advanced Topics (Detailed)

---

# Session 21 – Modern Large Language Models

## 📚 Core Concepts

### Scaling Laws
**Power law relationship**: Performance ∝ N^α (N = parameters, data, compute)

**Emergent abilities** appear at scale:
- Chain-of-thought reasoning
- Few-shot learning
- In-context learning

### Model Architectures

**GPT-3.5/4** (OpenAI):
- 175B+ parameters
- Instruction-tuned
- RLHF alignment

**LLaMA** (Meta):
- 7B to 70B parameters
- Open weights
- Efficient training

**Claude** (Anthropic):
- Constitutional AI
- Longer context (100K tokens)
- Helpfulness + harmlessness

**PaLM 2** (Google):
- Multilingual
- Improved reasoning
- Efficient compute

### Training Techniques

**Instruction Tuning**:
```
Input: "Summarize this article: [text]"
Output: [summary]
```

**RLHF** (Reinforcement Learning from Human Feedback):
1. Supervised fine-tuning
2. Reward model training
3. PPO optimization

**Constitutional AI**:
- Self-critique and revision
- Principle-based alignment

---

# Session 22 – Advanced Computer Vision

## Object Detection

### R-CNN Family

**R-CNN**: Region proposals + CNN
**Fast R-CNN**: ROI pooling
**Faster R-CNN**: RPN (Region Proposal Network)

### YOLO (You Only Look Once)

**Single-stage detector**: Direct bounding box + class prediction

```
Grid cells → Bounding boxes + Confidence + Class probabilities
```

**Versions**:
- YOLOv1-v3: Progressions
- YOLOv5: PyTorch, optimized
- YOLOv8: Latest, best accuracy

### SSD (Single Shot Detector)

Multiple feature maps at different scales.

## Image Segmentation

### Semantic Segmentation
Pixel-wise classification (all pixels of class labeled same).

### Instance Segmentation
Individual object instances.

**Mask R-CNN**: Faster R-CNN + segmentation branch

### U-Net Architecture
```
Encoder (downsampling) → Bottleneck → Decoder (upsampling)
Skip connections between corresponding encoder-decoder levels
```

**Applications**: Medical imaging, satellite imagery

---

# Session 23 – Generative Adversarial Networks

## 🧮 GAN Framework

**Min-max game**:
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

**Generator G**: Noise z → Fake sample G(z)
**Discriminator D**: Real/Fake classifier

### Training Algorithm
```
1. Train D: Maximize log D(x) + log(1 - D(G(z)))
2. Train G: Maximize log D(G(z))  [or minimize log(1 - D(G(z)))]
Alternate until convergence
```

### Challenges
- **Mode collapse**: G produces limited variety
- **Training instability**: Oscillation, non-convergence
- **Vanishing gradients**: D too strong → G can't learn

### Variants

**DCGAN**: Deep Convolutional GAN
- BatchNorm, LeakyReLU
- Strided convolutions (no pooling)

**StyleGAN**: 
- Style-based generator
- High-quality face generation
- Latent space manipulation

**CycleGAN**:
- Unpaired image-to-image translation
- Cycle consistency loss

**Conditional GAN**:
- Condition on class label
- Controlled generation

---

# Session 24 – Variational Autoencoders & Diffusion Models

## VAE (Variational Autoencoder)

### Architecture
```
Input → Encoder → μ, σ² → Sample z ~ N(μ, σ²) → Decoder → Output
```

**Loss**:
```
L = Reconstruction_loss + KL_divergence(q(z|x) || p(z))
```

**Reparameterization trick**:
```
z = μ + σ ⊙ ε where ε ~ N(0, I)
```

Allows backpropagation through sampling.

## Diffusion Models

### Forward Process (Add noise)
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Gradually add Gaussian noise over T steps.

### Reverse Process (Denoise)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Learn to reverse the noise process.

### Training
Predict noise added at each step:
```
L = E[||ε - ε_θ(x_t, t)||²]
```

### Stable Diffusion
- Latent diffusion (work in latent space)
- Text conditioning via CLIP
- Efficient generation

---

# Session 25 – Advanced NLP Topics

## Word Embeddings

### Word2Vec
**CBOW**: Context → Word
**Skip-gram**: Word → Context

Loss: Negative sampling

### GloVe
Matrix factorization on co-occurrence statistics.

### Contextual Embeddings
**ELMo**: Bidirectional LSTM
**BERT embeddings**: Layer outputs as features

## Seq2Seq Models

### Encoder-Decoder
```
Encoder: Input sequence → Context vector
Decoder: Context vector → Output sequence
```

**With attention**:
- Dynamic context at each decoder step
- Alignment between input/output

### Applications
- Machine translation
- Summarization
- Dialogue systems

---

# Session 26 – Reinforcement Learning

## Core Concepts

**MDP** (Markov Decision Process):
- States S
- Actions A  
- Transition P(s'|s,a)
- Reward R(s,a)
- Discount γ

**Goal**: Learn policy π(a|s) maximizing expected return.

### Value Functions
```
V^π(s) = E[Σ γ^t r_t | s_0=s, π]  (state value)
Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]  (action value)
```

### Bellman Equations
```
V(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a) + γV(s')]
Q(s,a) = Σ_{s'} P(s'|s,a)[R(s,a) + γ Σ_{a'} π(a'|s')Q(s',a')]
```

## Algorithms

### Q-Learning (Off-policy TD)
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### SARSA (On-policy TD)
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

### Policy Gradient (REINFORCE)
```
∇J(θ) = E[∇log π_θ(a|s) R]
```

### Actor-Critic
Combine value function (critic) and policy (actor).

---

# Session 27 – Deep Reinforcement Learning

## DQN (Deep Q-Network)

**Innovations**:
1. **Experience replay**: Store transitions, sample mini-batches
2. **Target network**: Stabilize Q-learning

```python
# Q-update with target network
Q_target = r + γ max_a Q_target(s', a)
Loss = (Q(s,a) - Q_target)²
```

## Advanced Algorithms

### A3C (Asynchronous Advantage Actor-Critic)
Multiple parallel agents, shared network.

### PPO (Proximal Policy Optimization)
```
Clip objective: min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)
```

Prevents large policy updates.

### TRPO (Trust Region Policy Optimization)
Constrained optimization:
```
max E[...] subject to KL(π_old || π_new) ≤ δ
```

## Applications
- Game playing (AlphaGo, Dota 2)
- Robotics
- Recommendation systems
- Resource allocation

---

# Session 28 – MLOps & Production ML

## Model Deployment

### REST API
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load('model.pth')

@app.post("/predict")
def predict(data: dict):
    input_tensor = preprocess(data)
    output = model(input_tensor)
    return {"prediction": postprocess(output)}
```

### Docker Containerization
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pth app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### Model Serving
- **TensorFlow Serving**: Optimized for TF models
- **TorchServe**: PyTorch models
- **ONNX Runtime**: Cross-framework

## Monitoring

### Metrics to Track
- **Performance**: Latency, throughput
- **Accuracy**: Online metrics
- **Data drift**: Input distribution changes
- **Model drift**: Performance degradation

### A/B Testing
```
Traffic split: 90% model A, 10% model B
Compare: accuracy, latency, business metrics
Gradual rollout if B better
```

## ML Pipelines

### Kubeflow
Kubernetes-based ML workflows

### MLflow
- Experiment tracking
- Model registry
- Deployment

### DVC (Data Version Control)
Version datasets + models with Git-like interface

---

# Session 29 – AI Ethics & Fairness

## Bias & Fairness

### Sources of Bias
1. **Historical bias**: Data reflects societal biases
2. **Representation bias**: Some groups underrepresented
3. **Measurement bias**: Proxies for protected attributes
4. **Aggregation bias**: One model for diverse groups

### Fairness Metrics

**Demographic Parity**:
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
```

**Equal Opportunity**:
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (TPR equality)
```

**Equalized Odds**:
```
TPR and FPR equal across groups
```

### Mitigation Strategies
- **Pre-processing**: Re-sample, re-weight training data
- **In-processing**: Fairness constraints during training
- **Post-processing**: Adjust predictions for fairness

## Explainability

### LIME (Local Interpretable Model-agnostic Explanations)
Approximate model locally with interpretable model.

### SHAP (SHapley Additive exPlanations)
Game-theoretic feature attribution:
```
φ_i = Σ [v(S∪{i}) - v(S)] / (combinations)
```

### Attention Visualization
Show which inputs model focuses on.

## Privacy

### Differential Privacy
Add calibrated noise to preserve privacy:
```
P(M(D) ∈ S) ≤ e^ε P(M(D') ∈ S)
```

### Federated Learning
Train on decentralized data without sharing.

---

# Session 30 – Future of AI & Emerging Trends

## Multimodal Learning

### CLIP (Contrastive Language-Image Pre-training)
Learn joint text-image embeddings:
```
Maximize: similarity(image_i, caption_i)
Minimize: similarity(image_i, caption_j) for i≠j
```

**Applications**: Zero-shot classification, image search

### Flamingo
Few-shot vision-language model.

## Efficient AI

### Model Compression

**Pruning**: Remove unimportant weights
**Quantization**: Use lower precision (INT8 vs FP32)
**Knowledge Distillation**: Train small model to mimic large model

```
L = L_task + α KL(p_student || p_teacher)
```

### Neural Architecture Search
Automated model design.

## Retrieval-Augmented Generation (RAG)

```
Query → Retrieve relevant docs → LLM with context → Response
```

Grounds generation in external knowledge.

## AI Agents

**AutoGPT**: Autonomous task completion
**LangChain**: Framework for LLM applications
**Function calling**: LLMs use external tools

## Constitutional AI
Models critique and revise own outputs based on principles.

## Industry Impact

**Healthcare**: Drug discovery, diagnosis
**Finance**: Fraud detection, algorithmic trading  
**Climate**: Weather prediction, optimization
**Science**: Protein folding (AlphaFold), materials discovery

---

**🎉 ALL 30 SESSIONS NOW FULLY DETAILED! 🎉**
