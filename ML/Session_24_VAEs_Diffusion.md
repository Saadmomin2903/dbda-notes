# Session 24 – VAEs & Diffusion Models

## 📚 Table of Contents
1. [Variational Autoencoders](#variational-autoencoders)
2. [Diffusion Models](#diffusion-models)
3. [Stable Diffusion](#stable-diffusion)
4. [Comparison](#comparison)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Variational Autoencoders

## 🧮 Architecture

```
Encoder: x → μ(x), σ²(x)
Sample: z ~ N(μ, σ²)
Decoder: z → x̂
```

**Key idea**: Learn continuous latent space.

## 🧮 Loss Function

```
L = Reconstruction_loss + KL_divergence

L = -E[log p(x|z)] + KL(q(z|x) || p(z))
```

**Reconstruction**: How well decoder reconstructs input
**KL divergence**: How close q(z|x) to prior p(z) = N(0,I)

**Expanded**:
```
KL = -0.5 Σ(1 + log σ² - μ² - σ²)
```

## 🧮 Reparameterization Trick

**Problem**: Can't backprop through sampling z ~ N(μ, σ²).

**Solution**: Reparameterize
```
z = μ + σ ⊙ ε where ε ~ N(0,I)
```

Now gradient flows through μ and σ.

## 🧪 Implementation

```python
class VAE(nn.Module):
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
```

---

# Diffusion Models

## 📘 Core Idea

**Forward process**: Gradually add Gaussian noise
**Reverse process**: Learn to denoise step by step

## 🧮 Forward Diffusion

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Over T steps (e.g., T=1000):
```
x_0 → x_1 → x_2 → ... → x_T ≈ N(0,I)
```

**Closed form** (using reparameterization):
```
x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
where ᾱ_t = ∏_{s=1}^t (1-β_s)
```

## 🧮 Reverse Diffusion

**Learn**: p_θ(x_{t-1} | x_t)

**Model predicts noise** ε_θ(x_t, t):
```
x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + σ_t z
```

## 🧮 Training Objective

**Simple loss** (Predicted noise vs actual noise):
```
L = E_t,x_0,ε[||ε - ε_θ(x_t, t)||²]
```

**Algorithm**:
```
1. Sample x_0 from data
2. Sample t ~ Uniform(1, T)
3. Sample ε ~ N(0,I)
4. Compute x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
5. Predict ε̂ = ε_θ(x_t, t)
6. Loss = ||ε - ε̂||²
```

## 🧮 Sampling

**Start from noise x_T ~ N(0,I), denoise T steps**:
```
for t = T down to 1:
    ε_pred = ε_θ(x_t, t)
    x_{t-1} = denoise(x_t, ε_pred, t)
return x_0
```

---

# Stable Diffusion

## 📘 Latent Diffusion

**Key innovation**: Diffusion in **latent space** (not pixel space).

**Architecture**:
```
VAE Encoder: Image → Latent z
Diffusion: Apply diffusion on z
VAE Decoder: Latent → Image
```

**Advantages**:
- Much faster (lower dimension)
- Less compute
- Better quality

## 🧮 Text Conditioning

**CLIP text encoder**: Text → embedding

**Cross-attention** in diffusion model:
```
Q = features from diffusion model
K, V = text embeddings from CLIP
Attention(Q, K, V) guides generation
```

## 🧪 Stable Diffusion Pipeline

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1"
)

prompt = "A cat wearing sunglasses on the beach"
image = pipe(prompt, num_inference_steps=50).images[0]
```

## 📊 Guidance

**Classifier-Free Guidance**:
```
ε_guided = ε_unconditional + w(ε_conditional - ε_unconditional)
```

Higher w → stronger prompt adherence (typical: w=7.5).

---

# Comparison

## VAE vs Diffusion

| Aspect | VAE | Diffusion |
|--------|-----|-----------|
| **Training** | Single pass | Iterative denoising |
| **Sampling** | Fast (single pass) | Slow (many steps) |
| **Quality** | Lower | Higher ✓ |
| **Latent space** | Structured ✓ | No explicit latent |
| **Mode coverage** | Can miss modes | Better coverage ✓ |

## GAN vs Diffusion

| Aspect | GAN | Diffusion |
|--------|-----|-----------|
| **Training** | Unstable | Stable ✓ |
| **Mode collapse** | Common | Rare ✓ |
| **Sampling** | Fast ✓ | Slow |
| **Quality** | High | Higher ✓ |
| **Diversity** | Lower | Higher ✓ |

---

# 🔥 MCQs

### Q1. VAE uses:
**Options:**
- A) GAN loss
- B) Reconstruction + KL divergence ✓
- C) Only MSE
- D) Cross-entropy

**Explanation**: VAE loss = reconstruction + KL regularization.

---

### Q2. Reparameterization trick:
**Options:**
- A) z = μ + σ ⊙ ε ✓
- B) z = μ × σ
- C) z ~ N(μ, σ²)
- D) z = μ - σ

**Explanation**: Allows backprop through sampling.

---

### Q3. Diffusion forward process:
**Options:**
- A) Removes noise
- B) Adds noise gradually ✓
- C) Generates images
- D) Trains classifier

**Explanation**: Forward process corrupts data with noise.

---

### Q4. Stable Diffusion operates in:
**Options:**
- A) Pixel space
- B) Latent space ✓
- C) Frequency domain
- D) No space

**Explanation**: Latent diffusion for efficiency.

---

### Q5. Diffusion sampling is:
**Options:**
- A) Single step
- B) Iterative denoising ✓
- C) Random
- D) Instant

**Explanation**: Reverse diffusion takes T steps.

---

# ⚠️ Common Mistakes

1. **Forgetting reparameterization in VAE**: Can't train without it
2. **Wrong KL weight**: Too high → blurry images
3. **Not enough diffusion steps**: Poor quality
4. **Using pixel space for diffusion**: Very slow
5. **Ignoring guidance scale**: Critical for prompt adherence
6. **Confusing forward/reverse process**: Forward adds noise, reverse removes
7. **Not normalizing inputs**: Both VAE and diffusion need normalized data

---

# ⭐ One-Line Exam Facts

1. **VAE loss**: Reconstruction + KL(q(z|x) || p(z))
2. **Reparameterization**: z = μ + σ ⊙ ε where ε ~ N(0,I)
3. **Forward diffusion**: Gradually add noise to x_0
4. **Reverse diffusion**: Learn to denoise from x_T to x_0
5. **Diffusion loss**: ||ε - ε_θ(x_t, t)||² (predict noise)
6. **Stable Diffusion**: Latent diffusion + CLIP text conditioning
7. **Classifier-free guidance**: Strengthen prompt adherence
8. **VAE latent space**: Continuous, structured
9. **Diffusion steps**: Typically T=1000 for training, 50+ for sampling
10. **Cross-attention**: Condition diffusion on text embeddings

---

**End of Session 24**
