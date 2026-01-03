# Session 23 – Generative Adversarial Networks (GANs)

## 📚 Table of Contents
1. [GAN Framework](#gan-framework)
2. [Training Dynamics](#training-dynamics)
3. [GAN Variants](#gan-variants)
4. [Applications](#applications)
5. [Challenges](#challenges)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# GAN Framework

## 🧮 Min-Max Game

**Two players**:
- **Generator G**: Noise z → Fake sample G(z)
- **Discriminator D**: Sample → Real/Fake probability

**Objective**:
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

**Intuition**:
- D tries to maximize (distinguish real from fake)
- G tries to minimize (fool D)

## ⚙️ Training Algorithm

```
for epoch in epochs:
    # Train Discriminator
    Sample real batch from data
    Sample noise, generate fake batch
    L_D = -[log D(real) + log(1 - D(fake))]
    Update D to minimize L_D
    
    # Train Generator
    Sample noise, generate new fake batch
    L_G = -log D(fake)  # or log(1 - D(fake))
    Update G to minimize L_G
```

**Non-saturating loss** for G: Maximize log D(G(z)) instead of minimizing log(1 - D(G(z))).

---

# Training Dynamics

## 📊 Nash Equilibrium

**Optimal solution**: D(x) = 0.5 everywhere (can't distinguish).

**Proof**: At equilibrium, p_g = p_data, so D(x) = 0.5.

## ⚠️ Challenges

### 1. Mode Collapse
G produces limited variety (ignores parts of data distribution).

**Solution**: Minibatch discrimination, unrolled GAN.

### 2. Training Instability
Oscillation, non-convergence.

**Solution**: Spectral normalization, gradient penalty.

### 3. Vanishing Gradients
D too strong → G gradient vanishes.

**Solution**: Non-saturating loss, Wasserstein loss.

---

# GAN Variants

## 📊 DCGAN (Deep Convolutional GAN)

**Architecture guidelines**:
- Replace pooling with strided convolutions
- Use BatchNorm (except G output, D input)
- Remove fully connected layers
- Generator: ReLU except output (Tanh)
- Discriminator: LeakyReLU

```python
# Generator
z → Dense → Reshape → UpConv → BN → ReLU → ... → Conv → Tanh

# Discriminator
Image → Conv → LeakyReLU → ... → Conv → Sigmoid
```

## 📊 Conditional GAN (cGAN)

**Condition on class label**:
```
G(z, y) generates image of class y
D(x, y) discriminates conditioned on y
```

**Objective**:
```
min_G max_D E[log D(x,y)] + E[log(1 - D(G(z,y), y))]
```

## 📊 CycleGAN

**Unpaired image-to-image translation**.

**Two generators**:
- G: X → Y
- F: Y → X

**Cycle consistency**:
```
L_cyc = E[||F(G(x)) - x||] + E[||G(F(y)) - y||]
```

Ensures F(G(x)) ≈ x (reconstruct original).

**Applications**: Photo → painting, horse → zebra, summer → winter.

## 📊 StyleGAN

**Style-based generator**:
- Maps latent z → w (intermediate latent space)
- Injects w at multiple scales (controls different styles)

**Features**:
- High-quality face generation
- Disentangled latent space (easy control)
- Style mixing (combine features from different images)

**StyleGAN2/3**: Further improvements in quality.

## 📊 Pix2Pix

**Paired image-to-image translation**.

**Generator**: U-Net architecture
**Discriminator**: PatchGAN (classifies image patches)

**Loss**:
```
L = L_GAN + λ L_L1
```

L1 loss encourages output close to ground truth.

---

# Applications

1. **Image generation**: Faces, art, landscapes
2. **Super-resolution**: Enhance image quality
3. **Style transfer**: Photo → artistic style
4. **Data augmentation**: Generate training samples
5. **Image-to-image**: Edges → photo, sketch → realistic
6. **Video generation**: Frame synthesis
7. **Text-to-image**: DALL-E (uses diffusion now, but GAN-inspired)

---

# Challenges

## 📊 Evaluation Metrics

### Inception Score (IS)
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

Higher = better quality + diversity.

### Fréchet Inception Distance (FID)
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real Σ_gen)^(1/2))
```

Lower = closer to real distribution.

**FID most commonly used** for GAN evaluation.

---

# 🔥 MCQs

### Q1. GAN objective function:
**Options:**
- A) Minimize loss
- B) Max loss
- C) Min-max game ✓
- D) No optimization

**Explanation**: D maximizes, G minimizes (adversarial).

---

### Q2. Mode collapse means:
**Options:**
- A) D fails
- B) G produces limited variety ✓
- C) Training fast
- D) Perfect generation

**Explanation**: G ignores parts of distribution.

---

### Q3. DCGAN uses:
**Options:**
- A) Fully connected
- B) Strided convolutions ✓
- C) Pooling
- D) RNN

**Explanation**: Strided conv for up/downsampling.

---

### Q4. CycleGAN requires:
**Options:**
- A) Paired data
- B) Unpaired data ✓
- C) Labels
- D) Supervision

**Explanation**: Works with unpaired image sets.

---

### Q5. FID metric:
**Options:**
- A) Higher better
- B) Lower better ✓
- C) Always 1
- D) Not useful

**Explanation**: Measures distance between real and generated distributions.

---

# ⚠️ Common Mistakes

1. **Training D and G equally**: Often need to train D more
2. **Ignoring mode collapse**: Check sample diversity
3. **Wrong learning rates**: Typically use lower LR for GANs
4. **Not using BatchNorm**: Critical for stable training
5. **Using wrong loss**: Non-saturating loss better for G
6. **Not monitoring FID**: Best metric for GAN quality
7. **Expecting quick convergence**: GANs need many epochs
8. **Using too powerful D**: Leads to vanishing G gradients

---

# ⭐ One-Line Exam Facts

1. **GAN**: min_G max_D game between generator and discriminator
2. **Generator**: Noise z → Fake sample G(z)
3. **Discriminator**: Sample → Real/Fake probability
4. **Mode collapse**: G produces limited variety
5. **DCGAN**: Convolutional GAN with BatchNorm, no pooling
6. **cGAN**: Conditional on class labels
7. **CycleGAN**: Unpaired image translation, cycle consistency
8. **StyleGAN**: Style-based generation, high-quality faces
9. **Pix2Pix**: Paired translation, U-Net generator
10. **FID**: Fréchet Inception Distance (lower = better)
11. **IS**: Inception Score (higher = better)
12. **Nash equilibrium**: D(x) = 0.5 everywhere
13. **Non-saturating loss**: Maximize log D(G(z))
14. **Vanishing gradients**: D too strong problem
15. **PatchGAN**: Discriminator for image patches

---

**End of Session 23**
