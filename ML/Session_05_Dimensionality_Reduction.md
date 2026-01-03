# Session 5 – Dimensionality Reduction

## 📚 Table of Contents
1. [Dimensionality Reduction Overview](#dimensionality-reduction-overview)
2. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [Kernel PCA](#kernel-pca)
4. [Random Projections](#random-projections)
5. [Other Techniques](#other-techniques)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Dimensionality Reduction Overview

## 📘 Concept

**Dimensionality Reduction** transforms high-dimensional data to lower dimensions while preserving important structure.

### Goals
1. **Visualization**: Project to 2D/3D for human interpretation
2. **Noise reduction**: Remove irrelevant dimensions
3. **Computational efficiency**: Fewer features → faster training
4. **Combat curse of dimensionality**: Avoid data sparsity in high dimensions

### Curse of Dimensionality

**Problem**: As dimensions increase:
- Data becomes sparse (exponentially more volume)
- Distance metrics become less meaningful (all points equidistant)
- Sample complexity grows exponentially

**Example**: To cover unit hypercube with 10 samples per dimension:
- 1D: 10 samples
- 2D: 10² = 100 samples  
- 10D: 10¹⁰ samples (impossible!)

### Approaches

1. **Feature Selection**: Select subset of original features
   - Filter methods (correlation, mutual information)
   - Wrapper methods (forward/backward selection)
   - Embedded methods (Lasso, tree importance)

2. **Feature Extraction**: Create new features as combinations
   - Linear: PCA, LDA
   - Non-linear: Kernel PCA, t-SNE, Autoencoders

---

# Principal Component Analysis (PCA)

## 📘 Concept Overview

**PCA** finds orthogonal directions (principal components) of maximum variance in data.

### Intuition

Project data onto lower-dimensional subspace that **preserves maximum variance**.

```
Original 2D:          After PCA:
    ●                    ●
  ●   ●       →        ●●●●●  (1D)
    ●                    ●
  (scatter)          (compressed)
```

## 🧮 Mathematical Foundation

### Problem Formulation

Given data X ∈ ℝⁿˣᵈ (n samples, d features), find k < d orthonormal directions that maximize projected variance.

### Objective

```
maximize Var(Xw) = w^T Σ w
subject to ‖w‖² = 1
```

Where Σ = covariance matrix of X.

### Solution via Eigendecomposition

**Covariance matrix**:
```
Σ = (1/n) X^T X  (assuming X is centered)
```

**Eigenvalue decomposition**:
```
Σ v = λ v
```

Where:
- v = eigenvector = principal component direction
- λ = eigenvalue = variance along that direction

**Principal components** = eigenvectors sorted by eigenvalue (descending).

### Step-by-Step Algorithm

```
1. Center data: X_centered = X - mean(X)

2. Compute covariance matrix: Σ = (1/n) X_centered^T X_centered

3. Eigendecomposition: Σ V = V Λ
   - V = eigenvectors (columns)
   - Λ = diagonal matrix of eigenvalues

4. Sort eigenvectors by eigenvalues (descending)

5. Select top k eigenvectors → W ∈ ℝᵈˣᵏ

6. Project data: X_reduced = X_centered W
```

### Singular Value Decomposition (SVD) Approach

More numerically stable than eigendecomposition.

```
X = U Σ V^T
```

Where:
- U ∈ ℝⁿˣⁿ: left singular vectors
- Σ ∈ ℝⁿˣᵈ: singular values (diagonal)
- V ∈ ℝᵈˣᵈ: right singular vectors (**principal components**)

**Relationship**: V from SVD = eigenvectors of X^T X

## 🧠 Key Concepts

### Explained Variance

Proportion of total variance explained by each component:

```
explained_variance_k = λₖ / Σᵢ λᵢ
```

**Cumulative explained variance**: Sum of first k components

```
cumulative_variance_k = (Σᵢ₌₁ᵏ λᵢ) / (Σᵢ₌₁ᵈ λᵢ)
```

### Choosing Number of Components k

**Rule of thumb**: Keep components that explain 80-95% of variance.

```python
# Scree plot
plt.plot(range(1, len(pca.explained_variance_) + 1), 
         pca.explained_variance_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')

# Cumulative variance
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
```

## ⚙️ PCA Algorithm Variants

### 1. Standard PCA

```python
from sklearn.decomposition import PCA

# Fit PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Access components
print(f"Principal Components:\n{pca.components_}")
print(f"Explained Variance: {pca.explained_variance_}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Inverse transform (reconstruct)
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
```

### 2. Incremental PCA

For datasets too large for memory.

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=100)

# Fit in batches
for batch in get_batches(X, batch_size=100):
    ipca.partial_fit(batch)

X_reduced = ipca.transform(X)
```

### 3. Randomized PCA

Faster approximation using randomized SVD (for large datasets).

```python
pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
X_reduced = pca.fit_transform(X)
```

### 4. Sparse PCA

Forces principal components to be sparse (many zero loadings).

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=10, alpha=0.1, random_state=42)
X_reduced = spca.fit_transform(X)
```

## 🧪 Complete Example

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data (64-dimensional)
X, y = load_digits(return_X_y=True)
print(f"Original shape: {X.shape}")

# Standardize (important for PCA!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)

print(f"Reduced shape: {X_reduced.shape}")
print(f"Components needed for 95% variance: {pca.n_components_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

# Visualize first 2 components
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.colorbar(scatter, label='Digit')
plt.title('PCA Projection of Digits Dataset')
plt.show()
```

## ⚠️ Assumptions & Limitations

### Assumptions
1. **Linearity**: Assumes linear combinations capture structure
2. **Variance = Information**: High variance directions are important
3. **Orthogonality**: Components must be perpendicular
4. **Gaussian-ish data**: Works best for roughly Gaussian distributions

### Limitations

1. **Linear only**: Can't capture non-linear manifolds
   ```
   Swiss roll:  ████    PCA fails to "unroll"
              ██████    
               ████     
   ```

2. **Sensitive to scale**: **Must standardize** features first

3. **Interpretability**: Components are linear combinations (hard to interpret)

4. **Outliers**: Sensitive to extreme values (influence covariance)

### When PCA Fails

- **Non-linear structure**: Use Kernel PCA, t-SNE, or autoencoders
- **Sparse high-dim data**: Use Sparse PCA or feature selection
- **Categorical features**: PCA assumes continuous variables

## 📊 PCA Applications

| Application | Why PCA? |
|-------------|----------|
| **Image compression** | Reduce pixel dimensions while preserving visual info |
| **Visualization** | Project high-dim to 2D/3D |
| **Noise filtering** | Keep top components, discard noisy low-variance ones |
| **Preprocessing** | Speed up ML algorithms (fewer features) |
| **Multicollinearity** | Remove correlated features (components are orthogonal) |
| **Eigenfaces** | Face recognition (PCA on face images) |

---

# Kernel PCA

## 📘 Concept Overview

**Kernel PCA** applies PCA in high-dimensional feature space via kernel trick.

Captures **non-linear** structure.

## 🧮 Mathematical Foundation

### Kernel Trick

Map data to high-dimensional space φ(x), then apply PCA:

```
X → φ(X) → PCA
```

**Problem**: φ(X) may be infinite-dimensional!

**Solution**: Never explicitly compute φ(X), only kernel matrix:

```
K_{ij} = k(xᵢ, xⱼ) = φ(xᵢ)^T φ(xⱼ)
```

### Common Kernels

#### 1. RBF (Gaussian) Kernel

```
k(x, y) = exp(-γ ‖x - y‖²)
```

**Most popular**, captures smooth non-linear manifolds.

#### 2. Polynomial Kernel

```
k(x, y) = (x^T y + c)ᵈ
```

Captures polynomial relationships of degree d.

#### 3. Sigmoid Kernel

```
k(x, y) = tanh(α x^T y + c)
```

Similar to neural network activation.

## ⚙️ Kernel PCA Algorithm

```
1. Compute kernel matrix: K_{ij} = k(xᵢ, xⱼ)

2. Center kernel matrix:
   K_centered = K - 1ₙK - K1ₙ + 1ₙK1ₙ
   (where 1ₙ = matrix of 1/n)

3. Eigendecomposition: K_centered v = λ v

4. Normalize eigenvectors: α = v / √λ

5. Project new point x:
   PC_k(x) = Σᵢ αₖᵢ k(x, xᵢ)
```

## 🧪 Python Implementation

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate non-linear data (two moons)
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standard PCA (fails to separate)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[0].set_title('Original Data')

# PCA projection
axes[1].scatter(X_pca, np.zeros_like(X_pca), c=y, cmap='viridis')
axes[1].set_title('PCA (Linear)')

# Kernel PCA projection
axes[2].scatter(X_kpca, np.zeros_like(X_kpca), c=y, cmap='viridis')
axes[2].set_title('Kernel PCA (RBF)')

plt.tight_layout()
plt.show()
```

## 🔄 PCA vs Kernel PCA

| Aspect | PCA | Kernel PCA |
|--------|-----|------------|
| **Linearity** | Linear | Non-linear |
| **Kernel** | None (linear) | RBF, Polynomial, etc. |
| **Complexity** | O(nd²) | O(n²d + n³) |
| **Interpretability** | High (components) | Low (implicit space) |
| **Inverse transform** | Exact | Approximate |
| **Use case** | Linear structure | Non-linear manifolds |

## ⚠️ Kernel PCA Challenges

1. **Hyperparameter tuning**: Kernel type and γ (RBF) must be tuned
2. **Computational cost**: O(n³) for eigendecomposition
3. **No exact inverse**: Reconstruction is approximate
4. **Overfitting risk**: Can overfit with wrong kernel parameters

---

# Random Projections

## 📘 Concept Overview

**Random Projections** project high-dimensional data to lower dimensions using **random matrix**.

Surprisingly, preserves distances well!

## 🧮 Mathematical Foundation

### Johnson-Lindenstrauss Lemma

For any ε > 0 and n points in ℝᵈ, there exists a mapping to ℝᵏ where:

```
k = O(log(n) / ε²)
```

Such that **all pairwise distances preserved** within (1±ε):

```
(1-ε) ‖xᵢ - xⱼ‖² ≤ ‖f(xᵢ) - f(xⱼ)‖² ≤ (1+ε) ‖xᵢ - xⱼ‖²
```

**Implication**: Can project from d=10,000 to k=100 while preserving distances!

### Random Projection Matrix

Generate random matrix R ∈ ℝᵈˣᵏ:

```
X_reduced = (1/√k) X R
```

**Entries of R**:
- Gaussian: R_{ij} ~ N(0, 1)
- Sparse: R_{ij} ~ {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}

## ⚙️ Algorithm

```
1. Choose target dimension k = O(log(n) / ε²)

2. Generate random matrix R ∈ ℝᵈˣᵏ
   (Gaussian or sparse)

3. Project: X_reduced = X R / √k

4. Done! (No training needed)
```

## 🧪 Python Implementation

```python
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# Gaussian random projection
grp = GaussianRandomProjection(n_components=50, random_state=42)
X_reduced_gauss = grp.fit_transform(X)

# Sparse random projection (faster for large d)
srp = SparseRandomProjection(n_components=50, density='auto', random_state=42)
X_reduced_sparse = srp.fit_transform(X)

# Compare distances
from sklearn.metrics import pairwise_distances

dist_original = pairwise_distances(X[:100])
dist_projected = pairwise_distances(X_reduced_gauss[:100])

# Check preservation
correlation = np.corrcoef(dist_original.flatten(), dist_projected.flatten())[0, 1]
print(f"Distance correlation: {correlation:.4f}")  # Should be close to 1
```

## 📊 Random Projections vs PCA

| Aspect | PCA | Random Projections |
|--------|-----|-------------------|
| **Training** | O(nd² + d³) | O(1) (no training!) |
| **Variance** | Maximized | Not optimized |
| **Distance preservation** | NOT guaranteed | Guaranteed (JL lemma) |
| **Deterministic** | Yes | No (random) |
| **Interpretability** | Components meaningful | No interpretation |
| **Data-dependent** | Yes (learns from X) | No (random) |
| **Large scale** | Slow | Fast ✓ |

## 🔄 When to Use Random Projections

✅ **Use when**:
- Very high dimensions (d > 10,000)
- Large datasets (n > 100,000)
- Need speed over optimality
- Downstream task is distance-based (k-NN, clustering)

❌ **Avoid when**:
- Need interpretable components
- Need to maximize variance
- Small datasets (PCA better)

---

# Other Techniques

## 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose**: Visualization (reduce to 2D/3D)

**Idea**: Preserve local neighborhood structure

**Pros**: Excellent for visualization, finds non-linear structure
**Cons**: Slow (O(n²)), non-deterministic, only for visualization (not preprocessing)

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Projection')
```

## 2. Linear Discriminant Analysis (LDA)

**Purpose**: Supervised dimensionality reduction (maximizes class separability)

**Idea**: Find projection that maximizes between-class variance / within-class variance

**Constraint**: At most (n_classes - 1) components

**Use**: Classification preprocessing

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)  # Requires labels!
```

## 3. Autoencoders (Neural Networks)

**Purpose**: Non-linear dimensionality reduction via deep learning

**Idea**: Train neural network to compress → reconstruct

```
Input → Encoder → Bottleneck (low-dim) → Decoder → Output
```

**Advantage**: Can learn complex non-linear manifolds

**Example** (PyTorch):
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Training
model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()

# Use encoder for dimensionality reduction
X_reduced = model.encode(torch.tensor(X, dtype=torch.float32)).detach().numpy()
```

## 4. UMAP (Uniform Manifold Approximation and Projection)

**Purpose**: Visualization + preprocessing (faster than t-SNE)

**Advantage**: Preserves global structure better than t-SNE

```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title('UMAP Projection')
```

---

# 🔥 MCQs

### Q1. PCA finds principal components by:
**Options:**
- A) Minimizing variance
- B) Maximizing variance ✓
- C) Minimizing correlation
- D) Maximizing entropy

**Explanation**: PCA finds orthogonal directions of maximum variance.

---

### Q2. Principal components are:
**Options:**
- A) Original features
- B) Eigenvectors of covariance matrix ✓
- C) Random projections
- D) Cluster centroids

**Explanation**: PCs are eigenvectors of Σ = X^T X / n.

---

### Q3. Before applying PCA, you should:
**Options:**
- A) Remove outliers
- B) Standardize features ✓
- C) Apply log transform
- D) Encode categorical features

**Explanation**: PCA is sensitive to scale; standardization ensures equal contribution.

---

### Q4. How many principal components can you have at most for d features?
**Options:**
- A) d² 
- B) d ✓
- C) d/2
- D) Infinite

**Explanation**: At most d components (one per original dimension).

---

### Q5. Kernel PCA is used for:
**Options:**
- A) Faster computation
- B) Non-linear dimensionality reduction ✓
- C) Supervised learning
- D) Feature selection

**Explanation**: Kernel PCA applies kernel trick to capture non-linear structure.

---

### Q6. Johnson-Lindenstrauss lemma guarantees:
**Options:**
- A) Variance preservation
- B) Distance preservation ✓
- C) Angle preservation
- D) Rank preservation

**Explanation**: Random projections preserve pairwise distances within (1±ε).

---

### Q7. Random projections require:
**Options:**
- A) Training on data
- B) Eigendecomposition
- C) No training ✓
- D) Labeled data

**Explanation**: Random projections use random matrix (no training needed).

---

### Q8. Which technique is supervised?
**Options:**
- A) PCA
- B) Kernel PCA
- C) LDA ✓
- D) Random Projections

**Explanation**: LDA uses class labels to maximize separability.

---

### Q9. t-SNE is best for:
**Options:**
- A) Preprocessing for classification
- B) Speeding up training
- C) Visualization ✓
- D) Feature engineering

**Explanation**: t-SNE preserves local structure (great for 2D viz), but not for preprocessing.

---

### Q10. PCA assumes:
**Options:**
- A) Non-linear relationships
- B) Linear relationships ✓
- C) Categorical features
- D) Sparse data

**Explanation**: PCA uses linear combinations (can't capture non-linear manifolds).

---

### Q11. Which has lowest computational complexity for large d?
**Options:**
- A) PCA
- B) Kernel PCA
- C) Random Projections ✓
- D) t-SNE

**Explanation**: Random projections are O(1) training (just matrix multiplication).

---

### Q12. Explained variance ratio tells you:
**Options:**
- A) Number of clusters
- B) Proportion of variance captured by component ✓
- C) Distance to centroid
- D) Classification accuracy

**Explanation**: λₖ / Σλᵢ = fraction of total variance explained by component k.

---

### Q13. Sparse PCA produces:
**Options:**
- A) Fewer components
- B) Components with many zero loadings ✓
- C) Faster computation
- D) Higher explained variance

**Explanation**: Sparse PCA enforces sparsity (easier interpretation).

---

### Q14. Inverse PCA transform:
**Options:**
- A) Is exact ✓
- B) Loses all information
- C) Is approximate
- D) Requires iteration

**Explanation**: PCA is linear, so inverse is exact: X_reconstructed = X_reduced @ components.T

---

### Q15. Kernel PCA with polynomial kernel of degree 2 captures:
**Options:**
- A) Linear relationships
- B) Quadratic relationships ✓
- C) Cubic relationships
- D) Exponential relationships

**Explanation**: Polynomial kernel of degree d captures up to d-order polynomial terms.

---

# ⚠️ Common Mistakes

1. **Not standardizing before PCA**: Features with large variance dominate

2. **Using PCA for non-linear data**: Use Kernel PCA or autoencoders instead

3. **Choosing k arbitrarily**: Use explained variance (80-95% threshold)

4. **Applying PCA to categorical data**: PCA assumes continuous features

5. **Interpreting PCA components as original features**: Components are linear combinations

6. **Using t-SNE for preprocessing**: t-SNE is for visualization only (non-deterministic)

7. **Ignoring explained variance**: May discard important information

8. **Not centering data**: PCA requires zero-mean data

9. **Using kernel PCA without tuning γ**: Performance heavily depends on kernel parameters

10. **Expecting random projections to maximize variance**: They preserve distances, not variance

---

# ⭐ One-Line Exam Facts

1. **PCA maximizes variance** along principal component directions

2. **Principal components = eigenvectors** of covariance matrix Σ = X^T X / n

3. **Explained variance ratio** = λₖ / Σλᵢ (proportion of variance by component k)

4. **Must standardize** before PCA (sensitive to feature scale)

5. **At most d principal components** for d features

6. **PCA assumes linearity**; use Kernel PCA for non-linear structure

7. **RBF kernel** most common for Kernel PCA (smooth non-linear manifolds)

8. **Random projections preserve distances** (Johnson-Lindenstrauss lemma)

9. **Random projections need k = O(log n / ε²) dimensions**

10. **t-SNE for visualization only** (2D/3D), not preprocessing

11. **LDA is supervised** (uses class labels), PCA is unsupervised

12. **Sparse PCA**: components have many zero loadings (interpretability)

13. **Incremental PCA** for datasets too large for memory

14. **Kernel PCA complexity O(n³)**; PCA complexity O(nd² + d³)

15. **Autoencoders** learn non-linear dimensionality reduction via neural networks

---

**End of Session 5**
