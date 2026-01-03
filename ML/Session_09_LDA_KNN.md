# Session 9 – LDA & K-Nearest Neighbors

## 📚 Table of Contents
1. [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
2. [LDA vs PCA](#lda-vs-pca)
3. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
4. [Distance Metrics](#distance-metrics)
5. [Curse of Dimensionality](#curse-of-dimensionality)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Linear Discriminant Analysis (LDA)

## 📘 Concept Overview

**LDA** is a **supervised** dimensionality reduction technique that finds projection maximizing **class separability**.

**Key difference from PCA**:
- **PCA**: Maximizes variance (unsupervised)
- **LDA**: Maximizes class separation (supervised)

## 🧮 Mathematical Foundation

### Objective

Find projection direction **w** that maximizes:

```
J(w) = (Between-class scatter) / (Within-class scatter)
```

**Intuition**: Want classes **far apart** (large between-class scatter) and **compact** (small within-class scatter).

### Fisher's Linear Discriminant (Binary Classification)

For two classes (C₁ and C₂):

**Project data onto line**: z = w^T x

**Means after projection**:
```
μ₁ = w^T m₁  (mean of class 1 projected)
μ₂ = w^T m₂  (mean of class 2 projected)
```

**Between-class scatter**:
```
S_B = (μ₁ - μ₂)²
```

**Within-class scatter**:
```
S_W = Σ_{x∈C₁} (z - μ₁)² + Σ_{x∈C₂} (z - μ₂)²
```

**Fisher criterion**:
```
J(w) = S_B / S_W = (μ₁ - μ₂)² / (σ₁² + σ₂²)
```

In matrix form:
```
J(w) = (w^T S_B w) / (w^T S_W w)
```

Where:
- **S_B = (m₁ - m₂)(m₁ - m₂)^T** (between-class scatter matrix)
- **S_W = Σ₁ + Σ₂** (within-class scatter matrix)

### Solution

**Optimal w** maximizes J(w):

```
S_W w = λ S_B w  (generalized eigenvalue problem)
```

**Closed-form solution**:
```
w ∝ S_W^(-1) (m₁ - m₂)
```

### Multi-Class LDA

For K classes:

**Between-class scatter matrix**:
```
S_B = Σᵢ nᵢ (mᵢ - m)(mᵢ - m)^T
```

Where:
- nᵢ = number of samples in class i
- mᵢ = mean of class i
- m = overall mean

**Within-class scatter matrix**:
```
S_W = Σᵢ Σ_{x∈Cᵢ} (x - mᵢ)(x - mᵢ)^T
```

**Objective**:
```
W = argmax_W |W^T S_B W| / |W^T S_W W|
```

**Solution**: Eigenvectors of S_W^(-1) S_B

**Maximum components**: min(K-1, d)
- K-1: At most K-1 discriminant directions
- d: Original dimensions

## ⚙️ LDA Algorithm

```
1. Compute class means: mᵢ = (1/nᵢ) Σ_{x∈Cᵢ} x

2. Compute overall mean: m = (1/n) Σ x

3. Compute within-class scatter: S_W = Σᵢ Σ_{x∈Cᵢ} (x-mᵢ)(x-mᵢ)^T

4. Compute between-class scatter: S_B = Σᵢ nᵢ(mᵢ-m)(mᵢ-m)^T

5. Solve eigenvalue problem: S_W^(-1) S_B v = λ v

6. Select top K-1 eigenvectors (largest eigenvalues)

7. Project data: X_reduced = X @ W
```

## 🧪 Python Implementation

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)  # Max 2 for 3 classes
X_lda = lda.fit_transform(X_train, y_train)

print(f"Original shape: {X_train.shape}")
print(f"LDA shape: {X_lda.shape}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

# Visualize
plt.figure(figsize=(10, 6))
for label in np.unique(y_train):
    mask = y_train == label
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {label}', alpha=0.6)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.title('LDA Projection')
plt.show()

# LDA as classifier (also does classification!)
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
accuracy = lda_clf.score(X_test, y_test)
print(f"LDA Classifier Accuracy: {accuracy:.3f}")
```

### From Scratch

```python
import numpy as np

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        
        # Compute class means
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            # Within-class scatter
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            
            # Between-class scatter
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if self.n_components is None:
            self.n_components = n_classes - 1
        
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        
        return self
    
    def transform(self, X):
        return X @ self.components
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

# Test
lda_custom = LDA(n_components=2)
X_lda_custom = lda_custom.fit_transform(X_train, y_train)
print(f"Custom LDA shape: {X_lda_custom.shape}")
print(f"Explained variance: {lda_custom.explained_variance_ratio_}")
```

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Supervised**: Uses class labels (better for classification)
2. **Class separation**: Maximizes discriminability
3. **Dimensionality reduction**: Reduces features while preserving class info
4. **Classification**: Can also be used as classifier
5. **Interpretability**: Linear projections easy to understand

### Weaknesses ✗
1. **Linearity**: Assumes linear class boundaries
2. **Gaussian assumption**: Assumes features Gaussian within classes
3. **Limited components**: At most K-1 components
4. **Small sample size**: Can fail if S_W not invertible (d > n)
5. **Outliers**: Sensitive to outliers (affects mean)
6. **Balanced classes**: Assumes similar covariances

## ⚠️ Assumptions

1. **Gaussian distribution**: Features normally distributed within each class
2. **Homoscedasticity**: All classes have same covariance matrix
3. **Linear boundaries**: Classes separable by linear decision boundary

---

# LDA vs PCA

## 📊 Comparison

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Supervision** | Unsupervised | Supervised ✓ |
| **Objective** | Maximize variance | Maximize class separation ✓ |
| **Uses labels** | No | Yes ✓ |
| **Max components** | d (all features) | K-1 (classes - 1) |
| **Use case** | General dimensionality reduction | Classification preprocessing ✓ |
| **Assumptions** | None (variance-based) | Gaussian, equal covariance |
| **Class info** | Ignores | Preserves ✓ |

## 🧪 Visual Comparison

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
for label in np.unique(y):
    mask = y == label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {label}', alpha=0.6)
axes[0].set_title('PCA (Unsupervised)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()

# LDA
for label in np.unique(y):
    mask = y == label
    axes[1].scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {label}', alpha=0.6)
axes[1].set_title('LDA (Supervised)')
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Observation**: LDA often gives **better class separation** than PCA.

## 🔄 When to Use Each

**Use PCA**:
- Unsupervised task (no labels)
- Visualization of general structure
- Preprocessing for any ML task
- Many dimensions to reduce

**Use LDA**:
- Classification task (labels available)
- Want to maximize class separability
- Number of classes << number of features
- Features approximately Gaussian

---

# K-Nearest Neighbors (KNN)

## 📘 Concept Overview

**KNN** is a **non-parametric**, **instance-based** (lazy learning) algorithm.

**Key idea**: Classify based on **majority vote** of k nearest neighbors.

## 🧮 Mathematical Foundation

### Classification

For new point x:

```
1. Find k nearest neighbors in training set (by distance)

2. Predict: ŷ = argmax_c (count of class c among k neighbors)
```

**Ties**: If multiple classes have same count, use smallest k' < k to break tie.

### Regression

```
ŷ = (1/k) Σ_{i∈neighbors} yᵢ
```

Average of k nearest neighbors' values.

### Distance-Weighted KNN

Give closer neighbors more weight:

```
ŷ = Σᵢ wᵢ yᵢ / Σᵢ wᵢ
```

Where:
```
wᵢ = 1 / d(x, xᵢ)²
```

## ⚙️ KNN Algorithm

```
Training:
  Store all training data (no actual "training"!)

Prediction for new point x:
  1. Compute distance to all training points
  2. Sort by distance
  3. Select k nearest neighbors
  4. Classification: Majority vote
     Regression: Average value
```

## 🧪 Python Implementation

### Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Scale features for KNN!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',    # or 'distance' for distance-weighted
    metric='euclidean',   # or 'manhattan', 'minkowski'
    algorithm='auto'      # 'ball_tree', 'kd_tree', 'brute'
)

knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Get neighbors for a point
distances, indices = knn.kneighbors(X_test_scaled[:1], n_neighbors=5)
print(f"\nNearest neighbors indices: {indices}")
print(f"Distances: {distances}")
```

### From Scratch

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        """Store training data."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """Predict labels for X."""
        predictions = []
        
        for x in X:
            # Compute distances to all training points
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)

# Test
knn_custom = KNN(k=5)
knn_custom.fit(X_train_scaled, y_train)
y_pred_custom = knn_custom.predict(X_test_scaled)
print(f"Custom KNN Accuracy: {accuracy_score(y_test, y_pred_custom):.3f}")
```

### Regression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")
```

## 📊 Choosing k

### Effect of k

```
Small k (k=1):
  - Low bias, high variance
  - Sensitive to noise
  - Complex decision boundary
  - Overfitting risk

Large k (k=n):
  - High bias, low variance
  - Smooth decision boundary
  - Underfitting risk
  - Approaches majority class prediction
```

### Elbow Method

```python
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Choosing k for KNN')
plt.grid(alpha=0.3)
plt.show()

optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
```

**Rule of thumb**: k = √n (n = training samples)

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Simple**: Easy to understand and implement
2. **No training**: No explicit training phase (lazy learning)
3. **Non-parametric**: No assumptions about data distribution
4. **Multi-class**: Naturally handles multiple classes
5. **Incremental**: Easy to add new training data
6. **Non-linear**: Can capture complex decision boundaries

### Weaknesses ✗
1. **Slow prediction**: O(nd) for each prediction
2. **Memory intensive**: Stores entire training set
3. **Curse of dimensionality**: Fails in high dimensions
4. **Sensitive to scale**: **Must standardize** features
5. **Imbalanced data**: Majority class dominates
6. **Irrelevant features**: Distance polluted by noise features
7. **No model interpretability**: Black box (no learned parameters)

## ⚙️ Hyperparameters

### 1. n_neighbors (k)

**Effect**: Bias-variance tradeoff
**Tuning**: Cross-validation

### 2. weights

- `'uniform'`: All neighbors equal weight
- `'distance'`: Weight inversely proportional to distance (closer = more influence)

**Use distance** when neighbors at varying distances.

### 3. metric

- `'euclidean'`: L2 distance (default)
- `'manhattan'`: L1 distance
- `'minkowski'`: L_p distance (p=2 is Euclidean)
- `'cosine'`: Cosine similarity

**Choose based on feature type** (continuous vs binary vs counts).

### 4. algorithm

- `'brute'`: Compute all distances (O(n²))
- `'kd_tree'`: KD-tree for fast lookup (O(log n))
- `'ball_tree'`: Ball tree for high dimensions
- `'auto'`: Automatically choose best

**High dimensions**: Use ball_tree or brute_force (kd_tree degrades).

## 🔄 Optimizing KNN

### 1. Feature Scaling (Critical!)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Dimensionality Reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn.fit(X_train_pca, y_train)
```

### 3. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### 4. Approximate Nearest Neighbors

For very large datasets, use approximate methods (faster but less accurate):

```python
# Example: Annoy library (not in sklearn)
# from annoy import AnnoyIndex
```

---

# Distance Metrics

## 📊 Common Metrics

### 1. Euclidean Distance (L2)

```
d(x, y) = √[Σᵢ (xᵢ - yᵢ)²]
```

**Properties**:
- Symmetric: d(x,y) = d(y,x)
- Non-negative: d(x,y) ≥ 0
- Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)

**Use**: Continuous features, magnitude matters

### 2. Manhattan Distance (L1, City Block)

```
d(x, y) = Σᵢ |xᵢ - yᵢ|
```

**Use**: Grid-like data, more robust to outliers

### 3. Minkowski Distance (L_p)

```
d(x, y) = (Σᵢ |xᵢ - yᵢ|ᵖ)^(1/p)
```

- p=1: Manhattan
- p=2: Euclidean
- p→∞: Chebyshev (max difference)

### 4. Cosine Distance

```
cosine_similarity(x, y) = (x · y) / (‖x‖ ‖y‖)
cosine_distance(x, y) = 1 - cosine_similarity(x, y)
```

**Use**: Text data (TF-IDF vectors), high-dimensional sparse data

### 5. Hamming Distance

```
d(x, y) = Σᵢ I(xᵢ ≠ yᵢ)
```

**Use**: Categorical or binary features

## 🧪 Distance Metric Impact

```python
from sklearn.neighbors import KNeighborsClassifier

metrics = ['euclidean', 'manhattan', 'cosine']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    accuracy = knn.score(X_test_scaled, y_test)
    print(f"{metric.capitalize()}: {accuracy:.3f}")
```

---

# Curse of Dimensionality

## 📘 Concept

In high dimensions, **distance becomes meaningless** — all points become equidistant.

## 🧮 Mathematical Insight

In d dimensions, distance ratio:

```
max_distance / min_distance → 1 as d → ∞
```

**Consequence**: Nearest and farthest neighbors have similar distances!

## 📊 Empirical Demonstration

```python
import numpy as np

def distance_concentration(n_samples=1000, dims=range(1, 101)):
    """Show distance concentration effect."""
    results = []
    
    for d in dims:
        # Random points in d dimensions
        X = np.random.randn(n_samples, d)
        
        # Distance from origin
        distances = np.linalg.norm(X, axis=1)
        
        # Ratio of max/min distance
        ratio = distances.max() / distances.min()
        results.append(ratio)
    
    return results

ratios = distance_concentration()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), ratios)
plt.xlabel('Dimensions')
plt.ylabel('Max Distance / Min Distance')
plt.title('Curse of Dimensionality: Distance Concentration')
plt.grid(alpha=0.3)
plt.show()
```

## ⚠️ Implications for KNN

1. **All neighbors equally far**: k-nearest is arbitrary
2. **Volume explosion**: Need exponentially more data
3. **Sparse data**: Most volume in corners (empty space)
4. **Solution**: 
   - Dimensionality reduction (PCA, LDA)
   - Feature selection
   - Use fewer features

## 📊 Sample Size Requirements

To maintain same density in d dimensions:

```
n_samples ∝ ρ^d
```

Where ρ = desired density per dimension.

**Example**: 10 samples per dimension
- 1D: 10 samples
- 2D: 100 samples
- 10D: 10^10 samples (impossible!)

---

# 🔥 MCQs

### Q1. LDA is:
**Options:**
- A) Unsupervised
- B) Supervised ✓
- C) Semi-supervised
- D) Reinforcement learning

**Explanation**: LDA uses class labels to maximize class separability.

---

### Q2. Maximum number of LDA components for K classes is:
**Options:**
- A) K
- B) K-1 ✓
- C) K+1
- D) 2K

**Explanation**: LDA finds at most K-1 discriminant directions.

---

### Q3. LDA assumes:
**Options:**
- A) Features are independent
- B) Classes have equal covariance ✓
- C) Non-linear boundaries
- D) No assumptions

**Explanation**: LDA assumes Gaussian features with same covariance matrix across classes.

---

### Q4. KNN is called "lazy learning" because:
**Options:**
- A) Slow training
- B) No explicit training phase ✓
- C) Requires less data
- D) Simple algorithm

**Explanation**: KNN stores data without learning; computation happens at prediction time.

---

### Q5. For KNN with k=1:
**Options:**
- A) High bias, low variance
- B) Low bias, high variance ✓
- C) Low bias, low variance
- D) High bias, high variance

**Explanation**: k=1 fits training data perfectly (low bias) but very sensitive to noise (high variance).

---

### Q6. KNN prediction complexity is:
**Options:**
- A) O(1)
- B) O(log n)
- C) O(n) ✓
- D) O(n²)

**Explanation**: Must compute distance to all n training points for each prediction.

---

### Q7. Before using KNN, you should:
**Options:**
- A) Remove outliers
- B) Standardize features ✓
- C) Apply PCA
- D) Encode labels

**Explanation**: KNN uses distance, so features must be on same scale.

---

### Q8. Cosine distance is best for:
**Options:**
- A) Categorical features
- B) Text/high-dimensional sparse features ✓
- C) Ordinal features
- D) Time series

**Explanation**: Cosine measures angle, not magnitude (good for TF-IDF vectors).

---

### Q9. Curse of dimensionality means:
**Options:**
- A) Too many features slow training
- B) Distances become meaningless in high dimensions ✓
- C) Need more memory
- D) Overfitting increases

**Explanation**: In high-d, all points become roughly equidistant (distance concentration).

---

### Q10. LDA maximizes:
**Options:**
- A) Variance
- B) Between-class scatter / Within-class scatter ✓
- C) Likelihood
- D) Information gain

**Explanation**: LDA maximizes Fisher criterion J(w) = S_B / S_W.

---

### Q11. Which is NOT an assumption of LDA?
**Options:**
- A) Gaussian features
- B) Equal covariances
- C) Linear boundaries
- D) Independent features ✓

**Explanation**: LDA doesn't assume feature independence (unlike Naïve Bayes).

---

### Q12. KNN with weights='distance':
**Options:**
- A) All neighbors equal weight
- B) Closer neighbors have more influence ✓
- C) Farther neighbors have more influence
- D) Random weights

**Explanation**: Distance weighting: wᵢ ∝ 1/d(x, xᵢ).

---

### Q13. For imbalanced data, KNN:
**Options:**
- A) Works perfectly
- B) May favor majority class ✓
- C) Requires resampling
- D) Needs different distance metric

**Explanation**: Majority class dominates among k neighbors.

---

### Q14. PCA vs LDA for classification preprocessing:
**Options:**
- A) Always use PCA
- B) LDA better (uses labels) ✓
- C) Same performance
- D) PCA faster

**Explanation**: LDA maximizes class separation (supervised), PCA ignores labels.

---

### Q15. Ball tree vs KD tree:
**Options:**
- A) Ball tree better for low dimensions
- B) KD tree better for high dimensions
- C) Ball tree better for high dimensions ✓
- D) No difference

**Explanation**: KD-tree degrades to O(n) in high-d; ball tree more robust.

---

# ⚠️ Common Mistakes

1. **Not scaling features for KNN**: Distance dominated by large-range features

2. **Using too many features for KNN**: Curse of dimensionality degrades performance

3. **Confusing LDA dimensionality reduction with LDA classifier**: Same algorithm, different use

4. **Using LDA for regression**: LDA is for classification only

5. **Expecting K-1 components when K > d**: Max components = min(K-1, d)

6. **Using Euclidean distance for categorical features**: Use Hamming instead

7. **Not handling class imbalance in KNN**: Can bias toward majority class

8. **Choosing k based on training accuracy**: Use cross-validation

9. **Assuming LDA always outperforms PCA**: Depends on data and task

10. **Using brute force for large datasets**: Use KD-tree or ball tree for speedup

---

# ⭐ One-Line Exam Facts

1. **LDA is supervised**, PCA is unsupervised

2. **LDA maximizes** between-class scatter / within-class scatter (Fisher criterion)

3. **Maximum LDA components** = min(K-1, d) where K = classes, d = features

4. **LDA assumes** Gaussian features with equal covariance across classes

5. **KNN is lazy learning** (no training, stores all data)

6. **KNN prediction** complexity = O(nd) for each point

7. **KNN requires feature scaling** (distance-based algorithm)

8. **Small k** → low bias, high variance (overfitting)

9. **Large k** → high bias, low variance (underfitting)

10. **Optimal k** typically √n via cross-validation

11. **Distance-weighted KNN**: wᵢ = 1/d(x, xᵢ) (closer neighbors more influence)

12. **Curse of dimensionality**: distances become meaningless in high-d

13. **Cosine distance** for text/sparse high-dimensional data

14. **Manhattan distance** more robust to outliers than Euclidean

15. **KNN with k=1** memorizes training data (zero training error)

---

**End of Session 9**
