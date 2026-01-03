# Session 12 – Support Vector Machines (SVM)

## 📚 Table of Contents
1. [SVM Fundamentals](#svm-fundamentals)
2. [Maximum Margin Classifier](#maximum-margin-classifier)
3. [Soft Margin SVM](#soft-margin-svm)
4. [Kernel Trick](#kernel-trick)
5. [Common Kernels](#common-kernels)
6. [SVM for Regression](#svm-for-regression)
7. [MCQs](#mcqs)
8. [Common Mistakes](#common-mistakes)
9. [One-Line Exam Facts](#one-line-exam-facts)

---

# SVM Fundamentals

## 📘 Concept Overview

**Support Vector Machine (SVM)** finds the **optimal hyperplane** that maximizes the **margin** between classes.

**Key idea**: Maximum margin → better generalization

## 🧮 Linear Separable Case

### Hyperplane Definition

Decision boundary in d dimensions:
```
w^T x + b = 0
```

Where:
- w = normal vector (perpendicular to hyperplane)
- b = bias (offset from origin)
- x = feature vector

### Decision Function

```
f(x) = sign(w^T x + b)

Predict class +1 if w^T x + b > 0
Predict class -1 if w^T x + b < 0
```

### Margin Definition

**Geometric margin** of point (x, y):
```
Distance to hyperplane = |w^T x + b| / ‖w‖
```

**Functional margin**: y(w^T x + b)

For correct classification: y(w^T x + b) > 0

---

# Maximum Margin Classifier

## 🧮 Mathematical Formulation

### Objective

Find w, b that maximize margin γ:

```
maximize γ
subject to yᵢ(w^T xᵢ + b) ≥ γ, ∀i
```

### Canonical Hyperplane

Scale w, b such that:
```
min_i |w^T xᵢ + b| = 1
```

Then margin = 1/‖w‖

### Optimization Problem

```
maximize 1/‖w‖
subject to yᵢ(w^T xᵢ + b) ≥ 1, ∀i
```

Equivalent to:
```
minimize (1/2) ‖w‖²
subject to yᵢ(w^T xᵢ + b) ≥ 1, ∀i
```

This is a **convex quadratic programming** problem!

## 🧠 Support Vectors

**Support vectors**: Points on the margin (yᵢ(w^T xᵢ + b) = 1)

**Key property**: Solution depends only on support vectors!

```
     ○  (support vector)
    ╱|╲
   ╱ | ╲  margin
  ╱  |  ╲
 ╱   |   ╲
○────┼────○  decision boundary
 ╲   |   ╱
  ╲  |  ╱
   ╲ | ╱
    ╲|╱
     ●  (support vector)
```

## 🧮 Dual Formulation

Using **Lagrange multipliers** α:

**Dual problem**:
```
maximize Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼ yᵢyⱼ xᵢ^T xⱼ
subject to αᵢ ≥ 0, ∀i
         Σᵢ αᵢyᵢ = 0
```

**Solution**:
```
w = Σᵢ αᵢyᵢxᵢ  (linear combination of support vectors!)
```

**Prediction**:
```
f(x) = sign(Σᵢ αᵢyᵢ xᵢ^T x + b)
```

Only support vectors (αᵢ > 0) contribute!

## 🧪 Python Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_clusters_per_class=1,
                          class_sep=2.0, random_state=42)
y = 2*y - 1  # Convert to {-1, +1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
svm = SVC(kernel='linear', C=1e10)  # Large C ≈ hard margin
svm.fit(X_train, y_train)

# Accuracy
accuracy = svm.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# Support vectors
print(f"Number of support vectors: {len(svm.support_vectors_)}")
print(f"Support vector indices: {svm.support_}")

# Visualize
def plot_svm_decision_boundary(X, y, model):
    """Plot SVM decision boundary and margins."""
    plt.figure(figsize=(10, 6))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'white', 'blue'])
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidths=2,
                label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary')
    plt.show()

plot_svm_decision_boundary(X_train, y_train, svm)
```

---

# Soft Margin SVM

## 📘 Concept Overview

**Problem**: Real data is rarely perfectly separable.

**Solution**: Allow some misclassifications using **slack variables** ξᵢ.

## 🧮 Mathematical Formulation

### Soft Margin Constraints

```
yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ∀i
ξᵢ ≥ 0
```

Where ξᵢ = violation amount:
- ξᵢ = 0: Correctly classified, outside margin
- 0 < ξᵢ < 1: Correctly classified, inside margin
- ξᵢ ≥ 1: Misclassified

### Optimization Problem

```
minimize (1/2) ‖w‖² + C Σᵢ ξᵢ
subject to yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ∀i
           ξᵢ ≥ 0, ∀i
```

**C parameter**: Regularization strength
- **Large C**: Hard margin (few violations, risk overfitting)
- **Small C**: Soft margin (many violations, smoother boundary)

### Interpretation

```
Minimize: Margin penalty + Classification error

C → ∞: Hard margin (no violations)
C → 0: Very soft margin (all weights → 0)
```

## 🔄 Effect of C

```python
# Compare different C values
C_values = [0.01, 1, 100]

plt.figure(figsize=(15, 4))

for i, C in enumerate(C_values):
    svm_c = SVC(kernel='linear', C=C)
    svm_c.fit(X_train, y_train)
    
    plt.subplot(1, 3, i+1)
    # Plot decision boundary (simplified)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
    plt.scatter(svm_c.support_vectors_[:, 0], svm_c.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidths=2)
    plt.title(f'C = {C}, SVs = {len(svm_c.support_vectors_)}')

plt.tight_layout()
plt.show()

# Observation: Smaller C → more support vectors (softer margin)
```

## 📊 Hinge Loss

Soft margin SVM equivalent to minimizing **hinge loss**:

```
L(y, f(x)) = max(0, 1 - y·f(x))

where f(x) = w^T x + b
```

**SVM objective**:
```
minimize (1/2)‖w‖² + C Σᵢ max(0, 1 - yᵢ(w^T xᵢ + b))
```

**Plot of hinge loss**:
```
Loss
  │╲
1 │ ╲
  │  ╲___________
  │             
0 ├──────┬───────> y·f(x)
  0      1
  
Hinge: max(0, 1 - y·f(x))
```

---

# Kernel Trick

## 📘 Concept Overview

**Problem**: Data not linearly separable in original space.

**Solution**: Map to **higher-dimensional space** where it becomes linearly separable.

## 🧮 Feature Mapping

Map x → φ(x) to higher dimension:

```
Original: x ∈ ℝ²
Mapped: φ(x) ∈ ℝ³ (or higher)

Example: φ([x₁, x₂]) = [x₁, x₂, x₁²+ x₂²]
```

**Problem**: Computing φ(x) explicitly can be expensive (or infinite-dimensional)!

## 🧮 Kernel Trick

**Key insight**: SVM dual only needs **inner products** xᵢ^T xⱼ

Define **kernel function**:
```
K(x, x') = φ(x)^T φ(x')
```

**Never compute φ(x) explicitly!** Just compute K(x, x').

### Dual with Kernel

```
maximize Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
subject to αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0
```

**Prediction**:
```
f(x) = sign(Σᵢ αᵢyᵢ K(xᵢ, x) + b)
```

## 🧠 Valid Kernels (Mercer's Theorem)

A function K is a valid kernel if:
1. **Symmetric**: K(x, x') = K(x', x)
2. **Positive semi-definite**: Kernel matrix K is PSD

---

# Common Kernels

## 1. Linear Kernel

```
K(x, x') = x^T x'
```

**Use**: Linearly separable data

```python
svm_linear = SVC(kernel='linear')
```

## 2. Polynomial Kernel

```
K(x, x') = (γ x^T x' + r)^d
```

Where:
- d = degree
- γ = gamma (coefficient)
- r = coef0 (constant term)

**Use**: When polynomial decision boundary needed

```python
svm_poly = SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
```

**Example**: Degree 2 in 2D
```
φ([x₁, x₂]) = [1, √2x₁, √2x₂, x₁², √2x₁x₂, x₂²]

K(x, x') = (x^T x' + 1)²  (no need to compute φ!)
```

## 3. RBF (Gaussian) Kernel

```
K(x, x') = exp(-γ ‖x - x'‖²)
```

Where γ > 0 controls width:
- **Small γ**: Wide kernel (smooth, may underfit)
- **Large γ**: Narrow kernel (wiggly, may overfit)

**Most popular kernel!** Works well for many problems.

```python
svm_rbf = SVC(kernel='rbf', gamma='scale')  # or gamma='auto' or float
```

**Intuition**: Similarity measure (close points → high similarity)

### RBF Kernel Interpretation

```
K(x, x') = exp(-γ ‖x - x'‖²)

‖x - x'‖ = 0 → K = 1 (maximum similarity)
‖x - x'‖ → ∞ → K → 0 (no similarity)
```

**Infinite-dimensional feature space!**

## 4. Sigmoid Kernel

```
K(x, x') = tanh(γ x^T x' + r)
```

**Use**: Similar to neural network with one hidden layer

```python
svm_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0)
```

**Note**: Not always positive semi-definite (may not be valid kernel)

## 🧪 Kernel Comparison

```python
from sklearn.datasets import make_moons

# Generate non-linearly separable data
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
y = 2*y - 1  # {-1, +1}

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(16, 4))

for i, kernel in enumerate(kernels):
    svm_kernel = SVC(kernel=kernel, gamma='scale')
    svm_kernel.fit(X, y)
    
    plt.subplot(1, 4, i+1)
    
    # Decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = svm_kernel.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(f'{kernel.capitalize()} Kernel')

plt.tight_layout()
plt.show()
```

## 📊 Gamma in RBF Kernel

```python
gammas = [0.1, 1, 10]

plt.figure(figsize=(15, 4))

for i, gamma in enumerate(gammas):
    svm_gamma = SVC(kernel='rbf', gamma=gamma)
    svm_gamma.fit(X, y)
    
    # Plot (similar to above)
    plt.subplot(1, 3, i+1)
    plt.title(f'γ = {gamma}')
    # ... plotting code

plt.tight_layout()
plt.show()

# Observation:
# Small γ: Smooth (may underfit)
# Large γ: Complex (may overfit)
```

---

# SVM for Regression

## 📘 Support Vector Regression (SVR)

**Goal**: Find function that has at most ε deviation from targets.

## 🧮 ε-insensitive Loss

```
L_ε(y, f(x)) = max(0, |y - f(x)| - ε)

No penalty if |y - f(x)| ≤ ε (inside ε-tube)
```

```
y
│     ╱ upper margin (f(x) + ε)
│    ╱
│   ╱──────── f(x)
│  ╱
│ ╱ lower margin (f(x) - ε)
└──────────────────> x
```

## ⚙️ SVR Objective

```
minimize (1/2)‖w‖² + C Σᵢ (ξᵢ + ξᵢ*)
subject to yᵢ - w^T xᵢ - b ≤ ε + ξᵢ
           w^T xᵢ + b - yᵢ ≤ ε + ξᵢ*
           ξᵢ, ξᵢ* ≥ 0
```

## 🧪 Python Implementation

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR with RBF kernel
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr.fit(X_train, y_train)

# Predictions
y_pred = svr.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Visualize
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = svr.predict(X_plot)

plt.scatter(X_train, y_train, label='Train', alpha=0.6)
plt.plot(X_plot, y_plot, 'r-', label='SVR prediction', linewidth=2)
plt.fill_between(X_plot.ravel(), y_plot - svr.epsilon, y_plot + svr.epsilon,
                alpha=0.2, label='ε-tube')
plt.legend()
plt.title('Support Vector Regression')
plt.show()
```

---

# 🔥 MCQs

### Q1. SVM maximizes:
**Options:**
- A) Accuracy
- B) Margin between classes ✓
- C) Number of support vectors
- D) Feature importance

**Explanation**: SVM finds hyperplane that maximizes margin for better generalization.

---

### Q2. Support vectors are:
**Options:**
- A) All training points
- B) Misclassified points
- C) Points on the margin ✓
- D) Outliers

**Explanation**: Support vectors lie on margin boundaries (yᵢ(w^T xᵢ + b) = 1).

---

### Q3. In soft margin SVM, parameter C controls:
**Options:**
- A) Number of features
- B) Regularization strength ✓
- C) Kernel width
- D) Learning rate

**Explanation**: Large C = hard margin (few violations), small C = soft margin.

---

### Q4. Kernel trick allows:
**Options:**
- A) Faster training
- B) Non-linear decision boundaries ✓
- C) More features
- D) Better interpretability

**Explanation**: Maps to high-dimensional space where data becomes linearly separable.

---

### Q5. RBF kernel is defined as:
**Options:**
- A) x^T x'
- B) (x^T x' + 1)^d
- C) exp(-γ‖x - x'‖²) ✓
- D) tanh(x^T x')

**Explanation**: Gaussian kernel = exp(-γ‖x - x'‖²).

---

### Q6. In RBF kernel, large γ causes:
**Options:**
- A) Smooth boundary (underfitting)
- B) Complex boundary (overfitting) ✓
- C) Linear boundary
- D) No effect

**Explanation**: Large γ = narrow kernel = wiggly boundary = potential overfitting.

---

### Q7. SVM dual formulation uses:
**Options:**
- A) Gradient descent
- B) Lagrange multipliers ✓
- C) Newton's method
- D) Coordinate descent

**Explanation**: Dual problem solved using Lagrangian and KKT conditions.

---

### Q8. Hinge loss is:
**Options:**
- A) (y - ŷ)²
- B) max(0, 1 - y·f(x)) ✓
- C) -y log(ŷ)
- D) |y - ŷ|

**Explanation**: SVM uses hinge loss = max(0, 1 - y·f(x)).

---

### Q9. Which kernel has infinite-dimensional feature space?
**Options:**
- A) Linear
- B) Polynomial
- C) RBF ✓
- D) Sigmoid

**Explanation**: RBF kernel maps to infinite-dimensional Hilbert space.

---

### Q10. SVR ε parameter defines:
**Options:**
- A) Regularization strength
- B) Width of insensitive tube ✓
- C) Kernel width
- D) Learning rate

**Explanation**: No loss penalty for errors within ε of true value.

---

### Q11. SVM assumes:
**Options:**
- A) Gaussian features
- B) Linear boundaries in some space ✓
- C) Independent features
- D) Balanced classes

**Explanation**: SVM finds linear boundary in (possibly infinite-dimensional) feature space.

---

### Q12. Polynomial kernel degree 2 in 2D creates:
**Options:**
- A) 2 features
- B) 4 features
- C) 6 features ✓
- D) 8 features

**Explanation**: [1, x₁, x₂, x₁², x₁x₂, x₂²] = 6 features.

---

### Q13. C → ∞ in soft margin SVM:
**Options:**
- A) All violations allowed
- B) Hard margin (no violations) ✓
- C) Soft margin
- D) Linear kernel only

**Explanation**: Large C heavily penalizes violations (approaches hard margin).

---

### Q14. For linearly separable data, best kernel is:
**Options:**
- A) RBF
- B) Polynomial
- C) Linear ✓
- D) Sigmoid

**Explanation**: Linear kernel sufficient and faster for linearly separable data.

---

### Q15. SVM is sensitive to:
**Options:**
- A) Feature scaling ✓
- B) Class labels
- C) Number of classes
- D) Sample order

**Explanation**: SVM uses distances, so features must be scaled.

---

# ⚠️ Common Mistakes

1. **Not scaling features**: SVM very sensitive to feature scale (use StandardScaler)

2. **Using RBF for linearly separable data**: Overkill; linear kernel faster and simpler

3. **Default C and γ values**: Always tune via cross-validation

4. **Forgetting class imbalance**: Use class_weight='balanced' or adjust C

5. **Too large γ**: Causes overfitting (very complex boundary)

6. **Too small C**: Under-penalizes violations (underfitting)

7. **Not normalizing for polynomial kernel**: Can cause numerical instability

8. **Assuming SVM always best**: Decision trees/ensembles often better for large datasets

9. **Ignoring computational cost**: SVM slow for large n (O(n²) to O(n³))

10. **Using linear SVM for non-linear data**: Need kernel (RBF, polynomial)

---

# ⭐ One-Line Exam Facts

1. **SVM maximizes margin** = distance between support vectors

2. **Support vectors**: Points on margin (αᵢ > 0)

3. **Hard margin**: No violations (linearly separable data only)

4. **Soft margin**: Allows violations via slack variables ξᵢ

5. **C parameter**: Large C = hard margin, small C = soft margin

6. **Kernel trick**: K(x, x') = φ(x)^T φ(x') (no explicit φ!)

7. **Linear kernel**: K(x, x') = x^T x'

8. **Polynomial kernel**: K(x, x') = (γx^T x' + r)^d

9. **RBF kernel**: K(x, x') = exp(-γ‖x - x'‖²)

10. **RBF γ**: Small = smooth, large = complex (overfitting)

11. **Hinge loss**: max(0, 1 - y·f(x))

12. **SVM dual**: Uses Lagrange multipliers, depends only on inner products

13. **SVR**: ε-insensitive loss (ε-tube around predictions)

14. **Mercer's theorem**: Valid kernel = symmetric + PSD

15. **SVM requires feature scaling** (distance-based)

---

**End of Session 12**

**Progress: 12/30 sessions completed!** You now have comprehensive notes covering all classical ML algorithms. Ready to continue with remaining sessions (Time Series, Recommendation Systems, Deep Learning, CNNs, RNNs, Transformers) whenever you'd like!
