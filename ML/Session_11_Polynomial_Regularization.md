# Session 11 – Polynomial Regression & Regularization

## 📚 Table of Contents
1. [Polynomial Regression](#polynomial-regression)
2. [Ridge Regression (L2)](#ridge-regression-l2)
3. [Lasso Regression (L1)](#lasso-regression-l1)
4. [Elastic Net](#elastic-net)
5. [Regularization Comparison](#regularization-comparison)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Polynomial Regression

## 📘 Concept Overview

**Polynomial Regression** captures non-linear relationships by creating polynomial features.

**Key insight**: Still **linear in parameters** (can use linear regression)!

## 🧮 Mathematical Foundation

### Model

For single feature x, degree d polynomial:

```
ŷ = w₀ + w₁x + w₂x² + w₃x³ + ... + wₐx^d
```

**Vector form**:
```
ŷ = w^T φ(x)
```

Where φ(x) = [1, x, x², ..., x^d]^T is **feature transformation**

### Multiple Features

For n features, polynomial of degree 2:

```
Original: [x₁, x₂]
Polynomial: [1, x₁, x₂, x₁², x₁x₂, x₂²]
```

**Number of features**: O(d^n) grows rapidly!

### Training

Same as linear regression after feature transformation:

```
1. Transform: X → φ(X)
2. Fit: w = (φ(X)^T φ(X))^(-1) φ(X)^T y
3. Predict: ŷ = w^T φ(x_new)
```

## 🧪 Python Implementation

### Using Sklearn

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

# Generate data with non-linear relationship
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 100)).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 100)

# Polynomial regression
degrees = [1, 3, 9, 15]

plt.figure(figsize=(16, 4))

for i, degree in enumerate(degrees):
    # Create pipeline
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_reg.fit(X, y)
    
    # Predictions
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred = poly_reg.predict(X_test)
    
    # Plot
    plt.subplot(1, 4, i+1)
    plt.scatter(X, y, s=10, alpha=0.5)
    plt.plot(X_test, y_pred, 'r-', linewidth=2)
    plt.title(f'Degree {degree}')
    plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
```

### From Scratch

```python
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
    
    def _create_polynomial_features(self, X):
        """Create polynomial features."""
        n_samples = X.shape[0]
        
        # Start with intercept
        features = [np.ones(n_samples)]
        
        # Add polynomial terms
        for d in range(1, self.degree + 1):
            features.append(X.ravel() ** d)
        
        return np.column_stack(features)
    
    def fit(self, X, y):
        """Fit polynomial regression."""
        # Transform features
        X_poly = self._create_polynomial_features(X)
        
        # Normal equation
        self.weights = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        
        return self
    
    def predict(self, X):
        """Predict using polynomial model."""
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.weights

# Test
poly_reg = PolynomialRegression(degree=3)
poly_reg.fit(X, y)
y_pred = poly_reg.predict(X_test)
```

## 📊 Bias-Variance Tradeoff

```
Degree 1 (Linear):
  - High bias (underfits curved data)
  - Low variance (stable)

Degree 3-5 (Moderate):
  - Balanced bias-variance
  - Good generalization

Degree 15+ (High):
  - Low bias (fits training data)
  - High variance (overfits, wiggly)
```

## ⚠️ Overfitting Example

```python
from sklearn.metrics import mean_squared_error

# Train/test split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

degrees = range(1, 20)
train_errors = []
test_errors = []

for degree in degrees:
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_reg.fit(X_train, y_train)
    
    train_errors.append(mean_squared_error(y_train, poly_reg.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, poly_reg.predict(X_test)))

# Plot
plt.plot(degrees, train_errors, label='Train Error')
plt.plot(degrees, test_errors, label='Test Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.title('Overfitting: Train vs Test Error')
plt.show()
```

**Observation**: Test error increases after optimal degree (overfitting)!

---

# Ridge Regression (L2)

## 📘 Concept Overview

**Ridge Regression** adds **L2 penalty** to prevent overfitting by shrinking coefficients.

Also called **Tikhonov regularization** or **weight decay**.

## 🧮 Mathematical Foundation

### Loss Function

```
L(w) = MSE + λ ‖w‖²₂
     = (1/n) ‖y - Xw‖² + λ Σⱼ wⱼ²
```

Where:
- λ ≥ 0 = regularization strength
- ‖w‖²₂ = w₁² + w₂² + ... + wₐ²

**Note**: Typically don't penalize intercept w₀

### Closed-Form Solution

Minimize L(w):

```
∂L/∂w = -2X^T(y - Xw) + 2λw = 0
X^T Xw + λIw = X^T y
(X^T X + λI)w = X^T y
```

**Ridge solution**:
```
w_ridge = (X^T X + λI)^(-1) X^T y
```

**Advantage over OLS**: Always invertible (X^T X + λI positive definite)!

### Effect of λ

```
λ = 0: No regularization (standard OLS)
λ → ∞: All weights → 0 (predict mean)

Small λ: Weak regularization (risk overfitting)
Large λ: Strong regularization (risk underfitting)
```

## 🧠 Geometric Interpretation

**Constrained optimization**:
```
minimize ‖y - Xw‖²
subject to ‖w‖² ≤ t
```

Ridge finds weights in **sphere** ‖w‖² ≤ t that minimize MSE.

```
        w₂
         │
         │    ╱│╲
         │  ╱  │  ╲
    ─────┼───●──────► w₁
         │╲  Circle  ╱
         │  ╲ ‖w‖²≤t╱
         
● = Ridge solution (on circle boundary)
```

## 🧪 Python Implementation

### Using Sklearn

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Scale features for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge with fixed lambda
ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train_scaled, y_train)

print(f"Ridge R²: {ridge.score(X_test_scaled, y_test):.3f}")

# Ridge with cross-validation to select best alpha
alphas = np.logspace(-6, 6, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {ridge_cv.alpha_:.3f}")
print(f"Ridge CV R²: {ridge_cv.score(X_test_scaled, y_test):.3f}")
```

### From Scratch

```python
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # λ
    
    def fit(self, X, y):
        """Fit Ridge regression."""
        n_features = X.shape[1]
        
        # Add intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Ridge solution: (X^T X + λI)^(-1) X^T y
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Don't penalize intercept
        
        self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        return self
    
    def predict(self, X):
        """Predict using Ridge model."""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

# Test
ridge_custom = RidgeRegression(alpha=1.0)
ridge_custom.fit(X_train_scaled, y_train)
y_pred = ridge_custom.predict(X_test_scaled)
```

## 📊 Regularization Path

```python
from sklearn.linear_model import Ridge

alphas = np.logspace(-6, 6, 200)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

# Plot coefficient paths
plt.figure(figsize=(10, 6))
for i in range(X_train.shape[1]):
    plt.plot(alphas, [coef[i] for coef in coefs], label=f'Feature {i}' if i < 5 else '')
plt.xscale('log')
plt.xlabel('Lambda (α)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regularization Path')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Observation**: As λ increases, all coefficients shrink toward 0.

---

# Lasso Regression (L1)

## 📘 Concept Overview

**Lasso** (Least Absolute Shrinkage and Selection Operator) uses **L1 penalty** for regularization and **automatic feature selection**.

## 🧮 Mathematical Foundation

### Loss Function

```
L(w) = MSE + λ ‖w‖₁
     = (1/n) ‖y - Xw‖² + λ Σⱼ |wⱼ|
```

Where ‖w‖₁ = |w₁| + |w₂| + ... + |wₐ|

### Key Difference from Ridge

**L1 penalty produces sparse solutions** (many weights exactly 0).

**Why?** Diamond-shaped constraint region touches axes.

### No Closed-Form Solution

L1 penalty not differentiable at 0 → Use **coordinate descent** or **proximal gradient**.

## 🧠 Geometric Interpretation

**Constrained optimization**:
```
minimize ‖y - Xw‖²
subject to ‖w‖₁ ≤ t
```

```
        w₂
         │
        ╱│╲
       ╱ │ ╲
    ──┼──●──┼──► w₁
       ╲ │ ╱
        ╲│╱
      Diamond
      ‖w‖₁≤t
         
● = Lasso solution (often on axis → sparse!)
```

**Ridge (circle)** rarely touches axes → Non-zero weights
**Lasso (diamond)** touches axes → Sparse solution

## 🧪 Python Implementation

### Using Sklearn

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso with fixed lambda
lasso = Lasso(alpha=0.1)  # alpha = λ
lasso.fit(X_train_scaled, y_train)

print(f"Lasso R²: {lasso.score(X_test_scaled, y_test):.3f}")

# Check sparsity
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"Non-zero coefficients: {n_nonzero} / {len(lasso.coef_)}")
print(f"Sparsity: {(1 - n_nonzero/len(lasso.coef_))*100:.1f}%")

# Lasso with cross-validation
alphas = np.logspace(-6, 1, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {lasso_cv.alpha_:.3f}")
print(f"Lasso CV R²: {lasso_cv.score(X_test_scaled, y_test):.3f}")
```

### Feature Selection with Lasso

```python
# Select features with non-zero coefficients
selected_features = np.where(lasso_cv.coef_ != 0)[0]

print(f"Selected {len(selected_features)} features:")
print(selected_features)

# Train model on selected features only
from sklearn.linear_model import LinearRegression

lr_selected = LinearRegression()
lr_selected.fit(X_train_scaled[:, selected_features], y_train)
score_selected = lr_selected.score(X_test_scaled[:, selected_features], y_test)

print(f"R² with selected features: {score_selected:.3f}")
```

## 📊 Regularization Path

```python
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_train_scaled, y_train, alphas=np.logspace(-6, 1, 200))

# Plot
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i], label=f'Feature {i}' if i < 5 else '')
plt.xscale('log')
plt.xlabel('Lambda (α)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regularization Path')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Observation**: Coefficients become exactly 0 as λ increases (sparse!).

---

# Elastic Net

## 📘 Concept Overview

**Elastic Net** combines **L1 and L2** penalties, getting benefits of both.

## 🧮 Mathematical Foundation

### Loss Function

```
L(w) = MSE + λ₁ ‖w‖₁ + λ₂ ‖w‖²₂
     = MSE + λ [α‖w‖₁ + (1-α)/2 ‖w‖²₂]
```

Where:
- λ = overall regularization strength
- α ∈ [0, 1] = L1 ratio
  - α = 0: Pure Ridge
  - α = 1: Pure Lasso
  - α = 0.5: Equal L1 and L2

### Advantages

1. **Sparsity** (from L1): Feature selection
2. **Stability** (from L2): Better with correlated features
3. **Grouped selection**: Selects correlated features together

## 🧪 Python Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Elastic Net with fixed parameters
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = α
elastic.fit(X_train_scaled, y_train)

print(f"Elastic Net R²: {elastic.score(X_test_scaled, y_test):.3f}")
print(f"Non-zero coefficients: {np.sum(elastic.coef_ != 0)}")

# Elastic Net with cross-validation
elastic_cv = ElasticNetCV(
    alphas=np.logspace(-6, 1, 50),
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    cv=5,
    random_state=42
)
elastic_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {elastic_cv.alpha_:.3f}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_:.3f}")
print(f"Elastic Net CV R²: {elastic_cv.score(X_test_scaled, y_test):.3f}")
```

## 📊 L1 Ratio Effect

```python
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]
results = []

for ratio in l1_ratios:
    elastic = ElasticNet(alpha=0.1, l1_ratio=ratio)
    elastic.fit(X_train_scaled, y_train)
    
    r2 = elastic.score(X_test_scaled, y_test)
    sparsity = 1 - np.sum(elastic.coef_ != 0) / len(elastic.coef_)
    
    results.append((ratio, r2, sparsity))
    print(f"l1_ratio={ratio:.2f}: R²={r2:.3f}, Sparsity={sparsity*100:.1f}%")
```

---

# Regularization Comparison

## 📊 Side-by-Side Comparison

| Aspect | Ridge (L2) | Lasso (L1) | Elastic Net |
|--------|------------|-----------|-------------|
| **Penalty** | ‖w‖²₂ = Σwᵢ² | ‖w‖₁ = Σ\|wᵢ\| | α‖w‖₁ + (1-α)‖w‖²₂ |
| **Sparsity** | No (all weights non-zero) | ✓ Yes (many exactly 0) | ✓ Yes |
| **Closed-form** | ✓ Yes | ✗ No (coordinate descent) | ✗ No |
| **Feature selection** | ✗ No | ✓ Yes | ✓ Yes |
| **Correlated features** | Distributes weight equally | Picks one arbitrarily | ✓ Groups together |
| **Multicollinearity** | ✓ Handles well | Can be unstable | ✓ Handles well |
| **Computation** | Fast (closed-form) | Moderate | Moderate |
| **Use case** | Many small weights | Sparse features | Balance of both |

## 🧪 Comparative Example

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(model.coef_ != 0)
    else:
        n_nonzero = len(model.coef_)
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Non-zero coefs': n_nonzero
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

## 🔄 When to Use Each

**Use Ridge when**:
- All features potentially useful
- Features are correlated
- Want smooth coefficient shrinkage
- Prefer computational speed

**Use Lasso when**:
- Need feature selection
- Many irrelevant features
- Want interpretable sparse model
- Features mostly independent

**Use Elastic Net when**:
- Need feature selection + stability
- Highly correlated features (groups)
- Best all-around choice for unknown data
- Tuning l1_ratio gives flexibility

## 📊 Bayesian Interpretation

Regularization = Prior distribution on weights

| Regularization | Equivalent Prior |
|----------------|------------------|
| **Ridge (L2)** | Gaussian: w ~ N(0, σ²I) |
| **Lasso (L1)** | Laplace: w ~ Laplace(0, b) |
| **Elastic Net** | Mixture of Gaussian and Laplace |

**MAP estimation** (Maximum A Posteriori):
```
argmax P(w|data) = argmax [P(data|w) × P(w)]
                 = argmin [-log P(data|w) - log P(w)]
                 = argmin [MSE + Regularization]
```

---

# 🔥 MCQs

### Q1. Polynomial regression is:
**Options:**
- A) Non-linear in parameters
- B) Linear in parameters ✓
- C) Always better than linear
- D) Doesn't use linear regression

**Explanation**: ŷ = w^T φ(x) is linear in w (non-linear feature transformation).

---

### Q2. Ridge regression penalty is:
**Options:**
- A) ‖w‖₁
- B) ‖w‖²₂ ✓
- C) ‖w‖₀
- D) ‖w‖∞

**Explanation**: Ridge uses L2 penalty = Σwᵢ².

---

### Q3. Lasso produces:
**Options:**
- A) All weights equal
- B) Sparse solutions (many zeros) ✓
- C) Smooth solutions
- D) Negative weights only

**Explanation**: L1 penalty forces many weights to exactly 0.

---

### Q4. Ridge closed-form solution is:
**Options:**
- A) (X^T X)^(-1) X^T y
- B) (X^T X + λI)^(-1) X^T y ✓
- C) X^T y
- D) (λI)^(-1) X^T y

**Explanation**: Ridge adds λI to make X^T X + λI invertible.

---

### Q5. Lasso is better than Ridge for:
**Options:**
- A) Computational speed
- B) Feature selection ✓
- C) Correlated features
- D) All features important

**Explanation**: Lasso L1 penalty produces sparse solutions (automatic feature selection).

---

### Q6. Elastic Net parameter l1_ratio = 0 gives:
**Options:**
- A) Lasso
- B) Ridge ✓
- C) OLS
- D) No regularization

**Explanation**: l1_ratio=0 → pure L2 penalty (Ridge).

---

### Q7. As λ → ∞ in Ridge:
**Options:**
- A) Weights → ∞
- B) Weights → 0 ✓
- C) Weights unchanged
- D) Weights → random

**Explanation**: Strong regularization shrinks all weights toward 0.

---

### Q8. Polynomial degree too high causes:
**Options:**
- A) Underfitting
- B) Overfitting ✓
- C) Better generalization
- D) Faster training

**Explanation**: High degree fits noise (high variance, overfitting).

---

### Q9. Ridge is better than Lasso for:
**Options:**
- A) Feature selection
- B) Sparse solutions
- C) Grouped correlated features ✓
- D) Interpretability

**Explanation**: Ridge distributes weight among correlated features; Lasso picks one arbitrarily.

---

### Q10. Which requires feature scaling?
**Options:**
- A) Decision trees
- B) Ridge regression ✓
- C) Neither
- D) Only Lasso

**Explanation**: Regularization penalties sensitive to feature scale (both Ridge and Lasso need scaling).

---

### Q11. L1 penalty geometric constraint is:
**Options:**
- A) Circle
- B) Diamond ✓
- C) Square
- D) Ellipse

**Explanation**: ‖w‖₁ ≤ t forms diamond in 2D (touches axes → sparsity).

---

### Q12. Elastic Net l1_ratio = 1 gives:
**Options:**
- A) Ridge
- B) Lasso ✓
- C) OLS
- D) Equal L1 and L2

**Explanation**: l1_ratio=1 → pure L1 penalty (Lasso).

---

### Q13. Ridge penalty on intercept:
**Options:**
- A) Always applied
- B) Typically not applied ✓
- C) Doubled
- D) Halved

**Explanation**: Usually don't penalize intercept (only slopes).

---

### Q14. Number of polynomial features for d features, degree 2:
**Options:**
- A) d
- B) d²
- C) d(d+1)/2 + d + 1 ✓
- D) 2d

**Explanation**: Includes original, squared, and cross terms: 1 + d + d(d+1)/2.

---

### Q15. Which has no closed-form solution?
**Options:**
- A) OLS
- B) Ridge
- C) Lasso ✓
- D) All have closed-form

**Explanation**: L1 penalty not differentiable at 0 (use coordinate descent).

---

# ⚠️ Common Mistakes

1. **Not scaling features before regularization**: Penalty unfair to large-scale features

2. **Regularizing intercept**: Usually should not penalize w₀

3. **Using too high polynomial degree**: Overfitting (use cross-validation)

4. **Confusing sklearn alpha with λ**: Sklearn's alpha = λ (regularization strength)

5. **Expecting Lasso to always outperform Ridge**: Depends on data sparsity

6. **Not cross-validating regularization strength**: Use CV to tune λ

7. **Using OLS when features correlated**: Ridge/Elastic Net better

8. **Assuming more features always better**: Regularization helps prevent overfitting

9. **Comparing models trained on unscaled data**: Regularization sensitive to scale

10. **Ignoring sparsity in Lasso**: Many coefficients exactly 0 (feature selection!)

---

# ⭐ One-Line Exam Facts

1. **Polynomial regression**: ŷ = w^T φ(x) where φ(x) contains polynomial terms (still linear in w)

2. **Ridge loss**: MSE + λ‖w‖²₂ (L2 penalty)

3. **Lasso loss**: MSE + λ‖w‖₁ (L1 penalty)

4. **Elastic Net loss**: MSE + λ[α‖w‖₁ + (1-α)‖w‖²₂]

5. **Ridge closed-form**: w = (X^T X + λI)^(-1) X^T y

6. **Lasso produces sparse solutions** (many weights exactly 0)

7. **Ridge produces non-sparse solutions** (all weights non-zero but small)

8. **L2 penalty** = Gaussian prior, **L1 penalty** = Laplace prior

9. **Ridge better for correlated features** (distributes weight)

10. **Lasso better for feature selection** (automatic sparsity)

11. **Elastic Net** combines benefits of Ridge and Lasso

12. **l1_ratio = 0** → Ridge, **l1_ratio = 1** → Lasso

13. **Higher λ** → stronger regularization → smaller weights

14. **Polynomial features** grow exponentially: O(d^n) for n features, degree d

15. **Always scale features** before applying Ridge, Lasso, or Elastic Net

---

**End of Session 11**
