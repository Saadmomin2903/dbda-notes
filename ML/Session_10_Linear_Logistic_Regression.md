# Session 10 – Linear & Logistic Regression

## 📚 Table of Contents
1. [Linear Regression](#linear-regression)
2. [Gradient Descent](#gradient-descent)
3. [Logistic Regression](#logistic-regression)
4. [Multi-class Logistic Regression](#multi-class-logistic-regression)
5. [Model Evaluation](#model-evaluation)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Linear Regression

## 📘 Concept Overview

**Linear Regression** models relationship between features X and continuous target y as **linear combination**.

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₐxₐ = w^T x
```

Where:
- w₀ = intercept (bias)
- w = [w₁, ..., wₐ]^T = weights (coefficients)
- x = [1, x₁, ..., xₐ]^T = features (augmented with 1 for intercept)

## 🧮 Mathematical Foundation

### Loss Function: Mean Squared Error

```
L(w) = (1/n) Σᵢ (yᵢ - ŷᵢ)²
     = (1/n) Σᵢ (yᵢ - w^T xᵢ)²
     = (1/n) ‖y - Xw‖²
```

**Objective**: Find w that minimizes L(w)

### Ordinary Least Squares (OLS) - Closed Form

**Minimize**: L(w) = ‖y - Xw‖²

**Take derivative and set to zero**:
```
∂L/∂w = -2X^T(y - Xw) = 0
X^T Xw = X^T y
```

**Normal Equation**:
```
w* = (X^T X)^(-1) X^T y
```

**Assumption**: X^T X is invertible (full rank)

### Derivation

Starting from L(w) = ‖y - Xw‖²:

```
L(w) = (y - Xw)^T (y - Xw)
     = y^T y - y^T Xw - w^T X^T y + w^T X^T Xw
     = y^T y - 2w^T X^T y + w^T X^T Xw

∂L/∂w = -2X^T y + 2X^T Xw = 0

X^T Xw = X^T y

w = (X^T X)^(-1) X^T y
```

### Geometric Interpretation

**Projection**: ŷ = Xw is projection of y onto column space of X

**Residual**: e = y - ŷ is orthogonal to column space

```
X^T e = X^T (y - Xw) = 0
```

This is exactly the normal equation!

## 🧪 Python Implementation

### Using Sklearn

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Coefficients
print(f"Intercept (w₀): {lr.intercept_:.3f}")
print(f"Coefficient (w₁): {lr.coef_[0]:.3f}")

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")

# Visualize
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, label='Actual', alpha=0.6)
plt.plot(X_test, y_pred, 'r-', label='Predicted', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
```

### From Scratch (Normal Equation)

```python
class LinearRegressionOLS:
    def fit(self, X, y):
        """Fit using normal equation."""
        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: w = (X^T X)^(-1) X^T y
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

# Test
lr_custom = LinearRegressionOLS()
lr_custom.fit(X_train, y_train)
y_pred_custom = lr_custom.predict(X_test)

print(f"Custom MSE: {mean_squared_error(y_test, y_pred_custom):.3f}")
print(f"Weights: {lr_custom.weights}")
```

## ⚙️ Assumptions of Linear Regression

### 1. Linearity

Relationship between X and y is linear.

**Test**: Residual plot should show no pattern

```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### 2. Independence

Observations are independent.

**Violation**: Time series (autocorrelation)

### 3. Homoscedasticity

Constant variance of errors (σ² same for all x).

**Test**: Residuals vs fitted plot (should be uniform spread)

**Violation**: Heteroscedasticity (funnel shape in residuals)

### 4. Normality

Errors are normally distributed.

**Test**: QQ-plot, Shapiro-Wilk test

```python
from scipy import stats

stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()
```

### 5. No Multicollinearity

Features not highly correlated.

**Test**: VIF (Variance Inflation Factor)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

**Rule**: VIF > 10 indicates problematic multicollinearity

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Simple & interpretable**: Coefficients show feature importance
2. **Fast training**: Closed-form solution O(d³)
3. **Low variance**: Stable predictions
4. **Probabilistic**: Can compute confidence intervals
5. **Baseline**: Good starting point

### Weaknesses ✗
1. **Assumes linearity**: Fails for non-linear relationships
2. **Sensitive to outliers**: MSE heavily penalizes large errors
3. **Multicollinearity**: Unstable coefficients
4. **Extrapolation**: Poor predictions outside training range
5. **Feature engineering**: May need manual polynomial features

---

# Gradient Descent

## 📘 Concept Overview

**Gradient Descent** is an iterative optimization algorithm to minimize loss function.

**Alternative to closed-form** when:
- X^T X not invertible
- Large datasets (d or n very large)
- Online learning

## 🧮 Algorithm

```
1. Initialize: w ← random values

2. Repeat until convergence:
   a) Compute gradient: ∇L(w) = ∂L/∂w
   b) Update: w ← w - α ∇L(w)
   
3. Return: w
```

Where α = learning rate (step size)

### Gradient Derivation

For MSE loss:
```
L(w) = (1/n) Σᵢ (yᵢ - w^T xᵢ)²

∂L/∂w = (2/n) Σᵢ (w^T xᵢ - yᵢ) xᵢ
      = (2/n) X^T (Xw - y)
```

### Update Rule

```
w ← w - α × (2/n) X^T (Xw - y)
```

## ⚙️ Variants

### 1. Batch Gradient Descent

Use **all** samples to compute gradient.

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for epoch in range(epochs):
        # Predictions
        y_pred = X @ w
        
        # Gradient
        gradient = (2/n_samples) * X.T @ (y_pred - y)
        
        # Update
        w -= learning_rate * gradient
        
        # Loss (optional, for monitoring)
        if epoch % 100 == 0:
            loss = np.mean((y - y_pred) ** 2)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return w
```

**Pros**: Stable convergence, exact gradient
**Cons**: Slow for large datasets (computes on all n samples)

### 2. Stochastic Gradient Descent (SGD)

Use **single** sample to compute gradient.

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for epoch in range(epochs):
        for i in range(n_samples):
            # Single sample
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            # Prediction
            y_pred = xi @ w
            
            # Gradient (single sample)
            gradient = 2 * xi.T @ (y_pred - yi)
            
            # Update
            w -= learning_rate * gradient
    
    return w
```

**Pros**: Fast updates, can escape local minima (noisy)
**Cons**: Noisy convergence, may not converge exactly

### 3. Mini-Batch Gradient Descent

Use **batch** of samples (e.g., 32, 64, 128).

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Predictions
            y_pred = X_batch @ w
            
            # Gradient
            gradient = (2/batch_size) * X_batch.T @ (y_pred - y_batch)
            
            # Update
            w -= learning_rate * gradient
    
    return w
```

**Pros**: Balance between batch and SGD, vectorized operations
**Cons**: Requires tuning batch size

## 📊 Learning Rate Selection

```
Too small: Slow convergence
Too large: Divergence (overshooting)
```

**Strategies**:
1. **Fixed**: α = constant
2. **Decay**: α = α₀ / (1 + decay × epoch)
3. **Adaptive**: Adam, RMSprop (adjust per parameter)

```python
# Learning rate decay
initial_lr = 0.1
decay_rate = 0.01

for epoch in range(epochs):
    lr = initial_lr / (1 + decay_rate * epoch)
    w -= lr * gradient
```

## 🧪 Sklearn SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',  # Decay
    eta0=0.01,                    # Initial learning rate
    random_state=42
)

sgd_reg.fit(X_train, y_train)
y_pred = sgd_reg.predict(X_test)
```

---

# Logistic Regression

## 📘 Concept Overview

**Logistic Regression** is a **classification** algorithm (despite the name!) that models probability of class membership.

**Binary classification**: y ∈ {0, 1}

## 🧮 Mathematical Foundation

### Logistic (Sigmoid) Function

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties**:
- Range: (0, 1) — perfect for probabilities!
- σ(0) = 0.5
- σ(z) → 1 as z → ∞
- σ(z) → 0 as z → -∞
- Derivative: σ'(z) = σ(z)(1 - σ(z))

### Model

```
P(y=1|x) = σ(w^T x) = 1 / (1 + e^(-w^T x))
```

**Log-odds (logit)**:
```
log[P(y=1|x) / P(y=0|x)] = w^T x
```

Linear in w!

### Decision Boundary

Predict class 1 if P(y=1|x) > 0.5:

```
σ(w^T x) > 0.5
w^T x > 0
```

**Decision boundary**: w^T x = 0 (hyperplane)

### Loss Function: Binary Cross-Entropy

Cannot use MSE (non-convex for sigmoid)!

**Negative log-likelihood**:

```
L(w) = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

Where ŷᵢ = σ(w^T xᵢ)

**Derivation** (maximum likelihood):

```
Likelihood: L = ∏ᵢ P(yᵢ|xᵢ) = ∏ᵢ ŷᵢ^yᵢ (1-ŷᵢ)^(1-yᵢ)

Log-likelihood: log L = Σᵢ [yᵢ log ŷᵢ + (1-yᵢ) log(1-ŷᵢ)]

Minimize negative log-likelihood → Cross-entropy
```

### Gradient

```
∂L/∂w = (1/n) Σᵢ (ŷᵢ - yᵢ) xᵢ
      = (1/n) X^T (ŷ - y)
```

**Same form as linear regression!** (but ŷ = σ(w^T x))

### Optimization

**No closed-form solution** → Use gradient descent

```
w ← w - α × (1/n) X^T (σ(Xw) - y)
```

## 🧪 Python Implementation

### Using Sklearn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
log_reg = LogisticRegression(
    solver='lbfgs',      # Optimization algorithm
    max_iter=1000,
    random_state=42
)

log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(f"\n{classification_report(y_test, y_pred)}")

# Coefficients
print(f"\nIntercept: {log_reg.intercept_[0]:.3f}")
print(f"Coefficients: {log_reg.coef_[0][:5]}")  # First 5
```

### From Scratch

```python
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
    
    def sigmoid(self, z):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train using gradient descent."""
        n_samples, n_features = X.shape
        
        # Add intercept
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Initialize weights
        self.weights = np.zeros(n_features + 1)
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Predictions
            z = X_b @ self.weights
            y_pred = self.sigmoid(z)
            
            # Gradient
            gradient = (1/n_samples) * X_b.T @ (y_pred - y)
            
            # Update
            self.weights -= self.lr * gradient
            
            # Loss (for monitoring)
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-8) + 
                               (1 - y) * np.log(1 - y_pred + 1e-8))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.weights)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

# Test
log_reg_custom = LogisticRegressionGD(learning_rate=0.1, epochs=1000)
log_reg_custom.fit(X_train, y_train)
y_pred_custom = log_reg_custom.predict(X_test)

print(f"\nCustom Accuracy: {accuracy_score(y_test, y_pred_custom):.3f}")
```

## 📊 Decision Boundary Visualization

```python
def plot_decision_boundary(X, y, model):
    """Plot 2D decision boundary."""
    h = 0.02  # Step size
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# For 2D data
X_2d, y_2d = make_classification(n_samples=200, n_features=2, n_informative=2,
                                  n_redundant=0, random_state=42)
model_2d = LogisticRegression().fit(X_2d, y_2d)
plot_decision_boundary(X_2d, y_2d, model_2d)
```

## ⚙️ Regularization

### L2 Regularization (Ridge)

```
L(w) = CrossEntropy + λ ‖w‖²
```

**Effect**: Shrinks weights, prevents overfitting

```python
log_reg_l2 = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ
```

### L1 Regularization (Lasso)

```
L(w) = CrossEntropy + λ ‖w‖₁
```

**Effect**: Sparse weights (feature selection)

```python
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
```

**Note**: Smaller C → stronger regularization

---

# Multi-class Logistic Regression

## 📘 Softmax Regression

For K classes, generalize to **softmax** function.

### Softmax Function

```
P(y=k|x) = exp(wₖ^T x) / Σⱼ exp(wⱼ^T x)
```

**Properties**:
- Σₖ P(y=k|x) = 1 (probabilities sum to 1)
- Reduces to sigmoid for K=2

### Loss: Categorical Cross-Entropy

```
L(W) = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
```

Where yᵢₖ = 1 if sample i is class k, else 0 (one-hot)

## ⚙️ Multi-class Strategies

### 1. One-vs-Rest (OvR)

Train K binary classifiers:
- Classifier k: class k vs all others

**Prediction**: Choose class with highest confidence

```python
log_reg_ovr = LogisticRegression(multi_class='ovr')
```

### 2. One-vs-One (OvO)

Train K(K-1)/2 binary classifiers:
- Classifier (i,j): class i vs class j

**Prediction**: Majority voting

```python
from sklearn.multiclass import OneVsOneClassifier

log_reg_ovo = OneVsOneClassifier(LogisticRegression())
```

### 3. Multinomial (Softmax)

Train single multi-class model.

```python
log_reg_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

## 🧪 Multi-class Example

```python
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Softmax regression
log_reg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg_multi.fit(X_train, y_train)

# Predictions
y_pred = log_reg_multi.predict(X_test)
y_proba = log_reg_multi.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Multi-class Accuracy: {accuracy:.3f}")

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(log_reg_multi, X_test, y_test)
plt.show()
```

---

# Model Evaluation

## 📊 Classification Metrics

### ROC-AUC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.plot(recall, precision, label=f'PR (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve')
plt.show()
```

## 📊 Regression Metrics

Covered in Session 6 (MAE, MSE, RMSE, R²).

---

# 🔥 MCQs

### Q1. Linear regression loss function is:
**Options:**
- A) Cross-entropy
- B) Hinge loss
- C) Mean Squared Error ✓
- D) Log loss

**Explanation**: Linear regression minimizes MSE = (1/n)Σ(yᵢ - ŷᵢ)².

---

### Q2. Normal equation is:
**Options:**
- A) w = X^T y
- B) w = (X^T X)^(-1) X^T y ✓
- C) w = X^(-1) y
- D) w = (X X^T)^(-1) y

**Explanation**: Closed-form OLS solution.

---

### Q3. Logistic regression is used for:
**Options:**
- A) Regression
- B) Classification ✓
- C) Clustering
- D) Dimensionality reduction

**Explanation**: Despite the name, logistic regression is a classifier.

---

### Q4. Sigmoid function range is:
**Options:**
- A) (-∞, ∞)
- B) [0, 1]
- C) (0, 1) ✓
- D) [-1, 1]

**Explanation**: σ(z) ∈ (0, 1) for all z (asymptotes at 0 and 1).

---

### Q5. Logistic regression decision boundary is:
**Options:**
- A) Non-linear
- B) Linear (hyperplane) ✓
- C) Quadratic
- D) Exponential

**Explanation**: w^T x = 0 is a hyperplane.

---

### Q6. Gradient descent update rule:
**Options:**
- A) w ← w + α ∇L
- B) w ← w - α ∇L ✓
- C) w ← α ∇L
- D) w ← w / α

**Explanation**: Move opposite to gradient (downhill).

---

### Q7. Mini-batch gradient descent uses:
**Options:**
- A) All samples
- B) Single sample
- C) Batch of samples ✓
- D) Random samples with replacement

**Explanation**: Processes batches (e.g., 32, 64) at a time.

---

### Q8. Logistic regression loss function is:
**Options:**
- A) MSE
- B) Binary cross-entropy ✓
- C) Hinge loss
- D) MAE

**Explanation**: Negative log-likelihood (cross-entropy) for logistic regression.

---

### Q9. OLS assumes:
**Options:**
- A) Non-linear relationship
- B) Constant error variance (homoscedasticity) ✓
- C) Correlated features preferred
- D) Outliers don't matter

**Explanation**: One of key OLS assumptions.

---

### Q10. For logistic regression, C parameter in sklearn:
**Options:**
- A) C = λ (regularization strength)
- B) C = 1/λ (inverse regularization) ✓
- C) C = learning rate
- D) C = number of classes

**Explanation**: Larger C → less regularization.

---

### Q11. Softmax is used for:
**Options:**
- A) Binary classification
- B) Multi-class classification ✓
- C) Regression
- D) Clustering

**Explanation**: Generalizes sigmoid to K classes.

---

### Q12. Sigmoid derivative is:
**Options:**
- A) σ(z)
- B) 1 - σ(z)
- C) σ(z)(1 - σ(z)) ✓
- D) σ(z)²

**Explanation**: d/dz σ(z) = σ(z)(1 - σ(z)).

---

### Q13. Linear regression is sensitive to:
**Options:**
- A) Feature scaling
- B) Outliers ✓
- C) Class imbalance
- D) Categorical features

**Explanation**: MSE heavily penalizes large errors (outliers).

---

### Q14. Which has no closed-form solution?
**Options:**
- A) Linear regression
- B) Logistic regression ✓
- C) Ridge regression
- D) LDA

**Explanation**: Logistic regression requires iterative optimization (gradient descent).

---

### Q15. One-vs-Rest trains:
**Options:**
- A) 1 classifier
- B) K classifiers ✓
- C) K(K-1)/2 classifiers
- D) K² classifiers

**Explanation**: K binary classifiers (one per class vs rest).

---

# ⚠️ Common Mistakes

1. **Not checking linear regression assumptions**: Leads to poor predictions

2. **Using MSE for logistic regression**: Non-convex, use cross-entropy

3. **Forgetting to scale features for gradient descent**: Slow convergence

4. **Using linear regression for classification**: Use logistic regression instead

5. **Confusing C and λ in sklearn**: C = 1/λ (larger C → less regularization)

6. **Not handling multicollinearity**: Unstable coefficients in linear regression

7. **Extrapolating with linear regression**: Poor predictions outside training range

8. **Interpreting logistic regression output as probability without calibration**: May need calibration

9. **Using batch GD for large datasets**: Too slow, use mini-batch or SGD

10. **Threshold at 0.5 for imbalanced data**: Adjust threshold based on precision-recall tradeoff

---

# ⭐ One-Line Exam Facts

1. **Linear regression**: ŷ = w^T x (minimizes MSE)

2. **Normal equation**: w = (X^T X)^(-1) X^T y (closed-form OLS)

3. **Gradient descent update**: w ← w - α ∇L(w)

4. **Sigmoid function**: σ(z) = 1/(1 + e^(-z)), range (0, 1)

5. **Logistic regression**: Predicts P(y=1|x) = σ(w^T x)

6. **Decision boundary**: w^T x = 0 (hyperplane)

7. **Logistic loss**: Binary cross-entropy (negative log-likelihood)

8. **Gradient** (linear & logistic): X^T (ŷ - y) (same form!)

9. **Batch GD**: Uses all samples (slow but stable)

10. **SGD**: Uses single sample (fast but noisy)

11. **Mini-batch GD**: Uses batch of samples (best trade-off)

12. **Softmax**: Multi-class generalization of sigmoid

13. **OvR**: K classifiers (class k vs rest)

14. **OvO**: K(K-1)/2 classifiers (class i vs class j)

15. **L2 penalty** smooths weights, **L1 penalty** creates sparsity

---

**End of Session 10**
