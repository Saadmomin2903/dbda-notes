# Session 7 – Tree-Based Models & Ensemble Methods

## 📚 Table of Contents
1. [Decision Trees](#decision-trees)
2. [Random Forest](#random-forest)
3. [Gradient Boosting](#gradient-boosting)
4. [XGBoost](#xgboost)
5. [LightGBM](#lightgbm)
6. [CatBoost](#catboost)
7. [Model Stacking](#model-stacking)
8. [MCQs](#mcqs)
9. [Common Mistakes](#common-mistakes)
10. [One-Line Exam Facts](#one-line-exam-facts)

---

# Decision Trees

## 📘 Concept Overview

**Decision Trees** recursively partition feature space using **if-then-else** rules to make predictions.

### Tree Structure

```
           [Root Node]
         Age <= 30?
        /          \
      Yes          No
       |            |
  [Salary<=50k?]  [Predict: Approved]
    /        \
  Yes        No
   |          |
[Reject]  [Approved]
```

**Components**:
- **Root Node**: Top decision (best first split)
- **Internal Nodes**: Decision based on feature
- **Leaf Nodes**: Final prediction
- **Branches**: Decision rules

## 🧮 Mathematical Foundation

### CART (Classification and Regression Trees)

Developed by Breiman et al. (1984).

### Objective

Find splits that **maximize homogeneity** (purity) of child nodes.

### Splitting Criterion

#### Classification: Gini Impurity

```
Gini(D) = 1 - Σᵢ pᵢ²
```

where pᵢ = proportion of class i in dataset D.

**Interpretation**:
- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity (binary, 50-50 split)

**Information Gain**:
```
IG = Gini(parent) - Σ (|Dᵢ|/|D|) Gini(Dᵢ)
```

**Example**:
```
Parent: [40 class-0, 60 class-1]
Gini(parent) = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48

Split on "Age <= 30":
  Left: [30 class-0, 10 class-1]
    Gini(left) = 1 - (0.75² + 0.25²) = 0.375
  Right: [10 class-0, 50 class-1]
    Gini(right) = 1 - (0.17² + 0.83²) = 0.28

Weighted Gini = (40/100)×0.375 + (60/100)×0.28 = 0.318
Information Gain = 0.48 - 0.318 = 0.162
```

#### Classification: Entropy (Alternative)

```
Entropy(D) = -Σᵢ pᵢ log₂(pᵢ)
```

**Same as Session 1 entropy!**

**Information Gain** (mutual information):
```
IG = Entropy(parent) - Σ (|Dᵢ|/|D|) Entropy(Dᵢ)
```

#### Regression: Variance Reduction

```
Variance(D) = (1/n) Σᵢ (yᵢ - ȳ)²
```

**Variance Reduction**:
```
VR = Var(parent) - Σ (|Dᵢ|/|D|) Var(Dᵢ)
```

Equivalent to minimizing **MSE** of predictions.

## ⚙️ CART Algorithm

```
function BuildTree(D, features):
    # Stopping criteria
    if all samples have same class:
        return LeafNode(majority_class)
    
    if max_depth reached or min_samples < threshold:
        return LeafNode(majority_class)
    
    # Find best split
    best_gain = 0
    best_split = None
    
    for feature in features:
        for threshold in unique_values(feature):
            D_left = samples where feature <= threshold
            D_right = samples where feature > threshold
            
            gain = InformationGain(D, D_left, D_right)
            
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold)
    
    # Create decision node
    node = DecisionNode(best_split)
    
    # Recursively build subtrees
    node.left = BuildTree(D_left, features)
    node.right = BuildTree(D_right, features)
    
    return node
```

## 🧠 Key Concepts

### 1. Stopping Criteria

- **Max depth**: Limit tree depth (e.g., max_depth=10)
- **Min samples split**: Minimum samples to split node (e.g., min_samples_split=20)
- **Min samples leaf**: Minimum samples in leaf (e.g., min_samples_leaf=10)
- **Max leaf nodes**: Maximum number of leaves
- **Min impurity decrease**: Minimum gain required to split

### 2. Tree Pruning

**Problem**: Deep trees overfit.

**Solution**: Prune to simplify.

#### Cost-Complexity Pruning (α-pruning)

Minimize:
```
Cost(T) = Error(T) + α × |Leaves(T)|
```

Where:
- Error(T) = misclassification error
- |Leaves(T)| = number of leaf nodes
- α ≥ 0 = complexity parameter (larger α → more pruning)

**Pre-pruning**: Stop growing early (max_depth, min_samples_split)
**Post-pruning**: Grow full tree, then remove branches

### 3. Handling Missing Values

CART handles missing values by:
1. **Surrogate splits**: Find alternative split that mimics best split
2. **Separate branch**: Create third branch for missing values

Scikit-learn: **Doesn't handle** missing values (must impute first)

## 🧪 Python Implementation

### Classification

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
clf = DecisionTreeClassifier(
    criterion='gini',           # or 'entropy'
    max_depth=3,                # limit depth
    min_samples_split=10,       # min samples to split
    min_samples_leaf=5,         # min samples in leaf
    random_state=42
)

clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, 
               feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
               class_names=['setosa', 'versicolor', 'virginica'],
               filled=True,
               rounded=True)
plt.show()

# Feature importance
importances = clf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.3f}")
```

### Regression

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")
```

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Interpretable**: Easy to visualize and explain
2. **No feature scaling needed**: Scale-invariant
3. **Handles non-linear relationships**: Arbitrary decision boundaries
4. **Mixed data types**: Continuous + categorical features
5. **Feature interactions**: Automatically captures
6. **Fast prediction**: O(log n) traversal

### Weaknesses ✗
1. **High variance**: Small data changes → completely different tree (unstable)
2. **Overfitting**: Deep trees memorize training data
3. **Greedy algorithm**: May miss global optimum (local search)
4. **Biased towards dominant classes**: In imbalanced data
5. **Axis-aligned splits**: Can't capture diagonal boundaries well
6. **Extrapolation**: Can't predict outside training range

### Bias-Variance Behavior

```
Shallow Tree → High Bias, Low Variance (Underfitting)
Deep Tree → Low Bias, High Variance (Overfitting)
```

## ⚠️ Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

---

# Random Forest

## 📘 Concept Overview

**Random Forest** is an ensemble of decision trees trained on **bootstrap samples** with **random feature subsets**.

**Combines**:
1. **Bagging** (Bootstrap Aggregating)
2. **Feature randomness**

## 🧮 Mathematical Foundation

### Bagging

**Idea**: Train multiple models on different bootstrap samples, then average predictions.

**Bootstrap sample**: Random sample with replacement (same size as original).

```
Original: [1, 2, 3, 4, 5]

Bootstrap 1: [1, 1, 3, 4, 5]
Bootstrap 2: [2, 3, 3, 4, 5]
Bootstrap 3: [1, 2, 4, 4, 5]
```

**Out-of-Bag (OOB)**: Samples not in bootstrap (~37% on average).

```
P(sample not selected) = (1 - 1/n)ⁿ → 1/e ≈ 0.37 as n → ∞
```

### Random Forest Algorithm

```
1. For b = 1 to B (number of trees):
   a) Draw bootstrap sample Dᵦ from training data
   b) Grow tree Tᵦ on Dᵦ:
      - At each split, randomly select m features (m << d)
      - Choose best split among m features
      - Grow to max size (no pruning)
   
2. Predictions:
   - Classification: Majority vote across trees
   - Regression: Average predictions
```

### Why It Works

**Variance Reduction**:

For B independent models with variance σ²:
```
Var(average) = σ² / B
```

**With correlation ρ**:
```
Var(RF) = ρσ² + (1-ρ)σ²/B
```

**Feature randomness** reduces correlation ρ between trees!

## ⚙️ Key Hyperparameters

### 1. n_estimators (B)

Number of trees.

**Effect**:
- More trees → Lower variance, better performance
- Diminishing returns after ~100-500 trees
- **Never overfits** (more trees always helps or plateaus)

### 2. max_features (m)

Number of random features per split.

**Defaults**:
- Classification: √d
- Regression: d/3

**Effect**:
- Smaller m → More randomness → Lower correlation → Lower variance
- Too small → High bias (not enough features to find good splits)

### 3. max_depth

Maximum tree depth.

**Effect**:
- Larger depth → Lower bias, higher variance per tree
- Random Forest less sensitive than single tree (ensemble averages)

### 4. min_samples_split, min_samples_leaf

Stopping criteria.

**Effect**: Control individual tree complexity.

### 5. bootstrap

Whether to use bootstrap samples.

**Default**: True
**If False**: Uses entire dataset for each tree (loses diversity)

### 6. oob_score

Use out-of-bag samples for validation.

**Advantage**: No need for separate validation set!

## 🧪 Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_features='sqrt',        # √d features per split
    max_depth=None,             # No depth limit (grow to purity)
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,             # Use bootstrap samples
    oob_score=True,             # Compute OOB score
    n_jobs=-1,                  # Parallel training
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
print(f"OOB Score: {rf.oob_score_:.3f}")
print(f"Train Accuracy: {rf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.3f}")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 5 Features:")
for i in indices[:5]:
    print(f"Feature {i}: {importances[i]:.3f}")

# Visualize feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### Regression

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_features='auto',  # d/3 for regression
    random_state=42
)

rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
```

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Low variance**: Ensemble reduces overfitting
2. **Robust to outliers**: Averaging smooths predictions
3. **Feature importance**: Built-in importance scores
4. **Parallel training**: Trees independent (fast with multi-core)
5. **OOB validation**: Free validation without holdout set
6. **Handles high dimensions**: Feature randomness prevents overfitting
7. **Non-linear**: Captures complex relationships

### Weaknesses ✗
1. **Less interpretable**: Ensemble of trees (black box)
2. **Memory intensive**: Stores B trees
3. **Slow prediction**: Must traverse B trees
4. **Extrapolation**: Can't predict outside training range (like trees)
5. **Imbalanced data**: May favor majority class
6. **Linear relationships**: Overkill (simple linear model better)

## 🔄 Random Forest vs Single Tree

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Variance** | High | Low (bagging) |
| **Overfitting** | Prone | Resistant |
| **Interpretability** | High | Low |
| **Training time** | Fast | Slower (B trees) |
| **Prediction time** | Fast | Slower (B trees) |
| **Hyperparameters** | Many | More (+ n_estimators) |
| **Accuracy** | Lower | Higher (usually) |

---

# Gradient Boosting

## 📘 Concept Overview

**Gradient Boosting** sequentially trains weak learners (usually shallow trees) where each corrects errors of previous ensemble.

**Key idea**: Fit new tree to **residuals** (errors) of current model.

## 🧮 Mathematical Foundation

### Boosting Framework

Train ensemble:
```
F(x) = Σᵢ₌₁ᴹ fᵢ(x)
```

Where each fᵢ is a weak learner (shallow tree).

### Gradient Boosting Algorithm

```
1. Initialize: F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   (e.g., average for MSE loss)

2. For m = 1 to M:
   a) Compute pseudo-residuals:
      rᵢₘ = -∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) |_{F=Fₘ₋₁}
   
   b) Fit tree hₘ to residuals {(xᵢ, rᵢₘ)}
   
   c) Find optimal step size:
      γₘ = argmin_γ Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + γhₘ(xᵢ))
   
   d) Update:
      Fₘ(x) = Fₘ₋₁(x) + ηγₘhₘ(x)
      
   (η = learning rate)

3. Output: F_M(x)
```

### For Regression (MSE Loss)

```
L(y, F(x)) = (1/2)(y - F(x))²

Gradient: -∂L/∂F = y - F(x) = residual
```

So fitting to gradient = fitting to residuals!

### For Classification (Log Loss)

```
L(y, F(x)) = log(1 + exp(-yF(x)))  (y ∈ {-1, +1})

Gradient: -∂L/∂F = y / (1 + exp(yF(x)))
```

## 🧠 Intuition

**Sequential Error Correction**:

```
Iteration 1: F₁(x) = f₁(x)
            Error: y - F₁(x)

Iteration 2: Train f₂ to predict errors
            F₂(x) = F₁(x) + η·f₂(x)
            
Iteration 3: Train f₃ to predict remaining errors
            F₃(x) = F₂(x) + η·f₃(x)

... and so on
```

**Gradient descent in function space!**

## ⚙️ Key Hyperparameters

### 1. n_estimators (M)

Number of boosting iterations.

**Effect**:
- More trees → Lower training error
- **Can overfit** if too large (unlike Random Forest!)
- Use early stopping

### 2. learning_rate (η)

Shrinkage applied to each tree.

**Effect**:
- **Smaller η → Better generalization** (but need more trees)
- Typical: 0.01 - 0.1
- **Tradeoff**: η vs M (smaller η requires larger M)

### 3. max_depth

Tree depth (usually shallow for boosting).

**Typical**: 3-8 (weak learners)
**Effect**: Deeper → More complex, higher variance

### 4. subsample

Fraction of samples for each tree (stochastic gradient boosting).

**Effect**:
- < 1.0 → Adds randomness → Reduces variance
- Typical: 0.5 - 1.0

### 5. min_samples_split, min_samples_leaf

Control tree complexity.

## 🧪 Python Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,              # Shallow trees
    subsample=0.8,            # 80% samples per tree
    random_state=42
)

gb_clf.fit(X_train, y_train)

print(f"Train Accuracy: {gb_clf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {gb_clf.score(X_test, y_test):.3f}")

# Training evolution
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gb_clf.staged_predict(X_train),
    gb_clf.staged_predict(X_test)
)):
    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Train')
plt.plot(test_scores, label='Test')
plt.xlabel('Boosting Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Gradient Boosting Learning Curve')
plt.show()
```

### Early Stopping

```python
gb_clf = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    validation_fraction=0.2,  # Use 20% for early stopping
    n_iter_no_change=10,      # Stop if no improvement for 10 iterations
    random_state=42
)

gb_clf.fit(X_train, y_train)
print(f"Best iteration: {gb_clf.n_estimators_}")
```

## 📊 Bagging vs Boosting

| Aspect | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|--------|------------------------|------------------------------|
| **Training** | Parallel (independent trees) | Sequential (dependent trees) |
| **Weak learners** | Deep trees | Shallow trees |
| **Weights** | Equal | Adaptive (focus on errors) |
| **Variance reduction** | ✓ Strong | ✓ Moderate |
| **Bias reduction** | Limited | ✓ Strong |
| **Overfitting** | Resistant | Can overfit |
| **Speed** | Fast (parallel) | Slower (sequential) |
| **Robust to noise** | ✓ Yes | ✗ Sensitive |

---

# XGBoost

## 📘 Concept Overview

**XGBoost** (Extreme Gradient Boosting) is an optimized, scalable implementation of gradient boosting.

**Key innovations**:
1. **Regularization** in objective (L1 + L2)
2. **Second-order** Taylor approximation (Newton boosting)
3. **Efficient tree learning** (histogram-based)
4. **Handling missing values**
5. **Parallel tree construction**
6. **Hardware optimization**

## 🧮 Mathematical Foundation

### Objective Function

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
```

Where:
- L = loss function (e.g., MSE, log loss)
- Ω = regularization term

**Regularization**:
```
Ω(f) = γT + (λ/2) Σⱼ wⱼ²
```

Where:
- T = number of leaves
- wⱼ = leaf weight (prediction value)
- γ = complexity penalty
- λ = L2 regularization

### Second-Order Approximation

Taylor expand loss around previous prediction:

```
L(yᵢ, ŷᵢ⁽ᵗ⁾) ≈ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + (1/2)hᵢfₜ²(xᵢ)
```

Where:
- gᵢ = ∂L/∂ŷ (first-order gradient)
- hᵢ = ∂²L/∂ŷ² (second-order Hessian)

**Optimal leaf weight**:
```
w*ⱼ = -Σᵢ∈leaf_j gᵢ / (Σᵢ∈leaf_j hᵢ + λ)
```

**Gain from split**:
```
Gain = (1/2) [(ΣgₗEFT)² / (ΣhₗEFT + λ) + (ΣgᴿIGHT)² / (Σhᴿ IGHT + λ) - (Σg)² / (Σh + λ)] - γ
```

## ⚙️ XGBoost Hyperparameters

### Tree Parameters
- `max_depth`: Maximum tree depth (default=6)
- `min_child_weight`: Minimum sum of Hessian in leaf (regularization)
- `gamma`: Minimum gain to split (γ in formula)
- `subsample`: Fraction of samples per tree
- `colsample_bytree`: Fraction of features per tree
- `colsample_bylevel`: Fraction of features per level
- `colsample_bynode`: Fraction of features per node

### Boosting Parameters
- `n_estimators`: Number of trees
- `learning_rate` (eta): Shrinkage (default=0.3)
- `objective`: Loss function (e.g., 'binary:logistic', 'reg:squarederror')
- `eval_metric`: Evaluation metric

### Regularization
- `reg_alpha`: L1 regularization on weights
- `reg_lambda`: L2 regularization on weights (default=1)

## 🧪 Python Implementation

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,          # L1 regularization
    reg_lambda=1,         # L2 regularization
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# Fit with early stopping
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=10,
    verbose=False
)

# Predictions
y_pred = xgb_clf.predict(X_test)
y_prob = xgb_clf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"Best iteration: {xgb_clf.best_iteration}")

# Feature importance
xgb.plot_importance(xgb_clf, max_num_features=10)
plt.show()
```

### Native API (More Features)

```python
# Convert to DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'max_depth': 5,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train with early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Predict
y_prob = model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)
```

## 📊 XGBoost vs Sklearn GradientBoosting

| Feature | Sklearn GB | XGBoost |
|---------|-----------|---------|
| **Regularization** | No | ✓ L1 + L2 |
| **Second-order** | No | ✓ Yes (Hessian) |
| **Missing values** | No | ✓ Learned direction |
| **Parallel** | No | ✓ Tree construction |
| **Speed** | Slower | ✓ Faster |
| **Memory** | Higher | ✓ Lower (column block) |
| **Cross-validation** | Manual | ✓ Built-in cv() |

---

# LightGBM

## 📘 Concept Overview

**LightGBM** (Light Gradient Boosting Machine) by Microsoft.

**Key innovations**:
1. **Leaf-wise growth** (vs level-wise)
2. **Gradient-based One-Side Sampling (GOSS)**
3. **Exclusive Feature Bundling (EFB)**
4. **Histogram-based** splitting

## 🧮 Key Differences from XGBoost

### 1. Leaf-Wise vs Level-Wise Growth

**Level-wise** (XGBoost):
```
        [ ]
       /   \
     [ ]   [ ]
    / \   / \
   [ ][ ][ ][ ]
```
Grows all nodes at same level.

**Leaf-wise** (LightGBM):
```
        [ ]
       /   \
     [ ]   [ ]
    / \       
   [ ][ ]     
  /
[ ]
```
Grows leaf with maximum gain (deeper, more asymmetric trees).

**Advantage**: Faster convergence, lower loss
**Risk**: Can overfit (use `max_depth` to control)

### 2. GOSS (Gradient-based One-Side Sampling)

**Idea**: Keep instances with large gradients (large errors), sample from small gradients.

**Why**: Large gradient = hard to predict = more important

**Result**: Faster training with minimal accuracy loss.

### 3. EFB (Exclusive Feature Bundling)

**Idea**: Bundle mutually exclusive sparse features (many zeros).

**Example**: One-hot encoded categorical → Bundle back into single feature

**Result**: Reduces feature count, faster training.

## ⚙️ LightGBM Hyperparameters

### Tree Parameters
- `num_leaves`: Maximum leaves (default=31) ⚠️ **Key parameter (controls complexity)**
- `max_depth`: Maximum depth (default=-1, no limit)
- `min_data_in_leaf`: Minimum samples in leaf
- `min_gain_to_split`: Minimum gain to split

### Boosting Parameters
- `learning_rate`: Shrinkage (default=0.1)
- `n_estimators`: Number of trees
- `objective`: Loss function
- `metric`: Evaluation metric

### GOSS Parameters
- `boosting_type`: 'gbdt', 'dart', 'goss'
- `top_rate`: Ratio of large gradients to keep (GOSS)
- `other_rate`: Ratio of small gradients to sample (GOSS)

### Regularization
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization
- `min_split_gain`: Minimum gain (like gamma in XGBoost)

## 🧪 Python Implementation

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,         # Key parameter!
    learning_rate=0.1,
    max_depth=-1,          # No limit (controlled by num_leaves)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective='binary',
    random_state=42,
    n_jobs=-1
)

# Fit
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)]
)

# Predictions
y_pred = lgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance
lgb.plot_importance(lgb_clf, max_num_features=10)
plt.show()
```

### Native API

```python
# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'verbose': -1
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)]
)
```

## 📊 LightGBM vs XGBoost

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Tree growth** | Level-wise | Leaf-wise ✓ Faster |
| **Sampling** | Random | GOSS ✓ Smarter |
| **Feature bundling** | No | EFB ✓ Yes |
| **Speed** | Fast | ✓ Faster (large datasets) |
| **Memory** | Moderate | ✓ Lower |
| **Overfitting** | Moderate | ✓ More prone (deep leaves) |
| **Categorical** | Encode first | ✓ Native support |

---

# CatBoost

## 📘 Concept Overview

**CatBoost** (Categorical Boosting) by Yandex.

**Key innovations**:
1. **Native categorical feature handling**
2. **Ordered boosting** (reduces overfitting)
3. **Symmetric trees** (oblivious trees)

## 🧮 Key Features

### 1. Categorical Feature Handling

**Problem**: Traditional encoding (one-hot, label) can cause:
- High cardinality explosion (one-hot)
- Ordering bias (label encoding)
- Target leakage

**CatBoost solution**: **Target Statistics** with **ordered encoding**

For each category value c at instance i:
```
TS(c, i) = (Σⱼ<ᵢ yⱼ I(catⱼ=c) + prior) / (Σⱼ<ᵢ I(catⱼ=c) + prior_weight)
```

Uses only **previous instances** (ordered) to avoid leakage!

### 2. Ordered Boosting

**Problem**: Traditional boosting uses same data to fit and compute gradients → overfitting.

**CatBoost solution**: Use different random permutations for gradient computation vs fitting.

Reduces **prediction shift** (distribution difference between train and test).

### 3. Symmetric Trees (Oblivious Trees)

**Structure**: Same split condition at each level.

```
        [x1 <= t1?]
       /          \
   [x2<=t2?]   [x2<=t2?]  (Same split!)
   /  \        /    \
  L1  L2      L3    L4
```

**Advantages**:
- Faster prediction (index-based table lookup)
- Less overfitting (simpler structure)
- Better CPU cache utilization

## ⚙️ CatBoost Hyperparameters

### Tree Parameters
- `depth`: Tree depth (default=6)
- `max_leaves`: Maximum leaves (for non-symmetric mode)

### Boosting Parameters
- `iterations`: Number of trees
- `learning_rate`: Shrinkage
- `l2_leaf_reg`: L2 regularization (default=3.0)

### Categorical
- `cat_features`: Indices of categorical features
- `one_hot_max_size`: Max unique values for one-hot (default=2)

### Boosting Type
- `boosting_type`: 'Ordered' (default) or 'Plain'

## 🧪 Python Implementation

```python
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost Classifier
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    verbose=False,
    random_state=42
)

# Fit (no need to encode categorical features!)
cat_clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)

# Predictions
y_pred = cat_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance
feature_importance = cat_clf.get_feature_importance()
```

### With Categorical Features

```python
import pandas as pd

# Example data with categorical features
df = pd.DataFrame({
    'cat1': ['A', 'B', 'A', 'C', 'B'],
    'cat2': ['X', 'Y', 'X', 'Z', 'Y'],
    'num1': [1.5, 2.3, 3.1, 4.0, 2.8],
    'target': [0, 1, 0, 1, 1]
})

# Specify categorical features
cat_features = ['cat1', 'cat2']

# Create Pool (CatBoost's data structure)
train_pool = Pool(
    data=df[['cat1', 'cat2', 'num1']],
    label=df['target'],
    cat_features=cat_features
)

# Train (CatBoost handles encoding internally!)
model = CatBoostClassifier(iterations=50, verbose=False)
model.fit(train_pool)
```

## 📊 CatBoost vs XGBoost vs LightGBM

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Categorical** | Encode first | Encode first | ✓ Native (ordered TS) |
| **Tree growth** | Level-wise | Leaf-wise | ✓ Symmetric |
| **Overfitting** | Moderate | Higher | ✓ Lower (ordered boosting) |
| **Speed** | Fast | ✓ Fastest | Moderate |
| **Tuning** | Many params | Many params | ✓ Fewer (good defaults) |
| **Prediction** | Fast | Fast | ✓ Fastest (symmetric) |
| **GPU** | ✓ Yes | ✓ Yes | ✓ Yes |

---

# Model Stacking

## 📘 Concept Overview

**Stacking** combines predictions from multiple models using a **meta-model**.

## 🧮 Algorithm

```
Level 0 (Base models):
- Model 1 (e.g., Random Forest)
- Model 2 (e.g., XGBoost)
- Model 3 (e.g., Neural Network)

Level 1 (Meta-model):
- Model M (e.g., Logistic Regression)
  Inputs: Predictions from Model 1, 2, 3
  Output: Final prediction
```

### Training Process

```
1. Split train data into K folds

2. For each base model:
   a) For each fold k:
      - Train on folds ≠ k
      - Predict on fold k (out-of-fold predictions)
   b) Train on full dataset → predict test set
   
3. Create meta-features:
   - Train: Out-of-fold predictions from step 2a
   - Test: Predictions from step 2b
   
4. Train meta-model on meta-features
```

## 🧪 Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Meta-model
meta_model = LogisticRegression()

# Generate out-of-fold predictions
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for meta-features
meta_train = np.zeros((X_train.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    # Out-of-fold predictions for train
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]
        
        model.fit(X_fold_train, y_fold_train)
        meta_train[val_idx, i] = model.predict_proba(X_fold_val)[:, 1]
    
    # Retrain on full training set and predict test
    model.fit(X_train, y_train)
    meta_test[:, i] = model.predict_proba(X_test)[:, 1]

# Train meta-model
meta_model.fit(meta_train, y_train)

# Final predictions
y_pred = meta_model.predict(meta_test)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Scikit-Learn StackingClassifier

```python
from sklearn.ensemble import StackingClassifier

# Define stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

---

# 🔥 MCQs

### Q1. Decision trees use which splitting criterion for classification in sklearn?
**Options:**
- A) MSE
- B) Gini impurity ✓
- C) MAE
- D) Variance

**Explanation**: Default is Gini; can also use entropy.

---

### Q2. Random Forest reduces variance through:
**Options:**
- A) Pruning
- B) Bagging (bootstrap aggregation) ✓
- C) Boosting
- D) Regularization

**Explanation**: RF averages predictions from bootstrap samples to reduce variance.

---

### Q3. In Gradient Boosting, each new tree fits:
**Options:**
- A) Original labels
- B) Residuals (errors) of current ensemble ✓
- C) Random labels
- D) Weighted labels

**Explanation**: Sequential error correction - fit to residuals.

---

### Q4. Which can overfit with too many trees?
**Options:**
- A) Random Forest
- B) Gradient Boosting ✓
- C) Both
- D) Neither

**Explanation**: RF doesn't overfit with more trees; GB can overfit (use early stopping).

---

### Q5. XGBoost uses ______ order approximation in its objective.
**Options:**
- A) Zeroth
- B) First
- C) Second ✓
- D) Third

**Explanation**: XGBoost uses Hessian (second derivative) for Newton boosting.

---

### Q6. LightGBM grows trees:
**Options:**
- A) Level-wise
- B) Leaf-wise ✓
- C) Breadth-first
- D) Random

**Explanation**: Leaf-wise = split leaf with max gain (faster, can overfit).

---

### Q7. CatBoost handles categorical features using:
**Options:**
- A) One-hot encoding
- B) Label encoding
- C) Ordered target statistics ✓
- D) Frequency encoding

**Explanation**: Ordered TS uses only previous instances to avoid leakage.

---

### Q8. Decision trees are NOT sensitive to:
**Options:**
- A) Outliers
- B) Missing values
- C) Feature scaling ✓
- D) Class imbalance

**Explanation**: Trees split based on thresholds, not distances (scale-invariant).

---

### Q9. Random Forest `max_features` default for classification is:
**Options:**
- A) d
- B) √d ✓
- C) d/3
- D) log₂(d)

**Explanation**: Classification uses √d; regression uses d/3.

---

### Q10. Bagging trains models:
**Options:**
- A) Sequentially
- B) In parallel ✓
- C) Iteratively
- D) Recursively

**Explanation**: Bootstrap samples are independent → parallel training.

---

### Q11. Which uses symmetric (oblivious) trees?
**Options:**
- A) XGBoost
- B) LightGBM
- C) CatBoost ✓
- D) Random Forest

**Explanation**: CatBoost uses symmetric trees (same split at each level).

---

### Q12. XGBoost regularization term includes:
**Options:**
- A) Number of leaves only
- B) L2 norm of weights only
- C) Both number of leaves and L2 norm ✓
- D) Neither

**Explanation**: Ω(f) = γT + (λ/2)Σwⱼ²

---

### Q13. Gradient Boosting learning_rate controls:
**Options:**
- A) Tree depth
- B) Contribution of each tree ✓
- C) Number of trees
- D) Sample size

**Explanation**: η shrinks contribution of each tree (regularization).

---

### Q14. LightGBM GOSS keeps instances with:
**Options:**
- A) Small gradients
- B) Large gradients ✓
- C) Random selection
- D) Low weights

**Explanation**: Large gradient = large error = more important.

---

### Q15. Stacking combines models using:
**Options:**
- A) Averaging
- B) Voting
- C) Meta-model ✓
- D) Weighted sum

**Explanation**: Meta-model learns to combine base model predictions.

---

# ⚠️ Common Mistakes

1. **Not limiting tree depth**: Deep trees overfit (use max_depth, min_samples_split)

2. **Using Random Forest for linear relationships**: Overkill; linear model simpler and better

3. **Too many boosting iterations**: Can overfit (use early stopping)

4. **Not scaling for neural nets but scaling for trees**: Trees don't need scaling!

5. **Ignoring class imbalance**: Use class_weight parameter

6. **Comparing feature importance across different scalings**: Scale first for consistency

7. **Using default hyperparameters**: Always tune (especially for GB)

8. **Not using early stopping in gradient boosting**: Wastes computation, risks overfitting

9. **Encoding categorical features for CatBoost**: CatBoost handles natively!

10. **Assuming more trees always better**: True for RF, NOT for GB

---

# ⭐ One-Line Exam Facts

1. **Decision trees split** using Gini impurity (classification) or variance (regression)

2. **Gini = 0** means pure node; **Gini = 0.5** maximum impurity (binary)

3. **Random Forest = Bagging + Feature randomness** (√d features for classification)

4. **RF doesn't overfit** with more trees (more trees always helps or plateaus)

5. **Gradient Boosting** fits new tree to residuals (sequential error correction)

6. **GB can overfit** with too many iterations (use early stopping)

7. **XGBoost uses second-order** (Hessian) approximation for Newton boosting

8. **XGBoost regularization** = γ×(# leaves) + λ×Σwⱼ²

9. **LightGBM grows leaf-wise** (max gain), XGBoost grows level-wise

10. **GOSS** keeps large gradients, samples small gradients (faster training)

11. **CatBoost uses ordered target statistics** for categorical features (no encoding needed)

12. **CatBoost symmetric trees** (oblivious) = same split at each level

13. **Learning rate vs iterations** trade-off: smaller η needs larger M

14. **Bagging reduces variance**, **Boosting reduces bias**

15. **Stacking uses meta-model** to combine base model predictions

---

**End of Session 7**
