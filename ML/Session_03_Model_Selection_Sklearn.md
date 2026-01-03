# Session 3 – Model Selection & Scikit-Learn

## 📚 Table of Contents
1. [Train/Validation/Test Split](#trainvalidationtest-split)
2. [Cross-Validation](#cross-validation)
3. [Scikit-Learn Architecture](#scikit-learn-architecture)
4. [Pipelines](#pipelines)
5. [Dataset Exploration & Correlation](#dataset-exploration--correlation)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Train/Validation/Test Split

## 📘 Concept Overview

Proper data splitting is **critical** for honest model evaluation and avoiding overfitting.

### Three-Way Split

```
Full Dataset
    │
    ├─ Training Set (60-80%)
    │  └─ Learn model parameters (weights, coefficients)
    │
    ├─ Validation Set (10-20%)
    │  └─ Tune hyperparameters, select models
    │
    └─ Test Set (10-20%)
       └─ Final evaluation (USE ONCE ONLY)
```

## 🧮 Mathematical Rationale

### Why Not Just Train/Test?

**Problem**: Tuning hyperparameters on test set causes **overfitting to test set**.

**Example workflow WITHOUT validation set (WRONG)**:
```python
# WRONG: Test set used multiple times
for alpha in [0.001, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)  # ❌ Leaking test info
    if test_score > best_score:
        best_alpha = alpha
        best_score = test_score

# Final test set score is optimistic estimate!
```

**Correct workflow WITH validation set**:
```python
# CORRECT: Validation for hyperparameter tuning
for alpha in [0.001, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)  # ✓ Use validation
    if val_score > best_score:
        best_alpha = alpha

# Train final model on train+validation
final_model = Ridge(alpha=best_alpha)
final_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

# Test ONCE to get unbiased estimate
test_score = final_model.score(X_test, y_test)  # ✓ Unbiased
```

## ⚙️ Practical Implementation

### Basic Split

```python
from sklearn.model_selection import train_test_split

# First split: Separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: Separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Train: {len(X_train)/len(X)*100:.0f}%")  # 60%
print(f"Val: {len(X_val)/len(X)*100:.0f}%")      # 20%
print(f"Test: {len(X_test)/len(X)*100:.0f}%")    # 20%
```

### Stratified Split (Classification)

**Preserves class distribution** in each split (critical for imbalanced data).

```python
# With stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Maintain class proportions
    random_state=42
)

# Check class distribution
print("Train distribution:", np.bincount(y_train) / len(y_train))
print("Test distribution:", np.bincount(y_test) / len(y_test))
```

### Time Series Split ⚠️

**NEVER shuffle temporal data** — use chronological splits.

```python
# WRONG for time series
X_train, X_test = train_test_split(X_time_series, shuffle=True)  # ❌

# CORRECT for time series
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]   # Earlier data
X_test = X[split_idx:]    # Future data ✓
```

## 📊 Split Size Recommendations

| Dataset Size | Train | Validation | Test | Notes |
|--------------|-------|------------|------|-------|
| **< 1,000** | 60% | 20% | 20% | Use k-fold CV instead |
| **1,000-10,000** | 70% | 15% | 15% | Standard split |
| **10,000-100,000** | 80% | 10% | 10% | More data for training |
| **> 100,000** | 90% | 5% | 5% | Validation/test can be smaller |
| **Time Series** | 70% | 15% | 15% | Chronological, no shuffle |

## ⚠️ Common Pitfalls

### 1. Data Leakage via Preprocessing

**WRONG**: Fit scaler on entire dataset
```python
# Data leakage! ❌
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
# Test statistics leaked into training via scaler
```

**CORRECT**: Fit scaler only on training data
```python
X_train, X_test = train_test_split(X)

# Fit scaler ONLY on training data
scaler.fit(X_train)

# Transform both sets using training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training mean/std
```

### 2. Test Set Contamination

**WRONG**: Using test set for any decisions
- Feature selection based on test set correlations
- Early stopping based on test set loss
- Multiple model evaluations on test set

**CORRECT**: Never touch test set until final evaluation

### 3. Ignoring Class Imbalance

Without stratification on imbalanced data (1% positive class):
- Training might get 2% positive (lucky)
- Test might get 0% positive (unlucky)
- Invalid comparison!

**Solution**: Always use `stratify=y` for classification.

---

# Cross-Validation

## 📘 Concept Overview

**Cross-Validation (CV)** uses multiple train/validation splits to get **robust performance estimate** and **better use of data**.

## 🧮 k-Fold Cross-Validation

### Algorithm

1. Split data into k equal folds
2. For i = 1 to k:
   - Use fold i as validation
   - Use remaining k-1 folds as training
   - Train model and evaluate on validation fold
3. Average k validation scores

```
Fold 1: [Val] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Val] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Val] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Val] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Val]

Average score = (Score₁ + Score₂ + Score₃ + Score₄ + Score₅) / 5
```

### Mathematical Justification

**Variance of single validation score**: σ²
**Variance of k-fold CV average**: σ²/k

More folds → Lower variance in estimate → More reliable

### Implementation

```python
from sklearn.model_selection import cross_val_score, KFold

# Basic k-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Manual k-fold for more control
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold+1}: {score:.3f}")
```

## 🔄 CV Variants

### 1. Stratified k-Fold

Maintains class distribution in each fold (essential for classification).

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### 2. Leave-One-Out CV (LOOCV)

**Special case**: k = n (number of samples)

- Train on n-1 samples, validate on 1 sample
- Repeat n times

**Advantages**:
- Maximum use of data
- Deterministic (no randomness)

**Disadvantages**:
- Extremely computationally expensive (n model fits)
- High variance in estimates
- Not recommended for large datasets

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
print(f"LOOCV score: {scores.mean():.3f}")  # Very expensive!
```

### 3. Leave-P-Out

Generalization of LOOCV (leave out p samples each time).

**Warning**: Combinatorially expensive — rarely used.

### 4. Repeated k-Fold

Repeat k-fold CV multiple times with different random splits.

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)

print(f"Repeated k-fold (50 fits): {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### 5. Time Series CV

**Forward chaining**: Validation set always after training set.

```
Fold 1: [Train] [Val] - - -
Fold 2: [Train] [Train] [Val] - -
Fold 3: [Train] [Train] [Train] [Val] -
Fold 4: [Train] [Train] [Train] [Train] [Val]
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # train_idx is always before val_idx (no data leakage)
```

### 6. Group k-Fold

Ensures samples from same group **never split** across train/validation.

**Use case**: Medical data (multiple samples per patient)

```python
from sklearn.model_selection import GroupKFold

# patients = [1, 1, 1, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)

for train_idx, val_idx in gkf.split(X, y, groups=patients):
    # Patient groups never split across train/val
    pass
```

## 📊 CV for Hyperparameter Tuning

### GridSearchCV

Exhaustive search over hyperparameter grid.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,  # Parallel
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Evaluate on test set (ONCE at the end)
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")
```

### RandomizedSearchCV

Sample hyperparameters randomly (faster for large search spaces).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Try 50 random combinations
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

## ⚠️ Nested Cross-Validation

**Problem**: Using CV for both model selection AND performance estimation leads to optimistic bias.

**Solution**: Nested CV (outer loop for evaluation, inner loop for hyperparameter tuning)

```python
from sklearn.model_selection import cross_val_score

# Outer CV: Performance estimation
outer_scores = []

for train_idx, test_idx in KFold(n_splits=5).split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV: Hyperparameter tuning (on training set only)
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model on outer test set
    score = grid_search.score(X_test_outer, y_test_outer)
    outer_scores.append(score)

print(f"Nested CV score: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores)*2:.3f})")
```

---

# Scikit-Learn Architecture

## 📘 Design Principles

Scikit-Learn follows **consistent API design**:

1. **Estimator**: Any object that learns from data (`fit` method)
2. **Predictor**: Estimator that can make predictions (`predict` method)
3. **Transformer**: Estimator that can transform data (`transform` method)
4. **Uniform interface**: All estimators have same method names

## 🧮 Key Interfaces

### 1. Estimators

All learning algorithms implement `fit(X, y)`:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)  # Learn from training data
```

**Learned parameters** end with underscore `_`:
```python
print(model.coef_)       # Learned weights
print(model.intercept_)  # Learned bias
```

### 2. Predictors

Supervised learning algorithms implement `predict(X)`:

```python
y_pred = model.predict(X_test)  # Classification: Class labels

# For probabilistic predictions
y_proba = model.predict_proba(X_test)  # Class probabilities (if available)
```

### 3. Transformers

Data preprocessing implements `fit`, `transform`, and `fit_transform`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data (compute mean, std)
scaler.fit(X_train)

# Transform training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Shortcut: fit_transform (fit + transform in one step, training only)
X_train_scaled = scaler.fit_transform(X_train)
```

### 4. Scoring

All supervised estimators implement `score(X, y)`:

```python
# Classification: accuracy by default
accuracy = model.score(X_test, y_test)

# Regression: R² by default
r2 = reg_model.score(X_test, y_test)
```

## 📊 Estimator Hierarchy

```
BaseEstimator (base class)
    │
    ├─ ClassifierMixin
    │  ├─ LogisticRegression
    │  ├─ SVC
    │  ├─ RandomForestClassifier
    │  └─ GradientBoostingClassifier
    │
    ├─ RegressorMixin
    │  ├─ LinearRegression
    │  ├─ Ridge
    │  ├─ SVR
    │  └─ RandomForestRegressor
    │
    ├─ TransformerMixin
    │  ├─ StandardScaler
    │  ├─ PCA
    │  ├─ LabelEncoder
    │  └─ OneHotEncoder
    │
    └─ ClusterMixin
       ├─ KMeans
       ├─ DBSCAN
       └─ AgglomerativeClustering
```

## ⚙️ Hyperparameters

Set during initialization (before `fit`):

```python
# Hyperparameters specified at creation
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    random_state=42      # Reproducibility
)

# Access hyperparameters
print(model.get_params())

# Modify hyperparameters
model.set_params(n_estimators=200,max_depth=15)
```

## 🧪 Model Persistence

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')

# Make predictions with loaded model
predictions = loaded_model.predict(X_new)
```

---

# Pipelines

## 📘 Concept Overview

**Pipelines** chain preprocessing and modeling steps into single object.

**Benefits**:
1. **Prevent data leakage**: Transformers fit only on training data
2. **Code simplicity**: One `fit`, one `predict`
3. **Hyperparameter tuning**: Tune pipeline as single entity
4. **Reproducibility**: Entire workflow encapsulated

## 🧮 Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Scale features
    ('pca', PCA(n_components=10)),       # Step 2: Dimensionality reduction
    ('classifier', LogisticRegression()) # Step 3: Classification
])

# Fit entire pipeline (each step applied sequentially on training data)
pipeline.fit(X_train, y_train)

# Predict (applies all transformations + prediction)
y_pred = pipeline.predict(X_test)

# Score
accuracy = pipeline.score(X_test, y_test)
```

### Pipeline Execution Flow

**Training**: 
```
X_train → scaler.fit_transform() → PCA.fit_transform() → classifier.fit()
```

**Prediction**:
```
X_test → scaler.transform() → PCA.transform() → classifier.predict()
```

## ⚙️ Advanced Pipeline Features

### 1. Access Pipeline Steps

```python
# Access intermediate steps
scaler = pipeline.named_steps['scaler']
print(f"Training mean: {scaler.mean_}")

# Access final estimator
classifier = pipeline.named_steps['classifier']
print(f"Coefficients: {classifier.coef_}")
```

### 2. Pipeline with GridSearchCV

```python
param_grid = {
    'pca__n_components': [5, 10, 20],       # PCA hyperparameters
    'classifier__C': [0.1, 1, 10],          # Classifier hyperparameters
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

### 3. ColumnTransformer (Different Transformations for Different Features)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Different preprocessing for numeric and categorical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [0, 1, 2]),           # Numeric columns
    ('cat', OneHotEncoder(), [3, 4])                # Categorical columns
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### 4. FeatureUnion (Combine Multiple Transformations)

```python
from sklearn.pipeline import FeatureUnion

# Apply multiple transformations and concatenate results
feature_union = FeatureUnion([
    ('pca', PCA(n_components=10)),
    ('poly', PolynomialFeatures(degree=2))
])

pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', LogisticRegression())
])
```

## 🧪 Complete Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Hyperparameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

# Grid search on pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Results
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")
```

---

# Dataset Exploration & Correlation

## 📘 Initial Data Exploration

### 1. Data Overview

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# Basic information
print(df.info())         # Column types, null counts
print(df.describe())     # Statistical summary
print(df.head())         # First 5 rows
print(df.shape)          # (rows, columns)

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(f"Duplicates: {df.duplicated().sum()}")
```

### 2. Target Variable Analysis

```python
# Classification: Class distribution
print(df['target'].value_counts())

# Check for class imbalance
class_counts = df['target'].value_counts()
print(f"Imbalance ratio: {class_counts.max() / class_counts.min():.2f}")

# Regression: Target distribution
df['target'].hist(bins=50)
plt.title('Target Distribution')
plt.show()
```

### 3. Feature Types

```python
# Identify numeric and categorical features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric: {len(numeric_features)}")
print(f"Categorical: {len(categorical_features)}")
```

## 🧮 Correlation Analysis

### 1. Pearson Correlation

Measures **linear** relationship between features.

```
r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² Σ(yᵢ - ȳ)²]
```

**Range**: -1 (perfect negative) to +1 (perfect positive)

```python
# Compute correlation matrix
corr_matrix = df.corr()

# Correlation with target
target_corr = corr_matrix['target'].sort_values(ascending=False)
print(target_corr)

# Visualize correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

### 2. Identifying Multicollinearity

**Multicollinearity**: High correlation between features (problematic for some models).

```python
# Find highly correlated feature pairs
high_corr = np.where(np.abs(corr_matrix) > 0.8)
high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                   for x, y in zip(*high_corr) if x != y and x < y]

for feat1, feat2, corr in high_corr_pairs:
    print(f"{feat1} - {feat2}: {corr:.3f}")
```

### 3. VIF (Variance Inflation Factor)

Quantifies multicollinearity severity.

```
VIF_i = 1 / (1 - R²_i)
```

where R²_i = R² when regressing feature i on all other features.

**Rule of thumb**:
- VIF < 5: Low multicollinearity
- 5 ≤ VIF < 10: Moderate
- VIF ≥ 10: High (consider removing feature)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features
vif_data["VIF"] = [variance_inflation_factor(df[numeric_features].values, i)
                   for i in range(len(numeric_features))]

print(vif_data.sort_values('VIF', ascending=False))
```

### 4. Feature-Target Relationship

```python
# Scatter plots for each feature vs target (regression)
for col in numeric_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col], df['target'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('Target')
    plt.title(f'{col} vs Target')
    plt.show()

# Box plots for categorical features (classification)
for col in categorical_features:
    df.boxplot(column='target', by=col)
    plt.title(f'{col} vs Target')
    plt.show()
```

## 📊 Advanced Exploration

### 1. Outlier Detection

```python
# Z-score method
from scipy.stats import zscore

z_scores = np.abs(zscore(df[numeric_features]))
outliers = (z_scores > 3).any(axis=1)
print(f"Outliers: {outliers.sum()}")

# IQR method
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((df[numeric_features] < (Q1 - 1.5 * IQR)) | 
                (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
```

### 2. Feature Distributions

```python
# Check for normality (many algorithms assume Gaussian features)
for col in numeric_features:
    df[col].hist(bins=50, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.show()
    
    # Skewness
    skew = df[col].skew()
    print(f"{col} skewness: {skew:.2f}")
    if abs(skew) > 1:
        print(f"  → Consider log transformation")
```

### 3. Pairplot

```python
# Visualize all pairwise relationships (expensive for many features)
sns.pairplot(df[[*numeric_features[:5], 'target']], hue='target')
plt.show()
```

---

# 🔥 MCQs

### Q1. In train/validation/test split, which set is used for hyperparameter tuning?
**Options:**
- A) Training set
- B) Validation set ✓
- C) Test set
- D) All three

**Explanation**: Validation set is for model selection and hyperparameter tuning. Test set is for final evaluation only.

---

### Q2. What is data leakage in the context of scaling?
**Options:**
- A) Losing data during preprocessing
- B) Test statistics influencing training ✓
- C) Training set being too large
- D) Missing values in dataset

**Explanation**: Fitting scaler on entire dataset before split leaks test statistics (mean, std) into training.

---

### Q3. In 5-fold cross-validation, what percentage of data is used for training in each fold?
**Options:**
- A) 20%
- B) 50%
- C) 80% ✓
- D) 100%

**Explanation**: Each fold uses 4/5 = 80% for training, 1/5 = 20% for validation.

---

### Q4. Which CV method is appropriate for time series data?
**Options:**
- A) Standard k-fold
- B) Stratified k-fold
- C) TimeSeriesSplit ✓
- D) LOOCV

**Explanation**: TimeSeriesSplit ensures validation set is always after training set (respects temporal order).

---

### Q5. What does high multicollinearity (VIF > 10) indicate?
**Options:**
- A) Features are uncorrelated
- B) Features are highly correlated ✓
- C) Model is overfitting
- D) Data has outliers

**Explanation**: High VIF means feature can be predicted well by other features (high correlation).

---

### Q6. Which method provides the most unbiased performance estimate?
**Options:**
- A) Training accuracy
- B) Validation accuracy
- C) Hold-out test set (used once) ✓
- D) Test set used multiple times

**Explanation**: Test set used once provides unbiased estimate. Multiple use leads to overfitting to test set.

---

### Q7. In scikit-learn, which method is used to save learned parameters?
**Options:**
- A) fit()
- B) transform()
- C) predict()
- D) None, parameters end with underscore ✓

**Explanation**: Learned parameters (e.g., `coef_`, `mean_`) have trailing underscore, set by `fit()`.

---

### Q8. What is the advantage of using Pipelines?
**Options:**
- A) Faster training
- B) Prevents data leakage ✓
- C) Improves model accuracy
- D) Reduces memory usage

**Explanation**: Pipelines ensure transformers fit only on training data, preventing test information leakage.

---

### Q9. Pearson correlation = 0 means:
**Options:**
- A) Features are independent
- B) No linear relationship ✓
- C) Strong relationship
- D) Perfect correlation

**Explanation**: Zero correlation means no **linear** relationship, but non-linear relationships may exist.

---

### Q10. GridSearchCV with 5-fold CV and 100 hyperparameter combinations trains how many models?
**Options:**
- A) 5
- B) 100
- C) 105
- D) 500 ✓

**Explanation**: 100 combinations × 5 folds = 500 model fits.

---

### Q11. Stratified split is important for:
**Options:**
- A) Regression problems
- B) Imbalanced classification ✓
- C) Time series
- D) Clustering

**Explanation**: Stratification maintains class distribution, critical for imbalanced datasets.

---

### Q12. LOOCV (Leave-One-Out CV) with n=1000 requires:
**Options:**
- A) 1 model fit
- B) 10 model fits
- C) 100 model fits
- D) 1000 model fits ✓

**Explanation**: LOOCV trains n models (one for each sample left out).

---

### Q13. In sklearn, `fit_transform()` should be used on:
**Options:**
- A) Both training and test sets
- B) Test set only
- C) Training set only ✓
- D) Validation set only

**Explanation**: `fit_transform()` fits and transforms. Only use on training data. Use `transform()` on test/validation.

---

### Q14. Nested cross-validation is used for:
**Options:**
- A) Faster training
- B) Unbiased performance estimation with hyperparameter tuning ✓
- C) Feature selection
- D) Handling missing values

**Explanation**: Outer loop evaluates performance, inner loop tunes hyperparameters (avoids optimistic bias).

---

### Q15. Which correlation coefficient captures non-linear relationships?
**Options:**
- A) Pearson
- B) Spearman ✓
- C) Both
- D) Neither

**Explanation**: Spearman (rank correlation) can detect monotonic non-linear relationships. Pearson only captures linear.

---

# ⚠️ Common Mistakes

1. **Using test set for hyperparameter tuning**: Leads to overestimated performance

2. **Fitting preprocessors on entire dataset before split**: Data leakage

3. **Not stratifying classification datasets**: Unbalanced splits, invalid comparisons

4. **Shuffling time series data**: Violates temporal independence assumption

5. **Using `fit_transform()` on test set**: Should use `transform()` only

6. **Ignoring multicollinearity**: Problematic for linear models, inflates coefficient variance

7. **Assuming correlation = causation**: Correlation doesn't imply causal relationship

8. **Using wrong CV method**: e.g., standard k-fold for time series

9. **Not checking class distribution after split**: Especially with small datasets

10. **Pipeline without proper naming**: Makes hyperparameter tuning difficult

---

# ⭐ One-Line Exam Facts

1. **Three-way split**: Train (learn params), Validation (tune hyperparams), Test (final eval, ONCE)

2. **Data leakage**: Fit preprocessors ONLY on training data, never on test data

3. **Stratified split maintains class proportions** (critical for imbalanced classification)

4. **Time series: NO shuffle**, use chronological split

5. **k-fold CV trains k models**, averaging reduces variance in performance estimate

6. **LOOCV has low bias but high variance**, computationally expensive

7. **GridSearchCV with k-fold and m hyperparameters** = k × m model fits

8. **Nested CV**: Outer loop (evaluation), Inner loop (hyperparameter tuning)

9. **Sklearn API**: `fit()` learns, `transform()` applies, `predict()` outputs

10. **Pipeline prevents data leakage** by ensuring fit on training data only

11. **Pearson correlation measures linear relationships** only

12. **VIF ≥ 10 indicates high multicollinearity** (consider removing feature)

13. **`fit_transform()` on training**, `transform()` on test

14. **Test set contamination** = using test set for ANY decisions before final evaluation

15. **ColumnTransformer applies different transformations** to different feature subsets

---

**End of Session 3**
