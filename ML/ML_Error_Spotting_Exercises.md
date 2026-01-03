# ML Error Spotting Exercises (20 Problems)

## 📘 Instructions
Each code snippet contains **one or more errors**. Identify ALL errors and provide corrections.

**Common error types**:
- Data leakage
- Incorrect API usage
- Logical errors
- Common ML pitfalls

---

## Problem 1: Data Leakage in Scaling 🐛

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ ERROR HERE

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

**Error**: Scaling BEFORE splitting causes data leakage (test statistics leak into training)

**Correction**:
```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Then scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)        # Transform test using train statistics
```

---

## Problem 2: Using Test Set for Hyperparameter Tuning 🐛

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_score = 0
best_C = None

for C in [0.1, 1, 10, 100]:
    model = SVC(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ ERROR: Using test set for tuning
    if score > best_score:
        best_score = score
        best_C = C

# Final model
final_model = SVC(C=best_C)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)  # ❌ Optimistic estimate!
```

**Error**: Using test set multiple times for hyperparameter selection causes overfitting to test set

**Correction**:
```python
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use cross-validation for tuning
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Test ONCE at the end
test_score = grid_search.score(X_test, y_test)  # ✓ Unbiased estimate
```

---

## Problem 3: Fitting Scaler on Test Data 🐛

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # ❌ ERROR
```

**Error**: `fit_transform` on test data computes new statistics from test set (data leakage)

**Correction**:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)         # Transform only (using train stats)
```

---

## Problem 4: Wrong Metric for Imbalanced Data 🐛

```python
from sklearn.metrics import accuracy_score

# Dataset: 95% class 0, 5% class 1
y_true = [0]*95 + [1]*5
y_pred = [0]*100  # Model predicts all class 0

accuracy = accuracy_score(y_true, y_pred)  # ❌ Shows 95% accuracy!
print(f"Model accuracy: {accuracy}")  # Misleading!
```

**Error**: Accuracy is misleading for imbalanced data

**Correction**:
```python
from sklearn.metrics import classification_report, f1_score, confusion_matrix

print(classification_report(y_true, y_pred))
print(f"F1-score: {f1_score(y_true, y_pred)}")  # ✓ Better metric
print(confusion_matrix(y_true, y_pred))
```

---

## Problem 5: Incorrect train_test_split Usage 🐛

```python
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)  # ❌ Wrong order
```

**Error**: Incorrect unpacking order

**Correction**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # ✓ Correct
```

---

## Problem 6: Not Stratifying Imbalanced Classification 🐛

```python
from sklearn.model_selection import train_test_split

# Highly imbalanced: 90% class 0, 10% class 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # ❌ No stratification
```

**Error**: Random split might create unrepresentative train/test distributions

**Correction**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # ✓ Maintains class proportions
)
```

---

## Problem 7: Using LabelEncoder on Features 🐛

```python
from sklearn.preprocessing import LabelEncoder

# Categorical feature: ['red', 'blue', 'green']
le = LabelEncoder()
X_encoded = le.fit_transform(X_categorical)  # ❌ Creates ordinal relationship
```

**Error**: LabelEncoder creates ordinal encoding (red=0, blue=1, green=2) implying order

**Correction**:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical.reshape(-1, 1))  # ✓ No ordinal assumption
```

---

## Problem 8: Forgotten Feature Scaling for k-NN 🐛

```python
from sklearn.neighbors import KNeighborsClassifier

# Features have different scales: age(0-100), salary(20000-200000)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # ❌ No scaling!
```

**Error**: k-NN uses distances; unscaled features make salary dominate

**Correction**:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
pipeline.fit(X_train, y_train)  # ✓ Automatic scaling
```

---

## Problem 9: Overfitting Decision Tree 🐛

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()  # ❌ No constraints
clf.fit(X_train, y_train)
# Likely 100% train accuracy, poor test accuracy
```

**Error**: Unconstrained tree will overfit (grow until pure leaves)

**Correction**:
```python
clf = DecisionTreeClassifier(
    max_depth=10,           # Limit depth
    min_samples_split=20,   # Minimum samples to split
    min_samples_leaf=10     # Minimum samples in leaf
)
clf.fit(X_train, y_train)  # ✓ Regularized
```

---

## Problem 10: Shuffling Time Series Data 🐛

```python
from sklearn.model_selection import train_test_split

# Time series data
X_train, X_test, y_train, y_test = train_test_split(
    X_timeseries, y_timeseries, test_size=0.2, shuffle=True  # ❌ Shuffling time series!
)
```

**Error**: Shuffling breaks temporal order → data leakage (future predicts past)

**Correction**:
```python
# Chronological split
split_idx = int(0.8 * len(X_timeseries))
X_train = X_timeseries[:split_idx]   # Earlier data
X_test = X_timeseries[split_idx:]    # Later data (future)
y_train = y_timeseries[:split_idx]
y_test = y_timeseries[split_idx:]

# OR use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X_timeseries):
    X_train, X_test = X_timeseries[train_idx], X_timeseries[test_idx]
```

---

## Problem 11: Incorrect Random Forest Parameters 🐛

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=100)  # ❌ Individual tree max_depth
rf.fit(X_train, y_train)
```

**Error**: Not an error per se, but likely ineffective regularization

**Better practices**:
```python
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # ✓ Reasonable depth
    max_features='sqrt',   # ✓ Feature randomness
    min_samples_leaf=5,    # ✓ Prevent overfitting
    n_jobs=-1              # ✓ Parallel processing
)
```

---

## Problem 12: Forgetting to Set random_state 🐛

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # ❌ Non-reproducible
```

**Error**: Different splits each run → non-reproducible results

**Correction**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # ✓ Reproducible
)
```

---

## Problem 13: Using .fit() on Test Data 🐛

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)  # ❌ Re-fitting on test!
```

**Error**: `fit_transform` re-fits PCA on test data (data leakage)

**Correction**:
```python
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)  # Fit on train
X_test_pca = pca.transform(X_test)        # Transform only
```

---

## Problem 14: Missing Class in Train Set 🐛

```python
from sklearn.preprocessing import LabelEncoder

y_train = ['cat', 'dog', 'cat']
y_test = ['cat', 'dog', 'bird']  # 'bird' not in training

le = LabelEncoder()
le.fit(y_train)
y_test_encoded = le.transform(y_test)  # ❌ Error: 'bird' unseen
```

**Error**: LabelEncoder doesn't handle unseen labels

**Solutions**:
```python
# Option 1: Ensure all classes in train
# Option 2: Handle unknown labels
try:
    y_test_encoded = le.transform(y_test)
except ValueError:
    # Handle unseen labels
    y_test_encoded = [le.transform([yt])[0] if yt in le.classes_ else -1 for yt in y_test]
```

---

## Problem 15: Incorrect Cross-Validation for Model Comparison 🐛

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt_scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
rf_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)  # ❌ Different splits!

print(f"DT: {dt_scores.mean()}, RF: {rf_scores.mean()}")  # Unfair comparison
```

**Error**: Different CV splits make comparison unfair (randomness affects results)

**Correction**:
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

dt_scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=kf)
rf_scores = cross_val_score(RandomForestClassifier(), X, y, cv=kf)  # ✓ Same splits

print(f"DT: {dt_scores.mean()}, RF: {rf_scores.mean()}")
```

---

## Problem 16: Incorrect Pipeline Usage 🐛

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Train
pipeline.fit(X_train, y_train)

# Test - WRONG WAY
X_test_scaled = pipeline['scaler'].transform(X_test)  # ❌ Manual transform
predictions = pipeline['svm'].predict(X_test_scaled)
```

**Error**: Manually accessing pipeline steps defeats the purpose

**Correction**:
```python
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)  # ✓ Pipeline handles everything
```

---

## Problem 17: Forgetting to Handle Missing Values 🐛

```python
from sklearn.linear_model import LogisticRegression

# X has NaN values
model = LogisticRegression()
model.fit(X_train, y_train)  # ❌ Error: NaN values
```

**Error**: Most sklearn models don't handle NaN

**Correction**:
```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle NaN
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)  # ✓ Works
```

---

## Problem 18: Wrong Shapes for Model Input 🐛

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1, 2, 3, 4, 5])  # ❌ 1D array
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)  # ❌ Error: X must be 2D
```

**Error**: sklearn expects 2D array for X (even single feature)

**Correction**:
```python
X = np.array([[1], [2], [3], [4], [5]])  # ✓ 2D: (n_samples, n_features)
# OR
X = X.reshape(-1, 1)
model.fit(X, y)  # ✓ Works
```

---

## Problem 19: Using Training Accuracy to Evaluate 🐛

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_train, y_train)  # ❌ Training accuracy!
print(f"Model performance: {accuracy}")  # Misleading
```

**Error**: Training accuracy is optimistic (model has seen this data)

**Correction**:
```python
accuracy = model.score(X_test, y_test)  # ✓ Test accuracy
# Better: use cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV accuracy: {cv_scores.mean()}")
```

---

## Problem 20: Incorrect MultiOutputClassifier Usage 🐛

```python
from sklearn.tree import DecisionTreeClassifier

# Multi-label classification: y has shape (n_samples, n_labels)
y_train = [[1, 0], [0, 1], [1, 1]]  
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # ❌ DecisionTree doesn't support multi-output natively
```

**Error**: Standard DecisionTree for multi-output needs wrapper

**Correction**:
```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

model = MultiOutputClassifier(DecisionTreeClassifier())
model.fit(X_train, y_train)  # ✓ Works for multi-output
```

---

# Common Error Patterns Summary

## Data Leakage Errors
1. Scaling before splitting
2. Fitting transformers on test data
3. Using test set for hyperparameter tuning
4. Shuffling time series data

## API Usage Errors
5. Wrong unpacking order (`train_test_split`)
6. Using `fit_transform` on test (should be `transform`)
7. LabelEncoder on features (should be OneHotEncoder)
8. Wrong array shapes (1D vs 2D)

## ML Best Practice Violations
9. No feature scaling for distance-based models
10. No stratification for imbalanced data
11. Evaluating on training set
12. No regularization (overfitting)

## Reproducibility Errors
13. Missing `random_state`
14. Different CV splits for model comparison

## Missing Preprocessing
15. Not handling missing values
16. Not handling unseen categories

---

**End of Error Spotting Exercises**
