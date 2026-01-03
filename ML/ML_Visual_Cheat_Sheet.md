# Machine Learning - Visual Cheat Sheet

**One-Page Quick Reference for PG-DBDA ML Exam**

---

## 🎯 Algorithm Selection Flowchart

```
START: What is your task?
│
├─ SUPERVISED LEARNING (Have labeled data)
│  │
│  ├─ REGRESSION (Predict continuous value)
│  │  ├─ Linear relationships → Linear Regression
│  │  ├─ Non-linear → Polynomial Regression
│  │  ├─ Prevent overfitting → Ridge/Lasso
│  │  ├─ Many features → ElasticNet
│  │  ├─ Non-linear complex → Decision Tree / Random Forest / Gradient Boosting
│  │  └─ Sequential data → LSTM / GRU
│  │
│  └─ CLASSIFICATION (Predict category)
│     ├─ Linear separable → Logistic Regression / SVM (linear)
│     ├─ Non-linear → SVM (RBF kernel)
│     ├─ Probabilistic → Naive Bayes
│     ├─ Interpretable → Decision Tree
│     ├─ High accuracy → Random Forest / Gradient Boosting / XGBoost
│     ├─ Distance-based → k-NN (but slow!)
│     ├─ Text data → Naive Bayes / Logistic Regression / BERT
│     ├─ Image data → CNN / ResNet / Vision Transformer
│     └─ Sequential → RNN / LSTM / Transformer
│
├─ UNSUPERVISED LEARNING (No labels)
│  │
│  ├─ CLUSTERING (Group similar data)
│  │  ├─ Know # clusters → k-Means
│  │  ├─ Unknown # clusters → DBSCAN / Hierarchical
│  │  ├─ Soft clusters → Gaussian Mixture Models
│  │  └─ High-dimensional → Spectral Clustering
│  │
│  ├─ DIMENSIONALITY REDUCTION
│  │  ├─ Linear → PCA
│  │  ├─ Non-linear → t-SNE / UMAP
│  │  ├─ Preserve variance → PCA
│  │  ├─ Visualization → t-SNE (2D/3D)
│  │  └─ With labels available → LDA
│  │
│  └─ ANOMALY DETECTION
│     ├─ Isolation Forest
│     ├─ One-Class SVM
│     └─ Autoencoder
│
└─ REINFORCEMENT LEARNING (Learn from environment)
   ├─ Q-Learning (discrete actions)
   ├─ DQN (deep Q-network)
   ├─ Policy Gradient (continuous actions)
   └─ Actor-Critic / PPO
```

---

## 📊 Model Comparison Matrix

| Model | Type | Pros | Cons | When to Use |
|-------|------|------|------|-------------|
| **Linear Regression** | Regression | Fast, interpretable | Assumes linearity | Linear relationships |
| **Logistic Regression** | Classification | Fast, probabilistic | Linear decision boundary | Baseline classifier |
| **Decision Tree** | Both | Interpretable, no scaling | Overfits, unstable | Need interpretability |
| **Random Forest** | Both | High accuracy, robust | Slow, black-box | General purpose |
| **Gradient Boosting** | Both | Best accuracy | Very slow, overfits | Competitions, critical tasks |
| **SVM** | Both | Works in high-dim | Slow, needs scaling | Small-medium datasets |
| **k-NN** | Both | Simple, non-parametric | Slow prediction, needs scaling | Small datasets |
| **Naive Bayes** | Classification | Fast, works with small data | Strong independence assumption | Text classification |
| **Neural Network** | Both | Learns complex patterns | Needs lots of data, slow | Large datasets, images, text |
| **k-Means** | Clustering | Fast, simple | Needs k, sensitive to init | Spherical clusters |
| **DBSCAN** | Clustering | Finds any shape, detects outliers | Sensitive to params | Arbitrary-shaped clusters |
| **PCA** | Dim Reduction | Fast, interpretable | Linear only | Preprocessing, visualization |

---

## 🔄 Train/Val/Test Split Strategy

```
Full Dataset (100%)
│
├─ Train Set (60-80%) ────────► FIT models, transformers
│
├─ Validation Set (10-20%) ───► TUNE hyperparameters, SELECT models
│
└─ Test Set (10-20%) ──────────► EVALUATE final model (USE ONCE!)
```

**Golden Rules**:
1. **Split BEFORE any preprocessing**
2. **Fit transformers on train only**
3. **Never touch test set until final evaluation**
4. **Stratify for classification** (`stratify=y`)
5. **NO shuffling for time series!**

---

## 📈 Evaluation Metrics Quick Reference

### Classification

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| **Accuracy** | (TP+TN) / Total | Balanced classes | [0, 1] |
| **Precision** | TP / (TP+FP) | Minimize false positives | [0, 1] |
| **Recall** | TP / (TP+FN) | Minimize false negatives | [0, 1] |
| **F1-Score** | 2·(P·R)/(P+R) | Imbalanced data | [0, 1] |
| **AUC-ROC** | Area under ROC curve | Overall performance | [0, 1] |

**Confusion Matrix**:
```
                Predicted
              Neg       Pos
Actual  Neg   TN        FP
        Pos   FN        TP
```

### Regression

| Metric | Formula | Interpretation | Range |
|--------|---------|----------------|-------|
| **MAE** | Mean(\|y-ŷ\|) | Average error | [0, ∞) |
| **MSE** | Mean((y-ŷ)²) | Penalizes large errors | [0, ∞) |
| **RMSE** | √MSE | Same unit as target | [0, ∞) |
| **R²** | 1 - SS_res/SS_tot | Variance explained | (-∞, 1] |

---

## ⚙️ Hyperparameter Tuning Guide

### Key Hyperparameters by Model

**Decision Tree**:
- `max_depth`: 3-20 (prevent overfitting)
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10

**Random Forest**:
- `n_estimators`: 100-1000 (more = better, slower)
- `max_depth`: 10-50
- `max_features`: 'sqrt' or 'log2'

**Gradient Boosting**:
- `n_estimators`: 100-1000
- `learning_rate`: 0.01-0.3 (lower = better but slower)
- `max_depth`: 3-10 (shallow trees!)

**SVM**:
- `C`: 0.1-100 (regularization, lower = more)
- `kernel`: 'linear', 'rbf', 'poly'
- `gamma`: 0.001-1 (for RBF)

**Neural Network**:
- `learning_rate`: 0.0001-0.1
- `batch_size`: 16-256
- `epochs`: 10-1000 (with early stopping)
- `dropout`: 0.2-0.5

---

## 🛠️ Preprocessing Checklist

### Numerical Features
- [ ] **Handle missing values**: SimpleImputer (mean/median/mode)
- [ ] **Scale features**: StandardScaler or MinMaxScaler
- [ ] **Remove outliers**: IQR method or Isolation Forest
- [ ] **Create polynomial features**: PolynomialFeatures

### Categorical Features
- [ ] **Encode target**: LabelEncoder
- [ ] **Encode features**: OneHotEncoder (nominal) or OrdinalEncoder (ordinal)
- [ ] **Handle high cardinality**: Target encoding or frequency encoding

### Text Features
- [ ] **Tokenization**: CountVectorizer or TfidfVectorizer
- [ ] **Remove stop words**: English stopwords
- [ ] **Stemming/Lemmatization**: NLTK or spaCy
- [ ] **Embeddings**: Word2Vec, GloVe, BERT

### Feature Engineering
- [ ] **Domain-specific features**: Based on business logic
- [ ] **Interaction terms**: Feature1 × Feature2
- [ ] **Binning**: Convert continuous to categorical
- [ ] **Date features**: Extract year, month, day, weekday

---

## 🔍 Bias-Variance Tradeoff

```
    Error
      ↑
      │     ╱Total Error
      │    ╱
      │   ╱╲ 
      │  ╱  ╲
      │ ╱    ╲___Variance
      │╱      
      ├────────────► Model Complexity
     Bias
     
Underfitting ←─── Optimal ───→ Overfitting
(High Bias)                   (High Variance)
```

**Signs of Overfitting**:
- Training accuracy >> Test accuracy
- Perfect training, poor test
- **Fix**: More data, regularization, simpler model, cross-validation

**Signs of Underfitting**:
- Low training AND test accuracy
- **Fix**: More features, complex model, less regularization

---

## 🚨 Common ML Pitfalls

### Data Leakage
```python
# ❌ WRONG
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✓ CORRECT
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Using Test Set for Tuning
```python
# ❌ WRONG: Tuning on test set
for param in params:
    model.fit(X_train, y_train)
    if model.score(X_test, y_test) > best:  # Leakage!
        best = model

# ✓ CORRECT: Use validation/CV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
final_score = grid_search.score(X_test, y_test)  # ONE time!
```

### Not Scaling for Distance-Based Models
- **Must scale**: k-NN, SVM, Neural Networks, k-Means
- **No need**: Decision Trees, Random Forest, Naive Bayes

### Imbalanced Data Mistakes
- Using accuracy (use F1, precision/recall instead)
- Not stratifying train/test split
- Not using class weights or SMOTE

---

## 🧮 Key Formulas

### Scaling
```
StandardScaler: (x - μ) / σ
MinMaxScaler: (x - min) / (max - min)
```

### Regularization
```
Ridge (L2): Loss + λ·Σ(β²)
Lasso (L1): Loss + λ·Σ|β|
ElasticNet: Loss + λ₁·Σ|β| + λ₂·Σ(β²)
```

### Distance Metrics
```
Euclidean: √[Σ(x-y)²]
Manhattan: Σ|x-y|
Cosine: (x·y) / (||x||·||y||)
```

### Information Theory
```
Entropy: -Σ p(x)·log₂(p(x))
Information Gain: Entropy(parent) - Weighted_Avg(Entropy(children))
Gini: 1 - Σ p²(x)
```

---

## 📚 Scikit-Learn API Pattern

```python
# 1. Import
from sklearn.xxx import YourModel

# 2. Instantiate
model = YourModel(param1=value1, param2=value2)

# 3. Fit (train)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
score = model.score(X_test, y_test)

# Key attributes (after fitting):
model.coef_          # Coefficients (linear models)
model.feature_importances_  # Importance (tree models)
model.n_features_in_ # Number of features
```

---

## 🎯 Quick Decision Guide

**Q: Small dataset (<1000 samples)?**
→ Use: Logistic Regression, Naive Bayes, SVM (avoid deep learning)

**Q: Need interpretability?**
→ Use: Linear Regression, Logistic Regression, Decision Tree

**Q: Have lots of data (>100K samples)?**
→ Use: Neural Networks, Gradient Boosting, Random Forest

**Q: High-dimensional data?**
→ Use: PCA for preprocessing, then any model

**Q: Imbalanced classes?**
→ Use: F1-score, SMOTE, class weights, stratified CV

**Q: Time series data?**
→ Use: ARIMA, LSTM, Prophet (NO random shuffling!)

**Q: Text data?**
→ Use: TF-IDF + Logistic Regression / Naive Bayes, or BERT

**Q: Image data?**
→ Use: CNN, Transfer Learning (ResNet, VGG), Vision Transformers

---

## 💡 Exam Tips

### Must Remember
1. **Train/test must be split BEFORE preprocessing**
2. **Accuracy is BAD for imbalanced data**
3. **k-NN, SVM, Neural Nets NEED scaling**
4. **LabelEncoder for target, OneHotEncoder for features**
5. **Cross-validation for hyperparameter tuning, NOT test set**
6. **Stratify for classification, especially if imbalanced**
7. **NO shuffling for time series**
8. **Pipeline prevents data leakage**
9. **Regularization prevents overfitting (λ↑ = regularization↑)**
10. **More trees in Random Forest = better (but slower)**

### Common MCQ Traps
- "Accuracy is best metric for all tasks" → **FALSE** (imbalanced!)
- "Test set used for hyperparameter tuning" → **FALSE** (validation!)
- "Scale before train/test split" → **FALSE** (data leakage!)
- "Decision trees need feature scaling" → **FALSE** (only distance-based!)
- "Overfitting means high train AND test error" → **FALSE** (that's underfitting!)

---

**Print this page for quick exam reference!** 📄

**End of Cheat Sheet**
