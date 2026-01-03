# Session 6 – Evaluation Metrics

## 📚 Table of Contents
1. [Classification Metrics](#classification-metrics)
2. [Regression Metrics](#regression-metrics)
3. [ROC-AUC Analysis](#roc-auc-analysis)
4. [Choosing the Right Metric](#choosing-the-right-metric)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Classification Metrics

## 📘 Confusion Matrix

Foundation of all classification metrics.

```
                  Predicted
                Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN
```

Where:
- **TP (True Positive)**: Correctly predicted positive class
- **TN (True Negative)**: Correctly predicted negative class
- **FP (False Positive)**: Type I error (predicted positive, actually negative)
- **FN (False Negative)**: Type II error (predicted negative, actually positive)

### Python Implementation

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)

# Visualize
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print(f"TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
```

## 🧮 Core Metrics

### 1. Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation**: Proportion of correct predictions

**Pros**: Simple, intuitive
**Cons**: **Misleading for imbalanced data**

**Example failure case**:
```
Dataset: 95% negative class, 5% positive class
Naive classifier: Always predict negative
Accuracy = 95% (but useless!)
```

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

### 2. Precision (Positive Predictive Value)

```
Precision = TP / (TP + FP)
```

**Interpretation**: Of all positive predictions, how many were actually positive?

**Use case**: When **false positives are costly**
- Spam detection (don't want to mark important emails as spam)
- Medical diagnosis (don't want to unnecessarily worry patients)

**High precision → Few false alarms**

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")
```

### 3. Recall (Sensitivity,True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Interpretation**: Of all actual positives, how many did we catch?

**Use case**: When **false negatives are costly**
- Cancer screening (don't want to miss cancer cases)
- Fraud detection (want to catch all fraud)

**High recall → Few missed positives**

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.3f}")
```

### 4. F1 Score

**Harmonic mean** of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
  = 2TP / (2TP + FP + FN)
```

**Why harmonic mean?** Penalizes extreme values (both precision and recall must be high).

**Comparison**:
```
Precision = 0.9, Recall = 0.1
  Arithmetic mean = 0.5 (misleading)
  Harmonic mean (F1) = 0.18 (realistic)
```

**Use**: When you need **balance** between precision and recall

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.3f}")
```

### 5. F-beta Score

Generalization of F1 that allows **weighting** precision vs recall:

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

Where:
- **β < 1**: Favor precision (minimize false positives)
- **β = 1**: F1 score (balanced)
- **β > 1**: Favor recall (minimize false negatives)

**Common**: F2 score (β=2, emphasizes recall)

```python
from sklearn.metrics import fbeta_score

f2 = fbeta_score(y_true, y_pred, beta=2)
print(f"F2 Score: {f2:.3f}")
```

### 6. Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

**Interpretation**: Of all actual negatives, how many were correctly identified?

**Use**: Medical tests (want high specificity to avoid false alarms)

### 7. False Positive Rate (FPR)

```
FPR = FP / (FP + TN) = 1 - Specificity
```

**Used in**: ROC curve (x-axis)

## 📊 Multi-Class Classification Metrics

### Averaging Strategies

For multi-class, compute metric per class then average:

#### 1. Macro Average

```
Macro = (1/k) Σᵢ metric_i
```

Treats all classes equally (good for balanced datasets).

#### 2. Weighted Average

```
Weighted = Σᵢ (nᵢ/n) × metric_i
```

Weights by class frequency (accounts for imbalance).

#### 3. Micro Average

```
Micro = metric(all TP, FP, FN summed across classes)
```

Aggregates contributions from all classes (dominated by large classes).

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Multi-class
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 2, 0, 1, 1, 0, 1, 2]

print(f"Precision (macro): {precision_score(y_true_multi, y_pred_multi, average='macro'):.3f}")
print(f"Precision (weighted): {precision_score(y_true_multi, y_pred_multi, average='weighted'):.3f}")
print(f"Precision (micro): {precision_score(y_true_multi, y_pred_multi, average='micro'):.3f}")
```

## 🔍 Probabilistic Metrics

### 1. Log Loss (Cross-Entropy Loss)

Measures quality of **probability predictions**:

```
LogLoss = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

Where pᵢ = predicted probability of positive class.

**Range**: [0, ∞)
- 0 = perfect predictions
- ∞ = infinitely wrong

**Penalizes confident wrong predictions heavily.**

**Example**:
```
True label: 1 (positive)
Predicted probability: 0.99 → Log Loss = -log(0.99) ≈ 0.01 (good)
Predicted probability: 0.01 → Log Loss = -log(0.01) ≈ 4.61 (bad!)
```

```python
from sklearn.metrics import log_loss

y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.2]  # Probabilities

loss = log_loss(y_true, y_prob)
print(f"Log Loss: {loss:.3f}")
```

### 2. Brier Score

Mean squared error between predicted probabilities and actual outcomes:

```
Brier = (1/n) Σᵢ (pᵢ - yᵢ)²
```

**Range**: [0, 1]
- 0 = perfect
- 1 = worst possible

**Similar to log loss but less sensitive to extreme predictions.**

```python
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_true, y_prob)
print(f"Brier Score: {brier:.3f}")
```

---

# Regression Metrics

## 🧮 Common Metrics

### 1. Mean Absolute Error (MAE)

```
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

**Interpretation**: Average absolute difference between prediction and truth

**Pros**: 
- Robust to outliers (linear penalty)
- Same unit as target variable

**Cons**: Not differentiable at 0 (optimization issue)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.3f}")
```

### 2. Mean Squared Error (MSE)

```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Interpretation**: Average squared difference

**Pros**:
- Differentiable everywhere
- Heavily penalizes large errors

**Cons**:
- Sensitive to outliers (squared term)
- Unit is y²

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.3f}")
```

### 3. Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/n) Σᵢ (yᵢ - ŷᵢ)²]
```

**Interpretation**: Like MAE but more sensitive to large errors

**Pros**:
- Same unit as y (interpretable)
- Penalizes large errors more than MAE

**Cons**: Still sensitive to outliers

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.3f}")
```

### 4. R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σᵢ (yᵢ - ŷᵢ)²     (residual sum of squares)
SS_tot = Σᵢ (yᵢ - ȳ)²      (total sum of squares)
```

**Interpretation**: Proportion of variance explained by model

**Range**: (-∞, 1]
- **R² = 1**: Perfect predictions
- **R² = 0**: Model as good as predicting mean
- **R² < 0**: Model worse than predicting mean (possible on test set!)

**Decomposition**:
```
R² = 1 - MSE(model) / Var(y)
```

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.3f}")
```

**Important**: R² can be negative on test set if model overfits!

### 5. Adjusted R²

Penalizes model complexity:

```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]
```

Where:
- n = number of samples
- p = number of features

**Use**: Model comparison (accounts for number of features)

### 6. Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) Σᵢ |yᵢ - ŷᵢ| / |yᵢ|
```

**Interpretation**: Average percentage error

**Pros**: Scale-independent (can compare across datasets)
**Cons**: 
- Undefined when yᵢ = 0
- Asymmetric (over-predictions penalized less)

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape(y_true, y_pred):.2f}%")
```

## 📊 Metric Comparison

| Metric | Outlier Sensitivity | Unit | Range | Use Case |
|--------|-------------------|------|-------|----------|
| **MAE** | Low | Same as y | [0, ∞) | Robust to outliers |
| **MSE** | High | y² | [0, ∞) | Optimization (differentiable) |
| **RMSE** | High | Same as y | [0, ∞) | Penalize large errors |
| **R²** | Medium | Unitless | (-∞, 1] | Variance explained |
| **MAPE** | Medium | Percentage | [0, ∞) | Scale-independent |

---

# ROC-AUC Analysis

## 📘 ROC Curve

**ROC (Receiver Operating Characteristic)** plots **TPR vs FPR** at different classification thresholds.

```
TPR = TP / (TP + FN)  (Recall, Sensitivity)
FPR = FP / (FP + TN)  (1 - Specificity)
```

### Interpretation

```
1.0 ┤           Perfect
TPR │        ●  Classifier
    │       ╱│
    │      ╱ │
    │     ╱  │  Good
    │    ╱   │  Classifier
    │   ╱    │
    │  ╱     │
    │ ╱      │  Random
    │╱_______|  Classifier
0.0 └─────────────────
    0.0     FPR    1.0
```

**Key points**:
- **(0, 0)**: Predict all negative (threshold = 1)
- **(1, 1)**: Predict all positive (threshold = 0)
- **Diagonal line**: Random classifier (AUC = 0.5)
- **Top-left corner**: Perfect classifier (TPR=1, FPR=0)

## 🧮 AUC (Area Under Curve)

```
AUC = ∫₀¹ TPR(FPR) d(FPR)
```

**Interpretation**: Probability that model ranks random positive example higher than random negative example.

**Range**: [0, 1]
- **AUC = 0.5**: Random classifier
- **AUC = 1.0**: Perfect classifier
- **AUC < 0.5**: Worse than random (flip predictions!)

### AUC Guidelines

| AUC | Interpretation |
|-----|----------------|
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.6-0.7 | Poor |
| 0.5-0.6 | Fail |

## 🧪 Python Implementation

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute AUC
roc_auc = roc_auc_score(y_test, y_prob)
# OR: roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"AUC: {roc_auc:.3f}")
```

### Multi-Class ROC

**One-vs-Rest** or **One-vs-One** approaches:

```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Get probabilities for each class
y_prob_multi = model.predict_proba(X_test)

# Compute ROC AUC for each class
roc_auc_ovr = roc_auc_score(y_test_bin, y_prob_multi, multi_class='ovr', average='weighted')
print(f"Multi-class AUC (OvR): {roc_auc_ovr:.3f}")
```

## 🔄 ROC vs Precision-Recall Curve

### Precision-Recall Curve

Better for **imbalanced datasets** (many negatives, few positives).

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**Why?** ROC can be overly optimistic when negatives dominate (TN inflates FPR denominator).

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### When to Use Each

| Scenario | ROC-AUC | PR-AUC |
|----------|---------|--------|
| **Balanced classes** | ✓ Good | ✓ Good |
| **Imbalanced (rare positive)** | Can be misleading | ✓ Better |
| **Costs symmetric** | ✓ Good | OK |
| **Costs asymmetric** | OK | ✓ Better |

---

# Choosing the Right Metric

## 📊 Decision Framework

### Classification

```
┌─────────────────────────────┐
│ Is dataset balanced?        │
└──────┬────────────────┬─────┘
       │ Yes            │ No
       ▼                ▼
   Accuracy       F1/Precision-Recall
                  
┌─────────────────────────────┐
│ Need probability scores?    │
└──────┬──────────────────────┘
       │ Yes
       ▼
   Log Loss / Brier Score
                  
┌─────────────────────────────┐
│ What's more costly?         │
└──┬──────────────────────┬───┘
   │ False Positives      │ False Negatives
   ▼                      ▼
Precision              Recall
```

### Regression

```
┌─────────────────────────────┐
│ Data has outliers?          │
└──────┬────────────────┬─────┘
       │ Yes            │ No
       ▼                ▼
      MAE      RMSE / MSE / R²
                  
┌─────────────────────────────┐
│ Need scale-independent?     │
└──────┬──────────────────────┘
       │ Yes
       ▼
     MAPE
```

## ⚠️ Common Pitfalls

### 1. Accuracy Paradox

Dataset: 99% class 0, 1% class 1

```
Classifier 1: Always predict 0 → Accuracy = 99%
Classifier 2: Balanced (TPR=90%, TNR=90%) → Accuracy = 90.9%
```

**Classifier 1 has higher accuracy but is useless!**

**Solution**: Use F1, precision-recall, or ROC-AUC.

### 2. Optimizing Wrong Metric

**Problem**: Optimize accuracy, but deploy where false negatives are costly.

**Example**: Cancer detection
- Optimize for accuracy: May miss many cancer cases
- Should optimize for recall: Catch more cancer cases

### 3. Ignoring Class Weights

**Imbalanced data**: Minority class contributes little to many metrics.

**Solution**: Use class weights or resampling (SMOTE).

---

# 🔥 MCQs

### Q1. For imbalanced binary classification, which metric is most appropriate?
**Options:**
- A) Accuracy
- B) F1 Score ✓
- C) Mean Squared Error
- D) R²

**Explanation**: F1 balances precision and recall, not biased by class imbalance.

---

### Q2. Precision measures:
**Options:**
- A) TP / (TP + FN)
- B) TP / (TP + FP) ✓
- C) TN / (TN + FP)
- D) TP / (TP + TN)

**Explanation**: Precision = TP / (TP + FP) = of predicted positives, how many are actually positive.

---

### Q3. Recall is also called:
**Options:**
- A) Specificity
- B) False Positive Rate
- C) Sensitivity ✓
- D) Precision

**Explanation**: Recall = Sensitivity = TPR = TP / (TP + FN).

---

### Q4. F1 score is the ______ mean of precision and recall.
**Options:**
- A) Arithmetic
- B) Geometric
- C) Harmonic ✓
- D) Quadratic

**Explanation**: F1 = 2/(1/Precision + 1/Recall) = harmonic mean.

---

### Q5. AUC = 0.5 indicates:
**Options:**
- A) Perfect classifier
- B) Random classifier ✓
- C) Worst possible classifier
- D) Good classifier

**Explanation**: AUC = 0.5 means classifier performs no better than random.

---

### Q6. R² can be negative when:
**Options:**
- A) Never
- B) On training set
- C) On test set (overfitting) ✓
- D) When using MAE

**Explanation**: R² < 0 means model worse than predicting mean (possible on test if overfit).

---

### Q7. RMSE is more sensitive to outliers than MAE because:
**Options:**
- A) It uses absolute value
- B) It squares errors ✓
- C) It has lower value
- D) It's not differentiable

**Explanation**: Squared term (yᵢ - ŷᵢ)² heavily penalizes large errors.

---

### Q8. Log loss penalizes:
**Options:**
- A) Any wrong prediction
- B) Confident wrong predictions ✓
- C) Only false positives
- D) Only false negatives

**Explanation**: -log(p) → ∞ as p → 0 (confident wrong prediction heavily penalized).

---

### Q9. For cancer screening, optimize for:
**Options:**
- A) Precision
- B) Recall ✓
- C) Accuracy
- D) Specificity

**Explanation**: Want to catch all cancer cases (minimize false negatives).

---

### Q10. ROC curve plots:
**Options:**
- A) Precision vs Recall
- B) TPR vs FPR ✓
- C) Accuracy vs Threshold
- D) F1 vs Threshold

**Explanation**: ROC = Receiver Operating Characteristic plots TPR (y-axis) vs FPR (x-axis).

---

### Q11. Macro average treats:
**Options:**
- A) All samples equally
- B) All classes equally ✓
- C) Large classes more
- D) Small classes more

**Explanation**: Macro = arithmetic mean of per-class metrics (unweighted).

---

### Q12. Brier score measures:
**Options:**
- A) Classification accuracy
- B) Mean squared error of probabilities ✓
- C) Log loss
- D) AUC

**Explanation**: Brier = MSE between predicted probabilities and true labels.

---

### Q13. Which metric has same unit as target variable?
**Options:**
- A) MSE
- B) R²
- C) RMSE ✓
- D) All of the above

**Explanation**: RMSE = √MSE has same unit as y; MSE has unit y².

---

### Q14. For imbalanced dataset, prefer:
**Options:**
- A) ROC-AUC
- B) Precision-Recall AUC ✓
- C) Accuracy
- D) MSE

**Explanation**: PR-AUC focuses on minority class performance (better for imbalance).

---

### Q15. False Positive Rate (FPR) =
**Options:**
- A) 1 - Specificity ✓
- B) 1 - Sensitivity
- C) 1 - Precision
- D) 1 - Recall

**Explanation**: FPR = FP/(FP+TN) = 1 - TN/(TN+FP) = 1 - Specificity.

---

# ⚠️ Common Mistakes

1. **Using accuracy for imbalanced data**: Misleading (accuracy paradox)

2. **Confusing precision and recall**: Precision = predicted positives, Recall = actual positives

3. **Ignoring class imbalance**: Leads to biased models

4. **Not checking confusion matrix**: Provides insights beyond single metric

5. **Using wrong averaging for multi-class**: Macro vs weighted vs micro depends on goal

6. **Optimizing metric != deployment goal**: Match metric to business objective

7. **Comparing R² across datasets**: R² is relative, not absolute measure

8. **Using MAPE with zeros**: Undefined when y=0

9. **Assuming AUC > 0.9 is always good**: Context-dependent (medical vs marketing)

10. **Not visualizing ROC/PR curves**: Curves provide more info than single AUC number

---

# ⭐ One-Line Exam Facts

1. **Accuracy = (TP + TN) / Total** (misleading for imbalanced data)

2. **Precision = TP / (TP + FP)** (of predicted positives, how many correct?)

3. **Recall = TP / (TP + FN)** (of actual positives, how many caught?)

4. **F1 = harmonic mean** of precision and recall = 2TP/(2TP + FP + FN)

5. **Specificity = TN / (TN + FP)**, FPR = 1 - Specificity

6. **ROC curve plots TPR vs FPR** at different thresholds

7. **AUC = 0.5** (random), **AUC = 1.0** (perfect)

8. **PR curve better than ROC for imbalanced** datasets

9. **Log loss penalizes confident wrong predictions** heavily

10. **MAE robust to outliers**, **RMSE sensitive** (squared errors)

11. **R² ∈ (-∞, 1]**, can be **negative on test set** if overfitting

12. **MSE has unit y²**, **RMSE has unit y** (interpretable)

13. **Macro average** = unweighted mean (treats all classes equally)

14. **Weighted average** = weighted by class frequency

15. **F-beta with β > 1 emphasizes recall**, β < 1 emphasizes precision

---

**End of Session 6**
