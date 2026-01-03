# Session 29 – AI Ethics & Fairness

## 📚 Table of Contents
1. [Bias in AI Systems](#bias-in-ai-systems)
2. [Fairness Metrics](#fairness-metrics)
3. [Explainability & Interpretability](#explainability--interpretability)
4. [Privacy-Preserving ML](#privacy-preserving-ml)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Bias in AI Systems

## 📊 Sources of Bias

### 1. Historical Bias
Data reflects past discrimination.

**Example**: Hiring data favoring men (historical gender imbalance).

### 2. Representation Bias
Some groups underrepresented in training data.

**Example**: Face recognition trained mostly on lighter skin tones.

### 3. Measurement Bias
Proxies don't accurately capture target variable.

**Example**: Using ZIP code as proxy for income.

### 4. Aggregation Bias
One model for diverse groups with different patterns.

**Example**: Single diabetes model for all ethnicities (different risk factors).

### 5. Evaluation Bias
Test data doesn't represent deployment population.

---

# Fairness Metrics

## 🧮 Group Fairness

### Demographic Parity
**Equal acceptance rate across groups**:
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
```

Where A = protected attribute (e.g., gender, race)

**Use**: When equal opportunity is goal

### Equalized Odds
**Equal TPR and FPR across groups**:
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (TPR)
P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (FPR)
```

**Strongest fairness criterion**.

### Equal Opportunity
**Equal TPR only**:
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)
```

**Use**: When false negatives more costly.

## 🧮 Individual Fairness

**Similar individuals treated similarly**:
```
d(x_i, x_j) small → d(f(x_i), f(x_j)) small
```

**Challenge**: Defining similarity metric.

## 📊 Fairness-Accuracy Tradeoff

**Impossibility results**: Can't satisfy all fairness criteria + perfect accuracy simultaneously.

**Tradeoff**: Often must sacrifice some accuracy for fairness.

## 🧪 Bias Mitigation

### Pre-processing
Modify training data:
- **Resampling**: Oversample minority groups
- **Reweighting**: Higher weight to underrepresented groups

### In-processing
Add fairness constraints during training:
```
L_total = L_task + λ L_fairness
```

### Post-processing
Adjust predictions to satisfy fairness:
- Threshold optimization per group
- Calibration

---

# Explainability & Interpretability

## 📊 LIME (Local Interpretable Model-agnostic Explanations)

**Idea**: Approximate model locally with interpretable model.

**Algorithm**:
```
1. Select instance x to explain
2. Generate perturbed samples around x
3. Get model predictions for samples
4. Fit simple model (e.g., linear) on samples
5. Explain using simple model coefficients
```

**Example**:
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['No', 'Yes'],
    mode='classification'
)

explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=5
)

explanation.show_in_notebook()
```

## 🧮 SHAP (SHapley Additive exPlanations)

**Game theory approach**: Shapley values.

**Feature contribution**:
```
φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S∪{i}) - v(S)]
```

Where:
- N = all features
- S = subset of features
- v(S) = model output with feature subset S

**Properties**:
- Σ φ_i = f(x) - E[f(X)] (additivity)
- Symmetry, dummy, efficiency

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

## 📊 Attention Visualization

For transformers, visualize attention weights:
```python
# Get attention weights
attentions = model(input_ids, output_attentions=True).attentions

# Visualize
import matplotlib.pyplot as plt

plt.imshow(attentions[0][0].detach().numpy())
plt.colorbar()
```

---

# Privacy-Preserving ML

## 🧮 Differential Privacy

**Definition**: Algorithm's output similar whether individual data included or not.

**ε-Differential Privacy**:
```
P(M(D) ∈ S) ≤ e^ε × P(M(D') ∈ S)
```

Where D, D' differ in one record.

**Implementation**: Add calibrated noise
```
Query result + Laplace(λ = Δf/ε)
```

Δf = global sensitivity (max change in output)

**DP-SGD** (Differentially Private SGD):
```
1. Clip gradients: g_i ← g_i / max(1, ||g_i||/C)
2. Add noise: g̃ = (1/N) Σ g_i + N(0, σ²C²)
3. Update: θ ← θ - η g̃
```

## 📊 Federated Learning

**Training without centralizing data**.

**Algorithm**:
```
Server:
  Initialize global model θ
  for each round:
    Send θ to selected clients
    Aggregate client updates → θ_new
    θ ← θ_new

Client:
  Receive θ from server
  Train on local data → Δθ
  Send Δθ to server
```

**Benefits**:
- Data stays on device
- Privacy preserved
- Reduces central data storage

## 📊 Data Anonymization

**Techniques**:
- **K-anonymity**: Each record indistinguishable from k-1 others
- **L-diversity**: At least L different sensitive values per group
- **T-closeness**: Distribution of sensitive attribute similar to overall

---

# 🔥 MCQs

### Q1. Demographic parity ensures:
**Options:**
- A) Equal accuracy
- B) Equal acceptance rate across groups ✓
- C) Individual fairness
- D) Privacy

**Explanation**: P(Ŷ=1|A=0) = P(Ŷ=1|A=1).

---

### Q2. SHAP uses:
**Options:**
- A) Linear regression
- B) Shapley values from game theory ✓
- C) Random sampling
- D) Decision trees

**Explanation**: Game-theoretic feature attribution.

---

### Q3. Differential privacy adds:
**Options:**
- A) Features
- B) Calibrated noise ✓
- C) Data
- D) Models

**Explanation**: Noise protects individual privacy.

---

### Q4. Federated learning:
**Options:**
- A) Centralizes data
- B) Keeps data on devices ✓
- C) Requires cloud
- D) Not private

**Explanation**: Training without data centralization.

---

### Q5. LIME approximates locally with:
**Options:**
- A) Deep network
- B) Interpretable model (e.g., linear) ✓
- C) Ensemble
- D) SVM

**Explanation**: Fits simple model around instance.

---

# ⚠️ Common Mistakes

1. **Ignoring fairness**: Can perpetuate discrimination
2. **Using only accuracy**: Doesn't reveal bias
3. **Not checking all fairness metrics**: Different metrics reveal different issues
4. **Assuming correlation means causation**: In fairness analysis
5. **No diversity in test data**: Can't detect bias
6. **Not explaining critical decisions**: Medical, legal need explainability
7. **Insufficient noise in DP**: Privacy not guaranteed
8. **Thinking fairness is purely technical**: Requires domain expertise

---

# ⭐ One-Line Exam Facts

1. **Demographic parity**: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
2. **Equalized odds**: Equal TPR and FPR across groups
3. **Equal opportunity**: Equal TPR only
4. **LIME**: Local approximation with interpretable model
5. **SHAP**: Shapley values for feature importance
6. **Differential privacy**: P(M(D)) ≤ e^ε P(M(D'))
7. **DP-SGD**: Gradient clipping + noise addition
8. **Federated learning**: Decentralized training
9. **K-anonymity**: Each record indistinguishable from k-1
10. **Historical bias**: Data reflects past discrimination
11. **Representation bias**: Underrepresented groups
12. **Fairness-accuracy tradeoff**: Can't optimize both perfectly
13. **Attention visualization**: Shows model focus
14. **Privacy budget ε**: Smaller = more private
15. **Shapley additivity**: Σ φ_i = f(x) - E[f(X)]

---

**End of Session 29**
