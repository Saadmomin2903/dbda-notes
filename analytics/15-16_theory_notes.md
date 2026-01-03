# 📚 Theory‑Focused Notes for PG‑DBDA (ACTS, Pune)

---
## 1️⃣ From Correlation to Prediction

### Why we start with **correlation**
- **Motivation**: In exploratory data analysis we need a *quick* quantitative sense of how two variables move together before committing to a model. Correlation tells us whether a linear relationship *might* be useful for prediction.
- **Historical context**: Pearson’s correlation (1895) was introduced to summarise the strength of linear association in biological data; it became the cornerstone of early regression work.

### Intuition & Definition
| Symbol | Meaning |
|--------|---------|
| \(X, Y\) | Random variables (population) |
| \(\mu_X, \mu_Y\) | Means |
| \(\sigma_X, \sigma_Y\) | Standard deviations |
| \(\rho_{XY}\) | Population Pearson correlation |

\[\rho_{XY}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}=\frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}\]

- **Interpretation**: \(\rho=+1\) → perfect positive linear relationship; \(\rho=0\) → no linear relationship (non‑linear may still exist).

### From Sample to Population
- Sample correlation \(r\) is an *estimator* of \(\rho\). It is unbiased only under bivariate normality.
- **Assumptions**: 
  - Linear relationship (or monotonic for Spearman).
  - Homoscedasticity (constant variance).
  - No extreme outliers (they can inflate/deflate \(r\)).

### Limitations & Edge Cases
| Situation | Why correlation fails | MCQ ALERT |
|-----------|----------------------|----------|
| Non‑linear monotonic trend | Pearson captures only linear trend; Spearman rank correlation is needed. | 1. *A dataset shows a perfect quadratic curve. Which correlation coefficient will be 0?* |
| Heteroscedastic data | Variance changes with X, violating constant variance assumption; \(r\) can be misleading. | 2. *When variance of Y increases with X, what happens to the significance of \(r\)?* |
| Outliers | A single outlier can drive \(r\) to ±1 even when bulk of data is unrelated. | 3. *Removing a single point changes \(r\) from 0.9 to 0.2. What does this indicate?* |

### Transition to **Prediction**
- **Linear regression** builds on correlation: \(\beta_1 = r\frac{\sigma_Y}{\sigma_X}\) under simple linear regression assumptions.
- **Key shift**: From *association* (does X relate to Y?) to *causation‑oriented modeling* (can we predict Y for new X?).
- **Assumptions for valid prediction**:
  1. Model correctly specified (linearity, correct functional form).
  2. Errors are i.i.d. Gaussian (for inference) – not required for point prediction but affects confidence intervals.
  3. No omitted variable bias (see next section on relevance vs causality).

---
## 2️⃣ Feature Relevance vs Causality

### Why distinguish them?
- **Relevance**: A feature that improves predictive performance (e.g., lowers MSE) regardless of underlying causal mechanism.
- **Causality**: A feature that *actually influences* the outcome; essential for policy decisions, interventions, and interpretability.

### Intuition
| Concept | Goal | Typical Metric |
|---------|------|----------------|
| **Relevance** | Maximise predictive accuracy | Feature importance (e.g., Gini, permutation) |
| **Causality** | Understand *why* the outcome changes | Average Treatment Effect, Do‑Calculus, Instrumental Variables |

### Common Misconceptions (MCQ traps)
1. *“If a variable has high importance, it must be causal.”* – False. Importance can arise from correlation with a causal proxy.
2. *“Removing a non‑causal but relevant feature always improves model interpretability without hurting performance.”* – Not always; may increase bias.

### Assumptions for Causal Inference
- **Ignorability/No unmeasured confounding** – all common causes are observed.
- **Positivity** – each level of treatment has non‑zero probability.
- **Stable Unit Treatment Value Assumption (SUTVA)** – no interference between units.

### Edge Cases
| Situation | Relevance‑Causality Gap |
|-----------|--------------------------|
| **Collider bias** | Conditioning on a collider creates spurious association → high relevance but no causality. |
| **Mediators** | A mediator may appear relevant; removing it changes total effect estimation. |
| **Time‑lagged variables** | Lagged predictor can be relevant for prediction but may not be causal for current outcome. |

---
## 3️⃣ Supervised Segmentation Logic

### Why segment?
- Real‑world data often contains *heterogeneous sub‑populations* (e.g., customers, patients). A single global model may under‑fit each subgroup.

### Intuition
- **Supervised segmentation** = *learn a partition of the feature space* such that within each segment a simple model (often linear) works well.
- It is a *two‑stage* process: (1) find segments, (2) fit local models.

### Formal Definition
Given data \((X_i, Y_i)\), find a mapping \(S: \mathcal{X} \rightarrow \{1,\dots,K\}\) that maximises a criterion, e.g., reduction in total SSE:
\[\text{Objective}=\sum_{k=1}^{K}\sum_{i: S(X_i)=k}(Y_i-\hat{Y}_{k,i})^2\]

### Typical Algorithms
| Algorithm | How it works | When to use |
|-----------|--------------|------------|
| **Decision‑tree based segmentation** | Tree splits create homogeneous leaves; each leaf fits a local model. | Small‑to‑medium data, interpretability needed. |
| **Mixture‑of‑Experts (MoE)** | Gating network learns soft assignment probabilities; experts are local regressors. | Large data, smooth transitions between segments. |
| **Cluster‑then‑regress** | Unsupervised clustering on X, then fit separate models per cluster. | When clusters are well‑separated and label noise is low. |

### Assumptions & Pitfalls
- **Sufficient data per segment** – otherwise variance explodes.
- **Segment stability** – if segments change drastically with new data, model maintenance is costly.
- **Over‑segmentation** – leads to overfitting (see Overfitting intuition below).

### MCQ ALERT
1. *Which of the following is *not* a typical reason to use supervised segmentation?* (Options: A) Heterogeneous error variance, B) Non‑linear relationships, C) Identical distributions across groups, D) Different optimal predictors.)

---
## 4️⃣ Tree Models as Rule Systems

### Why view trees as rules?
- Each root‑to‑leaf path encodes a **conjunctive rule** (AND of split conditions). This makes trees naturally interpretable.

### Formal Rule Extraction
For a binary tree, a leaf \(l\) defines rule \(R_l\):
\[R_l = \bigwedge_{j \in \text{ancestors}(l)} \text{condition}_j\]
Prediction = \(\hat{y}_l\) (mean for regression, majority class for classification).

### Example (textual)
```
if (Age <= 30) AND (Income > 50k) → Predict: High‑Spender
else if (Age > 30) AND (CreditScore <= 600) → Predict: Risky
else → Predict: Average
```

### Advantages
- **Transparency** – each decision can be traced.
- **Non‑parametric** – no distributional assumptions.
- **Handles mixed data** – categorical and numeric splits.

### Limitations & Edge Cases
| Issue | Why it matters |
|-------|----------------|
| **Depth explosion** | Deep trees produce many long rules, hurting interpretability. |
| **Axis‑aligned splits** | Trees split on one variable at a time → may need many splits to capture diagonal boundaries. |
| **Instability** | Small data perturbations can drastically change splits (high variance). |

### MCQ ALERT
2. *A depth‑3 decision tree can represent at most how many distinct conjunctive rules?* (Options: 4, 6, 8, 12)

---
## 5️⃣ Interpretability vs Accuracy Trade‑off

### Why the trade‑off exists
- **Bias‑Variance Perspective**: Simple, interpretable models (e.g., linear regression, shallow trees) have higher bias but lower variance. Complex models (e.g., deep ensembles, boosted trees) reduce bias but increase variance and become opaque.
- **Human‑cognition limit**: Humans can reliably process ~7±2 logical conditions; beyond that interpretability drops sharply.

### Intuition Table
| Model Family | Typical Accuracy (on benchmark) | Typical Interpretability |
|--------------|--------------------------------|--------------------------|
| Linear/Logistic Regression | Moderate | High (coefficients) |
| Shallow Decision Tree (depth ≤ 3) | Moderate‑Low | High (few rules) |
| Random Forest (100 trees) | High | Low (ensemble) |
| Gradient Boosted Trees (XGBoost) | Very High | Low‑Medium (feature importance) |
| Neural Networks (deep) | Highest | Very Low (black‑box) |

### Decision Framework (when to favour interpretability)
1. **Regulatory/Legal** – need to justify decisions (e.g., credit scoring).
2. **Domain expertise** – stakeholder wants to validate model logic.
3. **Data scarcity** – simpler models generalise better.

### Edge Cases
- **High‑dimensional sparse data**: Linear models with L1 regularisation can be both accurate *and* interpretable (sparse coefficients).
- **Post‑hoc explanations** (e.g., SHAP) can *appear* interpretable but may mislead – they are approximations.

### MCQ ALERT
3. *Which technique provides the most faithful local explanation for a tree‑based model?* (Options: LIME, SHAP, Partial Dependence, Decision Path)

---
## 📌 Summary of Key MCQ Traps
| Topic | Common Trap | Correct Reasoning |
|-------|--------------|-------------------|
| Correlation → Prediction | Assuming high \(r\) guarantees good predictions. | Prediction also needs correct model form and low error variance. |
| Feature Relevance vs Causality | Equating importance with causation. | Importance is predictive; causality requires structural assumptions. |
| Supervised Segmentation | Over‑segmenting → overfit. | Need enough data per segment; use validation to choose K. |
| Tree Rules | Counting rules incorrectly. | A binary tree with depth \(d\) yields at most \(2^{d}\) leaves → that many rules. |
| Interpretability vs Accuracy | Believing “black‑box = always better”. | Trade‑off depends on data size, domain, and regulatory constraints. |

---
## 📚 Further Reading (concise list)
- Pearson, K. (1895). *Notes on regression and correlation*. Philosophical Magazine.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*.
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- Friedman, J. (1991). *Multivariate Adaptive Regression Splines*. Annals of Statistics.
- Molnar, C. (2022). *Interpretable Machine Learning* (chapters on tree rules & SHAP).

---
*End of notes.*
