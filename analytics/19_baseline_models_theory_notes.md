# 📚 Theory‑Intensive Notes – Baseline Models, Evaluation & Data Investment (PG‑DBDA, ACTS, Pune)

---
## SECTION A: Baseline Models

### 1️⃣ What a **baseline model** is and why it is *mandatory*
- **Why it exists:** In any modelling project the first question is *"Is my model doing anything useful?"* A baseline provides the **minimum acceptable performance** against which every subsequent model must be compared.
- **How it is used:** Compute a simple, well‑defined metric (accuracy, MAE, etc.) for the baseline; any candidate model must **beat** this number **and** demonstrate a *meaningful* improvement.
- **Mandatory because:**
  1. **Detects trivial solutions** – prevents reporting inflated scores that arise from data leakage or label imbalance.
  2. **Sets a business‑level reference** – stakeholders can instantly see whether the effort adds value.
  3. **Guides resource allocation** – if a sophisticated model cannot surpass a naïve rule, further investment is unjustified.

### 2️⃣ Types of baseline models
| Type | Description | Typical Implementation |
|------|-------------|------------------------|
| **Random (chance) baseline** | Predicts uniformly at random (classification) or draws from the marginal distribution (regression). | `np.random.choice(labels)` or `np.random.normal(mean, std)`.
| **Majority‑class (mode) baseline** | Always predicts the most frequent class (classification) or the mean of the target (regression). | `np.bincount(y).argmax()` or `np.mean(y)`.
| **Simple rule‑based baseline** | Uses a deterministic, domain‑specific rule (e.g., “if age > 65 then high risk”). | Hand‑crafted if‑else logic based on expert knowledge.
| **Historical‑average baseline** | For time‑series, predicts the last observed value or a moving average. | `y[t-1]` or `np.mean(last_k)`.

### 3️⃣ Why **beating the baseline** matters more than a high absolute accuracy
- **Accuracy paradox:** A model can achieve 95 % accuracy on a dataset where 94 % of instances belong to the majority class, yet it adds **no value** beyond the majority‑class baseline.
- **Business relevance:** Stakeholders care about **incremental gain** (e.g., additional profit per correctly identified lead). A 1 % lift over baseline may be far more valuable than a 10 % absolute accuracy on an imbalanced problem.
- **Model complexity justification:** If a complex model only marginally exceeds the baseline, the extra computational and maintenance cost is rarely justified.

> **MCQ ALERT** – *Which statement is **false**?
> A) A model that matches the majority‑class baseline is acceptable if it is easier to interpret.
> B) Beating the baseline guarantees a model is useful for business decisions.
> C) High accuracy on a highly imbalanced dataset can be misleading.
> D) A random baseline provides a lower bound for performance.

---
## SECTION B: Performance Evaluation

### 1️⃣ Relative vs absolute performance
- **Absolute performance** – raw metric value (e.g., 0.82 MAE, 0.91 accuracy). Useful for *benchmarking* against industry standards.
- **Relative performance** – improvement over a reference (baseline or previous version). Expressed as **Δ%** or **lift**.
- **Why the distinction matters:** Decision makers allocate budget based on *incremental* benefit, not on an absolute number that may be meaningless without context.

### 2️⃣ Model improvement justification
| Justification | What to report | When it is convincing |
|---------------|----------------|-----------------------|
| **Statistical significance** | Confidence intervals, hypothesis test (e.g., paired t‑test on cross‑validation scores). | Large sample size, low variance.
| **Business impact** | Expected profit, cost savings, ROI (e.g., $10 k per month). | Clear mapping from metric to monetary value.
| **Operational feasibility** | Runtime, memory, interpretability score. | When deployment constraints are tight.

### 3️⃣ Overfitting disguised as performance gain
- **Symptoms:** Very high training score, modest validation score, large gap > 10 %.
- **Root causes:**
  1. **Leakage** – using future information in features.
  2. **Excessive hyper‑parameter tuning** on the validation set (treating it as training).
  3. **Complex models** with many parameters relative to data size.
- **How to detect:** Use a **hold‑out test set** never seen during model development, or employ **nested cross‑validation**.

> **MCQ ALERT** – *A model shows 98 % accuracy on the training set but only 70 % on the test set. Which explanation is most plausible?*
> A) The test set is corrupted.
> B) The model is under‑fitting.
> C) The model has over‑fitted the training data.
> D) The metric is unsuitable.

---
## SECTION C: Implications for Investment in Data

### 1️⃣ Cost of data collection vs performance gain
| Data Investment | Typical Cost (USD) | Expected Δ Performance (relative) |
|----------------|--------------------|-----------------------------------|
| **Additional 1 k labeled rows** | $2 000 – $5 000 | 1‑3 % lift (diminishing after ~5 k rows).
| **High‑resolution sensor data** | $10 000 – $30 000 | 0.5‑2 % lift (often noise‑dominated).
| **Feature engineering (domain expert time)** | $5 000 – $15 000 | 3‑8 % lift if features capture hidden structure.
| **Full data lake ingestion** | $50 000+ | < 1 % lift for many tabular problems.

- **Why diminishing returns:** Learning curves show a **logarithmic** relationship between data volume and error reduction; after a certain point, each extra datum contributes negligible information.

### 2️⃣ When **complex models** are **NOT justified**
| Situation | Reason it’s unnecessary |
|----------|--------------------------|
| **High baseline performance** (e.g., > 95 % accuracy with simple rule). | Incremental gain is marginal; maintenance cost outweighs benefit.
| **Small, well‑structured dataset** (< 5 k rows). | Complex models over‑fit; simple linear or tree models suffice.
| **Strict latency / interpretability requirements** (e.g., real‑time credit scoring). | Black‑box models add latency and cannot be explained to regulators.
| **Limited budget for data labeling**. | Investing in better data (cleaning, labeling) yields higher ROI than a fancier algorithm.

> **MCQ ALERT** – *Which scenario most strongly suggests that a deep neural network is an over‑investment?*
> A) Predicting churn with 1 M historical records and many categorical features.
> B) Classifying handwritten digits with 60 k images.
> C) Forecasting monthly sales using 200 rows of clean, numeric data.
> D) Detecting anomalies in high‑frequency sensor streams.

---
## 📊 Business Interpretation Tables

### Table 1 – Metric vs Business Decision
| Metric | What business question it answers | When to rely on it |
|--------|-----------------------------------|--------------------|
| **Baseline lift (%)** | "Is the new model worth the investment?" | When ROI is tied to incremental performance.
| **Cost‑sensitive loss** | "What is the expected monetary loss from errors?" | When misclassification costs are asymmetric.
| **ROI (Δ Revenue / Δ Cost)** | "Will the model generate net profit?" | When you can quantify revenue impact per correct prediction.
| **Time‑to‑predict** | "Can we meet SLA requirements?" | When latency is a hard constraint.

### Table 2 – Common False Exam Statements (and why they are wrong)
| Statement (exam‑style) | Why it is a trap |
|--------------------------|-------------------|
| *"A model with 99 % accuracy is always superior to a baseline of 95 %.* | Ignores class imbalance and business cost; may still be useless.
| *"If a model beats the majority‑class baseline, it is ready for production.* | Overlooks overfitting, data leakage, and operational constraints.
| *"Adding more data always improves model performance.* | Violates diminishing returns; may introduce noise and increase processing cost.
| *"Complex models guarantee higher ROI.* | ROI depends on incremental gain vs. added cost; complexity can reduce interpretability and increase maintenance.

---
## 📝 12 Examiner‑Style Conceptual MCQs (with explanations)
1. **Why is a baseline model considered the “floor” of performance?**
   - *Explanation:* It represents the simplest plausible prediction; any model must exceed it to demonstrate value.
2. **Which baseline is appropriate for a regression problem with a highly skewed target distribution?**
   - *Explanation:* The **median** baseline (or mean) provides a robust reference; a random baseline would be too noisy.
3. **What does the “accuracy paradox” illustrate?**
   - *Explanation:* High overall accuracy can mask poor performance on the minority class when data is imbalanced.
4. **When should you report *relative* performance instead of *absolute* performance?**
   - *Explanation:* When the business cares about incremental improvement over the current solution.
5. **Which of the following is *not* a legitimate justification for abandoning a complex model?**
   - *Explanation:* If the model yields a statistically significant 0.1 % lift, it may still be worthwhile if the monetary impact is large.
6. **In a cost‑sensitive setting, which metric directly reflects business loss?**
   - *Explanation:* Expected cost (weighted sum of FP and FN costs).
7. **Why does over‑fitting often appear as a large gap between training and test performance?**
   - *Explanation:* The model memorises training noise that does not generalise.
8. **What is the typical shape of a learning curve for a well‑specified model?**
   - *Explanation:* Logarithmic decay of error with increasing data size.
9. **Which scenario most likely suffers from diminishing returns on data collection?**
   - *Explanation:* Adding more rows to an already large, clean dataset.
10. **When is a majority‑class baseline insufficient?**
    - *Explanation:* When the minority class carries disproportionate business value (e.g., fraud detection).
11. **How does a simple rule‑based baseline differ from a random baseline?**
    - *Explanation:* It encodes domain knowledge, often yielding higher performance than pure chance.
12. **Why might a model with higher absolute accuracy still have lower ROI than a simpler baseline?**
    - *Explanation:* The extra accuracy may not translate into enough additional revenue to offset higher development and maintenance costs.

---
## 📚 Further Reading (concise list)
- *Machine Learning Yearning* – Andrew Ng (chapter on baselines).
- *Pattern Recognition and Machine Learning* – Bishop (section on model comparison).
- *The Elements of Statistical Learning* – Hastie, Tibshirani, Friedman (chapters on over‑fitting and learning curves).
- *Data‑Driven Decision Making* – Provost & Fawcett (business ROI of analytics).
- *Cost‑Sensitive Learning* – Elkan (2001) – seminal paper on misclassification costs.

---
*End of notes.*
