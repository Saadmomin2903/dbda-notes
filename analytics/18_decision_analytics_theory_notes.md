# 📚 Advanced Theory Notes – Decision Analytics & Model Evaluation (PG‑DBDA, ACTS, Pune)

---
## SECTION A: Decision Analytics

### 1️⃣ What *Decision Analytics* means
- **Why it exists:** Business environments rarely stop at describing what happened (descriptive) or forecasting what may happen (predictive). Managers must *choose* actions; decision analytics provides a systematic, quantitative framework to turn insights into *optimal choices*.
- **How it differs:** 
  - *Descriptive* → summarises past data (e.g., dashboards).
  - *Predictive* → builds models to estimate future outcomes (e.g., regression, classification).
  - *Decision* → embeds predictions into a **decision‑making model** (utility, cost, risk) to select the *best alternative*.

### 2️⃣ Why prediction alone is insufficient
- **Prediction ≠ Preference:** A model may forecast high sales for a product, but if the profit margin is low, the optimal decision could be to *not* launch it.
- **Uncertainty & Risk:** Predictions are point estimates with uncertainty; decisions must consider *distribution* of possible outcomes (e.g., via expected utility, Value‑At‑Risk).
- **Resource Constraints:** Real‑world actions are limited by budget, time, or capacity – factors not captured by pure prediction.

### 3️⃣ Decision‑making under uncertainty & risk
| Concept | Intuition | Typical Formalisation |
|---------|-----------|-----------------------|
| **Expected Utility** | Choose action that maximises average payoff weighted by risk attitude. | \(\max_a \; \mathbb{E}[U(\text{Outcome}\mid a)]\) |
| **Risk‑Adjusted Return** | Penalise variability; e.g., Sharpe ratio. | \(\frac{\mu - r_f}{\sigma}\) |
| **Scenario Analysis** | Evaluate decisions across a set of plausible futures. | Enumerate \(\{\text{scenario}_i\}\) and compute outcomes. |
| **Stochastic Dominance** | Preference ordering without specifying utility. | First‑order: \(F_A(x) \le F_B(x)\) for all \(x\). |

#### Common MCQ Traps
- *“The highest predicted sales always yields the best decision.”* – Ignores cost, risk, and constraints.
- *“Risk‑free decisions do not need probability modeling.”* – Even deterministic actions can have uncertain outcomes.

---
## SECTION B: Evaluating Classifiers (THEORY ONLY)

### 1️⃣ Why model evaluation is required
- **Why:** A classifier’s *training* performance is optimistic; we need *generalisation* assessment to ensure it will behave acceptably on unseen data and, crucially, to align with business objectives.
- **How:** Evaluation bridges the gap between statistical performance and *operational usefulness*.

### 2️⃣ Accuracy vs Business Usefulness
- **Accuracy** measures overall proportion of correct predictions but treats all errors equally.
- **Business usefulness** weighs errors by their *impact* (e.g., false‑negative fraud detection may cost millions, while false‑positive may be a minor inconvenience).

### 3️⃣ Core Concepts (interpretation only)
#### Confusion Matrix
```
                Predicted
                +      -
Actual   +   TP      FN
         -   FP      TN
```
- **TP** (True Positive): Correctly predicted positive class.
- **FN** (False Negative): Missed a positive case – often *costly* in safety‑critical domains.
- **FP** (False Positive): Incorrectly flagged negative as positive – may waste resources.
- **TN** (True Negative): Correctly predicted negative.

#### Derived Metrics (interpretation)
| Metric | Intuition | When it matters |
|--------|-----------|-----------------|
| **Accuracy** | Overall correctness. | Balanced classes, equal error costs. |
| **Precision** | Proportion of predicted positives that are true. | High cost of *false positives* (e.g., spam filters). |
| **Recall (Sensitivity)** | Proportion of actual positives captured. | High cost of *false negatives* (e.g., disease screening). |
| **F1‑Score** | Harmonic mean of precision & recall – balances both. | When you need a single summary under class imbalance. |

### 4️⃣ Cost‑Sensitive Decisions & Misclassification Costs
- **Why:** Real‑world decisions assign a monetary or utility cost to each type of error (\(C_{FP}, C_{FN}\)).
- **How:** Choose the class label that minimises *expected cost*:
\[\text{Predict Positive if } \; P(\text{Positive}\mid x) \cdot C_{FN} < (1-P(\text{Positive}\mid x)) \cdot C_{FP}\]
- **Threshold Tuning:** Adjust decision threshold away from 0.5 to reflect asymmetric costs.

#### MCQ ALERT
1. *Which metric is most appropriate when the cost of missing a fraud case (FN) is far higher than flagging a legitimate transaction (FP)?* (A) Accuracy, (B) Precision, (C) Recall, (D) F1‑Score.)

---
## SECTION C: Analytical Framework (Decision‑Oriented Analytics)

### 1️⃣ End‑to‑End Steps
| Step | Why it matters | Typical Activities |
|------|----------------|--------------------|
| **1. Problem Definition** | Align analytics with strategic goals. | Define decision alternatives, objectives, constraints. |
| **2. Data Acquisition & Preparation** | Quality data is the foundation; garbage in → garbage out. | Collect, clean, feature‑engineer, assess uncertainty. |
| **3. Model Development** | Translate data into predictive or prescriptive insight. | Choose appropriate predictive model, calibrate, validate. |
| **4. Decision Modeling** | Convert predictions into *actionable* recommendations. | Build utility/cost functions, perform optimisation, scenario analysis. |
| **5. Implementation & Monitoring** | Real‑world impact must be measured; feedback informs refinement. | Deploy decision rule, collect outcome data, update model (feedback loop). |

### 2️⃣ Data → Model → Decision → Outcome Feedback Loop
- **Why a loop?** Business environments evolve; static models become stale. Continuous learning ensures relevance and improves future decisions.

#### MCQ ALERT
2. *In the analytical framework, which step directly addresses *risk* associated with uncertain predictions?* (A) Data preparation, (B) Model development, (C) Decision modeling, (D) Monitoring.)

---
## SECTION D: Evaluation Philosophy

### 1️⃣ “Best model” depends on context
- **Why:** Different stakeholders value different outcomes (e.g., regulator vs. profit‑maximiser). The *optimal* model is the one that aligns with the **decision objective**, not necessarily the one with highest statistical score.

### 2️⃣ Trade‑off between False Positives & False Negatives
- **Why:** Adjusting the decision threshold moves the operating point along the ROC curve; the optimal point balances *business cost* of each error type.
- **How to visualise:** Cost curve or profit curve – plot expected profit vs. threshold.

#### MCQ ALERT
3. *A classifier with 99 % accuracy on a dataset where 1 % are positives suffers from which paradox?* (A) Overfitting, (B) Accuracy paradox, (C) Class‑imbalance, (D) None of the above.)

---
## 📊 Comparison Table – Metrics vs. Business Goals
| Business Goal | Preferred Metric(s) | Rationale |
|---------------|--------------------|-----------|
| Minimise costly false alarms (e.g., spam) | Precision, Cost‑sensitive loss | Penalises FP heavily. |
| Detect rare events (e.g., fraud) | Recall, F1‑Score, ROC‑AUC | Rewards capturing positives. |
| Balanced performance on balanced data | Accuracy, ROC‑AUC | Errors equally weighted. |
| Maximise overall profit | Expected Cost, Profit Curve | Directly incorporates monetary impact. |

---
## 📝 15 Examiner‑Style Conceptual MCQs (with explanations)
1. **Why is prediction alone insufficient for decision making?**
   - *Explanation:* Prediction provides *what may happen* but does not encode *preferences* (costs, utilities) or *constraints* needed to choose an action.
2. **In a confusion matrix, which cell directly contributes to *Recall*?**
   - *Explanation:* Recall = TP / (TP + FN); only TP and FN matter.
3. **What does the *accuracy paradox* illustrate?**
   - *Explanation:* High overall accuracy can mask poor performance on the minority class when data is imbalanced.
4. **When should you prefer the F1‑Score over Accuracy?**
   - *Explanation:* When class distribution is skewed and you need a balance between Precision and Recall.
5. **Which decision‑analytic concept explicitly incorporates the decision‑maker’s risk attitude?**
   - *Explanation:* Expected Utility Theory.
6. **How does *cost‑sensitive learning* differ from standard classification?**
   - *Explanation:* It weights errors by their real‑world costs rather than treating all mistakes equally.
7. **What is the primary purpose of *scenario analysis* in decision analytics?**
   - *Explanation:* To evaluate how decisions perform under a set of plausible future states.
8. **Why might a model with higher ROC‑AUC be unsuitable for a particular business problem?**
   - *Explanation:* ROC‑AUC ignores actual cost/benefit structure; a model with lower AUC but better alignment to cost may be preferable.
9. **Which metric is most appropriate for evaluating a medical test where missing a disease is catastrophic?**
   - *Explanation:* Recall (Sensitivity) because false negatives are extremely costly.
10. **What does *Stochastic Dominance* allow you to compare without specifying a utility function?**
    - *Explanation:* It provides a partial ordering of distributions based on risk‑averse preferences.
11. **In decision modeling, what role does the *utility function* play?**
    - *Explanation:* It translates outcomes into a scalar measure of desirability, reflecting stakeholder preferences.
12. **Why is the *feedback loop* essential in analytical frameworks?**
    - *Explanation:* It enables model updating and continuous improvement as real outcomes become available.
13. **Which error type is typically more concerning in credit‑card fraud detection?**
    - *Explanation:* False Negatives (missed fraud) because they incur direct monetary loss.
14. **How does *threshold tuning* affect the trade‑off between FP and FN?**
    - *Explanation:* Raising the threshold reduces FP but increases FN; lowering does the opposite.
15. **What is the key limitation of using *accuracy* as the sole evaluation metric in imbalanced datasets?**
    - *Explanation:* It can be misleading; a classifier that always predicts the majority class can achieve high accuracy while being useless for the minority class.

---
## 📚 Further Reading (concise list)
- *Decision Analysis for the Real World* – Clemen & Reilly (chapters on utility & risk).
- *The Elements of Statistical Learning* – Hastie, Tibshirani, Friedman (section on model evaluation).
- *Machine Learning: A Probabilistic Perspective* – Kevin Murphy (cost‑sensitive learning).
- *Data‑Driven Decision Making* – Provost & Fawcett (framework & feedback loop).
- *An Introduction to ROC Analysis* – Fawcett (2006).

---
*End of notes.*
