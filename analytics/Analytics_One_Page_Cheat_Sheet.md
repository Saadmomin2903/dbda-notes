# 📄 Advanced Analytics: The "One-Page" Theory Cheat Sheet
**For PG-DBDA Theory Exam**

---

## 🏗️ 1. Analytic Architectures & Strategy
| Concept | Definition | Key Distinction |
| :--- | :--- | :--- |
| **Descriptive** | "What happened?" | Dashboards, Hindsight. |
| **Predictive** | "What will happen?" | Models, Foresight. (Note: Probability, not certainty). |
| **Prescriptive** | "What should we do?" | Optimization, Action. Best outcome selection. |
| **Decision Analytics** | Analytics + Values + Constraints | Prediction is just input; Decision is the output. |
| **Strategy** | "Where to play & How to win" | Long-term, Differentiation. (Tactics = Short-term execution). |
| **Cost Leadership** | Win by being cheapest | Analytics focus: Efficiency, Supply Chain, Automation. |
| **Differentiation** | Win by being unique | Analytics focus: Personalization, UX, Sentiment. |

---

## 🔢 2. Evaluation Metrics (The Confusion Matrix)
**Actual (Rows) vs Predicted (Cols)**
- **Accuracy:** $(TP+TN)/Total$. *Trap:* Useless if classes imbalanced.
- **Precision:** $TP / (TP+FP)$. *Use when:* Cost of False Positive is High (e.g., Spam).
- **Recall (Sensitivity):** $TP / (TP+FN)$. *Use when:* Cost of False Negative is High (e.g., Cancer).
- **F1 Score:** Harmonic Mean of Prec & Recall. *Use when:* Balance needed & Imbalanced classes.
- **ROC-AUC:** Area Under Curve. *Use when:* Need robust comparison independent of threshold. 0.5 = Random.

---

## 🎲 3. Probability & Evidence
- **Frequentist:** Prob = Long-run frequency. Fixed parameters.
- **Bayesian:** Prob = Degree of Belief. Parameters are random variables.
- **Bayes Rule:** $P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$
    - **Posterior:** New Belief.
    - **Likelihood ($P(E|H)$):** How well H explains E.
    - **Prior ($P(H)$):** Old Belief (Base Rate).
    - **Trap:** Ignoring Prior = **Base Rate Neglect**.
- **Independence:** $P(A,B) = P(A)P(B)$. Essential for Naive Bayes.
- **P-Value:** Prob of DATA given Null Hypothesis. $\neq$ Prob of Null Hypothesis.

---

## 🔍 4. Factor Analysis vs PCA
| Feature | PCA (Principal Component Analysis) | FA (Factor Analysis) |
| :--- | :--- | :--- |
| **Goal** | **Data Reduction** (Summarize) | **Structure Detection** (Explain) |
| **Assumption** | Components = Weighted Sum of Variables | Latent Factors *Cause* Variables |
| **Variance** | Analyzes **Total** Variance | Analyzes **Common** Variance only |
| **Solution** | Unique solution | Indeterminate (Needs Rotation) |
| **Use Case** | Pre-processing, Compression | Psychology, Theory validation |

---

## ⚠️ 5. Top 5 "Exam Traps" to Avoid
1.  **Correlation $\neq$ Causation:** $X$ correlates with $Y$ could mean $Z$ causes both (Spurious).
2.  **Statistical $\neq$ Practical Significance:** With $N=1,000,000$, a correlation of $0.01$ is "Significant" ($p<0.05$) but useless.
3.  **Confusion of Inverse:** $P(\text{Positive}|\text{Disease}) \approx 99\%$, but $P(\text{Disease}|\text{Positive})$ can be $<1\%$ if disease is rare.
4.  **Operational $\neq$ Strategic:** Making a warehouse 10% faster is operational. Switching to Drop-shipping is strategic.
5.  **Accuracy Paradox:** A "dumb" model predicting "No Fraud" gets 99.9% accuracy but 0 value. Always check baseline.

---
**Memorize for Exam:**
- **Evidence** must be *informative* (change posterior).
- **Baseline** is the minimum bar; beating it > absolute accuracy.
- **Directional Data:** Mean of $1^\circ$ & $359^\circ$ is $0^\circ$ (North), NOT $180^\circ$.
