# 📚 Theory‑Intensive Notes – Evidence, Probability as Belief, and Evidence Aggregation (PG‑DBDA, ACTS, Pune)

---
## SECTION A: Nature of Evidence

### 1️⃣ What qualifies as **evidence** in analytics
- **Why it matters:** Evidence is the *justifiable support* for a claim or decision. In analytics we must distinguish between raw observations and the *interpreted* material that can influence a conclusion.
- **Formal definition:** A piece of information that, when combined with a prior belief, *updates* that belief via a probabilistic rule (Bayes’ theorem). It must be:
  1. **Relevant** – directly related to the hypothesis.
  2. **Reliable** – obtained through a trustworthy process (measurement protocol, sampling design).
  3. **Sufficient** – contains enough information to affect the posterior distribution.
- **Typical sources:** survey responses, sensor readings, transaction logs, expert judgments, published studies.

### 2️⃣ Evidence vs **data** vs **information**
| Concept | Essence | Role in analytics |
|---------|--------|-------------------|
| **Data** | Raw, unprocessed symbols (numbers, strings). | Input to preprocessing; may contain noise, bias, or irrelevant fields.
| **Information** | Data that has been *organized* and *contextualized* (e.g., descriptive statistics). | Provides a *summary* that can be communicated; still not a claim.
| **Evidence** | Information that *supports* or *refutes* a specific hypothesis, typically quantified via likelihood or probability. | Drives *inference* and *decision*; used in hypothesis testing, Bayesian updating, and model validation.

### 3️⃣ Uncertainty and **incompleteness** of evidence
- **Uncertainty** arises from measurement error, sampling variability, and model misspecification. It is captured by **probability distributions** (e.g., confidence intervals, posterior variance).
- **Incompleteness** means the evidence does *not* fully determine the truth; there remains residual ambiguity.
- **Key intuition:** Even a large dataset can be *incomplete* if it lacks variables that are causally relevant (omitted variable bias).
- **Common exam traps:** Assuming that more data automatically reduces uncertainty without considering systematic bias.

> **MCQ ALERT** – *Which of the following statements about evidence is **false**?*
> A) Evidence must be relevant to the hypothesis it supports.
> B) All data automatically qualify as evidence.
> C) Evidence can be quantified using probability.
> D) Incomplete evidence leads to residual uncertainty.

---
## SECTION B: Probabilities as Degree of Belief

### 1️⃣ Frequentist vs Subjective (Bayesian) probability intuition
| Perspective | Core intuition | Typical use case |
|------------|----------------|-----------------|
| **Frequentist** | Probability = long‑run relative frequency of an event under repeated identical trials. | Confidence intervals, hypothesis testing.
| **Subjective (Bayesian)** | Probability = personal degree of belief, updated via Bayes’ theorem when new evidence arrives. | Prior‑posterior updating, decision analysis under uncertainty.

- **Why probabilities quantify evidence strength:**
  1. **Additivity:** Allows combination of independent pieces of evidence (product of likelihoods).
  2. **Normalization:** Guarantees a coherent scale (0–1) for comparing alternative hypotheses.
  3. **Decision‑theoretic link:** Expected utility maximisation requires probabilities to weight outcomes.
- **Misconception trap:** Treating a *p‑value* as the probability that the null hypothesis is true (frequentist vs Bayesian confusion).

### 2️⃣ Why probability is used to *measure* evidence strength
- **Likelihood principle:** The probability of observed data given a hypothesis (likelihood) directly reflects how well that hypothesis explains the evidence.
- **Bayes factor:** Ratio of likelihoods for two competing hypotheses; a natural *evidence strength* metric.
- **Interpretation:** A Bayes factor of 10 means the data are ten times more likely under hypothesis H₁ than H₀ – a clear, intuitive statement of evidential support.

> **MCQ ALERT** – *Which statement correctly captures the Bayesian view of probability?*
> A) It is the long‑run frequency of an event.
> B) It is a subjective degree of belief that can be updated with evidence.
> C) It is always equal to the p‑value of a test.
> D) It never changes once assigned.

---
## SECTION C: Evidence Aggregation

### 1️⃣ Why a **single piece of evidence** is rarely sufficient
- **Information content:** One datum often carries limited *entropy*; multiple independent pieces increase total information (Shannon’s additive property).
- **Robustness:** Aggregating reduces the impact of outliers or measurement error.
- **Decision confidence:** Higher cumulative evidence lowers posterior uncertainty, enabling stronger actions.

### 2️⃣ Handling **conflicting evidence**
| Conflict type | Intuition | Formal handling |
|---------------|-----------|-----------------|
| **Contradictory likelihoods** | Two sources suggest opposite directions. | Multiply likelihoods; the posterior reflects the *balance* of evidence (may remain ambiguous). |
| **Dependent evidence** | Sources share common information (e.g., same sensor). | Adjust for **conditional dependence** using joint probability or a hierarchical model to avoid double‑counting. |
| **Qualitative vs quantitative** | Expert opinion vs measured data. | Encode qualitative input as a prior or as a likelihood with appropriate variance.

- **Key pitfall:** Treating dependent evidence as independent inflates confidence (over‑confidence paradox).

### 3️⃣ Conditional **dependence** and **independence**
- **Independence**: \(P(A\cap B) = P(A)P(B)\). Allows simple multiplication of likelihoods.
- **Conditional independence**: \(P(A\cap B\mid C) = P(A\mid C)P(B\mid C)\). Often assumed in Naïve Bayes classifiers.
- **Dependence**: Requires joint modeling; e.g., using a multivariate normal with covariance matrix \(\Sigma\) to capture correlation.
- **Exam trap:** Assuming independence when evidence originates from the same underlying process (e.g., repeated measurements from the same instrument).

> **MCQ ALERT** – *If two pieces of evidence are conditionally independent given hypothesis H, how should their combined likelihood be computed?*
> A) Add the two likelihoods.
> B) Multiply the two likelihoods.
> C) Take the maximum of the two.
> D) Divide one by the other.

---
## 📊 Logical Reasoning Examples Converted into MCQs
1. **Scenario:** A medical test reports a positive result with 95 % sensitivity and 90 % specificity. The disease prevalence is 1 %.
   - **Question:** What is the *posterior probability* that a randomly selected individual who tests positive actually has the disease?
   - **Explanation:** Apply Bayes’ theorem; the answer is ≈ 9 % – a classic *evidence strength vs probability* trap where high test accuracy does not imply high post‑test probability.

2. **Scenario:** Two independent surveys report the same proportion (60 %) of customers preferring product A.
   - **Question:** How does the combined evidence affect the confidence interval for the true preference proportion?
   - **Explanation:** Sample size effectively doubles, narrowing the interval by \(\sqrt{2}\); demonstrates aggregation of independent evidence.

3. **Scenario:** An analyst uses two sensor readings that are highly correlated (ρ = 0.9) to estimate a temperature.
   - **Question:** Treating them as independent will most likely **under‑** or **over‑** estimate the uncertainty?
   - **Explanation:** Over‑estimate certainty (under‑estimate variance) because dependence is ignored.

4. **Scenario:** Expert A assigns a prior probability of 0.7 to hypothesis H, while Expert B assigns 0.3. Both are equally credible.
   **Question:** What is a reasonable aggregated prior for H?
   - **Explanation:** A weighted average (0.5) or a hierarchical Bayesian model; illustrates handling conflicting evidence.

---
## 📚 Further Reading (concise list)
- *The Logic of Scientific Discovery* – Karl Popper (on evidence and falsifiability).
- *Probability Theory: The Logic of Science* – E.T. Jaynes (Bayesian interpretation of evidence).
- *Bayesian Data Analysis* – Gelman et al. (chapters on prior‑likelihood‑posterior framework).
- *Statistical Evidence and the Law* – B. Berger (evidence aggregation).
- *Information Theory, Inference, and Learning Algorithms* – David MacKay (entropy, evidence combination).

---
*End of notes.*
