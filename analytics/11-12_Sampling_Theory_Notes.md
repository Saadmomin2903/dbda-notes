# Sampling Theory – Rigorous Exam‑Focused Notes

---

## SECTION A – Why Sampling Is Necessary

### 1. Cost, Feasibility, and Time Constraints
> **Why?**  In most real‑world studies the *population* (all possible observations) is either **too large**, **inaccessible**, or **expensive** to enumerate completely.  Sampling provides a **tractable surrogate** that allows us to estimate population characteristics while respecting resource limits.

- **Cost** – Measuring every unit (e.g., every household in a country) would require prohibitive financial outlay.
- **Feasibility** – Some populations are physically unreachable (deep‑sea organisms, historical events).
- **Time** – Data collection may need to be completed before a decision deadline; a full census could take years.
- **Assumptions** implicit in using a sample: the sample must be *representative* of the target population for inference to be valid.
- **Edge Cases** – When the population size is tiny (e.g., <30 units), a census is often preferable; sampling adds unnecessary sampling error.

### 2. Population Inference Philosophy
- **Goal**: Use the sample to make **probabilistic statements** about the *entire* population (e.g., confidence intervals, hypothesis tests).
- **Frequentist View**: The population parameters are fixed but unknown; the sampling distribution of an estimator quantifies its variability.
- **Bayesian View** (brief): Treat parameters as random variables with prior distributions; the sample updates beliefs via the posterior.
- **Key Principle** – *Design‑based inference*: All randomness stems from the sampling design, not from the underlying population.
- **Common Misconception**: "A sample *is* the population." – **FALSE**; a sample provides information *about* the population, not the population itself.

> **MCQ ALERT**: *“If a sample is drawn without replacement from a finite population, the sample mean is an unbiased estimator of the population mean.”* – **Correct**, provided the sampling is **simple random** (each subset of size *n* equally likely).

---

## SECTION B – Sampling Techniques

### 1. Random vs. Biased (Non‑random) Sampling
- **Random Sampling** – Every unit has a known, non‑zero probability of selection.  Guarantees that the sampling distribution can be derived analytically.
  - *Simple Random Sampling (SRS)*: All \(\binom{N}{n}\) subsets equally likely.
  - *Systematic Sampling*: Select every *k*‑th unit after a random start – still random if the ordering is unrelated to the variable of interest.
- **Biased Sampling** – Selection probabilities depend on the unit’s characteristics (e.g., convenience, voluntary response).  Leads to **selection bias**, invalidating inference.
- **Assumptions for Random Sampling**: Independence of selections (or known dependence in stratified/cluster designs) and known inclusion probabilities.
- **Edge Cases**: In *cluster sampling*, units within a cluster are correlated; variance estimation must account for intra‑cluster correlation.

### 2. Univariate vs. Bivariate (Multivariate) Logic
- **Univariate Sampling** – Focuses on a single variable of interest (e.g., average income).  Sample size calculations use the variance of that variable alone.
- **Bivariate/Multivariate Sampling** – Simultaneously estimates relationships between two or more variables (e.g., correlation, regression coefficients).
  - **Joint Inclusion Probabilities** become crucial; designs like *stratified sampling* improve precision for multiple variables by allocating samples proportionally to strata variance.
  - **Assumption**: The joint distribution of variables is adequately captured by the sample; otherwise, multivariate estimates may be biased.
- **Common Trap**: Using a sample size derived for a *single* variable when the exam asks about *multiple* outcomes – leads to under‑powered inference.

### 3. Re‑sampling Intuition (Bootstrap & Jackknife)
- **Why Re‑sampling?**  When analytical formulas for the sampling distribution are unavailable (complex estimators, small samples), we approximate it by repeatedly drawing *pseudo‑samples* from the observed data.
- **Bootstrap** – Sample *with replacement* from the original data many times (e.g., 10,000 replicates) and compute the estimator each time.
  - Provides empirical standard errors, confidence intervals, bias estimates.
  - **Assumption**: The observed sample is a reasonable proxy for the population (i.e., the empirical distribution approximates the true distribution).
- **Jackknife** – Systematically leave‑out one observation (or groups) and recompute the estimator; useful for bias reduction.
- **Edge Cases**: For highly dependent data (time series), naïve bootstrap breaks the dependence structure; *block bootstrap* is required.

> **MCQ ALERT**: *“Bootstrap confidence intervals are always narrower than analytical intervals.”* – **Incorrect**; they can be wider, especially with small samples or heavy‑tailed data.

---

## SECTION C – Central Limit Theorem (CLT)

### 1. Why the CLT Is Powerful
- **Core Insight**: *Regardless of the original distribution* of i.i.d. observations, the **distribution of the sample mean** (properly scaled) approaches a **Normal** distribution as the sample size grows.
- **Practical Consequence**: Enables *approximate* inference (confidence intervals, hypothesis tests) even when the population is non‑Normal.
- **Historical Note**: First proved in a limited form by **Laplace (1810)**; later generalized by **Lyapunov** and **Lindeberg** (1900s) with weaker conditions.

### 2. What the CLT *Does* and *Does NOT* Say
| Statement | True? | Explanation |
|-----------|-------|-------------|
| The *sample mean* of i.i.d. variables converges in distribution to \(\mathcal{N}(\mu,\sigma^{2}/n)\). | ✅ | Classic CLT (Lindeberg‑Levy) for finite variance. |
| The *sample median* is also asymptotically Normal. | ✅ (with different variance) | A separate theorem; CLT does not directly guarantee this for the median. |
| The *raw data* become Normal as \(n\) increases. | ❌ | Only the **mean** (or sum) becomes Normal; individual observations retain their original distribution. |
| Any *non‑i.i.d.* sequence satisfies the CLT. | ❌ | Independence (or weak dependence) and identical distribution (or Lindeberg condition) are required. |
| The CLT provides an *exact* Normal distribution for any finite \(n\). | ❌ | It is an *asymptotic* result; for small \(n\) the approximation may be poor, especially with heavy tails. |

### 3. Role of Sample Size
- **Rule of Thumb**: For *moderately* symmetric distributions, \(n\ge30\) often yields a decent Normal approximation.  For *highly skewed* or *heavy‑tailed* distributions, a much larger \(n\) (e.g., >100) may be needed.
- **Berry‑Esseen Theorem** quantifies the convergence rate: the maximum difference between the true distribution of the standardized mean and the Normal CDF is bounded by \(C\frac{\rho}{\sigma^{3}\sqrt{n}}\), where \(\rho\) is the third absolute central moment.
- **Edge Cases**: If the underlying variance is infinite (e.g., Cauchy distribution), the CLT **does not apply**; the sum does not converge to Normal.

> **MCQ ALERT**: *“A sample of size 20 from an exponential distribution can be treated as Normal for constructing a 95% confidence interval for the mean.”* – **Potentially incorrect**; the exponential is skewed, and \(n=20\) may be insufficient for a reliable Normal approximation.

---

## MCQ‑Focused Traps & Guidance

### Sampling vs. Population Confusion
1. **Mean of Sample ≠ Mean of Population** – Only *expected* value of the sample mean equals the population mean under unbiased sampling.
2. **Sampling Error vs. Measurement Error** – Sampling error arises from using a subset; measurement error stems from inaccurate data collection.
3. **Finite Population Correction (FPC)** – When sampling *without* replacement from a finite population, the variance of the sample mean is multiplied by \((N-n)/(N-1)\). Ignoring FPC inflates the estimated variance.

### CLT Misuse in Exams
- **Assuming Normality for Small n** – Many students apply Z‑tests with \(n<30\) regardless of distribution shape.
- **Applying CLT to Proportions without Checking np and n(1‑p)** – The normal approximation for a proportion \(\hat{p}\) requires both \(np\) and \(n(1-p)\) ≥ 5 (or 10 for stricter criteria).
- **Treating the Sample Median as Normally Distributed without Adjusting Variance** – The variance of the median depends on the underlying density at the median; using \(\sigma/\sqrt{n}\) is wrong.

---

## QUICK‑LOOK DECISION TABLE

| Situation | Recommended Sampling Design | Key Assumption | Typical Sample Size Guideline |
|-----------|----------------------------|----------------|------------------------------|
| Large, homogeneous population, single variable | Simple Random Sampling | Independence, equal inclusion probability | \(n\ge30\) for CLT‑based mean inference |
| Heterogeneous population (known strata) | Stratified Sampling | Within‑stratum homogeneity | Allocate proportionally; ensure each stratum \(n_h\ge30\) if using CLT per stratum |
| Costly field measurement, clusters natural (e.g., schools) | Cluster Sampling | Intra‑cluster correlation accounted for | Larger overall \(n\) to offset design effect (often \(DEFF>1\)) |
| Need to assess estimator variability without analytic formula | Bootstrap / Jackknife | Sample is representative of population | \(B\) replicates (e.g., 10,000) – computational cost considered |
| Estimating proportion with rare event (p≈0.01) | Oversample or use *Poisson* approximation | np ≥ 5 for Normal approx; otherwise exact binomial

---

> **NOTE TO STUDENTS**: Always verify the **sampling design assumptions** before invoking the CLT or any Normal‑based inference. Mis‑matching design and inference method is the most frequent source of MCQ errors.

---

*Prepared for PG‑DBDA (ACTS, Pune) theory‑dominant examinations.*
