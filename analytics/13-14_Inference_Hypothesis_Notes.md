# Inference & Hypothesis Testing – Theory‑Rich Exam Notes

---

## SECTION A – Inference Framework

### 1. Decision‑Making Under Uncertainty
> **Why?**  In statistical inference we never know the true population parameters; we must **act** based on imperfect information.  The inference framework formalizes a *decision‑theoretic* process where actions are chosen to minimize expected loss.

- **Key Concepts**
  - **Parameter of interest** (θ) – unknown quantity we wish to estimate or test.
  - **Estimator / Test statistic** – a function of the sample that provides evidence about θ.
  - **Loss function** – quantifies the cost of making a wrong decision (e.g., squared error loss for estimation, 0‑1 loss for hypothesis decisions).
- **Assumptions**: The sampling design is known; the loss function reflects the real‑world stakes.
- **Edge Cases**: When loss is asymmetric (e.g., false‑positive cost >> false‑negative), the optimal decision rule may differ from the usual *α = 0.05* convention.

### 2. Role of Probability in Inference
- **Probability as a Model of Uncertainty** – Provides the *sampling distribution* of estimators, enabling calculation of **risk** (expected loss) and **confidence**.
- **Frequentist View**: Parameters are fixed; probability describes the long‑run behavior of the estimator under repeated sampling.
- **Bayesian View** (brief): Parameters are random with a prior distribution; inference updates this via the posterior, integrating probability directly into the parameter itself.
- **Common Misconception**: "A p‑value is the probability that the null hypothesis is true." – **FALSE**; it is the probability of observing data **as extreme or more extreme** under the null.

> **MCQ ALERT**: *“In a decision‑theoretic framework, the optimal rule minimizes the expected loss, not the Type I error rate alone.”* – **Correct**.

---

## SECTION B – Hypothesis Testing

### 1. Null Hypothesis Philosophy
- **Null (H₀)**: Represents the status‑quo or a specific value of θ (often a point hypothesis, e.g., μ = μ₀).  It is the *default* that must be disproved.
- **Alternative (H₁ or Hₐ)**: Encodes the scientific claim of interest (e.g., μ ≠ μ₀, μ > μ₀).
- **Why a Null?**: Provides a concrete probability model against which we can compute the sampling distribution of the test statistic.
- **Assumptions**: The null model is correctly specified (distributional form, variance, independence).
- **Edge Cases**: Composite nulls (e.g., μ ≤ μ₀) require *supremum* of the null distribution; misuse leads to conservative tests.

### 2. Error Trade‑offs (Type I vs. Type II)
| Error Type | Symbol | Definition | Typical Cost | Relationship to α / β |
|------------|--------|------------|--------------|-----------------------|
| **Type I** | α | Reject H₀ when it is true | False alarm; may lead to unnecessary action | Directly set by researcher (commonly 0.05) |
| **Type II** | β | Fail to reject H₀ when H₁ is true | Missed discovery; opportunity loss | Determined by power = 1‑β; depends on effect size, α, and sample size |

- **Power Analysis**: Prior to data collection, compute required *n* to achieve desired power (e.g., 0.80) for a given effect size.
- **Assumptions**: Correct specification of effect size and variance; independence of observations.
- **Edge Cases**: Very small α (e.g., 0.001) dramatically inflates β unless sample size is increased.

### 3. p‑value Interpretation
- **Definition**: \(p = P(T \ge t_{obs} \mid H₀)\) for a right‑tailed test (or appropriate tail). It is a *tail probability* under the null.
- **What it tells you**:
  1. How surprising the observed data are *if* H₀ were true.
  2. Not the probability that H₀ is true or false.
- **Common Misinterpretations**:
  - "A p‑value of 0.03 means there is a 97 % chance the alternative is true." – **FALSE**.
  - "If p > 0.05, the effect does not exist." – **FALSE**; may be under‑powered.
- **Reporting Guidelines** (APA, Indian Statistical Institute):
  - Report exact p‑value (e.g., p = 0.032) unless <0.001.
  - Provide effect size and confidence interval alongside.

> **MCQ ALERT**: *“A p‑value of 0.20 automatically implies the null hypothesis is true.”* – **Incorrect**.

---

## SECTION C – Parametric vs. Non‑Parametric Tests

### 1. Why Assumptions Matter
- **Parametric Tests** (t‑test, ANOVA, linear regression) assume a *specific distributional form* (often Normality) and sometimes *homoscedasticity* (equal variances).
  - **Benefit**: Greater statistical power when assumptions hold because the test statistic’s sampling distribution is known exactly.
  - **Risk**: Violation leads to inflated Type I error or loss of power.
- **Non‑Parametric Tests** (Wilcoxon, Mann‑Whitney, Kruskal‑Wallis) make **fewer** assumptions—typically only that the data are at least ordinal and independent.
  - **Benefit**: Robust to skewness, outliers, and heteroscedasticity.
  - **Cost**: Generally lower power when the parametric assumptions *are* satisfied (the test uses only rank information).

### 2. Cost of Assumption Violations
| Scenario | Parametric Test Consequence | Non‑Parametric Alternative | Relative Power (Assumptions Hold) |
|----------|----------------------------|----------------------------|-----------------------------------|
| Heavy‑tailed distribution | Inflated Type I error; confidence intervals too narrow | Mann‑Whitney U (two‑sample) | Non‑parametric ≈ 0.8× parametric |
| Heteroscedastic variances | t‑test may be liberal (α inflated) | Welch’s t‑test (still parametric) or rank‑based test | Welch retains power; rank‑based loses some |
| Ordinal data (e.g., Likert) | t‑test inappropriate (treats as interval) | Wilcoxon signed‑rank | Non‑parametric is the only valid choice |
| Small sample (n < 15) with unknown distribution | Normal approximation unreliable | Exact permutation test | Non‑parametric may be more accurate |

- **Assumption‑checking workflow**:
  1. **Normality** – Shapiro‑Wilk test, Q‑Q plot.
  2. **Equal variances** – Levene’s test, Bartlett’s test.
  3. **Independence** – Study design (randomization, no repeated measures).
  4. **If any test fails → consider a non‑parametric alternative or transform data (log, sqrt).**

> **MCQ ALERT**: *“If the data are skewed, the two‑sample t‑test is always preferable to the Mann‑Whitney test.”* – **Incorrect**; Mann‑Whitney is more reliable under skewness.

---

## MCQ‑Focused Traps & Guidance

### Type I vs. Type II Errors
- **Common Trap**: Confusing *α* with the probability that the null is true.  Emphasize that α is a *pre‑specified* bound on the Type I error rate, not a posterior probability.
- **Test Selection Pitfall**: Choosing a test solely because it yields a *smaller* p‑value without checking assumptions; this may inflate Type I error.

### Test Selection Traps
| Situation | Incorrect Choice | Correct Reasoning |
|-----------|------------------|-------------------|
| Small, skewed sample, comparing medians | Paired t‑test | Data are not Normal; use Wilcoxon signed‑rank. |
| Comparing more than two groups with unequal variances | One‑way ANOVA (classic) | Violates homoscedasticity; use Welch ANOVA or Kruskal‑Wallis. |
| Binary outcome, large sample, but model assumes Normal errors | Linear regression | Outcome is binary; use logistic regression (GLM with binomial link). |
| Testing a proportion with expected count < 5 | χ² test | Expected frequencies too low; use exact binomial test. |

---

## QUICK‑LOOK DECISION TABLE

| Goal | Recommended Test (Parametric) | Recommended Test (Non‑Parametric) | Key Assumption to Verify |
|------|-------------------------------|-----------------------------------|--------------------------|
| Compare two independent means | Independent two‑sample t‑test (equal variances) | Mann‑Whitney U | Normality, equal variances |
| Compare two related samples | Paired t‑test | Wilcoxon signed‑rank | Normality of differences |
| Compare >2 group means | One‑way ANOVA | Kruskal‑Wallis | Normality, homoscedasticity |
| Correlation (linear) | Pearson r | Spearman ρ | Bivariate Normality |
| Regression with continuous outcome | Linear regression | Rank‑based regression (e.g., Theil‑Sen) | Linear relationship, Normal errors |

---

> **NOTE TO STUDENTS**: Always start by *checking assumptions*; only after confirming them should you proceed with a parametric test. If assumptions fail, switch to the appropriate non‑parametric alternative or transform the data.

---

*Prepared for PG‑DBDA (ACTS, Pune) theory‑dominant examinations.*
