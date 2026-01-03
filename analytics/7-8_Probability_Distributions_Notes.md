# Probability Distributions – Theory‑Focused Exam Notes

---

## SECTION A – WHY PROBABILITY DISTRIBUTIONS EXIST

### 1. Modeling Uncertainty Mathematically
> **Why?**  Real‑world phenomena are rarely deterministic; outcomes vary due to hidden factors, measurement error, or inherent randomness.  A probability distribution provides a **formal language** to quantify this variability, enabling rigorous reasoning, inference, and decision‑making.

- **Intuition**: Think of a distribution as a *landscape* that tells you how likely you are to land on each point of the outcome space.
- **Assumptions**: The underlying process can be abstracted as a random variable (RV) with a well‑defined sample space.
- **Limitations**: If the process violates the assumptions (e.g., non‑stationary, dependent on unobserved variables), the chosen distribution may mis‑represent reality.
- **Edge Cases**: Degenerate distributions (all probability mass at a single point) – useful for deterministic limits.

### 2. Discrete vs Continuous World‑view
| Aspect | Discrete RV | Continuous RV |
|--------|-------------|---------------|
| **Support** | Countable set (e.g., {0,1,2,…}) | Uncountable interval (e.g., \([0,\infty)\)) |
| **Probability** | Mass function \(P(X=x)\) sums to 1 | Density function \(f(x)\) integrates to 1 |
| **Typical Applications** | Number of successes, arrivals, failures | Measurement of time, length, weight |
| **Common Misconception** | "A continuous variable can have a probability at a single point" – **FALSE** (probability is zero). |

> **MCQ ALERT**: *“If \(X\) is continuous, \(P(X=5) = 0.2\) is possible.”* – **Incorrect**; the probability of any exact value for a continuous RV is zero.

---

## SECTION B – DISCRETE DISTRIBUTIONS

### 1. Binomial Distribution \(X \sim \text{Bin}(n,p)\)
- **Why it exists**: Models the number of *successes* in a fixed number of *independent* trials, each with the same success probability \(p\).
- **Assumptions**
  1. Fixed number of trials \(n\).
  2. Trials are independent.
  3. Success probability \(p\) is constant across trials.
- **Probability Mass Function (PMF)**
  \[P(X=k)=\binom{n}{k}p^{k}(1-p)^{n-k},\quad k=0,1,…,n\]
- **Intuition**: Imagine flipping a fair coin \(n\) times; the binomial tells you the chance of getting exactly \(k\) heads.
- **Limitations & Edge Cases**
  - When \(p\) is very small and \(n\) large, the Binomial becomes computationally unwieldy; approximation (Poisson) is preferred.
  - If trials are *not* independent (e.g., sampling without replacement), the **Hypergeometric** distribution is appropriate instead.
- **Common Exam Trap**: Confusing *order* with *combination* – the binomial counts *combinations*; the term \(\binom{n}{k}\) already accounts for ordering.

### 2. Poisson Distribution \(X \sim \text{Pois}(\lambda)\)
- **Why it exists**: Captures the count of *rare* events occurring in a fixed interval of time/space when events happen **independently** at a constant average rate \(\lambda\).
- **Assumptions**
  1. Events occur one at a time.
  2. The average rate \(\lambda\) is constant.
  3. Occurrences in disjoint intervals are independent.
- **PMF**
  \[P(X=k)=\frac{e^{-\lambda}\lambda^{k}}{k!},\quad k=0,1,2,…\]
- **Intuition**: Think of the number of emails you receive in an hour when the average is \(\lambda=5\).
- **Limiting Relationship**: \(\text{Bin}(n,p) \approx \text{Pois}(\lambda=np)\) when \(n\to\infty\) and \(p\to0\) such that \(np=\lambda\) stays finite.
- **Edge Cases**
  - \(\lambda=0\) yields a degenerate distribution at 0.
  - For very large \(\lambda\), the distribution becomes approximately Normal (by CLT).
- **MCQ ALERT**: *“Poisson can model the number of heads in 10 coin flips.”* – **Incorrect**; the underlying process is not “rare” and the number of trials is fixed, so Binomial is appropriate.

### 3. Geometric Distribution \(X \sim \text{Geom}(p)\)
- **Why it exists**: Models the *waiting time* (number of trials) until the **first success** in a sequence of independent Bernoulli trials with success probability \(p\).
- **PMF (support \(k=1,2,…\))**
  \[P(X=k) = (1-p)^{k-1}p\]
- **Mean & Variance**
  \[E[X]=\frac{1}{p},\quad \text{Var}(X)=\frac{1-p}{p^{2}}\]
- **Intuition**: Number of dice rolls needed to get the first six.
- **Assumptions**: Same as Binomial – independent trials, constant \(p\).
- **Edge Cases**
  - As \(p\to1\), the distribution collapses to \(X=1\).
  - As \(p\to0\), the mean explodes → infinite expected waiting time.
- **Common Misconception**: Confusing *Geometric* with *Negative Binomial* (which counts trials until *r* successes). The geometric is the special case with \(r=1\).

---

## SECTION C – CONTINUOUS DISTRIBUTIONS

### 1. Uniform Distribution \(X \sim \text{U}(a,b)\)
- **Why it exists**: Represents a situation where **every outcome in an interval** \([a,b]\) is *equally likely* – the principle of insufficient reason.
- **PDF**
  \[f(x)=\frac{1}{b-a},\quad a\le x \le b\]
- **Common Misinterpretation**: "Uniform means the *probability* of each point is \(\frac{1}{b-a}\)." – **FALSE**; for continuous RVs the probability of any *single* point is zero; the density is constant.
- **Assumptions**: The underlying process truly has no bias toward any sub‑interval.
- **Edge Cases**
  - When \(a=b\) the distribution becomes degenerate at a point.
  - If the interval is unbounded, the uniform cannot be defined (integral diverges).
- **MCQ ALERT**: *“The probability that a Uniform(0,1) variable equals 0.5 is 0.5.”* – **Incorrect**; probability at a single point is zero.

### 2. Exponential Distribution \(X \sim \text{Exp}(\lambda)\)
- **Why it exists**: Models the *time between* successive *memoryless* events occurring at a constant rate \(\lambda\).
- **PDF & CDF**
  \[f(x)=\lambda e^{-\lambda x},\; x\ge0\]
  \[F(x)=1-e^{-\lambda x}\]
- **Memoryless Property**
  \[P(X> s+t \mid X> s) = P(X> t)\]
  *Only the exponential (and geometric) satisfy this property.*
- **Intuition**: Lifetime of a radioactive atom, or waiting time for the next bus when buses arrive at a constant average rate.
- **Assumptions**: Events occur independently and at a constant average rate.
- **Limitations**: Real‑world waiting times often exhibit *over‑dispersion* (e.g., bus bunching) – then a **Weibull** or **Gamma** may be more appropriate.
- **Edge Cases**
  - As \(\lambda\to0\), the distribution spreads out (mean → ∞).
  - As \(\lambda\to\infty\), the distribution collapses near 0.
- **Common Trap**: Assuming *exponential* can model *bounded* waiting times – it is defined on \([0,\infty)\) only.

### 3. Normal Distribution \(X \sim \mathcal{N}(\mu,\sigma^{2})\)
- **Why it exists**: Central Limit Theorem (CLT) shows that the sum (or average) of many independent, identically distributed random variables tends toward a Normal shape, regardless of the original distribution.
- **PDF**
  \[f(x)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\!\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)\]
- **Intuition**: Height of adult humans, measurement errors, and many natural phenomena cluster around a mean with symmetric spread.
- **Assumptions**: Underlying variables are independent, have finite variance, and are *identically* distributed (or at least satisfy Lindeberg’s condition).
- **Limitations & Edge Cases**
  - Heavy‑tailed data (e.g., income) violate the finite‑variance assumption → Normal may underestimate extreme events.
  - For bounded data, a **Beta** distribution may be more appropriate.
- **Historical Motivation**: First derived by Gauss (method of least squares) and Laplace (error theory) in the early 19th century.
- **MCQ ALERT**: *“All real‑world data are approximately Normal.”* – **Incorrect**; many datasets are skewed or have heavy tails.

---

## SECTION D – DISTRIBUTION SELECTION LOGIC

| Real‑World Scenario | Primary Candidate(s) | Reasoning (Why) | Pitfall (What NOT to choose) |
|----------------------|----------------------|-----------------|------------------------------|
| Number of defective items in a batch of size 200, defect probability 0.02 | **Binomial** (n=200, p=0.02) | Fixed number of independent trials, constant p | **Poisson** – only if n is huge and p tiny; here n is moderate.
| Calls arriving at a call‑center per hour (average 30) | **Poisson** (λ=30) | Rare events, independent arrivals, constant rate | **Binomial** – would require an artificial upper bound on trials.
| Time until next earthquake (rare, memoryless) | **Exponential** (λ based on historical rate) | Memoryless property, continuous waiting time | **Uniform** – would imply bounded waiting time, unrealistic.
| Height of adult males in a city | **Normal** (μ≈170 cm, σ≈7 cm) | CLT justification, symmetric spread | **Uniform** – would imply equal likelihood of extreme heights.
| Number of attempts needed to get first ‘6’ on a die | **Geometric** (p=1/6) | Waiting‑time for first success, independent trials | **Negative Binomial** – would be over‑kill (counts until r>1 successes).

### Consequences of Wrong Distribution Choice
1. **Bias in Parameter Estimates** – e.g., using Normal for count data yields negative probabilities.
2. **Incorrect Confidence Intervals** – variance formulas differ; using Binomial variance for Poisson data underestimates variability.
3. **Misleading Hypothesis Tests** – test statistics assume a specific distribution; violation inflates Type I/II errors.
4. **Poor Predictive Performance** – model fit deteriorates, leading to inaccurate forecasts.

---

## MCQ‑STYLE TRAPS & QUICK‑LOOK DECISION TABLE

| # | Trap Description | Correct Reasoning |
|---|-------------------|-------------------|
| 1 | "A continuous RV can have a non‑zero probability at a single point." | For continuous RVs, \(P(X=x)=0\) for any exact \(x\). |
| 2 | "Binomial and Poisson are interchangeable for any n and p." | Poisson is only an approximation when \(n\) large, \(p\) small, \(np=\lambda\). |
| 3 | "Mean = Variance implies a Poisson distribution." | Many distributions (e.g., Negative Binomial) can have equal mean and variance under specific parameters. |
| 4 | "Exponential can model waiting times that are bounded above." | Exponential support is \([0,\infty)\); bounded waiting times need truncated or other distributions. |
| 5 | "Uniform(0,1) has variance 0.5." | Variance of Uniform(a,b) is \((b-a)^{2}/12\); for (0,1) it is \(1/12≈0.0833\). |
| 6 | "Geometric counts the number of failures before the first success." | Depends on definition; some textbooks define support \{0,1,2,…\) for failures. Clarify the convention used. |
| 7 | "Normal distribution can model skewed data because of its flexibility." | Normal is symmetric; skewed data require transformations or asymmetric distributions. |
| 8 | "If events occur at a constant rate, the inter‑arrival times must be exponential." | True only if arrivals are independent (Poisson process). Correlated arrivals break memorylessness. |
| 9 | "A degenerate distribution is a special case of Uniform." | Degenerate is a point mass; Uniform requires an interval of positive length. |
|10|"Using a Poisson model for over‑dispersed count data is fine."| Over‑dispersion (variance > mean) violates Poisson’s equal mean‑variance property; consider Negative Binomial. |

---

### ONE‑GLANCE DECISION TABLE

| Variable Type | Key Characteristics | Recommended Distribution |
|---------------|---------------------|--------------------------|
| Count of successes in fixed trials | Fixed \(n\), constant \(p\), independent | **Binomial** |
| Rare count in fixed interval | Low rate, independent events | **Poisson** |
| Number of trials until first success | Independent Bernoulli trials | **Geometric** |
| Continuous measurement on bounded interval | No preference, all outcomes equally likely | **Uniform** |
| Continuous waiting time with memoryless property | Constant hazard rate | **Exponential** |
| Sum/average of many independent effects | Symmetric, bell‑shaped, CLT applies | **Normal** |

---

> **NOTE TO STUDENTS**: Always start by *identifying the data‑generating mechanism* (fixed trials vs. time‑based counts vs. waiting times) before looking at formulas. Mis‑identifying the mechanism is the most common source of MCQ errors.

---

*Prepared for PG‑DBDA (ACTS, Pune) theory‑dominant examinations.*
