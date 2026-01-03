# 📚 Theory‑Focused Notes: Simulation & Optimization

---
## 1️⃣ Why Analytical Solutions Fail

### Historical Motivation
- Early mathematical physics (Newton, Laplace) relied on closed‑form formulas. As problems grew (high‑dimensional integrals, non‑linear PDEs) exact solutions became intractable.
- **Key Insight**: Many real‑world models involve *no closed‑form* because the underlying equations are either non‑linear, involve stochastic components, or have boundary conditions that defy analytic integration.

### Core Reasons
| Reason | Explanation | Typical Example |
|--------|-------------|-----------------|
| **Non‑linearity** | Superposition no longer holds; equations cannot be solved by simple algebraic manipulation. | Navier‑Stokes equations for fluid flow. |
| **High Dimensionality** | Curse of dimensionality makes symbolic integration impossible. | Multivariate Gaussian integrals beyond 3 dimensions. |
| **Complex Boundary/Initial Conditions** | Irregular domains prevent separation of variables. | Heat equation on an irregular shape. |
| **Stochastic Components** | Randomness introduces expectations that lack closed forms. | Expected payoff of a path‑dependent option. |
| **Implicit Definitions** | Solutions defined only implicitly (e.g., root of transcendental equation). | Logistic growth model with time‑varying carrying capacity. |

### Common Exam Traps
- *“If a differential equation is linear, it always has an analytical solution.”* – False; linear ODEs with variable coefficients may still lack closed forms.
- *“Higher‑order moments are always analytically obtainable.”* – Many distributions (e.g., Cauchy) have undefined moments.

---
## 2️⃣ Role of Randomness in Simulation

### Why we **inject randomness**
- **Monte Carlo principle**: Approximate deterministic quantities (integrals, expectations) by averaging over random draws.
- Randomness provides *unbiased* sampling of the underlying probability space, enabling convergence to the true value via the **Law of Large Numbers**.

### Types of Randomness
| Type | Use‑case | Typical Generator |
|------|----------|-------------------|
| **Pseudo‑random** | General purpose simulations; reproducible via seed. | Mersenne Twister, PCG. |
| **Quasi‑random (low‑discrepancy)** | Variance reduction for integration. | Sobol, Halton sequences. |
| **True random (hardware)** | Cryptographic simulations, when independence is critical. | Quantum RNG, atmospheric noise. |

### Edge Cases & Pitfalls
- **Seed leakage**: Re‑using the same seed across experiments can create hidden dependencies.
- **Correlation in generated streams**: Poor RNGs may exhibit serial correlation, violating independence assumptions → biased estimates.
- **Dimensionality vs. uniformity**: In high dimensions, pseudo‑random points become sparse; quasi‑random sequences mitigate but require careful scrambling.

### MCQ ALERT
1. *Which statement about pseudo‑random number generators is **false**?* (A) They are deterministic given a seed, (B) They guarantee statistical independence, (C) They can be reproduced, (D) They are suitable for Monte Carlo integration.)

---
## 3️⃣ Monte Carlo Logic

### Core Idea
1. **Define** the quantity of interest \(\theta = \mathbb{E}[g(X)]\) where \(X\sim p(x)\).
2. **Draw** \(N\) independent samples \(X_1,\dots,X_N\) from \(p(x)\).
3. **Estimate** \(\hat{\theta}=\frac{1}{N}\sum_{i=1}^{N} g(X_i)\).
4. **Assess error** using Central Limit Theorem (CLT): \(\hat{\theta}\approx \mathcal{N}\big(\theta, \frac{\sigma^2}{N}\big)\).

### Variance Reduction Techniques (Why they matter)
| Technique | Mechanism | When to use |
|-----------|-----------|-------------|
| **Antithetic variates** | Pair each sample with its complement to cancel variance. | Smooth, monotonic integrands. |
| **Control variates** | Use a correlated variable with known expectation to adjust estimate. | When a cheap, correlated estimator exists. |
| **Importance sampling** | Sample from a distribution \(q(x)\) that over‑weights important regions; weight by \(p/q\). | Rare‑event probability estimation. |
| **Stratified sampling** | Partition space, sample proportionally within each stratum. | Heterogeneous domains. |

### Common Misconceptions (MCQ focus)
- *“Increasing the number of samples always linearly reduces error.”* – Error scales as \(1/\sqrt{N}\), not \(1/N\).
- *“Monte Carlo is only for high‑dimensional integrals.”* – It is also useful for low‑dimensional problems when analytic integration is hard.

---
## 4️⃣ Optimization Thinking in Simulation

### Why optimisation matters
- Simulations often need **parameter tuning** (e.g., choosing step size, proposal distribution) to minimise variance or computational cost.
- **Stochastic optimisation** (e.g., Simulated Annealing, Genetic Algorithms) uses Monte Carlo sampling to explore solution spaces.

### Key Concepts
| Concept | Description | Typical Algorithm |
|---------|-------------|-------------------|
| **Gradient‑free optimization** | No explicit derivatives; rely on function evaluations. | Nelder‑Mead, Bayesian Optimisation. |
| **Stochastic Gradient Descent (SGD)** | Uses random mini‑batches to approximate gradient. | Deep learning training. |
| **Meta‑heuristics** | Randomised search strategies inspired by nature. | Simulated Annealing, Particle Swarm. |
| **Convergence criteria** | Based on variance of estimates or change in objective. | Stopping when \(\Delta < \epsilon\). |

### Edge Cases
- **Noisy objective**: Optimization may converge to a region rather than a point; need smoothing or larger batch sizes.
- **Plateaus**: Random walks can get stuck; incorporate momentum or temperature schedules.

### MCQ ALERT
2. *In importance sampling, the optimal proposal distribution \(q^*(x)\) is proportional to:* (A) \(p(x)\), (B) \(|g(x)|p(x)\), (C) \(g(x)\), (D) Uniform distribution.

---
## 📌 Summary of Simulation‑Related MCQ Traps
| Trap | Why it’s wrong | Correct reasoning |
|------|----------------|-------------------|
| Believing more samples → linear error reduction | Error decreases with \(\sqrt{N}\) (CLT). | Doubling samples reduces error by ~29 %. |
| Assuming pseudo‑random generators guarantee independence | Deterministic algorithms can exhibit serial correlation. | Independence must be verified via statistical tests. |
| Using importance sampling without weighting | Leads to biased estimates. | Always weight by \(p/q\). |
| Ignoring variance reduction when estimating rare events | Variance can be astronomically high. | Apply importance sampling or stratification. |

---
## 📚 Further Reading (concise list)
- Metropolis, N., & Ulam, S. (1949). *The Monte Carlo method*. Journal of the American Statistical Association.
- Robert, C., & Casella, G. (2004). *Monte Carlo Statistical Methods*.
- Rubinstein, R. Y., & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method*.
- Nocedal, J., & Wright, S. (2006). *Numerical Optimization* (chapters on stochastic methods).
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.

---
*End of notes.*
