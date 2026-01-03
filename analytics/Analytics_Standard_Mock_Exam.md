# 📝 Comprehensive Candidate Assessment: Advanced Analytics Theories
**Target Audience:** PG-DBDA Students | **Focus:** Concepts & Theory (No Calculations)

---

## 🟢 Section 1: Decision Analytics & Evaluation
**Q1. In the context of decision analytics, why is "prediction" alone considered insufficient?**
A. Predictions are always inaccurate.
B. Predictions do not account for resource constraints and stakeholder preferences (utility).
C. Prediction models are computationally more expensive than decision models.
D. Predictions can only handle numerical data, not categorical.
> **Answer: B** | *Reasoning:* A prediction tells you "what might happen," but a decision model tells you "what to do" by considering costs, risks, and benefits.

**Q2. You are building a fraud detection model. The cost of missing a fraud (False Negative) is $10,000, while calling a customer to verify a transaction (False Positive) costs $5. Which metric should you optimize?**
A. Accuracy
B. Precision
C. Recall (Sensitivity)
D. Specificity
> **Answer: C** | *Reasoning:* Recall measures the ability to catch positives (frauds). In high-risk scenarios, missing a positive is catastrophic, so we maximize Recall even if Precision drops.

**Q3. What does the "Accuracy Paradox" refer to?**
A. As models get more complex, accuracy decreases.
B. A model with high accuracy may have zero predictive power for the minority class in imbalanced datasets.
C. Accuracy is theoretically impossible to calculate for regression problems.
D. Using more training data reduces the accuracy of the test set.
> **Answer: B** | *Reasoning:* If 99% of transactions are legit, a model that predicts "Legit" for everything has 99% accuracy but is useless for finding fraud.

**Q4. A "Baseline Model" in analytics is primarily used to:**
A. Generate the final predictions for production.
B. Impute missing values in the dataset.
C. Establish a minimum performance threshold to justify complex modeling efforts.
D. Visualize the correlation matrix.
> **Answer: C** | *Reasoning:* If a complex Neural Network cannot beat a simple "Average" or "Rule-based" baseline, the complexity is unjustified.

---

## 🔵 Section 2: Evidence & Probability
**Q5. According to the Bayesian view, "Probability" is best defined as:**
A. The long-run relative frequency of an event.
B. A measure of the randomness of the universe.
C. A subjective degree of belief updated by evidence.
D. The ratio of favorable cases to total cases.
> **Answer: C** | *Reasoning:* Frequentists view probability as a limit of repeated trials; Bayesians view it as a state of knowledge (belief) that changes with data.

**Q6. If \( P(A|B) \) is very high, does this imply \( P(B|A) \) is also high?**
A. Yes, always.
B. No, this is the "Confusion of the Inverse".
C. Only if the variables are independent.
D. Only if the sample size is large (>30).
> **Answer: B** | *Reasoning:* Example: Probability of Coughing given Lung Cancer is high. Probability of Lung Cancer given Coughing is low (Base Rate Fallacy).

**Q7. "Evidence" in analytics is only useful if it is:**
A. Large in volume (Big Data).
B. Numeric (Quantitative).
C. Relevant and capable of altering the posterior probability (Informative).
D. Collected from a generated simulation.
> **Answer: C** | *Reasoning:* Data becomes evidence only when it has the power to support or refute a hypothesis (update belief). Irrelevant data is noise.

**Q8. When aggregating multiple pieces of evidence, the assumption of "Conditional Independence" allows us to:**
A. Simply add the probabilities together.
B. Multiply the likelihoods of each evidence given the hypothesis.
C. Ignore the Prior probability.
D. Use a Linear Regression model.
> **Answer: B** | *Reasoning:* This is the foundation of Naive Bayes. If independent, \( P(E_1, E_2 | H) = P(E_1|H) \times P(E_2|H) \).

---

## 🟠 Section 3: Business Strategy
**Q9. Which of the following is an example of "Operational Effectiveness," NOT "Strategy"?**
A. Choosing to target a completely new unserved demographic.
B. Determining to be the lowest-cost provider in the industry.
C. Upgrading server software to process transactions 10% faster.
D. Pivoting from a product-based to a service-based business model.
> **Answer: C** | *Reasoning:* Operational effectiveness is "doing things better" (tactics). Strategy is "doing things differently" (positioning).

**Q10. Why is proprietary historical data considered a source of sustainable competitive advantage?**
A. Because storage is cheap.
B. Because it is a "Rare" and "Inimitable" resource that competitors cannot copy.
C. Because it ensures the algorithms will run faster.
D. Because it allows for unsupervised learning.
> **Answer: B** | *Reasoning:* Competitors can copy your code and hire your people, but they cannot buy your past data. This fits the RBV (Resource-Based View) of the firm.

**Q11. "Cost Leadership" strategy implies using analytics primarily for:**
A. Creating highly personalized, premium customer experiences.
B. Optimizing supply chains and automating processes to squeeze out waste.
C. Designing new logos and branding materials.
D. Finding niche market segments willing to pay more.
> **Answer: B** | *Reasoning:* Cost leaders compete on price; analytics must focus on efficiency (Operational Excellence).

---

## 🟣 Section 4: Factor Analysis & Dimensions
**Q12. The primary goal of Factor Analysis is to:**
A. Predict a dependent variable y.
B. Cluster similar observations into groups.
C. Reduce dimensionality by identifying underlying latent constructs that explain correlations.
D. Test if the means of two groups are different.
> **Answer: C** | *Reasoning:* FA looks for "hidden" causes (Latent variables) that explain why observed variables move together.

**Q13. In Factor Analysis, a "Factor Loading" of 0.80 indicates:**
A. The variable is 80% error.
B. There is a strong correlation between the variable and the factor.
C. The reliability of the dataset is 80%.
D. The factor explains 80% of the total variance in the dataset.
> **Answer: B** | *Reasoning:* Loadings are essentially correlation coefficients between the item and the factor.

**Q14. Why is the arithmetic mean inappropriate for Directional (Circular) data?**
A. Because 0 degrees and 360 degrees are the same point, but the arithmetic mean treats them as extremes.
B. Because directional data always has a negative skew.
C. Because directional data is qualitative, not quantitative.
D. Because the sample size is usually too small.
> **Answer: A** | *Reasoning:* The average of 1° and 359° should be 0° (North), but arithmetic mean gives 180° (South).

**Q15. Which test is used to verify if variables are sufficiently correlated to justify Factor Analysis?**
A. T-test
B. Chi-Square Test
C. Kaiser-Meyer-Olkin (KMO) Test & Bartlett’s Test of Sphericity
D. Shapiro-Wilk Test
> **Answer: C** | *Reasoning:* KMO measures sampling adequacy (>0.6 good) and Bartlett tests if the correlation matrix is significantly different from an identity matrix.

---
**End of Standard Mock Exam**
