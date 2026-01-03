# 🧠 The "Tricky" Exam: Edge Cases, Traps & Common Errors
**Target Audience:** Advanced PG-DBDA Candidates | **Focus:** Debunking Misconceptions

---

## 🚩 Section 1: The "Obvious" Traps
*(Questions where the gut feeling is usually wrong)*

**Q1. A dataset has a Pearson Correlation Coefficient ($r$) of exactly 0.0 between X and Y. What can you definitively conclude?**
A. X and Y are completely unrelated.
B. X and Y are constant variables.
C. There is no linear relationship between X and Y, but a strong non-linear relationship (e.g., quadratic) might still exist.
D. You need more data to calculate $r$ properly.
> **Answer: C** | *Trap:* $r=0$ only rules out straight lines. A perfect parabola ($Y=X^2$) can have $r=0$. Don't conflate "No Correlation" with "No Association".

**Q2. You train a classifier on a dataset with 99% Class A and 1% Class B. The model achieves 99% overall accuracy. Is this a good model?**
A. Yes, 99% is excellent.
B. Need to check the P-value.
C. Probably not; it likely predicts "Class A" for everything (The ZeroR/Baseline Strategy).
D. Yes, provided the test set was also 99% Class A.
> **Answer: C** | *Trap:* Accuracy is deceptive in imbalanced data. A "dumb" model (Baseline) gets 99% without learning anything. You need Kappa, F1-score, or AUC.

**Q3. If a medical test has 99% accuracy (Sensitivity=99%, Specificity=99%) and you use it on a disease with 0.1% prevalence, what is the probability a positive test means the person actually has the disease?**
A. 99%
B. Approx 9%
C. 50%
D. 0.1%
> **Answer: B** | *Trap:* Base Rate Neglect.
> *Reasoning:* In 1000 people, 1 has disease (Test: Positive). 999 healthy (Test: ~10 False Positives). Real Positive (1) vs False Positives (10). Probability is $1/(1+10) \approx 9\%$.

---

## 🚧 Section 2: P-Values & Significance
*(The most misinterpreted concept in statistics)*

**Q4. You run a hypothesis test and get a p-value of 0.03. Which sentence is technically CORRECT?**
A. There is a 97% chance the alternative hypothesis is true.
B. There is a 3% chance the null hypothesis is true.
C. Assuming the null hypothesis is true, there is a 3% chance of observing data this extreme.
D. The result is practically significant.
> **Answer: C** | *Trap:* P-value is NOT the probability of the hypothesis. It's the probability of the *data* given the *hypothesis*. Also, statistical significance (p < 0.05) $\neq$ practical significance (effect size).

**Q5. A researcher finds a "significant correlation" (p < 0.05) of 0.01 between eating jellybeans and acne in a sample of 1 million people. Implication?**
A. Eating jellybeans causes acne.
B. The correlation is strong.
C. The result is statistically significant due to huge sample size, but likely practically useless (tiny effect size).
D. The p-value calculation is wrong because the r value is too small.
> **Answer: C** | *Trap:* With massive N, even microscopic deviations from 0 Become "significant". Significance tells you "It's not zero", not "It's important".

---

## 🔮 Section 3: Causal vs. Predictive
*(Mistaking Correlation for Causation)*

**Q6. High ice cream sales are perfectly correlated with shark attacks ($r=0.9$). To save lives, the mayor bans ice cream. Why is this wrong?**
A. He should have banned sharks instead.
B. **Confounding Variable (Latent Factor):** Summer heat causes *both* swimming (shark exposure) and ice cream eating.
C. The corelation is spurious/accidental.
D. He needs to run a T-test first.
> **Answer: B** | *Trap:* This is the classic "Common Cause" scenario. Controlling for "Temperature" would make the partial correlation vanish.

**Q7. In a regression model ($Y = aX + b$), variable X has a high positive coefficient. Does increasing X *cause* Y to increase?**
A. Yes, that's what the coefficient means.
B. Only if X is a randomized treatment in a controlled experiment. Otherwise, it just predicts association.
C. No, regression can never prove causality, even in experiments.
D. Yes, if the $R^2$ is high enough.
> **Answer: B** | *Trap:* Coefficients in observational studies (passive data) only show association. Only *intervention* (Do-calculus/Experiments) proves causality.

---

## 📉 Section 4: Model Evaluation Edge Cases

**Q8. You have a test set. You try 100 different models on it and pick the one with the best accuracy. Why is the reported accuracy of this "winner" likely an overestimate?**
A. It isn't; you picked the best one.
B. **Multiple Comparisons Problem:** By trying 100 times, you essentially "trained" on the test set (Overfitting to the test set).
C. You should have used R-squared instead of accuracy.
D. The models disturb the test data.
> **Answer: B** | *Trap:* "Data Dredging" works on model selection too. If you select based on Test Set performance, the Test Set is no longer "Unseen". You need a **Validation Set** for selection and a fresh **Test Set** for final reporting.

**Q9. Which metric is robust to outliers?**
A. Mean Squared Error (MSE)
B. Root Mean Squared Error (RMSE)
C. Mean Absolute Error (MAE)
D. R-Squared
> **Answer: C** | *Trap:* Squaring the errors (MSE/RMSE/R2) disproportionately punishes large errors (outliers), making the metric sensitive to them. Absolute value (MAE) is linear and more robust.

**Q10. Why is PCA *not* a good method for Feature Selection usually?**
A. It's too slow.
B. It transforms features into new, uninterpretable components (PC1, PC2...) rather than selecting the best original features.
C. It only works for categorical data.
D. It increases dimensionality.
> **Answer: B** | *Trap:* Confusion between *Dimensionality Reduction* (PCA - changing the variable space) and *Feature Selection* (Lasso/Stepwise - keeping a subset of original variables).

---
**End of Tricky Exam**
