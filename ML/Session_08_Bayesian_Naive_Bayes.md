# Session 8 – Bayesian Analysis & Naïve Bayes

## 📚 Table of Contents
1. [Bayesian Probability Theory](#bayesian-probability-theory)
2. [Bayes' Theorem](#bayes-theorem)
3. [Naïve Bayes Classifier](#naïve-bayes-classifier)
4. [Gaussian Naïve Bayes](#gaussian-naïve-bayes)
5. [Multinomial Naïve Bayes](#multinomial-naïve-bayes)
6. [Bernoulli Naïve Bayes](#bernoulli-naïve-bayes)
7. [Bayesian vs Frequentist](#bayesian-vs-frequentist)
8. [MCQs](#mcqs)
9. [Common Mistakes](#common-mistakes)
10. [One-Line Exam Facts](#one-line-exam-facts)

---

# Bayesian Probability Theory

## 📘 Concept Overview

**Bayesian statistics** treats probability as **degree of belief** that can be updated with evidence.

### Core Philosophy

- **Frequentist**: Probability = long-run frequency of events
- **Bayesian**: Probability = degree of belief given information

**Key insight**: Update beliefs as new data arrives.

## 🧮 Probability Foundations

### Joint Probability

```
P(A, B) = P(A and B)
```

Probability that both A and B occur.

### Conditional Probability

```
P(A|B) = P(A, B) / P(B)
```

Probability of A given that B has occurred.

**Rearranged**:
```
P(A, B) = P(A|B) × P(B) = P(B|A) × P(A)
```

### Marginal Probability

```
P(A) = Σ_B P(A, B) = Σ_B P(A|B) P(B)
```

Sum over all possible values of B (marginalization).

### Independence

A and B are independent if:
```
P(A, B) = P(A) × P(B)
```

Equivalently:
```
P(A|B) = P(A)
```

Knowing B doesn't change probability of A.

### Conditional Independence

A and B are conditionally independent given C if:
```
P(A, B|C) = P(A|C) × P(B|C)
```

Equivalently:
```
P(A|B, C) = P(A|C)
```

**Important**: A and B can be dependent, but independent given C!

**Example**: 
- A = "grass is wet", B = "sidewalk is wet", C = "it rained"
- A and B are dependent (usually wet together)
- But A ⊥ B | C (if we know it rained, grass wetness doesn't tell us about sidewalk)

---

# Bayes' Theorem

## 🧮 Mathematical Derivation

From conditional probability:
```
P(A, B) = P(A|B) P(B) = P(B|A) P(A)
```

Solving for P(A|B):
```
P(A|B) = P(B|A) P(A) / P(B)
```

**This is Bayes' Theorem!**

### Expanded Form

```
P(A|B) = P(B|A) P(A) / [Σ_a P(B|A=a) P(A=a)]
```

Denominator ensures normalization (sums to 1).

## 🧠 Intuition

```
Posterior = (Likelihood × Prior) / Evidence

P(hypothesis|data) = P(data|hypothesis) × P(hypothesis) / P(data)
```

**Components**:

1. **Prior P(A)**: Belief before seeing data
2. **Likelihood P(B|A)**: Probability of data given hypothesis
3. **Evidence P(B)**: Normalizing constant (total probability of data)
4. **Posterior P(A|B)**: Updated belief after seeing data

## 📊 Example: Medical Diagnosis

**Problem**: Patient tests positive for disease. What's probability they have it?

**Given**:
- Disease prevalence: P(Disease) = 0.01 (1%)
- Test sensitivity (TPR): P(Positive|Disease) = 0.95 (95%)
- Test specificity (TNR): P(Negative|No Disease) = 0.90 (90%)

**Find**: P(Disease|Positive)

**Solution**:

```
P(Positive|No Disease) = 1 - 0.90 = 0.10

P(Positive) = P(Positive|Disease)P(Disease) + P(Positive|No Disease)P(No Disease)
            = 0.95 × 0.01 + 0.10 × 0.99
            = 0.0095 + 0.099
            = 0.1085

P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
                    = (0.95 × 0.01) / 0.1085
                    = 0.0876
                    ≈ 8.76%
```

**Surprising result**: Even with positive test, only 8.76% chance of having disease!

**Reason**: Low prior (1% prevalence) and moderate false positive rate (10%).

## 🧪 Python Implementation

```python
def bayes_theorem(prior, likelihood, evidence):
    """
    Compute posterior using Bayes' theorem.
    
    Args:
        prior: P(H)
        likelihood: P(E|H)
        evidence: P(E)
    
    Returns:
        posterior: P(H|E)
    """
    return (likelihood * prior) / evidence

# Medical diagnosis example
prior_disease = 0.01
likelihood_pos_given_disease = 0.95
likelihood_pos_given_healthy = 0.10

# Compute evidence
evidence_positive = (likelihood_pos_given_disease * prior_disease + 
                    likelihood_pos_given_healthy * (1 - prior_disease))

# Compute posterior
posterior_disease_given_pos = bayes_theorem(
    prior_disease,
    likelihood_pos_given_disease,
    evidence_positive
)

print(f"P(Disease|Positive) = {posterior_disease_given_pos:.4f}")
print(f"Percentage: {posterior_disease_given_pos * 100:.2f}%")
```

---

# Naïve Bayes Classifier

## 📘 Concept Overview

**Naïve Bayes** is a probabilistic classifier based on Bayes' theorem with **naïve** conditional independence assumption.

## 🧮 Mathematical Foundation

### Classification Goal

Given features X = (x₁, x₂, ..., xₐ), predict class y.

Choose class with highest posterior probability:
```
ŷ = argmax_y P(y|X)
```

### Applying Bayes' Theorem

```
P(y|X) = P(X|y) P(y) / P(X)
```

Since P(X) is constant for all classes:
```
ŷ = argmax_y P(X|y) P(y)
```

### The "Naïve" Assumption

**Assume features are conditionally independent given class**:
```
P(X|y) = P(x₁, x₂, ..., xₐ|y) = ∏ᵢ P(xᵢ|y)
```

**Why "naïve"?** Features rarely truly independent!

**Example violation**: Email spam detection
- x₁ = "contains 'free'"
- x₂ = "contains 'money'"
- These are correlated, but we assume independence

### Naïve Bayes Formula

```
ŷ = argmax_y P(y) ∏ᵢ P(xᵢ|y)
```

**Components**:
1. **Class prior**: P(y) = fraction of training samples in class y
2. **Feature likelihood**: P(xᵢ|y) = depends on feature type

### Log-Space Computation

To avoid numerical underflow (many small probabilities multiplied):

```
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
```

## ⚙️ Training Algorithm

```
1. For each class y:
   a) Compute class prior: P(y) = count(y) / total_samples
   
   b) For each feature xᵢ:
      Compute P(xᵢ|y) based on feature distribution
      (Gaussian, Multinomial, or Bernoulli)

2. Store priors and likelihoods
```

## ⚙️ Prediction Algorithm

```
For new sample X = (x₁, ..., xₐ):

1. For each class y:
   score[y] = log P(y) + Σᵢ log P(xᵢ|y)

2. Return: ŷ = argmax_y score[y]
```

## 📊 Strengths & Weaknesses

### Strengths ✓
1. **Fast training and prediction**: O(nd + kd) where k = classes
2. **Handles high dimensions**: Works well with many features
3. **Small data**: Requires less training data than discriminative models
4. **Probabilistic**: Outputs class probabilities
5. **Multi-class**: Naturally handles multiple classes
6. **Baseline**: Good starting point for text classification

### Weaknesses ✗
1. **Independence assumption**: Rarely true in practice
2. **Zero probability problem**: P(xᵢ|y)=0 makes entire product zero
3. **Feature correlations**: Can't capture feature interactions
4. **Continuous features**: Gaussian assumption may not hold
5. **Calibration**: Probability estimates can be poor

---

# Gaussian Naïve Bayes

## 📘 Concept Overview

For **continuous features**, assume each feature follows **Gaussian (normal) distribution** within each class.

## 🧮 Mathematical Foundation

### Likelihood

```
P(xᵢ|y) = (1/√(2πσ²ᵧᵢ)) exp(-(xᵢ - μᵧᵢ)² / (2σ²ᵧᵢ))
```

Where:
- μᵧᵢ = mean of feature i for class y
- σ²ᵧᵢ = variance of feature i for class y

### Parameter Estimation (MLE)

For class y and feature i:

```
μᵧᵢ = (1/nᵧ) Σ_{samples in class y} xᵢ

σ²ᵧᵢ = (1/nᵧ) Σ_{samples in class y} (xᵢ - μᵧᵢ)²
```

### Prediction Formula

```
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
  = argmax_y [log P(y) - Σᵢ (log(σᵧᵢ√(2π)) + (xᵢ-μᵧᵢ)²/(2σ²ᵧᵢ))]
```

## 🧪 Python Implementation

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naïve Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Inspect learned parameters
print(f"\nClass priors: {gnb.class_prior_}")
print(f"\nMeans (μ):\n{gnb.theta_}")
print(f"\nVariances (σ²):\n{gnb.var_}")

# Predict probabilities for first test sample
print(f"\nProbabilities for first test sample:")
for i, prob in enumerate(y_proba[0]):
    print(f"  Class {i}: {prob:.4f}")
```

### From Scratch

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Compute parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples
    
    def _gaussian_pdf(self, class_idx, x):
        """Compute Gaussian PDF for class."""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # Gaussian PDF
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return numerator / denominator
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Compute log posterior for each class
            posteriors = []
            
            for idx, c in enumerate(self.classes):
                # Log prior
                prior = np.log(self.priors[idx])
                
                # Log likelihood (sum of log probabilities)
                likelihood = np.sum(np.log(self._gaussian_pdf(idx, x)))
                
                # Log posterior
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Choose class with highest posterior
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)

# Test
gnb_custom = GaussianNaiveBayes()
gnb_custom.fit(X_train, y_train)
y_pred_custom = gnb_custom.predict(X_test)
print(f"Custom GNB Accuracy: {accuracy_score(y_test, y_pred_custom):.3f}")
```

## ⚠️ Assumptions & Limitations

### Assumptions
1. **Gaussian distribution**: Each feature normally distributed within class
2. **Independence**: Features independent given class
3. **Homoscedasticity** (not required): Can have different variances per class

### When Gaussian NB Fails
- **Categorical features**: Use Multinomial or Bernoulli instead
- **Multi-modal distributions**: Gaussian assumption violated
- **Heavy-tailed distributions**: Outliers influence mean/variance
- **Skewed distributions**: May need transformation (log, Box-Cox)

---

# Multinomial Naïve Bayes

## 📘 Concept Overview

For **count-based features** (e.g., word counts in documents), assume **multinomial distribution**.

**Common use**: Text classification (spam filtering, sentiment analysis, document categorization).

## 🧮 Mathematical Foundation

### Multinomial Distribution

For document with total word count n, probability of observing word counts (n₁, n₂, ..., nₐ):

```
P(X|y) = (n! / ∏ᵢ nᵢ!) ∏ᵢ θᵧᵢⁿⁱ
```

Where:
- θᵧᵢ = P(word i | class y)
- Σᵢ θᵧᵢ = 1

**For classification, constant terms cancel**:
```
P(X|y) ∝ ∏ᵢ θᵧᵢⁿⁱ
```

### Parameter Estimation (MLE with Laplace Smoothing)

```
θᵧᵢ = (count(word i in class y) + α) / (total words in class y + α×d)
```

Where:
- α = smoothing parameter (default α=1, Laplace smoothing)
- d = vocabulary size

**Why smoothing?** Avoid θᵧᵢ = 0 for unseen words.

### Prediction

```
ŷ = argmax_y [log P(y) + Σᵢ nᵢ log θᵧᵢ]
```

## 🧪 Python Implementation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load text data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train_data = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Convert text to word counts
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

y_train = train_data.target
y_test = test_data.target

# Train Multinomial Naïve Bayes
mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing
mnb.fit(X_train, y_train)

# Predictions
y_pred = mnb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

# Most informative features per class
feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(train_data.target_names):
    top_features = np.argsort(mnb.feature_log_prob_[i])[-10:]
    print(f"\n{category}:")
    print(", ".join([feature_names[j] for j in top_features]))
```

### TF-IDF with Multinomial NB

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF better than raw counts for many tasks
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data.data)
X_test_tfidf = tfidf_vectorizer.transform(test_data.data)

# Note: TF-IDF can have negative values after mean subtraction
# Use MultinomialNB with non-negative features or use GaussianNB
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_train_tfidf, y_train)
```

## 📊 Use Cases

| Domain | Features | Why Multinomial NB |
|--------|----------|-------------------|
| **Spam filtering** | Word counts | Fast, interpretable |
| **Sentiment analysis** | N-gram counts | Good baseline |
| **Document classification** | TF-IDF | Scalable to large corpus |
| **Author identification** | Character n-grams | Captures writing style |

---

# Bernoulli Naïve Bayes

## 📘 Concept Overview

For **binary features** (presence/absence), assume **Bernoulli distribution**.

**Difference from Multinomial**: Explicitly models feature absence.

## 🧮 Mathematical Foundation

### Bernoulli Distribution

For binary feature xᵢ ∈ {0, 1}:

```
P(xᵢ|y) = pᵧᵢˣⁱ (1 - pᵧᵢ)¹⁻ˣⁱ
```

Where pᵧᵢ = P(xᵢ=1 | y)

**Likelihood**:
```
P(X|y) = ∏ᵢ [pᵧᵢˣⁱ (1 - pᵧᵢ)¹⁻ˣⁱ]
```

### Parameter Estimation

```
pᵧᵢ = (count(xᵢ=1 in class y) + α) / (total samples in class y + 2α)
```

With Laplace smoothing (α=1):
```
pᵧᵢ = (count(xᵢ=1 in class y) + 1) / (total samples in class y + 2)
```

### Prediction

```
ŷ = argmax_y [log P(y) + Σᵢ (xᵢ log pᵧᵢ + (1-xᵢ) log(1-pᵧᵢ))]
```

## 🧪 Python Implementation

```python
from sklearn.naive_bayes import BernoulliNB

# Binarize features (word presence/absence)
from sklearn.preprocessing import Binarizer

X_train_binary = Binarizer().fit_transform(X_train)
X_test_binary = Binarizer().fit_transform(X_test)

# Train Bernoulli Naïve Bayes
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)

# Predictions
y_pred = bnb.predict(X_test_binary)

accuracy = accuracy_score(y_test, y_pred)
print(f"Bernoulli NB Accuracy: {accuracy:.3f}")
```

## 🔄 Multinomial vs Bernoulli

| Aspect | Multinomial NB | Bernoulli NB |
|--------|---------------|--------------|
| **Feature type** | Counts (word frequency) | Binary (presence/absence) |
| **Distribution** | Multinomial | Bernoulli |
| **Models absence** | No (only counts) | ✓ Yes (explicitly) |
| **Text length** | Sensitive (longer docs have higher counts) | ✓ Less sensitive |
| **Use case** | Frequency matters | Presence/absence matters |

**Example**: "free" appears 10 times vs 1 time
- Multinomial: Different (10x weight)
- Bernoulli: Same (both present)

---

# Bayesian vs Frequentist

## 📊 Philosophical Differences

| Aspect | Frequentist | Bayesian |
|--------|------------|----------|
| **Probability** | Long-run frequency | Degree of belief |
| **Parameters** | Fixed (unknown) | Random variables (with distribution) |
| **Inference** | Point estimates (MLE) | Posterior distributions |
| **Prior** | Not used | Central (encodes prior knowledge) |
| **Uncertainty** | Confidence intervals | Credible intervals |

## 🧮 Example: Coin Flip

**Question**: Estimate probability of heads.

**Data**: 7 heads in 10 flips.

### Frequentist Approach

```
θ̂_MLE = 7/10 = 0.7
```

**Interpretation**: Best point estimate is 0.7.

**Confidence interval** (95%):
```
CI = θ̂ ± 1.96√(θ̂(1-θ̂)/n)
   = 0.7 ± 1.96√(0.7×0.3/10)
   = 0.7 ± 0.28
   = [0.42, 0.98]
```

**Meaning**: If we repeat this experiment many times, 95% of such intervals will contain true θ.

### Bayesian Approach

**Prior**: θ ~ Beta(α, β) (conjugate prior for Bernoulli)
- Assume θ ~ Beta(2, 2) (slightly favors 0.5)

**Likelihood**: Binomial(7 | 10, θ)

**Posterior**: θ | data ~ Beta(α+7, β+3) = Beta(9, 5)

```
Posterior mean = α/(α+β) = 9/(9+5) = 0.643
```

**Credible interval** (95%): [0.38, 0.87]

**Meaning**: Given the data, there's 95% probability that θ ∈ [0.38, 0.87].

## 📊 Prior Selection

### Uninformative Priors
- **Uniform**: Beta(1, 1) (all values equally likely)
- **Jeffreys**: Beta(0.5, 0.5) (invariant to reparameterization)

### Informative Priors
- **Expert knowledge**: Beta(α, β) based on domain expertise
- **Empirical Bayes**: Estimate prior from data

### Impact of Prior

**Strong prior** + **little data** → Prior dominates
**Weak prior** + **lots of data** → Likelihood dominates

**Example**: 
- Prior: Beta(100, 100) (strong belief in θ=0.5)
- Data: 70 heads in 100 flips
- Posterior: Beta(170, 130) → Mean = 0.567 (pulled toward prior)

---

# 🔥 MCQs

### Q1. Naïve Bayes assumes:
**Options:**
- A) Features are independent
- B) Features are conditionally independent given class ✓
- C) Classes are independent
- D) Features are identically distributed

**Explanation**: Naïve assumption = P(X|y) = ∏P(xᵢ|y) (conditional independence).

---

### Q2. Gaussian Naïve Bayes assumes each feature follows:
**Options:**
- A) Uniform distribution
- B) Exponential distribution
- C) Normal distribution ✓
- D) Binomial distribution

**Explanation**: GNB models P(xᵢ|y) as Gaussian with class-specific mean and variance.

---

### Q3. Laplace smoothing (α=1) in Multinomial NB prevents:
**Options:**
- A) Overfitting
- B) Underfitting
- C) Zero probability for unseen words ✓
- D) High variance

**Explanation**: Adds α to counts so P(word|class) ≠ 0 even if word unseen in training.

---

### Q4. Bayes' theorem formula is:
**Options:**
- A) P(A|B) = P(A) / P(B)
- B) P(A|B) = P(B|A) P(A) / P(B) ✓
- C) P(A|B) = P(A) P(B)
- D) P(A|B) = P(B) / P(A)

**Explanation**: Posterior = (Likelihood × Prior) / Evidence.

---

### Q5. In Naïve Bayes, which is NOT needed for prediction?
**Options:**
- A) P(y)
- B) P(xᵢ|y)
- C) P(y|xᵢ) ✓
- D) P(X|y)

**Explanation**: We compute P(y|X) from P(y) and P(X|y) = ∏P(xᵢ|y).

---

### Q6. Multinomial NB is best for:
**Options:**
- A) Continuous features
- B) Binary features
- C) Count features (word counts) ✓
- D) Ordinal features

**Explanation**: Models word/feature counts with multinomial distribution.

---

### Q7. Bernoulli NB differs from Multinomial by:
**Options:**
- A) Uses different prior
- B) Explicitly models feature absence ✓
- C) Faster training
- D) Better for regression

**Explanation**: Bernoulli considers both presence (xᵢ=1) and absence (xᵢ=0).

---

### Q8. Naïve Bayes is a ______ model.
**Options:**
- A) Discriminative
- B) Generative ✓
- C) Non-parametric
- D) Instance-based

**Explanation**: Generative models learn P(X, y) = P(X|y)P(y), then apply Bayes' rule.

---

### Q9. Prior P(y) in Naïve Bayes is estimated as:
**Options:**
- A) 1 / number of classes
- B) Fraction of samples in class y ✓
- C) Uniform distribution
- D) Based on validation set

**Explanation**: P(y) = count(y) / total_samples (MLE).

---

### Q10. Why compute in log-space for Naïve Bayes?
**Options:**
- A) Faster computation
- B) More interpretable
- C) Avoid numerical underflow ✓
- D) Improve accuracy

**Explanation**: Many small probabilities multiplied → underflow. Use log: product → sum.

---

### Q11. If prior P(Disease)=0.01 and P(Positive|Disease)=0.99, after positive test:
**Options:**
- A) Definitely have disease
- B) < 100% probability (depends on false positive rate) ✓
- C) 99% probability
- D) 1% probability

**Explanation**: Must account for P(Positive|Healthy). Low prior + false positives → posterior < 99%.

---

### Q12. Gaussian NB variance parameter is estimated:
**Options:**
- A) Per feature across all classes
- B) Per class across all features
- C) Per feature per class ✓
- D) Globally for all features and classes

**Explanation**: σ²ᵧᵢ = variance of feature i in class y (class- and feature-specific).

---

### Q13. Bayesian credible interval means:
**Options:**
- A) Long-run coverage frequency
- B) Probability parameter is in interval (given data) ✓
- C) Confidence in estimate
- D) Margin of error

**Explanation**: E.g., "95% probability θ ∈ [a, b]" (frequentist CI has different meaning).

---

### Q14. Conjugate prior for Bernoulli likelihood is:
**Options:**
- A) Gaussian
- B) Beta ✓
- C) Gamma
- D) Uniform

**Explanation**: Beta prior + Bernoulli/Binomial likelihood → Beta posterior (closed form).

---

### Q15. Naïve Bayes independence assumption is:
**Options:**
- A) Always true
- B) Rarely true but often works well ✓
- C) Required for correctness
- D) Only for text data

**Explanation**: "Naïve" because features rarely independent, but NB still performs well in practice.

---

# ⚠️ Common Mistakes

1. **Assuming Naïve Bayes needs independent features**: It's an **assumption**, not a requirement (often violated but works)

2. **Forgetting Laplace smoothing**: Causes zero probabilities for unseen features

3. **Using Multinomial NB on continuous features**: Should use Gaussian NB

4. **Using Gaussian NB on count data**: Should use Multinomial or Bernoulli

5. **Misinterpreting Bayesian credible intervals as frequentist confidence intervals**: Different meanings!

6. **Ignoring class imbalance**: Use `class_prior` parameter or resampling

7. **Thinking posterior = likelihood**: Posterior also depends on prior!

8. **Using raw counts instead of probabilities**: Must normalize (divide by total)

9. **Forgetting to transform test data same as train**: Use same vectorizer

10. **Over-trusting probability calibration**: NB probabilities can be poorly calibrated (too extreme)

---

# ⭐ One-Line Exam Facts

1. **Bayes' theorem**: P(A|B) = P(B|A)P(A) / P(B)

2. **Naïve Bayes assumption**: P(X|y) = ∏P(xᵢ|y) (conditional independence)

3. **Naïve Bayes classifier**: ŷ = argmax_y P(y)∏P(xᵢ|y)

4. **Gaussian NB**: P(xᵢ|y) ~ N(μᵧᵢ, σ²ᵧᵢ) for continuous features

5. **Multinomial NB**: For count features (word counts, TF-IDF)

6. **Bernoulli NB**: For binary features (presence/absence)

7. **Laplace smoothing**: Add α to counts to avoid zero probabilities (α=1 default)

8. **Log-space computation**: Prevents underflow (∏ → Σ in log space)

9. **Generative model**: Learns P(X|y) and P(y), applies Bayes' rule for P(y|X)

10. **Prior P(y)**: Estimated as class frequency in training data

11. **Frequentist**: Parameters fixed, probability = frequency

12. **Bayesian**: Parameters random, probability = belief

13. **Conjugate prior**: Posterior same distribution as prior (computational convenience)

14. **Beta distribution**: Conjugate prior for Bernoulli/Binomial

15. **Independence ≠ Conditional independence**: A ⊥ B doesn't imply A ⊥ B | C

---

**End of Session 8**
