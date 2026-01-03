# 📚 Advanced Theory Notes: Factor Analysis & Directional Data

---
## SECTION A: Factor Analysis (FA)

### 1️⃣ Why Factor Analysis is Needed
- **The Problem:** We often collect many variables that are **highly correlated** (redundant). Studying them individually ignores their structural relationship and leads to "multicollinearity" issues in models.
- **The Solution:** FA assumes that these correlations exist because the observed variables are influenced by a smaller number of unobserved, underlying **Latent Variables** (Factors).
- **Goal:** To **explain the covariance** among variables using the fewest possible underlying factors. It is a "Structure Detection" method, not just data reduction.

### 2️⃣ Latent Variables vs. Observed Variables
| Concept | Definition | Example |
| :--- | :--- | :--- |
| **Observed Variables** (Manifest) | The actual data we measure directly. | Test scores in Math, Physics, Chemistry, Literature, History. |
| **Latent Variables** (Factors) | The theoretical constructs we *infer* to explain the scores. | "Quantitative Ability" (influences Math/Physics) and "Verbal Ability" (influences Lit/History). |
- **Intuition:** We cannot measure "Intelligence" directly; we measure it indirectly through test scores. FA bridges the gap.

### 3️⃣ Dimensionality Reduction Intuition (Data Compression)
- **Concept:** If 5 variables move together (Result A goes up, Result B goes up...), we don't need 5 dimensions to describe them. We can collapse them into 1 dimension (Factor) with minimal loss of *meaning* (though some loss of *variance*).
- **Parsimony:** Science prefers simpler explanations. Describing a person with 2 factors ("Smart", "Outgoing") is more useful than a list of 50 specific test scores.

### 4️⃣ Factor Loadings and Their Interpretation
- **Definition:** A **Factor Loading** is the **correlation coefficient** between an observed variable and the underlying factor.
- **Range:** -1 to +1.
- **Interpretation:**
    - High Loading (e.g., 0.8): The variable is strongly driven by this factor.
    - **Squared Loading ($R^2$):** The percent of variance in the variable explained by the factor.
    - **Cross-loading:** When a variable loads strongly on *multiple* factors (undesirable for clean interpretation; solved by **Rotation**).

---
## SECTION B: Assumptions & Limitations

### 1️⃣ Key Assumptions
- **Linearity:** The relationship between factors and variables is linear. (FA cannot detect non-linear latent structures).
- **No Outliers:** Outliers can artificially inflate or deflate correlations, distorting factors.
- **Factorability (Correlation Requirement):**
    - **Why:** If variables aren't correlated, they don't share a common source. FA is useless.
    - **Tests:**
        - **Bartlett’s Test of Sphericity:** Checks if the correlation matrix is different from an Identity matrix (random noise). Value should be significant ($p < 0.05$).
        - **KMO (Kaiser-Meyer-Olkin):** Measures sampling adequacy. Should be $> 0.6$.

### 2️⃣ Sample Size Considerations
- **Rule of Thumb:** FA is data-hungry.
    - **Ratio:** At least 10 observations per variable (10:1), ideally 20:1.
    - **Absolute:** rarely stable with $N < 50$; good stability requires $N > 300$.

### 3️⃣ When Factor Analysis Should NOT Be Used
- When correlations are **weak** ($r < 0.3$ for most pairs).
- When the data is **categorical/nominal** (requires specialized FA like MCA).
- When the goal is strictly **prediction** (use PCA or Partial Least Squares instead).

---
## SECTION C: Directional Data Analytics

### 1️⃣ Meaning of Directionality in Data
- **Definition:** Data dealing with **directions** (observations on a circle or sphere), not linear magnitudes.
- **The "Cyclic" Problem:**
    - On a linear scale: 359 and 1 are far apart (diff = 358).
    - On a directional (circular) scale: 359° and 1° are almost identical (diff = 2°).
- **Standard statistics fail here:** The arithmetic mean of 1° and 359° is 180° (South), which is completely wrong (should be 0°/North).

### 2️⃣ Direction vs. Magnitude Distinction
| Feature | Direction | Magnitude |
| :--- | :--- | :--- |
| **Focus** | Orientation / Angle | Intensity / Length |
| **Example** | Wind blowing *North-East* | Wind blowing at *20 km/h* |
| **Representation** | Unit Vector (Length = 1) | Scalar or Vector Length |
| **Analysis Method** | Circular Statistics (Von Mises Distribution) | Normal/Gaussian Statistics |

### 3️⃣ Use Cases of Directional Analytics
- **Meteorology:** Wind direction analysis.
- **Biology:** Migration paths of birds/turtles (orientation).
- **Geology:** Fault line orientations.
- **Machine Learning:** Text embeddings (Cosine similarity focuses on vector *direction*—topic relatedness—rather than magnitude—word count).

---
## 🔍 Comparison Table: PCA vs. Factor Analysis

| Feature | Principal Component Analysis (PCA) | Factor Analysis (EFA/CFA) |
| :--- | :--- | :--- |
| **Primary Goal** | **Data Reduction** (Summarize variance) | **Structure Detection** (Explain covariance) |
| **Variance Focus** | Total Variance (Common + Unique + Error) | **Common Variance** (Shared only) |
| **Assumed Cause** | Components are *aggregates* of variables (Composite) | Factors *cause* the variables (Latent) |
| **Solvability** | Unique mathematical solution | Indeterminate (requires Rotation) |
| **Use Case** | Pre-processing for ML, Image Compression | Psychology, Survey Validation, Theory Building |

---
## ⚠️ MCQ Focus: "Which Statement is FALSE?"

### Question 1: Feature Loadings
**Which statement regarding factor loadings is FALSE?**
- A) A loading represents the correlation between the variable and the factor.
- B) Squared loadings represent the variance explained by the factor.
- C) Ideally, a variable should have high loadings on all extracted factors.
- D) Rotation is used to maximize high loadings and minimize low ones for clarity.
> **Answer: C.** *Ideally, we want "Simple Structure"—a variable should load highly on ONE factor and near-zero on others. High loadings on all factors (Cross-loading) makes interpretation impossible.*

### Question 2: PCA vs Factor Analysis
**Which statement accurately distinguishes PCA from FA?**
- A) PCA assumes latent causality; FA does not.
- B) FA focuses only on shared variance; PCA analyzes total variance.
- C) PCA requires rotation; FA guarantees a unique solution without rotation.
- D) FA handles nominal data better than PCA.
> **Answer: B.** *FA separates "Common Variance" (signal) from "Unique Variance" (noise/error). PCA mashes it all together to maximize explained variance.*

### Question 3: Directional Data
**Why is the arithmetic mean inappropriate for directional data?**
- A) It cannot handle negative numbers.
- B) It ignores the magnitude of vectors.
- C) It fails to account for the cyclic continuity (0° = 360°).
- D) It assumes a Von Mises distribution.
> **Answer: C.** *The "wrap-around" effect at 360/0 means standard linear math produces opposite results (e.g., avg of 350° and 10° is 180° linearly, but 0° circularly).*

### Question 4: Factor Analysis Limitations
**When is Factor Analysis an inappropriate technique?**
- A) When variables are highly correlated (Multicollinearity).
- B) When KMO test value is 0.45.
- C) When N > 200.
- D) When searching for latent constructs.
> **Answer: B.** *KMO < 0.6 indicates "sampling inadequacy," meaning partial correlations are too small for FA to work reliably.*

