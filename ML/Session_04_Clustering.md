# Session 4 – Clustering Algorithms

## 📚 Table of Contents
1. [Clustering Overview](#clustering-overview)
2. [K-Means Clustering](#k-means-clustering)
3. [Hierarchical Clustering](#hierarchical-clustering)
4. [Distance Measures](#distance-measures)
5. [Scaling & Normalization](#scaling--normalization)
6. [Cluster Evaluation](#cluster-evaluation)
7. [MCQs](#mcqs)
8. [Common Mistakes](#common-mistakes)
9. [One-Line Exam Facts](#one-line-exam-facts)

---

# Clustering Overview

## 📘 Concept

**Clustering** is an unsupervised learning technique that groups similar data points together based on distance/similarity metrics.

### Goal
Partition data into k groups (clusters) such that:
- **High intra-cluster similarity**: Points within cluster are similar
- **Low inter-cluster similarity**: Points in different clusters are dissimilar

### Applications
- Customer segmentation (marketing)
- Image compression
- Document categorization
- Anomaly detection
- Gene expression analysis

### Challenges
- No ground truth labels
- Defining "similarity" is domain-dependent
- Choosing number of clusters k
- Sensitive to initialization and outliers

---

# K-Means Clustering

## 📘 Concept Overview

**K-Means** partitions data into k clusters by minimizing within-cluster variance.

## 🧮 Mathematical Foundation

### Objective Function

Minimize **Within-Cluster Sum of Squares (WCSS)**:

```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ‖x - μᵢ‖²
```

Where:
- k = number of clusters
- Cᵢ = set of points in cluster i
- μᵢ = centroid of cluster i
- ‖x - μᵢ‖² = squared Euclidean distance

### Centroid Update

```
μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x
```

(Mean of all points in cluster i)

## ⚙️ K-Means Algorithm

### Lloyd's Algorithm (Standard K-Means)

```
1. Initialize: Randomly select k centroids {μ₁, μ₂, ..., μₖ}

2. Repeat until convergence:
   a) Assignment Step:
      For each point xⱼ:
          Assign to cluster i* where i* = argminᵢ ‖xⱼ - μᵢ‖²
   
   b) Update Step:
      For each cluster i:
          μᵢ = mean of all points assigned to cluster i
   
3. Convergence when:
   - Centroids don't change
   - OR assignments don't change
   - OR max iterations reached
```

### Pseudocode

```python
def k_means(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Step 1: Initialize centroids (random samples)
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for iteration in range(max_iters):
        # Step 2a: Assignment
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Step 2b: Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids
```

## 🧠 Intuition

1. **Assignment**: Each point joins nearest centroid
2. **Update**: Centroids move to center of their cluster
3. **Iterate**: Repeat until stable

**Visual**:
```
Iteration 1:     Iteration 2:     Iteration 3:     Converged:
  +  +  +          +  +  +          +  +  +          +  +  +
 + C₁ +   →      + C₁ C₂  →       C₁  C₂   →       C₁  C₂
  +  C₂ +          +     +          +     +          +     +
```

## 🔄 Initialization Methods

### 1. Random Initialization

**Problem**: Can converge to poor local minima

**Solution**: Run multiple times with different initializations, keep best result

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
# Runs 10 times with different initializations, returns best
```

### 2. K-Means++ (Smart Initialization)

**Idea**: Choose initial centroids that are far apart

**Algorithm**:
```
1. Choose first centroid uniformly at random from data points
2. For i = 2 to k:
   - For each point x, compute distance D(x) to nearest existing centroid
   - Choose next centroid with probability ∝ D(x)²
3. Proceed with standard K-Means
```

**Advantage**: O(log k) approximation to optimal, faster convergence

```python
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
```

## ⚠️ Assumptions & Limitations

### Assumptions
1. **Spherical clusters**: Works best for round, convex clusters
2. **Similar sizes**: Struggles with very different cluster sizes
3. **Similar densities**: Assumes uniform density
4. **Known k**: Must specify number of clusters

### Failure Cases

#### 1. Non-Spherical Clusters
```
True clusters (moons):    K-Means result:
    ○○○○○                    ○○●●●
  ○○    ●●●●              ○○○  ●●●
 ○○      ●●              ○○      ●●
                         ✗ Wrong!
```

#### 2. Different Cluster Sizes
```
Large cluster + small:   K-Means:
 ●●●●●        ○          ●●●●●  ○○○
 ●●●●●        ○          ●●●●●  
 ●●●●●                   ✗ Splits large cluster
```

#### 3. Outliers

Outliers heavily influence centroid positions (mean is sensitive).

**Solution**: Use K-Medoids (median instead of mean)

## 📊 Complexity

- **Time**: O(n × k × i × d)
  - n = samples, k = clusters, i = iterations, d = dimensions
  - Typical i ≈ 10-100
- **Space**: O(n × d + k × d)

## 🧪 Python Implementation

### Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

# Results
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"Iterations: {kmeans.n_iter_}")

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=300, c='red', edgecolors='black', label='Centroids')
plt.legend()
plt.title('K-Means Clustering')
plt.show()
```

### From Scratch

```python
def kmeans_from_scratch(X, k, max_iters=100):
    """K-Means implementation from scratch."""
    n_samples, n_features = X.shape
    
    # Initialize centroids using k-means++
    centroids = kmeans_plusplus_init(X, k)
    
    for iteration in range(max_iters):
        # Assignment step
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Update step
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize randomly
                new_centroids[i] = X[np.random.randint(n_samples)]
        
        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            print(f"Converged in {iteration + 1} iterations")
            break
        
        centroids = new_centroids
    
    # Calculate inertia
    distances = np.sqrt(((X - centroids[labels]) ** 2).sum(axis=1))
    inertia = (distances ** 2).sum()
    
    return labels, centroids, inertia

def kmeans_plusplus_init(X, k):
    """K-means++ initialization."""
    n_samples = len(X)
    centroids = [X[np.random.randint(n_samples)]]
    
    for _ in range(1, k):
        # Distance to nearest centroid
        distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
        # Sample with probability proportional to distance squared
        probs = distances ** 2 / (distances ** 2).sum()
        centroids.append(X[np.random.choice(n_samples, p=probs)])
    
    return np.array(centroids)
```

## 🔍 Choosing k (Number of Clusters)

### 1. Elbow Method

Plot WCSS (inertia) vs k, look for "elbow" where rate of decrease slows.

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

### 2. Silhouette Score

Measures how similar a point is to its own cluster vs. other clusters.

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest other cluster

**Range**: -1 (wrong cluster) to +1 (perfect cluster)

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette={score:.3f}")

# Choose k with highest silhouette score
```

### 3. Gap Statistic

Compares WCSS to expected WCSS under null reference distribution.

---

# Hierarchical Clustering

## 📘 Concept Overview

**Hierarchical Clustering** builds a tree of clusters (dendrogram) without specifying k upfront.

### Types

1. **Agglomerative (Bottom-Up)**: Start with individual points, merge clusters
2. **Divisive (Top-Down)**: Start with all points, recursively split

**Agglomerative is more common.**

## ⚙️ Agglomerative Clustering Algorithm

```
1. Start: Each point is its own cluster (n clusters)

2. Repeat until 1 cluster remains:
   a) Find two closest clusters Cᵢ, Cⱼ
   b) Merge Cᵢ and Cⱼ into single cluster
   c) Update distance matrix

3. Result: Dendrogram (tree structure)
```

## 🧮 Linkage Criteria

How to measure distance between clusters?

### 1. Single Linkage (MIN)

```
d(Cᵢ, Cⱼ) = min {d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
```

**Pros**: Can handle non-spherical clusters
**Cons**: Sensitive to noise, "chaining" effect

### 2. Complete Linkage (MAX)

```
d(Cᵢ, Cⱼ) = max {d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
```

**Pros**: Less sensitive to outliers
**Cons**: Tends to break large clusters

### 3. Average Linkage

```
d(Cᵢ, Cⱼ) = (1/(|Cᵢ||Cⱼ|)) Σₓ∈Cᵢ Σᵧ∈Cⱼ d(x, y)
```

**Pros**: Balanced approach
**Cons**: Computationally expensive

### 4. Ward's Linkage

Minimize increase in total within-cluster variance.

```
d(Cᵢ, Cⱼ) = increase in WCSS if Cᵢ and Cⱼ are merged
```

**Pros**: Produces balanced clusters, works well in practice
**Cons**: Assumes Euclidean distance, convex clusters

## 📊 Dendrogram

Visual representation of hierarchical clustering.

```
     Height
       │
   10  ├─────────┐
       │         │
    8  ├───┐     │
       │   │     │
    5  │   ├──┐  │
       │   │  │  │
    2  ├─┐ │  │  │
       │ │ │  │  │
    0  A B C  D  E
```

**Cut at different heights** to get different numbers of clusters.

## 🧪 Python Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative clustering
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clust.fit_predict(X)

# Dendrogram
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

## 📊 Comparison: K-Means vs Hierarchical

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| **k required** | Yes (upfront) | No (cut dendrogram later) |
| **Cluster shape** | Spherical | Arbitrary (with right linkage) |
| **Time complexity** | O(nkid) | O(n² log n) or O(n³) |
| **Space complexity** | O(n) | O(n²) |
| **Deterministic** | No (random init) | Yes |
| **Large datasets** | ✓ Suitable | ✗ Too slow |
| **Interpretability** | Low | High (dendrogram) |

---

# Distance Measures

## 📘 Overview

Distance/similarity metrics are **critical** for clustering quality.

## 🧮 Common Distance Metrics

### 1. Euclidean Distance (L2)

```
d(x, y) = √[Σᵢ (xᵢ - yᵢ)²]
```

**Most common**, assumes continuous features, sensitive to scale.

### 2. Manhattan Distance (L1)

```
d(x, y) = Σᵢ |xᵢ - yᵢ|
```

**More robust to outliers** than Euclidean.

### 3. Cosine Distance

```
cosine_similarity(x, y) = (x · y) / (‖x‖ ‖y‖)
cosine_distance(x, y) = 1 - cosine_similarity(x, y)
```

**Measures angle**, not magnitude. Used for text (TF-IDF vectors).

### 4. Minkowski Distance (General Form)

```
d(x, y) = (Σᵢ |xᵢ - yᵢ|ᵖ)^(1/p)
```

- p = 1: Manhattan
- p = 2: Euclidean
- p → ∞: Chebyshev (max difference)

### 5. Hamming Distance

Number of positions where bits differ (for categorical/binary data).

```
d(x, y) = Σᵢ I(xᵢ ≠ yᵢ)
```

### 6. Jaccard Distance (Sets)

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Used for binary/categorical features.

## 🧪 Python Implementation

```python
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# Euclidean
dist_euclidean = euclidean_distances(X)

# Cosine
dist_cosine = cosine_distances(X)

# Custom distance in KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Use precomputed distance matrix
D = pairwise_distances(X, metric='manhattan')
# Note: KMeans doesn't support precomputed, but DBSCAN does
```

---

# Scaling & Normalization

## 📘 Why Scale?

**Problem**: Features with large ranges dominate distance calculations.

**Example**:
- Feature 1 (age): 20-80 (range = 60)
- Feature 2 (income): 20,000-100,000 (range = 80,000)

Distance dominated by income!

## 🧮 Scaling Methods

### 1. Standardization (Z-score)

```
x' = (x - μ) / σ
```

**Result**: Mean = 0, Std = 1

**Use**: When features have different units, Gaussian assumption

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Min-Max Normalization

```
x' = (x - min) / (max - min)
```

**Result**: Range [0, 1]

**Use**: When need bounded range, neural networks

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 3. Robust Scaling

```
x' = (x - median) / IQR
```

**Result**: Robust to outliers

**Use**: When data has many outliers

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

### 4. L2 Normalization (Unit Norm)

```
x' = x / ‖x‖₂
```

**Result**: Each sample has unit norm

**Use**: Text data, cosine similarity

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
X_normed = normalizer.fit_transform(X)
```

## ⚠️ When to Scale

| Algorithm | Scaling Required? |
|-----------|-------------------|
| **K-Means** | ✓ Yes (uses Euclidean distance) |
| **Hierarchical** | ✓ Yes (distance-based) |
| **DBSCAN** | ✓ Yes (density + distance) |
| **Decision Trees** | ✗ No (split-based, scale-invariant) |
| **Neural Networks** | ✓ Yes (gradient descent) |

---

# Cluster Evaluation

## 📘 Challenge

**No ground truth labels** → Hard to evaluate quality.

## 🧮 Intrinsic Metrics (No Labels)

### 1. Silhouette Score

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Range**: [-1, 1]
- +1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong cluster assignment

```python
from sklearn.metrics import silhouette_score, silhouette_samples

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")

# Per-sample silhouette
sample_scores = silhouette_samples(X, labels)
```

### 2. Davies-Bouldin Index

Average similarity ratio of each cluster with most similar cluster.

```
DB = (1/k) Σᵢ max_{j≠i} [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]
```

**Lower is better** (0 is perfect).

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin: {score:.3f}")
```

### 3. Calinski-Harabasz Index (Variance Ratio)

Ratio of between-cluster to within-cluster variance.

**Higher is better**.

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz: {score:.3f}")
```

## 🧮 Extrinsic Metrics (With Ground Truth)

### 1. Adjusted Rand Index (ARI)

Measures agreement between true and predicted labels.

**Range**: [-1, 1]
- 1: Perfect match
- 0: Random labeling
- Negative: Worse than random

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, y_pred)
print(f"ARI: {ari:.3f}")
```

### 2. Normalized Mutual Information (NMI)

Information-theoretic metric.

**Range**: [0, 1]
- 1: Perfect match
- 0: Independent

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"NMI: {nmi:.3f}")
```

---

# 🔥 MCQs

### Q1. K-Means minimizes which objective?
**Options:**
- A) Between-cluster variance
- B) Within-cluster sum of squares ✓
- C) Total variance
- D) Number of clusters

**Explanation**: K-Means minimizes WCSS (inertia), sum of squared distances to centroids.

---

### Q2. K-Means++ improves over random initialization by:
**Options:**
- A) Choosing centroids close together
- B) Choosing centroids far apart ✓
- C) Using median instead of mean
- D) Increasing number of iterations

**Explanation**: K-Means++ selects initial centroids with probability proportional to distance squared.

---

### Q3. Which linkage is most sensitive to outliers?
**Options:**
- A) Complete linkage
- B) Average linkage
- C) Single linkage ✓
- D) Ward's linkage

**Explanation**: Single linkage uses minimum distance, heavily influenced by outlier pairs.

---

### Q4. Time complexity of K-Means is:
**Options:**
- A) O(n² log n)
- B) O(nkid) ✓
- C) O(n³)
- D) O(k²)

**Explanation**: n samples, k clusters, i iterations, d dimensions.

---

### Q5. Silhouette score of 0 indicates:
**Options:**
- A) Perfect clustering
- B) Overlapping clusters ✓
- C) Wrong assignment
- D) Too many clusters

**Explanation**: 0 means point is on boundary between clusters (ambiguous).

---

### Q6. Which distance metric is appropriate for text features?
**Options:**
- A) Euclidean
- B) Manhattan
- C) Cosine ✓
- D) Hamming

**Explanation**: Cosine measures angle (semantic similarity), not magnitude.

---

### Q7. Standardization transforms features to have:
**Options:**
- A) Range [0, 1]
- B) Mean = 0, Std = 1 ✓
- C) Median = 0
- D) Max = 1

**Explanation**: Z-score normalization: (x - μ) / σ

---

### Q8. K-Means assumes:
**Options:**
- A) Non-convex clusters
- B) Spherical clusters ✓
- C) Hierarchical structure
- D) Different densities

**Explanation**: K-Means works best for round, convex, similar-size clusters.

---

### Q9. Which metric does NOT require ground truth labels?
**Options:**
- A) Adjusted Rand Index
- B) Normalized Mutual Information
- C) Silhouette Score ✓
- D) All require labels

**Explanation**: Silhouette uses only distances, no labels needed.

---

### Q10. Ward's linkage minimizes:
**Options:**
- A) Maximum distance
- B) Minimum distance
- C) Increase in within-cluster variance ✓
- D) Between-cluster variance

**Explanation**: Ward merges clusters that minimize increase in total WCSS.

---

### Q11. Empty cluster in K-Means occurs when:
**Options:**
- A) k is too large
- B) No points assigned to centroid ✓
- C) Outliers present
- D) Data not scaled

**Explanation**: If centroid is far from all points, cluster can be empty after assignment.

---

### Q12. Dendrogram is used in:
**Options:**
- A) K-Means
- B) DBSCAN
- C) Hierarchical clustering ✓
- D) Gaussian Mixture Models

**Explanation**: Dendrogram visualizes hierarchical cluster merging.

---

### Q13. Elbow method helps determine:
**Options:**
- A) Best linkage
- B) Optimal number of clusters ✓
- C) Distance metric
- D) Scaling method

**Explanation**: Plot inertia vs k, look for "elbow" where decrease slows.

---

### Q14. Which algorithm is deterministic?
**Options:**
- A) K-Means with random init
- B) Agglomerative clustering ✓
- C) K-Means with k-means++ (single run)
- D) DBSCAN with random sampling

**Explanation**: Agglomerative always produces same result (no randomness).

---

### Q15. Cosine distance measures:
**Options:**
- A) Magnitude difference
- B) Angle between vectors ✓
- C) Manhattan distance
- D) Euclidean distance

**Explanation**: Cosine similarity = cos(θ), independent of magnitude.

---

# ⚠️ Common Mistakes

1. **Not scaling features**: Dominates clustering with high-range features

2. **Assuming K-Means finds global optimum**: Only guaranteed local optimum

3. **Using Euclidean for categorical data**: Use Hamming or Jaccard instead

4. **Choosing k arbitrarily**: Use elbow method or silhouette score

5. **Applying K-Means to non-spherical data**: Use DBSCAN or hierarchical instead

6. **Ignoring outliers**: Heavily influence K-Means centroids

7. **Not running multiple initializations**: Single run may converge to poor solution

8. **Using silhouette with only 2-3 clusters**: Can be misleading for small k

9. **Comparing clusters across different k**: Metrics not directly comparable

10. **Scaling after clustering**: Must scale BEFORE clustering

---

# ⭐ One-Line Exam Facts

1. **K-Means minimizes WCSS** (within-cluster sum of squares)

2. **K-Means++ initialization** selects centroids far apart (O(log k) approximation)

3. **K-Means assumes spherical, similar-size clusters** with Euclidean distance

4. **Time complexity**: K-Means O(nkid), Hierarchical O(n² log n)

5. **Single linkage** (MIN) sensitive to outliers, **complete linkage** (MAX) breaks large clusters

6. **Ward's linkage minimizes increase in WCSS** when merging clusters

7. **Dendrogram** visualizes hierarchical clustering (cut at different heights for different k)

8. **Silhouette score [-1, 1]**: +1 perfect, 0 overlapping, -1 wrong

9. **Standardization**: mean=0, std=1 via (x-μ)/σ

10. **Min-Max normalization**: range [0,1] via (x-min)/(max-min)

11. **Cosine distance** measures angle (text/high-dim), **Euclidean** measures magnitude

12. **Elbow method**: plot inertia vs k, choose elbow point

13. **Empty cluster** in K-Means requires reinitialization (centroid too far)

14. **Scaling required** for K-Means, Hierarchical, DBSCAN (distance-based)

15. **ARI and NMI** require ground truth; **Silhouette and Davies-Bouldin** don't

---

**End of Session 4**
