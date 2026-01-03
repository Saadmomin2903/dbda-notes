# Session 15 – Recommendation Systems

## 📚 Table of Contents
1. [Recommendation Systems Overview](#recommendation-systems-overview)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Content-Based Filtering](#content-based-filtering)
4. [Matrix Factorization](#matrix-factorization)
5. [Hybrid Methods](#hybrid-methods)
6. [Evaluation Metrics](#evaluation-metrics)
7. [MCQs](#mcqs)
8. [Common Mistakes](#common-mistakes)
9. [One-Line Exam Facts](#one-line-exam-facts)

---

# Recommendation Systems Overview

## 📘 Concept Overview

**Recommendation Systems** predict user preferences and suggest relevant items.

**Goal**: Predict rating r_{ui} that user u would give to item i.

## 🧮 Types of Recommendation Systems

### 1. Collaborative Filtering
Uses **user-item interactions** (ratings, clicks, purchases).

**Assumption**: Similar users like similar items.

### 2. Content-Based Filtering
Uses **item features** (genre, description, tags).

**Assumption**: Users like items similar to what they liked before.

### 3. Hybrid Methods
Combines collaborative and content-based approaches.

## 📊 Use Cases

| Domain | Items | Signals |
|--------|-------|---------|
| **E-commerce** | Products | Purchases, views, ratings |
| **Streaming** | Movies/Music | Watches, ratings, skips |
| **Social Media** | Posts, Friends | Likes, shares, clicks |
| **News** | Articles | Clicks, time spent |

---

# Collaborative Filtering

## 📘 User-Based Collaborative Filtering

**Idea**: Find similar users, recommend what they liked.

## 🧮 Mathematical Foundation

### User Similarity

**Cosine Similarity**:
```
sim(u, v) = (r_u · r_v) / (‖r_u‖ ‖r_v‖)
```

Where r_u = vector of user u's ratings

**Pearson Correlation**:
```
sim(u, v) = Σᵢ(r_{ui} - r̄_u)(r_{vi} - r̄_v) / 
            √[Σᵢ(r_{ui} - r̄_u)²] √[Σᵢ(r_{vi} - r̄_v)²]
```

### Prediction

Predict rating of user u for item i:

```
r̂_{ui} = r̄_u + Σ_{v∈N(u)} sim(u,v) × (r_{vi} - r̄_v) / Σ_{v∈N(u)} |sim(u,v)|
```

Where:
- N(u) = k most similar users to u who rated item i
- r̄_u = average rating of user u

## 🧪 Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item rating matrix
ratings_data = {
    'User1': [5, 3, 0, 1, 4],
    'User2': [4, 0, 0, 1, 5],
    'User3': [1, 1, 0, 5, 4],
    'User4': [0, 0, 5, 4, 0],
    'User5': [0, 3, 4, 0, 0]
}

ratings_df = pd.DataFrame(ratings_data, 
                         index=['Item1', 'Item2', 'Item3', 'Item4', 'Item5']).T

print("User-Item Rating Matrix:")
print(ratings_df)

# Compute user similarity (cosine)
# Replace 0 with NaN for proper similarity calculation
ratings_filled = ratings_df.replace(0, np.nan)

# Compute similarity on non-zero ratings
user_similarity = cosine_similarity(ratings_df.fillna(0))
user_sim_df = pd.DataFrame(user_similarity, 
                           index=ratings_df.index,
                           columns=ratings_df.index)

print(f"\nUser Similarity Matrix:")
print(user_sim_df)

def predict_rating_user_based(user, item, ratings, user_sim, k=2):
    """Predict rating using user-based CF."""
    # Get users who rated this item
    item_ratings = ratings[item]
    rated_users = item_ratings[item_ratings > 0].index
    
    if user not in ratings.index:
        return ratings[item].mean()
    
    # Get similarity scores
    similarities = user_sim.loc[user, rated_users]
    
    # Get top-k similar users
    top_k = similarities.nlargest(k)
    
    if top_k.sum() == 0:
        return ratings.loc[user].mean()
    
    # Weighted average
    user_mean = ratings.loc[user].replace(0, np.nan).mean()
    
    weighted_sum = 0
    sim_sum = 0
    
    for similar_user in top_k.index:
        if similar_user != user:
            sim = top_k[similar_user]
            rating = item_ratings[similar_user]
            similar_user_mean = ratings.loc[similar_user].replace(0, np.nan).mean()
            
            weighted_sum += sim * (rating - similar_user_mean)
            sim_sum += abs(sim)
    
    if sim_sum == 0:
        return user_mean
    
    prediction = user_mean + weighted_sum / sim_sum
    return np.clip(prediction, 1, 5)

# Predict User1's rating for Item3
pred = predict_rating_user_based('User1', 'Item3', ratings_df, user_sim_df, k=2)
print(f"\nPredicted rating for User1, Item3: {pred:.2f}")
```

## 📘 Item-Based Collaborative Filtering

**Idea**: Find similar items, recommend based on what user liked.

**Advantage**: Item similarities more stable than user similarities.

### Item Similarity

Same formulas but applied to items (columns instead of rows).

### Prediction

```
r̂_{ui} = Σ_{j∈N(i)} sim(i,j) × r_{uj} / Σ_{j∈N(i)} |sim(i,j)|
```

Where N(i) = k most similar items to i that user u rated.

## 🧪 Python Implementation

```python
# Compute item similarity
item_similarity = cosine_similarity(ratings_df.T.fillna(0))
item_sim_df = pd.DataFrame(item_similarity,
                           index=ratings_df.columns,
                           columns=ratings_df.columns)

print("Item Similarity Matrix:")
print(item_sim_df)

def predict_rating_item_based(user, item, ratings, item_sim, k=2):
    """Predict rating using item-based CF."""
    if user not in ratings.index:
        return ratings[item].mean()
    
    # Get items rated by this user
    user_ratings = ratings.loc[user]
    rated_items = user_ratings[user_ratings > 0].index
    
    if len(rated_items) == 0:
        return ratings[item].mean()
    
    # Get similarity scores
    similarities = item_sim.loc[item, rated_items]
    
    # Get top-k similar items
    top_k = similarities.nlargest(k)
    
    if top_k.sum() == 0:
        return user_ratings.mean()
    
    # Weighted average
    weighted_sum = 0
    sim_sum = 0
    
    for similar_item in top_k.index:
        if similar_item != item:
            sim = top_k[similar_item]
            rating = user_ratings[similar_item]
            
            weighted_sum += sim * rating
            sim_sum += abs(sim)
    
    if sim_sum == 0:
        return user_ratings.replace(0, np.nan).mean()
    
    prediction = weighted_sum / sim_sum
    return np.clip(prediction, 1, 5)

# Predict
pred_item = predict_rating_item_based('User1', 'Item3', ratings_df, item_sim_df, k=2)
print(f"\nItem-based prediction for User1, Item3: {pred_item:.2f}")
```

## 📊 User-Based vs Item-Based

| Aspect | User-Based | Item-Based |
|--------|------------|-----------|
| **Scalability** | Poor (many users) | ✓ Better (fewer items) |
| **Stability** | User preferences change | ✓ Item features stable |
| **Serendipity** | ✓ More diverse recommendations | Less diverse |
| **Cold start** | Difficult for new users | ✓ Better for new users |
| **Use case** | Small user base | Large user base (e.g., Amazon) |

---

# Content-Based Filtering

## 📘 Concept Overview

**Idea**: Recommend items similar to what user liked, based on **item features**.

## 🧮 Mathematical Foundation

### Item Profile

Represent item as feature vector:
```
i = [genre₁, genre₂, ..., actor₁, actor₂, ...]
```

**TF-IDF** commonly used for text features.

### User Profile

Aggregate features of items user liked:
```
u = Σᵢ r_{ui} × i / Σᵢ r_{ui}
```

Weighted average of item profiles.

### Prediction

**Cosine similarity** between user profile and item:
```
r̂_{ui} = sim(u, i) = u · i / (‖u‖ ‖i‖)
```

## 🧪 Python Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Item descriptions
items = {
    'Movie1': 'action adventure sci-fi',
    'Movie2': 'romance drama love',
    'Movie3': 'action thriller crime',
    'Movie4': 'comedy romance',
    'Movie5': 'sci-fi thriller space'
}

# User ratings
user_ratings = {
    'Movie1': 5,
    'Movie2': 2,
    'Movie3': 4,
    'Movie5': 5
}

# TF-IDF vectorization
tfidf = TfidfVectorizer()
item_features = tfidf.fit_transform(items.values())

# Create user profile (weighted average)
user_profile = np.zeros(item_features.shape[1])
total_weight = 0

for movie, rating in user_ratings.items():
    idx = list(items.keys()).index(movie)
    user_profile += rating * item_features[idx].toarray()[0]
    total_weight += rating

user_profile = user_profile / total_weight
user_profile = user_profile.reshape(1, -1)

# Compute similarity with all items
similarities = cosine_similarity(user_profile, item_features)[0]

# Recommend items
recommendations = pd.DataFrame({
    'Movie': items.keys(),
    'Score': similarities
}).sort_values('Score', ascending=False)

print("Content-Based Recommendations:")
print(recommendations)
```

## 📊 Advantages & Disadvantages

### Advantages ✓
1. **No cold start for items**: Can recommend new items
2. **Transparency**: Explainable (similar features)
3. **No user data needed**: Works for single user
4. **Personalized**: Tailored to individual preferences

### Disadvantages ✗
1. **Limited novelty**: Only recommends similar items
2. **Feature engineering**: Requires good item features
3. **Cold start for users**: Need user history
4. **Overspecialization**: Filter bubble effect

---

# Matrix Factorization

## 📘 Concept Overview

**Idea**: Decompose user-item matrix into **latent factors**.

## 🧮 Mathematical Foundation

### Matrix Decomposition

```
R ≈ U × V^T
```

Where:
- R ∈ ℝ^{m×n} = rating matrix (m users, n items)
- U ∈ ℝ^{m×k} = user latent factors
- V ∈ ℝ^{n×k} = item latent factors
- k = number of latent dimensions

**Prediction**:
```
r̂_{ui} = u_u^T v_i = Σⱼ u_{uj} v_{ij}
```

### Objective Function

Minimize squared error + regularization:

```
L = Σ_{(u,i)∈observed} (r_{ui} - u_u^T v_i)² + λ(‖u_u‖² + ‖v_i‖²)
```

**Regularization** prevents overfitting.

### Optimization: Alternating Least Squares (ALS)

```
Repeat until convergence:
  1. Fix V, solve for U
  2. Fix U, solve for V
```

Each step is least squares problem (closed-form solution).

### Optimization: Stochastic Gradient Descent (SGD)

```
For each rating r_{ui}:
  e_{ui} = r_{ui} - u_u^T v_i
  
  u_u ← u_u + α(e_{ui} × v_i - λ × u_u)
  v_i ← v_i + α(e_{ui} × u_u - λ × v_i)
```

## 🧪 Python Implementation

### Using Surprise Library

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Create dataset
data_dict = {
    'userID': ['User1', 'User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4'],
    'itemID': ['Item1', 'Item2', 'Item4', 'Item1', 'Item4', 'Item1', 'Item4', 'Item3'],
    'rating': [5, 3, 1, 4, 1, 1, 5, 5]
}

df = pd.DataFrame(data_dict)

# Define Reader
reader = Reader(rating_scale=(1, 5))

# Load dataset
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# SVD algorithm (Matrix Factorization)
algo = SVD(n_factors=2, lr_all=0.005, reg_all=0.02, n_epochs=100)

# Cross-validation
cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

print(f"Mean RMSE: {cv_results['test_rmse'].mean():.3f}")
print(f"Mean MAE: {cv_results['test_mae'].mean():.3f}")

# Train on full dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict
prediction = algo.predict('User1', 'Item3')
print(f"\nPredicted rating for User1, Item3: {prediction.est:.2f}")

# Get latent factors
print(f"\nUser1 latent factors: {algo.pu[trainset.to_inner_uid('User1')]}")
print(f"Item3 latent factors: {algo.qi[trainset.to_inner_iid('Item3')]}")
```

### From Scratch (SGD)

```python
class MatrixFactorization:
    def __init__(self, n_factors=10, lr=0.01, reg=0.02, n_epochs=100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
    
    def fit(self, ratings_matrix):
        """Fit using SGD."""
        n_users, n_items = ratings_matrix.shape
        
        # Initialize latent factors
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Get observed ratings
        observed = np.argwhere(~np.isnan(ratings_matrix))
        
        # SGD
        for epoch in range(self.n_epochs):
            np.random.shuffle(observed)
            
            for u, i in observed:
                r_ui = ratings_matrix[u, i]
                
                # Prediction
                pred = self.U[u] @ self.V[i]
                
                # Error
                error = r_ui - pred
                
                # Update
                u_update = self.lr * (error * self.V[i] - self.reg * self.U[u])
                v_update = self.lr * (error * self.U[u] - self.reg * self.V[i])
                
                self.U[u] += u_update
                self.V[i] += v_update
            
            # Compute RMSE
            if epoch % 10 == 0:
                preds = self.U @ self.V.T
                mask = ~np.isnan(ratings_matrix)
                rmse = np.sqrt(np.mean((ratings_matrix[mask] - preds[mask])**2))
                print(f"Epoch {epoch}: RMSE = {rmse:.3f}")
        
        return self
    
    def predict(self, u, i):
        """Predict rating."""
        return self.U[u] @ self.V[i]

# Convert to numpy matrix
ratings_matrix = ratings_df.values.astype(float)
ratings_matrix[ratings_matrix == 0] = np.nan

# Fit
mf = MatrixFactorization(n_factors=2, lr=0.01, reg=0.02, n_epochs=50)
mf.fit(ratings_matrix)

# Predict
user_idx = 0  # User1
item_idx = 2  # Item3
pred_mf = mf.predict(user_idx, item_idx)
print(f"\nMF prediction for User1, Item3: {pred_mf:.2f}")
```

## 📊 Variants

### SVD++ (Singular Value Decomposition Plus Plus)

Adds **implicit feedback**:
```
r̂_{ui} = μ + b_u + b_i + u_u^T(v_i + |I_u|^{-0.5} Σ_{j∈I_u} y_j)
```

Where I_u = items user u interacted with.

### NMF (Non-Negative Matrix Factorization)

**Constraint**: U, V ≥ 0

**Advantage**: More interpretable factors.

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=2, init='random', random_state=42)
U_nmf = nmf.fit_transform(ratings_df.fillna(0))
V_nmf = nmf.components_
```

---

# Hybrid Methods

## 📊 Combination Strategies

### 1. Weighted Hybrid

```
r̂_{ui} = α × r̂_{CF} + (1-α) × r̂_{CB}
```

Weighted combination of collaborative and content-based.

### 2. Switching Hybrid

Use different methods based on context:
- New item → Content-based
- Established item → Collaborative

### 3. Feature Combination

Use content features as input to collaborative model.

### 4. Cascade Hybrid

Refine recommendations from one method using another.

## 🧪 Example: Weighted Hybrid

```python
def hybrid_prediction(user, item, cf_pred, cb_pred, alpha=0.7):
    """Weighted hybrid recommendation."""
    return alpha * cf_pred + (1 - alpha) * cb_pred

# Combine predictions
hybrid_pred = hybrid_prediction('User1', 'Item3', pred, 0.7, alpha=0.7)
print(f"Hybrid prediction: {hybrid_pred:.2f}")
```

---

# Evaluation Metrics

## 📊 Rating Prediction Metrics

### RMSE (Root Mean Squared Error)

```
RMSE = √[(1/n) Σ(r_{ui} - r̂_{ui})²]
```

**Lower = better**

### MAE (Mean Absolute Error)

```
MAE = (1/n) Σ|r_{ui} - r̂_{ui}|
```

**More robust to outliers** than RMSE.

## 📊 Ranking Metrics

### Precision@K

```
Precision@K = (Relevant items in top-K) / K
```

### Recall@K

```
Recall@K = (Relevant items in top-K) / (Total relevant items)
```

### NDCG (Normalized Discounted Cumulative Gain)

```
DCG@K = Σᵢ (2^{rel_i} - 1) / log₂(i + 1)

NDCG@K = DCG@K / IDCG@K
```

Where IDCG = ideal DCG (perfect ranking).

## 🧪 Python Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# True and predicted ratings
y_true = np.array([5, 3, 4, 2, 5])
y_pred = np.array([4.5, 3.2, 3.8, 2.1, 4.8])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
```

---

# 🔥 MCQs

### Q1. Collaborative filtering uses:
**Options:**
- A) Item features only
- B) User-item interactions ✓
- C) Content metadata
- D) Demographics

**Explanation**: CF relies on ratings/interactions between users and items.

---

### Q2. Content-based filtering recommends based on:
**Options:**
- A) Similar users
- B) Item features ✓
- C) Popularity
- D) Ratings only

**Explanation**: Uses item attributes (genre, tags, description).

---

### Q3. Matrix factorization decomposes R as:
**Options:**
- A) R = U + V
- B) R = U × V^T ✓
- C) R = U - V
- D) R = UV

**Explanation**: R ≈ UV^T where U=user factors, V=item factors.

---

### Q4. Cold start problem refers to:
**Options:**
- A) Slow algorithm
- B) New users/items with no history ✓
- C) Server issues
- D) Large datasets

**Explanation**: Hard to recommend for new users/items without data.

---

### Q5. Item-based CF is better than user-based for:
**Options:**
- A) Small item catalog
- B) Large user base ✓
- C) New users
- D) Personalization

**Explanation**: Item similarities more stable; scales better.

---

### Q6. Cosine similarity range is:
**Options:**
- A) [0, 1]
- B) [-1, 1] ✓
- C) [0, ∞)
- D) (-∞, ∞)

**Explanation**: Cosine of angle between vectors ∈ [-1, 1].

---

### Q7. In matrix factorization, k (latent factors) controls:
**Options:**
- A) Number of users
- B) Model complexity ✓
- C) Number of items
- D) Rating scale

**Explanation**: Higher k = more complex (risk overfitting).

---

### Q8. SVD++ adds:
**Options:**
- A) More users
- B) Implicit feedback ✓
- C) Content features
- D) Time dynamics

**Explanation**: Incorporates items user interacted with (beyond ratings).

---

### Q9. RMSE vs MAE:
**Options:**
- A) RMSE more robust to outliers
- B) MAE more robust to outliers ✓
- C) Identical
- D) RMSE always higher

**Explanation**: MAE uses absolute error (less sensitive to large errors).

---

### Q10. Precision@10 measures:
**Options:**
- A) Total recommendations
- B) Relevant items in top 10 / 10 ✓
- C) All relevant items found
- D) Rating accuracy

**Explanation**: Fraction of top-K that are relevant.

---

### Q11. Hybrid recommender combines:
**Options:**
- A) Multiple algorithms ✓
- B) Multiple datasets
- C) Multiple users
- D) Multiple items

**Explanation**: E.g., CF + content-based for better performance.

---

### Q12. NMF constraint is:
**Options:**
- A) U, V ≤ 0
- B) U, V ≥ 0 ✓
- C) U = V
- D) No constraint

**Explanation**: Non-negative matrix factorization requires U, V ≥ 0.

---

### Q13. Content-based filtering suffers from:
**Options:**
- A) Cold start for items
- B) Overspecialization (filter bubble) ✓
- C) Scalability
- D) Sparsity

**Explanation**: Only recommends similar items (limited diversity).

---

### Q14. ALS in matrix factorization:
**Options:**
- A) Alternates fixing U and V ✓
- B) Uses gradient descent
- C) Requires labels
- D) Non-iterative

**Explanation**: Alternating Least Squares fixes one, solves for other.

---

### Q15. NDCG accounts for:
**Options:**
- A) Position in ranking ✓
- B) Total items
- C) User count
- D) Rating scale

**Explanation**: Discounted Cumulative Gain weights by position.

---

# ⚠️ Common Mistakes

1. **Not handling sparsity**: Most user-item matrices very sparse (use matrix factorization)

2. **Ignoring cold start**: Need strategies for new users/items

3. **Using only popularity**: Personalization lost; use collaborative methods

4. **Not regularizing matrix factorization**: Overfitting to sparse data

5. **Wrong similarity metric**: Cosine for sparse, Pearson for dense data

6. **Forgetting to normalize**: Different users use rating scales differently

7. **Too many latent factors**: Overfitting (start with k=10-50)

8. **Not handling implicit feedback**: Clicks, views as important as ratings

9. **Evaluating on training data**: Always use held-out test set

10. **Ignoring temporal dynamics**: User preferences change over time

---

# ⭐ One-Line Exam Facts

1. **Collaborative filtering**: Uses user-item interactions (ratings, clicks)

2. **Content-based**: Uses item features (genre, tags, description)

3. **User-based CF**: Find similar users, recommend what they liked

4. **Item-based CF**: Find similar items, recommend based on user's history

5. **Matrix factorization**: R ≈ U × V^T (U=user, V=item latent factors)

6. **Cold start**: Problem for new users/items without history

7. **Cosine similarity**: sim(u,v) = u·v / (‖u‖‖v‖) ∈ [-1, 1]

8. **Hybrid methods**: Combine collaborative + content-based

9. **RMSE**: √[mean squared error] (penalizes large errors more)

10. **MAE**: Mean absolute error (more robust to outliers)

11. **ALS**: Alternating Least Squares (fix U, solve V; fix V, solve U)

12. **SGD for MF**: Update U and V using gradient descent

13. **Regularization**: Prevents overfitting in matrix factorization

14. **NMF**: Non-negative matrix factorization (U, V ≥ 0)

15. **NDCG**: Ranking metric that weights by position

---

**End of Session 15**

**Progress: 15/30 sessions (50% complete)!** Halfway through the comprehensive ML notes. Ready to continue with remaining 15 sessions.
