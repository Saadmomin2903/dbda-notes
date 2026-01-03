# Practical Machine Learning - Full Mock Exam (100 Questions)

## 📋 Exam Instructions

**Total Time**: 3 hours  
**Total Marks**: 100  
**Passing Marks**: 40

### Section Breakdown:
- **Section A: Fundamentals & Theory** (30 questions, 45 minutes) - Questions 1-30
- **Section B: Algorithms & Implementation** (40 questions, 90 minutes) - Questions 31-70
- **Section C: Applied ML & Advanced Topics** (30 questions, 45 minutes) - Questions 71-100

### Difficulty Distribution:
- 🟢 Easy: 30 questions (1 mark each)
- 🟡 Medium: 50 questions (1 mark each)
- 🔴 Hard: 20 questions (1 mark each)

### Instructions:
1. All questions are **multiple choice** (single correct answer)
2. **No negative marking**
3. Calculator allowed for Section B & C
4. Read questions carefully - many have tricky options!

---

# SECTION A: Fundamentals & Theory (30 Questions, 45 Minutes)

## Q1. 🟢 What is the primary goal of supervised learning?
**A)** Find patterns in unlabeled data  
**B)** Learn a mapping from input to output using labeled examples  
**C)** Maximize reward through trial and error  
**D)** Reduce dimensionality of data

**Answer**: B  
**Explanation**: Supervised learning learns from labeled data (X, y) to predict outcomes.

---

## Q2. 🟡 In bias-variance tradeoff, which statement is TRUE?
**A)** High bias models overfit training data  
**B)** High variance models underfit training data  
**C)** Increasing model complexity always reduces total error  
**D)** Optimal model balances bias and variance

**Answer**: D  
**Explanation**: Bias-variance tradeoff: Total Error = Bias² + Variance + Irreducible Error. Need to balance both.

---

## Q3. 🟢 Cross-validation is used to:
**A)** Train the final model  
**B)** Estimate model performance on unseen data  
**C)** Replace the test set  
**D)** Increase training data size

**Answer**: B  
**Explanation**: CV provides robust performance estimate by using multiple train/val splits.

---

## Q4. 🟡 Which is NOT a distance metric?
**A)** Euclidean distance  
**B)** Manhattan distance  
**C)** Correlation coefficient  
**D)** Cosine distance

**Answer**: C  
**Explanation**: Correlation coefficient measures linear relationship, not distance. All others are valid distance metrics.

---

## Q5. 🔴 What is the Vapnik-Chervonenkis (VC) dimension?
**A)** Number of features in model  
**B)** Capacity of a model (max points it can shatter)  
**C)** Number of support vectors in SVM  
**D)** Depth of decision tree

**Answer**: B  
**Explanation**: VC dimension measures model capacity - maximum number of points the model can correctly classify for all possible labelings.

---

## Q6. 🟢 In k-NN, increasing k generally:
**A)** Increases variance  
**B)** Decreases variance  
**C)** Has no effect on variance  
**D)** Always improves accuracy

**Answer**: B  
**Explanation**: Larger k → smoother decision boundary → lower variance (but higher bias).

---

## Q7. 🟡 Which metric is most suitable for imbalanced classification?
**A)** Accuracy  
**B)** F1-score  
**C)** Mean Squared Error  
**D)** R² score

**Answer**: B  
**Explanation**: F1-score balances precision and recall, better for imbalanced data than accuracy.

---

## Q8. 🟢 PCA is:
**A)** Supervised dimensionality reduction  
**B)** Unsupervised dimensionality reduction  
**C)** Classification algorithm  
**D)** Clustering algorithm

**Answer**: B  
**Explanation**: PCA is unsupervised - finds principal components without using labels.

---

## Q9. 🟡 The curse of dimensionality refers to:
**A)** High computational cost  
**B)** Data sparsity in high dimensions  
**C)** Too many features  
**D)** Overfitting risk

**Answer**: B  
**Explanation**: In high dimensions, data becomes sparse - distances become meaningless, nearest neighbors are far.

---

## Q10. 🔴 Information Gain in decision trees is calculated using:
**A)** Variance reduction  
**B)** Entropy reduction  
**C)** Mean squared error  
**D)** Correlation

**Answer**: B  
**Explanation**: Information Gain = Entropy(parent) - Weighted_Avg(Entropy(children)).

---

## Q11. 🟢 Which is an ensemble method?
**A)** Logistic Regression  
**B)** Decision Tree  
**C)** Random Forest  
**D)** K-Means

**Answer**: C  
**Explanation**: Random Forest combines multiple decision trees (ensemble).

---

## Q12. 🟡 Regularization in ML helps to:
**A)** Increase model complexity  
**B)** Prevent overfitting  
**C)** Speed up training  
**D)** Improve interpretability

**Answer**: B  
**Explanation**: Regularization adds penalty term to prevent overfitting by constraining weights.

---

## Q13. 🟢 Scikit-learn's `fit_transform()` method:
**A)** Only fits the model  
**B)** Only transforms data  
**C)** Fits on data and transforms same data  
**D)** Fits on train, transforms test

**Answer**: C  
**Explanation**: `fit_transform()` = `fit()` + `transform()` on same data. Use only on training data!

---

## Q14. 🟡 Which statement about Naive Bayes is FALSE?
**A)** Assumes feature independence  
**B)** Works well with high-dimensional data  
**C)** Can handle feature correlations well  
**D)** Based on Bayes' theorem

**Answer**: C  
**Explanation**: "Naive" assumption: features are independent. Performs poorly when features are highly correlated.

---

## Q15. 🔴 In SVM, support vectors are:
**A)** All training points  
**B)** Points on the margin boundaries  
**C)** Misclassified points  
**D)** Centroids of classes

**Answer**: B  
**Explanation**: Support vectors are points lying on or inside the margin boundaries that determine the hyperplane.

---

## Q16. 🟢 GridSearchCV performs:
**A)** Feature selection  
**B)** Hyperparameter tuning  
**C)** Model training  
**D)** Data preprocessing

**Answer**: B  
**Explanation**: GridSearchCV exhaustively searches hyperparameter space using CV.

---

## Q17. 🟡 Silhouette score is used for:
**A)** Classification evaluation  
**B)** Regression evaluation  
**C)** Clustering evaluation  
**D)** Dimensionality reduction

**Answer**: C  
**Explanation**: Silhouette score measures how well data points fit their assigned clusters.

---

## Q18. 🟢 Gradient Descent updates weights by:
**A)** Moving in direction of gradient  
**B)** Moving opposite to gradient  
**C)** Random direction  
**D)** Moving to global minimum directly

**Answer**: B  
**Explanation**: Weights updated as: w = w - learning_rate * gradient (opposite direction to minimize loss).

---

## Q19. 🟡 Bagging reduces:
**A)** Bias only  
**B)** Variance only  
**C)** Both bias and variance  
**D)** Neither

**Answer**: B  
**Explanation**: Bagging (Bootstrap Aggregating) reduces variance by averaging multiple models.

---

## Q20. 🔴 The Bayes optimal classifier achieves:
**A)** 100% accuracy  
**B)** Lowest possible error rate given the data distribution  
**C)** Zero training error  
**D)** Same performance as nearest neighbor

**Answer**: B  
**Explanation**: Bayes optimal classifier has lowest possible error (Bayes error rate) but still has irreducible error.

---

## Q21. 🟢 Confusion matrix is used for:
**A)** Regression  
**B)** Classification  
**C)** Clustering  
**D)** Dimensionality reduction

**Answer**: B  
**Explanation**: Confusion matrix shows TP, TN, FP, FN for classification tasks.

---

## Q22. 🟡 Early stopping in neural networks:
**A)** Stops when training error stops decreasing  
**B)** Stops when validation error stops decreasing  
**C)** Stops after fixed epochs  
**D)** Stops when test error is minimized

**Answer**: B  
**Explanation**: Early stopping monitors validation error to prevent overfitting.

---

## Q23. 🟢 Feature scaling is most important for:
**A)** Decision Trees  
**B)** K-Nearest Neighbors  
**C)** Random Forest  
**D)** Naive Bayes

**Answer**: B  
**Explanation**: k-NN uses distances, so feature scaling is critical. Tree-based methods don't need scaling.

---

## Q24. 🟡 SMOTE is used for:
**A)** Feature extraction  
**B)** Handling imbalanced data  
**C)** Dimensionality reduction  
**D)** Outlier detection

**Answer**: B  
**Explanation**: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for minority class.

---

## Q25. 🔴 The Rademacher complexity measures:
**A)** Model's ability to memorize  
**B)** Model's capacity to fit random labels  
**C)** Number of parameters  
**D)** Training time

**Answer**: B  
**Explanation**: Rademacher complexity measures model's ability to fit random noise - indicates capacity/complexity.

---

## Q26. 🟢 One-hot encoding is used for:
**A)** Numerical features  
**B)** Categorical features  
**C)** Text data  
**D)** Time series

**Answer**: B  
**Explanation**: One-hot encoding converts categorical variables into binary vectors.

---

## Q27. 🟡 Stratified sampling ensures:
**A)** Equal sample size from each class  
**B)** Proportional class distribution in splits  
**C)** Random sampling  
**D)** Maximum diversity

**Answer**: B  
**Explanation**: Stratified sampling maintains class proportions in train/test splits.

---

## Q28. 🟢 Precision is defined as:
**A)** TP / (TP + FP)  
**B)** TP / (TP + FN)  
**C)** TN / (TN + FP)  
**D)** (TP + TN) / Total

**Answer**: A  
**Explanation**: Precision = TP / (TP + FP) = "Of predicted positives, how many were correct?"

---

## Q29. 🟡 Dropout in neural networks:
**A)** Removes neurons permanently  
**B)** Randomly deactivates neurons during training  
**C)** Removes features  
**D)** Stops training early

**Answer**: B  
**Explanation**: Dropout randomly sets neuron activations to zero during training to prevent overfitting.

---

## Q30. 🔴 PAC (Probably Approximately Correct) learning guarantees:
**A)** Exact solution  
**B)** Solution that is probably close to optimal with high probability  
**C)**100% accuracy  
**D)** Convergence in polynomial time

**Answer**: B  
**Explanation**: PAC learning provides probabilistic bounds: with probability ≥ (1-δ), error ≤ ε.

---

# SECTION B: Algorithms & Implementation (40 Questions, 90 Minutes)

## Q31. 🟢 In linear regression, if X.shape = (100, 5), what is the shape of coefficients?
**A)** (100,)  
**B)** (5,)  
**C)** (100, 5)  
**D)** (5, 100)

**Answer**: B  
**Explanation**: One coefficient per feature: β has shape (5,) plus one intercept.

---

## Q32. 🟡 Which scikit-learn function splits data into train/test?
**A)** `split_data()`  
**B)** `train_test_split()`  
**C)** `split()`  
**D)** `divide_data()`

**Answer**: B  
**Explanation**: `from sklearn.model_selection import train_test_split`

---

## Q33. 🟢 What does `model.predict_proba()` return for binary classification?
**A)** Class labels (0 or 1)  
**B)** Probabilities for each class  
**C)** Confidence scores  
**D)** Decision function values

**Answer**: B  
**Explanation**: `predict_proba()` returns array of shape (n_samples, 2) with [P(class=0), P(class=1)] for each sample.

---

## Q34. 🟡 In k-means, what happens if you set random_state?
**A)** Clustering becomes deterministic  
**B)** Results are always random  
**C)** Number of clusters changes  
**D)** Algorithm runs faster

**Answer**: A  
**Explanation**: `random_state` controls initialization, making results reproducible.

---

## Q35. 🔴 What's the complexity of k-NN prediction with n samples and d features?
**A)** O(1)  
**B)** O(log n)  
**C)** O(nd)  
**D)** O(n²d)

**Answer**: C  
**Explanation**: Must compute distance to all n training points, each with d features: O(nd).

---

## Q36. 🟢 Which parameter controls tree depth in Decision Tree?
**A)** `n_estimators`  
**B)** `max_depth`  
**C)** `max_features`  
**D)** `min_samples_split`

**Answer**: B  
**Explanation**: `max_depth` limits maximum depth of tree.

---

## Q37. 🟡 In Random Forest, `n_estimators` controls:
**A)** Number of features  
**B)** Number of trees  
**C)** Tree depth  
**D)** Number of samples

**Answer**: B  
**Explanation**: `n_estimators` = number of trees in the forest.

---

## Q38. 🟢 StandardScaler transforms data to have:
**A)** Mean=0, Std=0  
**B)** Mean=1, Std=1  
**C)** Mean=0, Std=1  
**D)** Range [0, 1]

**Answer**: C  
**Explanation**: StandardScaler: (x - mean) / std → mean=0, std=1.

---

## Q39. 🟡 MinMaxScaler scales data to:
**A)** Mean=0, Std=1  
**B)** Range [0, 1]  
**C)** Range [-1, 1]  
**D)** No change

**Answer**: B  
**Explanation**: MinMaxScaler: (x - min) / (max - min) → range [0, 1].

---

## Q40. 🔴 For PCA with 100 features, if you keep 95% variance, how many components minimum?
**A)** Always 95  
**B)** Depends on data eigenvalues  
**C)** Always 100  
**D)** Always less than 50

**Answer**: B  
**Explanation**: Number of components depends on eigenvalue distribution - vary by dataset.

---

## Q41. 🟢 Logistic Regression outputs:
**A)** Continuous values  
**B)** Probabilities  
**C)** Binary labels only  
**D)** Distances

**Answer**: B  
**Explanation**: Logistic regression outputs P(y=1|x) via sigmoid function.

---

## Q42. 🟡 In SVM, the C parameter controls:
**A)** Margin width  
**B)** Regularization strength (inverse)  
**C)** Number of support vectors  
**D)** Kernel type

**Answer**: B  
**Explanation**: Larger C → less regularization → harder margin → more overfitting risk.

---

## Q43. 🟢 To use polynomial features with degree=2, you use:
**A)** `PolynomialFeatures(degree=2)`  
**B)** `degree=2` in model  
**C)** `model.polynomial(2)`  
**D)** Manual feature creation only

**Answer**: A  
**Explanation**: `from sklearn.preprocessing import PolynomialFeatures`

---

## Q44. 🟡 Ridge regression adds which penalty?
**A)** L0 (number of features)  
**B)** L1 (absolute values)  
**C)** L2 (squared values)  
**D)** No penalty

**Answer**: C  
**Explanation**: Ridge adds L2 penalty: λ Σ β²

---

## Q45. 🔴 Lasso regression can perform:
**A)** Only regularization  
**B)** Only feature selection  
**C)** Both regularization and feature selection  
**D)** Neither

**Answer**: C  
**Explanation**: Lasso (L1) can shrink coefficients to exactly zero → feature selection + regularization.

---

## Q46. 🟢 `cross_val_score` with cv=5 trains the model:
**A)** 1 time  
**B)** 5 times  
**C)** 10 times  
**D)** Depends on data

**Answer**: B  
**Explanation**: 5-fold CV trains model 5 times (once per fold).

---

## Q47. 🟡 For time series, which cross-validation is appropriate?
**A)** K-Fold  
**B)** Stratified K-Fold  
**C)** TimeSeriesSplit  
**D)** Leave-One-Out

**Answer**: C  
**Explanation**: TimeSeriesSplit respects temporal order (no data leakage).

---

## Q48. 🟢 DBSCAN requires specifying:
**A)** Number of clusters  
**B)** eps and min_samples  
**C)** Only eps  
**D)** Only min_samples

**Answer**: B  
**Explanation**: DBSCAN needs eps (neighborhood radius) and min_samples (density threshold).

---

## Q49. 🟡 Elbow method is used to find:
**A)** Optimal features  
**B)** Optimal k in k-means  
**C)** Optimal learning rate  
**D)** Optimal test size

**Answer**: B  
**Explanation**: Elbow method plots inertia vs k, look for "elbow" point.

---

## Q50. 🔴 In Gradient Boosting, learning_rate controls:
**A)** Number of trees  
**B)** Contribution of each tree  
**C)** Tree depth  
**D)** Sample size

**Answer**: B  
**Explanation**: learning_rate (shrinkage) scales contribution of each tree to prevent overfitting.

---

## Q51. 🟢 `model.score()` returns what for classification?
**A)** Accuracy  
**B)** F1-score  
**C)** Precision  
**D)** Recall

**Answer**: A  
**Explanation**: Default scoring for classifiers is accuracy.

---

## Q52. 🟡 `model.score()` returns what for regression?
**A)** MSE  
**B)** RMSE  
**C)** R² score  
**D)** MAE

**Answer**: C  
**Explanation**: Default scoring for regressors is R² (coefficient of determination).

---

## Q53. 🟢 To handle missing values with mean imputation:
**A)** `SimpleImputer(strategy='mean')`  
**B)** `fillna(mean())`  
**C)** Manual mean calculation  
**D)** `interpolate()`

**Answer**: A  
**Explanation**: `from sklearn.impute import SimpleImputer`

---

## Q54. 🟡 LabelEncoder should be used on:
**A)** Features only  
**B)** Target only  
**C)** Both features and target  
**D)** Neither

**Answer**: B  
**Explanation**: LabelEncoder for target. Use OneHotEncoder for categorical features (to avoid ordinal assumption).

---

## Q55. 🔴 Pipeline in scikit-learn ensures:
**A)** Faster training  
**B)** No data leakage in preprocessing  
**C)** Better accuracy  
**D)** Automatic hyperparameter tuning

**Answer**: B  
**Explanation**: Pipeline applies transformations correctly (fit on train, transform on test) preventing data leakage.

---

## Q56. 🟢 For multi-class classification with >2 classes, use:
**A)** BinaryClassifier only  
**B)** LogisticRegression  
**C)** LinearRegression  
**D)** Cannot use any classifier

**Answer**: B  
**Explanation**: LogisticRegression supports multi-class (one-vs-rest or softmax).

---

## Q57. 🟡 ROC curve plots:
**A)** Precision vs Recall  
**B)** TPR vs FPR  
**C)** Accuracy vs Threshold  
**D)** Loss vs Epochs

**Answer**: B  
**Explanation**: ROC plots True Positive Rate vs False Positive Rate at various thresholds.

---

## Q58. 🟢 AUC-ROC of 0.5 means:
**A)** Perfect classifier  
**B)** Random classifier  
**C)** Worst classifier  
**D)** Depends on data

**Answer**: B  
**Explanation**: AUC=0.5 → performance equal to random guessing.

---

## Q59. 🟡 For text classification, which vectorizer converts text to features?
**A)** LabelEncoder  
**B)** StandardScaler  
**C)** CountVectorizer  
**D)** PCA

**Answer**: C  
**Explanation**: CountVectorizer/TfidfVectorizer convert text to numerical features.

---

## Q60. 🔴 In neural networks, which activation for binary classification output?
**A)** ReLU  
**B)** Sigmoid  
**C)** Tanh  
**D)** Softmax

**Answer**: B  
**Explanation**: Sigmoid outputs probability P(y=1) ∈ [0,1] for binary classification.

---

## Q61. 🟢 Adam optimizer is:
**A)** Learning rate scheduler  
**B)** Adaptive learning rate method  
**C)** Loss function  
**D)** Activation function

**Answer**: B  
**Explanation**: Adam combines momentum and RMSprop with adaptive learning rates.

---

## Q62. 🟡 Batch normalization normalizes:
**A)** Input data  
**B)** Activations within layers  
**C)** Weights  
**D)** Gradients

**Answer**: B  
**Explanation**: Batch norm normalizes layer activations across mini-batch.

---

## Q63. 🟢 In CNN, pooling layer:
**A)** Increases spatial dimensions  
**B)** Reduces spatial dimensions  
**C)** Adds parameters  
**D)** Acts as activation

**Answer**: B  
**Explanation**: Pooling (max/average) downsamples feature maps.

---

## Q64. 🟡 Transfer learning involves:
**A)** Training from scratch  
**B)** Using pre-trained model on new task  
**C)** Transferring data between tasks  
**D)** Converting model to different framework

**Answer**: B  
**Explanation**: Transfer learning uses pre-trained weights as initialization for new task.

---

## Q65. 🔴 In LSTM, the forget gate decides:
**A)** What to output  
**B)** What to remember from previous state  
**C)** What to add to current state  
**D)** Learning rate

**Answer**: B  
**Explanation**: Forget gate (sigmoid) determines what to keep from previous cell state.

---

## Q66. 🟢 For sequential data, which model is most suitable?
**A)** CNN  
**B)** RNN/LSTM  
**C)** Random Forest  
**D)** k-NN

**Answer**: B  
**Explanation**: RNN/LSTM designed for sequential data with temporal dependencies.

---

## Q67. 🟡 Word embeddings (Word2Vec, GloVe) represent:
**A)** Words as high-dimensional sparse vectors  
**B)** Words as dense low-dimensional vectors  
**C)** Sentences  
**D)** Documents

**Answer**: B  
**Explanation**: Embeddings map words to dense vectors (e.g., 100-300 dimensions) capturing semantic meaning.

---

## Q68. 🟢 In reinforcement learning, the agent learns by:
**A)** Labeled examples  
**B)** Trial and error with rewards  
**C)** Clustering  
**D)** Dimensionality reduction

**Answer**: B  
**Explanation**: RL agent learns optimal policy through interaction and reward signals.

---

## Q69. 🟡 Q-learning is:
**A)** Supervised learning algorithm  
**B)** Unsupervised learning algorithm  
**C)** Model-free RL algorithm  
**D)** Deep learning architecture

**Answer**: C  
**Explanation**: Q-learning learns action-value function Q(s,a) without model of environment.

---

## Q70. 🔴 Actor-Critic methods in RL combine:
**A)** Two Q-functions  
**B)** Policy (actor) and value function (critic)  
**C)** Two policies  
**D)** Exploration and exploitation

**Answer**: B  
**Explanation**: Actor learns policy π(a|s), Critic learns value function V(s), training each other.

---

# SECTION C: Applied ML & Advanced Topics (30 Questions, 45 Minutes)

## Q71. 🟢 For highly imbalanced dataset (99% class 0, 1% class 1), accuracy is:
**A)** Best metric  
**B)** Misleading metric  
**C)** Unbiased metric  
**D)** Not applicable

**Answer**: B  
**Explanation**: Model predicting all class 0 gets 99% accuracy! Use F1/precision/recall instead.

---

## Q72. 🟡 Data leakage occurs when:
**A)** Test data influences training  
**B)** Missing values exist  
**C)** Features are correlated  
**D)** Model underfits

**Answer**: A  
**Explanation**: Data leakage: information from test set "leaks" into training (e.g., scaling before split).

---

## Q73. 🟢 To prevent overfitting, you should:
**A)** Increase model complexity  
**B)** Collect more data  
**C)** Reduce regularization  
**D)** Train longer

**Answer**: B  
**Explanation**: More data helps prevent overfitting. Also: regularization, cross-validation, simpler models.

---

## Q74. 🟡 Feature importance in Random Forest is calculated by:
**A)** Coefficient values  
**B)** Mean decrease in impurity  
**C)** Correlation with target  
**D)** P-values

**Answer**: B  
**Explanation**: RF feature importance based on average impurity decrease across all trees.

---

## Q75. 🔴 Explainable AI (XAI) techniques include:
**A)** SHAP values  
**B)** LIME  
**C)** Attention mechanisms  
**D)** All of the above

**Answer**: D  
**Explanation**: All are XAI methods to interpret black-box models.

---

## Q76. 🟢 AutoML automates:
**A)** Data collection  
**B)** Model selection and hyperparameter tuning  
**C)** Problem definition  
**D)** Code writing

**Answer**: B  
**Explanation**: AutoML automates ML pipeline: preprocessing, model selection, hyperparameter optimization.

---

## Q77. 🟡 For online learning, which algorithm is suitable?
**A)** Decision Tree  
**B)** SGD-based models  
**C)** k-NN  
**D)** Random Forest

**Answer**: B  
**Explanation**: Stochastic Gradient Descent allows incremental learning from streaming data.

---

## Q78. 🟢 Model deployment involves:
**A)** Training the model  
**B)** Making model available for predictions in production  
**C)** Hyperparameter tuning  
**D)** Feature engineering

**Answer**: B  
**Explanation**: Deployment makes trained model accessible (API, web service, edge device).

---

## Q79. 🟡 A/B testing in ML is used for:
**A)** Model training  
**B)** Comparing model performance in production  
**C)** Feature selection  
**D)** Data preprocessing

**Answer**: B  
**Explanation**: A/B testing compares two models in production with live traffic.

---

## Q80. 🔴 Concept drift refers to:
**A)** Model training drift  
**B)** Data distribution changing over time  
**C)** Feature drift during preprocessing  
**D)** Parameter drift

**Answer**: B  
**Explanation**: Concept drift: P(y|X) changes over time, requiring model retraining.

---

## Q81. 🟢 Model versioning is important for:
**A)** Tracking model changes  
**B)** Faster training  
**C)** Better accuracy  
**D)** Data storage

**Answer**: A  
**Explanation**: Versioning tracks models, data, and code for reproducibility and rollback.

---

## Q82. 🟡 Federated learning:
**A)** Trains model centrally  
**B)** Trains model distributedly without sharing data  
**C)** Ensemble method  
**D)** Transfer learning variant

**Answer**: B  
**Explanation**: Federated learning trains models across devices without centralizing data (privacy-preserving).

---

## Q83. 🟢 For recommendation systems, collaborative filtering uses:
**A)** Item features only  
**B)** User features only  
**C)** User-item interactions  
**D)** Content analysis

**Answer**: C  
**Explanation**: Collaborative filtering: "Users who liked X also liked Y" based on interaction patterns.

---

## Q84. 🟡 Matrix factorization in recommender systems decomposes:
**A)** Feature matrix  
**B)** User-item rating matrix  
**C)** Covariance matrix  
**D)** Confusion matrix

**Answer**: B  
**Explanation**: Factorize R ≈ U × V^T where U=user factors, V=item factors.

---

## Q85. 🔴 GANs consist of:
**A)** Generator only  
**B)** Discriminator only  
**C)** Generator and Discriminator  
**D)** Encoder and Decoder

**Answer**: C  
**Explanation**: GAN = Generator (creates fake data) + Discriminator (distinguishes real/fake).

---

## Q86. 🟢 In GAN training, mode collapse occurs when:
**A)** Discriminator fails  
**B)** Generator produces limited variety  
**C)** Training converges  
**D)** Both converge

**Answer**: B  
**Explanation**: Mode collapse: Generator produces only few types of outputs, missing data diversity.

---

## Q87. 🟡 Variational Autoencoders (VAEs) learn:
**A)** Deterministic encoding  
**B)** Probabilistic latent representation  
**C)** Supervised classification  
**D)** Clustering

**Answer**: B  
**Explanation**: VAE learns distribution over latent space (mean and variance), not deterministic encoding.

---

## Q88. 🟢 Attention mechanism in Transformers allows:
**A)** Sequential processing  
**B)** Focusing on relevant input parts  
**C)** Faster training  
**D)** Smaller models

**Answer**: B  
**Explanation**: Attention computes weighted sum of inputs, focusing on relevant positions.

---

## Q89. 🟡 BERT is pre-trained using:
**A)** Supervised learning  
**B)** Masked Language Modeling  
**C)** Reinforcement learning  
**D)** GANs

**Answer**: B  
**Explanation**: BERT pre-trained with MLM (predict masked tokens) + NSP (next sentence prediction).

---

## Q90. 🔴 GPT models use:
**A)** Encoder-only architecture  
**B)** Decoder-only architecture  
**C)** Encoder-decoder architecture  
**D)** No Transformer architecture

**Answer**: B  
**Explanation**: GPT uses decoder-only Transformer with causal (left-to-right) attention.

---

## Q91. 🟢 Zero-shot learning means:
**A)** Training without data  
**B)** Predicting classes not seen during training  
**C)** No hyperparameters  
**D)** No preprocessing

**Answer**: B  
**Explanation**: Zero-shot: model generalizes to unseen classes (e.g., via text descriptions).

---

## Q92. 🟡 Few-shot learning uses:
**A)** Large labeled datasets  
**B)** Very few examples per class  
**C)** No labeled data  
**D)** Only test data

**Answer**: B  
**Explanation**: Few-shot: learn from few examples (1-10) per class, often via meta-learning.

---

## Q93. 🟢 Meta-learning (learning to learn) aims to:
**A)** Learn faster on new tasks  
**B)** Learn bigger models  
**C)** Learn without data  
**D)** Learn interpretable models

**Answer**: A  
**Explanation**: Meta-learning learns initialization that adapts quickly to new tasks with few examples.

---

## Q94. 🟡 Neural Architecture Search (NAS) automates:
**A)** Data collection  
**B)** Model architecture design  
**C)** Hyperparameter tuning  
**D)** Feature engineering

**Answer**: B  
**Explanation**: NAS automatically finds optimal network architecture (layers, connections).

---

## Q95. 🔴 Which is NOT a challenge in production ML?
**A)** Model monitoring  
**B)** Model training (already done)  
**C)** Data drift  
**D)** Scalability

**Answer**: B  
**Explanation**: Training is offline phase. Production challenges: monitoring, drift, latency, scaling.

---

## Q96. 🟢 MLOps focuses on:
**A)** Model training only  
**B)** Operationalizing ML in production  
**C)** Data collection  
**D)** Algorithm design

**Answer**: B  
**Explanation**: MLOps combines ML + DevOps for deploying, monitoring, maintaining ML systems.

---

## Q97. 🟡 Model serving latency is:
**A)** Training time  
**B)** Time to return prediction  
**C)** Data loading time  
**D)** Model size

**Answer**: B  
**Explanation**: Latency = time from request to prediction response (critical for real-time systems).

---

## Q98. 🟢 Fairness in ML refers to:
**A)** Equal accuracy across groups  
**B)** No bias in predictions  
**C)** Equitable treatment of different groups  
**D)** All of the above

**Answer**: D  
**Explanation**: ML fairness encompasses accuracy parity, bias mitigation, equitable outcomes.

---

## Q99. 🟡 Differential privacy ensures:
**A)** Data encryption  
**B)** Individual privacy in datasets  
**C)** Model security  
**D)** Fair predictions

**Answer**: B  
**Explanation**: Differential privacy adds controlled noise to protect individual data points.

---

## Q100. 🔴 The AI alignment problem concerns:
**A)** Model architecture  
**B)** Aligning AI behavior with human values  
**C)** Data alignment  
**D)** Training speed

**Answer**: B  
**Explanation**: Alignment: ensuring AI systems behave according to human intentions and values.

---

# Answer Key Summary

## Section A (Questions 1-30):
1-B, 2-D, 3-B, 4-C, 5-B, 6-B, 7-B, 8-B, 9-B, 10-B,  
11-C, 12-B, 13-C, 14-C, 15-B, 16-B, 17-C, 18-B, 19-B, 20-B,  
21-B, 22-B, 23-B, 24-B, 25-B, 26-B, 27-B, 28-A, 29-B, 30-B

## Section B (Questions 31-70):
31-B, 32-B, 33-B, 34-A, 35-C, 36-B, 37-B, 38-C, 39-B, 40-B,  
41-B, 42-B, 43-A, 44-C, 45-C, 46-B, 47-C, 48-B, 49-B, 50-B,  
51-A, 52-C, 53-A, 54-B, 55-B, 56-B, 57-B, 58-B, 59-C, 60-B,  
61-B, 62-B, 63-B, 64-B, 65-B, 66-B, 67-B, 68-B, 69-C, 70-B

## Section C (Questions 71-100):
71-B, 72-A, 73-B, 74-B, 75-D, 76-B, 77-B, 78-B, 79-B, 80-B,  
81-A, 82-B, 83-C, 84-B, 85-C, 86-B, 87-B, 88-B, 89-B, 90-B,  
91-B, 92-B, 93-A, 94-B, 95-B, 96-B, 97-B, 98-D, 99-B, 100-B

---

# Scoring Guide

- **90-100**: Excellent! Ready for exam
- **75-89**: Good! Review sections with errors
- **60-74**: Satisfactory. More practice needed
- **40-59**: Pass, but needs significant review
- **<40**: Not ready. Study fundamentals again

---

# Tips for Exam Day

1. **Read questions carefully** - many have subtle differences in options
2. **Watch for "NOT" and "FALSE" questions** - easy to miss
3. **Eliminate obviously wrong answers** first
4. **Time management**: ~1 minute per question, flag difficult ones
5. **Trust your first instinct** for uncertain questions
6. **Review flagged questions** if time remains

---

**Good Luck! 🍀**

**End of Mock Exam**
