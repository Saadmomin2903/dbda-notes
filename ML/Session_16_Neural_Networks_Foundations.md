# Session 16 – Neural Networks & Deep Learning Foundations

## 📚 Table of Contents
1. [Perceptron](#perceptron)
2. [Activation Functions](#activation-functions)
3. [Feedforward Neural Networks](#feedforward-neural-networks)
4. [Backpropagation](#backpropagation)
5. [Gradient Descent Optimization](#gradient-descent-optimization)
6. [Regularization Techniques](#regularization-techniques)
7. [MCQs](#mcqs)
8. [Common Mistakes](#common-mistakes)
9. [One-Line Exam Facts](#one-line-exam-facts)

---

# Perceptron

## 📘 Concept Overview

**Perceptron**: Simplest neural network unit (single neuron).

## 🧮 Mathematical Foundation

### Model

```
ŷ = f(w^T x + b)
```

Where:
- x = input vector
- w = weights
- b = bias
- f = activation function (step function for perceptron)

### Binary Step Function

```
f(z) = {1 if z ≥ 0
       {0 if z < 0
```

### Decision Boundary

```
w^T x + b = 0
```

Linear boundary (hyperplane).

### Learning Rule

For misclassified point (x, y):

```
w ← w + α(y - ŷ)x
b ← b + α(y - ŷ)
```

Where α = learning rate

## 📊 Limitations

1. **Linear only**: Cannot learn XOR
2. **Binary classification**: Only 2 classes
3. **No probabilistic output**: Hard 0/1 predictions

**Solution**: Multi-layer networks with non-linear activations!

## 🧪 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
    
    def step_function(self, z):
        """Binary step activation."""
        return np.where(z >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train perceptron."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Compute output
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.step_function(z)
                
                # Update if misclassified
                if y[i] != y_pred:
                    self.weights += self.lr * (y[i] - y_pred) * X[i]
                    self.bias += self.lr * (y[i] - y_pred)
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)

# Test on linearly separable data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

perceptron = Perceptron(learning_rate=0.1, n_epochs=10)
perceptron.fit(X, y)

accuracy = np.mean(perceptron.predict(X) == y)
print(f"Perceptron Accuracy: {accuracy:.3f}")
```

---

# Activation Functions

## 📘 Purpose

**Non-linear activations** allow neural networks to learn complex patterns.

## 🧮 Common Activation Functions

### 1. Sigmoid (Logistic)

```
σ(z) = 1 / (1 + e^(-z))
```

**Range**: (0, 1)

**Derivative**:
```
σ'(z) = σ(z)(1 - σ(z))
```

**Use**: Output layer for binary classification

**Disadvantages**:
- Vanishing gradients (saturates at 0 and 1)
- Not zero-centered

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

### 2. Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Range**: (-1, 1)

**Derivative**:
```
tanh'(z) = 1 - tanh²(z)
```

**Advantages over sigmoid**:
- Zero-centered
- Stronger gradients

**Still suffers**: Vanishing gradients

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

### 3. ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

**Range**: [0, ∞)

**Derivative**:
```
ReLU'(z) = {1 if z > 0
           {0 if z ≤ 0
```

**Advantages**:
- ✓ No vanishing gradient (for z > 0)
- ✓ Computationally efficient
- ✓ Sparse activations

**Disadvantage**:
- Dying ReLU (neurons stuck at 0)

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
```

### 4. Leaky ReLU

```
LeakyReLU(z) = {z     if z > 0
               {αz    if z ≤ 0
```

Where α = small constant (e.g., 0.01)

**Fixes dying ReLU**: Non-zero gradient for z < 0

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

### 5. Softmax

```
softmax(z_i) = e^(z_i) / Σⱼ e^(z_j)
```

**Use**: Multi-class output layer

**Properties**:
- Σᵢ softmax(z_i) = 1 (probabilities)
- Range: (0, 1)

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / exp_z.sum(axis=0)
```

## 📊 Comparison

| Function | Range | Gradient | Use Case |
|----------|-------|----------|----------|
| **Sigmoid** | (0, 1) | Vanishing | Binary output |
| **Tanh** | (-1, 1) | Vanishing | Hidden layers (old) |
| **ReLU** | [0, ∞) | ✓ Better | ✓ Hidden layers (default) |
| **Leaky ReLU** | (-∞, ∞) | ✓ Better | Fixes dying ReLU |
| **Softmax** | (0, 1) | - | Multi-class output |

**Modern default**: ReLU for hidden layers

---

# Feedforward Neural Networks

## 📘 Architecture

**Layers**:
1. **Input Layer**: Receives features
2. **Hidden Layers**: Transform representations
3. **Output Layer**: Produces predictions

```
Input → Hidden₁ → Hidden₂ → ... → Output
```

## 🧮 Forward Pass

For layer l:

```
z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])
```

Where:
- a^[l] = activations of layer l
- W^[l] = weight matrix
- b^[l] = bias vector
- g^[l] = activation function

**Input**: a^[0] = x

## 🧪 Python Implementation

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of layer dimensions [input, hidden1, hidden2, ..., output]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1])
            b = np.zeros((layer_sizes[i], 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation."""
        self.activations = [X]
        self.z_values = []
        
        a = X
        
        for i in range(self.num_layers - 1):
            z = self.weights[i] @ a + self.biases[i]
            self.z_values.append(z)
            
            # ReLU for hidden layers, sigmoid for output
            if i < self.num_layers - 2:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            self.activations.append(a)
        
        return a
    
    def predict(self, X):
        """Make predictions."""
        output = self.forward(X.T)
        return (output > 0.5).astype(int).T

# Test
nn = NeuralNetwork([2, 4, 1])  # 2 inputs, 4 hidden, 1 output

X_test = np.array([[0.5, 0.3]])
output = nn.forward(X_test.T)
print(f"Network output: {output[0][0]:.4f}")
```

---

# Backpropagation

## 📘 Concept Overview

**Backpropagation**: Algorithm to compute gradients for weight updates.

**Chain rule** applied backwards through the network.

## 🧮 Mathematical Derivation

### Loss Function (Binary Cross-Entropy)

```
L = -(1/m) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

### Output Layer Gradient

```
δ^[L] = ∂L/∂z^[L] = a^[L] - y
```

For sigmoid output and cross-entropy loss.

### Hidden Layer Gradient

```
δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'^[l](z^[l])
```

Where ⊙ = element-wise product

### Weight and Bias Gradients

```
∂L/∂W^[l] = (1/m) δ^[l] (a^[l-1])^T
∂L/∂b^[l] = (1/m) Σᵢ δ^[l]
```

### Update Rule

```
W^[l] ← W^[l] - α ∂L/∂W^[l]
b^[l] ← b^[l] - α ∂L/∂b^[l]
```

## 🧪 Backpropagation Implementation

```python
def backward(self, X, y, learning_rate=0.01):
    """Backpropagation."""
    m = X.shape[1]  # Number of samples
    
    # Output layer gradient
    delta = self.activations[-1] - y.reshape(1, -1)
    
    # Iterate backwards
    for i in range(self.num_layers - 2, -1, -1):
        # Compute gradients
        dW = (1/m) * delta @ self.activations[i].T
        db = (1/m) * np.sum(delta, axis=1, keepdims=True)
        
        # Update weights and biases
        self.weights[i] -= learning_rate * dW
        self.biases[i] -= learning_rate * db
        
        # Propagate delta to previous layer
        if i > 0:
            delta = self.weights[i].T @ delta
            delta = delta * self.relu_derivative(self.z_values[i-1])

def train(self, X, y, epochs=1000, learning_rate=0.01):
    """Train neural network."""
    for epoch in range(epochs):
        # Forward pass
        output = self.forward(X.T)
        
        # Backward pass
        self.backward(X, y, learning_rate)
        
        # Compute loss
        if epoch % 100 == 0:
            loss = -np.mean(y * np.log(output + 1e-8) + 
                          (1 - y) * np.log(1 - output + 1e-8))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Add methods to NeuralNetwork class above
NeuralNetwork.backward = backward
NeuralNetwork.train = train

# Train on XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

nn_xor = NeuralNetwork([2, 4, 1])
nn_xor.train(X_xor, y_xor, epochs=5000, learning_rate=0.5)

# Test
predictions = nn_xor.predict(X_xor)
print(f"\nXOR Predictions:")
for i in range(len(X_xor)):
    print(f"Input: {X_xor[i]}, Predicted: {predictions[i][0]}, True: {y_xor[i]}")
```

---

# Gradient Descent Optimization

## 📊 Variants

### 1. Batch Gradient Descent

Use all samples:
```
W ← W - α (1/m) Σᵢ ∇L_i
```

**Pros**: Stable, exact gradient
**Cons**: Slow for large datasets

### 2. Stochastic Gradient Descent (SGD)

Use one sample:
```
W ← W - α ∇L_i
```

**Pros**: Fast updates
**Cons**: Noisy, high variance

### 3. Mini-Batch Gradient Descent

Use batch of samples:
```
W ← W - α (1/|B|) Σ_{i∈B} ∇L_i
```

**Best trade-off** (typical batch size: 32, 64, 128, 256)

## 📊 Advanced Optimizers

### Momentum

```
v_t = βv_{t-1} + (1-β)∇L
W ← W - αv_t
```

**Accelerates** in consistent direction

### RMSprop

```
s_t = βs_{t-1} + (1-β)(∇L)²
W ← W - α∇L / √(s_t + ε)
```

**Adaptive** learning rates per parameter

### Adam (Adaptive Moment Estimation)

**Combines momentum + RMSprop**:

```
m_t = β₁m_{t-1} + (1-β₁)∇L        (momentum)
v_t = β₂v_{t-1} + (1-β₂)(∇L)²     (RMSprop)

m̂_t = m_t / (1 - β₁^t)            (bias correction)
v̂_t = v_t / (1 - β₂^t)

W ← W - α m̂_t / (√v̂_t + ε)
```

**Default**: β₁=0.9, β₂=0.999, ε=1e-8

**Most popular optimizer**!

```python
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Using Adam
optimizer = Adam(learning_rate=0.001)
```

---

# Regularization Techniques

## 📊 Methods to Prevent Overfitting

### 1. L2 Regularization (Weight Decay)

Add penalty to loss:
```
L_total = L + (λ/2m) Σᵢ ‖W^[l]‖²
```

**Gradient**:
```
∂L/∂W += λW
```

### 2. L1 Regularization

```
L_total = L + (λ/m) Σᵢ |W^[l]|
```

**Produces sparse weights**

### 3. Dropout

Randomly **drop neurons** during training (p = dropout rate).

**Training**: Each neuron kept with probability (1-p)
**Testing**: Use all neurons, scale by (1-p)

```python
def dropout(a, dropout_rate=0.5, training=True):
    """Apply dropout."""
    if training:
        mask = np.random.rand(*a.shape) > dropout_rate
        return a * mask / (1 - dropout_rate)
    else:
        return a
```

**Prevents co-adaptation** of neurons.

### 4. Early Stopping

Stop training when validation loss stops improving.

### 5. Batch Normalization

Normalize layer inputs:
```
BN(z) = γ((z - μ_B) / √(σ²_B + ε)) + β
```

**Benefits**:
- Faster training
- Higher learning rates possible
- Regularization effect

```python
from tensorflow.keras.layers import BatchNormalization
```

### 6. Data Augmentation

Generate more training data through transformations.

**Images**: Rotation, flipping, cropping, color jittering

---

# 🔥 MCQs

### Q1. Perceptron can learn:
**Options:**
- A) XOR function
- B) Linear separable patterns only ✓
- C) Any function
- D) Non-linear patterns

**Explanation**: Single perceptron limited to linear decision boundaries.

---

### Q2. ReLU activation range is:
**Options:**
- A) (0, 1)
- B) (-1, 1)
- C) [0, ∞) ✓
- D) (-∞, ∞)

**Explanation**: ReLU(z) = max(0, z) ∈ [0, ∞).

---

### Q3. Vanishing gradient problem occurs with:
**Options:**
- A) ReLU
- B) Sigmoid ✓
- C) Leaky ReLU
- D) None

**Explanation**: Sigmoid saturates (gradient → 0) for large |z|.

---

### Q4. Softmax used for:
**Options:**
- A) Binary classification
- B) Multi-class classification ✓
- C) Regression
- D) Hidden layers

**Explanation**: Softmax outputs probability distribution over classes.

---

### Q5. Backpropagation computes:
**Options:**
- A) Activations
- B) Gradients ✓
- C) Predictions
- D) Loss

**Explanation**: Backprop uses chain rule to compute ∂L/∂W.

---

### Q6. Adam optimizer combines:
**Options:**
- A) SGD + RMSprop
- B) Momentum + RMSprop ✓
- C) SGD + Momentum
- D) Adagrad + RMSprop

**Explanation**: Adam = momentum (1st moment) + RMSprop (2nd moment).

---

### Q7. Dropout rate 0.5 means:
**Options:**
- A) Keep 50% of neurons ✓
- B) Drop 50% accuracy
- C) 50% learning rate
- D) 50 neurons

**Explanation**: Each neuron kept with probability 0.5 during training.

---

### Q8. He initialization scales weights by:
**Options:**
- A) √(1/n)
- B) √(2/n) ✓
- C) 1/n
- D) 2/n

**Explanation**: √(2/n) works well with ReLU activation.

---

### Q9. Cross-entropy loss for binary classification:
**Options:**
- A) (y - ŷ)²
- B) -[y log ŷ + (1-y) log(1-ŷ)] ✓
- C) |y - ŷ|
- D) max(0, 1 - yŷ)

**Explanation**: Binary cross-entropy measures probability error.

---

### Q10. Batch normalization normalizes:
**Options:**
- A) Weights
- B) Layer inputs ✓
- C) Gradients
- D) Loss

**Explanation**: BN normalizes activations (layer inputs).

---

### Q11. Dying ReLU problem means:
**Options:**
- A) Slow training
- B) Neurons always output 0 ✓
- C) Overfitting
- D) Vanishing gradients

**Explanation**: Negative weights cause z < 0 → ReLU=0 → no gradient.

---

### Q12. Momentum parameter β typically:
**Options:**
- A) 0.1
- B) 0.5
- C) 0.9 ✓
- D) 1.0

**Explanation**: β=0.9 common (90% previous velocity + 10% current gradient).

---

### Q13. L2 regularization equivalent to:
**Options:**
- A) Dropout
- B) Weight decay ✓
- C) Early stopping
- D) Batch normalization

**Explanation**: L2 penalty = weight decay in gradient update.

---

### Q14. For hidden layers, best default activation is:
**Options:**
- A) Sigmoid
- B) Tanh
- C) ReLU ✓
- D) Linear

**Explanation**: ReLU most commonly used (no vanishing gradient for z>0).

---

### Q15. Mini-batch size affects:
**Options:**
- A) Model architecture
- B) Training speed and stability ✓
- C) Number of layers
- D) Activation functions

**Explanation**: Larger batches = stable but slow; smaller = fast but noisy.

---

# ⚠️ Common Mistakes

1. **Not using non-linear activations**: Network becomes linear (no hidden layer benefit)

2. **Wrong learning rate**: Too large diverges, too small slow convergence

3. **Not normalizing inputs**: Features on different scales slow training

4. **Random weight initialization**: Use He or Xavier initialization

5. **Forgetting bias terms**: Can limit model expressiveness

6. **Using sigmoid for hidden layers**: Use ReLU instead

7. **Not shuffling training data**: Can cause poor convergence

8. **Overfitting without regularization**: Use dropout, L2, or early stopping

9. **Zero gradients with ReLU**: Use Leaky ReLU to avoid dying ReLU

10. **Not monitoring validation loss**: May overfit without knowing

---

# ⭐ One-Line Exam Facts

1. **Perceptron**: Single neuron, linear decision boundary only

2. **XOR problem**: Requires hidden layer (not linearly separable)

3. **Sigmoid**: σ(z) = 1/(1+e^(-z)), range (0,1), vanishing gradient

4. **ReLU**: max(0,z), range [0,∞), default for hidden layers

5. **Softmax**: Multi-class output, Σ probabilities = 1

6. **Forward pass**: z = Wa + b, a = g(z) for each layer

7. **Backpropagation**: Computes gradients via chain rule backwards

8. **Gradient descent**: W ← W - α∇L

9. **Adam optimizer**: Combines momentum + RMSprop (most popular)

10. **Dropout**: Randomly drop neurons during training (prevents overfitting)

11. **Batch normalization**: Normalize layer inputs, faster training

12. **L2 regularization**: Penalty λ‖W‖², prevents large weights

13. **Vanishing gradient**: Problem with sigmoid/tanh (use ReLU)

14. **Dying ReLU**: Neurons stuck at 0 (use Leaky ReLU)

15. **Mini-batch**: Balance between batch GD and SGD

---

**End of Session 16**

**Progress: 16/30 sessions completed (53%)!** Deep Learning foundations established. Ready to continue with remaining sessions.
