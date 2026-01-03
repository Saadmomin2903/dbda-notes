# Session 17 – Convolutional Neural Networks (CNNs)

## 📚 Table of Contents
1. [CNN Fundamentals](#cnn-fundamentals)
2. [Convolution Operations](#convolution-operations)
3. [Pooling Layers](#pooling-layers)
4. [CNN Architectures](#cnn-architectures)
5. [Transfer Learning](#transfer-learning)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# CNN Fundamentals

## 📘 Why CNNs for Images?

**Problems with fully-connected networks**:
1. Too many parameters (e.g., 224×224×3 image = 150K inputs)
2. No spatial structure preserved
3. Not translation invariant

**CNN advantages**:
- **Parameter sharing**: Same filter across image
- **Local connectivity**: Each neuron sees local region
- **Translation invariance**: Learns features regardless of position

## 🧮 Key Concepts

### Receptive Field
Region of input that affects a neuron's activation.

### Feature Maps
Output of applying filters to input.

### Depth
Number of filters (channels) in layer.

---

# Convolution Operations

## 📘 Concept Overview

**Convolution**: Slide filter over input, compute dot product.

## 🧮 Mathematical Foundation

### 2D Convolution

For input I and filter K:

```
(I * K)[i, j] = ΣₘΣₙ I[i+m, j+n] × K[m, n]
```

### Output Size

```
Output_height = (Input_height - Filter_height + 2×Padding) / Stride + 1
Output_width = (Input_width - Filter_width + 2×Padding) / Stride + 1
```

### Parameters

**Stride**: Step size for sliding filter
- Stride = 1: Dense coverage
- Stride = 2: Reduce spatial dimensions by half

**Padding**: Add zeros around border
- Valid: No padding (output smaller)
- Same: Pad to keep output size = input size

### Number of Parameters

For filter size k×k, input channels C_in, output channels C_out:

```
Parameters = k × k × C_in × C_out + C_out
```

(+ C_out for bias terms)

## 🧪 Python Implementation

```python
import numpy as np

def conv2d(input_img, kernel, stride=1, padding=0):
    """2D convolution (simplified)."""
    # Add padding
    if padding > 0:
        input_img = np.pad(input_img, padding, mode='constant')
    
    h_in, w_in = input_img.shape
    k_h, k_w = kernel.shape
    
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract region
            region = input_img[h_start:h_start+k_h, w_start:w_start+k_w]
            
            # Convolve
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example
img = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

# Edge detection filter
kernel = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

result = conv2d(img, kernel, stride=1, padding=0)
print("Convolution result:")
print(result)
```

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Conv layer: 32 filters, 3×3 kernel
conv_layer = Conv2D(filters=32, kernel_size=3, strides=1, 
                    padding='same', activation='relu')

# Input: batch_size × height × width × channels
input_shape = (None, 28, 28, 1)  # MNIST-like
```

## 📊 Common Filters

### Edge Detection (Horizontal)
```
[-1, -1, -1]
[ 0,  0,  0]
[ 1,  1,  1]
```

### Edge Detection (Vertical)
```
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]
```

### Sharpen
```
[ 0, -1,  0]
[-1,  5, -1]
[ 0, -1,  0]
```

### Blur (Average)
```
[1/9, 1/9, 1/9]
[1/9, 1/9, 1/9]
[1/9, 1/9, 1/9]
```

**CNNs learn optimal filters** automatically!

---

# Pooling Layers

## 📘 Purpose

**Downsampling** to:
1. Reduce spatial dimensions
2. Reduce parameters
3. Translation invariance
4. Control overfitting

## 🧮 Types

### Max Pooling

Take maximum value in region:

```
Max Pooling 2×2:
[1, 2]  →  3
[3, 0]
```

**Most common** (preserves strongest features).

### Average Pooling

Take average value:

```
Avg Pooling 2×2:
[1, 2]  →  1.5
[3, 0]
```

### Global Average Pooling

Average over entire feature map:

```
H×W×C  →  1×1×C
```

Used before final dense layer (reduces parameters).

## 🧪 Python Implementation

```python
def max_pool2d(input_img, pool_size=2, stride=2):
    """Max pooling."""
    h_in, w_in = input_img.shape
    
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            
            region = input_img[h_start:h_start+pool_size, 
                              w_start:w_start+pool_size]
            output[i, j] = np.max(region)
    
    return output

# Keras
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

max_pool = MaxPooling2D(pool_size=2, strides=2)
```

---

# CNN Architectures

## 📊 Classic Architectures

### LeNet-5 (1998)

**First successful CNN** (MNIST digit recognition).

```
Input (32×32×1)
  ↓ Conv 6 filters 5×5
  ↓ AvgPool 2×2
  ↓ Conv 16 filters 5×5
  ↓ AvgPool 2×2
  ↓ Flatten
  ↓ FC 120
  ↓ FC 84
  ↓ FC 10 (output)
```

**Parameters**: ~60K

### AlexNet (2012)

**ImageNet breakthrough** (won with 15.3% error vs 26.2%).

```
Input (227×227×3)
  ↓ Conv 96 filters 11×11, stride 4
  ↓ MaxPool 3×3, stride 2
  ↓ Conv 256 filters 5×5
  ↓ MaxPool 3×3, stride 2
  ↓ Conv 384 filters 3×3
  ↓ Conv 384 filters 3×3
  ↓ Conv 256 filters 3×3
  ↓ MaxPool 3×3, stride 2
  ↓ FC 4096 + Dropout
  ↓ FC 4096 + Dropout
  ↓ FC 1000 (output)
```

**Innovations**:
- ReLU activation
- Dropout regularization
- Data augmentation
- GPU training

**Parameters**: ~60M

### VGG-16 (2014)

**Very deep** (16 layers), **simple architecture**.

**Pattern**: Stack of 3×3 convolutions + max pooling

```
Input (224×224×3)
  ↓ Conv 64 (3×3) × 2
  ↓ MaxPool
  ↓ Conv 128 (3×3) × 2
  ↓ MaxPool
  ↓ Conv 256 (3×3) × 3
  ↓ MaxPool
  ↓ Conv 512 (3×3) × 3
  ↓ MaxPool
  ↓ Conv 512 (3×3) × 3
  ↓ MaxPool
  ↓ FC 4096 + Dropout
  ↓ FC 4096 + Dropout
  ↓ FC 1000
```

**Parameters**: ~138M (very large!)

**Key insight**: Multiple small 3×3 filters better than one large filter.

### ResNet (2015)

**Residual connections** allow training very deep networks (50, 101, 152 layers).

**Residual Block**:
```
x → [Conv → BN → ReLU → Conv → BN] → Add → ReLU
 └─────────────────────────────────────┘
         (Skip connection)
```

**Formula**:
```
Output = F(x) + x
```

Where F(x) = learned residual.

**Why it works**:
- Easier to learn residual F(x) than direct mapping
- Solves vanishing gradient problem (gradient flows through skip connections)

```python
# Keras ResNet block
from tensorflow.keras.layers import Add

def residual_block(x, filters):
    """Residual block."""
    shortcut = x
    
    # Main path
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x
```

**Performance**: ResNet-152 achieved 3.57% top-5 error on ImageNet (better than human ~5%)!

## 📊 Architecture Comparison

| Model | Year | Depth | Parameters | Top-5 Error |
|-------|------|-------|-----------|-------------|
| **LeNet-5** | 1998 | 5 | 60K | - |
| **AlexNet** | 2012 | 8 | 60M | 15.3% |
| **VGG-16** | 2014 | 16 | 138M | 7.3% |
| **ResNet-50** | 2015 | 50 | 25M | 3.8% |
| **ResNet-152** | 2015 | 152 | 60M | 3.57% |

## 🧪 Build Simple CNN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Simple CNN for MNIST
model = Sequential([
    # Conv block 1
    Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2),
    
    # Conv block 2
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2),
    
    # Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

print(model.summary())
```

---

# Transfer Learning

## 📘 Concept Overview

**Transfer Learning**: Use pre-trained model on new task.

**Why?**
- Trained on large dataset (ImageNet: 1.2M images)
- Learned useful features (edges, textures, patterns)
- Faster convergence
- Better performance with small datasets

## 🧮 Strategies

### 1. Feature Extraction

**Freeze** pre-trained layers, train only new classifier:

```python
from tensorflow.keras.applications import VGG16

# Load pre-trained model (without top classifier)
base_model = VGG16(weights='imagenet', include_top=False, 
                   input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add new classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # New task: 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2. Fine-Tuning

**Unfreeze** some top layers, train with low learning rate:

```python
# Unfreeze last few layers
base_model.trainable = True

# Freeze early layers, unfreeze last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile with low learning rate
model.compile(optimizer=Adam(lr=1e-5), 
             loss='categorical_crossentropy',
             metrics=['accuracy'])
```

## 📊 When to Use Each

| Dataset Size | Similarity to ImageNet | Strategy |
|--------------|------------------------|----------|
| **Small** | Similar | Feature extraction |
| **Small** | Different | Feature extraction + fine-tune top |
| **Large** | Similar | Fine-tune many layers |
| **Large** | Different | Train from scratch or heavy fine-tuning |

---

# 🔥 MCQs

### Q1. Convolution operation:
**Options:**
- A) Averages all pixels
- B) Slides filter and computes dot product ✓
- C) Pools regions
- D) Flattens input

**Explanation**: Convolution = filter sliding over input, element-wise multiply and sum.

---

### Q2. Padding='same' ensures:
**Options:**
- A) No padding
- B) Output size = input size ✓
- C) Stride = 1
- D) Filter size = 3

**Explanation**: 'Same' padding preserves spatial dimensions.

---

### Q3. Max pooling 2×2 reduces dimensions by:
**Options:**
- A) 25%
- B) 50% ✓
- C) 75%
- D) None

**Explanation**: 2×2 pooling halves height and width (75% reduction in total pixels).

---

### Q4. ResNet uses:
**Options:**
- A) Large filters
- B) Skip connections ✓
- C) No pooling
- D) Linear activations

**Explanation**: Residual connections allow very deep networks.

---

### Q5. For 5×5 filter on 32×32 input, stride=1, no padding, output size:
**Options:**
- A) 32×32
- B) 28×28 ✓
- C) 30×30
- D) 27×27

**Explanation**: (32 - 5)/1 + 1 = 28.

---

### Q6. Number of parameters in Conv2D(64 filters, 3×3, input 32 channels):
**Options:**
- A) 64
- B) 576
- C) 18,496 ✓
- D) 32

**Explanation**: 3×3×32×64 + 64 = 18,496 (weights + bias).

---

### Q7. Global Average Pooling converts H×W×C to:
**Options:**
- A) H×W×1
- B) 1×1×C ✓
- C) C×1×1
- D) HWC

**Explanation**: Average each channel → 1×1×C vector.

---

### Q8. VGG-16 primarily uses filter size:
**Options:**
- A) 1×1
- B) 3×3 ✓
- C) 5×5
- D) 11×11

**Explanation**: VGG stacks many 3×3 convolutions.

---

### Q9. Transfer learning works because:
**Options:**
- A) Same architecture
- B) Pre-trained features generalize ✓
- C) Faster training only
- D) Reduces parameters

**Explanation**: Lower layers learn general features (edges, textures).

---

### Q10. Receptive field is:
**Options:**
- A) Filter size
- B) Input region affecting a neuron ✓
- C) Output size
- D) Pooling size

**Explanation**: Region of input that influences an activation.

---

### Q11. LeNet-5 was designed for:
**Options:**
- A) ImageNet
- B) MNIST digits ✓
- C) CIFAR-10
- D) Face recognition

**Explanation**: LeNet pioneered CNNs for handwritten digit recognition.

---

### Q12. AlexNet used activation:
**Options:**
- A) Sigmoid
- B) Tanh
- C) ReLU ✓
- D) Linear

**Explanation**: First major CNN to use ReLU (faster training).

---

### Q13. Stride=2 in convolution:
**Options:**
- A) Doubles output size
- B) Halves output size (approximately) ✓
- C) No effect
- D) Adds padding

**Explanation**: Larger stride reduces output dimensions.

---

### Q14. Fine-tuning uses:
**Options:**
- A) High learning rate
- B) Low learning rate ✓
- C) No regularization
- D) Random initialization

**Explanation**: Low LR prevents destroying pre-trained weights.

---

### Q15. Residual block formula:
**Options:**
- A) F(x) × x
- B) F(x) + x ✓
- C) F(x) - x
- D) F(x)

**Explanation**: Skip connection adds input to learned residual.

---

# ⚠️ Common Mistakes

1. **Wrong input shape**: CNNs expect (batch, height, width, channels) in TensorFlow

2. **Not using padding**: Output shrinks rapidly without 'same' padding

3. **Too many pooling layers**: Loses too much spatial information

4. **Large learning rate for fine-tuning**: Destroys pre-trained weights

5. **Not freezing base model initially**: Fine-tune after feature extraction training

6. **Forgetting data augmentation**: Essential for small image datasets

7. **Using average pooling everywhere**: Max pooling usually better

8. **Not normalizing inputs**: Images should be in [0,1] or standardized

9. **1×1 convolutions misunderstood**: Reduce channels, add non-linearity

10. **Ignoring batch normalization**: Speeds up training significantly

---

# ⭐ One-Line Exam Facts

1. **Convolution**: Slide filter, compute dot product (parameter sharing)

2. **Output size**: (input - filter + 2×padding)/stride + 1

3. **Padding 'same'**: Output size = input size

4. **Padding 'valid'**: No padding (output smaller)

5. **Max pooling**: Takes max value in region (downsampling)

6. **Receptive field**: Input region affecting a neuron's activation

7. **LeNet-5**: First successful CNN (MNIST, 1998)

8. **AlexNet**: ImageNet breakthrough (ReLU, dropout, 2012)

9. **VGG**: Deep network with 3×3 filters repeated

10. **ResNet**: Skip connections F(x) + x (enables very deep networks)

11. **Transfer learning**: Use pre-trained model for new task

12. **Feature extraction**: Freeze base, train classifier

13. **Fine-tuning**: Unfreeze layers with low learning rate

14. **Global Average Pooling**: H×W×C → 1×1×C (replaces FC layers)

15. **3×3 filter**: Most common in modern CNNs

---

**End of Session 17**

**Progress: 17/30 sessions (57%)!** CNNs complete. Continuing with remaining 13 sessions.
