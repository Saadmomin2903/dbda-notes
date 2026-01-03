# Session 22 – Advanced Computer Vision

## 📚 Table of Contents
1. [Object Detection](#object-detection)
2. [YOLO Architecture](#yolo-architecture)
3. [Image Segmentation](#image-segmentation)
4. [U-Net Architecture](#u-net-architecture)
5. [Object Tracking](#object-tracking)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Object Detection

## 📘 Task Overview

**Goal**: Locate objects in image + classify them.

**Output**: Bounding boxes (x, y, w, h) + class labels + confidence scores.

## 📊 R-CNN Family

### R-CNN (Region-based CNN)

**Two-stage detector**:
```
1. Region proposals (~2000) using selective search
2. CNN feature extraction for each region
3. SVM classification
4. Bounding box regression
```

**Slow**: CNN runs on each region separately.

### Fast R-CNN

**Improvements**:
- Single CNN forward pass for entire image
- **ROI Pooling**: Extract features for each region from feature map

**Speed**: 10x faster than R-CNN.

### Faster R-CNN

**Key innovation**: **Region Proposal Network (RPN)**
- Replace selective search with learned proposals
- End-to-end trainable

**RPN**:
```
Sliding window on feature map
  → Predict objectness + box offsets
  → Generate proposals
  → ROI pooling → Classification
```

**Anchors**: Predefined boxes at multiple scales/ratios.

---

# YOLO Architecture

## 📘 You Only Look Once

**Single-stage detector**: Direct prediction in one pass.

## 🧮 YOLO Approach

**Grid-based**:
```
Divide image into S×S grid
Each grid cell predicts:
  - B bounding boxes (x, y, w, h, confidence)
  - C class probabilities
```

**Output shape**: S × S × (B×5 + C)

**Example**: 7×7 grid, 2 boxes, 20 classes → 7×7×30 tensor

## 🧮 Loss Function

```
L = λ_coord Σ (box coordinate errors)
  + λ_obj Σ (objectness errors for cells with objects)
  + λ_noobj Σ (objectness errors for cells without objects)  
  + Σ (classification errors)
```

**Weights**: λ_coord = 5, λ_noobj = 0.5 (balance losses)

## 📊 YOLO Versions

### YOLOv1 (2016)
- 45 FPS (real-time)
- 7×7 grid

### YOLOv3 (2018)
- Multi-scale predictions (3 scales)
- Feature Pyramid Network
- Darknet-53 backbone

### YOLOv5 (2020)
- PyTorch implementation
- Auto-learning bounding box anchors
- CSPDarknet backbone

### YOLOv8 (2023)
- Anchor-free
- Improved accuracy
- Efficient architecture

## 🧪 YOLO Implementation

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano version

# Inference
results = model('image.jpg')

# Get detections
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
        conf = box.conf[0]  # Confidence
        cls = box.cls[0]  # Class
        
        print(f"Class: {cls}, Confidence: {conf:.2f}")
        print(f"Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

---

# Image Segmentation

## 📊 Types

### Semantic Segmentation
- **Pixel-wise classification**
- All pixels of same class labeled identically
- No instance differentiation

### Instance Segmentation
- **Individual object instances**
- Each object gets unique label
- **Mask R-CNN**: Adds segmentation branch to Faster R-CNN

### Panoptic Segmentation
- Combines semantic + instance
- All pixels labeled with class + instance ID

---

# U-Net Architecture

## 📘 Encoder-Decoder with Skip Connections

**Designed for**: Medical image segmentation (works on small datasets).

## 🧮 Architecture

```
Encoder (Downsampling):
  Conv → Conv → MaxPool
  Conv → Conv → MaxPool
  ...
  Bottleneck

Decoder (Upsampling):
  UpConv → Concat(skip) → Conv → Conv
  UpConv → Concat(skip) → Conv → Conv
  ...
  Output
```

**Skip connections**: Concatenate encoder features to decoder at corresponding levels.

**Why skip connections?**
- Preserve spatial information
- Better localization
- Gradient flow

## 🧪 U-Net Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import *

def unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)
    
    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = UpSampling2D(2)(c3)
    u1 = Concatenate()([u1, c2])  # Skip connection
    c4 = Conv2D(128, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    
    u2 = UpSampling2D(2)(c4)
    u2 = Concatenate()([u2, c1])  # Skip connection
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    
    return tf.keras.Model(inputs, outputs)
```

---

# Object Tracking

## 📊 Techniques

### Single Object Tracking

**SORT** (Simple Online Realtime Tracking):
- Kalman filter for motion prediction
- Hungarian algorithm for data association

**Deep SORT**:
- Adds appearance features (CNN)
- Better re-identification

### Multi-Object Tracking

**Challenges**:
- Occlusions
- Similar appearances
- Variable number of objects

**Approach**:
```
1. Detection in each frame
2. Associate detections across frames
3. Track trajectories
```

---

# 🔥 MCQs

### Q1. YOLO is:
**Options:**
- A) Two-stage detector
- B) Single-stage detector ✓
- C) Three-stage detector
- D) Not a detector

**Explanation**: YOLO predicts boxes + classes in single forward pass.

---

### Q2. U-Net skip connections:
**Options:**
- A) Add encoder to decoder
- B) Concatenate encoder to decoder ✓
- C) No connections
- D) Multiply features

**Explanation**: U-Net concatenates encoder features to decoder.

---

### Q3. Faster R-CNN uses:
**Options:**
- A) Selective search
- B) Region Proposal Network (RPN) ✓
- C) No proposals
- D) Manual regions

**Explanation**: RPN replaces selective search, end-to-end trainable.

---

### Q4. Semantic segmentation:
**Options:**
- A) Instance-level labels
- B) Pixel-wise class labels ✓
- C) Bounding boxes
- D) Object detection

**Explanation**: Classifies each pixel, no instance differentiation.

---

### Q5. Mask R-CNN adds to Faster R-CNN:
**Options:**
- A) More anchors
- B) Segmentation branch ✓
- C) Larger backbone
- D) More classes

**Explanation**: Adds mask prediction branch for instance segmentation.

---

# ⚠️ Common Mistakes

1. **Confusing semantic vs instance segmentation**: Semantic doesn't distinguish instances
2. **Using wrong YOLO version**: Different versions have different APIs
3. **Not tuning anchors**: YOLO performance sensitive to anchor boxes
4. **Ignoring NMS (Non-Max Suppression)**: Gets rid of duplicate detections
5. **Wrong input size**: YOLO expects specific sizes (640×640, etc.)
6. **Not using skip connections in U-Net**: Critical for localization
7. **Forgetting data augmentation**: Essential for object detection
8. **Class imbalance**: Most boxes are background, need balancing

---

# ⭐ One-Line Exam Facts

1. **R-CNN**: Two-stage (proposals → CNN → classify)
2. **Faster R-CNN**: RPN for learned proposals
3. **YOLO**: Single-stage, grid-based, real-time
4. **YOLO loss**: Coordinate + objectness + classification
5. **U-Net**: Encoder-decoder + skip connections
6. **Skip connections**: Concatenate encoder to decoder
7. **Semantic segmentation**: Pixel-wise classification
8. **Instance segmentation**: Per-object masks
9. **Mask R-CNN**: Faster R-CNN + segmentation branch
10. **NMS**: Remove duplicate detections
11. **ROI pooling**: Extract features for region proposals
12. **Anchors**: Predefined boxes at multiple scales
13. **IOU**: Intersection over Union (bounding box metric)
14. **FPN**: Feature Pyramid Network (multi-scale)
15. **YOLOv8**: Latest, anchor-free, improved accuracy

---

**End of Session 22**
