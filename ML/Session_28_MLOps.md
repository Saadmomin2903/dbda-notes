# Session 28 – MLOps & Production ML

## 📚 Table of Contents
1. [Model Deployment](#model-deployment)
2. [Monitoring & Drift Detection](#monitoring--drift-detection)
3. [ML Pipelines](#ml-pipelines)
4. [Experiment Tracking](#experiment-tracking)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Model Deployment

## 📊 Deployment Strategies

### REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model
model = torch.load('model.pth')
model.eval()

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Preprocess
    input_tensor = torch.FloatTensor([request.features])
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()
        confidence = torch.sigmoid(output).item()
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Containerization

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pth .
COPY app.py .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run**:
```bash
docker build -t ml-model:v1 .
docker run -p 8000:8000 ml-model:v1
```

## 📊 Model Serving Frameworks

### TensorFlow Serving
- Optimized for TensorFlow models
- gRPC and REST APIs
- Model versioning
- Batching and caching

### TorchServe
- PyTorch model serving
- Multi-model support
- Metrics and logging
- Custom handlers

### ONNX Runtime
- Cross-framework (TF, PyTorch, etc.)
- High performance
- Hardware optimization

---

# Monitoring & Drift Detection

## 📊 Key Metrics

### Performance Metrics
- **Latency**: Response time (p50, p95, p99)
- **Throughput**: Requests per second
- **Error rate**: Failed predictions
- **Resource usage**: CPU, memory, GPU

### ML Metrics
- **Accuracy/F1**: Online evaluation
- **Prediction distribution**: Monitor outputs
- **Feature statistics**: Mean, variance

## 🧮 Data Drift

**Definition**: Input data distribution changes over time.

**Detection**: Compare current vs reference distribution

**Methods**:

### Kolmogorov-Smirnov Test
```python
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(reference_data, current_data)

if p_value < 0.05:
    print("Data drift detected!")
```

### Population Stability Index (PSI)
```
PSI = Σ (P_current - P_reference) × log(P_current / P_reference)
```

**Thresholds**:
- PSI < 0.1: No drift
- 0.1 < PSI < 0.2: Moderate drift
- PSI > 0.2: Significant drift

## 🧮 Model Drift

**Definition**: Model performance degrades over time.

**Causes**:
- Data drift
- Concept drift (y|x changes)
- Feature engineering issues

**Solution**: Retrain periodically or when drift detected.

## 📊 A/B Testing

```
Traffic split:
  50% → Model A (baseline)
  50% → Model B (new)

Compare metrics:
  - Accuracy
  - Business metrics (revenue, engagement)
  - Latency

Decision:
  If B significantly better → Gradual rollout
  Otherwise → Keep A
```

---

# ML Pipelines

## 📊 Pipeline Components

```
Data Ingestion → Data Validation → Preprocessing →
  Feature Engineering → Model Training → Model Validation →
  Model Deployment → Monitoring
```

## 🧪 Kubeflow

**ML workflows on Kubernetes**.

**Components**:
- **Pipelines**: Define ML workflows
- **Notebooks**: Interactive development
- **Training operators**: Distributed training
- **Serving**: Model deployment

```python
# Kubeflow Pipeline example
import kfp
from kfp import dsl

@dsl.component
def preprocess_data(input_path: str) -> str:
    # Data preprocessing
    return output_path

@dsl.component
def train_model(data_path: str) -> str:
    # Model training
    return model_path

@dsl.pipeline(name='ML Pipeline')
def ml_pipeline(input_path: str):
    preprocess_task = preprocess_data(input_path)
    train_task = train_model(preprocess_task.output)

kfp.compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')
```

---

# Experiment Tracking

## 📊 MLflow

**Platform for ML lifecycle**.

### Components

**Tracking**: Log parameters, metrics, artifacts
```python
import mlflow

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("batch_size", 64)

# Log metrics
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("loss", 0.15)

# Log model
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
```

**Model Registry**: Version and manage models
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/<run-id>/model"
mlflow.register_model(model_uri, "ProductionModel")

# Transition to production
client.transition_model_version_stage(
    name="ProductionModel",
    version=1,
    stage="Production"
)
```

## 📊 Weights & Biases

```python
import wandb

wandb.init(project="my-project")

wandb.config.learning_rate = 0.01
wandb.config.batch_size = 64

# Log metrics
wandb.log({"loss": loss, "accuracy": accuracy})

# Save model
wandb.save("model.h5")
```

## 📊 DVC (Data Version Control)

**Track datasets and models with Git**.

```bash
# Initialize DVC
dvc init

# Track dataset
dvc add data/train.csv

# Push to remote storage
dvc remote add -d storage s3://my-bucket/dvc-store
dvc push

# In other environment
dvc pull  # Download data
```

---

# 🔥 MCQs

### Q1. Data drift is:
**Options:**
- A) Model performance change
- B) Input distribution change ✓
- C) Bug in code
- D) Hardware failure

**Explanation**: Change in p(X) over time.

---

### Q2. A/B testing compares:
**Options:**
- A) Data versions
- B) Model versions ✓
- C) Hardware
- D) Datasets

**Explanation**: Split traffic between models to compare performance.

---

### Q3. MLflow provides:
**Options:**
- A) Only training
- B) Tracking, registry, deployment ✓
- C) Only deployment
- D) Data storage

**Explanation**: Full ML lifecycle management.

---

### Q4. Docker benefits:
**Options:**
- A) Faster training
- B) Reproducibility ✓
- C) Better accuracy
- D) More data

**Explanation**: Consistent environment across deployments.

---

### Q5. PSI > 0.2 indicates:
**Options:**
- A) No drift
- B) Significant drift ✓
- C) Good model
- D) Fast inference

**Explanation**: Population Stability Index > 0.2 means substantial drift.

---

# ⚠️ Common Mistakes

1. **Not monitoring in production**: Essential to catch issues early
2. **Ignoring data drift**: Leads to silent model degradation
3. **No versioning**: Can't reproduce or rollback
4. **Hardcoding paths**: Use environment variables
5. **Not logging predictions**: Needed for debugging
6. **Skipping A/B tests**: Direct deployment risky
7. **No automated retraining**: Models stale over time
8. **Insufficient error handling**: Production needs robust code

---

# ⭐ One-Line Exam Facts

1. **REST API**: FastAPI or Flask for model serving
2. **Docker**: Containerization for reproducible deployment
3. **Data drift**: Input distribution p(X) changes
4. **Model drift**: Performance degrades over time
5. **PSI**: Population Stability Index measures drift
6. **A/B testing**: Split traffic to compare models
7. **MLflow**: Tracking, registry, deployment platform
8. **Kubeflow**: ML pipelines on Kubernetes
9. **DVC**: Version control for data and models
10. **TensorFlow Serving**: Optimized TF model serving
11. **Monitoring metrics**: Latency, throughput, accuracy
12. **K-S test**: Statistical test for distribution change
13. **Model registry**: Centralized model versioning
14. **Gradual rollout**: Slowly increase traffic to new model
15. **Retraining**: Periodic or drift-triggered

---

**End of Session 28**
