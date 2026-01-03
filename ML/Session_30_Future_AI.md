# Session 30 – Future of AI & Emerging Trends

## 📚 Table of Contents
1. [Multimodal Learning](#multimodal-learning)
2. [Efficient AI](#efficient-ai)
3. [Retrieval-Augmented Generation](#retrieval-augmented-generation)
4. [AI Agents](#ai-agents)
5. [Industry Applications](#industry-applications)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Multimodal Learning

## 📊 CLIP (Contrastive Language-Image Pre-training)

**Goal**: Learn joint text-image embedding space.

**Training objective**:
```
Maximize: similarity(image_i, text_i)
Minimize: similarity(image_i, text_j) for i≠j
```

**Contrastive loss**:
```
L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
```

**Architecture**:
- Image encoder (Vision Transformer or ResNet)
- Text encoder (Transformer)
- Shared embedding space

**Applications**:
- Zero-shot image classification
- Text-to-image retrieval
- Image-to-text retrieval

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Zero-shot classification
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # Image-text similarity
probs = logits_per_image.softmax(dim=1)  # Probabilities
```

## 📊 Flamingo

**Few-shot vision-language model**.

**Key features**:
- Interleaved image-text input
- Few-shot learning from examples
- Strong zero-shot performance

---

# Efficient AI

## 📊 Model Compression

### 1. Pruning

**Remove unimportant weights**:
```
Set weights with |w| < threshold to 0
```

**Types**:
- **Unstructured**: Individual weights
- **Structured**: Entire neurons/channels

**Typical**: 50-90% weights can be  removed with minimal accuracy loss.

### 2. Quantization

**Reduce precision**: FP32 → INT8

**Benefits**:
- 4x smaller model
- 2-4x faster inference
- Lower memory bandwidth

**Post-training quantization**:
```python
import torch

model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**Quantization-aware training**: Simulate low precision during training.

### 3. Knowledge Distillation

**Train small model to mimic large model**.

**Loss**:
```
L = L_task + α KL(P_student || P_teacher)
```

**Temperature scaling**: Soften probabilities
```
P_i = exp(z_i/T) / Σ_j exp(z_j/T)
```

Higher T → softer distribution (more information).

```python
# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Soft targets
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_prob = F.log_softmax(student_logits / T, dim=1)
    
    # KL divergence
    kl_div = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)
    
    # Hard targets
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * kl_div + (1 - alpha) * ce_loss
```

## 📊 Neural Architecture Search (NAS)

**Automated model design**.

**Approaches**:
- **Reinforcement learning**: Controller generates architectures
- **Evolutionary algorithms**: Mutation and crossover
- **Gradient-based**: DARTS (differentiable architectures)

---

# Retrieval-Augmented Generation

## 🧮 RAG Pipeline

```
Query → Retrieve relevant docs → LLM(query + docs) → Response
```

**Benefits**:
- Grounds generation in facts
- Reduces hallucination
- Can update knowledge without retraining
- Cites sources

**Architecture**:
```
1. Encode query: q = Encoder(query)
2. Retrieve: docs = TopK(similarity(q, document_embeddings))
3. Generate: response = LLM(query + docs)
```

**Example**:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
response = qa_chain.run("What is the capital of France?")
```

---

# AI Agents

## 📊 Agent Framework

**Components**:
- **LLM**: Reasoning engine
- **Tools**: Functions agent can call
- **Memory**: Conversation history
- **Planning**: Break down tasks

## 🧮 AutoGPT

**Autonomous task completion**.

**Loop**:
```
1. Receive goal
2. Plan steps
3. Execute step (use tools)
4. Evaluate progress
5. Repeat until goal achieved
```

## 📊 LangChain

**Framework for LLM applications**.

**Components**:
- **Chains**: Combine LLM calls
- **Agents**: Autonomous decision-making
- **Memory**: Context management
- **Tools**: External functions

```python
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# Run
agent.run("What is the GDP of France in 2023?")
```

## 📊 Function Calling

LLMs can call external functions/APIs.

**Example**:
```python
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=functions
)
```

## 📊 Constitutional AI

**Self-alignment via principles**.

**Process**:
```
1. Generate initial response
2. Critique based on constitution (principles)
3. Revise response
4. Repeat
```

**Example principles**:
- "Choose helpful and harmless response"
- "Avoid stereotypes"
- "Be truthful"

---

# Industry Applications

## 📊 Healthcare

**Drug Discovery**:
- AlphaFold: Protein structure prediction
- Molecule generation: Design new drugs
- Clinical trials: Patient selection

**Diagnosis**:
- Medical image analysis
- Disease prediction
- Treatment recommendation

## 📊 Climate & Environment

- Weather forecasting: GraphCast (Google)
- Climate modeling: Predict long-term changes
- Optimization: Energy grid, traffic flow

## 📊 Scientific Discovery

- **Protein folding**: AlphaFold (DeepMind)
- **Materials science**: Discover new materials
- **Mathematics**: Theorem proving

## 📊 Finance

- Fraud detection: Anomaly detection
- Algorithmic trading: Pattern recognition
- Risk assessment: Credit scoring

---

# 🔥 MCQs

### Q1. CLIP learns:
**Options:**
- A) Text only
- B) Joint text-image embeddings ✓
- C) Images only
- D) Audio

**Explanation**: Contrastive learning for vision-language.

---

### Q2. Quantization reduces:
**Options:**
- A) Accuracy
- B) Precision (FP32 → INT8) ✓
- C) Parameters
- D) Layers

**Explanation**: Lower precision → smaller, faster model.

---

### Q3. RAG stands for:
**Options:**
- A) Random Access Generation
- B) Retrieval-Augmented Generation ✓
- C) Rapid AI Growth
- D) Resource Allocation Graph

**Explanation**: Retrieve docs then generate with LLM.

---

### Q4. Knowledge distillation:
**Options:**
- A) Removes data
- B) Student mimics teacher ✓
- C) Adds parameters
- D) Slows training

**Explanation**: Small model learns from large model.

---

### Q5. AutoGPT is:
**Options:**
- A) Training algorithm
- B) Autonomous agent ✓
- C) Model architecture
- D) Dataset

**Explanation**: Agent for autonomous task completion.

---

# ⚠️ Common Mistakes

1. **Over-quantizing**: Too aggressive → accuracy drop
2. **Not calibrating quantization**: Use representative data
3. **Ignoring RAG for factual tasks**: Reduces hallucination
4. **Pruning before convergence**: Prune trained model
5. **Wrong distillation temperature**: T=2-4 typical
6. **Not using agents for complex tasks**: Break down manually instead
7. **Forgetting to cite sources in RAG**: Provide references
8. **Assuming NAS always better**: Often manual design sufficient

---

# ⭐ One-Line Exam Facts

1. **CLIP**: Contrastive text-image pre-training
2. **Zero-shot**: CLIP can classify without task-specific training
3. **Pruning**: Remove unimportant weights (50-90% possible)
4. **Quantization**: FP32 → INT8 (4x smaller, 2-4x faster)
5. **Knowledge distillation**: L = L_task + α KL(student || teacher)
6. **RAG**: Query → Retrieve → Generate (grounds in facts)
7. **LangChain**: Framework for LLM applications
8. **AutoGPT**: Autonomous task-driven agent
9. **Function calling**: LLM invokes external APIs
10. **Constitutional AI**: Self-critique via principles
11. **AlphaFold**: Protein structure prediction
12. **NAS**: Automated architecture search
13. **Temperature T**: Soften probabilities for distillation
14. **Multimodal**: Multiple modalities (vision + language)
15. **Agent loop**: Plan → Execute → Evaluate → Repeat

---

**End of Session 30**

---

# 🎉 CONGRATULATIONS! 🎉

## All 30 Comprehensive ML Sessions Complete!

You now have detailed, exam-ready notes covering:
- ✅ Foundations (Information Theory, Learning Theory)
- ✅ Classical ML (Clustering, Trees, SVM, Regression)
- ✅ Deep Learning (Neural Networks, CNNs, RNNs, Transformers)
- ✅ Modern AI (BERT, GPT, LLMs, Generative Models)
- ✅ Advanced Topics (RL, MLOps, Ethics, Future Trends)

**Total**: 30 sessions × comprehensive coverage = Complete PG-DBDA ML preparation! 🚀

**Good luck with your exams!** 📚✨
