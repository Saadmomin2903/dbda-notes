# Session 25 – Advanced NLP & Sequence Models

## 📚 Table of Contents
1. [Word Embeddings](#word-embeddings)
2. [Sequence-to-Sequence Models](#sequence-to-sequence-models)
3. [Attention in Seq2Seq](#attention-in-seq2seq)
4. [Named Entity Recognition](#named-entity-recognition)
5. [MCQs](#mcqs)
6. [Common Mistakes](#common-mistakes)
7. [One-Line Exam Facts](#one-line-exam-facts)

---

# Word Embeddings

## 📘 Word2Vec

### Skip-Gram
**Objective**: Predict context from word
```
Input: "cat"
Output: ["the", "sat", "on", "mat"]
```

**Loss** (Negative sampling):
```
L = -log σ(v_w^T v_c) - Σ_neg log σ(-v_w^T v_neg)
```

### CBOW (Continuous Bag of Words)
**Objective**: Predict word from context
```
Input: ["the", "sat", "on", "mat"]
Output: "cat"
```

## 📘 GloVe (Global Vectors)

**Matrix factorization** on co-occurrence statistics:
```
X_ij = number of times word j appears in context of word i
```

**Objective**:
```
L = Σ f(X_ij)(w_i^T w_j + b_i + b_j - log X_ij)²
```

## 📘 Contextual Embeddings

**ELMo**: Bidirectional LSTM, context-dependent vectors
**BERT embeddings**: Layer outputs as features (dynamic per context)

---

# Sequence-to-Sequence Models

## 🧮 Architecture

```
Encoder: Input sequence → Context vector c
Decoder: Context c → Output sequence
```

**Encoder** (RNN/LSTM):
```
h_t = f(h_{t-1}, x_t)
c = h_T (final hidden state)
```

**Decoder**:
```
s_t = g(s_{t-1}, y_{t-1}, c)
y_t = softmax(W s_t)
```

## 📊 Applications
- Machine translation
- Text summarization
- Dialogue systems
- Image captioning

---

# Attention in Seq2Seq

## 🧮 Attention Mechanism

**Problem**: Single context vector bottleneck.

**Solution**: Dynamic context at each decoding step.

**Alignment scores**:
```
e_ij = score(s_{i-1}, h_j)  # decoder state i-1, encoder state j
α_ij = softmax(e_ij)  # attention weights
c_i = Σ α_ij h_j  # context vector for step i
```

**Scoring functions**:
- Dot product: s^T h
- Additive: v^T tanh(W_s s + W_h h)

## 🧪 Seq2Seq with Attention

```python
class Seq2SeqAttention(nn.Module):
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, seq_len, hidden)
        # decoder_hidden: (batch, hidden)
        
        # Compute attention scores
        scores = torch.bmm(
            encoder_outputs,
            decoder_hidden.unsqueeze(2)
        ).squeeze(2)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        
        return context, attn_weights
```

---

# Named Entity Recognition

## 📘 Task
Identify named entities (Person, Location, Organization, etc.)

**Example**:
```
"Barack Obama was born in Hawaii"
[Barack Obama]_PERSON was born in [Hawaii]_LOCATION
```

## 🧮 Approaches

### CRF (Conditional Random Field)
Model label dependencies:
```
P(y|x) ∝ exp(Σ_i,k λ_k f_k(y_i, y_{i-1}, x, i))
```

### BiLSTM-CRF
```
Input → BiLSTM → Emission scores → CRF → Label sequence
```

### BERT for NER
Token classification with [CLS] and fine-tuning.

---

# 🔥 MCQs

### Q1. Skip-gram predicts:
**Options:**
- A) Word from context
- B) Context from word ✓
- C) Next word
- D) Previous word

**Explanation**: Skip-gram: center word → context words.

---

### Q2. Seq2Seq bottleneck:
**Options:**
- A) Too many parameters
- B) Fixed-length context vector ✓
- C) Slow training
- D) No issue

**Explanation**: Single vector compresses entire sequence.

---

### Q3. Attention mechanism:
**Options:**
- A) Fixes bottleneck ✓
- B) Slows training
- C) Reduces parameters
- D) Not useful

**Explanation**: Dynamic context at each step.

---

### Q4. GloVe uses:
**Options:**
- A) Neural network
- B) Co-occurrence matrix ✓
- C) Transformers
- D) RNN

**Explanation**: Matrix factorization on word co-occurrences.

---

### Q5. BiLSTM-CRF for:
**Options:**
- A) Translation
- B) NER (sequence labeling) ✓
- C) Generation
- D) Classification

**Explanation**: Captures label dependencies for NER.

---

# ⚠️ Common Mistakes

1. **Using static embeddings for context-dependent tasks**: Use ELMo/BERT
2. **Not using attention for long sequences**: Critical for seq2seq
3. **Wrong decoding strategy**: Beam search usually better than greedy
4. **Ignoring label dependencies in NER**: CRF helps
5. **Not fine-tuning embeddings**: Often improves task performance

---

# ⭐ One-Line Exam Facts

1. **Word2Vec Skip-gram**: Word → context prediction
2. **CBOW**: Context → word prediction
3. **GloVe**: Matrix factorization on co-occurrences
4. **ELMo**: Contextual embeddings from BiLSTM
5. **Seq2Seq**: Encoder-decoder architecture
6. **Attention**: Dynamic context vector, fixes bottleneck
7. **Beam search**: Keep top-k hypotheses (better than greedy)
8. **NER**: Named entity recognition (sequence labeling)
9. **CRF**: Models label dependencies
10. **BiLSTM-CRF**: Standard for sequence labeling

---

**End of Session 25**
