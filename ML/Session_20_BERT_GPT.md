# Session 20 – BERT & GPT Models

## 📚 Table of Contents
1. [Conceptual Understanding](#conceptual-understanding)
2. [Pre-training Paradigm](#pre-training-paradigm)
3. [BERT Architecture](#bert-architecture)
4. [GPT Architecture](#gpt-architecture)
5. [Fine-tuning Strategies](#fine-tuning-strategies)
6. [Common Misconceptions](#common-misconceptions)
7. [Edge Cases & Failure Modes](#edge-cases--failure-modes)
8. [Real-World Applications](#real-world-applications)
9. [Comparison Tables](#comparison-tables)
10. [Expanded MCQs](#expanded-mcqs)
11. [Exam-Focused Theory Points](#exam-focused-theory-points)
12. [Quick Recall Facts](#quick-recall-facts)
13. [Common Mistakes](#common-mistakes)
14. [One-Line Exam Facts](#one-line-exam-facts)

---

# 📘 Conceptual Understanding

## The Pre-training Revolution

### Before BERT/GPT (2017 and earlier)
- **Word2Vec/GloVe**: Fixed embeddings, no context
  - "bank" always has same vector (river bank = financial bank)
- **Task-specific training**: Train from scratch for each task
- **Limited data**: Most NLP tasks have small labeled datasets
- **Poor transfer**: Can't leverage knowledge across tasks

### After BERT/GPT (2018+)
-**Contextual embeddings**: Word meaning changes with context
  - "bank" in "river bank" ≠ "bank" in "savings bank"
- **Transfer learning**: Pre-train once, fine-tune for many tasks
- **Leverages unlabeled data**: Billions of web pages for pre-training
- **State-of-the-art**: Massive performance improvements across all NLP tasks

**Analogy**: 
- **Before**: Learning to be a doctor by only reading medical textbooks (specialized knowledge)
- **After**: General education first (pre-training), then medical school (fine-tuning)

---

## BERT: The Bidirectional Breakthrough

### The Core Insight

**Previous approaches** (including original Transformer decoder):
- Read text **left-to-right** only
- When processing "bank", can only see words to the left
- Missing right context!

**BERT's innovation**:
- Read text **both directions simultaneously**
- When processing "bank", sees entire sentence
- Captures full context!

**Example**: "I went to the bank to deposit money"
- **Left-to-right model**: Sees "I went to the bank to..." → predicts "river"?
- **BERT**: Sees entire sentence → knows it's financial bank (deposit money)

### Why "Masked" Language Modeling?

**Problem**: If model sees full sentence during training, prediction becomes trivial (just copy the input!).

**Solution**: **Mask** some words, predict them from context.

**Process**:
```
Original: "The cat sat on the mat"
Masked:   "The cat [MASK] on the mat"
Task:     Predict [MASK] = "sat"
```

**Why this works**: Forces model to understand context deeply to fill in blanks.

**Real-world analogy**: Cloze test (fill-in-the-blank) from language learning!

---

## GPT: The Generative Approach

### The Core Insight

**GPT's philosophy**: Language modeling is all you need!

**Task**: Given "The cat sat", predict next word.

**Why this works**:
- To predict well, model must understand grammar, facts, reasoning
- One simple objective trains a general-purpose model
- Scales incredibly well (GPT-3: 175B parameters!)

### Autoregressive Generation Explained

**Autoregressive**: Each prediction depends on previous predictions.

**Generation process**:
```
Prompt: "Once upon a time"

Step 1: Predict next word given "Once upon a time" → "there"
Step 2: Predict next word given "Once upon a time there" → "was"
Step 3: Predict next word given "Once upon a time there was" → "a"
Step 4: ... continue until [EOS] or max length
```

**Key property**: Cannot change previous words (left-to-right only)

**Advantage**: Natural for generation tasks (write text sequentially)
**Disadvantage**: Can't use future context for understanding tasks

---

## Pre-training vs Fine-tuning

### Pre-training Phase

**Dataset**: Massive unlabeled text (Wikipedia, books, web pages)
- BERT: 3.3B words (Wikipedia + Books)
- GPT-3: ~500B tokens (most of the internet!)

**Objective**:
- BERT: Predict masked words + next sentence
- GPT: Predict next word

**Duration**: Weeks/months on hundreds of GPUs/TPUs
**Cost**: Millions of dollars for large models

**Result**: General-purpose language understanding

### Fine-tuning Phase

**Dataset**: Task-specific labeled data
- Sentiment analysis: 10K-100K labeled reviews
- Question answering: SQuAD (100K examples)

**Objective**: Task-specific (classification, QA, etc.)

**Duration**: Hours to days on single GPU
**Cost**: Hundreds of dollars

**Result**: Task-specific model

**Why this works**: Pre-trained model already understands language, just needs to adapt to specific task!

---

## Evolution: GPT-1 → GPT-2 → GPT-3 →GPT-4

### GPT-1 (2018)
- **Size**: 117M parameters
- **Innovation**: Demonstrated pre-training + fine-tuning works
- **Limitation**: Still needed fine-tuning for each task

### GPT-2 (2019)
- **Size**: 1.5B parameters (13x larger!)
- **Innovation**: **Zero-shot learning** - can do tasks without fine-tuning!
- **Example**: Give it translated examples in prompt, it learns to translate
- **Controversy**: Initially not released due to "dangerous" text generation potential

### GPT-3 (2020)
- **Size**: 175B parameters (117x larger!)
- **Innovation**: **Few-shot learning** - learns from few examples in prompt
- **Emergent abilities**: Arithmetic, code generation, reasoning
- **Limitation**: No fine-tuning API initially (later added)

### GPT-4 (2023)
- **Size**: Unknown (rumored 1T+ parameters)
- **Innovation**: **Multimodal** (text + images), better reasoning
- **Performance**: Professional-level on many benchmarks (bar exam, medical licensing)

**Trend**: Bigger models → better zero/few-shot performance → less fine-tuning needed!

---

# Pre-training Paradigm

## 📘 Transfer Learning in NLP

**Evolution**:
1. Word embeddings (Word2Vec, GloVe) - context-free
2. Contextualized embeddings (ELMo) - context-dependent
3. **Pre-training + Fine-tuning** (BERT, GPT) - task-agnostic

**Key insight**: Pre-train on massive unlabeled data, fine-tune on task-specific data.

## Why Transfer Learning Works

### Linguistic Hierarchy

Pre-training learns linguistic features at multiple levels:

**Low-level** (early layers):
- Syntax: Part-of-speech, grammar rules
- Morphology: Word structure (plural, tense)

**Mid-level** (middle layers):
- Semantics: Word meanings, relationships
- Coreference: "it" refers to what?

**High-level** (late layers):
- Pragmatics: Implied meaning, sarcasm
- World knowledge: Facts, common sense

**Fine-tuning**: Adapts these features to specific task!

### The Data Efficiency Argument

**Traditional approach**:
- Need 100K+ labeled examples per task
- Expensive, time-consuming to collect
- Many tasks don't have this much data

**Transfer learning approach**:
- Pre-train on billions of unlabeled examples (free!)
- Fine-tune on 1K-10K labeled examples
- Achieves better performance with less labeled data

---

# BERT Architecture

## 📘 BERT = Bidirectional Encoder Representations from Transformers

**Key innovation**: Bidirectional context (unlike left-to-right GPT).

## 🧮 Architecture

- **Base**: 12 layers, 768 hidden, 12 heads, 110M parameters
- **Large**: 24 layers, 1024 hidden, 16 heads, 340M parameters

**Encoder-only** Transformer (no decoder).

## 🧮 Pre-training Tasks

### 1. Masked Language Modeling (MLM)

**Objective**: Predict masked tokens.

```
Input:  The cat [MASK] on the mat
Target: sat
```

**Process**:
- Randomly mask 15% of tokens
- 80%: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep original

**Why 80%-10%-10% split?**

1. **80% [MASK]**: Standard masked prediction
2. **10% random**: Prevents model from learning "[MASK] always means predict something"
3. **10% unchanged**: Teaches model to copy when appropriate

**Loss**: Cross-entropy on masked positions only.

### 2. Next Sentence Prediction (NSP)

**Objective**: Predict if sentence B follows sentence A.

```
Input:  [CLS] Sentence A [SEP] Sentence B [SEP]
Label:  IsNext (50%) or NotNext (50%)
```

**Why NSP?** 
- Helps with sentence-pair tasks (QA, textual entailment)
- Later research (RoBERTa) showed NSP not essential!

## 🧮 Input Representation

```
Final Embedding = Token Embeddings + Segment Embeddings + Position Embeddings
```

### Token Embeddings
- **WordPiece tokenization**: Splits rare words into subwords
- "unbelievable" → "un" + "##believ" + "##able"
- Vocabulary: 30,000 tokens

### Segment Embeddings
- Distinguish sentence A vs sentence B
- Sentence A tokens: Embedding A
- Sentence B tokens: Embedding B

### Position Embeddings
- **Learned** (not sinusoidal like original Transformer!)
- Absolute positions up to 512
- Cannot generalize beyond 512 tokens

**Special tokens**:
- [CLS]: Classification token (output used for sentence-level tasks)
- [SEP]: Separator between sentences
- [MASK]: Masked token
- [PAD]: Padding for batching

## 🧪 Fine-tuning

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
text = "This movie was great!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Forward pass
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
```

## 📊 BERT Variants

### RoBERTa (Robustly Optimized BERT)
- **Improvements**: Remove NSP, larger batches, more data
- **Result**: Better performance than BERT

### ALBERT (A Lite BERT)
- **Innovation**: Parameter sharing across layers
- **Result**: 18x fewer parameters, similar performance

### DistilBERT
- **Innovation**: Knowledge distillation (student learns from teacher)
- **Result**: 40% smaller, 60% faster, 97% performance

### ELECTRA
- **Innovation**: Discriminator (detect replaced tokens) instead of generator
- **Result**: More efficient pre-training

## 📊 BERT Applications

1. **Text Classification**: Sentiment, topic categorization
2. **Named Entity Recognition**: Token-level classification
3. **Question Answering**: SQuAD dataset
4. **Sentence Similarity**: Semantic textual similarity

---

# GPT Architecture

## 📘 GPT = Generative Pre-trained Transformer

**Key innovation**: Autoregressive language modeling (left-to-right).

## 🧮 Architecture Evolution

### GPT-1 (2018)
- 12 layers, 768 hidden, 117M parameters
- **Decoder-only** Transformer
- Demonstration of pre-training effectiveness

### GPT-2 (2019)
- 48 layers, 1600 hidden, 1.5B parameters
- Zero-shot task transfer
- No fine-tuning needed for some tasks!

### GPT-3 (2020)
- 96 layers, 12288 hidden, 175B parameters
- Few-shot learning via prompting
- "Emergent" abilities at scale

### GPT-4 (2023)
- Multimodal (text + images)
- Enhanced reasoning and factuality
- Professional/expert level performance

## 🧮 Pre-training

**Causal Language Modeling**: Predict next token.

```
Input:  The cat sat
Target: on

Training: maximize P(on | The cat sat)
```

**Loss**:
```
L = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**Why causal?** Each token only sees previous tokens (autoregressive property).

## 🧮 Inference

**Autoregressive generation**:
```
1. Start with prompt
2. Predict next token
3. Append to sequence
4. Repeat until [EOS] or max length
```

### Sampling Strategies

**Greedy**: Always pick highest probability token
- **Pro**: Deterministic
- **Con**: Repetitive, boring

**Top-k**: Sample from top k most likely tokens
- **Pro**: More diverse
- **Con**: Fixed k not always optimal

**Top-p (Nucleus)**: Sample from smallest set with cumulative probability ≥ p
- **Pro**: Adapts to probability distribution
- **Con**: Can still be repetitive

**Temperature**: Scale logits before softmax
- High temperature (>1): More random (creative)
- Low temperature (<1): More deterministic (focused)

## 🧪 Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')

# Generate
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,      # Control randomness
    top_p=0.9,           # Nucleus sampling
    do_sample=True       # Enable sampling
)

generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

## 📊 GPT's Emerging Abilities

### Arithmetic (GPT-3+)
```
Prompt: "Q: 17 + 42 = ?"
Output: "A: 59"
```

### Code Generation (Codex, GPT-4)
```
Prompt: "# Python function to compute fibonacci"
Output: [Generates working code]
```

### Reasoning (GPT-4)
```
Prompt: "If I have 3 apples and buy 2 more, then give 1 to my friend, how many do I have?"
Output: "You have 4 apples."
```

**Note**: These were NOT explicitly trained! Emerged from scale.

---

# Fine-tuning Strategies

## 📊 Approaches

### 1. Full Fine-tuning
Update all parameters on task-specific data.

**Pros**: Best performance
**Cons**: Expensive, risk catastrophic forgetting

**When to use**: High-stakes tasks, sufficient compute

### 2. Feature Extraction
Freeze BERT/GPT, train only task head.

**Pros**: Fast, less data needed
**Cons**: Lower performance

**When to use**: Limited compute, very small datasets

### 3. Layer-wise Learning Rates
Different LR for different layers (lower for early, higher for late).

**Rationale**:
- Early layers: General features (don't change much)
- Late layers: Task-specific features (adapt more)

**Typical**: LR_early = 0.1 × LR_late

### 4. Adapter Modules
Insert small trainable modules between layers, freeze main model.

**Pros**: 
- Only 1-2% extra parameters
- Can maintain multiple tasks (one adapter per task)
- No catastrophic forgetting

**Cons**: Slightly lower performance than full fine-tuning

### 5. Prompt Tuning (Soft Prompts)
Optimize continuous embeddings prepended to input, freeze model.

**Example**:
```
[learnable_emb_1] [learnable_emb_2] Is this review positive? [ACTUAL TEXT]
```

**Pros**: Extremely parameter-efficient (<0.1%)
**Cons**: Requires large pre-trained models to work well

### 6. LoRA (Low-Rank Adaptation)
Add low-rank matrices to weight updates.

**Math**: Instead of updating W → W + ΔW, use W + AB where A, B are low-rank
**Pros**: ~0.1% parameters, matches full fine-tuning performance
**Cons**: Slightly more complex implementation

---

# 🎯 Common Misconceptions

## Misconception 1: "BERT Can Generate Text Like GPT"

**Wrong thinking**: BERT and GPT are both Transformers, so BERT can generate.

**Truth**:
- BERT is **encoder-only**: Bidirectional attention (sees future!)
- Cannot generate autoregressively (would "cheat" by seeing what's ahead)
- Designed for understanding, not generation

**MCQ Trap**: "Which model is better for text generation: BERT or GPT?" → GPT!

---

## Misconception 2: "[MASK] Token is Just a Placeholder"

**Wrong thinking**: [MASK] doesn't matter, could use any symbol.

**Truth**:
- [MASK] is a **learned embedding** in BERT's vocabulary
- Model specifically learns to predict when it sees [MASK]
- Problem: [MASK] never appears during fine-tuning → train/test mismatch!
- This is why 10% of masked tokens are random/unchanged

**MCQ Trap**: "Why does BERT sometimes replace masked tokens with random words?" → Reduce train/test mismatch

---

## Misconception 3: "NSP is Essential for BERT"

**Wrong thinking**: BERT's performance comes from NSP task.

**Truth**:
- **Research (RoBERTa)** showed NSP provides minimal benefit
- Sometimes even hurts performance!
- Modern BERT variants often remove NSP
- MLM alone is sufficient

**MCQ Trap**: "RoBERTa removes which pre-training task?" → NSP

---

## Misconception 4: "GPT-3 Is 'Just' GPT-2 But Bigger"

**Wrong thinking**: GPT-3 = scaled-up GPT-2, no fundamental changes.

**Truth - Emergent abilities**:
- GPT-2: Weak arithmetic, no code generation
- GPT-3: Can do arithmetic, generate code, etc.
- **Not explicitly trained** for these tasks!
- Abilities "emerge" at scale

**Scaling laws**: Performance improves predictably with scale (size, data, compute)

**MCQ Trap**: "GPT-3's few-shot learning ability was explicitly trained" → FALSE!

---

## Misconception 5: "Fine-tuning Always Improves Performance"

**Wrong thinking**: More fine-tuning = better results.

**Truth**:
- **Catastrophic forgetting**: Fine-tuning can destroy pre-trained knowledge
- **Overfitting**: Too much fine-tuning on small dataset → poor generalization
- **Optimal**: Early stopping, validation set monitoring

**Best practices**:
- Use small learning rate (1e-5 to 5e-5)
- Early stopping based on validation
- Gradual unfreezing (fine-tune late layers first)

---

## Misconception 6: "BERT/GPT Understand Language Like Humans"

**Wrong thinking**: These models truly "understand" meaning.

**Truth**:
- Models learn statistical patterns, not true understanding
- Can be fooled by adversarial examples
- No grounding in real world (text-only)
- No causal reasoning (mostly pattern matching)

**Example failure**:
```
Input: "I put my phone in the fridge because it was too hot"
BERT might think "it" = phone (correct)
But doesn't understand WHY (to cool it down)
```

---

# ⚠️ Edge Cases & Failure Modes

## Edge Case 1: Maximum Sequence Length

**Problem**: BERT/GPT have hard limits on sequence length.

**Limits**:
- BERT: 512 tokens
- GPT-2: 1024 tokens
- GPT-3: 2048 tokens (some variants 4096)
- GPT-4: 8K-32K tokens

**What happens beyond limit?**
- Must truncate input (lose information!)
- Positional embeddings won't work (BERT)
- Cannot process long documents in one pass

**Solutions**:
- **Sliding window**: Process in chunks, aggregate
- **Hierarchical**: Summarize chunks, then process summaries
- **Long-range models**: Longformer, BigBird (sparse attention)

---

## Edge Case 2: Domain Shift

**Problem**: Pre-training data != fine-tuning domain.

**Example**:
- BERT trained on Wikipedia + books (formal English)
- Fine-tune on Twitter (informal, slang, emojis)
- **Result**: Poor performance!

**Why?**:
- Vocabulary mismatch (out-of-vocabulary tokens)
- Style mismatch (formal vs informal)
- Topic mismatch (encyclopedic vs social)

**Solution**: Domain-adaptive pre-training
- Continue pre-training on domain-specific unlabeled data
- THEN fine-tune on task

---

## Edge Case 3: Low-Resource Languages

**Problem**: Most pre-training data is English.

**Consequence**:
- BERT-base is mostly English
- Poor performance on non-English or low-resource languages
- Tokenizer biased toward English

**Solutions**:
- **mBERT**: Multilingual BERT (104 languages)
- **XLM-R**: Cross-lingual model (100 languages)
- Language-specific pre-training

**Caveat**: Performance still worse than English due to less training data

---

## Edge Case 4: Rare Word Handling

**Problem**: WordPiece splits unknown words into subwords.

**Example**:
- "antidisestablishmentarianism" → "anti" + "##dis" + "##establish" + "##ment" + "##arian" + "##ism"
- Long sequences from single word!
- May exceed token limit
- Harder for model to learn word-level meaning

**When this matters**:
- Technical domains (medical, legal)
- Named entities (person names, places)
- Code (variable names)

**Solution**:
- Domain-specific vocabularies
- Character-level models (less common)

---

## Edge Case 5: Catastrophic Forgetting in Fine-tuning

**Problem**: Fine-tuning on Task A destroys performance on all other tasks.

**Example**:
```
1. Pre-train BERT (knows general language)
2. Fine-tune on sentiment analysis (learns sentiment)
3. Try on NER → Performance collapsed!
```

**Why?**: Updating all weights overwrites pre-trained knowledge.

**Solutions**:
- **Multi-task learning**: Fine-tune on multiple tasks simultaneously
- **Adapters**: Keep base model frozen
- **Small learning rate**: Update weights gently

---

# 🌍 Real-World Applications

## Application 1: Google Search (BERT)

**Deployed**: 2019

**Use case**: Understanding search queries

**Example improvement**:
- Query: "2019 brazil traveler to usa need a visa"
- Old system: Confused by "to" (Brazil→USA or USA→Brazil?)
- BERT: Correctly understands direction (traveler FROM Brazil TO USA)
- **Result**: Better search results for 1 in 10 queries!

**Why BERT?**: Bidirectional context crucial for understanding query intent

---

## Application 2: Customer Service Chatbots (GPT)

**Use case**: Automated customer support

**Architecture**:
- GPT fine-tuned on customer service conversations
- Prompt: Customer query + context
- Generation: Helpful response

**Advantages**:
- 24/7 availability
- Consistent responses
- Handles common queries automatically
- Escalates complex cases to humans

**Limitations**:
- Can hallucinate (make up information)
- Requires careful prompt engineering
- Needs human oversight

---

## Application 3: Code Completion (GitHub Copilot)

**Model**: Codex (GPT-3 variant fine-tuned on code)

**Use case**: Autocomplete code from comments/function names

**Example**:
```python
# Function to compute the nth Fibonacci number
def fibonacci(n):
    # [Copilot generates the rest]
```

**Impact**:
- Developers code ~55% faster (GitHub study)
- Especially helpful for boilerplate, tests
- Learns from billions of lines of public code

---

## Application 4: Content Moderation (BERT/RoBERTa)

**Use case**: Detect toxic/offensive content on social media

**Task**: Multi-label classification
- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

**Why BERT**:
- Understands context (same word can be offensive or not)
- Handles sarcasm better than keyword matching
- Multilingual variants for global platforms

**Challenge**: Bias in training data → unfair moderation

---

## Application 5: Summarization (BART, T5)

**Models**: BART (BERT + GPT), T5 (encoder-decoder)

**Use case**: Summarize long documents

**Types**:
- **Extractive**: Select important sentences
- **Abstractive**: Generate new summary text

**Applications**:
- News aggregation
- Legal document review
- Scientific paper summaries
- Meeting notes

**Example**: Summarize 10-page report → 1-page executive summary

---

## Application 6: Question Answering (BERT)

**Dataset**: SQuAD (Stanford Question Answering Dataset)

**Task**: Given passage + question, extract answer span

**Example**:
```
Passage: "The Apollo 11 mission landed on the Moon on July 20, 1969."
Question: "When did Apollo 11 land on the Moon?"
Answer: "July 20, 1969" (extracted from passage)
```

**Real systems**:
- Google Assistant
- Amazon Alexa
- Virtual assistants

**Why BERT**: Bidirectional context helps locate answer precisely

---

# 📊 Comparison Tables

## Comparison 1: BERT vs GPT - Core Differences

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional ✓ | Unidirectional (causal) |
| **Pre-training** | MLM + NSP | Causal LM |
| **Training Objective** | Predict masked tokens | Predict next token |
| **Best for** | Understanding tasks (classification, extraction) | Generation tasks (completion, dialogue) |
| **Input** | Can see full context | Only left context |
| **Output** | Contextualized embeddings | Token probabilities |
| **Examples** | Sentiment analysis, NER, QA | Text generation, code completion, chatbots |
| **Positional Encoding** | Learned (up to 512) | Learned (varies by version) |
| **Special Tokens** | [CLS], [SEP], [MASK] | [EOS], [BOS] |

**MCQ Focus**: Which model for which task!

---

## Comparison 2: Fine-tuning Strategies

| Strategy | Trainable Params | Performance | Compute | Use Case |
|----------|-----------------|-------------|---------|----------|
| **Full Fine-tuning** | 100% | Best | High | High-stakes tasks, large data |
| **Feature Extraction** | ~1% (head only) | Moderate | Low | Small data, limited compute |
| **Adapter Modules** | ~2% | Near-full | Medium | Multiple tasks, avoid forgetting |
| **Layer-wise LR** | 100% (different rates) | Best | High | When training instability |
| **Prompt Tuning** | <0.1% | Good (large models) | Very Low | Extremely limited compute |
| **LoRA** | ~0.1% | Near-full | Low | Best balance (modern choice) |

**MCQ Focus**: Parameter efficiency vs performance tradeoff

---

## Comparison 3: BERT Variants

| Model | Innovation | Parameters | Speed vs BERT | Performance vs BERT |
|-------|-----------|------------|---------------|---------------------|
| **BERT-base** | Original | 110M | 1x | 1x (baseline) |
| **RoBERTa** | Remove NSP, more data | 125M | 1x | +2-3% |
| **ALBERT** | Parameter sharing | 18M | 0.5x | ~0.98x |
| **DistilBERT** | Knowledge distillation | 66M (40% smaller) | 1.6x faster | 0.97x |
| **ELECTRA** | Replaced token detection | 110M | 1x | ~1.02x |
| **DeBERTa** | Disentangled attention | 140M | 0.8x | +3-4% |

**MCQ Focus**: Which variant for which constraint (size/speed/performance)

---

## Comparison 4: GPT Evolution

| Model | Parameters | Training Data | Key Capability | Best Use |
|-------|------------|--------------|----------------|----------|
| **GPT-1** | 117M | BooksCorpus (5GB) | Pre-training works | Proof of concept |
| **GPT-2** | 1.5B | WebText (40GB) | Zero-shot tasks | Text generation |
| **GPT-3** | 175B | CommonCrawl (570GB) | Few-shot learning | General purpose |
| **GPT-4** | Unknown (~1T) | Multimodal data | Multimodal, reasoning | Advanced tasks |

**Scale trend**: Bigger → Better zero/few-shot performance → Less fine-tuning needed

---

## Comparison 5: Pre-training Tasks

| Task | Model | Objective | Masking? | Bidirectional? |
|------|-------|-----------|----------|----------------|
| **Masked LM** | BERT | Predict masked tokens | Yes | Yes |
| **Causal LM** | GPT | Predict next token | No | No (left-only) |
| **NSP** | BERT | Predict if B follows A | N/A | Yes |
| **Permutation LM** | XLNet | Predict in random order | Implicit | Yes |
| **Replaced Token Detection** | ELECTRA | Detect replaced tokens | Different | Yes |

**MCQ Focus**: Which task corresponds to which model

---

## Comparison 6: Tokenization Strategies

| Strategy | Example | Vocab Size | OOV Handling | Used In |
|----------|---------|------------|--------------|---------|
| **Word-level** | ["The", "cat"] | 50K-100K | Poor (many OOV) | Word2Vec |
| **Character-level** | ["T","h","e"," ","c","a","t"] | <100 | Perfect (no OOV) | Rare (too long) |
| **Subword (BPE)** | ["The", "Ġcat"] | 30K-50K | Good | GPT-2, GPT-3 |
| **WordPiece** | ["The", "cat"] | 30K | Good | BERT |
| **SentencePiece** | ["▁The", "▁cat"] | 32K | Good | T5, XLM |

**MCQ Focus**: BERT uses WordPiece, GPT-2 uses BPE

---

# 🔥 Expanded MCQs

### Q1. BERT uses:
**Options:**
- A) Decoder-only
- B) Encoder-only ✓
- C) Encoder-decoder
- D) No transformer

**Explanation**: BERT is encoder-only for bidirectional context.

---

### Q2. MLM masks what percent of tokens?
**Options:**
- A) 5%
- B) 15% ✓
- C) 25%
- D) 50%

**Explanation**: BERT masks 15% of tokens during pre-training.

---

### Q3. GPT pre-training task:
**Options:**
- A) Masked LM
- B) Next sentence prediction
- C) Causal language modeling ✓
- D) Translation

**Explanation**: GPT predicts next token autoregressively.

---

### Q4. [CLS] token in BERT:
**Options:**
- A) Separator
- B) Classification output ✓
- C) Mask token
- D) End token

**Explanation**: [CLS] output used for sentence-level tasks.

---

### Q5. GPT-3 has approximately:
**Options:**
- A) 100M parameters
- B) 1B parameters
- C) 175B parameters ✓
- D) 1T parameters

**Explanation**: GPT-3 has 175 billion parameters.

---

### Q6. Which model is better for text generation?
**Options:**
- A) BERT
- B) GPT ✓
- C) Both equally good
- D) Neither can generate

**Explanation**: GPT's autoregressive decoder design is perfect for generation. BERT is encoder-only (bidirectional), not suitable.

---

### Q7. BERT's positional encoding is:
**Options:**
- A) Sinusoidal (fixed)
- B) Learned embeddings ✓
- C) Not used
- D) Relative positions

**Explanation**: BERT learns absolute positional embeddings (unlike original Transformer's sinusoidal).

---

### Q8. In BERT's MLM, masked tokens are replaced with [MASK] what percentage of the time?
**Options:**
- A) 100%
- B) 80% ✓
- C) 15%
- D) 50%

**Explanation**: 80% → [MASK], 10% → random token, 10% → unchanged.

---

### Q9. RoBERTa improves BERT by:
**Options:**
- A) Adding NSP
- B) Removing NSP ✓
- C) Using sinusoidal PE
- D) Making it smaller

**Explanation**: RoBERTa removes NSP task, trains longer with more data → better performance.

---

### Q10. GPT-3's few-shot learning works by:
**Options:**
- A) Fine-tuning on examples
- B) Including examples in the prompt ✓
- C) Explicit training
- D) Doesn't do few-shot

**Explanation**: GPT-3 learns from examples given in the prompt (in-context learning), no fine-tuning needed!

---

### Q11. Maximum sequence length of BERT-base:
**Options:**
- A) 128 tokens
- B) 256 tokens
- C) 512 tokens ✓
- D) 1024 tokens

**Explanation**: BERT-base positional embeddings go up to 512.

---

### Q12. Catastrophic forgetting refers to:
**Options:**
- A) Model forgetting training data
- B) Fine-tuning destroying pre-trained knowledge ✓
- C) Optimizer forgetting gradients
- D) Tokenizer forgetting vocabulary

**Explanation**: Aggressively fine-tuning can overwrite useful pre-trained representations.

---

### Q13. Which uses causal (unidirectional) attention?
**Options:**
- A) BERT encoder
- B) GPT decoder ✓
- C) Original Transformer encoder
- D) RoBERTa

**Explanation**: GPT is decoder-only with causal (left-to-right) attention.

---

### Q14. DistilBERT achieves efficiency by:
**Options:**
- A) Reducing layers via knowledge distillation ✓
- B) Using sparse attention
- C) Quantization only
- D) Smaller vocabulary

**Explanation**: DistilBERT is a smaller student model trained to mimic BERT (teacher) via distillation.

---

### Q15. The [SEP] token in BERT is used to:
**Options:**
- A) Mask tokens
- B) Separate sentences ✓
- C) End generation
- D) Mark classification

**Explanation**: [SEP] separates sentence A from sentence B (especially for NSP task).

---

# 💡 Exam-Focused Theory Points

## Must-Remember Concepts

### 1. BERT Core Architecture
- **Type**: Encoder-only Transformer
- **Attention**: Bidirectional (can see full context)
- **Pre-training**: 
  - MLM (Masked Language Modeling) - 15% masks
  - NSP (Next Sentence Prediction) - later found less important
- **Sizes**: Base (110M), Large (340M)
- **Max length**: 512 tokens

**Exam Tip**: BERT = bidirectional = understanding/classification tasks

### 2. GPT Core Architecture
- **Type**: Decoder-only Transformer
- **Attention**: Unidirectional causal (left-to-right only)
- **Pre-training**: Causal Language Modeling (predict next token)
- **Evolution**: GPT-1 (117M) → GPT-2 (1.5B) → GPT-3 (175B) → GPT-4 (multimodal)
- **Key ability**: Autore gressive text generation

**Exam Tip**: GPT = generation = autoregressive = decoder-only

### 3. MLM Masking Strategy
```
15% of tokens are selected for masking:
- 80% → [MASK]
- 10% → random token
- 10% → unchanged
```

**Why mixed strategy?** Reduce train/test mismatch ([MASK] only in training, not fine-tuning)

### 4. Pre-training vs Fine-tuning

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| Data | Billions of tokens (unlabeled) | Thousands of examples (labeled) |
| Objective | Language modeling | Task-specific |
| Duration | Weeks/months | Hours/days |
| Cost | Millions $$ | Hundreds $$ |
| Result | General language model | Task-specific model |

### 5. BERT vs GPT Decision Matrix

**Use BERT when:**
- Classification (sentiment, topic)
- Token labeling (NER, POS tagging)
- Question answering (extract answer)
- Similarity (sentence pairs)
- Need full context

**Use GPT when:**
- Text generation
- Completion
- Creative writing
- Dialogue/chat
- Code generation
- Zero/few-shot tasks

### 6. Special Tokens

**BERT**:
- `[CLS]`: Classification token (sentence representation at output)
- `[SEP]`: Separator between sentences
- `[MASK]`: Masked token
- `[PAD]`: Padding

**GPT**:
- `[BOS]`: Beginning of sequence
- `[EOS]`: End of sequence

### 7. Fine-tuning Best Practices
- **Learning rate**: 1e-5 to 5e-5 (small!)
- **Epochs**: 2-4 (few!)
- **Early stopping**: Monitor validation loss
- **Gradual unfreezing**: Fine-tune late layers first, gradually unfreeze earlier layers

### 8. Key Model Variants

**BERT family**:
- RoBERTa: Remove NSP, more training
- ALBERT: Parameter sharing → smaller
- DistilBERT: Knowledge distillation → faster
- ELECTRA: Replaced token detection → more efficient

**GPT family**:
- GPT-2: Zero-shot
- GPT-3: Few-shot
- Codex: Code generation
- ChatGPT: RLHF tuned for conversation

## Formula Quick Reference

**MLM Loss (BERT)**:
```
L = -Σ_(i∈masked) log P(x_i | context)
```

**Causal LM Loss (GPT)**:
```
L = -Σ_t log P(x_t | x_1,...,x_{t-1})
```

**Temperature Scaling (Generation)**:
```
P_i = exp(logit_i / T) / Σ_j exp(logit_j / T)
```
- T > 1: More random (creative)
- T < 1: More deterministic (focused)

## Common Exam Patterns

### Pattern 1: Model Selection
**Question**: "Which model for sentiment classification?"
**Answer**: BERT (understanding task, needs bidirectional context)

### Pattern 2: Masking Percentage
**Question**: "In BERT MLM, what percentage of tokens are masked?"
**Answer**: 15% (with 80%-10%-10% replacement strategy)

### Pattern 3: Architecture Type
**Question**: "GPT uses which Transformer component?"
**Answer**: Decoder-only (with causal attention)

### Pattern 4: Capability Comparison
**Question**: "Which model better for text generation?"
**Answer**: GPT (autoregressive decoder design)

### Pattern 5: Special Token Function
**Question**: "What is [CLS] token used for?"
**Answer**: Sentence representation for classification (use [CLS] output)

---

# ⚡ Quick Recall Facts

## BERT Quick Facts
- ✅ BERT = **Bidirectional** Encoder Representations from Transformers
- ✅ Architecture: **Encoder-only** Transformer
- ✅ Attention: **Bidirectional** (full context)
- ✅ Pre-training: **MLM** (15% masks) + **NSP**
- ✅ Sizes: Base (110M), Large (340M)
- ✅ Max length: **512 tokens**
- ✅ Positional encoding: **Learned** (not sinusoidal)
- ✅ Best for: **Classification, NER, QA**

## GPT Quick Facts
- ✅ GPT = **Generative** Pre-trained Transformer
- ✅ Architecture: **Decoder-only** Transformer
- ✅ Attention: **Causal** (left-to-right only)
- ✅ Pre-training: **Causal LM** (predict next token)
- ✅ Evolution: GPT-1 (117M) → GPT-2 (1.5B) → GPT-3 (175B)
- ✅ GPT-3: **Few-shot learning** via prompting
- ✅ Best for: **Text generation, completion, dialogue**

## Pre-training Quick Facts
- ✅ Goal: Learn **general language understanding**
- ✅ Data: **Billions** of unlabeled tokens
- ✅ Duration: **Weeks to months**
- ✅ Cost: **Millions of dollars** (large models)
- ✅ Result: **Reusable** for many tasks

## Fine-tuning Quick Facts
- ✅ Goal: Adapt to **specific task**
- ✅ Data: **Thousands** of labeled examples
- ✅ Duration: **Hours to days**
- ✅ Learning rate: **1e-5 to 5e-5** (very small!)
- ✅ Risk: **Catastrophic forgetting**

## MLM (Masked Language Modeling) Quick Facts
- ✅ Mask: **15%** of tokens
- ✅ [MASK]: **80%** of masked tokens
- ✅ Random: **10%** of masked tokens
- ✅ Unchanged: **10%** of masked tokens
- ✅ Loss: Only on **masked positions**

## Special Tokens Quick Facts
- ✅ [CLS]: **Classification** representation (BERT)
- ✅ [SEP]: **Separator** between sentences (BERT)
- ✅ [MASK]: **Masked** token (BERT)
- ✅ [PAD]: **Padding** for batching
- ✅ [EOS]: **End** of sequence (GPT)

## Tokenization Quick Facts
- ✅ BERT: **WordPiece** tokenization
- ✅ GPT-2/3: **BPE** (Byte-Pair Encoding)
- ✅ Vocabulary: **30K-50K** tokens typically
- ✅ Rare words: Split into **subwords**
- ✅ Example: "unbelievable" → "un" + "##believ" + "##able"

## Fine-tuning Strategies Quick Facts
- ✅ **Full**: Update all params → best performance
- ✅ **Feature extraction**: Freeze base, train head → fastest
- ✅ **Adapters**: Add small modules → parameter-efficient
- ✅ **LoRA**: Low-rank adaptation → modern best balance
- ✅ **Prompt tuning**: Optimize prompts → extreme efficiency

## Model Variants Quick Facts
- ✅ **RoBERTa**: BERT without NSP, more training
- ✅ **ALBERT**: Parameter sharing → 18M params
- ✅ **DistilBERT**: Knowledge distillation → 40% smaller, 60% faster
- ✅ **ELECTRA**: Replaced token detection → more efficient
- ✅ **DeBERTa**: Disentangled attention → state-of-the-art

## MCQ Trap Alerts 🚨
- 🚨 "BERT can generate text like GPT" → **FALSE** (encoder-only!)
- 🚨 "GPT is bidirectional" → **FALSE** (causal/unidirectional!)
- 🚨 "BERT uses sinusoidal PE" → **FALSE** (learned embeddings!)
- 🚨 "100% of masked tokens replaced with [MASK]" → **FALSE** (80%!)
- 🚨 "NSP is essential for BERT" → **FALSE** (RoBERTa shows it's not!)
- 🚨 "GPT-3 can only generate" → **FALSE** (can do classification too via prompting!)
- 🚨 "Fine-tuning always improves performance" → **FALSE** (can cause catastrophic forgetting!)
- 🚨 "BERT has decoder" → **FALSE** (encoder-only!)

---

# ⚠️ Common Mistakes

1. **Using BERT for generation**: BERT not designed for text generation (encoder-only, bidirectional)
2. **Not accounting for max sequence length**: BERT/GPT have limits (512/1024/2048)
3. **Ignoring special tokens**: [CLS], [SEP] critical for BERT tasks
4. **Wrong attention mask**: Causal mask for GPT, bidirectional for BERT
5. **Catastrophic forgetting**: Fine-tuning can destroy pre-trained knowledge (use small LR!)
6. **100% masking with [MASK]**: Should use 80%-10%-10% split
7. **Treating [CLS] as regular token**: [CLS] output specifically designed for classification
8. **Forgetting truncation**: Sequences > max length must be truncated/chunked
9. **Using high learning rates**: Fine-tuning needs 1e-5 to 5e-5, not 1e-3!
10. **Assuming GPT can't classify**: GPT can classify via clever prompting/few-shot

---

# ⭐ One-Line Exam Facts

1. **BERT**: Bidirectional encoder-only Transformer, MLM + NSP pre-training
2. **GPT**: Unidirectional decoder-only Transformer, causal LM pre-training
3. **MLM**: Mask 15% tokens (80% [MASK], 10% random, 10% unchanged), predict them
4. **[CLS]**: BERT classification token providing sentence representation from output
5. **Autoregressive**: GPT generates one token at a time, feeding output back as input
6. **Fine-tuning**: Adapt pre-trained model to specific task with tiny learning rate (1e-5)
7. **GPT-3**: 175B parameters, few-shot learning via prompting, no fine-tuning needed
8. **NSP**: Next Sentence Prediction in BERT, later found unnecessary (removed in RoBERTa)
9. **BERT applications**: Text classification, NER, question answering, sentence similarity
10. **GPT applications**: Text generation, completion, dialogue, code generation
11. **Transfer learning**: Pre-train on massive unlabeled data, fine-tune on small labeled data
12. **Catastrophic forgetting**: Fine-tuning overwrites pre-trained knowledge (use small LR!)
13. **RoBERTa**: BERT without NSP + more training = better performance
14. **DistilBERT**: Knowledge distillation makes BERT 40% smaller, 60% faster, 97% performance
15. **GPT evolution**: GPT-1 (supervised) → GPT-2 (zero-shot) → GPT-3 (few-shot) → GPT-4 (multimodal)

---

**End of Session 20**
