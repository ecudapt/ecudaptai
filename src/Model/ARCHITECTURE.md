# ECUdapt AI - LLM Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECUdapt AI LLM Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Scraper    │─────▶│    Model     │─────▶│  Inference   │
│  (Forums)    │      │  (Training)  │      │   (Chat)     │
└──────────────┘      └──────────────┘      └──────────────┘
```

## Detailed Architecture

### 1. Data Flow

```
Forum Websites
      │
      ▼
┌─────────────────┐
│  Web Scraper    │  ← Scrapy + Playwright
│  (Parallel)     │
└─────────────────┘
      │
      ▼ Raw JSONL
┌─────────────────┐
│  Data Cleaner   │  ← BeautifulSoup + Zero-shot ML
│  (Filter)       │
└─────────────────┘
      │
      ▼ Clean JSONL
┌─────────────────┐
│ Data Preparator │  ← Instruction pair extraction
│  (Transform)    │
└─────────────────┘
      │
      ▼ Training pairs
┌─────────────────┐
│  LLM Trainer    │  ← LoRA/QLoRA fine-tuning
│  (Fine-tune)    │
└─────────────────┘
      │
      ▼ Trained model
┌─────────────────┐
│ Inference Engine│  ← Generate responses
│  (Deploy)       │
└─────────────────┘
```

### 2. Training Pipeline Components

#### A. Data Preparation (`data_preparation.py`)

**Input**: Clean forum posts
```json
{
  "url": "https://forum.com/thread/123",
  "forum": "ecuedit.com",
  "title": "How to tune boost control?",
  "text": "I'm trying to tune boost on my N54..."
}
```

**Processing**:
1. Extract questions from titles
2. Identify technical concepts (boost, fuel, timing, etc.)
3. Find problem-solution patterns
4. Create instruction-response pairs

**Output**: Training data
```json
{
  "instruction": "How to tune boost control on N54?",
  "context": "From ecuedit.com tuning discussion",
  "response": "Boost control on N54 involves...",
  "metadata": {"type": "title_question", "forum": "ecuedit.com"}
}
```

#### B. Model Training (`train_llm.py`)

**Architecture**:
```
Base Model (e.g., Mistral-7B)
      │
      ▼
┌─────────────────────┐
│  4-bit Quantization │  ← QLoRA (reduces memory 4x)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   LoRA Adapters     │  ← Low-rank adaptation (trainable)
│   (r=16, α=32)      │     Only trains 0.1% of parameters
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Fine-tuned Model   │  ← Specialized for ECU tuning
└─────────────────────┘
```

**Training Process**:
```python
# Pseudo-code
model = load_base_model("mistralai/Mistral-7B")
model = apply_4bit_quantization(model)
model = add_lora_adapters(model, r=16, alpha=32)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        outputs = model(batch.input_ids, labels=batch.labels)
        loss = outputs.loss

        # Backward pass (only updates LoRA parameters)
        loss.backward()
        optimizer.step()
```

**Memory Efficiency**:
- **Without QLoRA**: 7B model = ~28GB VRAM
- **With QLoRA**: 7B model = ~8GB VRAM
- **Trainable params**: ~4M (0.06% of 7B)

#### C. Inference (`inference.py`)

**Generation Process**:
```
User Question
      │
      ▼
┌─────────────────────┐
│ Format Prompt       │  ← Add system prompt + context
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Tokenize            │  ← Convert to token IDs
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Generate Tokens     │  ← Autoregressive sampling
│ (temperature=0.7)   │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Decode Response     │  ← Token IDs → Text
└─────────────────────┘
      │
      ▼
Final Answer
```

### 3. Model Configurations

#### Standard Config (LLMConfig)
```python
base_model: "mistralai/Mistral-7B-v0.1"
max_seq_length: 2048
batch_size: 4
gradient_accumulation: 4
learning_rate: 2e-5
num_epochs: 3

use_lora: True
lora_r: 16
lora_alpha: 32
use_4bit: True  # QLoRA
```

**Requirements**: 12GB VRAM, 32GB RAM

#### Lightweight Config
```python
base_model: "microsoft/phi-2"  # 2.7B params
max_seq_length: 1024
batch_size: 1
gradient_accumulation: 8
num_epochs: 5
```

**Requirements**: 6GB VRAM, 16GB RAM

### 4. Hybrid RAG Architecture

Combine fine-tuned model with vector search for best results:

```
User Question
      │
      ├────────────────┐
      ▼                ▼
┌──────────┐    ┌──────────────┐
│  Vector  │    │  Fine-tuned  │
│  Search  │    │     LLM      │
│ (FAISS)  │    │              │
└──────────┘    └──────────────┘
      │                │
      ▼                │
  Top K Docs           │
      │                │
      └────────▶───────┘
             │
             ▼
      ┌──────────────┐
      │   Enhanced   │
      │   Response   │
      └──────────────┘
```

**Workflow**:
1. User asks question
2. Vector search finds relevant forum threads
3. LLM generates answer using:
   - Its fine-tuned knowledge
   - Retrieved context from vector search
4. Response combines both sources

### 5. Training Metrics & Monitoring

**TensorBoard Dashboard**:
```
┌─────────────────────────────────────┐
│  Training Loss       Validation Loss│
│  ─────────────      ───────────────│
│   ╱╲                    ╱╲          │
│  ╱  ╲___              ╱  ╲__       │
│              ▼                  ▼   │
│  Decreasing          Stable         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Learning Rate       GPU Memory     │
│  ─────────────      ───────────────│
│  ───────┐               ┌────────  │
│         └───────        │           │
│  Warmup + Decay         ~8GB        │
└─────────────────────────────────────┘
```

### 6. Evaluation Metrics

**ROUGE Scores** (measure text similarity):
- **ROUGE-1**: Unigram overlap (individual word matches)
- **ROUGE-2**: Bigram overlap (2-word phrase matches)
- **ROUGE-L**: Longest common subsequence

**Example**:
```
Reference: "Boost control on N54 uses wastegate solenoid duty cycle"
Generated: "N54 boost control works by modulating wastegate solenoid"

ROUGE-1: 0.67  (6/9 words match)
ROUGE-2: 0.43  (3/7 bigrams match)
ROUGE-L: 0.56  (longest sequence: "boost control wastegate solenoid")
```

### 7. Optimization Techniques

#### Memory Optimization
```
Full Precision (FP32)
  ├─ 7B model = 28GB
  └─ Training impossible on consumer GPUs

Mixed Precision (FP16)
  ├─ 7B model = 14GB
  └─ 2x faster, 2x less memory

8-bit Quantization
  ├─ 7B model = 7GB
  └─ 4x less memory, minimal quality loss

4-bit Quantization (QLoRA)
  ├─ 7B model = 3.5GB
  └─ 8x less memory, slight quality trade-off
  └─ + LoRA adapters = ~8GB total
```

#### Speed Optimization
```
Gradient Accumulation
  ├─ Effective batch size = batch_size × accumulation_steps
  ├─ batch_size=1, steps=16 → effective batch=16
  └─ Trades speed for memory

Gradient Checkpointing
  ├─ Recomputes activations during backward pass
  ├─ 30% slower, 40% less memory
  └─ Enabled by default

Flash Attention
  ├─ Optimized attention computation
  ├─ 2-4x faster than standard attention
  └─ Requires CUDA, Ampere GPU or newer
```

### 8. Production Deployment

#### Option 1: Local Inference
```python
from inference import ECUTuningLLM

llm = ECUTuningLLM("models/ecutuning-llm")
llm.load_model()
response = llm.generate(question)
```

#### Option 2: API Server
```python
from fastapi import FastAPI
from inference import ECUTuningLLM

app = FastAPI()
llm = ECUTuningLLM("models/ecutuning-llm")
llm.load_model()

@app.post("/generate")
def generate(question: str):
    return {"response": llm.generate(question)}
```

#### Option 3: Supabase Edge Function
```typescript
import { serve } from "https://deno.land/std/http/server.ts"

serve(async (req) => {
  const { question } = await req.json()

  // Call Python inference server
  const response = await fetch("http://inference:8000/generate", {
    method: "POST",
    body: JSON.stringify({ question })
  })

  return new Response(JSON.stringify(response))
})
```

### 9. Data Privacy & Security

- ✅ All training data from public forums
- ✅ No personally identifiable information (PII)
- ✅ Model runs locally (no cloud API)
- ✅ User conversations not logged
- ✅ Open-source base models (permissive licenses)

### 10. Future Enhancements

**Short-term**:
- [ ] Multi-turn conversation support
- [ ] Citation of source threads
- [ ] Confidence scores on responses

**Medium-term**:
- [ ] Multi-modal (include diagrams/charts)
- [ ] Active learning (user feedback)
- [ ] Domain-specific evaluation metrics

**Long-term**:
- [ ] Larger models (13B, 70B)
- [ ] Multi-language support
- [ ] Real-time learning from new posts
