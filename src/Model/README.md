# ECUdapt AI - Custom LLM Training

Build your own fine-tuned Large Language Model specialized in automotive ECU tuning, trained on real forum discussions.

## 🎯 Overview

This directory contains a complete pipeline for:
1. **Data Preparation**: Convert forum posts into instruction-response training pairs
2. **Model Training**: Fine-tune a base LLM using LoRA/QLoRA for efficient training
3. **Evaluation**: Test model performance on validation data
4. **Inference**: Use your trained model for ECU tuning assistance
5. **RAG Integration**: Combine with vector search for enhanced responses

## 📁 File Structure

```
Model/
├── llm_config.py           # Configuration for training/inference
├── data_preparation.py     # Convert forum data to training format
├── train_llm.py           # Main training script with LoRA/QLoRA
├── inference.py           # Load and use trained model
├── evaluate.py            # Evaluation framework
├── pipeline.py            # End-to-end orchestrator
├── embedder.py            # Create vector embeddings (existing)
├── data_cleaner.py        # Clean scraped data (existing)
├── query_engine.py        # Vector search (existing)
└── requirements_llm.txt   # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_llm.txt
```

**Note**: For GPU training, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Complete Pipeline

```bash
# Run everything in one command
python pipeline.py

# Or use lightweight config for smaller GPUs/CPU
python pipeline.py --lightweight
```

### 3. Use Your Trained Model

```bash
# Interactive chat
python inference.py

# Single question
python inference.py --question "How do I tune boost control on an N54?"
```

## 📊 Pipeline Steps

### Step 1: Data Cleaning
Filters and cleans scraped forum data:
```bash
python data_cleaner.py
```
- Removes HTML/formatting
- Filters for ECU tuning relevance
- Uses ML zero-shot classification
- Output: `data/clean/forum_posts_clean.jsonl`

### Step 2: Data Preparation
Converts posts to instruction-response pairs:
```bash
python data_preparation.py
```
- Extracts questions from titles
- Identifies technical concepts
- Creates problem-solution pairs
- Output: `data/training/train.jsonl` & `val.jsonl`

### Step 3: Training
Fine-tunes the base model:
```bash
# Default (Mistral-7B with 4-bit quantization)
python train_llm.py

# Lightweight (Phi-2 2.7B model)
python train_llm.py --lightweight

# Custom model
python train_llm.py --model "meta-llama/Llama-2-7b-hf"

# Override hyperparameters
python train_llm.py --epochs 5 --batch-size 2
```

**Training features:**
- ✅ 4-bit quantization (QLoRA) for consumer GPUs
- ✅ LoRA adapters for efficient fine-tuning
- ✅ Gradient checkpointing for memory efficiency
- ✅ Automatic mixed precision (FP16)
- ✅ TensorBoard logging

### Step 4: Evaluation
Test model performance:
```bash
python evaluate.py --max-samples 100
```

Metrics:
- ROUGE scores (response quality)
- Generation latency
- Success rate
- Sample outputs

### Step 5: Inference
Use the trained model:
```bash
# Interactive chat mode
python inference.py

# CLI mode
python inference.py \
  --question "What causes boost lag in turbocharged engines?" \
  --context "I have an N54 engine"
```

## ⚙️ Configuration

Edit `llm_config.py` to customize:

```python
@dataclass
class LLMConfig:
    # Model
    base_model: str = "mistralai/Mistral-7B-v0.1"

    # Training
    max_seq_length: int = 2048
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Quantization
    use_4bit: bool = True
```

## 🎓 Model Choices

### Recommended Base Models

**For Consumer GPUs (8-16GB VRAM):**
- `microsoft/phi-2` (2.7B params) - Fastest, great quality
- `mistralai/Mistral-7B-v0.1` (7B params) - Best balance
- `TinyLlama/TinyLlama-1.1B` (1.1B params) - CPU-friendly

**For High-End GPUs (24GB+ VRAM):**
- `meta-llama/Llama-2-13b-hf` (13B params)
- `mistralai/Mixtral-8x7B-v0.1` (46B params)

**Open Source & Commercial-Friendly:**
- `google/gemma-7b` (7B params)
- `stabilityai/stablelm-2-12b` (12B params)

## 💡 Training Tips

### Memory Requirements

| Model Size | Quantization | Min VRAM | Recommended |
|------------|--------------|----------|-------------|
| 1-2B       | 4-bit        | 4GB      | 6GB         |
| 7B         | 4-bit        | 8GB      | 12GB        |
| 7B         | 8-bit        | 12GB     | 16GB        |
| 13B        | 4-bit        | 12GB     | 16GB        |
| 13B        | 8-bit        | 20GB     | 24GB        |

### Optimization Strategies

**Limited GPU Memory:**
```bash
python train_llm.py \
  --lightweight \
  --batch-size 1 \
  --gradient-accumulation-steps 16
```

**Faster Training:**
```bash
python train_llm.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 3
```

**Better Quality:**
```bash
python train_llm.py \
  --model "mistralai/Mistral-7B-v0.1" \
  --epochs 5 \
  --learning-rate 1e-5
```

## 🔧 Advanced Usage

### Run Individual Steps

```bash
# Only prepare data
python pipeline.py --step prepare

# Only train
python pipeline.py --step train

# Only evaluate
python pipeline.py --step evaluate
```

### Custom Training Data

Create your own `train.jsonl`:
```json
{
  "instruction": "How to tune MAF scaling?",
  "context": "VW Golf GTI with stage 2 software",
  "response": "MAF scaling compensates for...",
  "metadata": {"source": "custom"}
}
```

### Integration with RAG (Retrieval-Augmented Generation)

Combine fine-tuned model with vector search:

```python
from inference import ECUTuningLLM
from query_engine import ForumRetriever

# Load model and retriever
llm = ECUTuningLLM(Path("models/ecutuning-llm"))
llm.load_model()
retriever = ForumRetriever()

# Query
question = "How to reduce turbo lag?"
docs = retriever.search(question, k=3)
context = "\n".join([d["url"] for d in docs])

# Generate with retrieved context
response = llm.generate(question, context)
```

## 📈 Monitoring Training

View training progress in TensorBoard:
```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to see:
- Training loss
- Validation loss
- Learning rate schedule
- GPU utilization

## 🧪 Testing

Test model on specific scenarios:

```python
from inference import ECUTuningLLM

llm = ECUTuningLLM(Path("models/ecutuning-llm"))
llm.load_model()

test_questions = [
    "Why does boost taper at high RPM?",
    "How to tune fuel trims?",
    "What's the difference between open and closed loop?",
]

for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {llm.generate(q)}\n")
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_llm.py --batch-size 1

# Use gradient accumulation
python train_llm.py --batch-size 1 --gradient-accumulation-steps 16

# Use smaller model
python train_llm.py --lightweight
```

### Model Not Loading

```bash
# Check model path
ls -la models/ecutuning-llm/

# Should contain:
# - config.json
# - adapter_model.bin (or pytorch_model.bin)
# - tokenizer files
```

### Poor Quality Responses

- Train for more epochs: `--epochs 5`
- Use larger base model
- Add more training data
- Adjust learning rate: `--learning-rate 1e-5`

## 📚 Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Model Hub](https://huggingface.co/models)

## 🎯 Next Steps

1. **Expand Training Data**: Scrape more forums
2. **Experiment with Models**: Try different base models
3. **Fine-tune Hyperparameters**: Adjust learning rate, epochs
4. **Deploy**: Create API endpoint with FastAPI
5. **RAG Integration**: Combine with vector search
6. **Continuous Learning**: Retrain with new forum data

## 📝 License

This project uses various base models - check their individual licenses:
- Mistral: Apache 2.0
- Phi-2: MIT
- Llama 2: Meta License (commercial use allowed)
