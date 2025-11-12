# Getting Started with ECUdapt AI LLM

A complete guide to building your own ECU tuning language model from scratch.

## 🎯 What You'll Build

A specialized AI assistant that:
- Understands ECU tuning terminology and concepts
- Answers questions about engine tuning, remapping, boost control, etc.
- Is trained on real forum discussions from the tuning community
- Runs locally on your hardware (no cloud dependencies)
- Can be customized and extended for your needs

## 📋 Prerequisites

### Hardware Requirements

**Minimum** (for training):
- CPU: 4+ cores
- RAM: 16GB
- GPU: 8GB VRAM (NVIDIA with CUDA support)
- Storage: 50GB free space

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB
- GPU: 16GB+ VRAM (RTX 3090, 4090, or similar)
- Storage: 100GB free space

**Note**: You can train on CPU with the lightweight config, but it's very slow.

### Software Requirements

- Python 3.8 or newer
- CUDA 11.8+ (for GPU training)
- Git (for downloading models)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd /path/to/project/src/Model

# Install all required packages
pip install -r requirements_llm.txt

# For GPU support (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Collect Training Data

```bash
# Go to Scraper directory
cd ../Scraper

# Run the web scraper
./run_scraper.sh urls.txt

# This will collect forum posts (takes 10-30 minutes)
```

### Step 3: Train Your Model

```bash
# Go back to Model directory
cd ../Model

# Run the automated pipeline
./quick_start.sh

# Or manually:
python pipeline.py --lightweight
```

That's it! Your model will be ready in `models/ecutuning-llm/`

## 📚 Detailed Walkthrough

### Understanding the Pipeline

The training pipeline has 5 main stages:

```
1. Clean Data     → Remove noise, filter relevant content
2. Prepare Data   → Create instruction-response pairs
3. Train Model    → Fine-tune base LLM with LoRA
4. Evaluate       → Test model performance
5. Create Index   → Build vector embeddings for RAG
```

### Stage 1: Data Cleaning

**What it does**: Filters scraped forum posts to keep only ECU tuning content.

```bash
python data_cleaner.py
```

**Input**: Raw forum posts from scraper
**Output**: `data/clean/forum_posts_clean.jsonl`

**Example**:
```
Before: 1,000 forum threads
After: 250 ECU tuning threads (75% filtered out)
```

### Stage 2: Data Preparation

**What it does**: Converts forum posts into training examples.

```bash
python data_preparation.py
```

**Strategies**:
1. Questions from thread titles
2. Technical concept explanations
3. Problem-solution pairs

**Example transformation**:
```
Forum Post:
  Title: "N54 boost drops at 5000 RPM?"
  Content: "My boost drops from 20psi to 15psi at high RPM..."

Training Pair:
  Instruction: "Why does boost drop at high RPM on N54?"
  Response: "Boost drop at high RPM can be caused by..."
```

### Stage 3: Model Training

**What it does**: Fine-tunes a base model on your ECU tuning data.

```bash
# Full model (requires 12GB+ VRAM)
python train_llm.py

# Lightweight (requires 6GB VRAM)
python train_llm.py --lightweight

# Custom
python train_llm.py --model "microsoft/phi-2" --epochs 5
```

**What happens**:
1. Downloads base model (Mistral-7B or Phi-2)
2. Applies 4-bit quantization (saves memory)
3. Adds LoRA adapters (efficient training)
4. Trains for 3-5 epochs (~2-6 hours on RTX 3090)
5. Saves trained model

**Training progress**:
```
Epoch 1/3: 100%|████████| 500/500 [45:23<00:00]
Train Loss: 1.234
Val Loss: 1.456

Epoch 2/3: 100%|████████| 500/500 [45:01<00:00]
Train Loss: 0.876
Val Loss: 1.012
...
```

### Stage 4: Evaluation

**What it does**: Tests your model's performance.

```bash
python evaluate.py --max-samples 100
```

**Metrics**:
- **ROUGE-1/2/L**: How similar responses are to references
- **Latency**: How fast the model generates answers
- **Success rate**: % of questions answered without errors

**Example results**:
```
ROUGE-1:  0.6234  (good overlap with reference answers)
ROUGE-L:  0.5123  (reasonable quality)
Avg latency: 2.3s (fast enough for chat)
Success: 98/100 (98%)
```

### Stage 5: Vector Embeddings

**What it does**: Creates searchable index for RAG (retrieval).

```bash
python embedder.py
```

This allows combining your fine-tuned model with vector search for even better responses.

## 💡 Using Your Trained Model

### Interactive Chat

```bash
python inference.py
```

```
🚗 ECUdapt AI - Your ECU Tuning Assistant

❓ You: How do I reduce turbo lag on N54?

🤖 ECUdapt AI: Turbo lag on the N54 can be reduced through
several methods:

1. Wastegate tuning - Adjust wastegate duty cycle to build
   boost faster
2. Throttle map - Modify throttle response in lower RPM ranges
3. Overboost - Allow temporary higher boost during spool
4. Ignition timing - Advance timing slightly during boost build

The most effective approach is combining wastegate tuning
with throttle map adjustments...
```

### Single Question

```bash
python inference.py --question "What is knock and how to prevent it?"
```

### Python Integration

```python
from pathlib import Path
from inference import ECUTuningLLM

# Load model
llm = ECUTuningLLM(Path("models/ecutuning-llm"))
llm.load_model()

# Generate response
question = "How to tune MAF scaling?"
response = llm.generate(question)
print(response)
```

### With RAG (Retrieval)

```python
from inference import ECUTuningLLM
from query_engine import ForumRetriever

# Load both
llm = ECUTuningLLM(Path("models/ecutuning-llm"))
llm.load_model()
retriever = ForumRetriever()

# Get relevant threads
question = "N54 boost control issues"
relevant_docs = retriever.search(question, k=3)

# Generate with context
context = "\n".join([d["url"] for d in relevant_docs])
response = llm.generate(question, context)
```

## 🎛️ Customization

### Training on Your Own Data

Create `custom_data.jsonl`:
```json
{"instruction": "Your question", "context": "Optional context", "response": "Your answer"}
{"instruction": "Another question", "context": "", "response": "Another answer"}
```

Update config:
```python
# llm_config.py
config.train_data_path = Path("data/training/custom_data.jsonl")
```

### Adjusting Model Behavior

In `llm_config.py`:
```python
# More creative/diverse responses
config.temperature = 0.9
config.top_p = 0.95

# More focused/deterministic responses
config.temperature = 0.3
config.top_p = 0.5

# Reduce repetition
config.repetition_penalty = 1.5
```

### Using Different Base Models

```bash
# Smaller, faster
python train_llm.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Larger, better quality
python train_llm.py --model "meta-llama/Llama-2-13b-hf"

# Code-optimized (if including logs/code)
python train_llm.py --model "codellama/CodeLlama-7b-hf"
```

## 🐛 Common Issues

### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during training

**Solutions**:
```bash
# Use smaller batch size
python train_llm.py --batch-size 1

# Use gradient accumulation
python train_llm.py --batch-size 1 --gradient-accumulation-steps 16

# Use lightweight model
python train_llm.py --lightweight

# Use CPU (very slow)
export CUDA_VISIBLE_DEVICES=""
python train_llm.py --lightweight
```

### 2. Model Not Loading

**Problem**: Trained model can't be loaded

**Check**:
```bash
# Verify model files exist
ls models/ecutuning-llm/

# Should contain:
# - config.json
# - adapter_model.bin (LoRA weights)
# - adapter_config.json
# - tokenizer.json
# - tokenizer_config.json
```

**Fix**: Retrain or check for errors in training logs

### 3. Poor Response Quality

**Problem**: Model gives incorrect or irrelevant answers

**Solutions**:
1. Train longer: `--epochs 5`
2. Use more data: Scrape more forums
3. Use larger model: `--model "mistralai/Mistral-7B-v0.1"`
4. Adjust learning rate: Lower LR = more careful learning
5. Check training data quality: Review `data/training/train.jsonl`

### 4. Slow Generation

**Problem**: Model takes too long to respond

**Solutions**:
1. Reduce max_new_tokens: `--max-new-tokens 256`
2. Use smaller model: Phi-2 instead of Mistral-7B
3. Enable Flash Attention (requires newer GPU)
4. Batch multiple questions together

## 📊 Monitoring & Debugging

### View Training Progress

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in browser

### Check GPU Usage

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

### Inspect Training Data

```bash
# View first 10 training examples
head -10 data/training/train.jsonl | python -m json.tool
```

### Test Model on Specific Questions

```python
test_cases = [
    "What causes knock in turbocharged engines?",
    "How to adjust fuel trims?",
    "Best ignition timing for E85?",
]

for q in test_cases:
    print(f"Q: {q}")
    print(f"A: {llm.generate(q)}\n")
```

## 🎓 Next Steps

1. **Expand Training Data**
   - Add more forums to urls.txt
   - Include manufacturer-specific forums
   - Scrape technical documentation

2. **Improve Model Quality**
   - Experiment with different base models
   - Adjust hyperparameters
   - Add domain-specific evaluation

3. **Deploy as Service**
   - Create FastAPI endpoint
   - Add authentication
   - Deploy to cloud/VPS

4. **Build UI**
   - Web interface with chat
   - Mobile app
   - Discord/Telegram bot

5. **Add Features**
   - Multi-turn conversations
   - Image understanding (dyno charts, logs)
   - Code generation (tuning scripts)

## 📖 Additional Resources

- **Hugging Face Hub**: Browse and download models
- **PEFT Documentation**: Learn about LoRA/QLoRA
- **Transformers Docs**: Model architectures and APIs
- **ECU Tuning Forums**: Source data for training

## 🤝 Contributing

Ideas for improvement:
- Add more scrapers for different forum types
- Optimize inference speed
- Create benchmark datasets
- Improve instruction extraction
- Add multilingual support

## 📄 License

This training code is open source. Base models have their own licenses:
- Mistral-7B: Apache 2.0
- Phi-2: MIT
- Llama 2: Meta Community License (commercial use allowed)

Check model cards on Hugging Face for specific license terms.

---

**Ready to build your own ECU tuning AI?** Start with `./quick_start.sh` 🚀
