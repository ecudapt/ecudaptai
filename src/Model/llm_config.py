"""
LLM Configuration for ECUdapt AI
Defines all hyperparameters and paths for training/fine-tuning
"""
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM training and inference"""

    # Model selection
    base_model: str = "mistralai/Mistral-7B-v0.1"  # Base model to fine-tune
    model_type: str = "causal"  # causal or seq2seq

    # Paths
    data_dir: Path = Path("data")
    raw_data_path: Path = Path("data/clean/forum_posts_clean.jsonl")
    train_data_path: Path = Path("data/training/train.jsonl")
    val_data_path: Path = Path("data/training/val.jsonl")
    output_dir: Path = Path("models/ecutuning-llm")
    checkpoint_dir: Path = Path("models/checkpoints")
    logs_dir: Path = Path("logs")

    # Training hyperparameters
    max_seq_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will use default for model type

    # Quantization for memory efficiency
    use_8bit: bool = False
    use_4bit: bool = True  # 4-bit quantization for consumer GPUs

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Data processing
    train_test_split: float = 0.9
    min_text_length: int = 50
    max_text_length: int = 4096

    # System prompts
    system_prompt: str = """You are ECUdapt AI, an expert automotive ECU tuning assistant.
You provide accurate, technical guidance on engine tuning, ECU remapping, and performance optimization.
Base your responses on real forum discussions and proven tuning practices."""

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "training").mkdir(parents=True, exist_ok=True)


# Alternative lightweight config for CPU/small GPU training
@dataclass
class LightweightConfig(LLMConfig):
    """Lightweight configuration for resource-constrained environments"""
    base_model: str = "microsoft/phi-2"  # Smaller 2.7B model
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_4bit: bool = True
    num_epochs: int = 5


# For instruction fine-tuning
@dataclass
class InstructionTuningConfig(LLMConfig):
    """Configuration for instruction-based fine-tuning"""
    instruction_template: str = """### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""

    instruction_key: str = "instruction"
    context_key: str = "context"
    response_key: str = "response"
