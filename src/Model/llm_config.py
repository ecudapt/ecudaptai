# src/Model/llm_config.py
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """
    Configuration for LLM training and inference (v2).
    Defaults target an 8B instruct model with LoRA/QLoRA on an A100/L40S.
    """

    # Stronger base for reasoning
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_type: str = "causal"

    # Paths
    data_dir: Path = Path("data")
    raw_data_path: Path = Path("data/clean/forum_posts_clean.jsonl")
    train_data_path: Path = Path("data/training/train.jsonl")
    val_data_path: Path = Path("data/training/val.jsonl")
    output_dir: Path = Path("models/ecutuning-llm-v2")
    checkpoint_dir: Path = Path("models/checkpoints-v2")
    logs_dir: Path = Path("logs")

    # Training hyperparams
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.5e-5
    num_epochs: int = 2
    warmup_steps: int = 0
    weight_decay: float = 0.01

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # defaulted in code to attn proj layers

    # Quantization
    use_8bit: bool = False
    use_4bit: bool = True  # QLoRA

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.35
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15

    # Data
    train_test_split: float = 0.9
    min_text_length: int = 50
    max_text_length: int = 4096

    # System prompt guiding *decisions*, not quotes
    system_prompt: str = (
        "You are ECUdapt AI, an expert automotive ECU tuning assistant. "
        "Provide concise, original guidance with clear decisions and short reasoning. "
        "Do not quote forum posts verbatim; summarize in your own words. "
        "Always include safety constraints when proposing changes."
    )

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "training").mkdir(parents=True, exist_ok=True)


@dataclass
class LightweightConfig(LLMConfig):
    """CPU or small-GPU fallback."""
    base_model: str = "microsoft/phi-2"
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
