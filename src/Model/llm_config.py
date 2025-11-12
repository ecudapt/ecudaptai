"""
LLM Configuration for ECUdapt AI
Defines all hyperparameters and paths for training/fine-tuning
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM training and inference"""

    # ------------------------------------------------------------------
    # Model selection
    # For 8 GB GPUs, prefer smaller bases (e.g., microsoft/phi-2) or use QLoRA.
    # You can override this on the CLI (e.g., --model mistralai/Mistral-7B-v0.1)
    # ------------------------------------------------------------------
    base_model: str = "microsoft/phi-2"    # safer default for 8 GB cards
    model_type: str = "causal"             # "causal" or "seq2seq"

    # ------------------------------------------------------------------
    # Paths (resolved relative to this file for robustness)
    # ------------------------------------------------------------------
    _here: Path = field(default_factory=lambda: Path(__file__).resolve().parent, init=False, repr=False)
    _root: Path = field(init=False, repr=False)

    data_dir: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    train_data_path: Path = field(init=False)
    val_data_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # ------------------------------------------------------------------
    # Training hyperparameters
    # For 8 GB GPUs, start with seq_len=512, batch_size=1, high grad_accum.
    # ------------------------------------------------------------------
    max_seq_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # ------------------------------------------------------------------
    # LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
    # Target attention proj layers only to save VRAM on 8 GB GPUs.
    # ------------------------------------------------------------------
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # ------------------------------------------------------------------
    # Quantization for memory efficiency
    # QLoRA (4-bit) is ideal on consumer GPUs; auto-disabled in code if no CUDA.
    # ------------------------------------------------------------------
    use_8bit: bool = False
    use_4bit: bool = True

    # ------------------------------------------------------------------
    # Generation parameters (used by inference/eval)
    # ------------------------------------------------------------------
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # ------------------------------------------------------------------
    # Data processing
    # ------------------------------------------------------------------
    train_test_split: float = 0.9
    min_text_length: int = 50
    max_text_length: int = 4096

    # ------------------------------------------------------------------
    # System prompt (ASCII only for Windows consoles)
    # ------------------------------------------------------------------
    system_prompt: str = (
        "You are ECUdapt AI, an expert automotive ECU tuning assistant.\n"
        "You provide accurate, technical guidance on engine tuning, ECU remapping, and performance optimization.\n"
        "Base your responses on real forum discussions and proven tuning practices."
    )

    def __post_init__(self):
        # Establish project root as parent of this file's directory (i.e., .../src/Model -> .../src)
        object.__setattr__(self, "_root", self._here.parent)

        # Build stable, absolute paths relative to the repository layout
        data_dir = self._here / "data"
        object.__setattr__(self, "data_dir", data_dir)

        object.__setattr__(self, "raw_data_path", data_dir / "clean" / "forum_posts_clean.jsonl")
        object.__setattr__(self, "train_data_path", data_dir / "training" / "train.jsonl")
        object.__setattr__(self, "val_data_path", data_dir / "training" / "val.jsonl")
        object.__setattr__(self, "output_dir", self._here / "models" / "ecutuning-llm")
        object.__setattr__(self, "checkpoint_dir", self._here / "models" / "checkpoints")
        object.__setattr__(self, "logs_dir", self._here / "logs")

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "training").mkdir(parents=True, exist_ok=True)


# Lightweight config for CPU/smaller GPUs (kept for compatibility/CLI flag)
@dataclass
class LightweightConfig(LLMConfig):
    """Lightweight configuration for resource-constrained environments"""
    base_model: str = "microsoft/phi-2"
    max_seq_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    use_4bit: bool = True
    num_epochs: int = 5


# Instruction-tuning prompt template (used by data_preparation/inference)
@dataclass
class InstructionTuningConfig(LLMConfig):
    """Configuration for instruction-based fine-tuning"""
    instruction_template: str = (
        "### Instruction:\n"
        "{instruction}\n\n"
        "### Context:\n"
        "{context}\n\n"
        "### Response:\n"
        "{response}"
    )
    instruction_key: str = "instruction"
    context_key: str = "context"
    response_key: str = "response"
