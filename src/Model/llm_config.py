from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    # ---- Model + training ----
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_name: Optional[str] = None  # defaults to base_model_name
    max_seq_length: int = 2048

    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.5
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # ---- Paths ----
    # "project_root" here is actually the Model folder:
    # C:/Users/jonat/Desktop/ecudapt/ecudapt-ai/src/Model
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    data_dir: Path = field(init=False)
    raw_forum_file: Path = field(init=False)
    tagged_forum_file: Path = field(init=False)
    sft_train_file: Path = field(init=False)
    sft_eval_file: Path = field(init=False)
    rag_index_dir: Path = field(init=False)
    memory_file: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        # data directory under Model:
        # src/Model/data/...
        self.data_dir = self.project_root / "data"

        # cleaned forum data
        self.raw_forum_file = self.data_dir / "clean" / "forum_posts_clean.jsonl"
        self.tagged_forum_file = self.data_dir / "clean" / "forum_posts_tagged.jsonl"

        # SFT data
        self.sft_train_file = self.data_dir / "train.jsonl"
        self.sft_eval_file = self.data_dir / "eval.jsonl"

        # RAG index + memory
        self.rag_index_dir = self.data_dir / "rag_index"
        self.memory_file = self.data_dir / "memory.json"

        # fine-tuned model output
        # keep checkpoints at repo root level if you want, or inside Model
        self.output_dir = self.project_root / "checkpoints"

        # ---- Create directories ----
        (self.data_dir / "clean").mkdir(parents=True, exist_ok=True)
        self.rag_index_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
