from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    # ---- Model + training ----
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
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
    # .../ecudapt-ai/src/Model
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    data_dir: Path = field(init=False)
    raw_forum_file: Path = field(init=False)
    tagged_forum_file: Path = field(init=False)

    # New: which file RAG should read from (usually the tagged forum file)
    rag_source_file: Path = field(init=False)

    sft_train_file: Path = field(init=False)
    sft_eval_file: Path = field(init=False)
    rag_index_dir: Path = field(init=False)
    memory_file: Path = field(init=False)
    output_dir: Path = field(init=False)

    # New: where the final SFT model lives (used at inference time)
    sft_model_dir: Path = field(init=False)

    def __post_init__(self):
        # data directory under Model:
        # src/Model/data/...
        self.data_dir = self.project_root / "data"

        clean_dir = self.data_dir / "clean"

        # cleaned forum data
        self.raw_forum_file = clean_dir / "forum_posts_clean.jsonl"
        self.tagged_forum_file = clean_dir / "forum_posts_tagged.jsonl"

        # RAG should usually use the richer tagged file
        self.rag_source_file = self.tagged_forum_file

        # SFT data
        self.sft_train_file = self.data_dir / "train.jsonl"
        self.sft_eval_file = self.data_dir / "eval.jsonl"

        # RAG index + memory
        self.rag_index_dir = self.data_dir / "rag_index"
        self.memory_file = self.data_dir / "memory.json"

        # fine-tuned model output
        # Trainer will write checkpoints here.
        self.output_dir = self.project_root / "checkpoints"

        # For now, assume we load the model directly from output_dir
        # (later you can point this at a specific subfolder like checkpoints/final)
        self.sft_model_dir = self.output_dir

        # ---- Create directories ----
        clean_dir.mkdir(parents=True, exist_ok=True)
        self.rag_index_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
