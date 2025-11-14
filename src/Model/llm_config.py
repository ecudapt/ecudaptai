from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    # -----------------------------
    # Model + Training Parameters
    # -----------------------------
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_name: Optional[str] = None
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

    # -----------------------------
    # Path Structure
    # -----------------------------
    #
    # project_root = src/Model/
    #
    # data/
    #   clean/
    #       forum_posts_clean.jsonl
    #       forum_posts_tagged.jsonl
    #   rag_index/
    #       faiss_index.bin
    #       docs_meta.json
    #       embeddings.npy
    #   train.jsonl
    #   eval.jsonl
    #
    # checkpoints/
    #
    # -----------------------------
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    # These get set in __post_init__
    data_dir: Path = field(init=False)
    clean_dir: Path = field(init=False)

    raw_forum_file: Path = field(init=False)
    tagged_forum_file: Path = field(init=False)
    rag_source_file: Path = field(init=False)

    sft_train_file: Path = field(init=False)
    sft_eval_file: Path = field(init=False)

    rag_index_dir: Path = field(init=False)
    memory_file: Path = field(init=False)

    output_dir: Path = field(init=False)
    sft_model_dir: Path = field(init=False)

    # ----------------------------------------------------
    # Init: Build unified directory + file paths
    # ----------------------------------------------------
    def __post_init__(self):

        # /data folder under Model
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # /data/clean folder
        self.clean_dir = self.data_dir / "clean"
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        # Forum files
        self.raw_forum_file = self.clean_dir / "forum_posts_clean.jsonl"
        self.tagged_forum_file = self.clean_dir / "forum_posts_tagged.jsonl"

        # RAG should try tagged first (best)
        self.rag_source_file = self.tagged_forum_file

        # SFT data
        self.sft_train_file = self.data_dir / "train.jsonl"
        self.sft_eval_file = self.data_dir / "eval.jsonl"

        # RAG index output
        self.rag_index_dir = self.data_dir / "rag_index"
        self.rag_index_dir.mkdir(parents=True, exist_ok=True)

        # memory store (for agent memory, not training)
        self.memory_file = self.data_dir / "memory.json"

        # Model output
        self.output_dir = self.project_root / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # final model folder used at inference time
        self.sft_model_dir = self.output_dir

    # ----------------------------------------------------
    # Debug helper — prints paths cleanly
    # ----------------------------------------------------
    def print_config(self):
        print("\n=== LLM CONFIG PATHS ===")
        print(f"project_root      = {self.project_root}")
        print(f"data_dir          = {self.data_dir}")
        print(f"clean_dir         = {self.clean_dir}")
        print(f"raw_forum_file    = {self.raw_forum_file}")
        print(f"tagged_forum_file = {self.tagged_forum_file}")
        print(f"sft_train_file    = {self.sft_train_file}")
        print(f"sft_eval_file     = {self.sft_eval_file}")
        print(f"rag_index_dir     = {self.rag_index_dir}")
        print(f"rag_source_file   = {self.rag_source_file}")
        print(f"memory_file       = {self.memory_file}")
        print(f"output_dir        = {self.output_dir}")
        print(f"sft_model_dir     = {self.sft_model_dir}")
        print("========================\n")
