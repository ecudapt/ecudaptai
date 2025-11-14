# ECUDapt LLaMA SFT Run – 2025-11-13

- Commit: <git commit hash>
- Base model: meta-llama/Llama-3.1-8B-Instruct
- Training script: train_sft_v2.py
- Config (llm_config.py):
  - max_seq_length: 2048
  - train_batch_size: ...
  - gradient_accumulation_steps: ...
  - num_train_epochs: ...
  - learning_rate: ...
- Dataset:
  - train.jsonl size: 15060 examples
  - eval.jsonl size: 1673 examples
  - Source: forum_posts_clean → tagged → data_preparation.py
- Notes:
  - First serious ECU SFT run on A100.
  - Gradient checkpointing: ON
  - QLoRA: yes (trainable params ~42M, ~0.52%)
