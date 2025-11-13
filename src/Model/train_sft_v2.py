#!/usr/bin/env python3
"""
Supervised fine-tuning with LoRA/QLoRA on your ECU forum SFT dataset.

Reads:
  - src/Model/data/train.jsonl
  - src/Model/data/eval.jsonl

Writes fine-tuned adapter + tokenizer to:
  - src/Model/checkpoints/
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from llm_config import LLMConfig


def format_example(example):
    """
    Convert {instruction, input, output} into a chat-style training text.

    This works well for LLaMA/Qwen instruct models that support the chat template.
    """
    instruction = example["instruction"]
    inp = example.get("input") or ""
    output = example["output"]

    # Fallback formatting; we'll mostly rely on chat_template at generation time,
    # but this is fine for SFT.
    if inp:
        text = f"<s>[INST] {instruction}\n\n{inp} [/INST]\n{output}</s>"
    else:
        text = f"<s>[INST] {instruction} [/INST]\n{output}</s>"

    return {"text": text}


def tokenize_example(example, tokenizer, max_length):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
    )
    # causal LM: labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()  # reserved for later if we want CLI overrides

    cfg = LLMConfig()

    # ---- Dataset ----
    data_files = {
        "train": str(cfg.sft_train_file),
        "eval": str(cfg.sft_eval_file),
    }

    if not Path(cfg.sft_train_file).exists():
        raise FileNotFoundError(f"Train file not found: {cfg.sft_train_file}")
    if not Path(cfg.sft_eval_file).exists():
        raise FileNotFoundError(f"Eval file not found: {cfg.sft_eval_file}")

    raw_ds = load_dataset("json", data_files=data_files)

    formatted = raw_ds.map(format_example)

    tokenizer_name = cfg.tokenizer_name or cfg.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = formatted.map(
        lambda ex: tokenize_example(ex, tokenizer, cfg.max_seq_length),
        remove_columns=formatted["train"].column_names,
    )

    # ---- Model ----
    print(f"[Train] Loading base model: {cfg.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        device_map="auto",
        load_in_4bit=cfg.use_qlora,  # QLoRA: 4-bit base weights
    )

    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # For LLaMA/Qwen, typical target modules:
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training ----
        training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=20,
        fp16=True,
        optim="paged_adamw_8bit" if cfg.use_qlora else "adamw_torch",
        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        tokenizer=tokenizer,
    )

    print("[Train] Starting training...")
    trainer.train()

    print("[Train] Saving model + tokenizer...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[Train] Done. Checkpoints in {cfg.output_dir}")


if __name__ == "__main__":
    main()
