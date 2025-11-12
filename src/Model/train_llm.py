#!/usr/bin/env python3
"""
LLM Training Script with LoRA/QLoRA
Fine-tune a base model on ECU tuning forum data
(Updated: trains ONLY on the Response via label masking)
"""
import os, sys, json, torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from time import time
from math import ceil
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    import bitsandbytes as bnb
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("        Install with: pip install transformers peft datasets bitsandbytes accelerate")
    sys.exit(1)

from llm_config import LLMConfig, LightweightConfig

# --------------------------- Progress Callback ---------------------------

class _Throughput:
    def __init__(self):
        self.last_time = None
        self.last_step = 0
        self.tok_per_sec = 0.0

class ProgressCallback(TrainerCallback):
    """
    ASCII-only live progress for Windows consoles.
    Shows step/epoch, loss, lr, tokens/sec, ETA; also logs to a file.
    """
    def __init__(self, total_steps: int, tokens_per_step: int, log_path: Path):
        self.total_steps = int(total_steps)
        self.tokens_per_step = int(tokens_per_step)
        self.tp = _Throughput()
        self.pbar = None
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, line: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.pbar = tqdm(total=self.total_steps, desc="TRAIN", leave=True, ncols=100)
        self.tp.last_time = time()
        self._log("[START] Training")

    def on_log(self, args, state, control, **kwargs):
        if not state.log_history:
            return
        rec = state.log_history[-1]
        step = state.global_step

        now = time()
        dt = max(1e-9, now - self.tp.last_time)
        dstep = max(0, step - self.tp.last_step)
        if dstep > 0:
            self.tp.tok_per_sec = (dstep * self.tokens_per_step) / dt
            self.tp.last_time = now
            self.tp.last_step = step

        loss = rec.get("loss", rec.get("train_loss"))
        lr = rec.get("learning_rate")

        remaining = max(0, self.total_steps - step)
        tps = max(1e-9, self.tp.tok_per_sec)
        eta_s = remaining * self.tokens_per_step / tps
        eta_min = int(eta_s // 60); eta_sec = int(eta_s % 60)

        self.pbar.n = min(step, self.total_steps)
        status = f"step={step}/{self.total_steps}"
        if loss is not None: status += f"  loss={loss:.4f}"
        if lr is not None:   status += f"  lr={lr:.2e}"
        status += f"  tok/s={self.tp.tok_per_sec:,.0f}  ETA={eta_min:02d}:{eta_sec:02d}"
        self.pbar.set_postfix_str(status[:60])
        self.pbar.refresh()
        self._log(f"[LOG] {status}")

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = state.epoch if state.epoch is not None else 0
        self._log(f"[EPOCH] {ep:.2f} finished")

    def on_save(self, args, state, control, **kwargs):
        self._log(f"[SAVE] checkpoint at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.n = min(self.total_steps, state.global_step)
        self.pbar.close()
        self._log("[DONE] Training finished")

# --------------------------- Trainer Wrapper -----------------------------

class ECUTuningTrainer:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Using device: {self.device}")
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def load_model_and_tokenizer(self):
        print(f"\n[LOAD] Base model: {self.config.base_model}")
        bnb_config = None
        if self.config.use_4bit:
            print("[LOAD] Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            print("[LOAD] Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print("[LOAD] Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[LOAD] Model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        if bnb_config is not None:
            print("[LOAD] Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)

        print("[OK] Model and tokenizer loaded")

    def apply_lora(self):
        if not self.config.use_lora:
            return
        print("\n[SETUP] Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("[OK] LoRA adapters applied")

    # ---------- NEW: label-masked dataset pipeline ----------
    def _format_example(self, ex: dict) -> dict:
        prompt = f"""{self.config.system_prompt}

### Instruction:
{ex['instruction']}

### Context:
{ex.get('context','')}

### Response:
"""
        target = ex["response"].strip() + (self.tokenizer.eos_token or "</s>")
        return {"prompt": prompt, "target": target}

    def _tokenize_mask_supervised(self, examples):
        prompts = examples["prompt"]
        targets = examples["target"]

        prompt_tok = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            add_special_tokens=True,
        )
        with self.tokenizer.as_target_tokenizer():
            target_tok = self.tokenizer(
                targets,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                add_special_tokens=False,
            )

        input_ids, attn, labels = [], [], []
        for pi, ti in zip(prompt_tok["input_ids"], target_tok["input_ids"]):
            ids = pi + ti
            am  = [1]*len(ids)
            lb  = [-100]*len(pi) + ti  # <-- mask prompt tokens

            # clip to max length
            ids  = ids[:self.config.max_seq_length]
            am   = am[:self.config.max_seq_length]
            lb   = lb[:self.config.max_seq_length]

            input_ids.append(ids)
            attn.append(am)
            labels.append(lb)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    def load_training_data(self):
        print(f"\n[DATA] Loading training data...")
        if not self.config.train_data_path.exists():
            print(f"[ERROR] Training data not found: {self.config.train_data_path}")
            print("        Run data_preparation.py first")
            sys.exit(1)

        dataset = load_dataset(
            "json",
            data_files={
                "train": str(self.config.train_data_path),
                "validation": str(self.config.val_data_path),
            }
        )
        print(f"[DATA] Training samples:   {len(dataset['train'])}")
        print(f"[DATA] Validation samples: {len(dataset['validation'])}")

        print("[DATA] Formatting + tokenizing with label masking...")
        dataset = dataset.map(self._format_example)
        tokenized = dataset.map(
            self._tokenize_mask_supervised,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        print("[OK] Data loaded and tokenized")
        return tokenized

    def _estimate_steps_and_tokens(self, args: TrainingArguments, train_dataset_len: int) -> tuple[int, int]:
        world_size = getattr(args, "world_size", 1)
        micro_bs = args.per_device_train_batch_size
        grad_acc = args.gradient_accumulation_steps
        steps_per_epoch = ceil(train_dataset_len / (micro_bs * max(1, world_size)))
        total_steps = steps_per_epoch * int(args.num_train_epochs)
        if grad_acc > 1:
            total_steps = ceil(total_steps / grad_acc)
        avg_seq_len = self.config.max_seq_length
        global_batch = micro_bs * max(1, world_size)
        tokens_per_step = avg_seq_len * global_batch
        return total_steps, tokens_per_step

    def train(self):
        print("\n[TRAIN] Starting LLM training...")
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=str(self.config.logs_dir),
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to=[],  # set to ["tensorboard"] if you want TB
            fp16=(self.device == "cuda"),
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if self.config.use_4bit else "adamw_torch",
        )

        tokenized = self.load_training_data()

        # We already masked labels; use a simple collator that does NOT add MLM.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
        )

        total_steps, tokens_per_step = self._estimate_steps_and_tokens(
            training_args, len(tokenized["train"])
        )
        log_path = self.config.logs_dir / "training_progress.log"
        trainer.add_callback(ProgressCallback(total_steps, tokens_per_step, log_path))
        print(f"[INFO] Progress log -> {log_path}")

        print("\n[TRAIN] Configuration:")
        print(f"  Model:                {self.config.base_model}")
        print(f"  Epochs:               {self.config.num_epochs}")
        print(f"  Batch size/device:    {self.config.batch_size}")
        print(f"  Grad accumulation:    {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Learning rate:        {self.config.learning_rate}")

        trainer.train()

        print("\n[OK] Training complete")
        print(f"\n[SAVE] Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        print("\n[DONE] Model + tokenizer saved")

    def run(self):
        try:
            self.load_model_and_tokenizer()
            self.apply_lora()
            self.train()
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ECUdapt LLM")
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    if args.lightweight:
        print("[CONFIG] Using lightweight configuration")
        config = LightweightConfig()
    else:
        config = LLMConfig()

    if args.model:  config.base_model = args.model
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size

    ECUTuningTrainer(config).run()

if __name__ == "__main__":
    main()
