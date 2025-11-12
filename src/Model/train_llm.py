#!/usr/bin/env python3
"""
LLM Training Script with LoRA/QLoRA
Fine-tune a base model on ECU tuning forum data
"""
import os
import sys
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    import bitsandbytes as bnb
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Install with: pip install transformers peft datasets bitsandbytes accelerate")
    sys.exit(1)

from llm_config import LLMConfig, LightweightConfig


class ECUTuningTrainer:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🖥️  Using device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def load_model_and_tokenizer(self):
        """Load base model with quantization if enabled"""
        print(f"\n📦 Loading base model: {self.config.base_model}")

        # Configure quantization
        bnb_config = None
        if self.config.use_4bit:
            print("   Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            print("   Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        print("   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print("   Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Prepare for training if using quantization
        if bnb_config is not None:
            print("   Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)

        print("✅ Model and tokenizer loaded")

    def apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        if not self.config.use_lora:
            return

        print("\n🔧 Applying LoRA adapters...")

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("✅ LoRA adapters applied")

    def load_training_data(self):
        """Load and tokenize training data"""
        print(f"\n📚 Loading training data...")

        if not self.config.train_data_path.exists():
            print(f"❌ Training data not found: {self.config.train_data_path}")
            print("   Run data_preparation.py first")
            sys.exit(1)

        # Load datasets
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(self.config.train_data_path),
                "validation": str(self.config.val_data_path),
            }
        )

        print(f"   Training samples: {len(dataset['train'])}")
        print(f"   Validation samples: {len(dataset['validation'])}")

        # Tokenize datasets
        print("   Tokenizing data...")

        def format_instruction(example):
            """Format instruction-context-response into prompt"""
            prompt = f"""{self.config.system_prompt}

### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
            return {"text": prompt}

        def tokenize_function(examples):
            """Tokenize texts"""
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        # Format and tokenize
        dataset = dataset.map(format_instruction)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        print("✅ Data loaded and tokenized")
        return tokenized_dataset

    def train(self):
        """Main training loop"""
        print("\n🚀 Starting LLM training...")

        # Setup training arguments
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
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            fp16=self.device == "cuda",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if self.config.use_4bit else "adamw_torch",
        )

        # Load datasets
        tokenized_dataset = self.load_training_data()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )

        # Train!
        print("\n🔥 Training started...")
        print(f"   Model: {self.config.base_model}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"   Learning rate: {self.config.learning_rate}")

        trainer.train()

        print("\n✅ Training complete!")

        # Save final model
        print(f"\n💾 Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        print("\n🎉 All done! Your ECU tuning LLM is ready!")
        print(f"   Model saved at: {self.config.output_dir}")
        print(f"   Logs saved at: {self.config.logs_dir}")

    def run(self):
        """Run complete training pipeline"""
        try:
            self.load_model_and_tokenizer()
            self.apply_lora()
            self.train()
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train ECUdapt LLM")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight config for limited resources"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override base model (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size per device"
    )

    args = parser.parse_args()

    # Select config
    if args.lightweight:
        print("📦 Using lightweight configuration")
        config = LightweightConfig()
    else:
        config = LLMConfig()

    # Override settings
    if args.model:
        config.base_model = args.model
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Run training
    trainer = ECUTuningTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
