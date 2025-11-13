# src/Model/dpo_train.py
#!/usr/bin/env python3
"""
DPO preference optimization on top of SFT output.
Dataset format (JSONL):
  {"prompt": "<full prompt with Instruction/Context/Response:>",
   "chosen": "Decision: ...\nWhy: ...\nSafety: ...",
   "rejected": "Vague/unsafe/rambling alt"}
"""
import sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from llm_config import LLMConfig

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with fields: prompt, chosen, rejected")
    ap.add_argument("--out", default="models/ecutuning-llm-v2-dpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    cfg = LLMConfig()                    # loads SFT output dir for tokenizer compatibility
    base = cfg.output_dir                # start from SFT checkpoint

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={"train": args.data})

    cfg_dpo = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=args.epochs,
        bf16=True, fp16=False, tf32=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        max_grad_norm=1.0,
        logging_steps=20, save_steps=200, save_total_limit=3,
        report_to=[],
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model, args=cfg_dpo, tokenizer=tok,
        train_dataset=ds["train"], max_length=cfg.max_seq_length, max_target_length=512
    )
    trainer.train()
    trainer.save_model(args.out); tok.save_pretrained(args.out)
    print(f"[OK] DPO done -> {args.out}")

if __name__ == "__main__":
    main()
