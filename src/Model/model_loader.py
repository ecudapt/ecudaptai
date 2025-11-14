# src/Model/model_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from llm_config import LLMConfig

# Cached globals so we only load once
_TOKENIZER = None
_MODEL = None


def load_sft_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load the base LLaMA 3.1 8B model in 4-bit + your LoRA adapter
    from cfg.sft_model_dir (usually src/Model/checkpoints).

    This assumes:
    - Training used QLoRA/PEFT (which you did).
    - Trainer saved the adapter weights into cfg.output_dir.
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    cfg = LLMConfig()
    model_dir: Path = cfg.sft_model_dir

    if not model_dir.exists():
        raise FileNotFoundError(
            f"SFT model directory not found: {model_dir} "
            "(did training finish and save a checkpoint here?)"
        )

    base_name = cfg.base_model_name
    tok_name = cfg.tokenizer_name or base_name

    print(f"[ModelLoader] Loading tokenizer: {tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[ModelLoader] Loading base model in 4-bit: {base_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=False,
    )

    print(f"[ModelLoader] Attaching LoRA adapter from: {model_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        is_trainable=False,
    )
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    return tokenizer, model


def generate_completion(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Convenience helper: run a single prompt through the SFT model.
    Used by the Agent.
    """
    cfg = LLMConfig()
    tokenizer, model = load_sft_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text
