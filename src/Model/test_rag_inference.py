#!/usr/bin/env python3
"""
Quick manual test for ECUDapt SFT + RAG.

Usage:
    cd src/Model
    python test_rag_inference.py
"""

from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_config import LLMConfig
from rag_index import RagIndex


def load_model_and_tokenizer(cfg: LLMConfig):
    """
    Load your fine-tuned SFT model from cfg.sft_model_dir.
    Assumes HuggingFace-style save from Trainer / PEFT merge.
    """
    model_path = cfg.sft_model_dir
    print(f"[MODEL] Loading model from: {model_path}")

    # A100 likes bfloat16; fall back to float16 if needed
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(query: str, docs: List[Dict]) -> str:
    """
    Build a chat-style prompt for Llama 3.1 using its chat template.
    We stuff the retrieved context + the user question.
    """
    # Build context block from RAG docs
    context_lines = []
    for i, d in enumerate(docs, start=1):
        title = d.get("thread_title") or d.get("title") or ""
        url = d.get("url") or ""
        snippet = d.get("content")[:500].replace("\n", " ")
        context_lines.append(
            f"[{i}] {title}\nURL: {url}\nText: {snippet}\n"
        )
    context_text = "\n\n".join(context_lines) if context_lines else "No context retrieved."

    system_prompt = (
        "You are ECUDapt, an expert automotive ECU tuning and diagnostics assistant. "
        "You specialize in realistic, safe street builds and interpreting forum knowledge. "
        "Use the retrieved forum context when it is relevant, but DO NOT just copy it. "
        "Explain your reasoning clearly and conservatively. If something is unsafe, say so."
    )

    user_content = (
        f"CONTEXT FROM FORUM THREADS:\n{context_text}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Using the context above and your own knowledge, give a detailed but concise answer. "
        "If you reference something from a specific thread, mention it like: (source: [1])"
    )

    # We’ll let the tokenizer's chat template wrap this correctly
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return messages


def generate_answer(model, tokenizer, messages, max_new_tokens: int = 512) -> str:
    # Turn messages into a chat-formatted prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
        )

    # Only decode the new tokens after the prompt
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def main():
    cfg = LLMConfig()

    # ---- Load RAG index ----
    print(f"[RAG] Loading index from {cfg.rag_index_dir} ...")
    rag = RagIndex(cfg)
    rag.load()

    # ---- Load model ----
    model, tokenizer = load_model_and_tokenizer(cfg)

    print("\n✅ ECUDapt SFT + RAG loaded.")
    print("Type a question about tuning/diagnostics. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Goodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("[EXIT] Goodbye.")
            break

        # ---- Retrieve context ----
        docs = rag.retrieve(query, k=5)
        print(f"[RAG] Retrieved {len(docs)} docs.")

        # ---- Build prompt & generate ----
        messages = build_prompt(query, docs)
        answer = generate_answer(model, tokenizer, messages)

        print("\nECUDapt:")
        print(answer)

        # ---- Show sources ----
        print("\n----- SOURCES -----")
        for i, d in enumerate(docs, start=1):
            title = d.get("thread_title") or d.get("title") or ""
            url = d.get("url") or ""
            print(f"[{i}] {title}  ({url})")
        print("-------------------\n")


if __name__ == "__main__":
    main()
