# src/Model/agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_config import LLMConfig
from prompts import build_system_prompt, build_user_message
from rag_index import RagIndex
from memory import UserMemory


@dataclass
class ECUDaptAgent:
    cfg: LLMConfig

    def __post_init__(self):
        model_path = self.cfg.output_dir  # use fine-tuned checkpoint
        print(f"[Agent] Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )

        self.rag = RagIndex(self.cfg)
        self.memory = UserMemory(self.cfg)

    # ---------- Prompt construction ----------

    def _format_docs(self, docs: List[Dict]) -> str:
        parts = []
        for d in docs:
            title = d.get("thread_title") or ""
            domain = d.get("domain") or ""
            url = d.get("url") or ""
            snippet = (d.get("content") or "").strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."

            header_bits = []
            if domain:
                header_bits.append(domain)
            if title:
                header_bits.append(title)
            header = " | ".join(header_bits) or "Forum post"

            parts.append(f"[{header}]\n{snippet}\n(Source: {url})")
        return "\n\n".join(parts)

    def _build_messages(self, user_id: str, query: str) -> List[Dict[str, str]]:
        profile = self.memory.get_profile(user_id)
        system_prompt = build_system_prompt(profile)

        # RAG
        docs = self.rag.retrieve(query, k=5)
        docs_text = self._format_docs(docs)

        # Recent convo (optional, kept short)
        history = self.memory.get_recent_history(user_id)

        user_content = build_user_message(
            query=query,
            retrieved_docs=docs_text,
            extra_instructions=(
                "Use the retrieved excerpts as noisy hints. "
                "Correct any misinformation and be conservative about risk."
            ),
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": user_content})
        return messages

    # ---------- Inference ----------

    @torch.no_grad()
    def chat(self, user_id: str, query: str, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(user_id, query)

        # Use model's chat template if present
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: crude concatenation
            prompt_text = ""
            for m in messages:
                role = m["role"]
                content = m["content"]
                prompt_text += f"{role.upper()}: {content}\n"
            prompt_text += "ASSISTANT:"

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_seq_length,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        self.memory.add_turn(user_id, query, text)
        return text


def chat_cli():
    cfg = LLMConfig()
    agent = ECUDaptAgent(cfg)

    user_id = "local-dev"  # in production, use auth user id
    print("ECUDapt Agent CLI. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in {"exit", "quit"}:
            break

        reply = agent.chat(user_id, q)
        print(f"\nECUDapt: {reply}\n")


if __name__ == "__main__":
    chat_cli()
