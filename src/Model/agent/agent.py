# src/Model/agent/agent.py
from __future__ import annotations

from typing import Dict, Any, Optional, List

from ..llm_config import LLMConfig
from ..rag_index import RagIndex
from .memory import SessionMemory
from .prompts import build_agent_prompt


# In-memory store of session_id -> SessionMemory
_SESSION_MEMORY: Dict[str, SessionMemory] = {}

# Lazy-loaded global RAG index
_RAG_INDEX: Optional[RagIndex] = None


def get_session_memory(session_id: str) -> SessionMemory:
    """
    Get or create a SessionMemory object for this session_id.
    In production, you might back this with Supabase/Postgres.
    """
    mem = _SESSION_MEMORY.get(session_id)
    if mem is None:
        mem = SessionMemory()
        _SESSION_MEMORY[session_id] = mem
    return mem


def get_rag_index() -> RagIndex:
    """
    Lazy load a global RagIndex.
    Assumes LLMConfig.rag_index_dir and raw_forum_file are configured.
    """
    global _RAG_INDEX
    if _RAG_INDEX is None:
        cfg = LLMConfig()
        _RAG_INDEX = RagIndex(cfg)
        # We don't need to call build_from_jsonl here; it should already be built.
        # retrieve() will auto-call load() when needed.
    return _RAG_INDEX


def generate_model_reply(prompt: str) -> str:
    """
    Placeholder: this is where you'll plug in your trained LLaMA SFT model.

    Later you'll do something like:
        model, tokenizer = load_sft_model(...)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(...)

    For now, this is a stub so the agent pipeline runs end-to-end.
    """
    return (
        "[MODEL STUB] This is where the fine-tuned ECUDapt LLaMA model will respond.\n\n"
        "Prompt I received was:\n"
        "----------------------\n"
        f"{prompt}\n"
        "----------------------\n"
        "Once the model is wired in, this stub will be replaced."
    )


def agent_reply(
    session_id: str,
    user_message: str,
    domains: Optional[List[str]] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    High-level ECUDapt agent interface.

    - Updates the session memory with new info from the user.
    - Uses RAG to retrieve relevant forum snippets.
    - Builds a persona + memory + RAG-augmented prompt.
    - Calls the (currently stubbed) model reply function.
    - Returns a structured response that an API/CLI can consume.
    """
    # 1) Load/update memory
    mem = get_session_memory(session_id)
    mem.update_from_message(user_message)
    memory_summary = mem.to_summary()

    # 2) Retrieve relevant docs from RAG
    rag = get_rag_index()
    # RagIndex.retrieve currently signature: retrieve(query, k, domains=None)
    retrieved_docs = rag.retrieve(user_message, k=k) if domains is None else rag.retrieve(user_message, k=k, domains=domains)

    # 3) Build final prompt
    prompt = build_agent_prompt(memory_summary, user_message, retrieved_docs)

    # 4) Get model answer (stub for now)
    answer = generate_model_reply(prompt)

    return {
        "session_id": session_id,
        "answer": answer,
        "debug": {
            "memory_summary": memory_summary,
            "retrieved_docs": retrieved_docs,
            "used_domains_filter": domains or [],
        },
    }
