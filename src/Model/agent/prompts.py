# src/Model/agent/prompts.py
from __future__ import annotations

from typing import List, Dict, Any

SYSTEM_PROMPT = """
You are ECUDapt, an ECU tuning assistant.

Core principles:
- Safety first: you never give reckless advice or extreme values.
- Data-driven: you ask for logs, hardware details, and fuel quality before specific tuning changes.
- Conservative: you favor reliability over maximum power.
- Anti-bro-science: you politely reject myths and unsafe shortcuts.
- Transparent: you explain your reasoning step by step.
- Boundaries: you do NOT output raw timing/boost/fuel tables or exact map values. Those are produced by a separate numeric tune model/tool.
- You encourage the user to verify with datalogs, wideband, and knock detection, and to consult a professional where appropriate.

When unsure, you say you are unsure and suggest safe next steps rather than guessing.
""".strip()


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved RAG docs into a compact text block for the prompt.
    We only include light metadata + a short snippet of the 'content'.
    """
    if not docs:
        return "No external forum snippets were retrieved for this question."

    lines: List[str] = ["Here are some ECU-tuning related forum snippets that may be relevant:"]

    for i, doc in enumerate(docs, start=1):
        title = doc.get("thread_title") or "Unknown thread"
        platform = doc.get("platform") or "Unknown platform"
        domains = doc.get("domains") or []
        url = doc.get("url") or ""
        content = doc.get("content") or ""

        # Snip content so prompt doesn't blow up
        snippet = content[:600].replace("\n", " ")
        if len(content) > 600:
            snippet += " ..."

        lines.append(
            f"\n[{i}] {title} ({platform})\n"
            f"Domains: {', '.join(domains) if domains else 'none'}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}"
        )

    return "\n".join(lines)


def build_agent_prompt(
    memory_summary: str,
    user_message: str,
    retrieved_docs: List[Dict[str, Any]],
) -> str:
    """
    Build a single text prompt for the SFT LLaMA model.

    If you later want to use a chat template, you can adapt this to return
    a list of messages instead of a single string.
    """
    docs_block = format_retrieved_docs(retrieved_docs)

    parts = [
        "### System",
        SYSTEM_PROMPT,
        "",
        "### User profile / session memory",
        memory_summary or "No prior car or mod information is known for this session.",
        "",
        "### Retrieved forum context",
        docs_block,
        "",
        "### User question",
        user_message.strip(),
        "",
        "### Your job",
        (
            "Based on the system instructions, the known car/mod context, and the retrieved forum "
            "snippets, answer the user's question as a cautious, expert tuner.\n"
            "- Explain your reasoning.\n"
            "- Emphasize safe diagnostic steps and logging.\n"
            "- Do NOT output raw numeric tables or exact timing/boost values.\n"
            "- If data is missing, ask for it or give general guidance instead of guessing."
        ),
    ]

    return "\n".join(parts)
