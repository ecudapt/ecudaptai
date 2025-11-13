# src/Model/prompts.py
from typing import Optional, Dict, List


BASE_PERSONA = """
You are ECUDapt, an ECU tuning assistant.

You:
- Specialize in modern ECUs (N54, B-series Hondas, TDIs, etc.)
- Prioritize engine safety, reliability, and data-driven decisions.
- Hate guessing: you say when something is uncertain or requires logs.
- Refuse to give obviously dangerous advice or numbers without context.

Style:
- Speak like a technically savvy, calm friend.
- Explain your reasoning step by step when it matters.
- Always call out risks and suggest datalogging + verification steps.
"""


def build_system_prompt(user_profile: Optional[Dict] = None) -> str:
    """Build a system prompt that includes long-lived user info."""
    prompt = BASE_PERSONA.strip()

    if user_profile:
        prefs_lines: List[str] = ["", "User profile / preferences:"]
        if name := user_profile.get("name"):
            prefs_lines.append(f"- Name: {name}")
        if cars := user_profile.get("cars"):
            prefs_lines.append(f"- Cars: {', '.join(cars)}")
        if risk := user_profile.get("risk_tolerance"):
            prefs_lines.append(f"- Risk tolerance: {risk} (favor reliability)")
        if goals := user_profile.get("goals"):
            prefs_lines.append(f"- Goals: {', '.join(goals)}")

        prompt += "\n" + "\n".join(prefs_lines)

    return prompt


def build_user_message(query: str, retrieved_docs: str, extra_instructions: str = "") -> str:
    """
    Wrap user query + retrieved forum/docs into a single content string.
    This is the 'context' that goes into the chat template as the user message.
    """
    parts = []

    if extra_instructions:
        parts.append(extra_instructions.strip())

    if retrieved_docs:
        parts.append(
            "Here are some potentially relevant excerpts from tuning forums and docs:\n"
            f"{retrieved_docs.strip()}\n\n"
            "Use these as noisy context — they may contain mixed quality info, "
            "so reason carefully and prioritize safety."
        )

    parts.append(f"User query:\n{query.strip()}")
    return "\n\n".join(parts)
