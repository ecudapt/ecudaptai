#!/usr/bin/env python3
"""
normalize_posts.py

Normalize forum posts for LLM training:

- Strip forum chrome: usernames, role labels, joined/posts lines, quote headers.
- Try to extract `author` and `timestamp` into separate fields.
- Leave only the actual post content in `content`.

Input:  src/Model/data/clean/forum_posts_clean.jsonl
Output: src/Model/data/clean/forum_posts_normalized.jsonl
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from llm_config import LLMConfig


# --------- Helper regexes ----------

DATE_PATTERN = re.compile(
    r"\b("
    r"Mon|Tue|Wed|Thu|Fri|Sat|Sun"
    r")?\s*,?\s*"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}"
    r"(?:\s+\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)?",
    re.IGNORECASE,
)

QUOTE_HEADER_PATTERNS = [
    re.compile(r"^Quote Post by\b", re.IGNORECASE),
    re.compile(r"^Quote from\b", re.IGNORECASE),
    re.compile(r"^Originally posted by\b", re.IGNORECASE),
]

USER_HEADER_PATTERN = re.compile(
    # e.g. "geartech Gear Head Posts: 16 Joined: Fri Nov 26, 2010 8:32 am"
    r"^\s*[A-Za-z0-9_\-]+\s+(?:Newbie|Member|Corporal|Sergeant|Captain|"
    r"Administrator|Moderator|Gear Head|Vendor)\b.*",
    re.IGNORECASE,
)

JOINED_POSTS_PATTERN = re.compile(
    r"\bJoined:\b|\bPosts:\b|\bLocation:\b", re.IGNORECASE
)


def get_text_from_obj(obj: Dict) -> Tuple[str, List[str]]:
    """Return raw text and list of source fields used."""
    fields = []
    texts = []
    for key in ("content", "text", "body", "post", "message"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            fields.append(key)
            texts.append(val.strip())
    return ("\n".join(texts), fields)


def extract_author_and_timestamp(lines: List[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Try to pull author + timestamp from top lines.
    Returns (cleaned_lines, author, timestamp_str).
    """

    author = None
    timestamp = None
    cleaned = []

    # First, scan for obvious quote headers / user headers / joined/posts lines
    for line in lines:
        stripped = line.strip()

        # Skip empty
        if not stripped:
            continue

        # Skip XenForo / BB quote headers
        if any(pat.search(stripped) for pat in QUOTE_HEADER_PATTERNS):
            continue

        # Skip 'Joined: ..., Posts: ...'
        if JOINED_POSTS_PATTERN.search(stripped):
            continue

        # Skip composite user header lines like "geartech Gear Head Posts: 16 Joined: ..."
        if USER_HEADER_PATTERN.match(stripped):
            # Try to grab first token as author
            if author is None:
                author = stripped.split()[0]
            # Don't keep this line
            continue

        # If we haven't seen a timestamp yet, try to detect in any line
        if timestamp is None:
            m = DATE_PATTERN.search(stripped)
            if m:
                timestamp = m.group(0)
                # Often those lines are pure date; drop them
                # But if they also contain real text, keep them minus date
                # For simplicity, drop entire line if it's short
                if len(stripped) < 40:
                    continue

        cleaned.append(line)

    return cleaned, author, timestamp


def normalize_content(raw: str) -> str:
    """
    Apply line-level normalization:
    - Remove stray 'Quote Post by...' lines (handled above, but this is a second pass).
    - Strip leading/trailing whitespace.
    """
    lines = raw.splitlines()
    out_lines: List[str] = []

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            out_lines.append("")
            continue

        if any(pat.search(stripped) for pat in QUOTE_HEADER_PATTERNS):
            # skip embedded quote headers
            continue

        out_lines.append(stripped)

    # Strip leading/trailing blank lines
    while out_lines and not out_lines[0].strip():
        out_lines.pop(0)
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    return "\n".join(out_lines)


def main():
    cfg = LLMConfig()

    input_path = cfg.data_dir / "clean" / "forum_posts_clean.jsonl"
    output_path = cfg.data_dir / "clean" / "forum_posts_normalized.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_text, fields_used = get_text_from_obj(obj)
            if not raw_text:
                # nothing to normalize
                continue

            # Split into lines and try to pull author/timestamp out of header-ish lines
            raw_lines = raw_text.splitlines()
            cleaned_lines, author, timestamp = extract_author_and_timestamp(raw_lines)
            cleaned_text = normalize_content("\n".join(cleaned_lines))

            if not cleaned_text.strip():
                # all header, no content
                continue

            # Update object
            obj["content"] = cleaned_text  # normalized main field

            if author and not obj.get("author"):
                obj["author"] = author
            if timestamp and not obj.get("timestamp"):
                obj["timestamp"] = timestamp

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[Normalize] Read {n_in} lines, wrote {n_out} normalized posts → {output_path}")


if __name__ == "__main__":
    main()
