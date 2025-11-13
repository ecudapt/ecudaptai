#!/usr/bin/env python3
"""
tag_dataset.py

Stage 1: Improve your domain dataset.

- Reads data/clean/forum_posts_clean.jsonl
- Adds:
  - tags: ["N54", "Honda B-series", "Fueling", "Boost", ...]
  - post_type: "question" | "answer" | "problem" | "other"
- Writes: data/clean/forum_posts_tagged.jsonl
"""

import json
import re
from pathlib import Path
from typing import Dict, List


# ---------- Heuristic keyword sets ----------

ENGINE_TAGS = {
    "N54": [r"\bn54\b", r"I8A0S", r"IJE0S"],
    "N55": [r"\bn55\b"],
    "B58": [r"\bb58\b"],
    "S55": [r"\bs55\b"],
    "N52": [r"\bn52\b"],

    "Honda B-series": [r"\bb16a\b", r"\bb18c\b", r"\bb20b\b", r"\bB18B\b"],
    "Honda K-series": [r"\bk20\b", r"\bk24\b"],
    "VW/Audi TDI": [r"\btdi\b", r"\bcjaa\b"],
}

TOPIC_TAGS = {
    "Boost": [
        r"\bboost\b", r"\bpsi\b", r"\bwgdc\b", r"wastegate", r"boost creep"
    ],
    "Fueling": [
        r"\bafr\b", r"lambda", r"fuel trim", r"lp fp", r"hpfp", r"lpfp",
        r"\binjector\b", r"\binjectors\b"
    ],
    "Ignition": [
        r"\bignition\b", r"\btiming\b", r"timing pull", r"\bspark\b"
    ],
    "Knock": [
        r"\bknock\b", r"detonation", r"pre-ignition"
    ],
    "ECU / Tune": [
        r"\btune\b", r"\btuning\b", r"\bflash\b", r"\bmap\b", r"\bstage 1\b",
        r"\bstage 2\b", r"\bcalibration\b", r"\brom\b"
    ],
    "Sensors / Logging": [
        r"\blog\b", r"\blogged\b", r"\bdatalog\b", r"\bOBD2\b", r"\bMHD\b",
        r"\bHondata\b", r"\bHP Tuners\b"
    ],
    "Turbo / Hardware": [
        r"\bturbo\b", r"\bturbos\b", r"\bintercooler\b", r"\bdownpipe\b",
        r"\bdownpipes\b", r"\bcatless\b"
    ],
}

QUESTION_PATTERNS = [
    r"\?\s*$",
    r"any ideas",
    r"what should I",
    r"does anyone know",
    r"can I",
    r"how do I",
]

PROBLEM_PATTERNS = [
    r"\bcode\b", r"\bCEL\b", r"check engine", r"misfire", r"limp mode",
    r"smoke", r"won't start", r"hard start", r"stumble", r"hesitation",
]


def normalize_text(text: str) -> str:
    return text.lower()


def match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def get_engine_tags(text: str) -> List[str]:
    tags = []
    for label, patterns in ENGINE_TAGS.items():
        if match_any(patterns, text):
            tags.append(label)
    return tags


def get_topic_tags(text: str) -> List[str]:
    tags = []
    for label, patterns in TOPIC_TAGS.items():
        if match_any(patterns, text):
            tags.append(label)
    return tags


def infer_post_type(text: str, title: str) -> str:
    full = f"{title}\n{text}".lower()

    if match_any(QUESTION_PATTERNS, full):
        return "question"
    if match_any(PROBLEM_PATTERNS, full):
        return "problem"

    # crude heuristic: "here's what I did" / "solved it"
    if "fixed it" in full or "here's log" in full or "for anyone wondering" in full:
        return "answer"

    return "other"


def tag_post(obj: Dict) -> Dict:
    content = (obj.get("content") or "").strip()
    title = (obj.get("thread_title") or "").strip()

    text_for_tags = f"{title}\n{content}"
    text_for_tags_norm = normalize_text(text_for_tags)

    engine_tags = get_engine_tags(text_for_tags_norm)
    topic_tags = get_topic_tags(text_for_tags_norm)
    post_type = infer_post_type(content, title)

    tags = sorted(set(engine_tags + topic_tags))

    obj["tags"] = tags
    obj["post_type"] = post_type
    return obj


def main():
    input_path = Path("data/clean/forum_posts_clean.jsonl")
    output_path = Path("data/clean/forum_posts_tagged.jsonl")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    n_in = 0
    n_out = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            tagged = tag_post(obj)
            fout.write(json.dumps(tagged, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[Tag] Read {n_in} lines, wrote {n_out} tagged posts → {output_path}")


if __name__ == "__main__":
    main()
