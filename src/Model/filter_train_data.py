#!/usr/bin/env python3
"""
filter_train_data.py

Final cleanup for SFT training data:
- Drop examples where INPUT or OUTPUT clearly contains forum editor toolbar JSON
  (toolbarButtons, xfSmilie, xfInsert, etc.).

Input:  src/Model/data/train.jsonl
Output: src/Model/data/train_filtered.jsonl
"""

from pathlib import Path
import json

BAD_MARKERS = [
    "toolbarButtons",
    "xfSmilie",
    "xfInsert",
    "xfCustom_trigger_attachment",
    "froalaEditor",
    "tinymce",
    "ckeditor",
]

MIN_OUTPUT_CHARS = 40  # skip ultra-tiny outputs too


def is_bad_example(example: dict) -> bool:
    inp = example.get("input", "") or ""
    out = example.get("output", "") or ""
    blob = (inp + "\n" + out)

    # Filter by obvious toolbar/editor markers
    lowered = blob.lower()
    for marker in BAD_MARKERS:
        if marker.lower() in lowered:
            return True

    # Skip examples with super tiny outputs
    if len(out.strip()) < MIN_OUTPUT_CHARS:
        return True

    return False


def main():
    in_path = Path("data/train.jsonl")
    out_path = Path("data/train_filtered.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(f"Input train.jsonl not found at {in_path}")

    kept = 0
    dropped = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            if is_bad_example(ex):
                dropped += 1
                continue

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[Filter] Kept {kept} examples, dropped {dropped} → {out_path}")


if __name__ == "__main__":
    main()
