# src/Model/data_preparation.py
#!/usr/bin/env python3
"""
Data Preparation for LLM Training

- Reads cleaned forum posts (JSONL).
- Converts them into instruction-response pairs.
- Splits into train / eval sets.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from llm_config import LLMConfig


class DataPreparator:
    def __init__(self, config: LLMConfig):
        self.cfg = config
        self.rng = random.Random(42)

    # ---------- Loading ----------

    def load_forum_posts(self) -> List[Dict]:
        posts: List[Dict] = []

        # Prefer tagged file; if not present, fall back to clean file
        if self.cfg.tagged_forum_file.exists():
            path = self.cfg.tagged_forum_file
            print(f"[Prep] Using TAGGED forum file: {path}")
        elif self.cfg.raw_forum_file.exists():
            path = self.cfg.raw_forum_file
            print(f"[Prep] Tagged file not found, using CLEAN file: {path}")
        else:
            raise FileNotFoundError(
                f"Neither tagged nor clean forum file found.\n"
                f"Expected:\n  {self.cfg.tagged_forum_file}\n  {self.cfg.raw_forum_file}"
            )

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    posts.append(obj)
                except json.JSONDecodeError:
                    continue

        return posts
    # ---------- Conversion ----------

    def make_pairs(self, posts: List[Dict]) -> List[Dict]:
        """
        Convert raw/ tagged posts into instruction/response pairs.

        This version is defensive:
        - Tries multiple possible text fields: content, text, body, post, message.
        - Uses tags + post_type if present, but doesn't require them.
        """
        pairs: List[Dict] = []

        # helpful counter for quick debugging
        skipped_empty = 0

        for p in posts:
            # --- find the main text field ---
            text_candidates = []
            for key in ("content", "text", "body", "post", "message"):
                val = p.get(key)
                if isinstance(val, str) and val.strip():
                    text_candidates.append(val.strip())

            if not text_candidates:
                skipped_empty += 1
                continue

            content = "\n".join(text_candidates)

            # --- meta fields ---
            title = (p.get("thread_title") or p.get("title") or "").strip()
            tags = p.get("tags") or []
            post_type = p.get("post_type") or "other"

            # --- build instruction based on tags / post_type if available ---
            base_instr = "You are an ECU tuning expert."

            if isinstance(tags, list):
                if "N54" in tags:
                    base_instr += " You specialize in BMW N54 engines."
                if any(t in tags for t in ["Honda B-series", "Honda K-series"]):
                    base_instr += " You are very familiar with Honda B/K-series tuning."
                if "Fueling" in tags:
                    base_instr += " Pay special attention to fueling, AFR, and trims."
                if "Boost" in tags:
                    base_instr += " Consider boost control, WGDC, and turbo limits."

            if post_type == "question":
                base_instr += (
                    " Answer the user's tuning question clearly, including relevant "
                    "background, likely causes, and safe next steps."
                )
            elif post_type == "problem":
                base_instr += (
                    " Diagnose the problem, propose likely causes, and suggest safe "
                    "debugging steps. Avoid guessing numeric tuning values."
                )
            elif post_type == "answer":
                base_instr += (
                    " Summarize and clean up the following tuning advice, explaining "
                    "why it works and any caveats."
                )
            else:
                base_instr += (
                    " Explain the following post, focusing on ECU-tuning-related concepts."
                )

            context_parts = []
            if title:
                context_parts.append(f"Thread title: {title}")
            if tags:
                # ensure tags is stringifiable
                if isinstance(tags, list):
                    tags_str = ", ".join(map(str, tags))
                else:
                    tags_str = str(tags)
                context_parts.append(f"Tags: {tags_str}")
            context_parts.append(f"Post:\n{content}")
            context_text = "\n\n".join(context_parts)

            pairs.append(
                {
                    "instruction": base_instr,
                    "input": context_text,
                    # for now we echo; later you can generate improved outputs
                    "output": content,
                }
            )

        print(f"[Prep] Skipped {skipped_empty} posts with no usable text")
        return pairs

    # ---------- Split + save ----------

    def train_eval_split(
        self, pairs: List[Dict], eval_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        self.rng.shuffle(pairs)
        n_eval = max(1, int(len(pairs) * eval_ratio))
        eval_set = pairs[:n_eval]
        train_set = pairs[n_eval:]
        return train_set, eval_set

    def save_jsonl(self, items: List[Dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---------- Orchestration ----------

    def run(self) -> None:
        print(f"[Prep] Loading posts from {self.cfg.raw_forum_file}")
        posts = self.load_forum_posts()
        print(f"[Prep] Loaded {len(posts)} posts")

        print("[Prep] Converting to instruction/response pairs...")
        pairs = self.make_pairs(posts)
        print(f"[Prep] Built {len(pairs)} pairs")

        train_set, eval_set = self.train_eval_split(pairs)

        print(f"[Prep] Saving train → {self.cfg.sft_train_file}")
        self.save_jsonl(train_set, self.cfg.sft_train_file)

        print(f"[Prep] Saving eval  → {self.cfg.sft_eval_file}")
        self.save_jsonl(eval_set, self.cfg.sft_eval_file)

        print("[Prep] Done.")


if __name__ == "__main__":
    cfg = LLMConfig()
    prep = DataPreparator(cfg)
    prep.run()
