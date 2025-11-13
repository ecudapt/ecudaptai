# src/Model/rag_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_config import LLMConfig


@dataclass
class RagIndex:
    cfg: LLMConfig
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        self.embedder = SentenceTransformer(self.model_name)
        self.index_path = self.cfg.rag_index_dir / "faiss_index.bin"
        self.meta_path = self.cfg.rag_index_dir / "docs_meta.json"

        self.index: Optional[faiss.IndexFlatIP] = None
        self.docs: List[Dict] = []

    # ---------- Building ----------

    def build_from_jsonl(self, max_docs: Optional[int] = None) -> None:
        docs: List[Dict] = []

        src = self.cfg.raw_forum_file
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = (obj.get("content") or "").strip()
                if not content:
                    continue
                docs.append(
                    {
                        "content": content,
                        "thread_title": obj.get("thread_title"),
                        "url": obj.get("url"),
                        "domain": obj.get("domain"),
                    }
                )

        self.docs = docs

        texts = [d["content"] for d in docs]
        print(f"[RAG] Encoding {len(texts)} docs...")
        emb = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        emb = emb.astype("float32")
        faiss.normalize_L2(emb)

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        self.index = index

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

        print(f"[RAG] Saved index → {self.index_path}")
        print(f"[RAG] Saved meta  → {self.meta_path}")

    # ---------- Loading ----------

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("RAG index not built yet. Run build_from_jsonl first.")
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.docs = json.load(f)

    # ---------- Query ----------

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None or not self.docs:
            self.load()

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = dict(self.docs[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results


def build_index_cli():
    cfg = LLMConfig()
    rag = RagIndex(cfg)
    rag.build_from_jsonl()
    print("[RAG] Build complete.")


if __name__ == "__main__":
    build_index_cli()
