# src/Model/rag_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_config import LLMConfig

# Try to import faiss; fall back to pure NumPy if unavailable (e.g. Python 3.12)
try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class RagIndex:
    cfg: LLMConfig
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        self.embedder = SentenceTransformer(self.model_name)

        self.index_path = self.cfg.rag_index_dir / "faiss_index.bin"
        self.meta_path = self.cfg.rag_index_dir / "docs_meta.json"
        self.emb_path = self.cfg.rag_index_dir / "embeddings.npy"

        self.use_faiss: bool = faiss is not None

        self.index: Optional[Any] = None           # faiss index if available
        self.embeddings: Optional[np.ndarray] = None  # fallback embeddings
        self.docs: List[Dict] = []

    # ---------- Internal helpers ----------

    def _read_source_docs(self) -> List[Dict]:
        """
        Read from tagged forum file if available, else fall back to clean forum file.
        """
        tagged = self.cfg.tagged_forum_file
        clean = self.cfg.raw_forum_file

        if tagged.exists():
            src = tagged
            print(f"[RAG] Reading tagged docs from {src}")
        elif clean.exists():
            src = clean
            print(f"[RAG] Tagged file missing. Using CLEAN file: {src}")
        else:
            raise FileNotFoundError(
                f"No RAG source file found. Neither {tagged} nor {clean} exists."
            )

        docs: List[Dict] = []
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
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
                        "tags": obj.get("tags"),
                    }
                )

        print(f"[RAG] Loaded {len(docs)} docs from source.")
        return docs

    # ---------- Building ----------

    def build_from_jsonl(self, max_docs: Optional[int] = None) -> None:
        docs = self._read_source_docs()
        if max_docs is not None:
            docs = docs[:max_docs]

        self.docs = docs
        texts = [d["content"] for d in docs]

        print(
            f"[RAG] Encoding {len(texts)} docs with "
            f"{self.model_name} (use_faiss={self.use_faiss})..."
        )
        emb = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        emb = emb.astype("float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb_norm = emb / norms

        self.cfg.rag_index_dir.mkdir(parents=True, exist_ok=True)

        # Save docs metadata
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

        # Save embeddings for fallback/no-faiss mode
        np.save(self.emb_path, emb_norm)

        if self.use_faiss:
            dim = emb_norm.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb_norm)
            self.index = index
            faiss.write_index(index, str(self.index_path))
            print(f"[RAG] Saved FAISS index → {self.index_path}")
        else:
            self.embeddings = emb_norm
            print("[RAG] FAISS not available; using NumPy cosine similarity fallback.")

        print(f"[RAG] Saved docs meta     → {self.meta_path}")
        print(f"[RAG] Saved embeddings    → {self.emb_path}")

    # ---------- Loading ----------

    def load(self) -> None:
        if not self.meta_path.exists():
            raise FileNotFoundError(
                "RAG index not built yet. Run build_from_jsonl first."
            )

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.docs = json.load(f)

        if self.use_faiss and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.embeddings = None
            print(f"[RAG] Loaded FAISS index from {self.index_path}")
        else:
            # fall back to NumPy embeddings
            if not self.emb_path.exists():
                raise FileNotFoundError(
                    "Embeddings file not found and FAISS is unavailable. "
                    "Run build_from_jsonl to rebuild the RAG index."
                )
            self.embeddings = np.load(self.emb_path)
            self.index = None
            print("[RAG] Loaded NumPy embeddings fallback.")

    # ---------- Query ----------

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not self.docs:
            self.load()

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = q_emb[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        if self.use_faiss and self.index is not None:
            scores, idxs = self.index.search(q_norm[None, :], k)
            scores = scores[0]
            idxs = idxs[0]
        else:
            assert self.embeddings is not None, "Embeddings not loaded."
            emb = self.embeddings
            scores = emb @ q_norm
            idxs = np.argsort(scores)[::-1][:k]

        results: List[Dict] = []
        for score, idx in zip(scores, idxs):
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
