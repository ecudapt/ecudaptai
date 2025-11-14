# src/Model/rag_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_config import LLMConfig

# Try to import faiss; fall back to pure NumPy if unavailable
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


@dataclass
class RagIndex:
    cfg: LLMConfig
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        # Path fixes — always correct relative to project root
        self.rag_dir: Path = self.cfg.rag_index_dir
        self.rag_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.rag_dir / "faiss_index.bin"
        self.meta_path = self.rag_dir / "docs_meta.json"
        self.emb_path = self.rag_dir / "embeddings.npy"

        self.embedder = SentenceTransformer(self.model_name)
        self.use_faiss: bool = faiss is not None

        self.index: Optional[Any] = None
        self.embeddings: Optional[np.ndarray] = None
        self.docs: List[Dict] = []

        print(f"[RAG] Using rag_index dir: {self.rag_dir}")

    # ---------- Internal helpers ----------

    def _load_docs_from_file(self, path: Path, label: str) -> List[Dict]:
        if not path.exists():
            print(f"[RAG] {label} file does not exist: {path}")
            return []

        print(f"[RAG] Reading {label} docs from {path}")
        docs: List[Dict] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
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

        print(f"[RAG] Loaded {len(docs)} docs from {label}.")
        return docs

    def _read_source_docs(self) -> List[Dict]:
        """
        Ordered sources:
        1) tagged forum posts
        2) clean forum posts
        3) SFT train file
        """
        candidates = [
            ("TAGGED", self.cfg.tagged_forum_file),
            ("CLEAN", self.cfg.raw_forum_file),
            ("TRAIN", self.cfg.sft_train_file),
        ]

        print("\n[RAG] Searching for source documents...")

        for label, path in candidates:
            print(f"[RAG] Checking {label} → {path}")
            docs = self._load_docs_from_file(path, label)
            if docs:
                print(f"[RAG] Using {label} as RAG source.")
                return docs

        raise FileNotFoundError(
            f"[RAG] ERROR: No valid RAG source found.\n"
            f"Tried:\n"
            f"- {self.cfg.tagged_forum_file}\n"
            f"- {self.cfg.raw_forum_file}\n"
            f"- {self.cfg.sft_train_file}"
        )

    # ---------- Build RAG index ----------

    def build_from_jsonl(self, max_docs: Optional[int] = None) -> None:
        docs = self._read_source_docs()

        if max_docs is not None:
            docs = docs[:max_docs]

        if not docs:
            print("[RAG] No docs found to index. Aborting.")
            return

        self.docs = docs
        texts = [d["content"] for d in docs]

        print(
            f"[RAG] Encoding {len(texts)} docs with "
            f"{self.model_name} (use_faiss={self.use_faiss})..."
        )

        emb = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        ).astype("float32")

        if emb.size == 0:
            print("[RAG] ERROR: Encoder returned empty embeddings.")
            return

        # Normalize for cosine similarity
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        # Save docs metadata
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

        # Save embeddings for fallback mode
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
            print("[RAG] FAISS not available; using NumPy fallback.")

        print(f"[RAG] Saved metadata → {self.meta_path}")
        print(f"[RAG] Saved embeddings → {self.emb_path}")

    # ---------- Load RAG index ----------

    def load(self) -> None:
        if not self.meta_path.exists():
            raise FileNotFoundError(
                "[RAG] Index not built. Run build_from_jsonl() first."
            )

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.docs = json.load(f)

        if self.use_faiss and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.embeddings = None
            print(f"[RAG] Loaded FAISS index → {self.index_path}")
        else:
            if not self.emb_path.exists():
                raise FileNotFoundError(
                    "[RAG] No FAISS index or embeddings file found."
                )
            self.embeddings = np.load(self.emb_path)
            self.index = None
            print("[RAG] Loaded NumPy embeddings.")

    # ---------- Query ----------

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not self.docs:
            self.load()

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        if self.use_faiss and self.index is not None:
            scores, idxs = self.index.search(q_norm[None, :], k)
            scores = scores[0]
            idxs = idxs[0]
        else:
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
    print(f"[RAG] Building index with cfg.rag_index_dir = {cfg.rag_index_dir}")
    rag = RagIndex(cfg)
    rag.build_from_jsonl()
    print("[RAG] Build complete.")


if __name__ == "__main__":
    build_index_cli()
