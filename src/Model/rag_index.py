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

        self.index: Optional[Any] = None              # faiss index (if used)
        self.embeddings: Optional[np.ndarray] = None  # NumPy fallback
        self.docs: List[Dict] = []

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
        Try multiple sources in priority order:
        1) tagged forum file
        2) clean forum file
        3) SFT train file (as a last resort)
        """
        candidates = [
            ("TAGGED", self.cfg.tagged_forum_file),
            ("CLEAN", self.cfg.raw_forum_file),
            ("TRAIN", self.cfg.sft_train_file),
        ]

        for label, path in candidates:
            docs = self._load_docs_from_file(path, label)
            if docs:
                print(f"[RAG] Using {label} as RAG source.")
                return docs

        raise FileNotFoundError(
            "No non-empty RAG source file found. Tried tagged, clean, and train."
        )

    # ---------- Building ----------

    def build_from_jsonl(self, max_docs: Optional[int] = None) -> None:
        docs = self._read_source_docs()
        if max_docs is not None:
            docs = docs[:max_docs]

        if not docs:
            print("[RAG] No docs found to index. Aborting RAG build.")
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
            print("[RAG] Encoder returned empty embeddings. Aborting RAG build.")
            return

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

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")[0]
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
def _extract_content(self, obj: Dict) -> Optional[str]:
        """
        Accepts multiple possible field names and attempts to extract text content.
        """
        POSSIBLE_FIELDS = [
            "content",
            "text",
            "post",
            "body",
            "message",
            "raw",
            "cleaned",
            "normalized",
            "chunk",
        ]

        for key in POSSIBLE_FIELDS:
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key].strip()

        # Fallback: combine all short string fields
        string_parts = []
        for v in obj.values():
            if isinstance(v, str) and 10 < len(v) < 5000:
                string_parts.append(v.strip())

        if string_parts:
            return "\n".join(string_parts)

        return None


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

            content = self._extract_content(obj)
            if not content:
                continue

            docs.append(
                {
                    "content": content,
                    "thread_title": obj.get("thread_title") or obj.get("title"),
                    "url": obj.get("url"),
                    "domain": obj.get("domain"),
                    "tags": obj.get("tags"),
                }
            )

    print(f"[RAG] Loaded {len(docs)} docs from {label}.")
    return docs

def build_index_cli():
    cfg = LLMConfig()
    rag = RagIndex(cfg)
    rag.build_from_jsonl()
    print("[RAG] Build complete.")


if __name__ == "__main__":
    build_index_cli()
