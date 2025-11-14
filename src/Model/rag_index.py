# src/Model/rag_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_config import LLMConfig


@dataclass
class RagIndex:
    """
    RAG index builder + retriever.

    Uses a JSONL forum file (cfg.raw_forum_file) where each line is a dict.
    Tries to be flexible about field names so it can work with:
      - forum_posts_normalized.jsonl
      - forum_posts_tagged.jsonl
      - etc.
    """
    cfg: LLMConfig
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        self.embedder = SentenceTransformer(self.model_name)
        self.index_path = self.cfg.rag_index_dir / "faiss_index.bin"
        self.meta_path = self.cfg.rag_index_dir / "docs_meta.json"

        self.index: Optional[faiss.Index] = None
        self.docs: List[Dict[str, Any]] = []

    # ---------- Building ----------

    def _extract_text_and_meta(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extracts a display text and metadata from a raw forum JSON object.
        Tries multiple key names to be robust to schema changes.
        """

        # Text content: try multiple keys
        text = (
            obj.get("content")
            or obj.get("post_text")
            or obj.get("text")
            or obj.get("body")
            or obj.get("normalized_text")
            or ""
        ).strip()
        if not text:
            return None

        # Thread title / URL / platform
        thread_title = (
            obj.get("thread_title")
            or obj.get("title")
            or ""
        )
        url = obj.get("thread_url") or obj.get("url") or ""
        platform = obj.get("platform") or obj.get("forum") or ""

        # Domains/tags – normalize to a list
        domains_raw = (
            obj.get("domains")
            or obj.get("tags")
            or obj.get("domain")
            or []
        )
        if isinstance(domains_raw, str):
            domains = [domains_raw]
        elif isinstance(domains_raw, list):
            domains = [str(d) for d in domains_raw]
        else:
            domains = [str(domains_raw)]

        # Build a richer text block so the embedder sees context
        header_lines = []
        if thread_title:
            header_lines.append(f"[Thread] {thread_title}")
        if platform:
            header_lines.append(f"[Platform] {platform}")
        if domains:
            header_lines.append(f"[Domains] {', '.join(domains)}")

        header = "\n".join(header_lines)
        full_text = f"{header}\n\nPost:\n{text}".strip()

        return {
            "content": full_text,       # what gets embedded
            "thread_title": thread_title or None,
            "url": url or None,
            "platform": platform or None,
            "domains": domains,
        }

    def build_from_jsonl(self, max_docs: Optional[int] = None) -> None:
        """
        Build FAISS index + docs metadata from cfg.raw_forum_file.
        Intended source: your *tagged/normalized* forum JSONL file.
        """

        src = self.cfg.rag_source_file
        if not src.exists():
            raise FileNotFoundError(f"RAG source file not found: {src}")

        docs: List[Dict[str, Any]] = []

        print(f"[RAG] Reading from {src}")
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs is not None and i >= max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                out = self._extract_text_and_meta(obj)
                if out is None:
                    continue

                docs.append(out)

        if not docs:
            raise RuntimeError(f"[RAG] No valid docs extracted from {src}")

        self.docs = docs

        texts = [d["content"] for d in docs]
        print(f"[RAG] Encoding {len(texts)} docs with {self.model_name}...")
        emb = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        emb = emb.astype("float32")
        faiss.normalize_L2(emb)

        dim = emb.shape[1]
        print(f"[RAG] Embedding dim = {dim}")
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        self.index = index

        # Ensure dir exists
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
        print(f"[RAG] Loading index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.docs = json.load(f)
        print(f"[RAG] Loaded {len(self.docs)} docs.")

    # ---------- Query ----------

    def retrieve(self, query: str, k: int = 5, domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k docs for a query.
        Optionally filter/boost by domain tags.
        """
        if self.index is None or not self.docs:
            self.load()

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k * 3)  # oversample, then filter by domain

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.docs):
                continue

            doc = dict(self.docs[idx])  # copy
            doc["score"] = float(score)

            if domains:
                doc_domains = doc.get("domains") or []
                # Require at least one overlap:
                if not any(d in doc_domains for d in domains):
                    continue

            results.append(doc)
            if len(results) >= k:
                break

        return results


def build_index_cli():
    cfg = LLMConfig()
    rag = RagIndex(cfg)
    rag.build_from_jsonl()
    print("[RAG] Build complete.")


if __name__ == "__main__":
    build_index_cli()
