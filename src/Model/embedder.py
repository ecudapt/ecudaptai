import json, numpy as np, faiss, time, os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def build_index(input_path: Path, index_path: Path):
    print("⚙️  Initializing sentence embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"📂 Reading cleaned forum data from {input_path}")
    texts, metas = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="📜 Loading JSONL", ncols=100):
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if len(text) < 25:
                    continue
                texts.append(text)
                metas.append(obj.get("url", ""))
            except Exception as e:
                print(f"⚠️ Skipping malformed line: {e}")

    print(f"🧠 Total valid documents to embed: {len(texts)}")
    if not texts:
        print("❌ No valid documents to process — aborting.")
        return

    print("\n🚀 Generating embeddings...")
    start_time = time.time()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )
    duration = time.time() - start_time
    print(f"✅ Generated embeddings for {len(texts)} docs in {duration:.1f}s")

    # --- Build FAISS index ---
    dim = embeddings.shape[1]
    print(f"\n🧩 Building FAISS index (dim={dim})...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # --- Save index and metadata ---
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(index_path.with_suffix(".meta.npy"), np.array(metas))
    size = os.path.getsize(index_path) / 1e6
    print(f"💾 Saved FAISS index → {index_path} ({size:.1f} MB)")
    print(f"💾 Saved metadata → {index_path.with_suffix('.meta.npy')}")
    print(f"[✓] Indexed {len(texts)} total documents ✅")

if __name__ == "__main__":
    build_index(
        Path("data/clean/forum_posts_clean.jsonl"),
        Path("data/vector/forum.faiss")
    )
