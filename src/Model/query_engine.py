import faiss, numpy as np, json
from sentence_transformers import SentenceTransformer

class ForumRetriever:
    def __init__(self, index_path="data/vector/forum.faiss", meta_path="data/vector/forum.meta.npy"):
        print("Loading vector index and metadata...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        self.meta = np.load(meta_path, allow_pickle=True)

    def search(self, query, k=5):
        q_vec = self.model.encode([query])
        D, I = self.index.search(q_vec, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({"url": self.meta[idx], "score": float(score)})
        return results