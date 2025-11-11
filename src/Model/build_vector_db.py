from data_cleaner import process_raw_threads
from embedder import build_index
from pathlib import Path

RAW_THREADS = Path("data/raw/threads")
CLEAN_FILE = Path("data/clean/forum_posts_clean.jsonl")
INDEX_FILE = Path("data/vector/forum.faiss")

def run_pipeline():
    process_raw_threads(RAW_THREADS, CLEAN_FILE)
    build_index(CLEAN_FILE, INDEX_FILE)

if __name__ == "__main__":
    run_pipeline()
