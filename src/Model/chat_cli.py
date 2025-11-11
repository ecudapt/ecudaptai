import os, sys, faiss, json, numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
INDEX_PATH = "data/vector/forum.faiss"
META_PATH = "data/vector/forum.meta.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Load your LLM provider key (OpenAI or Mistral endpoint)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# ────────────────────────────────────────────────────────────────
# INITIALIZE COMPONENTS
# ────────────────────────────────────────────────────────────────
print("⚙️  Loading embedding model and FAISS index...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
meta = np.load(META_PATH, allow_pickle=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# ────────────────────────────────────────────────────────────────
# RETRIEVAL + ANSWER LOGIC
# ────────────────────────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = TOP_K):
    """Find top_k most relevant documents."""
    q_vec = model.encode([query])
    D, I = index.search(q_vec, top_k)
    docs = []
    for idx, score in zip(I[0], D[0]):
        url = meta[idx]
        docs.append(f"- {url}  (score={score:.4f})")
    return "\n".join(docs)

def ask_model(question: str, context: str):
    """Send question + retrieved context to the LLM."""
    prompt = f"""
You are an automotive ECU tuning expert.
Use the following forum references to answer the question accurately and concisely.
If information is missing, reason based on your tuning knowledge.

Forum references:
{context}

Question: {question}
Answer:
"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",   # You can change this to mistral if using your own endpoint
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error contacting model: {e}]"

# ────────────────────────────────────────────────────────────────
# CLI LOOP
# ────────────────────────────────────────────────────────────────
def chat_loop():
    print("\n🚗 ECUdapt AI – ECU Tuning Chat Assistant")
    print("Type your question below (or 'exit' to quit).")
    print("──────────────────────────────────────────\n")

    while True:
        try:
            question = input("❓ > ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                print("👋 Exiting ECUdapt AI.")
                break

            print("\n🔎 Searching knowledge base...")
            context = retrieve_context(question)

            print("💬 Generating answer...\n")
            answer = ask_model(question, context)

            print("──────────────────────────────────────────")
            print(answer)
            print("──────────────────────────────────────────\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  Error: {e}\n")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chat_loop()
