from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from query_engine import ForumRetriever
from pathlib import Path

client = MistralClient(api_key="YOUR_MISTRAL_API_KEY")
retriever = ForumRetriever(Path("data/vector/forum.faiss"))

def ask(question: str):
    urls, _ = retriever.search(question, k=5)
    context = "\n".join(urls)
    prompt = f"You are an ECU tuning expert.\nRelevant notes:\n{context}\n\nQuestion: {question}"
    messages = [ChatMessage(role="user", content=prompt)]
    resp = client.chat(model="mistral-medium", messages=messages)
    print("\n💡 Answer:\n", resp.choices[0].message.content)

if __name__ == "__main__":
    ask("Why does boost taper at high RPM on an N54 engine?")
