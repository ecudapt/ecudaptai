# src/Model/inference.py
#!/usr/bin/env python3
"""
Inference engine with self-consistency and anti-quote decoding.
"""
import sys, torch
from pathlib import Path
from typing import Optional, List, Dict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"❌ Missing package: {e}\nInstall: pip install transformers torch")
    sys.exit(1)

from llm_config import LLMConfig


class ECUTuningLLM:
    def __init__(self, model_path: Path, config: Optional[LLMConfig] = None):
        self.model_path = model_path
        self.config = config or LLMConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        print(f"🖥️  Device: {self.device}")

    def load_model(self):
        print(f"\n📦 Loading model from {self.model_path}")
        if not self.model_path.exists():
            print(f"❌ Not found: {self.model_path}")
            sys.exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        print("✅ Model loaded")

    def _format_prompt(self, instruction: str, context: str = "") -> str:
        return f"""{self.config.system_prompt}

### Instruction:
{instruction}

### Context:
{context}

### Response:
"""

    def _extract_response(self, full_text: str) -> str:
        if "### Response:" in full_text:
            parts = full_text.split("### Response:")
            if len(parts) > 1:
                return parts[-1].strip()
        return full_text.strip()

    def generate(
        self, instruction: str, context: str = "",
        max_new_tokens: Optional[int] = None, temperature: Optional[float] = None,
        top_p: Optional[float] = None, top_k: Optional[int] = None
    ) -> str:
        prompt = self._format_prompt(instruction, context)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=self.config.max_seq_length).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature if temperature is None else temperature,
                top_p=self.config.top_p if top_p is None else top_p,
                top_k=self.config.top_k if top_k is None else top_k,
                do_sample=True,
                repetition_penalty=max(self.config.repetition_penalty, 1.15),
                no_repeat_ngram_size=6,                 # reduce verbatim copying
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_response(full)

    # Simple consensus: sample N answers, pick the best by structure/length
    def generate_consensus(self, instruction: str, context: str = "", n: int = 3) -> str:
        cands=[]
        for _ in range(n):
            cands.append(self.generate(instruction, context, temperature=0.35, top_p=0.9, top_k=50))

        def score(t: str) -> int:
            s=0
            if "Decision:" in t: s+=2
            if "Why:" in t: s+=2
            if "Safety:" in t: s+=3
            L=len(t.split())
            if 60 <= L <= 250: s+=2
            if "Originally posted" in t or '"' in t: s-=3
            return s

        return max(cands, key=score)

    def chat(self):
        print("\n🚗 ECUdapt AI — LLM (type 'exit' to quit)")
        hist=[]
        while True:
            try:
                q=input("\n❓ You: ").strip()
                if q.lower() in {"exit","quit","bye"}: print("👋"); break
                ctx="Previous conversation:\n"+"\n".join([f"Q:{h['q']}\nA:{h['a']}" for h in hist[-2:]]) if hist else ""
                print("💭 Thinking…", end="", flush=True)
                a=self.generate_consensus(q, ctx, n=3)
                print("\r", end="")
                print(f"🤖 ECUdapt AI:\n{a}\n")
                hist.append({"q":q,"a":a})
                hist=hist[-3:]
            except KeyboardInterrupt:
                print("\n👋"); break
            except Exception as e:
                print(f"\n⚠️  {e}")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=Path("models/ecutuning-llm-v2"))
    p.add_argument("--question", type=str)
    p.add_argument("--context", type=str, default="")
    a=p.parse_args()

    llm=ECUTuningLLM(a.model_path)
    llm.load_model()

    if a.question:
        print(f"\n❓ {a.question}")
        if a.context: print(f"📋 {a.context}")
        print("\n💭 Generating…")
        ans=llm.generate_consensus(a.question, a.context, n=3)
        print(f"\n🤖 Answer:\n{ans}\n")
    else:
        llm.chat()

if __name__=="__main__":
    main()
