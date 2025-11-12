#!/usr/bin/env python3
"""
Inference Engine for Fine-tuned ECU Tuning LLM
Load trained model and generate responses
"""
import torch
from pathlib import Path
from typing import Optional, List, Dict
import sys

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("        Install with: pip install transformers peft torch")
    sys.exit(1)

from llm_config import LLMConfig


class ECUTuningLLM:
    """Inference engine for the fine-tuned ECU tuning model"""

    def __init__(self, model_path: Path, config: Optional[LLMConfig] = None):
        self.model_path = model_path
        self.config = config or LLMConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        print(f"[INFO] Device: {self.device}")

    def load_model(self):
        """Load the fine-tuned model"""
        print(f"\n[LOAD] Loading model from {self.model_path}")

        if not self.model_path.exists():
            print(f"[ERROR] Model not found at {self.model_path}")
            print("        Train the model first with: python train_llm.py")
            sys.exit(1)

        # Load tokenizer
        print("        Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load model
        print("        Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

        print("[OK]    Model loaded successfully")

    def generate(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate response to an instruction"""

        # Format prompt
        prompt = self._format_prompt(instruction, context)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                top_k=top_k or self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after "### Response:")
        response = self._extract_response(full_response)

        return response

    def _format_prompt(self, instruction: str, context: str = "") -> str:
        """Format instruction and context into model prompt"""
        prompt = f"""{self.config.system_prompt}

### Instruction:
{instruction}

### Context:
{context}

### Response:
"""
        return prompt

    def _extract_response(self, full_text: str) -> str:
        """Extract just the response from full generation"""
        # Split on "### Response:" and take everything after
        if "### Response:" in full_text:
            parts = full_text.split("### Response:")
            if len(parts) > 1:
                return parts[-1].strip()

        return full_text.strip()

    def chat(self):
        """Interactive chat loop"""
        print("\nECUdapt AI - Your ECU Tuning Assistant")
        print("Fine-tuned on real forum discussions")
        print("Type 'exit' or 'quit' to end session")
        print("-" * 50)

        conversation_history = []

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nGoodbye! Thanks for using ECUdapt AI.")
                    break

                # Build context from conversation history
                context = self._build_context(conversation_history)

                # Generate response
                print("\n[INFO] Thinking...", end="", flush=True)
                response = self.generate(user_input, context)
                print("\r" + " " * 40 + "\r", end="")  # Clear "Thinking..."

                print(f"ECUdapt AI: {response}")

                # Update conversation history
                conversation_history.append({
                    "instruction": user_input,
                    "response": response
                })

                # Keep only last 3 exchanges for context
                conversation_history = conversation_history[-3:]

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")

    def _build_context(self, history: List[Dict]) -> str:
        """Build context from conversation history"""
        if not history:
            return "This is the start of the conversation."

        context_parts = ["Previous conversation:"]
        for exchange in history[-2:]:  # Last 2 exchanges
            context_parts.append(f"Q: {exchange['instruction'][:100]}")
            context_parts.append(f"A: {exchange['response'][:100]}")

        return "\n".join(context_parts)

    def batch_generate(self, instructions: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """Generate responses for multiple instructions"""
        if contexts is None:
            contexts = [""] * len(instructions)

        responses = []
        for instruction, context in zip(instructions, contexts):
            response = self.generate(instruction, context)
            responses.append(response)

        return responses


def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="ECUdapt AI Inference")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ecutuning-llm"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (non-interactive)"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Additional context for the question"
    )

    args = parser.parse_args()

    # Initialize model
    llm = ECUTuningLLM(args.model_path)
    llm.load_model()

    # Run inference
    if args.question:
        # Single question mode
        print(f"\nQuestion: {args.question}")
        if args.context:
            print(f"Context: {args.context}")

        print("\n[INFO] Generating answer...")
        response = llm.generate(args.question, args.context)

        print(f"\nAnswer:\n{response}\n")
    else:
        # Interactive chat mode
        llm.chat()


if __name__ == "__main__":
    main()
