#!/usr/bin/env python3
"""
Evaluation Framework for ECU Tuning LLM
Test model performance on held-out validation set
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys

try:
    from rouge_score import rouge_scorer
    from tqdm import tqdm
except ImportError:
    print("⚠️  Optional packages missing. Install with:")
    print("   pip install rouge-score tqdm")

from inference import ECUTuningLLM
from llm_config import LLMConfig


@dataclass
class EvaluationResult:
    """Results from model evaluation"""
    avg_rouge1: float
    avg_rouge2: float
    avg_rougeL: float
    avg_latency: float
    total_questions: int
    successful: int
    failed: int
    samples: List[Dict]


class ModelEvaluator:
    """Evaluate fine-tuned model performance"""

    def __init__(self, model_path: Path, config: LLMConfig):
        self.model_path = model_path
        self.config = config
        self.llm = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def load_model(self):
        """Load the model for evaluation"""
        print("📦 Loading model for evaluation...")
        self.llm = ECUTuningLLM(self.model_path, self.config)
        self.llm.load_model()

    def load_validation_data(self) -> List[Dict]:
        """Load validation dataset"""
        val_path = self.config.val_data_path

        if not val_path.exists():
            print(f"❌ Validation data not found: {val_path}")
            sys.exit(1)

        print(f"📚 Loading validation data from {val_path}")
        data = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"✅ Loaded {len(data)} validation samples")
        return data

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate model on a single sample"""
        instruction = sample["instruction"]
        context = sample.get("context", "")
        reference_response = sample["response"]

        # Time the generation
        start_time = time.time()
        try:
            generated_response = self.llm.generate(instruction, context)
            success = True
        except Exception as e:
            generated_response = f"[ERROR: {str(e)}]"
            success = False
        latency = time.time() - start_time

        # Calculate ROUGE scores
        scores = self.scorer.score(reference_response, generated_response)

        return {
            "instruction": instruction,
            "context": context,
            "reference": reference_response,
            "generated": generated_response,
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
            "latency": latency,
            "success": success,
        }

    def evaluate(self, max_samples: int = 100) -> EvaluationResult:
        """Run full evaluation on validation set"""
        print("\n🔬 Starting model evaluation...\n")

        # Load validation data
        val_data = self.load_validation_data()

        # Limit samples if requested
        if max_samples and max_samples < len(val_data):
            print(f"   Evaluating on {max_samples} samples (from {len(val_data)} total)")
            val_data = val_data[:max_samples]

        # Evaluate each sample
        results = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        latencies = []
        successful = 0
        failed = 0

        for sample in tqdm(val_data, desc="Evaluating"):
            result = self.evaluate_sample(sample)
            results.append(result)

            if result["success"]:
                rouge1_scores.append(result["rouge1"])
                rouge2_scores.append(result["rouge2"])
                rougeL_scores.append(result["rougeL"])
                latencies.append(result["latency"])
                successful += 1
            else:
                failed += 1

        # Calculate averages
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return EvaluationResult(
            avg_rouge1=avg_rouge1,
            avg_rouge2=avg_rouge2,
            avg_rougeL=avg_rougeL,
            avg_latency=avg_latency,
            total_questions=len(val_data),
            successful=successful,
            failed=failed,
            samples=results[:10],  # Keep first 10 for inspection
        )

    def print_results(self, result: EvaluationResult):
        """Print evaluation results in a nice format"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)

        print(f"\n📈 Performance Metrics:")
        print(f"   ROUGE-1:  {result.avg_rouge1:.4f}")
        print(f"   ROUGE-2:  {result.avg_rouge2:.4f}")
        print(f"   ROUGE-L:  {result.avg_rougeL:.4f}")

        print(f"\n⚡ Speed:")
        print(f"   Avg latency: {result.avg_latency:.2f}s per question")

        print(f"\n✅ Success Rate:")
        print(f"   Successful: {result.successful}/{result.total_questions} ({result.successful/result.total_questions*100:.1f}%)")
        if result.failed > 0:
            print(f"   Failed: {result.failed}")

        print(f"\n📋 Sample Outputs:")
        print("-" * 60)

        for i, sample in enumerate(result.samples[:3], 1):
            print(f"\n{i}. Question: {sample['instruction'][:100]}...")
            print(f"   Reference: {sample['reference'][:150]}...")
            print(f"   Generated: {sample['generated'][:150]}...")
            print(f"   ROUGE-L: {sample['rougeL']:.3f} | Latency: {sample['latency']:.2f}s")

        print("\n" + "=" * 60)

    def save_results(self, result: EvaluationResult, output_path: Path):
        """Save detailed results to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "metrics": {
                "rouge1": result.avg_rouge1,
                "rouge2": result.avg_rouge2,
                "rougeL": result.avg_rougeL,
                "avg_latency": result.avg_latency,
            },
            "stats": {
                "total": result.total_questions,
                "successful": result.successful,
                "failed": result.failed,
                "success_rate": result.successful / result.total_questions,
            },
            "samples": result.samples,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_path}")

    def run_evaluation(self, max_samples: int = 100, save_path: Optional[Path] = None):
        """Run complete evaluation pipeline"""
        self.load_model()
        result = self.evaluate(max_samples)
        self.print_results(result)

        if save_path:
            self.save_results(result, save_path)

        return result


def main():
    """CLI for evaluation"""
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser(description="Evaluate ECUdapt LLM")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ecutuning-llm"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Path to save evaluation results"
    )

    args = parser.parse_args()

    # Run evaluation
    config = LLMConfig()
    evaluator = ModelEvaluator(args.model_path, config)
    evaluator.run_evaluation(args.max_samples, args.output)


if __name__ == "__main__":
    main()
