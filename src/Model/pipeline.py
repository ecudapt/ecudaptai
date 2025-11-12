#!/usr/bin/env python3
"""
End-to-End LLM Pipeline Orchestrator
Automates the complete workflow from raw data to trained model
"""
import sys
import subprocess
from pathlib import Path
from typing import Optional
import time

from llm_config import LLMConfig, LightweightConfig

# Base dirs (independent of where you run Python from)
MODEL_DIR = Path(__file__).resolve().parent          # .../src/Model
SRC_DIR   = MODEL_DIR.parent                         # .../src
SCRAPER_RAW_DIR = SRC_DIR / "Scraper" / "data" / "raw"

class LLMPipeline:
    """Orchestrates the complete LLM training pipeline"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.steps_completed = []

    def print_header(self, step: str, step_num: int, total_steps: int):
        """Print formatted step header"""
        print("\n" + "=" * 70)
        print(f"STEP {step_num}/{total_steps}: {step}")
        print("=" * 70 + "\n")

    def run_step(self, name: str, func, *args, **kwargs):
        """Run a pipeline step with error handling"""
        start_time = time.time()
        try:
            print(f"[START] {name}")
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"[OK]    {name} (took {duration:.1f}s)")
            self.steps_completed.append(name)
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"[FAIL]  {name} (after {duration:.1f}s)")
            print(f"        Error: {e}")
            raise

    def check_prerequisites(self):
        """Check if required files and packages exist"""
        print("[CHECK] Checking prerequisites...")

        # Check for scraped data
        scraper_data = SCRAPER_RAW_DIR
        if not scraper_data.exists() or not list(scraper_data.glob("*.jsonl")):
            print("[WARN] No scraped data found!")
            print(f"   Expected location: {scraper_data.resolve()}")
            print("   Run the scraper first: cd src/Scraper && python <your run script>")
            return False

        # Check Python packages
        required_packages = [
            "transformers",
            "peft",
            "datasets",
            "torch",
            "accelerate",
            "bitsandbytes",
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            print(f"[FAIL] Missing required packages: {', '.join(missing)}")
            print("\nInstall with:")
            print(f"   pip install {' '.join(missing)}")
            return False

        print("[OK] All prerequisites satisfied")
        return True

    def step1_clean_data(self):
        """Step 1: Clean and filter scraped forum data"""
        from data_cleaner import process_raw_threads

        raw_dir = SCRAPER_RAW_DIR
        output_file = self.config.raw_data_path  # data/clean/forum_posts_clean.jsonl

        process_raw_threads(raw_dir, output_file, use_ml=False)

    def step2_prepare_training_data(self):
        """Step 2: Convert forum posts to instruction-response pairs"""
        from data_preparation import DataPreparator

        preparator = DataPreparator(self.config)
        preparator.prepare_training_data()

    def step3_train_model(self):
        """Step 3: Fine-tune the LLM"""
        from train_llm import ECUTuningTrainer

        trainer = ECUTuningTrainer(self.config)
        trainer.run()

    def step4_evaluate_model(self):
        """Step 4: Evaluate trained model"""
        from evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config.output_dir, self.config)
        evaluator.run_evaluation(
            max_samples=50,
            save_path=self.config.output_dir / "evaluation_results.json"
        )

    def step5_create_embeddings(self):
        """Step 5: Create vector embeddings for RAG"""
        from embedder import build_index

        build_index(
            self.config.raw_data_path,
            Path("data/vector/forum.faiss")
        )

    def run_full_pipeline(self):
        """Execute complete pipeline from start to finish"""
        print("\n[PIPELINE] ECUdapt AI - Full LLM Training Pipeline")
        print(f"   Base model: {self.config.base_model}")
        print(f"   Output directory: {self.config.output_dir}")

        # Check prerequisites
        if not self.check_prerequisites():
            print("\n[ABORT] Prerequisites not met. Exiting.")
            sys.exit(1)

        total_steps = 5
        current_step = 0

        try:
            # Step 1: Clean data
            current_step += 1
            self.print_header("Clean Forum Data", current_step, total_steps)
            self.run_step("Data Cleaning", self.step1_clean_data)

            # Step 2: Prepare training data
            current_step += 1
            self.print_header("Prepare Training Data", current_step, total_steps)
            self.run_step("Training Data Preparation", self.step2_prepare_training_data)

            # Step 3: Train model
            current_step += 1
            self.print_header("Train LLM", current_step, total_steps)
            self.run_step("Model Training", self.step3_train_model)

            # Step 4: Evaluate
            current_step += 1
            self.print_header("Evaluate Model", current_step, total_steps)
            self.run_step("Model Evaluation", self.step4_evaluate_model)

            # Step 5: Create embeddings for RAG
            current_step += 1
            self.print_header("Create Vector Embeddings", current_step, total_steps)
            self.run_step("Embedding Generation", self.step5_create_embeddings)

            # Success!
            print("\n" + "=" * 70)
            print("[DONE] PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\n[SUMMARY] Steps completed: {len(self.steps_completed)}/{total_steps}")
            for i, step in enumerate(self.steps_completed, 1):
                print(f"   {i}. {step}")

            print(f"\n[MODEL] Trained model is at:")
            print(f"   {self.config.output_dir.resolve()}")

            print(f"\n[USAGE] To use your model:")
            print(f"   python inference.py --model-path {self.config.output_dir}")

            print(f"\n[USAGE] To start interactive chat:")
            print("   python inference.py")

        except Exception as e:
            print("\n" + "=" * 70)
            print("[ERROR] PIPELINE FAILED")
            print("=" * 70)
            print(f"\nCompleted {len(self.steps_completed)}/{total_steps} steps:")
            for i, step in enumerate(self.steps_completed, 1):
                print(f"   [OK] {i}. {step}")

            print(f"\n[ERROR] Failed at: Step {current_step}")
            print(f"        Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """CLI for pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run complete ECUdapt LLM training pipeline"
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight config (smaller model, less memory)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override base model name"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["clean", "prepare", "train", "evaluate", "embed"],
        help="Run only a specific step (default: all)"
    )

    args = parser.parse_args()

    # Select configuration
    if args.lightweight:
        print("[CONFIG] Using lightweight configuration")
        config = LightweightConfig()
    else:
        config = LLMConfig()

    if args.model:
        config.base_model = args.model

    # Create pipeline
    pipeline = LLMPipeline(config)

    # Run specific step or full pipeline
    if args.step:
        step_map = {
            "clean": (pipeline.step1_clean_data, "Clean Forum Data"),
            "prepare": (pipeline.step2_prepare_training_data, "Prepare Training Data"),
            "train": (pipeline.step3_train_model, "Train LLM"),
            "evaluate": (pipeline.step4_evaluate_model, "Evaluate Model"),
            "embed": (pipeline.step5_create_embeddings, "Create Embeddings"),
        }

        func, name = step_map[args.step]
        print(f"\n[STEP] Running single step: {name}\n")
        pipeline.run_step(name, func)
    else:
        # Run full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
