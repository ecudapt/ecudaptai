#!/usr/bin/env python3
"""
Data Preparation for LLM Training
Converts forum posts into instruction-response pairs for fine-tuning
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from llm_config import LLMConfig


class DataPreparator:
    def __init__(self, config: LLMConfig):
        self.config = config
        random.seed(42)

    def create_instruction_pairs(self, forum_post: Dict) -> List[Dict]:
        """
        Convert a forum post into multiple instruction-response training pairs.

        Strategies:
        1. Question extraction from titles
        2. Context-based Q&A from discussions
        3. Technical explanation pairs
        """
        pairs = []
        text = forum_post.get("text", "")
        title = forum_post.get("title", "")
        url = forum_post.get("url", "")
        forum = forum_post.get("forum", "")

        # Strategy 1: Title as question
        if title and len(title) > 10:
            if any(q in title.lower() for q in ["?", "how", "what", "why", "where", "when", "which"]):
                pairs.append({
                    "instruction": title,
                    "context": f"From {forum} tuning discussion",
                    "response": self._extract_answer(text, title),
                    "metadata": {"url": url, "forum": forum, "type": "title_question"}
                })

        # Strategy 2: Extract technical concepts
        technical_pairs = self._extract_technical_concepts(text, forum)
        pairs.extend(technical_pairs)

        # Strategy 3: Problem-solution pairs
        if any(word in text.lower() for word in ["problem", "issue", "help", "fixed", "solved"]):
            problem_solution = self._extract_problem_solution(text, forum)
            if problem_solution:
                pairs.append(problem_solution)

        return pairs

    def _extract_answer(self, text: str, question: str, max_length: int = 500) -> str:
        """Extract relevant answer from text based on question"""
        # Simple extraction: first paragraph or first 500 chars
        sentences = text.split(". ")
        answer = ". ".join(sentences[:3])
        if len(answer) > max_length:
            answer = answer[:max_length] + "..."
        return answer if answer else text[:max_length]

    def _extract_technical_concepts(self, text: str, forum: str) -> List[Dict]:
        """Extract technical concept explanations from text"""
        pairs = []

        # Common ECU tuning concepts
        concepts = {
            "boost control": ["boost", "wastegate", "overboost"],
            "fuel mapping": ["fuel", "afr", "lambda", "injector"],
            "ignition timing": ["timing", "advance", "knock", "spark"],
            "MAF tuning": ["maf", "mass airflow", "airflow sensor"],
            "turbo lag": ["lag", "spool", "turbo response"],
            "dyno tuning": ["dyno", "dyno run", "power run"],
        }

        text_lower = text.lower()
        for concept, keywords in concepts.items():
            if any(kw in text_lower for kw in keywords):
                # Create instruction pair
                instruction = f"Explain {concept} in ECU tuning"
                response = self._extract_relevant_section(text, keywords)
                if response and len(response) > 50:
                    pairs.append({
                        "instruction": instruction,
                        "context": f"Based on {forum} forum discussion",
                        "response": response,
                        "metadata": {"forum": forum, "concept": concept, "type": "concept_explanation"}
                    })

        return pairs

    def _extract_relevant_section(self, text: str, keywords: List[str], window: int = 400) -> str:
        """Extract text section most relevant to keywords"""
        text_lower = text.lower()
        best_section = ""
        max_matches = 0

        # Sliding window approach
        words = text.split()
        for i in range(0, len(words), 50):
            section = " ".join(words[i:i + window])
            section_lower = section.lower()
            matches = sum(1 for kw in keywords if kw in section_lower)
            if matches > max_matches:
                max_matches = matches
                best_section = section

        return best_section.strip()

    def _extract_problem_solution(self, text: str, forum: str) -> Dict:
        """Extract problem-solution pairs from discussion threads"""
        text_lower = text.lower()

        # Look for problem indicators
        problem_markers = ["problem", "issue", "help", "trouble", "error"]
        solution_markers = ["fixed", "solved", "solution", "resolved", "worked"]

        has_problem = any(marker in text_lower for marker in problem_markers)
        has_solution = any(marker in text_lower for marker in solution_markers)

        if has_problem and has_solution:
            # Split text and try to identify problem and solution sections
            sentences = text.split(".")
            problem_text = []
            solution_text = []

            in_solution = False
            for sentence in sentences:
                sent_lower = sentence.lower()
                if any(marker in sent_lower for marker in solution_markers):
                    in_solution = True

                if in_solution:
                    solution_text.append(sentence)
                else:
                    problem_text.append(sentence)

            if problem_text and solution_text:
                problem = ". ".join(problem_text[:3]).strip()
                solution = ". ".join(solution_text[:3]).strip()

                return {
                    "instruction": f"How to resolve: {problem[:200]}",
                    "context": f"From {forum} troubleshooting discussion",
                    "response": solution,
                    "metadata": {"forum": forum, "type": "problem_solution"}
                }

        return None

    def prepare_training_data(self):
        """Main pipeline: Load forum data, create instruction pairs, split train/val"""
        print("[DATA] Starting data preparation for LLM training...")

        if not self.config.raw_data_path.exists():
            print(f"[ERROR] Raw data not found: {self.config.raw_data_path}")
            print("        Run data_cleaner.py first to create clean forum data")
            return

        # Load clean forum data
        print(f"[INFO] Loading clean forum data from {self.config.raw_data_path}")
        forum_posts = []
        with open(self.config.raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    forum_posts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"[INFO] Loaded {len(forum_posts)} forum posts")

        # Create instruction-response pairs
        print("\n[INFO] Creating instruction-response pairs...")
        all_pairs = []
        for post in tqdm(forum_posts, desc="Processing posts"):
            pairs = self.create_instruction_pairs(post)
            all_pairs.extend(pairs)

        print(f"[INFO] Created {len(all_pairs)} training pairs from {len(forum_posts)} posts")

        # Remove duplicates
        print("\n[INFO] Removing duplicate instructions...")
        unique_pairs = self._deduplicate(all_pairs)
        print(f"[INFO] Kept {len(unique_pairs)} unique pairs")

        # Train/validation split
        print(
            f"\n[INFO] Splitting data (train: {self.config.train_test_split * 100:.0f}%, "
            f"val: {(1 - self.config.train_test_split) * 100:.0f}%)"
        )
        random.shuffle(unique_pairs)
        split_idx = int(len(unique_pairs) * self.config.train_test_split)
        train_pairs = unique_pairs[:split_idx]
        val_pairs = unique_pairs[split_idx:]

        # Save training data
        print("\n[INFO] Saving training data...")
        self._save_jsonl(train_pairs, self.config.train_data_path)
        self._save_jsonl(val_pairs, self.config.val_data_path)

        print("\n[DONE] Data preparation complete!")
        print(f"       Training samples:   {len(train_pairs)}")
        print(f"       Validation samples: {len(val_pairs)}")
        print(f"       Train file:         {self.config.train_data_path}")
        print(f"       Validation file:    {self.config.val_data_path}")

        # Show sample
        if train_pairs:
            print("\n[SAMPLE] Training pair:")
            sample = train_pairs[0]
            print(f"   Instruction: {sample['instruction'][:100]}...")
            print(f"   Context:     {sample['context']}")
            print(f"   Response:    {sample['response'][:150]}...")

    def _deduplicate(self, pairs: List[Dict]) -> List[Dict]:
        """Remove duplicate instruction pairs"""
        seen_instructions = set()
        unique_pairs = []

        for pair in pairs:
            instruction = pair["instruction"].lower().strip()
            if instruction not in seen_instructions:
                seen_instructions.add(instruction)
                unique_pairs.append(pair)

        return unique_pairs

    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """Save data as JSONL file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    config = LLMConfig()
    preparator = DataPreparator(config)
    preparator.prepare_training_data()
