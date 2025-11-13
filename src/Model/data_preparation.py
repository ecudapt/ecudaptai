#!/usr/bin/env python3
"""
Data Preparation for LLM Training
Converts forum posts into instruction-response pairs for fine-tuning.
This version is noisy by design so you SEE what's happening.
"""

from __future__ import annotations
import os, sys, json, argparse, logging, traceback
from pathlib import Path
from typing import List, Dict, Tuple
import random
from tqdm import tqdm

# ---------- import LLMConfig robustly ----------
# Try same dir first; then package relative; then parent.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
try:
    from llm_config import LLMConfig
except Exception:
    try:
        from .llm_config import LLMConfig  # type: ignore
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.error("[IMPORT] Could not import LLMConfig. "
                      "Make sure llm_config.py is in the same folder or importable.")
        raise

# ---------- noisy logger ----------
def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    # force flush on every log line
    logging.getLogger().handlers[0].flush = sys.stdout.flush  # type: ignore

# ---------- DataPreparator (your improved version) ----------
class DataPreparator:
    def __init__(self, config: LLMConfig):
        self.config = config
        random.seed(42)

        self.solution_markers = {"fixed","solved","solution","resolve","resolved","workaround","it worked","ended up","root cause"}
        self.problem_markers  = {"problem","issue","help","trouble","error","fail","fails","failing","won't","cannot","doesn't"}
        self.tech_tokens = {
            "afr","lambda","boost","psi","bar","wastegate","map","maf","iac","ignition","timing","advance","spark",
            "knock","injector","duty","ms","rpm","load","cam","tps","o2","wideband","closed loop","open loop",
            "datalog","log","trim","fuel","table","hex",".xdf",".bin",".ori",".kp",".kp2",".xdfx",
            "stage","turbo","supercharger","haltech","cobb","hondata","megasquirt","link","mhd"
        }

    def _thread_text(self, forum_post: Dict) -> Tuple[str, str, str]:
        title = forum_post.get("title", "") or ""
        posts = forum_post.get("posts")
        if isinstance(posts, list) and posts:
            op_chunks, reply_chunks = [], []
            for i, p in enumerate(posts):
                body = (p.get("text") or p.get("content") or "").strip()
                if not body: continue
                (op_chunks if i == 0 else reply_chunks).append(body)
            op_text = "\n\n".join(op_chunks).strip()
            replies_text = "\n\n".join(reply_chunks).strip()
        else:
            t = forum_post.get("text", "") or ""
            op_text, replies_text = t, ""
        return title, op_text, replies_text

    def create_instruction_pairs(self, forum_post: Dict) -> List[Dict]:
        pairs = []
        forum = forum_post.get("forum", "") or ""
        url   = forum_post.get("url", "") or ""
        title, op_text, replies_text = self._thread_text(forum_post)

        if title and len(title) > 10 and any(k in title.lower() for k in ["?","how","what","why","where","when","which"]):
            answer = self._best_answer_for_question(title, op_text, replies_text)
            if self._is_acceptable(answer):
                pairs.append({
                    "instruction": self._normalize_question(title),
                    "context": f"From {forum} tuning discussion",
                    "response": self._tighten_answer(answer),
                    "metadata": {"url": url, "forum": forum, "type": "title_question"}
                })

        concept_pairs = self._extract_technical_concepts(op_text + "\n\n" + replies_text, forum)
        pairs.extend(concept_pairs)

        ps = self._extract_problem_solution(op_text, replies_text, forum)
        if ps: pairs.append(ps)

        pairs = [p for p in pairs if self._is_acceptable(p.get("response",""))]
        return pairs

    # --- internals (same as I sent previously, shortened slightly to fit here) ---
    import re as _re

    def _best_answer_for_question(self, question: str, op_text: str, replies_text: str) -> str:
        reply_blocks = [b.strip() for b in self._re.split(r"\n{2,}", replies_text or "") if b.strip()]
        candidates = []
        for block in reply_blocks:
            score = 0; L = block.lower()
            if any(m in L for m in self.solution_markers): score += 3
            tech_hits = sum(1 for t in self.tech_tokens if t in L); score += min(3, tech_hits)
            num_hits = len(self._re.findall(r"\b\d+(\.\d+)?\b", block)); score += min(2, num_hits // 2)
            if self._re.search(r"0x[0-9a-fA-F]+|\.bin|\.xdf|\.kp2?|\.ori", block): score += 2
            if "http://" in L or "https://" in L: score += 1
            if "```" in block or "\t" in block: score += 1
            length = len(block)
            if 200 <= length <= 1600: score += 2
            elif length < 120: score -= 2
            elif length > 2500: score -= 1
            candidates.append((score, block))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_score, top_block = candidates[0]
            if top_score >= 3 and len(top_block) >= 160:
                return self._tighten_answer(top_block)
        corpus = (op_text or "") + "\n\n" + (replies_text or "")
        return self._extractive_answer(corpus, question)

    def _extractive_answer(self, text: str, query: str, max_chars: int = 900) -> str:
        sents = self._re.split(r"(?<=[.!?])\s+", (text or "").strip())
        if not sents: return (text or "")[:max_chars]
        q_tokens = set(self._re.findall(r"[a-z0-9\.\-]+", query.lower()))
        def score(s: str) -> int:
            L = s.lower()
            toks = set(self._re.findall(r"[a-z0-9\.\-]+", L))
            overlap = len(q_tokens & toks)
            tech = sum(1 for t in self.tech_tokens if t in L)
            nums = len(self._re.findall(r"\b\d+(\.\d+)?\b", L))
            sol = any(m in L for m in self.solution_markers)
            return overlap + min(2, tech) + min(2, nums//2) + (3 if sol else 0)
        ranked = sorted(sents, key=score, reverse=True)
        out, total = [], 0
        for s in ranked:
            if len(s) < 20: continue
            out.append(s.strip()); total += len(s)
            if total >= max_chars: break
        return self._tighten_answer(" ".join(out))

    def _extract_technical_concepts(self, text: str, forum: str) -> List[Dict]:
        pairs = []
        concepts = {
            "boost control": ["boost","wastegate","overboost","psi","bar","duty"],
            "fuel mapping": ["fuel","afr","lambda","injector","trim","ve","table"],
            "ignition timing": ["timing","advance","knock","spark","btdc","degrees"],
            "MAF tuning": ["maf","mass airflow","airflow sensor","scaling"],
            "turbo lag": ["lag","spool","response","transient"],
            "dyno tuning": ["dyno","power run","pull","whp","wtq","rpm"],
        }
        L = (text or "").lower()
        for concept, kws in concepts.items():
            if any(k in L for k in kws):
                section = self._extract_relevant_section(text, kws)
                if self._is_acceptable(section, min_len=120):
                    pairs.append({
                        "instruction": f"Explain {concept} in ECU tuning",
                        "context": f"Based on {forum} forum discussion",
                        "response": self._tighten_answer(section),
                        "metadata": {"forum": forum, "concept": concept, "type": "concept_explanation"}
                    })
        return pairs

    def _extract_relevant_section(self, text: str, keywords: List[str], window_words: int = 220) -> str:
        words = (text or "").split()
        best, best_hits = "", -1
        for i in range(0, max(1, len(words) - window_words + 1), 40):
            section = " ".join(words[i:i+window_words])
            hits = sum(1 for kw in keywords if kw in section.lower())
            if hits > best_hits: best, best_hits = section, hits
        return best.strip()

    def _extract_problem_solution(self, op_text: str, replies_text: str, forum: str) -> Dict | None:
        opL = (op_text or "").lower(); repL = (replies_text or "").lower()
        if not (any(m in opL for m in self.problem_markers) or any(m in repL for m in self.problem_markers)):
            return None
        if not any(m in repL for m in self.solution_markers):
            return None
        op_sents = self._re.split(r"(?<=[.!?])\s+", op_text or "")
        prob_sents = [s for s in op_sents if any(m in s.lower() for m in self.problem_markers)]
        problem = " ".join(prob_sents[:3]).strip() or " ".join(op_sents[:3]).strip()
        answer = self._best_answer_for_question(problem or "How to resolve this issue", op_text, replies_text)
        if not self._is_acceptable(answer): return None
        instr = f"How do I resolve: {self._truncate(problem, 180)}"
        return {
            "instruction": instr,
            "context": f"From {forum} troubleshooting discussion",
            "response": self._tighten_answer(answer),
            "metadata": {"forum": forum, "type": "problem_solution"}
        }

    def _is_acceptable(self, s: str, min_len: int = 160) -> bool:
        if not s or not s.strip(): return False
        s = s.strip()
        if len(s) < min_len: return False
        if len(set(self._re.findall(r"[a-z]+", s.lower()))) < 20: return False
        return True

    def _normalize_question(self, q: str) -> str:
        q = q.strip()
        if not q.endswith("?") and not q.lower().startswith(("how","what","why","where","when","which")):
            q = q.rstrip(".") + "?"
        return q

    def _tighten_answer(self, s: str, max_len: int = 1200) -> str:
        s = self._re.sub(r"\n{3,}", "\n\n", s or "").strip()
        s = self._re.sub(r"[ \t]{2,}", " ", s)
        if len(s) > max_len: s = s[:max_len].rstrip() + "..."
        return s

    def _truncate(self, s: str, n: int) -> str:
        return (s[:n] + "...") if len(s) > n else s

    def prepare_training_data(self):
        logging.info("[DATA] Starting data preparation for LLM training...")

        if not self.config.raw_data_path.exists():
            logging.error(f"[ERROR] Raw data not found: {self.config.raw_data_path}")
            logging.error("        Run data_cleaner.py first to create clean forum data")
            return

        logging.info(f"[INFO] Loading clean forum data from {self.config.raw_data_path}")
        forum_posts = []
        with open(self.config.raw_data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    forum_posts.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.debug(f"[WARN] Skipped bad JSON line {i}")
        logging.info(f"[INFO] Loaded {len(forum_posts)} forum posts")

        logging.info("\n[INFO] Creating instruction-response pairs...")
        all_pairs = []
        for post in tqdm(forum_posts, desc="Processing posts"):
            try:
                pairs = self.create_instruction_pairs(post)
                all_pairs.extend(pairs)
            except Exception:
                logging.debug("[EXTRACT] error on a post:\n" + traceback.format_exc())

        logging.info(f"[INFO] Created {len(all_pairs)} training pairs from {len(forum_posts)} posts")

        logging.info("\n[INFO] Removing duplicate instructions...")
        unique_pairs = self._deduplicate(all_pairs)
        logging.info(f"[INFO] Kept {len(unique_pairs)} unique pairs")

        logging.info(
            f"\n[INFO] Splitting data (train: {self.config.train_test_split * 100:.0f}%, "
            f"val: {(1 - self.config.train_test_split) * 100:.0f}%)"
        )
        random.shuffle(unique_pairs)
        split_idx = int(len(unique_pairs) * self.config.train_test_split)
        train_pairs = unique_pairs[:split_idx]
        val_pairs = unique_pairs[split_idx:]

        logging.info("\n[INFO] Saving training data...")
        self._save_jsonl(train_pairs, self.config.train_data_path)
        self._save_jsonl(val_pairs, self.config.val_data_path)

        logging.info("\n[DONE] Data preparation complete!")
        logging.info(f"       Training samples:   {len(train_pairs)}")
        logging.info(f"       Validation samples: {len(val_pairs)}")
        logging.info(f"       Train file:         {self.config.train_data_path}")
        logging.info(f"       Validation file:    {self.config.val_data_path}")

        if train_pairs:
            sample = train_pairs[0]
            logging.info("\n[SAMPLE] Training pair:")
            logging.info(f"   Instruction: {sample['instruction'][:100]}...")
            logging.info(f"   Context:     {sample['context']}")
            logging.info(f"   Response:    {sample['response'][:150]}...")

    def _deduplicate(self, pairs: List[Dict]) -> List[Dict]:
        seen, uniq = set(), []
        for p in pairs:
            instr = (p.get("instruction","").lower().strip())
            resp  = self._re.sub(r"\s+", " ", (p.get("response","") or "").lower().strip())
            key = hash(instr + "|" + resp[:300])
            if key not in seen:
                seen.add(key); uniq.append(p)
        return uniq

    def _save_jsonl(self, data: List[Dict], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"[WRITE] {filepath}  ({len(data)} rows)")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Prepare instruction/response pairs for LLM fine-tuning")
    ap.add_argument("--raw", type=str, help="Path to clean forum JSONL (overrides config)")
    ap.add_argument("--train", type=str, help="Output train.jsonl (overrides config)")
    ap.add_argument("--val", type=str, help="Output val.jsonl (overrides config)")
    ap.add_argument("--split", type=float, default=None, help="Train split ratio, e.g. 0.9")
    ap.add_argument("--quiet", action="store_true", help="Less verbose logs")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(verbose=not args.quiet)
    logging.info("[BOOT] data_preparation.py starting...")

    cfg = LLMConfig()
    if args.raw:   cfg.raw_data_path = Path(args.raw)
    if args.train: cfg.train_data_path = Path(args.train)
    if args.val:   cfg.val_data_path = Path(args.val)
    if args.split is not None: cfg.train_test_split = float(args.split)

    logging.info(f"[CFG] raw={cfg.raw_data_path}")
    logging.info(f"[CFG] train={cfg.train_data_path}")
    logging.info(f"[CFG] val={cfg.val_data_path}")
    logging.info(f"[CFG] split={cfg.train_test_split}")

    try:
        prep = DataPreparator(cfg)
        prep.prepare_training_data()
    except Exception:
        logging.error("[FATAL] Unhandled error:\n" + traceback.format_exc())

if __name__ == "__main__":
    main()
