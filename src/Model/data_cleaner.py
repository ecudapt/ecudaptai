import re, json, html, time, traceback
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

RAW_DIR = Path("data/raw/")
OUTPUT_FILE = Path("data/clean/forum_posts_clean.jsonl")

# Use fast keyword-based filter first, ML for uncertain cases
USE_ML = True
ML_THRESHOLD = 0.6

RELEVANT_KEYWORDS = [
    "ecu", "tune", "tuning", "flash", "boost", "ignition", "fuel", "afr",
    "spark", "knock", "advance", "rpm", "engine", "injector", "dyno",
    "sensor", "turbo", "supercharger", "wastegate", "lambda", "airflow",
    "map sensor", "mhd", "hondata", "megasquirt", "cobb", "link g4",
    "openecu", "obd2", "fuel trims", "stage", "hp", "torque"
]

EXCLUDE_KEYWORDS = [
    "off topic", "classifieds", "for sale", "meet", "wheels", "tires",
    "paint", "audio", "detailing", "body kit", "suspension", "rims"
]

# ----------------------------------------------------------------------
# INITIALIZE MODEL
# ----------------------------------------------------------------------
classifier = None
if USE_ML:
    print("⚙️  Loading zero-shot relevance classifier (facebook/bart-large-mnli)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def keyword_relevant(text: str) -> bool:
    """Fast keyword filter for relevance."""
    t = text.lower()
    if any(bad in t for bad in EXCLUDE_KEYWORDS):
        return False
    return any(k in t for k in RELEVANT_KEYWORDS)


def clean_text(raw_html: str) -> str:
    """Strip HTML, quotes, and normalize whitespace."""
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(" ")
    text = html.unescape(text)
    text = re.sub(r"(Quote:|Originally Posted by).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ml_relevance(text: str, threshold=ML_THRESHOLD) -> bool:
    """ML zero-shot classification fallback."""
    try:
        labels = ["ECU tuning / engine tuning", "Not about tuning"]
        result = classifier(text[:2000], candidate_labels=labels)
        label, score = result["labels"][0], result["scores"][0]
        return label.lower().startswith("ecu") and score >= threshold
    except Exception as e:
        print(f"⚠️ ML relevance check failed: {e}")
        return keyword_relevant(text)


# ----------------------------------------------------------------------
# MAIN PROCESSING
# ----------------------------------------------------------------------

def process_raw_threads(input_dir: Path, output_file: Path, use_ml=True):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_raw = total_clean = total_kept = 0
    start_time = time.time()

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"❌ No .jsonl files found in {input_dir.resolve()}")
        return

    print(f"📂 Found {len(files)} raw forum files under {input_dir}")
    with open(output_file, "w", encoding="utf-8") as fout:
        for file in files:
            forum_name = file.stem.replace("_", ".")
            lines = list(open(file, "r", encoding="utf-8"))
            print(f"\n🧹 Cleaning forum: {forum_name} ({len(lines)} threads)")
            for line in tqdm(lines, desc=f"→ {forum_name}", ncols=90):
                total_raw += 1
                try:
                    obj = json.loads(line)
                    # Some spiders output {"url":..., "title":..., "posts":[{"author":...,"content":...}]}
                    posts = obj.get("posts", [])
                    if isinstance(posts, list):
                        joined = " ".join(p.get("content", "") for p in posts)
                    else:
                        joined = str(posts)

                    clean = clean_text(joined)
                    if len(clean.split()) < 25:
                        continue
                    total_clean += 1

                    # Relevance filtering
                    relevant = keyword_relevant(clean)
                    if not relevant and use_ml:
                        relevant = ml_relevance(clean)

                    if relevant:
                        fout.write(json.dumps({
                            "url": obj.get("url", ""),
                            "forum": forum_name,
                            "title": obj.get("title", ""),
                            "text": clean
                        }, ensure_ascii=False) + "\n")
                        total_kept += 1

                except Exception as e:
                    print(f"⚠️ Error parsing line in {file.name}: {e}")
                    traceback.print_exc()

    # --- Summary ---
    duration = time.time() - start_time
    print("\n──────────────────────────────")
    print(f"🧩 Total threads read: {total_raw}")
    print(f"🧼 Valid after cleaning: {total_clean}")
    print(f"✅ ECU-related threads kept: {total_kept}")
    print(f"💾 Output file: {output_file.resolve()}")
    print(f"⏱️ Duration: {duration:.1f}s ({total_kept/duration:.1f} items/sec)")
    print("──────────────────────────────")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    process_raw_threads(RAW_DIR, OUTPUT_FILE, use_ml=USE_ML)
