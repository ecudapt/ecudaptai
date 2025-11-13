# src/Model/decisionify_dataset.py
#!/usr/bin/env python3
import json
from pathlib import Path

IN_PATH  = Path("data/training/train.jsonl")
OUT_PATH = Path("data/training/train_decision.jsonl")

def infer_decision(resp: str) -> str:
    r = resp.lower()
    if "knock" in r:
        return "Reduce ignition advance in the high-load band and slightly enrich lambda near peak torque."
    if "afr" in r or "lambda" in r or "lean" in r:
        return "Enrich lambda in the affected load/rpm band and re-check trims."
    if "overboost" in r or "boost error" in r:
        return "Lower boost target at high rpm and cap WGDC until error stabilizes."
    if "misfire" in r:
        return "Check coil/plug health and slightly reduce timing; review fueling at the affected RPM."
    return "Apply conservative bounded deltas to fueling/timing/boost based on logs and re-evaluate."

def infer_why(resp: str) -> str:
    r = resp.lower()
    bits=[]
    if "knock" in r: bits.append("Knock events observed at higher load")
    if "trim" in r:  bits.append("Fuel trims out of range")
    if "boost" in r and "error" in r: bits.append("Boost error suggests control saturation")
    if "misfire" in r: bits.append("Misfire indicates spark/fuel instability")
    if not bits: bits.append("Observed behavior suggests conservative adjustment is warranted")
    return "; ".join(bits) + "."

def main():
    kept=0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            resp = ex.get("response","")
            ex["decision"] = ex.get("decision") or infer_decision(resp)
            ex["why"] = ex.get("why") or infer_why(resp)
            ex["safety"] = ex.get("safety") or "Respect octane timing caps, lambda floors at high load, and WGDC limits."
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept+=1
    print(f"[OK] Wrote {OUT_PATH} ({kept} samples). "
          f"Point LLMConfig.train_data_path to this file or overwrite train.jsonl.")

if __name__ == "__main__":
    main()
