"""
Run both relation extractors over all 8 filing text files and save outputs.

Outputs:
  results/rule_based_triples.json
  results/groq_triples.json
  results/groq_errors.json   (sentences where Groq returned invalid JSON)
"""

import json
import pathlib
import sys
import os
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from baselines import rule_based, zero_shot_llm

FILINGS_DIR = pathlib.Path(__file__).parent.parent / "data" / "filings"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILING_FILES = sorted(FILINGS_DIR.glob("*_2023.txt"))


def run_rule_based():
    print("\n=== RULE-BASED EXTRACTION ===")
    all_triples = []
    for fp in FILING_FILES:
        triples = rule_based.extract_file(fp)
        print(f"  {fp.name}: {len(triples)} triples")
        all_triples.extend(t.to_dict() for t in triples)

    out_path = RESULTS_DIR / "rule_based_triples.json"
    # Also save to results/ root for easy access
    root_path = pathlib.Path(__file__).parent.parent / "results" / "rule_based_triples.json"
    payload = {"total": len(all_triples), "triples": all_triples}
    for p in (out_path, root_path):
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Total: {len(all_triples)} triples saved to {root_path}")
    return all_triples


def run_groq(max_sentences_per_file: int = 60):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n=== GROQ EXTRACTION: SKIPPED (GROQ_API_KEY not set) ===")
        return [], []

    print(f"\n=== GROQ EXTRACTION (max {max_sentences_per_file} sentences/file) ===")
    all_triples = []
    all_errors  = []

    for fp in FILING_FILES:
        triples, errors = zero_shot_llm.extract_file(fp, max_sentences=max_sentences_per_file)
        print(f"  {fp.name}: {len(triples)} triples | {len(errors)} parse errors")
        all_triples.extend(t.to_dict() for t in triples)
        all_errors.extend(errors)

    root = pathlib.Path(__file__).parent.parent / "results"
    (root / "groq_triples.json").write_text(
        json.dumps({"total": len(all_triples), "triples": all_triples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "groq_errors.json").write_text(
        json.dumps(all_errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "groq_triples.json").write_text(
        json.dumps({"total": len(all_triples), "triples": all_triples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  Total: {len(all_triples)} triples, {len(all_errors)} parse errors")
    print(f"  Saved to results/groq_triples.json and results/groq_errors.json")
    return all_triples, all_errors


if __name__ == "__main__":
    rb = run_rule_based()
    g, e = run_groq()
    print("\nDone.")
