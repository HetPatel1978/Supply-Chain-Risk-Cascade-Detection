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


def run_groq(max_sentences_per_file: int = 60, resume: bool = True):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n=== GROQ EXTRACTION: SKIPPED (GROQ_API_KEY not set) ===")
        return [], []

    root = pathlib.Path(__file__).parent.parent / "results"
    triples_path = root / "groq_triples.json"
    errors_path  = root / "groq_errors.json"

    # Load existing results so we can resume without re-processing done files
    done_files: set[str] = set()
    all_triples: list[dict] = []
    all_errors:  list[dict] = []
    if resume and triples_path.exists():
        existing = json.loads(triples_path.read_text(encoding="utf-8"))
        all_triples = existing.get("triples", [])
        done_files  = {t["source_file"] for t in all_triples}
    if resume and errors_path.exists():
        existing_errors = json.loads(errors_path.read_text(encoding="utf-8"))
        # Only keep errors from already-done files so we don't lose them on resume
        all_errors = [e for e in existing_errors if e.get("source_file", "") in done_files]

    print(f"\n=== GROQ EXTRACTION (max {max_sentences_per_file} sentences/file, model={zero_shot_llm.MODEL}) ===")
    if done_files:
        print(f"  Resuming — skipping already-done files: {sorted(done_files)}")

    for fp in FILING_FILES:
        if fp.name in done_files:
            print(f"  {fp.name}: skipped (already done)")
            continue
        triples, errors = zero_shot_llm.extract_file(fp, max_sentences=max_sentences_per_file)
        # Tag errors with source file for resume tracking
        for e in errors:
            e["source_file"] = fp.name
        print(f"  {fp.name}: {len(triples)} triples | {len(errors)} errors")
        all_triples.extend(t.to_dict() for t in triples)
        all_errors.extend(errors)

        # Save after every file so a mid-run crash doesn't lose everything
        triples_path.write_text(
            json.dumps({"total": len(all_triples), "triples": all_triples}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        errors_path.write_text(
            json.dumps(all_errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (RESULTS_DIR / "groq_triples.json").write_text(
        json.dumps({"total": len(all_triples), "triples": all_triples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  Total: {len(all_triples)} triples, {len(all_errors)} errors")
    print(f"  Saved to results/groq_triples.json and results/groq_errors.json")
    return all_triples, all_errors


if __name__ == "__main__":
    rb = run_rule_based()
    g, e = run_groq(max_sentences_per_file=100)
    print("\nDone.")
