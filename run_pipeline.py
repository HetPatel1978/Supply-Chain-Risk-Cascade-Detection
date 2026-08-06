"""
End-to-end pipeline runner for Supply Chain Risk Cascade Detection.

Steps
-----
1. Rule-based extraction  (spaCy + dependency patterns)
2. Zero-shot extraction   (LLaMA 3.1-8B via Groq)
3. Clean + merge triples  (canonicalization, anchor filter, dedup)
4. Launch dashboard       (Streamlit)

Usage
-----
    python run_pipeline.py            # runs steps 1-3, then prints launch cmd
    python run_pipeline.py --skip-groq  # skips step 2 (useful if no API key)
"""

import argparse
import json
import pathlib
import sys
import os

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FILINGS_DIR = ROOT / "data" / "filings"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def step1_rule_based():
    from baselines import rule_based

    print("\n" + "=" * 60)
    print("STEP 1 · Rule-based extraction (spaCy)")
    print("=" * 60)

    files = sorted(FILINGS_DIR.glob("*_2023.txt"))
    if not files:
        print(f"  No filing .txt files found in {FILINGS_DIR}")
        return []

    all_triples = []
    for fp in files:
        triples = rule_based.extract_file(fp)
        print(f"  {fp.name:<22} {len(triples):>4} triples")
        all_triples.extend(t.to_dict() for t in triples)

    out = RESULTS_DIR / "rule_based_triples.json"
    out.write_text(
        json.dumps({"total": len(all_triples), "triples": all_triples},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Total: {len(all_triples)} triples → {out.name}")
    return all_triples


def step2_groq(max_sentences: int = 100):
    from baselines import zero_shot_llm

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n" + "=" * 60)
        print("STEP 2 · Groq extraction  [SKIPPED — no GROQ_API_KEY]")
        print("=" * 60)
        print("  Add GROQ_API_KEY to your .env file to enable zero-shot extraction.")
        return []

    print("\n" + "=" * 60)
    print(f"STEP 2 · Zero-shot extraction (LLaMA 3.1-8B via Groq, max {max_sentences} sentences/file)")
    print("=" * 60)

    files = sorted(FILINGS_DIR.glob("*_2023.txt"))
    triples_path = RESULTS_DIR / "groq_triples.json"
    errors_path  = RESULTS_DIR / "groq_errors.json"

    # Resume: skip files already processed
    done_files: set[str] = set()
    all_triples: list[dict] = []
    all_errors:  list[dict] = []
    if triples_path.exists():
        existing = json.loads(triples_path.read_text(encoding="utf-8"))
        all_triples = existing.get("triples", [])
        done_files  = {t["source_file"] for t in all_triples}
        if done_files:
            print(f"  Resuming — {len(done_files)} file(s) already done")

    for fp in files:
        if fp.name in done_files:
            print(f"  {fp.name:<22} skipped (already done)")
            continue
        triples, errors = zero_shot_llm.extract_file(fp, max_sentences=max_sentences)
        for e in errors:
            e["source_file"] = fp.name
        print(f"  {fp.name:<22} {len(triples):>4} triples | {len(errors)} errors")
        all_triples.extend(t.to_dict() for t in triples)
        all_errors.extend(errors)

        # Save after every file (crash-safe)
        triples_path.write_text(
            json.dumps({"total": len(all_triples), "triples": all_triples},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        errors_path.write_text(
            json.dumps(all_errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\n  Total: {len(all_triples)} triples, {len(all_errors)} errors → {triples_path.name}")
    return all_triples


def step3_clean():
    print("\n" + "=" * 60)
    print("STEP 3 · Clean + merge triples")
    print("=" * 60)

    from graph.clean_triples import clean_and_merge
    triples = clean_and_merge()
    return triples


def step4_graph():
    print("\n" + "=" * 60)
    print("STEP 4 · Build knowledge graph")
    print("=" * 60)

    from graph.build_graph import build_graph, load_triples, MERGED_TRIPLES_PATH

    triples = load_triples(ROOT / "results" / "merged_triples.json")
    G = build_graph(triples)

    print(f"  Nodes (companies) : {G.number_of_nodes()}")
    print(f"  Edges (relations) : {G.number_of_edges()}")

    import collections
    rel_counts = collections.Counter(
        d["relation"] for _, _, d in G.edges(data=True)
    )
    print(f"\n  Relation breakdown:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel:<22} {cnt}")

    print(f"\n  Note: 3D graph HTML is pre-built at results/supply_chain_graph_3d.html")
    return G


def main():
    parser = argparse.ArgumentParser(description="Supply Chain Risk Pipeline")
    parser.add_argument("--skip-groq", action="store_true",
                        help="Skip the Groq zero-shot extraction step")
    parser.add_argument("--max-sentences", type=int, default=100,
                        help="Max sentences per file for Groq extraction (default: 100)")
    args = parser.parse_args()

    print("\n🏭  Supply Chain Risk Cascade Detection Pipeline")
    print("   " + "─" * 50)

    step1_rule_based()

    if not args.skip_groq:
        step2_groq(max_sentences=args.max_sentences)
    else:
        print("\n[STEP 2 skipped — --skip-groq flag set]")

    step3_clean()
    step4_graph()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print("\nOutputs saved to results/")
    print("  merged_triples.json       — clean knowledge graph triples")
    print("  supply_chain_graph_3d.html — interactive 3D visualization\n")


if __name__ == "__main__":
    main()
