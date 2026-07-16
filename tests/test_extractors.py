"""
Smoke-test both extractors on 5 hand-picked sentences before the full filing run.

Sentences are chosen to:
  1. Match the user-specified AVGO sentence (self-description, tests LLM handling)
  2. Cover multiple relation types with explicitly named company entities
  3. Represent real sentences pulled from LRCX_2023.txt and NVDA_2023.txt
"""

import os, sys, json, pathlib, textwrap, spacy
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from baselines import rule_based, zero_shot_llm

# ── Test sentences ─────────────────────────────────────────────────────────
TEST_SENTENCES = [
    {
        "text": (
            "We are a global technology leader that designs, develops and supplies "
            "a broad range of semiconductor and infrastructure software solutions."
        ),
        "source_file": "AVGO_2023.txt",
        "note": "AVGO self-description (no explicit target entity — LLM vs rule-based contrast)",
    },
    {
        "text": (
            "Our most significant customers during the fiscal years ending June 25, 2023, "
            "June 26, 2022, and June 27, 2021 included Intel Corporation; Kioxia Corporation; "
            "Micron Technology, Inc.; Samsung Electronics Company, Ltd.; Taiwan Semiconductor "
            "Manufacturing Company, Limited; and SK hynix Inc."
        ),
        "source_file": "LRCX_2023.txt",
        "note": "LRCX customer list — customer_of / supplier_of signal with multiple ORG entities",
    },
    {
        "text": (
            "In the etch market, our primary competitors are Applied Materials, Inc.; "
            "Hitachi, Ltd.; and Tokyo Electron, Ltd., and our primary competitors in "
            "the deposition market include Applied Materials, Inc. and Tokyo Electron, Ltd."
        ),
        "source_file": "LRCX_2023.txt",
        "note": "LRCX competitor list — competitor_of with multiple explicit ORG entities",
    },
    {
        "text": (
            "We utilize suppliers, such as Taiwan Semiconductor Manufacturing Company "
            "Limited and Samsung Electronics Co., Ltd., to manufacture our products."
        ),
        "source_file": "NVDA_2023.txt",
        "note": "NVDA foundry relationships — customer_of / depends_on TSMC and Samsung",
    },
    {
        "text": (
            "In February 2022, NVIDIA and SoftBank Group Corp. announced the termination "
            "of the Share Purchase Agreement whereby NVIDIA would have acquired Arm Limited "
            "from SoftBank."
        ),
        "source_file": "NVDA_2023.txt",
        "note": "NVDA/Arm deal — subsidiary_of signal between explicitly named entities",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def show_triples(triples, label: str):
    if not triples:
        print(f"  [{label}] (no triples extracted)")
    for t in triples:
        print(f"  [{label}] {t}")


def show_ner(text: str):
    """Show what spaCy finds as named entities (for debugging)."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    ents = [(e.text, e.label_) for e in doc.ents if e.label_ in ("ORG","GPE","PRODUCT")]
    print(f"  [NER] {ents}")


# ── Run rule-based extractor ───────────────────────────────────────────────

def run_rule_based():
    section("RULE-BASED EXTRACTOR")
    all_triples = []
    for i, item in enumerate(TEST_SENTENCES, 1):
        print(f"\n[Sent {i}] {item['note']}")
        print(f"  TEXT: {textwrap.shorten(item['text'], 110)}")
        show_ner(item["text"])
        triples = rule_based.extract_sentence(
            item["text"],
            source_file=item["source_file"],
            source_sentence=item["text"],
        )
        show_triples(triples, "RB")
        all_triples.extend(t.to_dict() for t in triples)
    return all_triples


# ── Run Groq LLM extractor ────────────────────────────────────────────────

def run_groq():
    section("GROQ ZERO-SHOT EXTRACTOR  (llama-3.3-70b-versatile)")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  GROQ_API_KEY not set — skipping Groq tests.")
        print("  Set it with:  $env:GROQ_API_KEY='gsk_...'")
        return []

    all_triples = []
    for i, item in enumerate(TEST_SENTENCES, 1):
        print(f"\n[Sent {i}] {item['note']}")
        print(f"  TEXT: {textwrap.shorten(item['text'], 110)}")
        triples, err = zero_shot_llm.extract_sentence(
            item["text"],
            source_file=item["source_file"],
            source_sentence=item["text"],
        )
        if err:
            print(f"  [GROQ] PARSE ERROR: {err}")
        show_triples(triples, "GROQ")
        all_triples.extend(t.to_dict() for t in triples)
    return all_triples


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    rb_triples  = run_rule_based()
    llm_triples = run_groq()

    section("SUMMARY")
    print(f"  Rule-based : {len(rb_triples)} triples from {len(TEST_SENTENCES)} sentences")
    print(f"  Groq LLM   : {len(llm_triples)} triples from {len(TEST_SENTENCES)} sentences")
    print()
