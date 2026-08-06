# Supply Chain Risk Cascade Detection

Turns unstructured SEC 10-K filings from major semiconductor companies into a
structured knowledge graph of supplier / customer / ownership / competitor
relationships, then uses graph traversal + RAG to answer questions like
*"If TSMC is disrupted, who is affected downstream?"* — with every claim
traceable back to a source sentence in a real filing.

## How it works

```
data/filings/*.txt (9 SEC 10-Ks)
        │
        ├─► Rule-based extraction ──┐
        │   (spaCy dependency        │
        │    parsing + triggers)     │
        │                            ├─► clean + merge ──► results/merged_triples.json
        ├─► Zero-shot LLM extraction │    (canonicalize,        │
        │   (Llama 3.1-8B via Groq)  │     dedupe, filter)      │
        └────────────────────────────┘                          ▼
                                                          networkx knowledge graph
                                                                  │
                                          BFS cascade traversal ──┤
                                          (retrieve_paths.py)     │
                                                                  ▼
                                          RAG context + Groq  ──► grounded, cited answer
                                          (build_context.py)
```

Two independent extractors read the same filings and their outputs are
merged — a rule-based spaCy pipeline (fast, deterministic, no API cost) and
a zero-shot LLM pipeline (Llama 3.1-8B via Groq, better recall on
non-formulaic phrasing). Agreement between them, plus a canonicalization +
quality-filter pass, is what keeps the final graph clean.

## Project structure

```
run_pipeline.py            End-to-end pipeline runner (4 steps)
data/filings/               9 SEC 10-K filings (Item 1/1A text), one per ticker
src/
  baselines/
    schema.py               Shared Triple dataclass + 6-relation vocabulary
    rule_based.py            spaCy dependency-parse extractor
    zero_shot_llm.py          Groq/Llama zero-shot extractor
  graph/
    clean_triples.py          Canonicalize, dedupe, quality-filter, merge
    build_graph.py             Build the networkx MultiDiGraph
    retrieve_paths.py           BFS cascade-path traversal + evidence retrieval
    visualize_paths.py           Render a cascade as a layered PNG
  rag/
    build_context.py            RAG: question → graph evidence → grounded answer
results/
  rule_based_triples.json      Raw rule-based extractions
  groq_triples.json             Raw LLM extractions
  groq_errors.json               LLM responses that failed to parse as JSON
  merged_triples.json             Final clean knowledge graph (74 triples, 48 companies)
  supply_chain_graph.json          Node/edge export
  supply_chain_graph_3d.html        Pre-built interactive 3D graph view
  kg_path_evaluation_results.csv    LLM-judge scores for cascade-path quality
  qa_evaluation_results.csv          LLM-judge scores for RAG answer quality
```

## Setup

Requires Python 3.13 and a free [Groq API key](https://console.groq.com/keys)
(used for the LLM extraction step and the RAG answer step; the pipeline still
runs without one, just skipping those steps).

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

## Usage

Run the full pipeline (extraction → clean/merge → graph build):

```bash
python run_pipeline.py                       # all 4 steps
python run_pipeline.py --skip-groq             # skip LLM extraction (no API key needed)
python run_pipeline.py --max-sentences 50       # cap sentences sent to Groq per filing
```

It's incremental/crash-safe — `groq_triples.json` tracks which filings are
already processed, so re-running only calls the API for new/unfinished
filings.

Ask a supply-chain risk question grounded in the graph:

```bash
python src/rag/build_context.py
# Enter a supply-chain risk question: How could a disruption at TSMC affect Huawei?
```

This finds the companies mentioned, retrieves cascade paths (up to 5 hops)
between them, builds a citation-constrained prompt from the graph evidence,
and asks Groq to answer — refusing to invent relationships not present in
the retrieved evidence. It also renders `results/affected_supply_chain.png`,
a layered diagram of the cascade.

## The knowledge graph, currently

| | |
|---|---|
| Companies (nodes) | 48 |
| Relationships (edges) | 74 |
| Source filings | 9 (AMAT, AMD, AVGO, INTC, KLAC, LRCX, MU, NVDA, QCOM — FY2023 10-Ks) |

Relation breakdown:

| Relation | Count | Meaning |
|---|---|---|
| `supplier_of` | 35 | head sells to / delivers to tail (also absorbs the logical inverse `customer_of`) |
| `subsidiary_of` | 14 | head is owned by / a division of tail |
| `competitor_of` | 12 | head competes with tail |
| `depends_on` | 10 | head critically relies on tail for key inputs |
| `partner_of` | 3 | head has a strategic alliance with tail |

Sample edges:

```
AMD        --depends_on--> TSMC
AMD        --depends_on--> GLOBALFOUNDRIES
ASML       --supplier_of--> KLA
Apple      --competitor_of--> Intel
Intel      --depends_on--> Samsung
Intel      --competitor_of--> AMD, NVIDIA, Qualcomm
```

Every edge carries `confidence`, `source_file`, and `source_sentence` —
the exact filing sentence it was extracted from — so any claim in the graph
can be traced back to source.

## Evaluation

Two Groq-judged (`openai/gpt-oss-120b` as judge) evaluation harnesses live in
`results/`:

- **`evaluate_knowledge_graph_paths.py`** — scores retrieved cascade paths
  (1–4 hops between fixed company pairs) on path continuity, evidence
  coverage, direction consistency, cascade relevance, and path quality.
  Current average: **8.0 / 10** across 5 paths.
- **`evaluate_question_answering-v2.py`** — scores RAG answers to 5 fixed
  questions (direct, two-hop, multi-hop, direction-reasoning, and
  insufficient-evidence cases) on context relevance, faithfulness,
  relationship direction, completeness, and answer relevance.
  Current average: **7.8 / 10** across 5 questions.

Both scripts require `GROQ_API_KEY` and write their results to the CSVs in
`results/`; re-run them after any change to the extraction/merge pipeline to
check for regressions.

## Notes on data quality

- 8 of the 9 filings are the standard SEC `Item 1 (Business)` /
  `Item 1A (Risk Factors)` sections. Intel's 10-K uses a non-standard,
  reorganized layout, so `INTC_2023.txt` instead covers its equivalent
  "Introduction to Our Business" / "Risk Factors" / "Sales and Marketing"
  sections pulled directly from its FY2023 10-K on SEC EDGAR.
- `clean_triples.py`'s `ANCHOR_COMPANIES` and noise/blacklist filters are
  hand-curated per filing — a genuinely new company or filing will likely
  surface extraction noise that needs a new filter entry before merging (see
  the `_NOISE`, `_NOT_COMPANIES`, `_CATEGORY_SUFFIXES` sets in that file).
- The rule-based extractor trades recall for precision (no API cost, fully
  deterministic); the LLM extractor trades the reverse. Only relationships
  where the non-anchor entity passes the quality gate survive into the final
  merged graph.
