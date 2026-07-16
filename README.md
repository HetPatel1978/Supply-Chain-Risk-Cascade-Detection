# Supply Chain Risk Cascade Detection

End-to-end pipeline for extracting inter-company financial relations from SEC 10-K filings,
constructing a supply chain knowledge graph, and detecting multi-hop risk cascades via
retrieval-augmented generation.

## Pipeline Overview

1. **Fine-tune RoBERTa-base** on REFinD for financial relation extraction
2. **Zero-shot LLM baseline** via Groq API (Llama-3 / Mixtral)
3. **Rule-based baseline** using dependency patterns and keyword heuristics
4. **Knowledge graph construction** with NetworkX from extracted triples
5. **Multi-hop RAG** over the graph for risk cascade queries
6. **Evaluation** — intrinsic (precision/recall/F1 on RE) + extrinsic (RAG faithfulness, cascade hit-rate)

---

## Folder Structure

```
NLP_2/
├── data/
│   ├── raw/            # Original REFinD dataset files (do not modify)
│   ├── processed/      # Tokenized / label-encoded splits ready for training
│   └── sec_filings/    # Downloaded 10-K HTML/text files from SEC EDGAR
│
├── src/
│   ├── models/         # RoBERTa fine-tuning code (train.py, model.py, dataset.py)
│   ├── baselines/      # zero_shot_llm.py and rule_based.py
│   ├── graph/          # build_graph.py — triple → NetworkX KG construction
│   ├── rag/            # retrieval.py, generate.py — multi-hop RAG pipeline
│   └── evaluation/     # intrinsic.py (RE metrics), extrinsic.py (RAG metrics)
│
├── notebooks/          # Exploratory analysis, visualizations, demo walkthroughs
│
├── results/
│   ├── metrics/        # JSON/CSV files with evaluation scores
│   └── graphs/         # Exported graph files (.graphml, .png)
│
├── tests/              # Unit tests + one-line smoke tests per library
│
├── requirements.txt    # Pinned dependencies
├── .env.example        # Template for GROQ_API_KEY and other secrets
└── README.md
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Smoke-test all core libraries
python tests/smoke_test.py

# 4. Copy and fill in secrets
cp .env.example .env
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq Cloud API key for zero-shot LLM baseline |

## Key Dependencies

| Library | Role |
|---|---|
| `transformers` | RoBERTa fine-tuning and tokenization |
| `datasets` | REFinD dataset loading and preprocessing |
| `networkx` | Knowledge graph construction and traversal |
| `groq` | Zero-shot LLM baseline via Groq Cloud |
| `sentence-transformers` | Dense embeddings for RAG retrieval |
| `faiss-cpu` | Vector similarity search for retrieval |
| `sec-edgar-downloader` | Automated 10-K filing downloads |
