"""
Groq-based risk cascade narrative generator.
"""

import os
from groq import Groq

MODEL = "llama-3.1-8b-instant"


def _build_prompt(context: dict, scenario: str) -> str:
    start = context["start"]
    paths = context["cascade_paths"]
    sentences = context["source_sentences"]
    suppliers = context["suppliers"]

    # Format downstream cascade paths
    path_lines = []
    for p in paths:
        chain = " → ".join(p["path"])
        rels = " → ".join(e["relation"] for e in p["edges"])
        path_lines.append(f"  [{p['hops']} hop] {chain}  ({rels})")
    paths_str = "\n".join(path_lines) if path_lines else "  (no direct downstream customers found)"

    # Format upstream suppliers
    sup_lines = [f"  {s['company']} ({s['relation']})" for s in suppliers]
    sups_str = "\n".join(sup_lines) if sup_lines else "  (no supplier data found)"

    # Filing evidence
    sents_str = "\n".join(f"  • {s}" for s in sentences[:6]) if sentences else "  (none)"

    scenario_clause = f"\n\nDisruption scenario: {scenario}" if scenario.strip() else ""

    return f"""You are a supply chain risk analyst specializing in the semiconductor industry.{scenario_clause}

A major disruption has occurred at **{start}**.

DOWNSTREAM CASCADE — companies that depend on {start} (supply chain graph):
{paths_str}

UPSTREAM EXPOSURE — companies that {start} relies on:
{sups_str}

EVIDENCE from SEC 10-K filings:
{sents_str}

Write a concise risk cascade analysis (4-6 sentences).
- Name the specific companies at immediate risk and why.
- Describe the second-order effects if any.
- Mention which product lines or business segments are most exposed.
- Conclude with the overall severity (Low / Medium / High / Critical).
Do not repeat the data verbatim; synthesize into an analyst narrative."""


def generate_cascade_analysis(
    context: dict,
    scenario: str = "",
    api_key: str = "",
) -> str:
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "⚠️  GROQ_API_KEY not set — add it to your .env file to enable LLM analysis."

    client = Groq(api_key=key)
    prompt = _build_prompt(context, scenario)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=450,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"Generation error: {exc}"
