"""
Zero-shot relation extractor using the Groq API (llama-3.3-70b-versatile).

Each sentence is sent to the LLM with a structured prompt that asks for:
  - Named entity recognition (especially organizations)
  - Relation extraction over the fixed 6-relation vocabulary

The model is required to return strict JSON. Parse failures are captured
and returned as-is so they are visible in evaluation.
"""

import os
import re
import json
import time
import pathlib
import textwrap
from typing import Any

from groq import Groq
from .schema import Triple, RELATION_VOCAB, normalize_triples

# ---------------------------------------------------------------------------
# Client (lazy init so missing API key doesn't crash at import time)
# ---------------------------------------------------------------------------
_CLIENT: Groq | None = None

# llama-3.1-8b-instant: 500k tokens/day free tier (vs 100k for 70b)
MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 5
RETRY_DELAY = 2.0          # base delay; overridden by 429 wait time when available
INTER_REQUEST_DELAY = 0.3  # polite pause between calls

_WAIT_RE = re.compile(r"try again in ([\d]+)m([\d.]+)s")


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file and run:\n"
                "  $env:GROQ_API_KEY='gsk_...'"
            )
        _CLIENT = Groq(api_key=key)
    return _CLIENT


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a financial relation extraction system. Your ONLY output is valid JSON.
    Given a sentence from a corporate SEC 10-K filing, identify named-entity pairs
    that have a supply-chain or corporate relationship.

    ── RELATION DEFINITIONS (head, relation, tail direction is STRICT) ──────────

    supplier_of(head, tail)
      head SELLS TO / DELIVERS TO tail.  head=seller, tail=buyer.
      Example: "TSMC manufactures chips for NVIDIA"
        → {"head":"TSMC","relation":"supplier_of","tail":"NVIDIA","confidence":0.95}

    customer_of(head, tail)
      head PURCHASES FROM tail.  head=buyer, tail=seller.
      Example: "NVIDIA buys chips from TSMC"
        → {"head":"NVIDIA","relation":"customer_of","tail":"TSMC","confidence":0.95}
      WARNING: "Our customers include Intel" means Intel is the buyer — Intel is
      the customer OF us, so: {"head":"Intel","relation":"customer_of","tail":"<filing company>"}

    subsidiary_of(head, tail)
      head IS OWNED BY / IS A DIVISION OF tail.  head=child, tail=parent.
      Example: "Arm Limited was a subsidiary of SoftBank"
        → {"head":"Arm Limited","relation":"subsidiary_of","tail":"SoftBank","confidence":0.95}

    competitor_of(head, tail)
      head COMPETES WITH tail in the same market.
      Example: "AMD competes with Intel in CPUs"
        → {"head":"AMD","relation":"competitor_of","tail":"Intel","confidence":0.90}

    partner_of(head, tail)
      head HAS A STRATEGIC ALLIANCE OR PARTNERSHIP WITH tail.
      Example: "NVIDIA partners with Microsoft on AI infrastructure"
        → {"head":"NVIDIA","relation":"partner_of","tail":"Microsoft","confidence":0.90}

    depends_on(head, tail)
      head CRITICALLY RELIES ON tail for key inputs, components, or services.
      Example: "Apple depends on TSMC for its most advanced chips"
        → {"head":"Apple","relation":"depends_on","tail":"TSMC","confidence":0.90}

    ── RULES ───────────────────────────────────────────────────────────────────

    1. Resolve pronouns: "we", "our", "the Company", "the Corporation" → FILING COMPANY.
    2. Only extract relationships EXPLICITLY stated; do not infer.
    3. If a relationship exists but does NOT clearly fit one of the six types above
       (e.g., a terminated deal, legal dispute, regulatory action, investment, or
       historical event with no current supply/ownership link), OMIT IT.
       Do NOT force an ambiguous relationship into the nearest label.
    4. If no relationship qualifies, return {"triples": []}.
    5. confidence: float 0.0–1.0 reflecting certainty about BOTH the entities and
       the relation direction.

    ── OUTPUT FORMAT (STRICT — no markdown, no extra keys, no explanation) ─────
    {"triples":[{"head":"...","relation":"...","tail":"...","confidence":0.0}]}
""")

USER_TEMPLATE = textwrap.dedent("""\
    Filing company: {company}
    Sentence: {sentence}
""")


# ---------------------------------------------------------------------------
# Ticker → company name (mirrors rule_based.py)
# ---------------------------------------------------------------------------
TICKER_NAMES: dict[str, str] = {
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA Corporation",
    "MU":   "Micron Technology",
    "NVDA": "NVIDIA",
    "AMD":  "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "QCOM": "Qualcomm",
    "INTC": "Intel",
}


def _filer_name(source_file: str) -> str:
    stem = pathlib.Path(source_file).stem
    ticker = stem.split("_")[0].upper()
    return TICKER_NAMES.get(ticker, "Unknown Company")


# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_FIRST_BRACE = re.compile(r"\{", re.DOTALL)


def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract the first JSON object."""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1)
    start = _FIRST_BRACE.search(raw)
    if start:
        return raw[start.start():]
    return raw


def _parse_response(raw: str) -> tuple[list[dict], str | None]:
    """
    Try to parse the model response as JSON.
    Returns (triples_list, error_message).
    On success error_message is None; on failure triples_list is [].
    """
    cleaned = _clean_json(raw.strip())
    try:
        obj = json.loads(cleaned)
        return obj.get("triples", []), None
    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}  |  raw={raw!r}"


# ---------------------------------------------------------------------------
# Single-sentence extractor
# ---------------------------------------------------------------------------

def extract_sentence(
    sentence: str,
    source_file: str = "",
    source_sentence: str = "",
) -> tuple[list[Triple], str | None]:
    """
    Call the LLM and return (triples, parse_error).
    parse_error is None on success; the raw error string on failure.
    """
    client = _get_client()
    company = _filer_name(source_file) if source_file else "Unknown Company"
    user_msg = USER_TEMPLATE.format(company=company, sentence=sentence)

    raw_response = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw_response = resp.choices[0].message.content or ""
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < MAX_RETRIES - 1:
                # Parse "try again in Xm Y.Zs" from the error message
                m = _WAIT_RE.search(err_str)
                if m:
                    wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
                else:
                    wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [rate limit] waiting {wait:.0f}s before retry {attempt+1}/{MAX_RETRIES-1}...")
                time.sleep(wait)
                continue
            return [], f"API error: {e}"

    time.sleep(INTER_REQUEST_DELAY)

    raw_triples, parse_error = _parse_response(raw_response)
    triples: list[Triple] = []

    for t in raw_triples:
        rel = t.get("relation", "")
        if rel not in RELATION_VOCAB:
            # Log but don't drop — keep with relation flagged
            rel = f"INVALID:{rel}"
        head = str(t.get("head", "")).strip()
        tail = str(t.get("tail", "")).strip()
        conf = float(t.get("confidence", 0.5))
        if head and tail:
            triples.append(Triple(
                head=head, relation=rel, tail=tail,
                confidence=min(max(conf, 0.0), 1.0),
                source_file=source_file,
                source_sentence=source_sentence or sentence,
            ))

    return normalize_triples(triples), parse_error


# ---------------------------------------------------------------------------
# File-level extractor
# ---------------------------------------------------------------------------
# We filter sentences by relevance before sending to the API to stay within
# free-tier rate limits and avoid burning tokens on boilerplate.

_RELEVANCE_RE = re.compile(
    r"\b(suppli|customer|purchas|partner|subsidiar|compet|acqui|depend|rely|"
    r"manufactur|vendor|source|client|alliance|joint venture)\b",
    re.IGNORECASE,
)


def sentence_filter_stats(
    filepath: str | pathlib.Path,
) -> tuple[int, int, list[str]]:
    """
    Return (total_sentences, relevant_count, relevant_sentences) for a filing
    without making any API calls.  Used for the pre-filter audit table.
    """
    filepath = pathlib.Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    text = re.sub(r"=== ITEM \d+[A-Z]?:.*?===", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [
        s.strip() for s in re.split(r"(?<=[a-z0-9\"\')])\.\s+(?=[A-Z\"])", text)
        if 40 < len(s.strip()) < 600
    ]
    relevant = [s for s in sentences if _RELEVANCE_RE.search(s)]
    return len(sentences), len(relevant), relevant


def extract_file(
    filepath: str | pathlib.Path,
    max_sentences: int = 80,
    print_stats: bool = False,
) -> tuple[list[Triple], list[dict]]:
    """
    Extract triples from a filing. Returns (triples, error_log).
    error_log entries: {"sentence": ..., "error": ...}
    Triples are normalized (customer_of → supplier_of) before returning.
    """
    filepath = pathlib.Path(filepath)
    total, n_relevant, relevant_sents = sentence_filter_stats(filepath)
    capped = relevant_sents[:max_sentences]

    if print_stats:
        pct = 100 * n_relevant / total if total else 0
        capped_note = f" (capped at {max_sentences})" if n_relevant > max_sentences else ""
        print(
            f"  {filepath.name:<20}  total={total:4d}  "
            f"relevant={n_relevant:4d} ({pct:5.1f}%){capped_note}  "
            f"sending={len(capped):4d}"
        )

    all_triples: list[Triple] = []
    error_log: list[dict] = []

    for sent in capped:
        triples, err = extract_sentence(
            sent,
            source_file=filepath.name,
            source_sentence=sent,
        )
        all_triples.extend(triples)
        if err:
            error_log.append({"sentence": sent, "error": err})

    return normalize_triples(all_triples), error_log
