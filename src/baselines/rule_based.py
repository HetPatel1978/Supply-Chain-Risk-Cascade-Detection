"""
Rule-based relation extractor using spaCy dependency parsing + keyword patterns.

Strategy
--------
For each sentence:
  1. Parse the ORIGINAL text with spaCy so POS tags are clean.
  2. Collect NER entities (ORG / GPE / PRODUCT).
  3. Map first-person pronouns (we / our / us) to the filing company at the
     entity level — not in the raw text, which would corrupt POS tagging.
  4. Scan every token for relation-trigger lemmas (verbs & nouns).
  5. Use the dependency tree to find grammatical subject + object, then verify
     each against the NER entity list.  Only accept bare-noun fallbacks for
     entities we cannot find in NER, and only when at least ONE side of the
     triple is a recognised entity.
  6. Deduplicate by (head, relation, tail).
"""

import re
import pathlib
import spacy
from typing import Iterator

from .schema import Triple, RELATION_VOCAB, normalize_triples

# ---------------------------------------------------------------------------
# spaCy model (loaded once)
# ---------------------------------------------------------------------------
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------------------
# Company identity map
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
}

# Tokens that refer to the filing company
SELF_TOK = {"we", "our", "us", "the company", "the corporation", "the firm",
            "company", "corporation"}

# Tokens to never accept as entity heads/tails
SKIP_TOK = {"that", "it", "they", "this", "which", "who", "its", "their",
            "he", "she", "a", "an", "the", "and", "or", "but"}


def _filer_name(source_file: str) -> str | None:
    stem = pathlib.Path(source_file).stem
    ticker = stem.split("_")[0].upper()
    return TICKER_NAMES.get(ticker)


# ---------------------------------------------------------------------------
# Relation trigger vocabulary
# ---------------------------------------------------------------------------
TRIGGERS: list[dict] = [
    {
        "relation": "supplier_of",
        "verbs": {"supply", "provide", "manufacture", "produce", "make", "deliver",
                  "sell", "offer", "ship", "distribute", "fabricate"},
        "nouns": {"supplier", "vendor", "manufacturer", "producer", "maker",
                  "provider", "distributor", "foundry"},
        "confidence_verb": 0.75,
        "confidence_noun": 0.85,
        "subj_is_head": True,
        "obj_preps": {"to", "for"},   # preps that lead to the CUSTOMER entity
        "entity_pref": "ORG",
    },
    {
        "relation": "customer_of",
        "verbs": {"purchase", "buy", "procure", "adopt", "utilize", "use",
                  "source"},
        "nouns": {"customer", "buyer", "purchaser", "client"},
        "confidence_verb": 0.70,
        "confidence_noun": 0.85,
        "subj_is_head": True,
        "obj_preps": {"from", "of"},
        "entity_pref": "ORG",
    },
    {
        "relation": "subsidiary_of",
        "verbs": {"acquire", "own", "control", "absorb"},
        "nouns": {"subsidiary", "division", "affiliate", "acquisition"},
        "confidence_verb": 0.65,
        "confidence_noun": 0.90,
        "subj_is_head": False,   # verb-subj is the PARENT → swap
        "obj_preps": {"of"},
        "entity_pref": "ORG",
    },
    {
        "relation": "competitor_of",
        "verbs": {"compete", "rival"},
        "nouns": {"competitor", "rival", "competition"},
        "confidence_verb": 0.80,
        "confidence_noun": 0.80,
        "subj_is_head": True,
        "obj_preps": {"against", "with", "of"},
        "entity_pref": "ORG",
    },
    {
        "relation": "partner_of",
        "verbs": {"partner", "collaborate", "cooperate", "team"},
        "nouns": {"partner", "partnership", "alliance", "collaboration",
                  "joint venture"},
        "confidence_verb": 0.72,
        "confidence_noun": 0.82,
        "subj_is_head": True,
        "obj_preps": {"with", "of"},
        "entity_pref": "ORG",
    },
    {
        "relation": "depends_on",
        "verbs": {"depend", "rely", "hinge", "require"},
        "nouns": {"dependence", "reliance", "dependency"},
        "confidence_verb": 0.78,
        "confidence_noun": 0.82,
        "subj_is_head": True,
        "obj_preps": {"on"},
        "entity_pref": "ORG",
    },
]

# Fast lookup dicts
_VERB_MAP: dict[str, list[dict]] = {}
_NOUN_MAP: dict[str, list[dict]] = {}
for _t in TRIGGERS:
    for v in _t["verbs"]:
        _VERB_MAP.setdefault(v, []).append(_t)
    for n in _t["nouns"]:
        _NOUN_MAP.setdefault(n, []).append(_t)


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------
ENTITY_LABELS = {"ORG", "GPE", "PRODUCT"}


def _build_ent_set(doc, filer: str | None) -> tuple[list[tuple], set[str]]:
    """
    Returns:
      ents   – list of (text, start_char, end_char, label)
      ent_texts – set of entity texts (lower-cased) for O(1) lookup

    Filters out junk NER spans: very long spans or spans containing legal/
    document keywords that spaCy mis-classifies as ORG.
    """
    JUNK_RE = re.compile(
        r"\b(agreement|termination|amendment|whereby|schedule|exhibit|section"
        r"|fiscal year|table of contents|10-K|form|report)\b",
        re.IGNORECASE,
    )
    ents = []
    for e in doc.ents:
        if e.label_ not in ENTITY_LABELS:
            continue
        txt = e.text.strip()
        if len(txt) > 80 or JUNK_RE.search(txt):
            continue
        ents.append((txt, e.start_char, e.end_char, e.label_))
    ent_texts = {e[0].lower() for e in ents}
    return ents, ent_texts


def _is_entity(text: str, ent_texts: set[str], filer: str | None) -> bool:
    """True if `text` is an NER entity or the filing company name."""
    if not text or text.lower() in SKIP_TOK:
        return False
    if filer and text.lower() == filer.lower():
        return True
    # Also accept partial matches for long org names
    tl = text.lower()
    return tl in ent_texts or any(tl in et or et in tl for et in ent_texts)


def _nearest_ent_before(
    ents, char_pos: int, window: int = 300, prefer_label: str = "ORG"
) -> str | None:
    """Closest NER entity ending before char_pos; ORG/PRODUCT preferred over GPE."""
    cands = [e for e in ents if e[2] <= char_pos and char_pos - e[2] <= window]
    if not cands:
        return None
    org_cands = [e for e in cands if e[3] in ("ORG", "PRODUCT")]
    return (org_cands or cands)[-1][0]


def _nearest_ent_after(
    ents, char_pos: int, window: int = 300, prefer_label: str = "ORG"
) -> str | None:
    """Closest NER entity starting after char_pos; ORG/PRODUCT preferred."""
    cands = [e for e in ents if e[1] >= char_pos and e[1] - char_pos <= window]
    if not cands:
        return None
    org_cands = [e for e in cands if e[3] in ("ORG", "PRODUCT")]
    return (org_cands or cands)[0][0]


def _resolve_pronoun(tok_text: str, filer: str | None) -> str | None:
    """Map self-reference pronouns to the filer company name."""
    if filer and tok_text.lower() in SELF_TOK:
        return filer
    return None


def _tok_to_ent(token, ents, filer) -> str | None:
    """
    Return the entity text for a token, or the filer name if it's a pronoun,
    or None if it's neither.
    """
    # Check if token falls inside an NER span
    for ent_text, es, ee, _lbl in ents:
        if es <= token.idx < ee:
            return ent_text
    # Pronoun → filer
    resolved = _resolve_pronoun(token.text, filer)
    if resolved:
        return resolved
    return None


def _prep_pobj_ent(token, allowed_preps: set[str], ents, filer, ent_texts) -> str | None:
    """
    Find entity in a prepositional phrase attached to `token`.
    Searches `token.rights` for a prep whose lemma is in `allowed_preps`,
    then its pobj (or appositive of pobj for "such as X, Y" patterns).
    """
    for child in token.rights:
        if child.dep_ == "prep" and child.lemma_.lower() in allowed_preps:
            for gc in child.rights:
                if gc.dep_ in ("pobj", "nmod"):
                    ent = _tok_to_ent(gc, ents, filer)
                    if ent:
                        return ent
                    # "such as COMPANY" → pobj may be "such" with appos
                    for ggc in gc.rights:
                        if ggc.dep_ in ("appos", "conj"):
                            ent = _tok_to_ent(ggc, ents, filer)
                            if ent:
                                return ent
                    # fallback: first ORG entity after this token
                    near = _nearest_ent_after(ents, gc.idx, window=200)
                    if near:
                        return near
    return None


# ---------------------------------------------------------------------------
# Subject / object extraction via dependency tree
# ---------------------------------------------------------------------------

def _get_subj(token, ents, filer) -> str | None:
    """Grammatical subject of `token` (nsubj / nsubjpass)."""
    for child in token.lefts:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
            ent = _tok_to_ent(child, ents, filer)
            return ent  # may be None
    # also check parent's subject (for verb chains like "is designed to supply")
    head = token.head
    if head != token:
        for child in head.lefts:
            if child.dep_ in ("nsubj", "nsubjpass"):
                ent = _tok_to_ent(child, ents, filer)
                return ent
    return None


def _get_obj(token, ents, filer, cfg_preps: set[str] | None = None) -> str | None:
    """
    Direct or prepositional object of `token`.
    cfg_preps: relation-specific preferred preps (tried first).
    """
    ALL_PREPS = {"with", "to", "from", "of", "on", "for", "against", "such"}
    search_preps = (cfg_preps or set()) | ALL_PREPS

    # dobj / attr first (only if it's an actual entity)
    for child in token.rights:
        if child.dep_ in ("dobj", "attr"):
            ent = _tok_to_ent(child, ents, filer)
            if ent:
                return ent

    # Relation-preferred preps first, then others
    ordered = list(cfg_preps or []) + [p for p in ALL_PREPS if p not in (cfg_preps or [])]
    for target_prep in ordered:
        for child in token.rights:
            if child.dep_ == "prep" and child.lemma_.lower() == target_prep:
                for gc in child.rights:
                    if gc.dep_ in ("pobj", "nmod"):
                        ent = _tok_to_ent(gc, ents, filer)
                        if ent:
                            return ent
                        # "such as X" → scan for appositives / conjuncts
                        for ggc in gc.rights:
                            if ggc.dep_ in ("appos", "conj"):
                                ent = _tok_to_ent(ggc, ents, filer)
                                if ent:
                                    return ent
                        # fallback window
                        near = _nearest_ent_after(ents, gc.idx, window=250)
                        if near:
                            return near
    return None


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_sentence(
    sentence: str,
    source_file: str = "",
    source_sentence: str = "",
) -> list[Triple]:
    nlp = _get_nlp()
    filer = _filer_name(source_file)
    doc = nlp(sentence)

    ents, ent_texts = _build_ent_set(doc, filer)
    triples: list[Triple] = []
    seen: set[tuple] = set()

    for token in doc:
        lemma = token.lemma_.lower()

        # ── VERB triggers ──────────────────────────────────────────────────
        if token.pos_ in ("VERB", "AUX", "NOUN") and lemma in _VERB_MAP:
            for cfg in _VERB_MAP[lemma]:
                subj = _get_subj(token, ents, filer)
                obj  = _get_obj(token, ents, filer, cfg.get("obj_preps"))

                # Window fallback — NER entities only, prefer ORG
                if not subj:
                    subj = _nearest_ent_before(ents, token.idx, prefer_label="ORG") or \
                           (filer if filer else None)
                if not obj:
                    obj = _nearest_ent_after(ents, token.idx + len(token.text),
                                             prefer_label="ORG")

                # Require both to be non-trivial
                if not subj or not obj or subj == obj:
                    continue
                if subj.lower() in SKIP_TOK or obj.lower() in SKIP_TOK:
                    continue
                # At least one must be a known entity or the filer
                if not (_is_entity(subj, ent_texts, filer) or
                        _is_entity(obj, ent_texts, filer)):
                    continue

                if cfg["subj_is_head"]:
                    head, tail = subj, obj
                else:
                    head, tail = obj, subj

                key = (head, cfg["relation"], tail)
                if key not in seen:
                    seen.add(key)
                    triples.append(Triple(
                        head=head, relation=cfg["relation"], tail=tail,
                        confidence=cfg["confidence_verb"],
                        source_file=source_file,
                        source_sentence=source_sentence or sentence,
                    ))

        # ── NOUN triggers ──────────────────────────────────────────────────
        elif token.pos_ in ("NOUN", "PROPN") and lemma in _NOUN_MAP:
            for cfg in _NOUN_MAP[lemma]:
                # Pattern A: "X is a <noun> of Y"  → head=X, tail=Y
                # Pattern B: "our <noun>s include X" → possessive construction:
                #   - "our customers" → X is the customer OF filer → head=X, tail=filer
                #   - "our suppliers" → X is the supplier to filer → head=X, tail=filer
                #   - "our competitors" → filer competes with X → head=filer, tail=X (no swap)
                POSSESSIVE_SWAP_RELATIONS = {"supplier_of", "customer_of"}

                # "our suppliers" (poss) OR "utilize suppliers" (dobj) both mean
                # the listed entities play the supplier/customer role; filer is other party
                is_possessed_by_filer = filer and (
                    any(child.dep_ in ("poss", "det") and child.text.lower() in SELF_TOK
                        for child in token.lefts)
                    or token.dep_ == "dobj"
                )

                subj = None
                if token.head and token.head.pos_ in ("VERB", "AUX"):
                    subj = _get_subj(token.head, ents, filer)
                if not subj:
                    subj = _nearest_ent_before(ents, token.idx, prefer_label="ORG") or \
                           (filer if filer else None)

                preps = cfg.get("noun_customer_prep", {"of"})
                tail = _prep_pobj_ent(token, preps, ents, filer, ent_texts)
                if not tail:
                    tail = _nearest_ent_after(ents, token.idx + len(token.text),
                                              window=200, prefer_label="ORG")

                if not subj or not tail or subj == tail:
                    continue
                if subj.lower() in SKIP_TOK or tail.lower() in SKIP_TOK:
                    continue
                if not (_is_entity(subj, ent_texts, filer) or
                        _is_entity(tail, ent_texts, filer)):
                    continue

                # Possession swap: "our suppliers/customers include X" → X is the
                # supplier/customer; the filer is the other party → swap head/tail
                if is_possessed_by_filer and cfg["relation"] in POSSESSIVE_SWAP_RELATIONS:
                    # named entity (tail position) becomes head; filer becomes tail
                    if _is_entity(tail, ent_texts, None):  # tail is a real entity
                        final_head, final_tail = tail, filer
                    else:
                        final_head, final_tail = subj, tail
                else:
                    final_head, final_tail = subj, tail

                key = (final_head, cfg["relation"], final_tail)
                if key not in seen:
                    seen.add(key)
                    triples.append(Triple(
                        head=final_head, relation=cfg["relation"], tail=final_tail,
                        confidence=cfg["confidence_noun"],
                        source_file=source_file,
                        source_sentence=source_sentence or sentence,
                    ))

    return normalize_triples(triples)


def extract_file(filepath: str | pathlib.Path) -> list[Triple]:
    filepath = pathlib.Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    text = re.sub(r"=== ITEM \d+[A-Z]?:.*?===", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[a-z0-9\"\')])\.\s+(?=[A-Z\"])", text)
    all_triples: list[Triple] = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 800:
            continue
        triples = extract_sentence(
            sent,
            source_file=filepath.name,
            source_sentence=sent,
        )
        all_triples.extend(triples)
    return normalize_triples(all_triples)
