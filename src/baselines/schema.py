"""Shared schema for all relation extraction outputs."""

from dataclasses import dataclass, asdict
from typing import Literal

RELATION_VOCAB = frozenset({
    "supplier_of",
    "customer_of",
    "subsidiary_of",
    "competitor_of",
    "partner_of",
    "depends_on",
})

RelationType = Literal[
    "supplier_of", "customer_of", "subsidiary_of",
    "competitor_of", "partner_of", "depends_on"
]


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    confidence: float
    source_file: str
    source_sentence: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"({self.head!r}, {self.relation}, {self.tail!r}) "
            f"conf={self.confidence:.2f} src={self.source_file}"
        )


# ---------------------------------------------------------------------------
# Canonical normalization
# ---------------------------------------------------------------------------
# supplier_of and customer_of are logical inverses.  We keep supplier_of as the
# single canonical direction so that triples from different extractors and
# different sentence phrasings can be deduplicated / merged cleanly.
#
#   (A, customer_of, B)  →  (B, supplier_of, A)
#
# All other relations are kept as-is.

def normalize_triple(t: "Triple") -> "Triple":
    """Convert customer_of(A, B) → supplier_of(B, A); pass others through."""
    if t.relation == "customer_of":
        return Triple(
            head=t.tail,
            relation="supplier_of",
            tail=t.head,
            confidence=t.confidence,
            source_file=t.source_file,
            source_sentence=t.source_sentence,
        )
    return t


def normalize_triples(triples: list["Triple"]) -> list["Triple"]:
    """Apply normalize_triple to every item in a list."""
    return [normalize_triple(t) for t in triples]
