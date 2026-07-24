"""
Multi-hop graph traversal for supply chain risk cascade detection.

Cascade logic:
  - A supplier_of B  → if A is disrupted, B is immediately at risk
  - A depends_on  B  → if B is disrupted, A is immediately at risk
  BFS up to max_hops from the disrupted node.
"""

import collections
import networkx as nx

# Relations that propagate risk downstream (outgoing edges)
_DOWNSTREAM_RELS = {"supplier_of", "partner_of"}
# Relations where the disrupted node is the *tail* and risk flows to the *head*
_UPSTREAM_DEPENDS = {"depends_on"}


def find_cascade_paths(
    G: nx.MultiDiGraph,
    start: str,
    max_hops: int = 2,
) -> list[dict]:
    """
    BFS from `start`.  Returns list of path dicts, one per affected company.
    A company is 'affected' if it relies (directly or transitively) on `start`.
    """
    if start not in G:
        return []

    results: list[dict] = []
    visited: set[str] = {start}
    # queue items: (current_node, path_so_far, edge_list)
    queue: collections.deque = collections.deque([(start, [start], [])])

    while queue:
        node, path, edges = queue.popleft()
        hop = len(path) - 1

        if hop > 0:
            results.append({"path": list(path), "edges": list(edges), "hops": hop})

        if hop >= max_hops:
            continue

        # Outgoing supplier_of / partner_of → tail is a customer of `node`
        for _, nbr, data in G.out_edges(node, data=True):
            if data["relation"] in _DOWNSTREAM_RELS and nbr not in visited:
                visited.add(nbr)
                queue.append((
                    nbr, path + [nbr],
                    edges + [{"from": node, "to": nbr,
                               "relation": data["relation"],
                               "confidence": data.get("confidence", 0),
                               "source_sentence": data.get("source_sentence", "")}],
                ))

        # Incoming depends_on → head depends on `node`, so head is at risk
        for src, _, data in G.in_edges(node, data=True):
            if data["relation"] in _UPSTREAM_DEPENDS and src not in visited:
                visited.add(src)
                queue.append((
                    src, path + [src],
                    edges + [{"from": src, "to": node,
                               "relation": data["relation"],
                               "confidence": data.get("confidence", 0),
                               "source_sentence": data.get("source_sentence", "")}],
                ))

    return sorted(results, key=lambda x: (x["hops"], x["path"][-1]))


def get_suppliers(G: nx.MultiDiGraph, company: str) -> list[dict]:
    """Return companies that directly supply `company` (its upstream)."""
    suppliers = []
    for src, _, data in G.in_edges(company, data=True):
        if data["relation"] in _DOWNSTREAM_RELS:
            suppliers.append({"company": src, "relation": data["relation"],
                               "source_sentence": data.get("source_sentence", "")})
    for _, tgt, data in G.out_edges(company, data=True):
        if data["relation"] in _UPSTREAM_DEPENDS:
            suppliers.append({"company": tgt, "relation": data["relation"],
                               "source_sentence": data.get("source_sentence", "")})
    return suppliers


def retrieve_context(
    G: nx.MultiDiGraph,
    start: str,
    max_hops: int = 2,
) -> dict:
    """
    Returns a context dict with cascade paths + source evidence for the generator.
    """
    paths = find_cascade_paths(G, start, max_hops)
    suppliers = get_suppliers(G, start)

    # Collect all edge source sentences related to `start`
    sentences: list[str] = []
    seen: set[str] = set()
    for u, v, data in G.edges(data=True):
        if u == start or v == start:
            s = data.get("source_sentence", "").strip()
            if s and s not in seen:
                seen.add(s)
                sentences.append(s)

    # Also collect sentences from hop-1 edges in cascade paths
    for p in paths:
        if p["hops"] == 1:
            for e in p["edges"]:
                s = e.get("source_sentence", "").strip()
                if s and s not in seen:
                    seen.add(s)
                    sentences.append(s)

    return {
        "start": start,
        "cascade_paths": paths,
        "suppliers": suppliers,
        "source_sentences": sentences[:12],
    }
