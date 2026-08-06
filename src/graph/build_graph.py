import json
from pathlib import Path
import networkx as nx


MERGED_TRIPLES_PATH = Path("results/merged_triples.json")


def load_triples(path: Path) -> list[dict]:
    """Load extracted relation triples from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the triples file: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    triples = data.get("triples")

    if not isinstance(triples, list):
        raise ValueError(
            "Expected the JSON file to contain a 'triples' list."
        )

    return triples

def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    """Build a directed graph while preserving source evidence."""
    graph = nx.MultiDiGraph()

    for triple in triples:
        head = triple.get("head")
        tail = triple.get("tail")
        relation = triple.get("relation")

        if not head or not tail or not relation:
            continue

        # Skip self-links because they are not useful for path retrieval.
        if head == tail:
            continue

        graph.add_edge(
            head,
            tail,
            relation=relation,
            confidence=triple.get("confidence"),
            source_file=triple.get("source_file"),
            source_sentence=triple.get("source_sentence"),
        )

    return graph


def print_sample_edges(graph: nx.MultiDiGraph, limit: int = 5) -> None:
    """Print a small sample of graph relationships."""
    for index, (head, tail, attributes) in enumerate(
        graph.edges(data=True)
    ):
        if index >= limit:
            break

        relation = attributes.get("relation", "unknown")
        print(f"{head} --{relation}--> {tail}")


if __name__ == "__main__":
    triples = load_triples(MERGED_TRIPLES_PATH)
    graph = build_graph(triples)

    print(f"Loaded triples: {len(triples):,}")
    print(f"Graph nodes: {graph.number_of_nodes():,}")
    print(f"Graph edges: {graph.number_of_edges():,}")

    print("\nSample edges:")
    print_sample_edges(graph)
