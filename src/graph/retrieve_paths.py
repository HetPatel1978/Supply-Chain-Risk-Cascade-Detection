from graph.build_graph import (
    MERGED_TRIPLES_PATH,
    build_graph,
    load_triples,
)

def get_impacted_neighbors(graph, company: str) -> list[str]:
    """Find companies that could be affected if this company is disrupted."""
    impacted = set()

    # supplier_of already points from supplier to customer, so risk follows the edge forward.
    for _, tail, attributes in graph.out_edges(company, data=True):
        if attributes.get("relation") == "supplier_of":
            impacted.add(tail)

    # depends_on and customer_of point from the dependent/customer toward the supplier, so risk follows those edges backward.
    for head, _, attributes in graph.in_edges(company, data=True):
        if attributes.get("relation") in {"depends_on", "customer_of"}:
            impacted.add(head)

    return sorted(impacted)


def find_paths(
    graph,
    start_node: str,
    max_hops: int = 5,
) -> list[list[str]]:
    """Find supply-chain risk cascade paths from a disrupted company."""
    if start_node not in graph:
        raise ValueError(f"{start_node!r} is not in the graph.")

    paths = []
    queue = [(start_node, [start_node])]

    while queue:
        current_node, current_path = queue.pop(0)
        hops = len(current_path) - 1

        if hops > 0:
            paths.append(current_path)

        if hops == max_hops:
            continue

        for next_node in get_impacted_neighbors(graph, current_node):
            if next_node in current_path:
                continue

            queue.append(
                (
                    next_node,
                    current_path + [next_node],
                )
            )

    return paths

def path_to_evidence(graph, path: list[str]) -> dict:
    """Convert a cascade path into structured relationship evidence."""
    relationships = []

    for disrupted_company, impacted_company in zip(path, path[1:]):
        matching_edges = []

        # supplier_of follows the stored edge direction.
        forward_edges = graph.get_edge_data(
            disrupted_company,
            impacted_company,
            default={},
        )

        for attributes in forward_edges.values():
            if attributes.get("relation") == "supplier_of":
                matching_edges.append(
                    {
                        "head": disrupted_company,
                        "relation": attributes.get("relation"),
                        "tail": impacted_company,
                        "confidence": attributes.get("confidence"),
                        "source_file": attributes.get("source_file"),
                        "source_sentence": attributes.get("source_sentence"),
                        "traversal_direction": "forward",
                    }
                )

        # depends_on and customer_of are stored in the opposite
        # direction from the way disruption propagates.
        reverse_edges = graph.get_edge_data(
            impacted_company,
            disrupted_company,
            default={},
        )

        for attributes in reverse_edges.values():
            if attributes.get("relation") in {
                "depends_on",
                "customer_of",
            }:
                matching_edges.append(
                    {
                        "head": impacted_company,
                        "relation": attributes.get("relation"),
                        "tail": disrupted_company,
                        "confidence": attributes.get("confidence"),
                        "source_file": attributes.get("source_file"),
                        "source_sentence": attributes.get("source_sentence"),
                        "traversal_direction": "reverse",
                    }
                )

        relationships.extend(matching_edges)

    return {
        "path": path,
        "relationships": relationships,
    }

def retrieve_evidence(
    graph,
    start_node: str,
    max_hops: int = 2,
    max_paths: int = 10,
) -> list[dict]:
    """Retrieve structured evidence for paths from a starting node."""
    paths = find_paths(
        graph,
        start_node=start_node,
        max_hops=max_hops,
    )

    return [
        path_to_evidence(graph, path)
        for path in paths[:max_paths]
    ]

if __name__ == "__main__":
    triples = load_triples(MERGED_TRIPLES_PATH)
    graph = build_graph(triples)

    start_company = "TSMC"

    results = retrieve_evidence(
        graph,
        start_node=start_company,
        max_hops=2,
        max_paths=10,
    )

    print(
        f"Retrieved {len(results)} evidence paths "
        f"for disruption at {start_company}."
    )

    for result in results:
        print("\nCascade path:")
        print(" -> ".join(result["path"]))

        for relationship in result["relationships"]:
            print(
                f"{relationship['head']} "
                f"--{relationship['relation']}--> "
                f"{relationship['tail']}"
            )
            print(
                f"Traversal: "
                f"{relationship['traversal_direction']}"
            )
            print(f"Source: {relationship['source_file']}")
            print(f"Evidence: {relationship['source_sentence']}")
