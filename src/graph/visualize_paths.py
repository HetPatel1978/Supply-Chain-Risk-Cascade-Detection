from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


OUTPUT_PATH = Path("results/affected_supply_chain.png")


def build_path_graph(evidence_paths: list[dict]) -> nx.DiGraph:
    """Build a smaller graph containing only the retrieved cascade paths."""
    path_graph = nx.DiGraph()

    for result in evidence_paths:
        path = result["path"]

        for source, target in zip(path, path[1:]):
            path_graph.add_edge(
                source,
                target,
                label="risk flows to",
            )

    return path_graph


def get_layered_positions(
    path_graph: nx.DiGraph,
    disrupted_company: str,
) -> tuple[dict, dict]:
    """Place companies in layers based on their distance from the disruption."""
    distances = nx.single_source_shortest_path_length(
        path_graph,
        disrupted_company,
    )

    layers = {}

    for node, hop in distances.items():
        layers.setdefault(hop, []).append(node)

    ordered_layers = {}

    # The disrupted company is always first.
    ordered_layers[0] = [disrupted_company]

    # Sort directly affected companies consistently.
    if 1 in layers:
        ordered_layers[1] = sorted(layers[1])

    # Group later-hop companies near their parent in the prior layer.
    for hop in range(2, max(layers, default=0) + 1):
        previous_layer = ordered_layers.get(hop - 1, [])
        previous_positions = {
            node: index
            for index, node in enumerate(previous_layer)
        }

        def parent_position(node: str) -> tuple[float, str]:
            parents = [
                parent
                for parent in path_graph.predecessors(node)
                if parent in previous_positions
            ]

            if not parents:
                return float("inf"), node

            average_position = sum(
                previous_positions[parent]
                for parent in parents
            ) / len(parents)

            return average_position, node

        ordered_layers[hop] = sorted(
            layers.get(hop, []),
            key=parent_position,
        )

    positions = {}

    for hop, nodes in ordered_layers.items():
        count = len(nodes)

        for index, node in enumerate(nodes):
            y_position = (count - 1) / 2 - index

            positions[node] = (
                hop * 3.8,
                y_position * 1.7,
            )

    return positions, distances


def visualize_paths(
    evidence_paths: list[dict],
    disrupted_company: str,
    target_company: str | None = None,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Visualize direct and indirect supply-chain impacts."""
    path_graph = build_path_graph(evidence_paths)

    if path_graph.number_of_nodes() == 0:
        print("No graph paths were available to visualize.")
        return

    positions, distances = get_layered_positions(
        path_graph,
        disrupted_company,
    )

    node_sizes = []
    node_colors = []

    for node in path_graph.nodes:
        hop = distances.get(node, 0)

        if node == disrupted_company:
            node_sizes.append(3200)
            node_colors.append("#d9534f")
        elif hop == 1:
            node_sizes.append(2500)
            node_colors.append("#f0ad4e")
        else:
            node_sizes.append(2200)
            node_colors.append("#5b9bd5")

    fig, ax = plt.subplots(figsize=(9, 5))

    nx.draw_networkx_nodes(
        path_graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )

    nx.draw_networkx_edges(
        path_graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=26,
        node_size=node_sizes,
        min_source_margin=18,
        min_target_margin=24,
        width=1.6,
        connectionstyle="arc3,rad=0.02",
        ax=ax,
    )

    nx.draw_networkx_labels(
        path_graph,
        positions,
        font_size=9,
        ax=ax,
    )

    layer_headings = {
        0: "Disrupted company",
        1: "Direct impact",
        2: "Indirect impact",
    }

    highest_y = max(
        y_position
        for _, y_position in positions.values()
    )

    for hop, heading in layer_headings.items():
        if any(distance == hop for distance in distances.values()):
            ax.text(
                hop * 3.8,
                highest_y + 1.0,
                heading,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        f"Supply-Chain Risk Propagation from {disrupted_company}",
        fontsize=16,
        fontweight="bold",
        pad=40,
    )

    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"\nGraph visualization saved to: {output_path}")
