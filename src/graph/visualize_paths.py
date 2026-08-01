from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


OUTPUT_PATH = Path("results/affected_supply_chain.png")

HOP_COLORS = [
    "#E31A1C",  # disrupted company
    "#FDAE61",  # direct impact
    "#FFFFBF",  # hop 2
    "#ABDDA4",  # hop 3
    "#3288BD",  # hop 4
    "#8e63b6",  # hop 5
]


def build_path_graph(evidence_paths: list[dict]) -> nx.DiGraph:
    """Build a smaller graph containing only the retrieved cascade paths."""
    path_graph = nx.DiGraph()

    for result in evidence_paths:
        path = result["path"]

        for source, target in zip(path, path[1:]):
            path_graph.add_edge(source, target)

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

    ordered_layers = {0: [disrupted_company]}

    if 1 in layers:
        ordered_layers[1] = sorted(layers[1])

    # Group later-hop companies near their parent in the previous layer.
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
    """Visualize supply-chain impacts using a separate color for each hop."""
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
        color_index = min(hop, len(HOP_COLORS) - 1)

        node_colors.append(HOP_COLORS[color_index])

        if node == disrupted_company:
            node_sizes.append(3200)
        elif node == target_company:
            node_sizes.append(2700)
        elif hop == 1:
            node_sizes.append(2500)
        else:
            node_sizes.append(2200)

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
        font_size=8,
        ax=ax,
    )

    layer_headings = {
        hop: (
            "Disrupted company"
            if hop == 0
            else f"{hop}-hop"
        )
        for hop in sorted(set(distances.values()))
    }

    highest_y = max(
        y_position
        for _, y_position in positions.values()
    )

    for hop, heading in layer_headings.items():
        ax.text(
            hop * 3.8,
            highest_y + 0.6,
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
        pad=75,
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
