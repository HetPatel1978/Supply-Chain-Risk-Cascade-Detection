"""Build a NetworkX knowledge graph from merged triples and visualize in 3D."""

import json
import pathlib
import collections
import networkx as nx
import plotly.graph_objects as go
import numpy as np

ROOT    = pathlib.Path(__file__).parent.parent.parent
RESULTS = ROOT / "results"

# ── Relation styling ─────────────────────────────────────────────────────────
REL_COLOR = {
    "supplier_of":   "#3B82F6",   # blue
    "depends_on":    "#F97316",   # orange
    "competitor_of": "#EF4444",   # red
    "subsidiary_of": "#A855F7",   # purple
    "partner_of":    "#22C55E",   # green
    "customer_of":   "#06B6D4",   # cyan
}

REL_DASH = {
    "supplier_of":   "solid",
    "depends_on":    "dot",
    "competitor_of": "dash",
    "subsidiary_of": "longdash",
    "partner_of":    "dashdot",
    "customer_of":   "solid",
}

# Our 8 filing companies — highlighted differently in the graph
FILER_COMPANIES = {
    "NVIDIA", "AMD", "Applied Materials", "Lam Research",
    "KLA", "Micron", "Broadcom", "Qualcomm", "Intel",
}


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for t in triples:
        h, r, tl, c = t["head"], t["relation"], t["tail"], t["confidence"]
        if not G.has_node(h):
            G.add_node(h, is_filer=(h in FILER_COMPANIES))
        if not G.has_node(tl):
            G.add_node(tl, is_filer=(tl in FILER_COMPANIES))
        G.add_edge(h, tl, relation=r, confidence=c,
                   source_file=t.get("source_file",""),
                   source_sentence=t.get("source_sentence",""))
    return G


def graph_stats(G: nx.MultiDiGraph):
    print(f"\n── Graph Statistics ─────────────────────────────────────────")
    print(f"  Nodes (companies)   : {G.number_of_nodes()}")
    print(f"  Edges (relations)   : {G.number_of_edges()}")
    print(f"  Filing companies    : {sum(1 for _,d in G.nodes(data=True) if d.get('is_filer'))}")

    rel_counts = collections.Counter(d["relation"] for _,_,d in G.edges(data=True))
    print(f"\n  Relation breakdown:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel:<20} {cnt}")

    # Most connected companies (in-degree = most depended upon)
    in_deg  = sorted(G.in_degree(),  key=lambda x: -x[1])
    out_deg = sorted(G.out_degree(), key=lambda x: -x[1])
    print(f"\n  Most depended-on (highest in-degree):")
    for node, deg in in_deg[:5]:
        print(f"    {node:<30} {deg} incoming edges")
    print(f"\n  Most outgoing relations (highest out-degree):")
    for node, deg in out_deg[:5]:
        print(f"    {node:<30} {deg} outgoing edges")


def visualize_3d(G: nx.MultiDiGraph, out_path: pathlib.Path):
    # 3D spring layout
    np.random.seed(42)
    pos = nx.spring_layout(G, dim=3, seed=42, k=2.5)

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))

    # ── Edge traces (one per relation type, for legend) ──────────────────────
    rel_edges: dict[str, list] = collections.defaultdict(list)
    for u, v, data in edges:
        rel_edges[data["relation"]].append((u, v, data))

    edge_traces = []
    for rel, elist in sorted(rel_edges.items()):
        xs, ys, zs = [], [], []
        hover = []
        for u, v, data in elist:
            x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
            hover += [
                f"{u} → {v}<br>relation: {rel}<br>conf: {data['confidence']:.2f}<br>{data['source_file']}",
                f"{u} → {v}<br>relation: {rel}<br>conf: {data['confidence']:.2f}<br>{data['source_file']}",
                None,
            ]
        edge_traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            name=rel,
            line=dict(color=REL_COLOR.get(rel, "#888888"), width=3),
            hoverinfo="text", text=hover,
            legendgroup=rel,
        ))

    # ── Arrow cones (show direction) ─────────────────────────────────────────
    cone_u, cone_v, cone_w = [], [], []
    cone_x, cone_y, cone_z = [], [], []
    cone_colors = []
    for u, v, data in edges:
        x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
        mid_x = x0 + 0.75*(x1-x0)
        mid_y = y0 + 0.75*(y1-y0)
        mid_z = z0 + 0.75*(z1-z0)
        cone_x.append(mid_x); cone_y.append(mid_y); cone_z.append(mid_z)
        cone_u.append(x1-x0); cone_v.append(y1-y0); cone_w.append(z1-z0)
        cone_colors.append(REL_COLOR.get(data["relation"], "#888"))

    cone_trace = go.Cone(
        x=cone_x, y=cone_y, z=cone_z,
        u=cone_u, v=cone_v, w=cone_w,
        colorscale=[[0,"#888"],[1,"#888"]], showscale=False,
        sizemode="absolute", sizeref=0.15,
        hoverinfo="skip", showlegend=False,
    )

    # ── Node traces (filer vs non-filer) ─────────────────────────────────────
    filer_x,  filer_y,  filer_z,  filer_text,  filer_sizes  = [], [], [], [], []
    other_x,  other_y,  other_z,  other_text,  other_sizes  = [], [], [], [], []

    in_deg_map = dict(G.in_degree())
    out_deg_map = dict(G.out_degree())

    for node, data in nodes:
        x, y, z = pos[node]
        deg = in_deg_map[node] + out_deg_map[node]
        size = max(12, min(40, 10 + deg * 4))
        hover_txt = (
            f"<b>{node}</b><br>"
            f"In: {in_deg_map[node]}  Out: {out_deg_map[node]}<br>"
            f"Relations: {', '.join(set(d['relation'] for _,_,d in G.edges(node, data=True)))}"
        )
        if data.get("is_filer"):
            filer_x.append(x); filer_y.append(y); filer_z.append(z)
            filer_text.append(node); filer_sizes.append(size)
        else:
            other_x.append(x); other_y.append(y); other_z.append(z)
            other_text.append(f"  {node}"); other_sizes.append(size)

    filer_node_trace = go.Scatter3d(
        x=filer_x, y=filer_y, z=filer_z, mode="markers+text",
        name="Filing company (8)", text=filer_text,
        textposition="top center",
        marker=dict(size=filer_sizes, color="#FACC15", symbol="diamond",
                    line=dict(color="#000", width=1.5)),
        hovertext=filer_text, hoverinfo="text",
    )
    other_node_trace = go.Scatter3d(
        x=other_x, y=other_y, z=other_z, mode="markers+text",
        name="Other company", text=other_text,
        textposition="top center",
        marker=dict(size=other_sizes, color="#94A3B8", symbol="circle",
                    line=dict(color="#334155", width=1)),
        hovertext=other_text, hoverinfo="text",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(
        data=edge_traces + [cone_trace, filer_node_trace, other_node_trace],
        layout=go.Layout(
            title=dict(
                text="Supply Chain Risk — Knowledge Graph (3D Interactive)",
                font=dict(size=20, color="#F1F5F9"),
                x=0.5,
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            legend=dict(
                font=dict(color="#F1F5F9", size=12),
                bgcolor="rgba(30,41,59,0.8)",
                bordercolor="#475569",
                borderwidth=1,
                x=0.01, y=0.99,
            ),
            scene=dict(
                bgcolor="#0F172A",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=800,
        )
    )

    html_path = out_path / "supply_chain_graph_3d.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"\n  3D graph saved → {html_path}")
    return html_path


def save_graph_json(G: nx.MultiDiGraph, out_path: pathlib.Path):
    data = {
        "nodes": [
            {"id": n, "is_filer": d.get("is_filer", False),
             "in_degree": G.in_degree(n), "out_degree": G.out_degree(n)}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v,
             "relation": d["relation"], "confidence": d["confidence"],
             "source_file": d.get("source_file",""),
             "source_sentence": d.get("source_sentence","")}
            for u, v, d in G.edges(data=True)
        ],
    }
    p = out_path / "supply_chain_graph.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Graph JSON saved  → {p}")


if __name__ == "__main__":
    triples = json.loads(
        (RESULTS / "merged_triples.json").read_text(encoding="utf-8")
    )["triples"]

    G = build_graph(triples)
    graph_stats(G)
    html = visualize_3d(G, RESULTS)
    save_graph_json(G, RESULTS)

    print("\n  Done. Open the HTML file in your browser to explore the 3D graph.")
    print(f"  → {html}")
