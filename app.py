"""
Supply Chain Risk Cascade Detector — Streamlit Dashboard
Run: streamlit run app.py
"""

import json
import os
import pathlib
import sys
import collections

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from graph.build_graph import build_graph, REL_COLOR, FILER_COMPANIES
from rag.retriever import retrieve_context, find_cascade_paths
from rag.generator import generate_cascade_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Cascade Detector",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge graph…")
def load_data():
    results = ROOT / "results"
    triples = json.loads(
        (results / "merged_triples.json").read_text(encoding="utf-8")
    )["triples"]
    G = build_graph(triples)
    return G, triples


G, triples = load_data()
all_companies = sorted({t["head"] for t in triples} | {t["tail"] for t in triples})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/SEC_EDGAR_logo.png/320px-SEC_EDGAR_logo.png",
        width=180,
    )
    st.title("Controls")
    st.caption("SEC 10-K filings · 2023 · 8 companies")

    st.divider()
    st.markdown("**Dataset snapshot**")
    col_a, col_b = st.columns(2)
    col_a.metric("Triples", len(triples))
    col_b.metric("Companies", G.number_of_nodes())

    rel_counts = collections.Counter(t["relation"] for t in triples)
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        color = REL_COLOR.get(rel, "#888")
        st.markdown(
            f'<span style="color:{color}; font-weight:600">■</span> '
            f'{rel.replace("_", " ")} · **{cnt}**',
            unsafe_allow_html=True,
        )

    st.divider()
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        st.success("Groq API · Connected", icon="✅")
    else:
        st.warning("Groq API · Not set\n\nAdd GROQ_API_KEY to .env for LLM analysis.", icon="⚠️")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_graph, tab_rag, tab_explorer = st.tabs([
    "📊  Knowledge Graph",
    "⚡  Risk Cascade",
    "📋  Triple Explorer",
])

# ── Tab 1 · Knowledge Graph ───────────────────────────────────────────────────
with tab_graph:
    st.subheader("Supply Chain Knowledge Graph (3D Interactive)")
    st.caption(
        "Yellow diamonds = filing companies (NVIDIA, AMD, Intel…) · "
        "Grey circles = supply chain partners · "
        "Drag to rotate, scroll to zoom, hover for details."
    )

    html_path = ROOT / "results" / "supply_chain_graph_3d.html"
    if html_path.exists():
        html_bytes = html_path.read_text(encoding="utf-8")
        st.components.v1.html(html_bytes, height=760, scrolling=False)
    else:
        st.error("Graph HTML not found. Run `python src/graph/build_graph.py` first.")

    # Stats row
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Filing companies", sum(1 for _, d in G.nodes(data=True) if d.get("is_filer")))
    c4.metric("Relation types", len(rel_counts))

    # Most connected
    st.markdown("**Most connected nodes (by total degree)**")
    deg_df = pd.DataFrame(
        [(n, G.in_degree(n), G.out_degree(n), G.in_degree(n) + G.out_degree(n))
         for n in G.nodes()],
        columns=["Company", "In-degree", "Out-degree", "Total"],
    ).sort_values("Total", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(deg_df, use_container_width=True, hide_index=True)


# ── Tab 2 · Risk Cascade ─────────────────────────────────────────────────────
with tab_rag:
    st.subheader("Supply Chain Risk Cascade Detector")
    st.caption(
        "Select a company, simulate a disruption, and trace the downstream impact "
        "across the supply chain graph. LLaMA 3.1 synthesizes an analyst narrative."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Disruption settings**")

        default_co = "TSMC" if "TSMC" in all_companies else all_companies[0]
        company = st.selectbox(
            "Disrupted company",
            all_companies,
            index=all_companies.index(default_co),
        )

        max_hops = st.slider("Max cascade hops", min_value=1, max_value=3, value=2,
                             help="How many supply-chain steps to follow downstream")

        scenario = st.text_area(
            "Scenario description (optional)",
            placeholder="e.g. TSMC halts EUV production due to geopolitical tensions with China",
            height=100,
        )

        run_btn = st.button("⚡ Detect Risk Cascade", type="primary", use_container_width=True)

        # Show direct relations of selected company
        st.divider()
        st.markdown(f"**Direct edges for {company}**")
        direct_rows = []
        for u, v, d in G.edges(data=True):
            if u == company or v == company:
                direct_rows.append({
                    "From": u, "Relation": d["relation"], "To": v,
                    "Conf": f"{d['confidence']:.2f}",
                })
        if direct_rows:
            st.dataframe(
                pd.DataFrame(direct_rows).sort_values("Relation"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No direct edges found for this company.")

    with right:
        if run_btn:
            with st.spinner("Traversing supply chain graph…"):
                ctx = retrieve_context(G, company, max_hops=max_hops)

            paths = ctx["cascade_paths"]
            suppliers = ctx["suppliers"]

            if not paths and not suppliers:
                st.warning(
                    f"**{company}** has no supply-chain connections in the current graph. "
                    "Try a more central company like TSMC, NVIDIA, or AMD."
                )
            else:
                # Cascade paths
                st.markdown(f"#### Downstream cascade from **{company}**")
                if paths:
                    rows = []
                    for p in paths:
                        rows.append({
                            "Hops": p["hops"],
                            "Cascade chain": " → ".join(p["path"]),
                            "Relations": " → ".join(e["relation"] for e in p["edges"]),
                        })
                    cascade_df = pd.DataFrame(rows).sort_values("Hops")
                    st.dataframe(cascade_df, use_container_width=True, hide_index=True)

                    # Mini Plotly cascade tree
                    affected = {p["path"][-1] for p in paths}
                    direct = {p["path"][-1] for p in paths if p["hops"] == 1}
                    st.markdown(
                        f"**{len(direct)}** companies at immediate risk · "
                        f"**{len(affected) - len(direct)}** secondary"
                    )
                else:
                    st.info("No downstream customers found in graph.")

                # Upstream suppliers
                if suppliers:
                    st.markdown(f"#### Upstream suppliers of **{company}**")
                    sup_df = pd.DataFrame(suppliers).rename(
                        columns={"company": "Supplier", "relation": "Relation"}
                    )[["Supplier", "Relation"]]
                    st.dataframe(sup_df, use_container_width=True, hide_index=True)

                # Source evidence
                if ctx["source_sentences"]:
                    with st.expander("📄 Source evidence from 10-K filings"):
                        for s in ctx["source_sentences"]:
                            st.markdown(f"- *{s}*")

                # LLM analysis
                st.markdown("#### Risk Cascade Analysis")
                if groq_key:
                    with st.spinner("Generating analysis with LLaMA 3.1-8B…"):
                        analysis = generate_cascade_analysis(ctx, scenario, groq_key)
                    st.info(analysis)
                else:
                    st.warning(
                        "LLM analysis disabled — add `GROQ_API_KEY` to your `.env` file."
                    )
        else:
            st.markdown(
                """
                <div style="
                    border: 1px dashed #475569;
                    border-radius: 8px;
                    padding: 2rem;
                    text-align: center;
                    color: #94a3b8;
                    margin-top: 2rem;
                ">
                    <div style="font-size:3rem">⚡</div>
                    <div style="font-size:1.1rem; margin-top:0.5rem">
                        Select a company and click <strong>Detect Risk Cascade</strong>
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.9rem">
                        The graph will trace which companies are exposed<br>
                        to a disruption at the selected node.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Tab 3 · Triple Explorer ───────────────────────────────────────────────────
with tab_explorer:
    st.subheader("Triple Explorer")
    st.caption("All 60 clean triples extracted from SEC 10-K filings.")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    filter_rel = fc1.multiselect(
        "Relation type",
        options=sorted(rel_counts.keys()),
        default=sorted(rel_counts.keys()),
    )
    filter_head = fc2.selectbox("Head company", ["(all)"] + all_companies)
    filter_tail = fc3.selectbox("Tail company", ["(all)"] + all_companies)

    rows = []
    for t in triples:
        if t["relation"] not in filter_rel:
            continue
        if filter_head != "(all)" and t["head"] != filter_head:
            continue
        if filter_tail != "(all)" and t["tail"] != filter_tail:
            continue
        rows.append({
            "Head": t["head"],
            "Relation": t["relation"],
            "Tail": t["tail"],
            "Conf": round(t["confidence"], 2),
            "Extractor": t.get("extractor", ""),
            "Source file": t.get("source_file", ""),
            "Evidence": t.get("source_sentence", "")[:120] + "…"
            if len(t.get("source_sentence", "")) > 120
            else t.get("source_sentence", ""),
        })

    st.markdown(f"**{len(rows)} triples** match the current filters.")
    if rows:
        explorer_df = pd.DataFrame(rows)

        # Colour-code relation column
        def colour_rel(val):
            c = REL_COLOR.get(val, "#888888")
            return f"color: {c}; font-weight: 600"

        st.dataframe(
            explorer_df.style.map(colour_rel, subset=["Relation"]),
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.info("No triples match the current filters.")
