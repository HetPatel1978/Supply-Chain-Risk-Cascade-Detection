"""
Supply Chain Risk Cascade Detector — Streamlit Dashboard
Run: streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before pyplot import

import json
import os
import pathlib
import sys
import collections
import io

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from graph.build_graph import build_graph, load_triples, MERGED_TRIPLES_PATH
from graph.retrieve_paths import retrieve_evidence
from graph.visualize_paths import visualize_paths, build_path_graph
from rag.build_context import (
    build_rag_context,
    build_rag_prompt,
    generate_answer,
    find_company_in_question,
    select_deepest_paths,
    MAX_CASCADE_HOPS,
    MAX_EVIDENCE_PATHS,
    MAX_SELECTED_PATHS,
)

# ── Constants that used to live in build_graph.py ────────────────────────────
REL_COLOR = {
    "supplier_of":   "#3B82F6",
    "depends_on":    "#F97316",
    "competitor_of": "#EF4444",
    "subsidiary_of": "#A855F7",
    "partner_of":    "#22C55E",
    "customer_of":   "#06B6D4",
}

FILER_COMPANIES = {
    "NVIDIA", "AMD", "Applied Materials", "Lam Research",
    "KLA", "Micron", "Broadcom", "Qualcomm", "Intel",
}

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
    triples_path = ROOT / "results" / "merged_triples.json"
    triples = load_triples(triples_path)
    G = build_graph(triples)
    return G, triples


G, triples = load_data()
all_companies = sorted({t["head"] for t in triples} | {t["tail"] for t in triples})
rel_counts = collections.Counter(t["relation"] for t in triples)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 Supply Chain Risk")
    st.caption("SEC 10-K filings · 2023 · 8 filers")

    st.divider()
    st.markdown("**Dataset snapshot**")
    col_a, col_b = st.columns(2)
    col_a.metric("Triples", len(triples))
    col_b.metric("Companies", G.number_of_nodes())

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
        st.warning("Add GROQ_API_KEY to .env\nto enable LLM analysis.", icon="⚠️")

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
        "Yellow diamonds = filing companies · Grey circles = supply chain partners · "
        "Drag to rotate, scroll to zoom, hover for details."
    )

    html_path = ROOT / "results" / "supply_chain_graph_3d.html"
    if html_path.exists():
        st.components.v1.html(
            html_path.read_text(encoding="utf-8"),
            height=760,
            scrolling=False,
        )
    else:
        st.warning(
            "3D graph HTML not found. "
            "The graph was built before this session — check `results/` directory."
        )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Filing companies", sum(1 for n in G.nodes() if n in FILER_COMPANIES))
    c4.metric("Relation types", len(rel_counts))

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
        "Select a disrupted company and optionally describe a scenario. "
        "The pipeline traverses the knowledge graph up to 5 hops and uses "
        "LLaMA 3.1 to synthesize a grounded risk narrative."
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

        max_hops = st.slider(
            "Max cascade hops",
            min_value=1, max_value=MAX_CASCADE_HOPS, value=2,
            help="How many supply-chain steps to follow from the disrupted node",
        )

        scenario = st.text_area(
            "Scenario / question (optional)",
            placeholder=(
                "e.g. What happens to NVIDIA if TSMC halts production "
                "due to geopolitical tensions?"
            ),
            height=110,
        )

        run_btn = st.button(
            "⚡ Detect Risk Cascade", type="primary", use_container_width=True
        )

        st.divider()
        st.markdown(f"**Direct edges for {company}**")
        direct_rows = [
            {"From": u, "Relation": d["relation"], "To": v,
             "Conf": f"{d.get('confidence', 0):.2f}"}
            for u, v, d in G.edges(data=True)
            if u == company or v == company
        ]
        if direct_rows:
            st.dataframe(
                pd.DataFrame(direct_rows).sort_values("Relation"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No direct edges for this company.")

    with right:
        if run_btn:
            with st.spinner("Traversing supply chain graph…"):
                evidence_paths = retrieve_evidence(
                    G,
                    start_node=company,
                    max_hops=max_hops,
                    max_paths=MAX_EVIDENCE_PATHS,
                )

            if not evidence_paths:
                st.warning(
                    f"**{company}** has no reachable downstream dependencies "
                    "in the current graph. Try TSMC, NVIDIA, AMD, or Lam Research."
                )
            else:
                selected = select_deepest_paths(evidence_paths, limit=MAX_SELECTED_PATHS)
                max_depth = max(len(p["path"]) - 1 for p in evidence_paths)
                direct_count = sum(1 for p in evidence_paths if len(p["path"]) == 2)

                # Summary metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total affected paths", len(evidence_paths))
                m2.metric("Direct (1-hop)", direct_count)
                m3.metric("Max depth found", f"{max_depth} hops")

                # Cascade paths table
                st.markdown(f"#### Cascade paths from **{company}**")
                rows = []
                for p in evidence_paths:
                    rows.append({
                        "Hops": len(p["path"]) - 1,
                        "Cascade chain": " → ".join(p["path"]),
                        "Relations": " → ".join(
                            r["relation"] for r in p["relationships"]
                        ),
                    })
                st.dataframe(
                    pd.DataFrame(rows).sort_values("Hops"),
                    use_container_width=True, hide_index=True,
                )

                # Matplotlib layered cascade diagram
                st.markdown("#### Cascade diagram")
                try:
                    viz_path = ROOT / "results" / "_cascade_latest.png"
                    visualize_paths(
                        evidence_paths=selected,
                        disrupted_company=company,
                        output_path=viz_path,
                    )
                    if viz_path.exists():
                        st.image(str(viz_path), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Diagram generation failed: {exc}")

                # Source evidence
                context_str = build_rag_context(selected)
                with st.expander("📄 Source evidence from 10-K filings"):
                    st.code(context_str, language=None)

                # LLM analysis
                st.markdown("#### Risk Analysis (LLaMA 3.1-8B)")
                if groq_key:
                    question = scenario.strip() or (
                        f"What are the supply-chain risks if {company} is disrupted?"
                    )
                    prompt = build_rag_prompt(question=question, context=context_str)
                    with st.spinner("Generating grounded analysis…"):
                        try:
                            answer = generate_answer(prompt)
                            st.info(answer)
                        except Exception as exc:
                            st.error(f"Generation error: {exc}")
                else:
                    st.warning(
                        "LLM analysis disabled — add `GROQ_API_KEY` to your `.env` file."
                    )
        else:
            st.markdown(
                """
                <div style="
                    border: 1px dashed #475569; border-radius: 8px;
                    padding: 2rem; text-align: center; color: #94a3b8; margin-top:2rem;
                ">
                    <div style="font-size:3rem">⚡</div>
                    <div style="font-size:1.1rem; margin-top:0.5rem">
                        Select a company and click <strong>Detect Risk Cascade</strong>
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.9rem">
                        Optionally add a scenario to focus the LLM analysis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Tab 3 · Triple Explorer ───────────────────────────────────────────────────
with tab_explorer:
    st.subheader("Triple Explorer")
    st.caption(f"All {len(triples)} clean triples extracted from SEC 10-K filings.")

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
        sentence = t.get("source_sentence", "")
        rows.append({
            "Head": t["head"],
            "Relation": t["relation"],
            "Tail": t["tail"],
            "Conf": round(t.get("confidence", 0), 2),
            "Extractor": t.get("extractor", ""),
            "Source file": t.get("source_file", ""),
            "Evidence": sentence[:120] + "…" if len(sentence) > 120 else sentence,
        })

    st.markdown(f"**{len(rows)} triples** match the current filters.")
    if rows:
        explorer_df = pd.DataFrame(rows)

        def colour_rel(val):
            return f"color: {REL_COLOR.get(val, '#888888')}; font-weight: 600"

        st.dataframe(
            explorer_df.style.map(colour_rel, subset=["Relation"]),
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.info("No triples match the current filters.")
