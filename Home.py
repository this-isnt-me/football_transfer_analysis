"""Football Transfer Network Analysis — Streamlit home page.

Run with:  streamlit run Home.py

This is the Phase 1 scaffold: it wires up the shared data layer (graphs + tidy
edge frames + the P1/P2 mechanisms) and provides stub navigation. None of the 37
analyses are implemented yet — each section page is a placeholder.
"""
import pandas as pd
import streamlit as st

from src.data_layer import (
    NETWORKS,
    get_all_graphs,
    get_edges,
    get_league_names,
    get_p1,
    get_p2,
    p1_invariant_report,
    p2_violations,
)

st.set_page_config(
    page_title="Transfer Network Analysis",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Transfer Network Analysis")
st.caption(
    "Four directed multigraphs of football transfers — player movement and money "
    "flow, at club and league scale. This app will implement 37 analyses across 5 "
    "sections; this page confirms the shared data layer is wired up."
)

# --------------------------------------------------------------------------- #
# Data-layer status — proves graphs, edge frames, P1 and P2 all load & cache.
# --------------------------------------------------------------------------- #
graphs = get_all_graphs()
league_names = get_league_names()

st.subheader("Networks")
overview = pd.DataFrame(
    [
        {
            "Network": name,
            "Grain": "league" if "league" in name else "club",
            "Layer": "finance" if "finance" in name else "movement",
            "Nodes": graphs[name].number_of_nodes(),
            "Edges": graphs[name].number_of_edges(),
            "Edge direction": (
                "buyer → seller (money)" if "finance" in name
                else "seller → buyer (player)"
            ),
        }
        for name in NETWORKS
    ]
)
st.dataframe(overview, hide_index=True, width="stretch")
st.caption(
    "Finance edges run **buyer → seller** — the reverse of movement. Finance "
    "out-strength = spend, in-strength = sales revenue."
)

# --------------------------------------------------------------------------- #
# P1 / P2 health check
# --------------------------------------------------------------------------- #
st.subheader("Core mechanisms")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**P1 — `transfer_id` join (movement ⇄ finance)**")
    p1 = get_p1("club")
    rep = p1_invariant_report(p1)
    c = st.columns(3)
    c[0].metric("Movement deals", f"{rep['n_movement']:,}")
    c[1].metric("Matched to a fee", f"{rep['n_matched']:,}")
    c[2].metric("No fee (NULL)", f"{rep['n_unmatched']:,}")
    ok = (
        rep["endpoint_violations"] == 0
        and rep["position_divergences"] == 0
        and rep["unmatched_fee_all_null"]
        and rep["n_fee_equals_zero"] == 0
    )
    st.success("Reversal invariant holds; missing fees are NULL, never 0.") if ok else (
        st.error(f"P1 invariant FAILED: {rep}")
    )

with col2:
    st.markdown("**P2 — time-dependent club → league mapping**")
    p2 = get_p2()
    viol = p2_violations(p2)
    c = st.columns(2)
    c[0].metric("(club, season, window) keys", f"{p2.groupby(['club','season','window']).ngroups:,}")
    c[1].metric("Ambiguous (>1 league)", f"{len(viol):,}")
    if viol.empty:
        st.success("Each (club, season, window) resolves to exactly one league.")
    else:
        st.error(f"{len(viol)} ambiguous mappings — data-quality flag.")
        st.dataframe(viol.head(20), hide_index=True)

with st.expander("Peek at a tidy edge frame (movement_club)"):
    st.dataframe(get_edges("movement_club").head(20), hide_index=True, width="stretch")

st.divider()
st.subheader("Sections")
st.markdown(
    "Use the sidebar to navigate. Analyses are not implemented yet — each page is a stub.\n\n"
    "- **Section 1** — Single-network analyses (#1–15)\n"
    "- **Section 2** — Cross-network, same type: club vs league (#16–18)\n"
    "- **Section 3** — Cross-network, same granularity: movement vs finance (#19–25)\n"
    "- **Section 4** — All four networks combined (#26–32)\n"
    "- **Section 5** — Temporal league-level Sankeys (#33–37)\n"
)