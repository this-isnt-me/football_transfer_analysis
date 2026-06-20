"""Section 2 — cross-network, same type: club vs league (#16–18).

All three analyses compare the **club** and **league** views of the *same*
network type (movement or finance), bridged by the P2 ``(club, season, window)
-> league`` mapping from the data layer. Aggregation never crosses
movement<->finance, so edge direction (and the finance reversal) is preserved.

Reads everything from cached ``src.metrics`` helpers — no graph is reloaded.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import metrics as M
from . import ui
from .data_layer import get_league_names
from .ui import PALETTE


# --------------------------------------------------------------------------- #
# #16 Aggregation consistency check (P2 reconciliation)
# --------------------------------------------------------------------------- #
def render_16():
    st.subheader("#16 — Aggregation Consistency Check (validates P2)")
    st.caption("Relabel each club edge's endpoints with their (season, window) league "
               "via P2, sum by (source league, target league, season, window, position), "
               "and diff against the league network. A clean P2 makes every cell zero.")
    layer = st.radio("Network type", ["movement", "finance"], horizontal=True, key="lyr16")
    rec = M.aggregation_reconciliation(layer)

    if rec["clean"]:
        st.success(
            f"✅ Reconciliation **clean** — the {rec['n_club_edges']:,} {layer} club edges "
            f"roll up to the league network exactly. P2 is validated for this type.",
            icon="✅",
        )
    else:
        st.error(
            f"❌ Reconciliation found {rec['count_mismatch_cells']:,} mismatched cells — "
            "see the flagged rows below.", icon="❌",
        )

    cols = st.columns(5)
    cols[0].metric("Club edges", f"{rec['n_club_edges']:,}")
    cols[1].metric("League edges", f"{rec['n_league_edges']:,}")
    cols[2].metric("Rollup cells", f"{rec['n_rollup_cells']:,}")
    cols[3].metric("Count-mismatch cells", f"{rec['count_mismatch_cells']:,}")
    cols[4].metric("Σ|count diff|", f"{rec['sum_abs_count_diff']:,}")

    cols2 = st.columns(4)
    cols2[0].metric("P2 violations (triple→>1 league)", f"{rec['p2_violations']:,}")
    cols2[1].metric("Unmapped endpoints", f"{rec['unmapped_endpoints']:,}")
    if layer == "finance":
        cols2[2].metric("Fee-mismatch cells", f"{rec['fee_mismatch_cells']:,}")
        cols2[3].metric("Σ|fee diff| (€)", f"{rec['sum_abs_fee_diff']:,.0f}")

    # 11×11 discrepancy heatmap (should be all-zero) — summed |count diff| per league pair.
    names = get_league_names()
    ids = sorted(names)
    mat = pd.DataFrame(0, index=ids, columns=ids, dtype=float)
    mism = rec["mismatches"]
    if not mism.empty:
        agg = mism.groupby(["source_league", "target_league"])["n_diff"].apply(
            lambda s: s.abs().sum())
        for (s, t), v in agg.items():
            if s in mat.index and t in mat.columns:
                mat.loc[s, t] = v
    disp = mat.rename(index=names, columns=names)
    fig = px.imshow(disp, aspect="auto", color_continuous_scale="Reds",
                    title="Discrepancy heatmap — Σ|count diff| by league pair (all-zero = reconciled)",
                    text_auto=True, zmin=0)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="target league", yaxis_title="source league")
    st.plotly_chart(fig, width="stretch")

    if mism.empty:
        st.caption("No flagged cells — every (league pair × season × window × position) cell "
                   "reconciles exactly (diff ≡ 0).")
    else:
        st.markdown("**Flagged cells (rollup ≠ league):**")
        show_cols = ["source", "target", "season", "window", "position", "n_roll", "n_league", "n_diff"]
        if layer == "finance":
            show_cols += ["fee_roll", "fee_league", "fee_diff"]
        st.dataframe(mism[show_cols], hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# #17 Node ranking correlation (league-net vs club-aggregate)
# --------------------------------------------------------------------------- #
_METRICS = {
    "movement": {"pagerank": "PageRank", "degree": "Total degree"},
    "finance": {"pagerank": "PageRank", "spend": "Spend", "revenue": "Revenue"},
}


def render_17():
    st.subheader("#17 — Node Ranking Correlation (broad-based vs star-carried)")
    st.caption("Each league's standing measured two ways: directly on the 11-node "
               "league network, vs the membership-weighted rollup of its member clubs' "
               "metric. A tight diagonal = scale-consistent; off-diagonal leagues are "
               "carried by a few mega-clubs (or vice-versa).")
    c1, c2 = st.columns(2)
    with c1:
        layer = st.radio("Network type", ["movement", "finance"], horizontal=True, key="lyr17")
    with c2:
        opts = _METRICS[layer]
        mkey = st.radio("Metric", list(opts), format_func=lambda k: opts[k],
                        horizontal=True, key=f"m17_{layer}")

    df, stats = M.ranking_correlation(layer, mkey)
    st.caption("Club metrics are split across the leagues a club played in, weighted by "
               "its share of activity there (P2), so league-switchers are attributed fairly.")

    fig = px.scatter(df, x="league_metric", y="club_rollup", text="name",
                     color="name", color_discrete_sequence=PALETTE,
                     title=f"League {opts[mkey]} vs club-rollup — ρ = {stats['spearman']:.3f}")
    fig.update_traces(textposition="top center", marker=dict(size=13))
    # rank-parity diagonal
    m = max(df["league_metric"].max(), df["club_rollup"].max())
    ui.add_parity_line(fig, m)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                      xaxis_title=f"league-network {opts[mkey]}",
                      yaxis_title=f"club rollup {opts[mkey]}")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    c1.metric("Spearman ρ", f"{stats['spearman']:.3f}", help=f"p = {stats['spearman_p']:.3g}")
    c2.metric("Kendall τ", f"{stats['kendall']:.3f}", help=f"p = {stats['kendall_p']:.3g}")
    show = df.copy()
    show["league_rank"] = show["league_metric"].rank(ascending=False).astype(int)
    show["rollup_rank"] = show["club_rollup"].rank(ascending=False).astype(int)
    show["rank_gap"] = show["league_rank"] - show["rollup_rank"]
    st.dataframe(show[["name", "league_metric", "club_rollup", "league_rank", "rollup_rank", "rank_gap"]],
                 hide_index=True, width="stretch")
    st.caption("rank_gap ≠ 0 ⇒ the league ranks differently as an atomic node than as the "
               "sum of its clubs — the divergence #17 is about.")


# --------------------------------------------------------------------------- #
# #18 Temporal divergence between scales
# --------------------------------------------------------------------------- #
_BASE = {"volume": "Movement volume", "spend": "Finance spend", "revenue": "Finance revenue"}
_CONC = {"top_share": "Top-club share", "hhi": "HHI", "gini": "Gini"}


def render_18():
    st.subheader("#18 — Temporal Divergence Between Scales")
    st.caption("League-scale total vs within-league concentration, per season. The classic "
               "pattern: flat/falling league total while concentration rises = rich-get-richer "
               "*inside* the league. Shaded seasons = the two trajectories move opposite ways.")
    names = get_league_names()
    ids = sorted(names, key=lambda i: names[i])
    c1, c2, c3 = st.columns(3)
    with c1:
        league_id = st.selectbox("League", ids, format_func=lambda i: f"{names[i]} [{i}]",
                                 index=ids.index("GB1") if "GB1" in ids else 0, key="lg18")
    with c2:
        base = st.radio("Base metric", list(_BASE), format_func=lambda k: _BASE[k],
                        horizontal=False, key="base18")
    with c3:
        conc = st.radio("Concentration (club scale)", list(_CONC),
                        format_func=lambda k: _CONC[k], key="conc18")

    df, stats = M.temporal_divergence(league_id, base, conc)
    if df.empty or df["league_total"].dropna().empty:
        st.info("No data for this league / metric.")
        return

    yfac = 1e6 if base in ("spend", "revenue") else 1
    ylabel = f"{_BASE[base]} (€m)" if base in ("spend", "revenue") else _BASE[base]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["season"], y=df["league_total"] / yfac, mode="lines+markers",
                             name=f"League {_BASE[base]}", line=dict(color=PALETTE[0], width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["season"], y=df["concentration"], mode="lines+markers",
                             name=f"Club {_CONC[conc]}", line=dict(color=PALETTE[1], width=3)),
                  secondary_y=True)
    for s in df.loc[df["divergent"], "season"]:
        fig.add_vrect(x0=s - 0.5, x1=s + 0.5, fillcolor="orange", opacity=0.12, line_width=0)
    fig.update_yaxes(title_text=ylabel, secondary_y=False)
    fig.update_yaxes(title_text=f"{_CONC[conc]} (0–1)", secondary_y=True, range=[0, 1])
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10),
                      title=f"{names[league_id]} — league {_BASE[base]} vs club {_CONC[conc]}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman ρ (trajectories)", f"{stats['spearman']:.3f}",
              help=f"p = {stats['spearman_p']:.3g}; n = {stats['n_seasons']} seasons")
    c2.metric("Divergence windows", f"{stats['n_divergent']}")
    c3.metric("Seasons covered", f"{stats['n_seasons']}")
    st.caption("Fees are nominal (no deflation) — read trends, not absolute eras. "
               "Shaded = league total and club concentration moved in opposite directions "
               "that season (normalised year-on-year).")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ANALYSES = {
    "#16 — Aggregation Consistency (P2 check)": render_16,
    "#17 — Node Ranking Correlation": render_17,
    "#18 — Temporal Divergence Between Scales": render_18,
}
