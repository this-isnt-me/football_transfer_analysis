"""Section 3 — cross-network, same granularity: movement vs finance (#19–25).

Every analysis joins the two layers through **P1** (the ``transfer_id`` join).
Movement corridors run ``(sell, buy)``; finance runs ``(buy, sell)``, so the
join already aligns the flip — each movement edge carries its own deal fee.

Fee rules (CLAUDE.md + task): the ~22k unmatched moves carry NULL fee and are
**excluded** from every fee statistic (never zeroed); player counts include all
moves; fee stats use the **median** (fees are right-skewed).

Reads from cached ``src.metrics`` helpers — no graph is reloaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import metrics as M
from . import ui
from .ui import PALETTE


# --------------------------------------------------------------------------- #
# #19 Player flow vs money flow
# --------------------------------------------------------------------------- #
def render_19():
    st.subheader("19 · Players Moved vs Money Spent")
    st.caption("Players moved vs money moved per directed corridor (sell → buy). "
               "Top-left = high-volume / low-value feeder pipelines; bottom-right = "
               "low-volume / high-value marquee corridors. Log–log; counts include all "
               "moves, money excludes NULL-fee deals.")
    grain = ui.grain_control("19")
    cf = M.corridor_flow_money(grain)
    paid = cf[cf["total_fee"] > 0].copy()

    c1, _ = st.columns(2)
    with c1:
        min_p = st.slider("Min players in corridor", 1, 30, 3 if grain == "club" else 1, key="mp19")
    sub = paid[paid["n_players"] >= min_p].copy()
    sub["fee_m"] = sub["total_fee"] / 1e6

    fig = px.scatter(
        sub, x="n_players", y="fee_m", hover_name="corridor",
        color="os_involved" if grain == "club" else None,
        color_discrete_map={False: PALETTE[0], True: "#cccccc"},
        opacity=0.55, render_mode="webgl",
        title=f"{len(sub):,} corridors — players vs €m (log–log)",
        labels={"n_players": "players moved", "fee_m": "total fee (€m)",
                "os_involved": "OS involved"},
    )
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    # annotate a few outliers: biggest money and biggest volume
    for _, r in pd.concat([sub.nlargest(5, "fee_m"), sub.nlargest(5, "n_players")]).drop_duplicates("corridor").iterrows():
        fig.add_annotation(x=np.log10(r["n_players"]), y=np.log10(max(r["fee_m"], 1e-3)),
                           text=r["corridor"][:28], showarrow=True, arrowwidth=1,
                           font=dict(size=9), opacity=0.8)
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(paid):,} corridors carry recorded money; "
               f"{len(cf) - len(paid):,} are money-less (free/loan/unrecorded) and omitted "
               "from the log axes.")


# --------------------------------------------------------------------------- #
# #20 Fee per player (edge-level efficiency)
# --------------------------------------------------------------------------- #
def render_20():
    st.subheader("20 · Who Sells the Priciest Players")
    st.caption("Median fee per individual deal (P1). High = quality-over-quantity "
               "sellers/corridors; low = feeders. NULL-fee moves excluded from the median.")
    grain = ui.grain_control("20")
    c1, c2, c3 = st.columns(3)
    with c1:
        by = st.radio("Group by", ["corridor", "seller", "buyer"], horizontal=True, key="by20")
    with c2:
        n = st.slider("Top-X", 5, 40, 20, key="n20")
    with c3:
        min_d = st.slider("Min deals", 1, 25, 5, key="md20",
                          help="Filter thin groups whose median is noise.")

    fpp = M.fee_per_player(grain, by=by, min_deals=min_d)
    if grain == "club" and "os_involved" in fpp.columns:
        if st.checkbox("Exclude Outside-System", value=True, key="osx20"):
            fpp = fpp[~fpp["os_involved"]]
    fpp = fpp.copy()
    fpp["median_m"] = fpp["median_fee"] / 1e6
    label_col = "corridor" if by == "corridor" else "label"
    top = fpp.nlargest(n, "median_fee")
    ui.ranked_bar(top, "median_m", label_col,
                  f"Top {n} {by}s by median fee/player (€m)",
                  height=560, hover_data=["n_deals", "median_m"])

    fees = M.matched_fees(grain) / 1e6
    fig2 = px.histogram(fees, nbins=60, title="Per-deal fee distribution (€m, log x)",
                        color_discrete_sequence=PALETTE)
    fig2.update_xaxes(type="log", title="fee (€m)")
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"Median per-deal fee overall: €{fees.median():.2f}m · "
               f"{len(fees):,} matched deals (median, not mean — fees are skewed).")


# --------------------------------------------------------------------------- #
# #21 Prestige divergence
# --------------------------------------------------------------------------- #
def render_21():
    st.subheader("21 · Talent Magnets vs Cash Magnets")
    st.caption("Gap between movement-PageRank rank and finance-PageRank rank, with "
               "aligned semantics. Large positive gap = high sporting prestige at modest "
               "fees; large negative = pure cash extractor.")
    c1, c2, c3 = st.columns(3)
    with c1:
        grain = ui.grain_control("21")
    with c2:
        lens = st.radio("Lens", ["selling", "buying"], horizontal=True, key="lens21",
                        help="selling: reversed movement vs as-is finance; "
                             "buying: as-is movement vs reversed finance.")
    with c3:
        n = st.slider("Top-X divergent", 5, 30, 15, key="n21")
    df, stats = M.prestige_divergence(grain, lens)
    if grain == "club":
        df = M.drop_non_clubs(df)

    st.metric("Spearman ρ (movement-PR vs finance-PR)", f"{stats['spearman']:.3f}",
              help=f"p = {stats['spearman_p']:.3g}; n = {stats['n']}")

    div = df.reindex(df["abs_gap"].nlargest(n).index).sort_values("rank_gap")
    fig = go.Figure()
    for _, r in div.iterrows():
        fig.add_trace(go.Scatter(x=[r["mv_rank"], r["fn_rank"]], y=[r["label"], r["label"]],
                                 mode="lines", line=dict(color="#bbb", width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=div["mv_rank"], y=div["label"], mode="markers",
                             marker=dict(color=PALETTE[0], size=11), name="movement rank"))
    fig.add_trace(go.Scatter(x=div["fn_rank"], y=div["label"], mode="markers",
                             marker=dict(color=PALETTE[1], size=11), name="finance rank"))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10),
                      title=f"Top {n} divergent {grain}s — movement ↔ finance PageRank rank",
                      xaxis_title="rank (1 = highest)", legend=dict(orientation="h", y=1.02),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dumbbell spans a node's two prestige ranks; a wide span = divergence. "
               "Movement dot far left of finance dot ⇒ talent magnet that isn't a cash magnet.")


# --------------------------------------------------------------------------- #
# #22 Positional fee efficiency over time
# --------------------------------------------------------------------------- #
def render_22():
    st.subheader("22 · Position Price Trends")
    st.caption("Median fee per player by position × season (P1). Reveals positional "
               "inflation cycles. Fees nominal — no deflation applied.")
    grain = ui.grain_control("22", default="league")
    pt = M.positional_fee_time(grain)
    mode = st.radio("Display", ["Heatmap", "Multi-line"], horizontal=True, key="mode22")
    pt = pt.copy()
    pt["median_m"] = pt["median_fee"] / 1e6
    order = [p for p in M.POSITIONS if p in pt["position"].unique()]
    if mode == "Heatmap":
        mat = pt.pivot_table(index="position", columns="season", values="median_m").reindex(order)
        fig = px.imshow(mat, aspect="auto", color_continuous_scale="Viridis", text_auto=".1f",
                        title="Median fee (€m) — position × season")
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    else:
        fig = px.line(pt.sort_values("season"), x="season", y="median_m", color="position",
                      markers=True, category_orders={"position": order},
                      title="Median fee/player (€m) by position",
                      color_discrete_sequence=PALETTE)
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("position agrees on both sides of every matched deal (0 divergences), so "
               "the movement position is authoritative here.")


# --------------------------------------------------------------------------- #
# #23 Window arbitrage
# --------------------------------------------------------------------------- #
def render_23():
    st.subheader("23 · The January Price Premium")
    st.caption("Fee-per-player differences by window, controlled for position (like for "
               "like). The January premium/discount.")
    grain = ui.grain_control("23")
    wf = M.window_fee(grain).dropna(subset=["position"])
    wf = wf.copy()
    wf["fee_m"] = wf["fee"] / 1e6
    order = [p for p in M.POSITIONS if p in wf["position"].unique()]
    fig = px.box(wf, x="position", y="fee_m", color="window", points=False,
                 category_orders={"position": order},
                 color_discrete_sequence=PALETTE,
                 title="Fee/player by window, faceted by position (log €m)")
    fig.update_yaxes(type="log", title="fee (€m)")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=90), xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")

    med = (wf.groupby(["position", "window"], observed=True)["fee"].median() / 1e6).unstack()
    med["winter − summer"] = med.get("winter") - med.get("summer")
    med = med.reindex(order)
    st.dataframe(med.round(2).reset_index(), hide_index=True, width="stretch")
    overall = wf.groupby("window", observed=True)["fee"].median() / 1e6
    st.caption(f"Overall median fee — summer €{overall.get('summer', float('nan')):.2f}m · "
               f"winter €{overall.get('winter', float('nan')):.2f}m (median, fees skewed).")


# --------------------------------------------------------------------------- #
# #24 Motif analysis
# --------------------------------------------------------------------------- #
def render_24():
    st.subheader("24 · Repeating Trade Patterns")
    st.caption("Directed triad census of the aggregated movement graph vs a "
               "degree-preserving random null. Positive z = over-represented motif "
               "(talent escalators 030T, recycling loops 030C, cliques 300).")
    grain = ui.grain_control("24", default="league")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_null = st.slider("Null samples", 10, 50, 20, key="nn24")
    if grain == "club":
        with c2:
            top_k = st.slider("Top-k active clubs", 30, 120, 60, step=10, key="tk24")
        st.warning("Club level is **sampled** — restricted to the top-k most active clubs; "
                   "read as an estimate, not a full club-scale census.", icon="⚠️")
    else:
        top_k = 60
    prof, meta = M.triad_profile(grain, top_k=top_k, n_null=n_null)

    cols = st.columns(4)
    cols[0].metric("Nodes", meta["n_nodes"])
    cols[1].metric("Edges (no self-loops)", f"{meta['n_edges']:,}")
    cols[2].metric("Null samples", meta["n_null"])
    cols[3].metric("Mode", "sampled top-k" if meta["sampled"] else "full enumeration")

    nn = meta["n_nodes"]
    if not meta["sampled"] and meta["n_edges"] >= nn * (nn - 1):
        st.info("The 11-node league movement graph is **complete** (every league trades "
                "both ways with every other), so all triads are saturated 300-cliques and "
                "the null is identical — z ≡ 0. Motif structure is only informative at the "
                "sparse **club** level.", icon="ℹ️")

    fig = px.bar(prof, x="label", y="z", color="z", color_continuous_scale="RdBu",
                 title="Triad z-scores vs degree-preserving null")
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=90), xaxis_tickangle=-35,
                      coloraxis_showscale=False, yaxis_title="z-score (σ vs null)")
    st.plotly_chart(fig, width="stretch")

    show = prof[["triad", "label", "observed", "null_mean", "null_std", "z"]].copy()
    show[["observed", "null_mean", "null_std", "z"]] = show[
        ["observed", "null_mean", "null_std", "z"]].round(2)
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("030T = feed-forward (A→B→C, A→C: talent escalator); 030C = 3-cycle "
               "(money-recycling loop); 300 = complete triad. z computed against "
               f"{meta['n_null']} random edge-swapped graphs.")


# --------------------------------------------------------------------------- #
# #25 Capital flow asymmetry by position
# --------------------------------------------------------------------------- #
def render_25():
    st.subheader("25 · Where Clubs Earn & Spend")
    st.caption("Net money per node × position = revenue (sold) − spend (bought). "
               "Red = net spender on that position, blue = net earner. Reveals "
               "'sell expensive strikers, buy cheap defenders' patterns.")
    grain = ui.grain_control("25")
    n = st.slider("Top-X nodes (by gross money)", 5, 30, 18, key="n25")
    ca = M.capital_asymmetry(grain)
    if grain == "club":
        if st.checkbox("Exclude Outside System (OS1)", value=True, key="os25"):
            ca = M.drop_non_clubs(ca)
    totals = ca.groupby(["node", "label"], observed=True)["gross"].sum().reset_index()
    keep = totals.nlargest(n if grain == "club" else 11, "gross")["node"]
    sub = ca[ca["node"].isin(keep)].copy()
    sub["net_m"] = sub["net"] / 1e6
    order = [p for p in M.POSITIONS if p in sub["position"].unique()]
    mat = sub.pivot_table(index="label", columns="position", values="net_m", fill_value=0).reindex(columns=order)
    row_order = sub.groupby("label", observed=True)["gross"].sum().sort_values(ascending=False).index
    mat = mat.loc[[r for r in row_order if r in mat.index]]
    lim = np.nanmax(np.abs(mat.values)) or 1.0
    fig = px.imshow(mat, aspect="auto", color_continuous_scale="RdBu", zmin=-lim, zmax=lim,
                    text_auto=".0f", title="Net money (€m) by node × position — blue earns, red spends")
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=90), xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")
    st.caption("Centred at parity (0 = balanced trade in that position). Values in €m; "
               "money-out = spend = finance out-strength (reversal handled via deal orientation).")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ANALYSES = {
    "19 · Players Moved vs Money Spent": render_19,
    "20 · Who Sells the Priciest Players": render_20,
    "21 · Talent Magnets vs Cash Magnets": render_21,
    "22 · Position Price Trends": render_22,
    "23 · The January Price Premium": render_23,
    "24 · Repeating Trade Patterns": render_24,
    "25 · Where Clubs Earn & Spend": render_25,
}

# Plain-English, one-line explanations shown above each analysis.
EXPLAIN = {
    "19 · Players Moved vs Money Spent": "For each route between two clubs, how many players moved compared with how much money changed hands.",
    "20 · Who Sells the Priciest Players": "Which clubs and routes sell the priciest players on average.",
    "21 · Talent Magnets vs Cash Magnets": "Clubs that attract top talent without commanding top fees — and the reverse.",
    "22 · Position Price Trends": "How the typical fee for each position has risen or fallen over the years.",
    "23 · The January Price Premium": "Whether the same kind of player costs more in the winter window than in summer.",
    "24 · Repeating Trade Patterns": "Common three-club trading patterns, like talent ladders where players step up from club to club.",
    "25 · Where Clubs Earn & Spend": "For each club, which positions it earns money on and which it spends money on.",
}
