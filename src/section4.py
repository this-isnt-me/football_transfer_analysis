"""Section 4 — all four networks combined (#26–32).

Finance is aligned to movement by **reversing the finance layer** (so both run
sell→buy along the player path); P1 says which edges correspond, P2 does any
club→league rollup. Heavy work (multilayer #26, community #27/#29) uses igraph +
leidenalg/infomap, on top-N subgraphs where a full draw is impossible — those
are labelled estimates. Community detection is **club-level only** (11 leagues
won't cluster).

Reads from cached ``src.metrics`` helpers — no graph is reloaded.
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import metrics as M
from . import ui
from .ui import PALETTE, QUAL


# --------------------------------------------------------------------------- #
# #26 Multi-layer network analysis
# --------------------------------------------------------------------------- #
def render_26():
    st.subheader("26 · All-Round Heavyweights")
    st.caption("Cross-layer centrality on the **aligned** layers (finance reversed to "
               "sell→buy). x = talent pull (movement PageRank), y = money pull (reversed-"
               "finance PageRank). Top-right = versatile nodes central in *both*.")
    grain = ui.grain_control("26")
    df, stats = M.multilayer_centrality(grain)
    if grain == "club":
        df = M.drop_non_clubs(df)
        n = st.slider("Label top-N versatile", 5, 40, 15, key="n26")
    else:
        n = 11
    st.metric("Inter-layer Spearman ρ (talent vs money)", f"{stats['spearman']:.3f}",
              help=f"p = {stats['spearman_p']:.3g}; nodes in both layers = {stats['n_both']}")

    plot = df[(df["talent"] > 0) & (df["money"] > 0)].copy()
    fig = px.scatter(plot, x="talent", y="money", hover_name="label", color="versatility",
                     color_continuous_scale="Viridis", opacity=0.6, render_mode="webgl",
                     title="Talent vs money centrality (log–log; colour = versatility)")
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    for _, r in df.head(n).iterrows():
        if r["talent"] > 0 and r["money"] > 0:
            fig.add_annotation(x=np.log10(r["talent"]), y=np.log10(r["money"]),
                               text=r["label"].split(" [")[0][:18], font=dict(size=9),
                               showarrow=False, yshift=8, opacity=0.85)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    show = df.head(n)[["label", "talent", "money", "versatility"]].copy()
    show[["talent", "money", "versatility"]] = show[["talent", "money", "versatility"]].round(4)
    st.markdown("**Most versatile nodes (high in both layers):**")
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Finance layer reversed to align with the player path; versatility = "
               "geometric mean of the two percentile ranks.")


# --------------------------------------------------------------------------- #
# #27 Community detection & cross-layer comparison (club only)
# --------------------------------------------------------------------------- #
def render_27():
    st.subheader("27 · Trading Circles")
    st.caption("Trading blocs of clubs, detected per layer (aligned), then compared. "
               "Club-level only. Force layout is a **top-N drawable subgraph** — the full "
               "5,598-node graph is never drawn.")
    c1, c2 = st.columns(2)
    with c1:
        method = st.radio("Method", ["leiden", "infomap"], horizontal=True, key="m27",
                          help="Leiden = modularity; Infomap = flow-based, direction-aware.")
    with c2:
        top_n = st.slider("Subgraph size (top clubs by volume)", 30, 120, 70, step=10, key="tn27")

    comm, stats = M.cross_layer_communities(method)
    cols = st.columns(4)
    cols[0].metric("Movement communities", stats["n_movement_comm"])
    cols[1].metric("Finance communities", stats["n_finance_comm"])
    cols[2].metric("NMI (movement vs finance)", f"{stats['nmi']:.3f}")
    cols[3].metric("ARI", f"{stats['ari']:.3f}")

    nodes, edges = M.community_subgraph(method, top_n)
    fig = go.Figure()
    for r in edges.itertuples(index=False):
        fig.add_trace(go.Scatter(x=[r.x0, r.x1], y=[r.y0, r.y1], mode="lines",
                                 line=dict(width=0.4, color="#ddd"), hoverinfo="skip",
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers", text=nodes["name"], hoverinfo="text",
        marker=dict(size=8 + 2 * np.sqrt(nodes["deg"]),
                    color=nodes["comm"], colorscale="Rainbow", line=dict(width=0.5, color="#333")),
        showlegend=False,
    ))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10),
                      title=f"Top {top_n} clubs — movement communities ({method})",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")

    # Alluvial: movement community vs finance community on shared nodes (top blocs only)
    shared = comm.dropna(subset=["movement_comm", "finance_comm"]).copy()

    def _topk_relabel(col, k=8):
        top = shared[col].value_counts().head(k).index
        return shared[col].where(shared[col].isin(top), other=-1).map(
            lambda v: f"C{int(v)}" if v != -1 else "other")
    shared["mv"] = _topk_relabel("movement_comm")
    shared["fn"] = _topk_relabel("finance_comm")
    al = shared.groupby(["mv", "fn"], observed=True).size().reset_index(name="n")
    fig2 = px.parallel_categories(al, dimensions=["mv", "fn"], color="n",
                                  color_continuous_scale="Viridis",
                                  labels={"mv": "movement bloc", "fn": "finance bloc"},
                                  title="Cross-layer bloc flow (do money & player communities align?)")
    fig2.update_layout(height=460, margin=dict(l=40, r=40, t=50, b=10))
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"NMI {stats['nmi']:.2f} / ARI {stats['ari']:.2f} on {stats['n_shared']:,} shared "
               "clubs — higher = player-trading blocs and money blocs coincide.")


# --------------------------------------------------------------------------- #
# #28 Dominance index
# --------------------------------------------------------------------------- #
def render_28():
    st.subheader("28 · Overall Power Ranking")
    st.caption("Composite of z-scored components across all four nets: net talent gain "
               "(movement in−out), financial muscle (spend−revenue), prestige (movement + "
               "reversed-finance PageRank). Adjust weights to test sensitivity.")
    grain = ui.grain_control("28")
    c1, c2, c3 = st.columns(3)
    with c1:
        wt = st.slider("Weight: net talent", 0.0, 2.0, 1.0, 0.25, key="wt28")
    with c2:
        wm = st.slider("Weight: financial muscle", 0.0, 2.0, 1.0, 0.25, key="wm28")
    with c3:
        wp = st.slider("Weight: prestige", 0.0, 2.0, 1.0, 0.25, key="wp28")
    dom = M.dominance_index(grain, wt, wm, wp)
    if grain == "club":
        if st.checkbox("Exclude non-club nodes (OS1, Without Club, UnknownUnknown)",
                       value=True, key="nc28"):
            dom = M.drop_non_clubs(dom)
    top = dom.head(15)

    fig = px.bar(top, x="dominance", y="label", orientation="h",
                 color="dominance", color_continuous_scale="Plasma",
                 title="Top 15 by composite dominance")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    comp = top[["label", "z_talent", "z_muscle", "z_prestige"]].copy()
    fig2 = px.parallel_coordinates(
        comp.assign(idx=range(len(comp))),
        dimensions=["z_talent", "z_muscle", "z_prestige"], color="idx",
        color_continuous_scale="Plasma",
        labels={"z_talent": "net talent (z)", "z_muscle": "muscle (z)", "z_prestige": "prestige (z)"},
        title="Component breakdown of the leaders (z-scores)")
    fig2.update_layout(height=420, margin=dict(l=60, r=40, t=50, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")
    st.caption("Finance inputs direction-corrected (muscle = spend−revenue; prestige uses "
               "reversed-finance PageRank). Equal weights by default — slide to test robustness.")


# --------------------------------------------------------------------------- #
# #29 Temporal community drift (club)
# --------------------------------------------------------------------------- #
def render_29():
    st.subheader("29 · How Alliances Shift")
    st.caption("Community detection per season on the club movement graph; major "
               "communities tracked across seasons by Jaccard overlap. Streamgraph bands "
               "= persistent trading blocs forming, growing, splitting.")
    c1, c2 = st.columns(2)
    with c1:
        method = st.radio("Method", ["leiden", "infomap"], horizontal=True, key="m29")
    with c2:
        top_k = st.slider("Major communities / season", 3, 8, 5, key="tk29")
    df, stats = M.community_drift(method, top_k=top_k)
    if df.empty:
        st.info("No communities tracked.")
        return
    df = df.copy()
    df["stream"] = "bloc " + df["stream_id"].astype(str)
    fig = px.area(df.sort_values("season"), x="season", y="size", color="stream",
                  color_discrete_sequence=QUAL,
                  title=f"Major community sizes over season ({method})")
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Seasons", stats["n_seasons"])
    c2.metric("Distinct blocs tracked", stats["n_streams"])
    c3.metric("Mean continuity (Jaccard)", f"{stats['mean_jaccard']:.2f}")
    st.caption("Persistent ids assigned by greedy best-Jaccard match to the prior season "
               "(≥0.10); a new band = a bloc with no clear predecessor (formation/split).")


# --------------------------------------------------------------------------- #
# #30 Shock detection & propagation
# --------------------------------------------------------------------------- #
def render_30():
    st.subheader("30 · Market Shocks & Crashes")
    st.caption("Anomalies on the **detrended** (year-on-year) series — both metrics trend "
               "up, so shocks like the 2020 COVID dip show in YoY, not the raw level.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        grain = ui.grain_control("30", default="league")
    with c2:
        metric = st.radio("Metric", ["fee", "volume"], horizontal=True, key="met30")
    with c3:
        thr = st.slider("|robust z| threshold", 1.5, 4.0, 2.0, 0.5, key="thr30")
    with c4:
        excl = st.checkbox("Exclude OS flows", value=False, key="os30",
                           help="External flows can mask domestic shocks.") if grain == "club" else False
    s = M.shock_series(grain, metric, excl)
    flagged = s[s["rz"].abs() > thr]

    yfac = 1e9 if metric == "fee" else 1
    ylab = "total fee (€bn)" if metric == "fee" else "transfer volume"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["season"], y=s["value"] / yfac, mode="lines+markers",
                             line=dict(color=PALETTE[0], width=3), name=ylab))
    fig.add_trace(go.Scatter(x=flagged["season"], y=flagged["value"] / yfac, mode="markers",
                             marker=dict(color="red", size=14, symbol="x"), name="shock"))
    for _, r in flagged.iterrows():
        fig.add_annotation(x=r["season"], y=r["value"] / yfac,
                           text=f"{r['yoy']*100:+.0f}%", font=dict(size=10, color="red"),
                           showarrow=True, arrowcolor="red")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{ylab} per season — {len(flagged)} flagged", yaxis_title=ylab)
    st.plotly_chart(fig, width="stretch")

    if not flagged.empty:
        pick = st.selectbox("Trace propagation for flagged season",
                            flagged["season"].tolist(), key="pick30")
        casc = M.shock_cascade(grain, int(pick))
        if grain == "club":
            casc = M.drop_non_clubs(casc)
        casc = casc.head(15).copy()
        casc["delta_m"] = casc["delta"] / 1e6
        casc = casc.sort_values("delta_m")
        fig2 = px.bar(casc, x="delta_m", y="label", orientation="h",
                      color="delta_m", color_continuous_scale="RdBu_r",
                      title=f"Net-spend change vs prior season — {pick} (€m, who drove it)")
        fig2.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")
    st.caption("Note 2004 is the dataset's first season (a low base → large YoY); read the "
               "early spike as a ramp-up artifact, not a market shock.")


# --------------------------------------------------------------------------- #
# #31 Position-stratified multilayer (league × position)
# --------------------------------------------------------------------------- #
def render_31():
    st.subheader("31 · Who Rules Each Position")
    st.caption("Which leagues own which positional submarket — by **volume** (players) vs "
               "by **value** (fees). Columns normalised within each position so they sum to "
               "1 (share of that submarket).")
    c1, c2, c3 = st.columns(3)
    with c1:
        value = st.radio("Measure", ["volume", "fee"], horizontal=True, key="v31")
    with c2:
        side = st.radio("Side", ["buyer", "seller"], horizontal=True, key="s31")
    with c3:
        norm = st.checkbox("Normalise within position", value=True, key="n31")
    pm = M.position_league_matrix(value, side)
    order = [p for p in M.POSITIONS if p in pm["position"].unique()]
    mat = pm.pivot_table(index="league", columns="position", values="val",
                         fill_value=0).reindex(columns=order)
    if norm:
        mat = mat.divide(mat.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
        scale, fmt = "Blues", ".0%"
    else:
        if value == "fee":
            mat = mat / 1e6
        scale, fmt = "Blues", ".0f"
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(mat, aspect="auto", color_continuous_scale=scale, text_auto=fmt,
                    title=f"{value} ({side} side) — league × position"
                          + (" (share of submarket)" if norm else ""))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=80), xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")
    st.caption("Side = buyer (imports/spend; finance source via reversal) or seller "
               "(exports/revenue). Compare volume vs value to spot leagues that buy cheap "
               "but many, or few but expensive, in a given position.")


# --------------------------------------------------------------------------- #
# #32 Feeder club specialisation
# --------------------------------------------------------------------------- #
def render_32():
    st.subheader("32 · Selling & Feeder Clubs")
    st.caption("Feeder signature = sell many players (movement out) **+ net money IN** "
               "(revenue > spend). Bubble = destination concentration (Herfindahl); colour "
               "= league. Top-right with big bubbles = specialised selling/development clubs.")
    grain = ui.grain_control("32")
    fd = M.feeder_clubs(grain)
    if grain == "club":
        fd = M.drop_non_clubs(fd)
        n = st.slider("Top-X feeders (by players sold, net money in)", 10, 50, 25, key="n32")
    else:
        n = 11
    fd = fd.copy()
    fd["net_m"] = fd["net_money_in"] / 1e6
    feeders = fd[fd["net_money_in"] > 0].nlargest(n, "out_degree")

    fig = px.scatter(feeders, x="out_degree", y="net_m", size="dest_hhi", color="league",
                     hover_name="label", size_max=28, color_discrete_sequence=QUAL,
                     title="Players sold vs net money in (bubble = destination concentration)")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="players sold (movement out-degree)",
                      yaxis_title="net money in (€m)")
    st.plotly_chart(fig, width="stretch")

    tbl = feeders.sort_values("out_degree", ascending=False)[
        ["label", "league", "out_degree", "net_m", "dest_hhi", "top_destinations"]].copy()
    tbl["net_m"] = tbl["net_m"].round(1)
    tbl["dest_hhi"] = tbl["dest_hhi"].round(3)
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.caption("Net money in = finance in-strength − out-strength (revenue − spend; reversal-"
               "aware). Higher Herfindahl = sells to a more concentrated set of destinations.")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ANALYSES = {
    "26 · All-Round Heavyweights": render_26,
    "27 · Trading Circles": render_27,
    "28 · Overall Power Ranking": render_28,
    "29 · How Alliances Shift": render_29,
    "30 · Market Shocks & Crashes": render_30,
    "31 · Who Rules Each Position": render_31,
    "32 · Selling & Feeder Clubs": render_32,
}

# Plain-English, one-line explanations shown above each analysis.
EXPLAIN = {
    "26 · All-Round Heavyweights": "Which clubs are central to both the talent market and the money market at the same time.",
    "27 · Trading Circles": "The natural groups of clubs that trade together a lot, and whether the money groups match the player groups.",
    "28 · Overall Power Ranking": "A single, adjustable score ranking the most dominant clubs across talent, money and prestige.",
    "29 · How Alliances Shift": "How groups of frequently-trading clubs form, grow, split and fade across the seasons.",
    "30 · Market Shocks & Crashes": "Seasons where the market suddenly surged or crashed, and which clubs drove the swing.",
    "31 · Who Rules Each Position": "Which leagues dominate the market for each position, by both player numbers and money spent.",
    "32 · Selling & Feeder Clubs": "Which clubs specialise in developing and selling players for profit, and where they tend to sell them.",
}
