"""Section 1 — Single-network analyses (#1–15).

Each ``render_NN`` function draws one analysis, following the companion display
spec (chart type, top-X, Outside-System handling). They read everything from the
cached ``src.metrics`` / ``src.data_layer`` helpers — no graph is reloaded here.

Per the build instruction, the ONE real node-link drawing in this section is the
#15 squad-rebuilding ego-net; every other club view uses ranked bars / tables /
heatmaps / distributions (club graph has ~5,600 nodes — never drawn whole).
League views (11 nodes) use bars, heatmaps and chord-style matrices.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import metrics as M
from .data_layer import get_edges

PALETTE = px.colors.qualitative.Safe


# --------------------------------------------------------------------------- #
# Shared controls
# --------------------------------------------------------------------------- #
def _grain_control(key: str, default: str = "club") -> str:
    return st.radio(
        "Grain", ["club", "league"], horizontal=True,
        index=0 if default == "club" else 1, key=f"grain_{key}",
    )


def _topx(key: str, default: int = 20, lo: int = 5, hi: int = 40) -> int:
    return st.slider("Top-X", lo, hi, default, key=f"topx_{key}")


def _exclude_os(key: str, grain: str, default: bool = True) -> bool:
    if grain != "club":
        return False
    return st.checkbox(
        "Exclude Outside System (OS1) from rankings", value=default, key=f"os_{key}",
        help="OS1 is the single external-clubs catch-all node; it dominates volume "
             "and is shown separately rather than ranked beside real clubs.",
    )


def _maybe_drop_os(df, grain, exclude, id_col="node"):
    if grain == "club" and exclude:
        os_row = df[df[id_col] == M.OUTSIDE_SYSTEM_ID]
        return M.drop_outside_system(df, id_col), os_row
    return df, df.iloc[0:0]


def _bar(df, x, y, title, color=None, orientation="h"):
    fig = px.bar(df, x=x, y=y, title=title, color=color,
                 color_discrete_sequence=PALETTE, orientation=orientation)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=480)
    if orientation == "h":
        fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #1 In/Out-Degree
# --------------------------------------------------------------------------- #
def render_01():
    st.subheader("#1 — In / Out-Degree")
    st.caption("Movement: out-degree = players sold, in-degree = players recruited. "
               "Net importers (above the diagonal) vs net exporters (below).")
    grain = _grain_control("01")
    c1, c2 = st.columns(2)
    with c1:
        n = _topx("01")
    with c2:
        excl = _exclude_os("01", grain)

    deg = M.degree_table(grain)
    ranked, os_row = _maybe_drop_os(deg, grain, excl)

    col1, col2 = st.columns(2)
    with col1:
        top_out = ranked.nlargest(n, "out_degree")
        _bar(top_out, "out_degree", "label", f"Top {n} — out-degree (players sold)")
    with col2:
        top_in = ranked.nlargest(n, "in_degree")
        _bar(top_in, "in_degree", "label", f"Top {n} — in-degree (players recruited)")

    fig = px.scatter(
        ranked, x="out_degree", y="in_degree", hover_name="label",
        title="In vs Out degree (all nodes) — diagonal = parity",
        opacity=0.5, color_discrete_sequence=PALETTE,
    )
    m = max(ranked["out_degree"].max(), ranked["in_degree"].max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    if not os_row.empty:
        r = os_row.iloc[0]
        st.caption(f"Outside System (OS1) — excluded above: out={int(r.out_degree):,}, "
                   f"in={int(r.in_degree):,}.")


# --------------------------------------------------------------------------- #
# #2 Weighted PageRank (movement)
# --------------------------------------------------------------------------- #
def render_02():
    st.subheader("#2 — Weighted PageRank (movement)")
    st.caption("Talent-pull hierarchy. As-is: prestige of **destinations** (where "
               "players go). Reversed: prestige as a **seller**. Parallel edges are "
               "aggregated to transfer counts first.")
    grain = _grain_control("02")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = _topx("02", 25)
    with c2:
        excl = _exclude_os("02", grain)
    with c3:
        reverse = st.toggle("Reverse (selling prestige)", key="rev02")

    pr = M.pagerank_table("movement", grain, reverse=reverse)
    ranked, os_row = _maybe_drop_os(pr, grain, excl)
    _bar(ranked.head(n), "pagerank", "label",
         f"Top {n} PageRank — {'selling' if reverse else 'destination'} prestige")
    if not os_row.empty:
        st.caption(f"Outside System (OS1) PageRank = {os_row.iloc[0]['pagerank']:.4f} (excluded).")


# --------------------------------------------------------------------------- #
# #3 Betweenness (club only, approximate)
# --------------------------------------------------------------------------- #
def render_03():
    st.subheader("#3 — Betweenness Centrality (brokers)")
    st.warning("Club-level only and **approximate** — Brandes with k pivot sources "
               "on hop-distance. Read as ranks, not exact values.", icon="⚠️")
    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.select_slider("Pivot sources (k)", [200, 300, 500, 750, 1000], value=500, key="k03")
    with c2:
        n = _topx("03", 15)
    with c3:
        excl = _exclude_os("03", "club")
    bt = M.betweenness_table(k=k)
    ranked, os_row = _maybe_drop_os(bt, "club", excl)
    show = ranked.head(n)[["node", "name", "betweenness"]].copy()
    show.insert(0, "rank", range(1, len(show) + 1))
    st.dataframe(show, hide_index=True, width="stretch")
    if not os_row.empty:
        st.caption(f"Outside System (OS1) betweenness ≈ {os_row.iloc[0]['betweenness']:.4f} (excluded).")


# --------------------------------------------------------------------------- #
# #4 Position-filtered subgraph (heatmap)
# --------------------------------------------------------------------------- #
def render_04():
    st.subheader("#4 — Position-Filtered Volume (club × position)")
    st.caption("Specialisation by position. Heatmap of transfer counts; clubs as rows.")
    grain = _grain_control("04")
    c1, c2, c3 = st.columns(3)
    with c1:
        side = st.radio("Side", ["out (sold)", "in (bought)"], horizontal=True, key="side04")
        side = side.split()[0]
    with c2:
        n = _topx("04", 20)
    with c3:
        excl = _exclude_os("04", grain)

    pv = M.position_volume(grain, side=side)
    if grain == "club" and excl:
        pv = M.drop_outside_system(pv)
    totals = pv.groupby(["node", "label"], observed=True)["count"].sum().reset_index()
    top_nodes = totals.nlargest(n, "count")["node"]
    sub = pv[pv["node"].isin(top_nodes)]
    mat = sub.pivot_table(index="label", columns="position", values="count",
                          fill_value=0, aggfunc="sum").reindex(columns=M.POSITIONS, fill_value=0)
    order = mat.sum(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    fig = px.imshow(mat, aspect="auto", color_continuous_scale="Blues",
                    title=f"Top {len(mat)} {grain}s × position — {side} volume", text_auto=True)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #5 Reciprocity
# --------------------------------------------------------------------------- #
def render_05():
    st.subheader("#5 — Reciprocity")
    st.caption("Share of edges with a reciprocal partner — repeat trading corridors / swaps.")
    grain = _grain_control("05")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Movement reciprocity", f"{M.reciprocity_overall('movement', grain):.3f}")
    with c2:
        st.metric("Finance reciprocity", f"{M.reciprocity_overall('finance', grain):.3f}")

    layer = st.radio("Dyads from", ["movement", "finance"], horizontal=True, key="lyr05")
    if grain == "league":
        agg = M.aggregated_edges(layer, "league")
        names = M.node_name_map("league")
        agg = agg[agg["source"] != agg["target"]]
        val = "n" if layer == "movement" else "fee"
        mat = agg.pivot_table(index="source", columns="target", values=val,
                              fill_value=0, aggfunc="sum")
        mat.index = [names.get(i, i) for i in mat.index]
        mat.columns = [names.get(c, c) for c in mat.columns]
        fig = px.imshow(mat, aspect="auto", color_continuous_scale="Magma",
                        title=f"11×11 {layer} flow matrix (source → target)", text_auto=False)
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        dy = M.reciprocal_dyads(layer, "club", top=25)
        if dy.empty:
            st.info("No reciprocal dyads found.")
            return
        show = dy[["a_name", "b_name", "a_to_b", "b_to_a", "mutual_min", "total"]]
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Top reciprocal club dyads (both directions trade). "
                   "mutual_min = strength of the weaker direction.")


# --------------------------------------------------------------------------- #
# #6 Seasonal degree evolution
# --------------------------------------------------------------------------- #
def render_06():
    st.subheader("#6 — Seasonal Degree Evolution")
    grain = _grain_control("06")
    c1, c2, c3 = st.columns(3)
    with c1:
        side = st.radio("Side", ["out (sold)", "in (bought)"], horizontal=True, key="side06").split()[0]
    with c2:
        n = _topx("06", 20)
    with c3:
        excl = _exclude_os("06", grain)
    sd = M.seasonal_degree(grain, side=side)
    if grain == "club" and excl:
        sd = M.drop_outside_system(sd)
    totals = sd.groupby(["node", "label"], observed=True)["degree"].sum().reset_index()

    if grain == "league":
        fig = px.line(sd.sort_values("season"), x="season", y="degree", color="label",
                      title=f"League {side}-degree over season (all 11)",
                      color_discrete_sequence=PALETTE, markers=True)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        top_nodes = totals.nlargest(n, "degree")["node"]
        sub = sd[sd["node"].isin(top_nodes)]
        mat = sub.pivot_table(index="label", columns="season", values="degree",
                              fill_value=0, aggfunc="sum")
        order = mat.sum(axis=1).sort_values(ascending=False).index
        fig = px.imshow(mat.loc[order], aspect="auto", color_continuous_scale="Viridis",
                        title=f"Top {n} clubs × season — {side}-degree")
        fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #7 Window comparison
# --------------------------------------------------------------------------- #
def render_07():
    st.subheader("#7 — Transfer Window Comparison (Summer vs Winter)")
    grain = _grain_control("07")
    c1, c2 = st.columns(2)
    with c1:
        n = _topx("07", 15)
    with c2:
        excl = _exclude_os("07", grain)
    wv = M.window_volume(grain)
    if grain == "club" and excl:
        wv = M.drop_outside_system(wv)
    totals = wv.groupby(["node", "label"], observed=True)["count"].sum().reset_index()
    top_nodes = totals.nlargest(n if grain == "club" else 11, "count")["node"]
    sub = wv[wv["node"].isin(top_nodes)]
    fig = px.bar(sub, x="label", y="count", color="window", barmode="group",
                 title=f"Summer vs Winter outgoing volume — top {len(top_nodes)} {grain}s",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=80), xaxis_tickangle=-40)
    st.plotly_chart(fig, width="stretch")
    tot = wv.groupby("window", observed=True)["count"].sum()
    st.caption(f"Overall split — summer: {int(tot.get('summer', 0)):,} · "
               f"winter: {int(tot.get('winter', 0)):,}.")


# --------------------------------------------------------------------------- #
# #8 Position supply trends over time
# --------------------------------------------------------------------------- #
def render_08():
    st.subheader("#8 — Position Supply Trends Over Time")
    st.caption("Movement volume per position per season (weight = supply).")
    grain = _grain_control("08", default="league")
    ps = M.position_supply(grain)
    mode = st.radio("Display", ["Stacked area", "Multi-line"], horizontal=True, key="mode08")
    if mode == "Stacked area":
        fig = px.area(ps.sort_values("season"), x="season", y="count", color="position",
                      title="Position supply over season (stacked)",
                      color_discrete_sequence=PALETTE)
    else:
        fig = px.line(ps.sort_values("season"), x="season", y="count", color="position",
                      markers=True, title="Position supply over season",
                      color_discrete_sequence=PALETTE)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #9 Out/In-Strength (finance reversal)
# --------------------------------------------------------------------------- #
def render_09():
    st.subheader("#9 — Out / In-Strength (spend vs revenue)")
    st.caption("**Finance reversal:** out-strength = spend (money paid), "
               "in-strength = revenue (money received); net profit = revenue − spend.")
    grain = _grain_control("09")
    c1, c2 = st.columns(2)
    with c1:
        n = _topx("09", 20)
    with c2:
        excl = _exclude_os("09", grain)
    fs = M.finance_strength_table(grain)
    ranked, os_row = _maybe_drop_os(fs, grain, excl)

    col1, col2 = st.columns(2)
    with col1:
        top_spend = ranked.nlargest(n, "spend").copy()
        top_spend["spend_m"] = top_spend["spend"] / 1e6
        _bar(top_spend, "spend_m", "label", f"Top {n} spenders (€m paid)")
    with col2:
        top_rev = ranked.nlargest(n, "revenue").copy()
        top_rev["revenue_m"] = top_rev["revenue"] / 1e6
        _bar(top_rev, "revenue_m", "label", f"Top {n} earners (€m received)")

    sc = ranked.copy()
    sc["spend_m"] = sc["spend"] / 1e6
    sc["revenue_m"] = sc["revenue"] / 1e6
    fig = px.scatter(sc, x="spend_m", y="revenue_m", hover_name="label", opacity=0.5,
                     title="Spend vs revenue (€m) — diagonal = break-even",
                     color_discrete_sequence=PALETTE)
    m = max(sc["spend_m"].max(), sc["revenue_m"].max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    net = pd.concat([ranked.nlargest(15, "net_spend"), ranked.nlargest(15, "net_profit")])
    net = net.drop_duplicates("node")
    net["net_spend_m"] = net["net_spend"] / 1e6
    net = net.sort_values("net_spend_m")
    fig2 = px.bar(net, x="net_spend_m", y="label", orientation="h",
                  title="Net spend (€m): net buyers (+) vs net sellers (−)",
                  color="net_spend_m", color_continuous_scale="RdBu_r")
    fig2.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, width="stretch")
    if not os_row.empty:
        r = os_row.iloc[0]
        st.caption(f"Outside System (OS1) excluded: spend €{r.spend/1e6:,.0f}m, "
                   f"revenue €{r.revenue/1e6:,.0f}m.")


# --------------------------------------------------------------------------- #
# #10 Flow concentration (Gini / Lorenz)
# --------------------------------------------------------------------------- #
def render_10():
    st.subheader("#10 — Flow Concentration (Gini / Lorenz)")
    st.caption("Inequality of flow across nodes. Expect movement < finance "
               "(money is far more concentrated than players).")
    grain = _grain_control("10")
    excl = _exclude_os("10", grain)

    deg = M.degree_table(grain)
    fs = M.finance_strength_table(grain)
    if grain == "club" and excl:
        deg = M.drop_outside_system(deg)
        fs = M.drop_outside_system(fs)

    series = {
        "Movement out-degree": deg["out_degree"].to_numpy(),
        "Movement in-degree": deg["in_degree"].to_numpy(),
        "Finance spend": fs["spend"].to_numpy(),
        "Finance revenue": fs["revenue"].to_numpy(),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"), name="equality"))
    cols = st.columns(len(series))
    for (name, vals), col, color in zip(series.items(), cols, PALETTE):
        p, c = M.lorenz_points(vals)
        fig.add_trace(go.Scatter(x=p, y=c, mode="lines", name=name, line=dict(color=color)))
        col.metric(f"Gini — {name}", f"{M.gini(vals):.3f}")
    fig.update_layout(title="Lorenz curves", height=520, xaxis_title="cumulative share of nodes",
                      yaxis_title="cumulative share of flow", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #11 Weighted PageRank on finance
# --------------------------------------------------------------------------- #
def render_11():
    st.subheader("#11 — Weighted PageRank on Finance")
    st.caption("Finance edges run buyer → seller. As-is: **selling power** "
               "(extracting big fees from wealthy buyers). Reversed: buying/spending prestige.")
    grain = _grain_control("11")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = _topx("11", 25)
    with c2:
        excl = _exclude_os("11", grain)
    with c3:
        reverse = st.toggle("Reverse (buying prestige)", key="rev11")
    pr = M.pagerank_table("finance", grain, reverse=reverse)
    ranked, os_row = _maybe_drop_os(pr, grain, excl)
    _bar(ranked.head(n), "pagerank", "label",
         f"Top {n} finance PageRank — {'buying' if reverse else 'selling'} prestige")
    if not os_row.empty:
        st.caption(f"Outside System (OS1) PageRank = {os_row.iloc[0]['pagerank']:.4f} (excluded).")


# --------------------------------------------------------------------------- #
# #12 Seasonal spending trajectory
# --------------------------------------------------------------------------- #
def render_12():
    st.subheader("#12 — Seasonal Spending Trajectory")
    st.caption("Per-node spend / revenue across seasons. Fees are nominal — "
               "compare trends, not absolute eras (no deflation applied).")
    grain = _grain_control("12")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.radio("Metric", ["spend", "revenue", "net_spend"], horizontal=True, key="m12")
    with c2:
        n = _topx("12", 10)
    with c3:
        excl = _exclude_os("12", grain)
    sf = M.seasonal_finance(grain)
    if grain == "club" and excl:
        sf = M.drop_outside_system(sf)
    totals = sf.groupby(["node", "label"], observed=True)[metric].sum().abs().reset_index()
    keep = totals.nlargest(n if grain == "club" else 11, metric)["node"]
    sub = sf[sf["node"].isin(keep)].copy()
    sub[metric + "_m"] = sub[metric] / 1e6
    fig = px.line(sub.sort_values("season"), x="season", y=metric + "_m", color="label",
                  markers=True, title=f"{metric} (€m) over season",
                  color_discrete_sequence=PALETTE)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# #13 Winter vs Summer financial comparison
# --------------------------------------------------------------------------- #
def render_13():
    st.subheader("#13 — Winter vs Summer Financial Comparison")
    grain = _grain_control("13")
    fe = M.fee_edges(grain)
    c1, c2 = st.columns(2)
    with c1:
        tot = fe.groupby("window")["fee"].sum() / 1e6
        fig = px.bar(tot.reset_index(), x="window", y="fee",
                     title="Total fee volume by window (€m)",
                     color="window", color_discrete_sequence=PALETTE)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(fe, x="window", y="fee", title="Fee distribution by window (log scale)",
                     color="window", color_discrete_sequence=PALETTE, points=False)
        fig.update_yaxes(type="log")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    med = fe.groupby("window")["fee"].median()
    st.caption(f"Median fee — summer: €{med.get('summer', float('nan'))/1e6:,.2f}m · "
               f"winter: €{med.get('winter', float('nan'))/1e6:,.2f}m (median, not mean — fees are skewed).")


# --------------------------------------------------------------------------- #
# #14 Position-based valuation
# --------------------------------------------------------------------------- #
def render_14():
    st.subheader("#14 — Position-Based Valuation")
    st.caption("Fee distribution by position (median-sorted). Fees right-skewed → "
               "box/violin on a log scale; report medians.")
    grain = _grain_control("14")
    fe = M.fee_edges(grain).dropna(subset=["position"])
    order = fe.groupby("position")["fee"].median().sort_values(ascending=False).index.tolist()
    kind = st.radio("Chart", ["Box", "Violin"], horizontal=True, key="k14")
    if kind == "Box":
        fig = px.box(fe, x="position", y="fee", category_orders={"position": order},
                     color="position", color_discrete_sequence=PALETTE, points=False)
    else:
        fig = px.violin(fe, x="position", y="fee", category_orders={"position": order},
                        color="position", color_discrete_sequence=PALETTE, box=True, points=False)
    fig.update_yaxes(type="log", title="fee (log €)")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=80),
                      xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    med = (fe.groupby("position")["fee"].median() / 1e6).reindex(order).round(2)
    st.dataframe(med.reset_index().rename(columns={"fee": "median fee (€m)"}),
                 hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# #15 Squad rebuilding ego-net (the one node-link drawing)
# --------------------------------------------------------------------------- #
def render_15():
    st.subheader("#15 — Squad Rebuilding (ego-network)")
    st.caption("A single club's buys (in) and sells (out) for one window. "
               "Edge colour = position, width = fee (via the P1 transfer_id join). "
               "This is the only full node-link drawing in Section 1.")
    ranking = M.club_volume_ranking()
    ranking = M.drop_outside_system(ranking)
    top = ranking.head(60)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        choice = st.selectbox("Club", top["label"].tolist(), key="club15")
        club_id = top.loc[top["label"] == choice, "node"].iloc[0]
    edges_all = get_edges("movement_club")
    club_edges = edges_all[(edges_all["source"] == club_id) | (edges_all["target"] == club_id)]
    with c2:
        seasons = sorted(club_edges["season"].dropna().unique().tolist())
        if not seasons:
            st.info("No transfers for this club.")
            return
        season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="season15")
    with c3:
        windows = sorted(club_edges[club_edges["season"] == season]["window"].dropna().unique())
        window = st.selectbox("Window", windows, key="window15")

    ego = M.ego_edges(club_id, int(season), window)
    if ego.empty:
        st.info("No transfers in this season/window for the selected club.")
        return

    names = M.node_name_map("club")
    G = nx.DiGraph()
    G.add_node(club_id)
    for r in ego.itertuples(index=False):
        G.add_edge(r.source, r.target, position=r.position, fee=r.fee)
    pos = nx.spring_layout(G, seed=1, k=0.9)

    fee_vals = ego["fee"].dropna()
    fmax = fee_vals.max() if not fee_vals.empty else 1.0
    pos_colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(M.POSITIONS)}

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        fee = d.get("fee")
        width = 1.5 + 6 * (fee / fmax) if pd.notna(fee) and fmax else 1.0
        label = f"{names.get(u, u)} → {names.get(v, v)}<br>{d.get('position')}<br>" + \
                (f"€{fee/1e6:,.1f}m" if pd.notna(fee) else "fee: NULL (free/unrecorded)")
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=width, color=pos_colors.get(d.get("position"), "#999")),
            hoverinfo="text", text=label, showlegend=False,
        ))
    node_x, node_y, node_text, node_size = [], [], [], []
    for nd in G.nodes():
        x, y = pos[nd]
        node_x.append(x); node_y.append(y)
        node_text.append(names.get(nd, nd))
        node_size.append(28 if nd == club_id else 12)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        marker=dict(size=node_size, color="#222"), hoverinfo="text", showlegend=False,
    )
    fig = go.Figure(edge_traces + [node_trace])
    # legend proxies for positions
    for p, col in pos_colors.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=col, width=4), name=p))
    fig.update_layout(
        title=f"{choice} — {window} {season}  ({len(ego)} transfers; line width = fee)",
        height=640, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, width="stretch")
    n_null = int(ego["fee"].isna().sum())
    st.caption(f"{len(ego)} transfers · {len(ego) - n_null} with a recorded fee · "
               f"{n_null} with NULL fee (free/loan/unrecorded — not drawn as €0).")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ANALYSES = {
    "#1 — In/Out-Degree": render_01,
    "#2 — Weighted PageRank (movement)": render_02,
    "#3 — Betweenness Centrality": render_03,
    "#4 — Position-Filtered Subgraph": render_04,
    "#5 — Reciprocity": render_05,
    "#6 — Seasonal Degree Evolution": render_06,
    "#7 — Window Comparison": render_07,
    "#8 — Position Supply Trends": render_08,
    "#9 — Out/In-Strength": render_09,
    "#10 — Flow Concentration (Gini)": render_10,
    "#11 — Finance PageRank": render_11,
    "#12 — Seasonal Spending Trajectory": render_12,
    "#13 — Winter vs Summer Financial": render_13,
    "#14 — Position-Based Valuation": render_14,
    "#15 — Squad Rebuilding (ego-net)": render_15,
}
