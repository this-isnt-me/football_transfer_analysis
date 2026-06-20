"""Section 5 — temporal league-level Sankeys (#33–37).

A state-transition Sankey: one node per (league, stage), stages ordered left→
right by (season, window). A window's transfers are the transition stage t→t+1,
aggregated by (season, window, source_league, target_league) summing weight
(players for movement, fees for finance).

Encoding (display spec):
  * #33/#34 every league rendered at every stage (epsilon carry edges keep lanes
    alive); the diagram is **flow, not stock** — no real retention edges.
  * #35 Outside System pinned to the bottom edge of each stage and toggleable;
    tiny flows thresholded into an "Other" band.
  * #36 finance shown **as-is** (paying league → receiving league); the two
    Sankeys are deliberately mirrored — "left = origin of the thing flowing".
  * #37 comparative view stacks both with aligned stage axes + a per-league
    net-flow small-multiple.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import metrics as M
from .data_layer import get_league_names


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _lane_y(leagues: list[str], include_os: bool, has_other: bool) -> dict[str, float]:
    """y per lane (0 top → 1 bottom). OS pinned to the bottom edge, Other above it."""
    real = [l for l in leagues if l not in (M.OUTSIDE_SYSTEM_ID, "Other")]
    y: dict[str, float] = {}
    top = 0.80 if (include_os or has_other) else 0.95
    denom = max(len(real) - 1, 1)
    for i, l in enumerate(real):
        y[l] = 0.04 + (top - 0.04) * i / denom
    if has_other:
        y["Other"] = 0.90
    if include_os:
        y[M.OUTSIDE_SYSTEM_ID] = 0.98
    return y


def build_sankey(layer: str, include_os: bool, thr_pct: float,
                 lo: int, hi: int) -> tuple[go.Figure, dict]:
    """Build one temporal Sankey for ``layer`` over stages [lo, hi]."""
    names = get_league_names()
    cmap = M.league_color_map()
    flows = M.sankey_flows(layer)
    flows = flows[(flows["stage"] >= lo) & (flows["stage"] <= hi)].copy()
    if not include_os:
        flows = flows[(flows["source"] != M.OUTSIDE_SYSTEM_ID) &
                      (flows["target"] != M.OUTSIDE_SYSTEM_ID)]

    # Threshold tiny flows into an "Other" band (percentile of in-view weights).
    hidden = 0.0
    has_other = False
    if thr_pct > 0 and not flows.empty:
        cut = flows["weight"].quantile(thr_pct)
        small = flows["weight"] < cut
        hidden = float(flows.loc[small, "weight"].sum())
        if small.any():
            has_other = True
            flows.loc[small, "source"] = "Other"
            flows.loc[small, "target"] = "Other"
            flows = flows.groupby(["stage", "source", "target"], observed=True)["weight"].sum().reset_index()

    m = hi - lo + 1                       # number of transition windows in view
    n_layers = m + 1                      # node columns 0..m
    leagues = sorted([l for l in names if include_os or l != M.OUTSIDE_SYSTEM_ID])
    if has_other:
        leagues = leagues + ["Other"]
    yman = _lane_y(leagues, include_os, has_other)

    # node index per (league, layer)
    node_idx: dict[tuple[str, int], int] = {}
    node_x, node_y, node_label, node_color, node_custom = [], [], [], [], []
    for j in range(n_layers):
        for lg in leagues:
            node_idx[(lg, j)] = len(node_label)
            node_x.append(j / m if m else 0.0)
            node_y.append(yman.get(lg, 0.5))
            node_label.append(lg if j == 0 else "")     # label only the leftmost column
            node_color.append(cmap.get(lg, "#888"))
            node_custom.append(names.get(lg, lg))

    src, tgt, val, lcolor = [], [], [], []
    for r in flows.itertuples(index=False):
        j = int(r.stage) - lo
        s_idx = node_idx[(r.source, j)]
        t_idx = node_idx[(r.target, j + 1)]
        src.append(s_idx); tgt.append(t_idx); val.append(float(r.weight))
        lcolor.append(_rgba(cmap.get(r.source, "#888"), 0.45))

    # Epsilon carry edges ONLY for nodes that no real flow touches, so inactive
    # leagues' lanes still render (epsilon height) without inflating active nodes.
    present = set(src) | set(tgt)
    eps = max((flows["weight"].median() * 0.02) if not flows.empty else 1.0, 1e-9)
    for (lg, j), idx in node_idx.items():
        if idx in present:
            continue
        if j < m:
            a, b = idx, node_idx[(lg, j + 1)]
        else:
            a, b = node_idx[(lg, j - 1)], idx
        src.append(a); tgt.append(b); val.append(eps)
        lcolor.append("rgba(200,200,200,0.12)")
        present.update((a, b))

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=node_label, x=node_x, y=node_y, color=node_color,
                  pad=8, thickness=12, line=dict(width=0.3, color="#fff"),
                  customdata=node_custom, hovertemplate="%{customdata}<br>%{value:,.0f}<extra></extra>"),
        link=dict(source=src, target=tgt, value=val, color=lcolor,
                  hovertemplate="%{source.customdata} → %{target.customdata}<br>%{value:,.0f}<extra></extra>"),
    ))
    # Stage (window) labels along the top, centred between layers — aligned axes.
    stages = M.sankey_stages().set_index("stage")["label"]
    for s in range(lo, hi + 1):
        j = s - lo
        fig.add_annotation(x=(j + 0.5) / m if m else 0.5, y=1.06, xref="paper", yref="paper",
                           text=stages.get(s, str(s)), showarrow=False, font=dict(size=10),
                           textangle=-20)
    unit = "players" if layer == "movement" else "€"
    flowdir = "selling league → buying league (player path)" if layer == "movement" \
        else "paying league → receiving league (money path, as-is)"
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=70, b=10),
                      title=f"{layer.capitalize()} Sankey — {flowdir} · width = {unit}")
    summary = {"n_flows": int(len(flows)), "total": float(flows["weight"].sum()),
               "hidden_other": hidden, "has_other": has_other, "n_stages": m}
    return fig, summary


# --------------------------------------------------------------------------- #
# #37 companion: net-flow small multiples
# --------------------------------------------------------------------------- #
def _net_flow_smallmultiples(lo: int, hi: int, include_os: bool):
    nf = M.league_net_flows()
    nf = nf[(nf["stage"] >= lo) & (nf["stage"] <= hi)].copy()
    if not include_os:
        nf = nf[nf["league"] != M.OUTSIDE_SYSTEM_ID]
    # normalise each series per league to [-1, 1] for shape comparison
    parts = []
    for lg, g in nf.groupby("league_name"):
        g = g.sort_values("stage").copy()
        for col in ("net_players", "net_money"):
            mx = g[col].abs().max()
            g[col + "_n"] = g[col] / mx if mx else 0.0
        parts.append(g)
    nf = pd.concat(parts, ignore_index=True)
    tidy = nf.melt(id_vars=["league_name", "stage", "label"],
                   value_vars=["net_players_n", "net_money_n"],
                   var_name="metric", value_name="value")
    tidy["metric"] = tidy["metric"].map({"net_players_n": "net players",
                                         "net_money_n": "net money"})
    fig = px.line(tidy.sort_values("stage"), x="stage", y="value", color="metric",
                  facet_col="league_name", facet_col_wrap=3, markers=True,
                  color_discrete_sequence=["#2E91E5", "#E15F99"],
                  title="Net flow per league per window (normalised: net players vs net money)")
    fig.add_hline(y=0, line_color="#bbb", line_width=1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.04))
    return fig


# --------------------------------------------------------------------------- #
# Page render
# --------------------------------------------------------------------------- #
def render():
    stages = M.sankey_stages()
    labels = stages["label"].tolist()
    n = len(labels)
    default_lo = max(0, n - 8)

    st.markdown("##### Controls (shared across both Sankeys)")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        lo_lab, hi_lab = st.select_slider(
            "Stage range (ordered season → window)", options=labels,
            value=(labels[default_lo], labels[-1]), key="range5")
        lo, hi = labels.index(lo_lab), labels.index(hi_lab)
        if lo > hi:
            lo, hi = hi, lo
    with c2:
        include_os = st.toggle("Show Outside System", value=False, key="os5",
                               help="OS1 is pinned to the bottom edge. Off = intra-system "
                                    "inter-league structure; On = true totals.")
    with c3:
        thr_pct = st.slider("Bucket flows below pct", 0.0, 0.8, 0.3, 0.05, key="thr5",
                            help="Flows below this percentile of in-view weight are pooled "
                                 "into an 'Other' band.")
    st.caption("⬅️ **Left = origin of the thing flowing.** Movement follows the *player* "
               "(selling→buying league); finance follows the *money* (paying→receiving "
               "league), shown as-is — so the two diagrams are **mirrored**: the same deal "
               "puts a league on opposite sides. Diagram is **flow, not stock** (only movers "
               "are recorded; no retention edges). Faint grey threads are epsilon lane-"
               "carries so inactive leagues' lanes persist.")

    tab_m, tab_f, tab_c = st.tabs(["Movement Sankey (#33–35)", "Finance Sankey (#36)",
                                   "Comparative view (#37)"])
    with tab_m:
        fig, s = build_sankey("movement", include_os, thr_pct, lo, hi)
        st.plotly_chart(fig, width="stretch")
        _summary_caption(s, "players")
    with tab_f:
        fig, s = build_sankey("finance", include_os, thr_pct, lo, hi)
        st.plotly_chart(fig, width="stretch")
        _summary_caption(s, "€")
    with tab_c:
        st.markdown("**Movement** (player path) and **finance** (money path) with aligned "
                    "stage axes. Read each league per window: big movement *outflow* + big "
                    "finance *inflow* = profitable feeder; big movement *inflow* + finance "
                    "*outflow* = premium buyer; where volume and value diverge = market "
                    "(in)efficiency.")
        figm, _ = build_sankey("movement", include_os, thr_pct, lo, hi)
        figf, _ = build_sankey("finance", include_os, thr_pct, lo, hi)
        figm.update_layout(height=420, title="Movement — player path (selling → buying)")
        figf.update_layout(height=420, title="Finance — money path, as-is (paying → receiving)")
        st.plotly_chart(figm, width="stretch")
        st.plotly_chart(figf, width="stretch")
        st.divider()
        st.plotly_chart(_net_flow_smallmultiples(lo, hi, include_os), width="stretch")
        st.caption("Net players = movement in − out; net money = finance revenue − spend. "
                   "Each series normalised per league to [−1, 1] for shape comparison. "
                   "Opposite-signed lines (e.g. net players ↓ while net money ↑) = a "
                   "selling league cashing in.")


def _summary_caption(s: dict, unit: str):
    msg = f"{s['n_flows']:,} flows across {s['n_stages']} windows · total {s['total']:,.0f} {unit}."
    if s["has_other"]:
        msg += f" {s['hidden_other']:,.0f} {unit} pooled into 'Other'."
    st.caption(msg)
