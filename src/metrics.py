"""Reusable, cached metric computations for the transfer-network app.

Everything here builds on the Phase 1 data layer (``src.data_layer``) and is
cached with ``st.cache_data`` so pages never recompute on rerun. Functions are
keyed by ``grain`` (``"club"`` / ``"league"``) and, where relevant, ``layer``
(``"movement"`` / ``"finance"``).

Invariants respected (see CLAUDE.md):
  * **Finance reversal** — finance edges run buyer -> seller, so finance
    *out*-strength = spend (money paid) and *in*-strength = revenue (money
    received); net profit = in - out.
  * **Parallel edges aggregated to counts** before PageRank / reciprocity /
    betweenness (movement weight is 1 = one player).
  * **Outside System** at club level is the single node ``OS1`` — labelled by
    node id, optionally excluded from leaderboards and shown separately.
  * Fees are right-skewed -> use **median/percentiles**, never the mean.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from .data_layer import get_edges, get_graph, node_names

OUTSIDE_SYSTEM_ID = "OS1"
POSITIONS = [
    "Goalkeeper", "Centre-Back", "Full-Back", "Central Midfielder",
    "Attacking Mid", "Winger / Wide Attacker", "Striker",
]


# --------------------------------------------------------------------------- #
# Names / labelling
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def node_name_map(grain: str) -> dict[str, str]:
    """``node_id -> name`` merged across both layers of a grain."""
    names: dict[str, str] = {}
    for layer in ("movement", "finance"):
        names.update(node_names(get_graph(f"{layer}_{grain}")))
    return names


def with_names(df: pd.DataFrame, grain: str, id_col: str = "node") -> pd.DataFrame:
    """Attach a ``name`` column (and a display ``label``) keyed by node id.

    Club identity is the node id, not the name (some names map to two ids), so
    ``label`` keeps the id visible: ``"Arsenal FC [11]"`` at club grain, plain
    name at league grain.
    """
    names = node_name_map(grain)
    out = df.copy()
    out["name"] = out[id_col].map(names).fillna(out[id_col])
    if grain == "club":
        out["label"] = out["name"] + " [" + out[id_col].astype(str) + "]"
    else:
        out["label"] = out["name"]
    return out


def drop_outside_system(df: pd.DataFrame, id_col: str = "node") -> pd.DataFrame:
    return df[df[id_col] != OUTSIDE_SYSTEM_ID].copy()


# --------------------------------------------------------------------------- #
# Aggregated simple graphs (parallel edges collapsed)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Aggregating edges…")
def aggregated_edges(layer: str, grain: str) -> pd.DataFrame:
    """One row per directed ``(source, target)`` pair with:
    ``n`` = transfer count, ``fee`` = summed finance weight (NaN for movement)."""
    df = get_edges(f"{layer}_{grain}")
    g = df.groupby(["source", "target"], observed=True)
    agg = g.size().reset_index(name="n")
    if layer == "finance":
        agg["fee"] = g["weight"].sum().to_numpy()
    else:
        agg["fee"] = np.nan
    agg["source"] = agg["source"].astype(str)
    agg["target"] = agg["target"].astype(str)
    return agg


def _digraph(agg: pd.DataFrame, weight_col: str) -> nx.DiGraph:
    return nx.from_pandas_edgelist(
        agg, "source", "target", edge_attr=weight_col, create_using=nx.DiGraph
    )


# --------------------------------------------------------------------------- #
# #1 Degree
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def degree_table(grain: str) -> pd.DataFrame:
    """Movement in/out-degree per node (out = players sold, in = recruited)."""
    df = get_edges(f"movement_{grain}")
    out = df.groupby("source", observed=True).size().rename("out_degree")
    inn = df.groupby("target", observed=True).size().rename("in_degree")
    t = pd.concat([out, inn], axis=1).fillna(0).astype(int)
    t.index = t.index.astype(str)
    t.index.name = "node"
    t = t.reset_index()
    t["net"] = t["in_degree"] - t["out_degree"]  # net importer (+) / exporter (-)
    return with_names(t, grain)


# --------------------------------------------------------------------------- #
# #2 / #11 PageRank
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing PageRank…")
def pagerank_table(layer: str, grain: str, reverse: bool = False) -> pd.DataFrame:
    weight_col = "n" if layer == "movement" else "fee"
    G = _digraph(aggregated_edges(layer, grain), weight_col)
    if reverse:
        G = G.reverse(copy=True)
    pr = nx.pagerank(G, weight=weight_col)
    t = pd.DataFrame({"node": list(pr.keys()), "pagerank": list(pr.values())})
    t = with_names(t, grain).sort_values("pagerank", ascending=False, ignore_index=True)
    t["rank"] = t.index + 1
    return t


# --------------------------------------------------------------------------- #
# #3 Betweenness (movement, club, approximate via k pivots)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Estimating betweenness (k-pivot sample)…")
def betweenness_table(k: int = 500, seed: int = 42) -> pd.DataFrame:
    """Approximate betweenness on the aggregated movement-club graph.

    Hop-distance (unweighted) Brandes with ``k`` pivot sources — an estimate,
    not exact. Club-level only (degenerate at 11 league nodes)."""
    G = _digraph(aggregated_edges("movement", "club"), "n")
    k = min(k, G.number_of_nodes())
    bc = nx.betweenness_centrality(G, k=k, seed=seed, weight=None, normalized=True)
    t = pd.DataFrame({"node": list(bc.keys()), "betweenness": list(bc.values())})
    return with_names(t, "club").sort_values("betweenness", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# #9 / #12 Finance strength (reversal-aware)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def finance_strength_table(grain: str) -> pd.DataFrame:
    """Per-node spend / revenue / net (finance reversal applied).

    spend   = out-strength (buyer->seller, money paid out)
    revenue = in-strength  (money received)
    net_profit = revenue - spend ; net_spend = spend - revenue
    """
    df = get_edges(f"finance_{grain}")
    spend = df.groupby("source", observed=True)["weight"].sum().rename("spend")
    revenue = df.groupby("target", observed=True)["weight"].sum().rename("revenue")
    t = pd.concat([spend, revenue], axis=1).fillna(0.0)
    t.index = t.index.astype(str)
    t.index.name = "node"
    t = t.reset_index()
    t["net_profit"] = t["revenue"] - t["spend"]
    t["net_spend"] = t["spend"] - t["revenue"]
    return with_names(t, grain)


# --------------------------------------------------------------------------- #
# #10 Gini / Lorenz
# --------------------------------------------------------------------------- #
def gini(values) -> float:
    x = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    x = x[~np.isnan(x)]
    x = x[x >= 0]
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def lorenz_points(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    x = x[~np.isnan(x)]
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return np.array([0, 1.0]), np.array([0, 1.0])
    cum = np.insert(np.cumsum(x) / x.sum(), 0, 0.0)
    p = np.linspace(0.0, 1.0, len(cum))
    return p, cum


# --------------------------------------------------------------------------- #
# #5 Reciprocity
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def reciprocity_overall(layer: str, grain: str) -> float:
    G = _digraph(aggregated_edges(layer, grain), "n" if layer == "movement" else "fee")
    if G.number_of_edges() == 0:
        return float("nan")
    return nx.overall_reciprocity(G)


@st.cache_data(show_spinner=False)
def reciprocal_dyads(layer: str, grain: str, top: int = 25) -> pd.DataFrame:
    """Top mutual pairs: both A->B and B->A exist. Strength = min(both directions)."""
    agg = aggregated_edges(layer, grain)
    val = "n" if layer == "movement" else "fee"
    fwd = {(r.source, r.target): getattr(r, val) for r in agg.itertuples(index=False)}
    rows = []
    seen = set()
    for (a, b), w in fwd.items():
        if a == b or (b, a) in seen:
            continue
        if (b, a) in fwd:
            seen.add((a, b))
            rows.append((a, b, w, fwd[(b, a)]))
    if not rows:
        return pd.DataFrame(columns=["a", "b", "a_to_b", "b_to_a", "mutual_min", "total"])
    d = pd.DataFrame(rows, columns=["a", "b", "a_to_b", "b_to_a"])
    d["mutual_min"] = d[["a_to_b", "b_to_a"]].min(axis=1)
    d["total"] = d["a_to_b"] + d["b_to_a"]
    names = node_name_map(grain)
    d["a_name"] = d["a"].map(names).fillna(d["a"])
    d["b_name"] = d["b"].map(names).fillna(d["b"])
    return d.sort_values("mutual_min", ascending=False, ignore_index=True).head(top)


# --------------------------------------------------------------------------- #
# #4 Position-filtered volume (club/league x position)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def position_volume(grain: str, side: str = "out") -> pd.DataFrame:
    """Transfer counts per node x position. ``side`` = out (sold) / in (bought)."""
    df = get_edges(f"movement_{grain}")
    key = "source" if side == "out" else "target"
    t = (
        df.groupby([key, "position"], observed=True).size()
        .reset_index(name="count")
        .rename(columns={key: "node"})
    )
    t["node"] = t["node"].astype(str)
    return with_names(t, grain)


# --------------------------------------------------------------------------- #
# #6 Seasonal degree ; #12 seasonal spend
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def seasonal_degree(grain: str, side: str = "out") -> pd.DataFrame:
    df = get_edges(f"movement_{grain}")
    key = "source" if side == "out" else "target"
    t = (
        df.groupby([key, "season"], observed=True).size()
        .reset_index(name="degree").rename(columns={key: "node"})
    )
    t["node"] = t["node"].astype(str)
    return with_names(t, grain)


@st.cache_data(show_spinner=False)
def seasonal_finance(grain: str) -> pd.DataFrame:
    """Per node x season spend (out) and revenue (in) on finance."""
    df = get_edges(f"finance_{grain}")
    spend = (df.groupby(["source", "season"], observed=True)["weight"].sum()
             .reset_index().rename(columns={"source": "node", "weight": "spend"}))
    rev = (df.groupby(["target", "season"], observed=True)["weight"].sum()
           .reset_index().rename(columns={"target": "node", "weight": "revenue"}))
    t = spend.merge(rev, on=["node", "season"], how="outer").fillna({"spend": 0.0, "revenue": 0.0})
    t["node"] = t["node"].astype(str)
    t["net_spend"] = t["spend"] - t["revenue"]
    return with_names(t, grain)


# --------------------------------------------------------------------------- #
# #7 Window comparison ; #8 position supply trends
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def window_volume(grain: str) -> pd.DataFrame:
    """Per-node transfer volume split by window (out-degree based)."""
    df = get_edges(f"movement_{grain}")
    t = (df.groupby(["source", "window"], observed=True).size()
         .reset_index(name="count").rename(columns={"source": "node"}))
    t["node"] = t["node"].astype(str)
    return with_names(t, grain)


@st.cache_data(show_spinner=False)
def position_supply(grain: str) -> pd.DataFrame:
    """Movement transfer counts per season x position (supply over time)."""
    df = get_edges(f"movement_{grain}")
    return (df.groupby(["season", "position"], observed=True).size()
            .reset_index(name="count"))


# --------------------------------------------------------------------------- #
# #13 / #14 / #23 Fee distributions (finance, edge level)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def fee_edges(grain: str) -> pd.DataFrame:
    """Edge-level finance fees with position/window/season for distributions."""
    df = get_edges(f"finance_{grain}")
    return df[["weight", "position", "window", "season"]].rename(columns={"weight": "fee"})


# --------------------------------------------------------------------------- #
# #15 Ego network (movement + fee via P1), club level
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def club_volume_ranking() -> pd.DataFrame:
    """Clubs by total movement volume (for the #15 club picker)."""
    deg = degree_table("club")
    deg["total"] = deg["in_degree"] + deg["out_degree"]
    return deg.sort_values("total", ascending=False, ignore_index=True)


@st.cache_data(show_spinner="Building ego-network…")
def ego_edges(club_id: str, season: int, window: str) -> pd.DataFrame:
    """Directed movement edges touching ``club_id`` in one (season, window),
    with the per-deal fee attached via the P1 ``transfer_id`` join.

    Returns columns: source, target, position, fee (NaN if no finance edge),
    direction (in/out relative to the club).
    """
    mv = get_edges("movement_club")
    mv = mv[(mv["season"] == season) & (mv["window"] == window)]
    mv = mv[(mv["source"] == club_id) | (mv["target"] == club_id)]
    fin = get_edges("finance_club")[["transfer_id", "weight"]].rename(columns={"weight": "fee"})
    out = mv.merge(fin, on="transfer_id", how="left")
    out["direction"] = np.where(out["source"] == club_id, "out", "in")
    return out[["source", "target", "position", "fee", "direction", "transfer_id"]]
