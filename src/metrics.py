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

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from .data_layer import (
    get_edges,
    get_graph,
    get_league_names,
    get_p1,
    get_p2,
    node_names,
    p2_violations,
)

OUTSIDE_SYSTEM_ID = "OS1"
# Club-level pseudo-nodes that are NOT real competitor clubs. OS1 (Outside
# System) is documented; "Without Club" (free agency) and "UnknownUnknown" were
# discovered in the data (they otherwise top dominance/feeder rankings). Key by
# node id — labelling by name is unsafe (names are not unique at club level).
NON_CLUB_IDS = {OUTSIDE_SYSTEM_ID, "515", "75"}  # OS1, Without Club, UnknownUnknown
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


def drop_non_clubs(df: pd.DataFrame, id_col: str = "node") -> pd.DataFrame:
    """Drop all non-club pseudo-nodes (OS1, Without Club, UnknownUnknown)."""
    return df[~df[id_col].isin(NON_CLUB_IDS)].copy()


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


# =========================================================================== #
# Section 2 — cross-network, same type: CLUB vs LEAGUE via P2
#
# Aggregation stays *within one type* (movement->movement, finance->finance) so
# the edge direction (and the finance reversal) is preserved end-to-end. The P2
# map only relabels endpoints from club id to league id at the right
# (season, window); it never crosses movement<->finance.
# =========================================================================== #
@st.cache_data(show_spinner=False)
def p2_lookup() -> pd.DataFrame:
    """1:1 ``(club, season, window) -> league`` lookup.

    P2 resolves each triple to exactly one league (see ``p2_violation_count``);
    we drop_duplicates defensively so an ambiguity could never duplicate edges
    in a rollup (it would instead surface as a reconciliation mismatch)."""
    p2 = get_p2()
    return p2.drop_duplicates(["club", "season", "window"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def p2_violation_count() -> int:
    """Number of ``(club, season, window)`` triples mapping to >1 league (should be 0)."""
    return int(len(p2_violations(get_p2())))


def _relabel_to_league(edges: pd.DataFrame) -> pd.DataFrame:
    """Add ``source_league`` / ``target_league`` to club edges via the P2 lookup,
    matched on the endpoint's ``(club, season, window)``."""
    lut = p2_lookup()
    src = lut.rename(columns={"club": "source", "league": "source_league"})
    tgt = lut.rename(columns={"club": "target", "league": "target_league"})
    e = edges.merge(src, on=["source", "season", "window"], how="left")
    e = e.merge(tgt, on=["target", "season", "window"], how="left")
    return e


# --------------------------------------------------------------------------- #
# #16 Aggregation consistency check (P2 reconciliation)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reconciling club→league rollup (P2)…")
def aggregation_reconciliation(layer: str) -> dict:
    """Roll club edges up to leagues via P2 and diff against the league network.

    Relabel each club edge's endpoints with their ``(season, window)`` league,
    group by ``(source_league, target_league, season, window, position)`` and
    sum (count for movement; count + summed fee for finance), then full-outer
    compare to the league network grouped identically. A clean P2 makes every
    diff exactly zero. Returns a summary dict plus a (hopefully empty) frame of
    mismatched cells with league names attached.
    """
    club = get_edges(f"{layer}_club")
    league = get_edges(f"{layer}_league")
    e = _relabel_to_league(club)
    unmapped = int(e["source_league"].isna().sum() + e["target_league"].isna().sum())

    keys = ["source_league", "target_league", "season", "window", "position"]
    lkeys = ["source", "target", "season", "window", "position"]
    if layer == "movement":
        roll = e.groupby(keys, observed=True).size().reset_index(name="n_roll")
        lk = league.groupby(lkeys, observed=True).size().reset_index(name="n_league")
    else:
        roll = (e.groupby(keys, observed=True)
                .agg(n_roll=("transfer_id", "size"), fee_roll=("weight", "sum")).reset_index())
        lk = (league.groupby(lkeys, observed=True)
              .agg(n_league=("transfer_id", "size"), fee_league=("weight", "sum")).reset_index())
    lk = lk.rename(columns={"source": "source_league", "target": "target_league"})

    m = roll.merge(lk, on=keys, how="outer")
    m["n_roll"] = m["n_roll"].fillna(0).astype(int)
    m["n_league"] = m["n_league"].fillna(0).astype(int)
    m["n_diff"] = m["n_roll"] - m["n_league"]
    bad_mask = m["n_diff"] != 0
    if layer == "finance":
        m["fee_roll"] = m["fee_roll"].fillna(0.0)
        m["fee_league"] = m["fee_league"].fillna(0.0)
        m["fee_diff"] = m["fee_roll"] - m["fee_league"]
        bad_mask = bad_mask | (m["fee_diff"].abs() > 1e-3)

    names = get_league_names()
    mism = m[bad_mask].copy()
    mism["source"] = mism["source_league"].map(names).fillna(mism["source_league"])
    mism["target"] = mism["target_league"].map(names).fillna(mism["target_league"])

    summary = {
        "layer": layer,
        "n_club_edges": int(len(club)),
        "n_league_edges": int(len(league)),
        "p2_violations": p2_violation_count(),
        "unmapped_endpoints": unmapped,
        "n_rollup_cells": int(len(m)),
        "count_mismatch_cells": int(bad_mask.sum()),
        "sum_abs_count_diff": int(m["n_diff"].abs().sum()),
        "mismatches": mism.reset_index(drop=True),
    }
    if layer == "finance":
        summary["fee_mismatch_cells"] = int((m["fee_diff"].abs() > 1e-3).sum())
        summary["sum_abs_fee_diff"] = float(m["fee_diff"].abs().sum())
    summary["clean"] = (
        summary["count_mismatch_cells"] == 0
        and summary["unmapped_endpoints"] == 0
        and summary["p2_violations"] == 0
        and summary.get("fee_mismatch_cells", 0) == 0
    )
    return summary


# --------------------------------------------------------------------------- #
# #17 Node ranking correlation (league-net metric vs club-aggregate)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def club_league_membership(layer: str) -> pd.DataFrame:
    """Per ``(club, league)`` the fraction of the club's endpoint appearances (in
    ``layer`` edges) that fall in that league — its time-weighted membership.

    Used to allocate a club node-metric across the leagues it played in, so
    league-switchers are split proportionally rather than force-assigned."""
    e = _relabel_to_league(get_edges(f"{layer}_club"))
    a = e[["source", "source_league"]].rename(columns={"source": "club", "source_league": "league"})
    b = e[["target", "target_league"]].rename(columns={"target": "club", "target_league": "league"})
    appear = pd.concat([a, b], ignore_index=True).dropna()
    w = appear.groupby(["club", "league"], observed=True).size().rename("appearances").reset_index()
    w["frac"] = w["appearances"] / w.groupby("club")["appearances"].transform("sum")
    return w


def _club_node_metric(layer: str, metric: str) -> pd.DataFrame:
    """Per-club value of ``metric`` -> columns ``club``, ``cval``."""
    if metric == "pagerank":
        t = pagerank_table(layer, "club")[["node", "pagerank"]].rename(columns={"pagerank": "cval"})
    elif metric == "degree":
        d = degree_table("club")
        t = d.assign(cval=d["in_degree"] + d["out_degree"])[["node", "cval"]]
    else:  # spend / revenue
        fs = finance_strength_table("club")
        t = fs[["node", metric]].rename(columns={metric: "cval"})
    return t.rename(columns={"node": "club"})


def _league_node_metric(layer: str, metric: str) -> pd.DataFrame:
    """Per-league (11-node network) value of ``metric`` -> ``node``, ``league_metric``."""
    if metric == "pagerank":
        return pagerank_table(layer, "league")[["node", "pagerank"]].rename(
            columns={"pagerank": "league_metric"})
    if metric == "degree":
        d = degree_table("league")
        return d.assign(league_metric=d["in_degree"] + d["out_degree"])[["node", "league_metric"]]
    fs = finance_strength_table("league")
    return fs[["node", metric]].rename(columns={metric: "league_metric"})


@st.cache_data(show_spinner="Correlating club rollup vs league metric…")
def ranking_correlation(layer: str, metric: str) -> tuple[pd.DataFrame, dict]:
    """League-net metric vs the membership-weighted rollup of member-club metrics.

    Returns ``(frame, stats)`` where ``frame`` has 11 rows (one per league) with
    ``league_metric`` and ``club_rollup``, and ``stats`` holds Spearman/Kendall."""
    from scipy.stats import kendalltau, spearmanr

    lm = _league_node_metric(layer, metric)
    cm = _club_node_metric(layer, metric)
    mem = club_league_membership(layer).merge(cm, on="club", how="left")
    mem["contrib"] = mem["cval"].fillna(0.0) * mem["frac"]
    rolled = mem.groupby("league", observed=True)["contrib"].sum().rename("club_rollup").reset_index()

    out = lm.merge(rolled, left_on="node", right_on="league", how="outer")
    out["club_rollup"] = out["club_rollup"].fillna(0.0)
    out["league_metric"] = out["league_metric"].fillna(0.0)
    names = node_name_map("league")
    out["name"] = out["node"].map(names).fillna(out["node"])
    out = out[["node", "name", "league_metric", "club_rollup"]].sort_values(
        "league_metric", ascending=False, ignore_index=True)

    rho, p_rho = spearmanr(out["league_metric"], out["club_rollup"])
    tau, p_tau = kendalltau(out["league_metric"], out["club_rollup"])
    stats = {"spearman": float(rho), "spearman_p": float(p_rho),
             "kendall": float(tau), "kendall_p": float(p_tau), "n": int(len(out))}
    return out, stats


# --------------------------------------------------------------------------- #
# #18 Temporal divergence between scales
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def club_league_by_season() -> pd.DataFrame:
    """``(club, season) -> league`` resolving the (rare) cross-window case by mode."""
    p2 = get_p2()
    g = (p2.groupby(["club", "season", "league"], observed=True).size()
         .reset_index(name="n").sort_values("n", ascending=False)
         .drop_duplicates(["club", "season"]))
    return g[["club", "season", "league"]].reset_index(drop=True)


def _club_season_metric(metric: str) -> pd.DataFrame:
    """Per ``(club, season)`` value of the base metric -> ``club``, ``season``, ``value``."""
    if metric == "volume":
        df = get_edges("movement_club")
        out = df.groupby(["source", "season"], observed=True).size().rename_axis(["club", "season"])
        inn = df.groupby(["target", "season"], observed=True).size().rename_axis(["club", "season"])
        s = out.add(inn, fill_value=0).rename("value").reset_index()
    else:  # spend / revenue (finance reversal: spend=out/source, revenue=in/target)
        df = get_edges("finance_club")
        col = "source" if metric == "spend" else "target"
        s = (df.groupby([col, "season"], observed=True)["weight"].sum()
             .rename_axis(["club", "season"]).rename("value").reset_index())
    return s


def _league_season_metric(metric: str) -> pd.DataFrame:
    """Per ``(league, season)`` total from the league network -> ``league``, ``season``, ``league_total``."""
    if metric == "volume":
        df = get_edges("movement_league")
        out = df.groupby(["source", "season"], observed=True).size().rename_axis(["league", "season"])
        inn = df.groupby(["target", "season"], observed=True).size().rename_axis(["league", "season"])
        s = out.add(inn, fill_value=0).rename("league_total").reset_index()
    else:
        df = get_edges("finance_league")
        col = "source" if metric == "spend" else "target"
        s = (df.groupby([col, "season"], observed=True)["weight"].sum()
             .rename_axis(["league", "season"]).rename("league_total").reset_index())
    return s


def _hhi(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[v > 0]
    if v.sum() == 0:
        return float("nan")
    s = v / v.sum()
    return float((s ** 2).sum())


@st.cache_data(show_spinner="Computing cross-scale trajectories…")
def temporal_divergence(league_id: str, metric: str, concentration: str) -> tuple[pd.DataFrame, dict]:
    """Per-season league-scale total vs club-scale concentration *within* that league.

    ``concentration`` in {``top_share``, ``hhi``, ``gini``} over member clubs.
    A season is a *divergence window* when the two series (min-max normalised)
    move in opposite year-on-year directions. Returns ``(frame, stats)``."""
    from scipy.stats import spearmanr

    lt = _league_season_metric(metric)
    lt = lt[lt["league"] == league_id][["season", "league_total"]]

    csm = _club_season_metric(metric).merge(club_league_by_season(), on=["club", "season"], how="left")
    csm = csm[csm["league"] == league_id]

    rows = []
    for season, grp in csm.groupby("season", observed=True):
        vals = grp["value"].to_numpy()
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        if concentration == "top_share":
            conc = float(vals.max() / vals.sum())
        elif concentration == "hhi":
            conc = _hhi(vals)
        else:
            conc = gini(vals)
        rows.append({"season": int(season), "concentration": conc, "n_clubs": int(vals.size)})
    conc_df = pd.DataFrame(rows)

    df = lt.merge(conc_df, on="season", how="outer").sort_values("season", ignore_index=True)
    df["season"] = df["season"].astype(int)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0

    df["league_norm"] = _norm(df["league_total"].fillna(0.0))
    df["conc_norm"] = _norm(df["concentration"].fillna(0.0))
    # divergence: YoY directions disagree
    dl = df["league_norm"].diff()
    dc = df["conc_norm"].diff()
    df["divergent"] = (np.sign(dl) * np.sign(dc) < 0)

    valid = df.dropna(subset=["league_total", "concentration"])
    if len(valid) >= 3:
        rho, p = spearmanr(valid["league_total"], valid["concentration"])
    else:
        rho, p = float("nan"), float("nan")
    stats = {"spearman": float(rho), "spearman_p": float(p),
             "n_seasons": int(len(valid)), "n_divergent": int(df["divergent"].sum())}
    return df, stats


# =========================================================================== #
# Section 3 — cross-network, same granularity: MOVEMENT vs FINANCE via P1
#
# P1 attaches each deal's fee to its movement edge by transfer_id. The movement
# corridor is (sell=source, buy=target); finance runs (buy, sell), so the join
# already aligns the flip. The ~22k unmatched moves carry NULL fee — pandas
# median/mean skip NaN, so fee stats exclude them automatically (never zeroed),
# while player counts include every move.
# =========================================================================== #
def _label_corridor(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Attach ``source_label`` / ``target_label`` / ``corridor`` and an OS flag."""
    names = node_name_map(grain)
    out = df.copy()
    for end in ("source", "target"):
        nm = out[end].map(names).fillna(out[end])
        out[f"{end}_label"] = nm + " [" + out[end].astype(str) + "]" if grain == "club" else nm
    out["corridor"] = out["source_label"] + " → " + out["target_label"]
    if grain == "club":
        out["os_involved"] = (out["source"] == OUTSIDE_SYSTEM_ID) | (out["target"] == OUTSIDE_SYSTEM_ID)
    else:
        out["os_involved"] = False
    return out


# --------------------------------------------------------------------------- #
# #19 Player flow vs money flow (per corridor)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Aggregating corridors (P1)…")
def corridor_flow_money(grain: str) -> pd.DataFrame:
    """Per movement corridor ``(sell, buy)``: player count (all moves) vs money
    (summed/median matched fee). Money excludes NULL-fee moves; counts include all."""
    p1 = get_p1(grain)
    g = p1.groupby(["source", "target"], observed=True)
    out = g.agg(
        n_players=("transfer_id", "size"),
        n_paid=("matched", "sum"),
        total_fee=("fee", "sum"),
        median_fee=("fee", "median"),
    ).reset_index()
    out["n_paid"] = out["n_paid"].astype(int)
    out["total_fee"] = out["total_fee"].fillna(0.0)
    return _label_corridor(out, grain)


# --------------------------------------------------------------------------- #
# #20 Fee per player (edge-level efficiency)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def fee_per_player(grain: str, by: str = "corridor", min_deals: int = 3) -> pd.DataFrame:
    """Median fee per deal grouped by corridor / seller / buyer (matched deals only).

    ``min_deals`` filters out thin groups whose median is noise. Median (not mean)
    per the skew rule; the ~22k NULL-fee moves never enter the stat."""
    matched = get_p1(grain)
    matched = matched[matched["matched"]]
    if by == "corridor":
        keys = ["source", "target"]
    elif by == "seller":
        keys = ["source"]
    else:  # buyer
        keys = ["target"]
    g = matched.groupby(keys, observed=True)
    out = g.agg(median_fee=("fee", "median"), mean_fee=("fee", "mean"),
                n_deals=("fee", "size"), total_fee=("fee", "sum")).reset_index()
    out = out[out["n_deals"] >= min_deals]
    if by == "corridor":
        return _label_corridor(out, grain)
    out = out.rename(columns={keys[0]: "node"})
    out["node"] = out["node"].astype(str)
    res = with_names(out, grain)
    res["os_involved"] = res["node"] == OUTSIDE_SYSTEM_ID if grain == "club" else False
    return res


@st.cache_data(show_spinner=False)
def matched_fees(grain: str) -> pd.Series:
    """All per-deal matched fees (for the #20 distribution histogram)."""
    p1 = get_p1(grain)
    return p1.loc[p1["matched"], "fee"].dropna()


# --------------------------------------------------------------------------- #
# #21 Prestige divergence (movement-PR rank vs finance-PR rank)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing prestige divergence…")
def prestige_divergence(grain: str, lens: str = "selling") -> tuple[pd.DataFrame, dict]:
    """Rank gap between movement- and finance-PageRank with aligned semantics.

    ``selling``: reversed movement (selling prestige) vs as-is finance (selling power).
    ``buying`` : as-is movement (destination pull) vs reversed finance (buying prestige).
    Returns ``(frame, stats)``; ``rank_gap`` = movement_rank − finance_rank (large
    positive ⇒ talent magnet that is not a cash magnet)."""
    from scipy.stats import spearmanr

    if lens == "selling":
        mv = pagerank_table("movement", grain, reverse=True)
        fn = pagerank_table("finance", grain, reverse=False)
    else:
        mv = pagerank_table("movement", grain, reverse=False)
        fn = pagerank_table("finance", grain, reverse=True)
    mv = mv[["node", "label", "pagerank"]].rename(columns={"pagerank": "mv_pr"})
    fn = fn[["node", "pagerank"]].rename(columns={"pagerank": "fn_pr"})
    m = mv.merge(fn, on="node", how="inner")
    m["mv_rank"] = m["mv_pr"].rank(ascending=False, method="min").astype(int)
    m["fn_rank"] = m["fn_pr"].rank(ascending=False, method="min").astype(int)
    m["rank_gap"] = m["mv_rank"] - m["fn_rank"]
    m["abs_gap"] = m["rank_gap"].abs()
    rho, p = spearmanr(m["mv_pr"], m["fn_pr"])
    stats = {"spearman": float(rho), "spearman_p": float(p), "n": int(len(m))}
    return m.sort_values("mv_rank", ignore_index=True), stats


# --------------------------------------------------------------------------- #
# #22 Positional fee efficiency over time
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def positional_fee_time(grain: str) -> pd.DataFrame:
    """Median fee per ``(position, season)`` from matched deals (P1)."""
    p1 = get_p1(grain)
    m = p1[p1["matched"]].dropna(subset=["position", "season"])
    med = (m.groupby(["position", "season"], observed=True)["fee"]
           .median().reset_index().rename(columns={"fee": "median_fee"}))
    med["season"] = med["season"].astype(int)
    return med


# --------------------------------------------------------------------------- #
# #23 Window arbitrage
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def window_fee(grain: str) -> pd.DataFrame:
    """Matched per-deal fees with window/position/season (for #23 paired boxes)."""
    p1 = get_p1(grain)
    m = p1[p1["matched"]].dropna(subset=["window"])
    return m[["window", "position", "season", "fee"]].copy()


# --------------------------------------------------------------------------- #
# #24 Motif analysis (directed triad census + null-model z-scores)
# --------------------------------------------------------------------------- #
# Connected directed triads (drop the empty/dyadic 003/012/102) with readable names.
TRIAD_NAMES = {
    "021D": "021D out-star", "021U": "021U in-star", "021C": "021C chain",
    "111D": "111D", "111U": "111U", "030T": "030T feed-forward",
    "030C": "030C 3-cycle", "120D": "120D", "120U": "120U", "120C": "120C",
    "210": "210", "300": "300 clique",
}


@st.cache_data(show_spinner="Computing triad census + null model…")
def triad_profile(grain: str, top_k: int = 60, n_null: int = 20, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Directed triad census of the aggregated movement graph, with z-scores vs a
    degree-preserving null (random edge swaps).

    League (11 nodes): full enumeration. Club: restricted to the ``top_k`` most
    active nodes (by total transfer volume) — full club enumeration is too
    expensive — and the result is labelled an estimate."""
    agg = aggregated_edges("movement", grain)
    G = _digraph(agg, "n")
    sampled = False
    if grain == "club":
        deg = degree_table("club")
        deg = drop_outside_system(deg)
        deg["total"] = deg["in_degree"] + deg["out_degree"]
        keep = set(deg.nlargest(top_k, "total")["node"])
        G = G.subgraph(keep).copy()
        sampled = True
    # Triads are defined on distinct nodes — self-loops (intra-league / intra-club
    # transfers collapsed onto one node) are not part of any triad.
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    obs = nx.triadic_census(G)
    rng = np.random.default_rng(seed)
    n_edges = G.number_of_edges()
    null_rows = []
    for i in range(n_null):
        R = G.copy()
        try:
            nx.directed_edge_swap(R, nswap=max(1, 2 * n_edges),
                                  max_tries=20 * n_edges, seed=int(rng.integers(1 << 31)))
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass  # graph too small/constrained to swap; null ≈ observed for that draw
        null_rows.append(nx.triadic_census(R))
    nulldf = pd.DataFrame(null_rows)

    rows = []
    for code, label in TRIAD_NAMES.items():
        o = float(obs.get(code, 0))
        mu = float(nulldf[code].mean()) if code in nulldf else 0.0
        sd = float(nulldf[code].std(ddof=1)) if code in nulldf else 0.0
        z = (o - mu) / sd if sd > 0 else 0.0
        rows.append({"triad": code, "label": label, "observed": o,
                     "null_mean": mu, "null_std": sd, "z": z})
    prof = pd.DataFrame(rows)
    meta = {"n_nodes": G.number_of_nodes(), "n_edges": n_edges,
            "sampled": sampled, "top_k": top_k if sampled else None, "n_null": n_null}
    return prof, meta


# --------------------------------------------------------------------------- #
# #25 Capital flow asymmetry by position
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing capital-flow asymmetry…")
def capital_asymmetry(grain: str) -> pd.DataFrame:
    """Per ``(node, position)`` net money = revenue (sold) − spend (bought), from
    matched deals. Positive ⇒ net earner in that position; negative ⇒ net spender.

    Finance reversal handled via the movement orientation: in a movement edge the
    source SOLD (receives fee = revenue), the target BOUGHT (pays fee = spend)."""
    p1 = get_p1(grain)
    m = p1[p1["matched"]].dropna(subset=["position"])
    revenue = (m.groupby(["source", "position"], observed=True)["fee"].sum()
               .rename("revenue").rename_axis(["node", "position"]))
    spend = (m.groupby(["target", "position"], observed=True)["fee"].sum()
             .rename("spend").rename_axis(["node", "position"]))
    t = pd.concat([revenue, spend], axis=1).fillna(0.0).reset_index()
    t["node"] = t["node"].astype(str)
    t["net"] = t["revenue"] - t["spend"]
    t["gross"] = t["revenue"] + t["spend"]
    return with_names(t, grain)


# =========================================================================== #
# Section 4 — ALL FOUR NETWORKS COMBINED (#26-32)
#
# Alignment: movement runs sell->buy; finance runs buy->sell. We *reverse the
# finance layer* so both run sell->buy (along the player path) and are directly
# comparable. P1 tells us which movement/finance edges correspond; P2 does any
# club->league rollup. Heavy work (multilayer #26, community #27/#29) uses
# igraph + leidenalg/infomap and, where noted, top-N subgraphs (labelled est.).
# =========================================================================== #
def _aligned_agg(layer: str, grain: str) -> pd.DataFrame:
    """Aggregated simple edges oriented **sell -> buy** with column ``w``.

    Movement is already sell->buy (w = transfer count). Finance is buy->sell, so
    we swap endpoints to align it to the player path (w = summed fee = money the
    seller received from the buyer)."""
    a = aggregated_edges(layer, grain).copy()
    if layer == "finance":
        a = a.rename(columns={"source": "target", "target": "source"})
        a["w"] = a["fee"]
    else:
        a["w"] = a["n"]
    return a[["source", "target", "w"]]


def _igraph_from_edges(df: pd.DataFrame, weight: str = "w") -> ig.Graph:
    """Directed weighted igraph from a (source, target, weight) frame; node
    ``name`` attribute carries the original ids."""
    nodes = pd.unique(pd.concat([df["source"], df["target"]], ignore_index=True))
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(zip(df["source"].map(idx), df["target"].map(idx)))
    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    g.vs["name"] = list(nodes)
    g.es["weight"] = df[weight].to_numpy(dtype=float)
    return g


def _detect(g: ig.Graph, method: str, seed: int = 42):
    """Community membership list aligned to ``g.vs``; Leiden (leidenalg, modularity)
    or Infomap (igraph, flow-based, direction-aware)."""
    if method == "infomap":
        return g.community_infomap(edge_weights="weight").membership
    part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", seed=seed)
    return part.membership


# --------------------------------------------------------------------------- #
# #26 Multi-layer network analysis (cross-layer centrality + versatility)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing multilayer centrality…")
def multilayer_centrality(grain: str) -> tuple[pd.DataFrame, dict]:
    """Per-node talent vs money centrality on the **aligned** layers + versatility.

    talent  = movement PageRank (as-is sell->buy) — pull as a talent destination.
    money   = finance PageRank on the **reversed** finance layer — pull as a money
              destination (buying power), aligned to the same orientation.
    versatility = geometric mean of the two percentile ranks (high in *both*)."""
    from scipy.stats import spearmanr

    mv = pagerank_table("movement", grain, reverse=False)[["node", "label", "pagerank"]].rename(
        columns={"pagerank": "talent"})
    fn = pagerank_table("finance", grain, reverse=True)[["node", "pagerank"]].rename(
        columns={"pagerank": "money"})
    m = mv.merge(fn, on="node", how="outer")
    m["talent"] = m["talent"].fillna(0.0)
    m["money"] = m["money"].fillna(0.0)
    m["label"] = m["label"].fillna(m["node"])
    m["talent_pct"] = m["talent"].rank(pct=True)
    m["money_pct"] = m["money"].rank(pct=True)
    m["versatility"] = np.sqrt(m["talent_pct"] * m["money_pct"])
    both = m[(m["talent"] > 0) & (m["money"] > 0)]
    rho, p = spearmanr(both["talent"], both["money"]) if len(both) > 2 else (float("nan"), float("nan"))
    stats = {"spearman": float(rho), "spearman_p": float(p), "n_both": int(len(both))}
    return m.sort_values("versatility", ascending=False, ignore_index=True), stats


# --------------------------------------------------------------------------- #
# #27 Community detection & cross-layer comparison (club only)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Detecting communities (both layers)…")
def cross_layer_communities(method: str = "leiden", seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Leiden/Infomap communities on the aligned movement and finance club graphs,
    one row per node with both memberships (NaN if absent from a layer), plus
    NMI/ARI on the shared nodes. Club-level only (11 leagues won't cluster)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out = {}
    for layer in ("movement", "finance"):
        g = _igraph_from_edges(_aligned_agg(layer, "club"))
        mem = _detect(g, method, seed)
        out[layer] = pd.DataFrame({"node": g.vs["name"], f"{layer}_comm": mem})
    df = out["movement"].merge(out["finance"], on="node", how="outer")
    names = node_name_map("club")
    df["name"] = df["node"].map(names).fillna(df["node"])

    shared = df.dropna(subset=["movement_comm", "finance_comm"])
    if len(shared) > 1:
        nmi = float(normalized_mutual_info_score(shared["movement_comm"], shared["finance_comm"]))
        ari = float(adjusted_rand_score(shared["movement_comm"], shared["finance_comm"]))
    else:
        nmi = ari = float("nan")
    stats = {
        "method": method, "nmi": nmi, "ari": ari, "n_shared": int(len(shared)),
        "n_movement_comm": int(df["movement_comm"].nunique()),
        "n_finance_comm": int(df["finance_comm"].nunique()),
    }
    return df, stats


@st.cache_data(show_spinner="Building community subgraph…")
def community_subgraph(method: str, top_n: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-``top_n`` clubs (by movement volume, OS excluded) as an induced movement
    subgraph for a drawable community-coloured layout. Returns (nodes, edges) with
    a spring layout and the movement-community colour."""
    comm, _ = cross_layer_communities(method, seed)
    rank = drop_outside_system(club_volume_ranking()).head(top_n)
    keep = set(rank["node"])
    agg = _aligned_agg("movement", "club")
    sub = agg[(agg["source"].isin(keep)) & (agg["target"].isin(keep))]

    G = nx.DiGraph()
    G.add_nodes_from(keep)
    for r in sub.itertuples(index=False):
        G.add_edge(r.source, r.target, weight=r.w)
    pos = nx.spring_layout(G, seed=seed, k=0.5)
    cmap = dict(zip(comm["node"], comm["movement_comm"]))
    names = node_name_map("club")
    nodes = pd.DataFrame({
        "node": list(G.nodes()),
        "x": [pos[n][0] for n in G.nodes()],
        "y": [pos[n][1] for n in G.nodes()],
        "comm": [cmap.get(n, -1) for n in G.nodes()],
        "name": [names.get(n, n) for n in G.nodes()],
        "deg": [G.degree(n) for n in G.nodes()],
    })
    edges = pd.DataFrame(
        [(pos[u][0], pos[u][1], pos[v][0], pos[v][1]) for u, v in G.edges()],
        columns=["x0", "y0", "x1", "y1"],
    )
    return nodes, edges


# --------------------------------------------------------------------------- #
# #28 Dominance index (composite across all four nets)
# --------------------------------------------------------------------------- #
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


@st.cache_data(show_spinner="Building dominance index…")
def dominance_index(grain: str, w_talent: float = 1.0, w_spend: float = 1.0,
                    w_prestige: float = 1.0) -> pd.DataFrame:
    """Composite dominance = weighted sum of z-scored components (direction-corrected):

    net talent gain = movement in - out (net importer);
    financial muscle = spend - revenue (net buyer, reversal-aware);
    prestige = movement PageRank + reversed-finance (buying) PageRank.
    """
    deg = degree_table(grain)[["node", "label", "in_degree", "out_degree"]]
    fs = finance_strength_table(grain)[["node", "spend", "revenue"]]
    mv = pagerank_table("movement", grain, reverse=False)[["node", "pagerank"]].rename(
        columns={"pagerank": "mv_pr"})
    fn = pagerank_table("finance", grain, reverse=True)[["node", "pagerank"]].rename(
        columns={"pagerank": "fn_pr"})
    t = deg.merge(fs, on="node", how="outer").merge(mv, on="node", how="outer").merge(
        fn, on="node", how="outer").fillna(0.0)
    t["net_talent"] = t["in_degree"] - t["out_degree"]
    t["muscle"] = t["spend"] - t["revenue"]
    t["prestige"] = t["mv_pr"] + t["fn_pr"]
    t["z_talent"] = _z(t["net_talent"])
    t["z_muscle"] = _z(t["muscle"])
    t["z_prestige"] = _z(t["prestige"])
    wsum = (w_talent + w_spend + w_prestige) or 1.0
    t["dominance"] = (w_talent * t["z_talent"] + w_spend * t["z_muscle"]
                      + w_prestige * t["z_prestige"]) / wsum
    t["label"] = t["label"].fillna(t["node"])
    return t.sort_values("dominance", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# #29 Temporal community drift (club, per season)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Detecting communities per season…")
def community_drift(method: str = "leiden", top_k: int = 5, min_size: int = 15,
                    seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Per-season Leiden/Infomap on the season's club movement graph; track the
    ``top_k`` largest communities and give them persistent ids by greedy Jaccard
    matching to the previous season. Returns a tidy (season, stream_id, size)
    frame for a streamgraph + a stability stat (mean best-Jaccard)."""
    mv = get_edges("movement_club")
    seasons = sorted(mv["season"].dropna().unique().astype(int))
    prev: list[set] = []           # member-sets of last season's tracked streams
    prev_ids: list[int] = []
    next_id = 0
    jaccards = []
    rows = []
    for s in seasons:
        grp = mv[mv["season"] == s]
        agg = (grp.groupby(["source", "target"], observed=True).size()
               .reset_index(name="w"))
        if agg.empty:
            continue
        g = _igraph_from_edges(agg, weight="w")
        mem = _detect(g, method, seed)
        comm = pd.DataFrame({"node": g.vs["name"], "c": mem})
        sizes = comm.groupby("c")["node"].agg(list)
        big = [(c, set(v)) for c, v in sizes.items() if len(v) >= min_size]
        big.sort(key=lambda kv: -len(kv[1]))
        big = big[:top_k]

        assigned, used = [], set()
        for _, members in big:
            best_j, best_i = 0.0, None
            for i, pm in enumerate(prev):
                if i in used:
                    continue
                j = len(members & pm) / len(members | pm) if (members | pm) else 0.0
                if j > best_j:
                    best_j, best_i = j, i
            if best_i is not None and best_j >= 0.1:
                sid = prev_ids[best_i]; used.add(best_i); jaccards.append(best_j)
            else:
                sid = next_id; next_id += 1
            assigned.append((sid, members))
            rows.append({"season": s, "stream_id": sid, "size": len(members)})
        prev = [m for _, m in assigned]
        prev_ids = [sid for sid, _ in assigned]
    df = pd.DataFrame(rows)
    stats = {"method": method, "n_seasons": len(seasons),
             "n_streams": int(df["stream_id"].nunique()) if not df.empty else 0,
             "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan")}
    return df, stats


# --------------------------------------------------------------------------- #
# #30 Shock detection & propagation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def shock_series(grain: str, metric: str, exclude_os: bool) -> pd.DataFrame:
    """Aggregate per-season series with a robust (median/MAD) anomaly z-score.

    metric = ``volume`` (movement edges) or ``fee`` (finance €). ``exclude_os``
    drops any edge touching OS1, since external flows can mask domestic shocks."""
    if metric == "volume":
        df = get_edges(f"movement_{grain}")
        if exclude_os and grain == "club":
            df = df[(df["source"] != OUTSIDE_SYSTEM_ID) & (df["target"] != OUTSIDE_SYSTEM_ID)]
        s = df.groupby("season", observed=True).size().rename("value")
    else:
        df = get_edges(f"finance_{grain}")
        if exclude_os and grain == "club":
            df = df[(df["source"] != OUTSIDE_SYSTEM_ID) & (df["target"] != OUTSIDE_SYSTEM_ID)]
        s = df.groupby("season", observed=True)["weight"].sum().rename("value")
    t = s.reset_index()
    t["season"] = t["season"].astype(int)
    t = t.sort_values("season", ignore_index=True)
    t["yoy"] = t["value"].pct_change()
    # Detect shocks on the *detrended* series (year-on-year change): both metrics
    # trend upward, so a robust z on the raw level misses dips like 2020. A robust
    # (median/MAD) z on YoY surfaces change-points (e.g. the COVID fee drop).
    y = t["yoy"]
    med = y.median()
    mad = (y - med).abs().median() * 1.4826
    t["rz"] = (y - med) / mad if mad and mad > 0 else 0.0
    return t


@st.cache_data(show_spinner=False)
def shock_cascade(grain: str, season: int) -> pd.DataFrame:
    """For a flagged season, the per-node net-spend change vs the prior season —
    a lightweight propagation view (who drove the shock)."""
    sf = seasonal_finance(grain)
    cur = sf[sf["season"] == season][["node", "label", "net_spend"]]
    prev = sf[sf["season"] == season - 1][["node", "net_spend"]].rename(
        columns={"net_spend": "prev"})
    m = cur.merge(prev, on="node", how="left").fillna({"prev": 0.0})
    m["delta"] = m["net_spend"] - m["prev"]
    return m.reindex(m["delta"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- #
# #31 Position-stratified multilayer (league × position)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def position_league_matrix(value: str, side: str) -> pd.DataFrame:
    """League × position matrix. ``value`` = ``volume`` (movement counts) or
    ``fee`` (finance €). ``side`` = ``buyer`` (imports/spend) or ``seller``
    (exports/revenue). Aligned via the finance reversal: buyer = finance source."""
    if value == "volume":
        df = get_edges("movement_league")
        key = "target" if side == "buyer" else "source"   # movement: target=buyer
        t = df.groupby([key, "position"], observed=True).size().reset_index(name="val")
    else:
        df = get_edges("finance_league")
        key = "source" if side == "buyer" else "target"   # finance: source=buyer (reversal)
        t = df.groupby([key, "position"], observed=True)["weight"].sum().reset_index(name="val")
    t = t.rename(columns={t.columns[0]: "league"})
    names = get_league_names()
    t["league"] = t["league"].map(names).fillna(t["league"])
    return t


# --------------------------------------------------------------------------- #
# #32 Feeder club specialisation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Identifying feeder clubs…")
def feeder_clubs(grain: str) -> pd.DataFrame:
    """Per node: players sold (out-degree), net money in (revenue - spend),
    destination concentration (Herfindahl of out-targets), modal league, and
    the top-3 destinations. Feeder signature = sell players + net money IN."""
    deg = degree_table(grain)[["node", "label", "out_degree", "in_degree"]]
    fs = finance_strength_table(grain)[["node", "spend", "revenue"]]
    edges = get_edges(f"movement_{grain}")
    out_pairs = edges.groupby(["source", "target"], observed=True).size().reset_index(name="n")

    # Herfindahl of each seller's destination distribution
    tot = out_pairs.groupby("source", observed=True)["n"].transform("sum")
    out_pairs["share2"] = (out_pairs["n"] / tot) ** 2
    hhi = out_pairs.groupby("source", observed=True)["share2"].sum().rename("dest_hhi")

    # top-3 destinations by volume
    names = node_name_map(grain)
    top_dest = (out_pairs.sort_values("n", ascending=False)
                .groupby("source", observed=True)
                .head(3).groupby("source", observed=True)["target"]
                .agg(lambda s: ", ".join(names.get(x, str(x)) for x in s)).rename("top_destinations"))

    t = deg.merge(fs, on="node", how="left")
    t["node"] = t["node"].astype(str)
    t = t.merge(hhi, left_on="node", right_index=True, how="left")
    t = t.merge(top_dest, left_on="node", right_index=True, how="left")
    t["net_money_in"] = t["revenue"].fillna(0.0) - t["spend"].fillna(0.0)
    t["dest_hhi"] = t["dest_hhi"].fillna(0.0)

    if grain == "club":
        cl = club_league_by_season().groupby("club")["league"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        lnames = get_league_names()
        t["league"] = t["node"].map(cl).map(lnames).fillna("—")
    else:
        t["league"] = t["label"]
    return t


# =========================================================================== #
# Section 5 — TEMPORAL LEAGUE-LEVEL SANKEYS (#33-37)
#
# State-transition Sankey: one node per (league, stage); stages are the ordered
# (season, window) windows. A window's transfers are the transition stage t ->
# t+1. Aggregate by groupby(season, window, source_league, target_league) summing
# weight (player counts for movement, fees for finance) — transfer_id not needed.
# Movement follows the PLAYER (selling league -> buying league); finance is shown
# AS-IS, following the MONEY (paying/buyer league -> receiving/seller league), so
# the two diagrams are deliberately mirrored.
# =========================================================================== #
WINDOW_ORDER = {"summer": 0, "winter": 1}


@st.cache_data(show_spinner=False)
def sankey_stages() -> pd.DataFrame:
    """Ordered (season, window) stages with an integer ``stage`` index and label."""
    mv = get_edges("movement_league")[["season", "window"]].dropna().drop_duplicates()
    mv["season"] = mv["season"].astype(int)
    mv["wo"] = mv["window"].map(WINDOW_ORDER)
    mv = mv.sort_values(["season", "wo"]).reset_index(drop=True)
    mv["stage"] = range(len(mv))
    mv["label"] = mv["window"].str.capitalize() + " " + mv["season"].astype(str)
    return mv[["stage", "season", "window", "label"]]


@st.cache_data(show_spinner="Aggregating league flows…")
def sankey_flows(layer: str) -> pd.DataFrame:
    """Per (stage, source_league, target_league) flow weight for one layer.

    Movement weight = player count (transfers); finance weight = summed fee.
    ``stage`` is the ordered (season, window) index from :func:`sankey_stages`.
    Orientation is the raw edge orientation: movement = seller->buyer (player
    path); finance = buyer->seller (money path), shown as-is."""
    df = get_edges(f"{layer}_league").dropna(subset=["season", "window", "source", "target"])
    g = df.groupby(["season", "window", "source", "target"], observed=True)
    agg = (g.size().reset_index(name="weight") if layer == "movement"
           else g["weight"].sum().reset_index())
    agg["season"] = agg["season"].astype(int)
    stages = sankey_stages()[["season", "window", "stage", "label"]]
    agg = agg.merge(stages, on=["season", "window"], how="left")
    return agg[["stage", "label", "source", "target", "weight"]]


def _in_out_per_league_stage(df: pd.DataFrame) -> pd.DataFrame:
    """From a (stage, source, target, weight) flow frame, per (league, stage)
    total outflow (as source) and inflow (as target)."""
    out = (df.groupby(["stage", "source"], observed=True)["weight"].sum()
           .reset_index().rename(columns={"source": "league", "weight": "out"}))
    inn = (df.groupby(["stage", "target"], observed=True)["weight"].sum()
           .reset_index().rename(columns={"target": "league", "weight": "in"}))
    return out.merge(inn, on=["stage", "league"], how="outer").fillna(0.0)


@st.cache_data(show_spinner=False)
def league_net_flows() -> pd.DataFrame:
    """Per (league, stage): net players (movement in − out) and net money
    (finance revenue − spend) — the numeric companion to the #37 comparison.

    Movement: out = league as seller (source), in = as buyer (target).
    Finance (reversal): out = as buyer/payer (source) = spend, in = as seller (target) = revenue."""
    stages = sankey_stages()[["stage", "label", "season", "window"]]
    mvt = _in_out_per_league_stage(sankey_flows("movement")).rename(
        columns={"out": "players_out", "in": "players_in"})
    fnt = _in_out_per_league_stage(sankey_flows("finance")).rename(
        columns={"out": "spend", "in": "revenue"})
    t = mvt.merge(fnt, on=["stage", "league"], how="outer").fillna(0.0)
    t["net_players"] = t["players_in"] - t["players_out"]      # movement: in − out
    t["net_money"] = t["revenue"] - t["spend"]                 # finance: revenue − spend
    names = get_league_names()
    t = t.merge(stages, on="stage", how="left")
    t["league_name"] = t["league"].map(names).fillna(t["league"])
    return t


# Stable, brand league palette (consistent colours across both Sankeys).
from . import theme as _theme

_LEAGUE_PALETTE = list(_theme.LEAGUE_PALETTE)


def league_color_map() -> dict[str, str]:
    """Stable league_id -> colour, consistent across both Sankeys."""
    ids = sorted(get_league_names())
    cmap = {lid: _LEAGUE_PALETTE[i % len(_LEAGUE_PALETTE)] for i, lid in enumerate(ids)}
    cmap[OUTSIDE_SYSTEM_ID] = "#9e9e9e"   # OS parked in neutral grey
    cmap["Other"] = "#d0d0d0"
    return cmap
