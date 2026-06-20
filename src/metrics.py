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
