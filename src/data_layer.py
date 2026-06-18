"""Shared data layer for the football transfer-network app.

Single source of truth for the four GraphML networks. Exposes them as:
  * networkx ``MultiDiGraph`` objects (``get_graph`` / ``get_all_graphs``,
    cached with ``st.cache_resource``), and
  * tidy edge ``DataFrame``s (``get_edges``, cached with ``st.cache_data``).

It also implements the two core mechanisms from CLAUDE.md:
  * **P1** — the ``transfer_id`` join between movement and finance (LEFT join
    movement -> finance; a missing fee stays NULL, never 0), and
  * **P2** — the time-dependent ``(club, season, window) -> league`` mapping
    derived from the data.

The pure functions (``read_graph``, ``edge_frame``, ``build_p1``, ``build_p2``,
``p1_invariant_report``, ``p2_violations``) take/return plain objects and have no
Streamlit dependency, so they are directly unit-testable. The ``get_*`` wrappers
add Streamlit caching on top.

Schema notes (see CLAUDE.md "Verified schema", data regenerated 2026-06-18):
  * ``transfer_id`` is globally unique per file (one edge per deal); no dedup needed.
  * The position attribute key is ``player_position`` in three files but ``position``
    in ``finance_league`` -> normalised to a single ``position`` column here.
  * Movement ``weight`` is 1; finance ``weight`` is the fee (float, no nulls).
  * Finance edges run buyer -> seller (the reverse of movement's seller -> buyer).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NETWORKS: dict[str, str] = {
    "movement_club":   "movement_club_net.graphml.xml",
    "movement_league": "movement_league_net.graphml.xml",
    "finance_club":    "finance_club_net.graphml.xml",
    "finance_league":  "finance_league_net.graphml.xml",
}

# Columns of every tidy edge frame, in order.
EDGE_COLUMNS = [
    "source", "target", "transfer_id", "season", "window", "position",
    "weight", "player_id", "player_home_country", "player_value",
]

# The position attribute is stored under one of these keys (file-dependent).
_POSITION_KEYS = ("position", "player_position")


# ---------------------------------------------------------------------------
# Pure (Streamlit-free) loaders
# ---------------------------------------------------------------------------
def read_graph(name: str) -> nx.MultiDiGraph:
    """Load one network as a networkx ``MultiDiGraph`` (parallel edges kept)."""
    if name not in NETWORKS:
        raise KeyError(f"Unknown network {name!r}; expected one of {list(NETWORKS)}")
    path = DATA_DIR / NETWORKS[name]
    return nx.read_graphml(path, force_multigraph=True)


def node_names(G: nx.MultiDiGraph) -> dict[str, str]:
    """``node_id -> name`` for a loaded graph."""
    return {nid: d.get("name", nid) for nid, d in G.nodes(data=True)}


def edge_frame(name: str, G: nx.MultiDiGraph | None = None) -> pd.DataFrame:
    """Tidy one-row-per-edge DataFrame for a network.

    The ``position`` column is normalised across files (reads ``position`` or
    ``player_position``, whichever is present). Columns absent in a given file
    (e.g. ``player_value`` outside movement_club / finance_league) are filled
    with NA so every frame shares the same schema.
    """
    if G is None:
        G = read_graph(name)

    records = []
    for u, v, d in G.edges(data=True):
        position = next((d[k] for k in _POSITION_KEYS if k in d), None)
        records.append((
            u,
            v,
            d.get("transfer_id"),
            d.get("season"),
            d.get("window"),
            position,
            d.get("weight"),
            d.get("player_id"),
            d.get("player_home_country"),
            d.get("player_value"),
        ))

    df = pd.DataFrame.from_records(records, columns=EDGE_COLUMNS)

    # Stable dtypes. season is int; weight numeric (int for movement, float for
    # finance). player_value is a valuation distinct from the fee.
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["player_value"] = pd.to_numeric(df["player_value"], errors="coerce").astype("Int64")
    for col in ("source", "target", "transfer_id", "window", "position",
                "player_id", "player_home_country"):
        df[col] = df[col].astype("string")
    return df


# ---------------------------------------------------------------------------
# P1 — the transfer_id join (movement <-> finance)
# ---------------------------------------------------------------------------
def build_p1(movement_df: pd.DataFrame, finance_df: pd.DataFrame) -> pd.DataFrame:
    """LEFT join movement -> finance on ``transfer_id``.

    Returns one row per movement edge. Finance columns are suffixed ``fin_``;
    the fee is ``fee`` (finance ``weight``). Rows with no finance counterpart
    get **NA fee — never 0**. ``matched`` is a boolean structural flag derived
    from the merge indicator (independent of the fee value).

    Finance endpoints are the reverse of movement's, so for a matched row the
    invariant is ``source == fin_target`` and ``target == fin_source`` (checked
    by :func:`p1_invariant_report`).
    """
    fin = finance_df[["transfer_id", "source", "target", "position", "weight"]].rename(
        columns={
            "source": "fin_source",
            "target": "fin_target",
            "position": "fin_position",
            "weight": "fee",
        }
    )
    merged = movement_df.merge(fin, on="transfer_id", how="left", indicator=True)
    merged["matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")
    return merged


def p1_invariant_report(p1_df: pd.DataFrame) -> dict:
    """Validate the P1 reversal invariant on matched rows.

    Checks (on rows where ``matched``):
      * ``source == fin_target`` and ``target == fin_source`` (finance reversal),
      * ``position == fin_position`` (the deal's position agrees on both sides).
    Also reports the unmatched count and confirms no fee was coerced to 0.
    """
    matched = p1_df[p1_df["matched"]]
    endpoint_ok = (matched["source"] == matched["fin_target"]) & (
        matched["target"] == matched["fin_source"]
    )
    position_ok = matched["position"] == matched["fin_position"]

    unmatched = p1_df[~p1_df["matched"]]
    return {
        "n_movement": int(len(p1_df)),
        "n_matched": int(len(matched)),
        "n_unmatched": int(len(unmatched)),
        "endpoint_violations": int((~endpoint_ok).sum()),
        "position_divergences": int((~position_ok).sum()),
        "unmatched_fee_all_null": bool(unmatched["fee"].isna().all()),
        "matched_fee_any_null": bool(matched["fee"].isna().any()),
        "n_fee_equals_zero": int((p1_df["fee"] == 0).sum()),
    }


# ---------------------------------------------------------------------------
# P2 — time-dependent club -> league mapping (derived from the data)
# ---------------------------------------------------------------------------
def build_p2(movement_club_df: pd.DataFrame,
             movement_league_df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``(club, season, window) -> league`` by joining the club and
    league movement networks on ``transfer_id``.

    Each shared transfer reveals the league of *both* its endpoints at that
    ``(season, window)``: the club edge gives ``club_source -> club_target`` and
    the league edge gives ``league_source -> league_target``. We harvest both
    the source-side and target-side ``(club, season, window, league)`` tuples
    and de-duplicate. The result keys league by its node id (e.g. ``GB1``);
    use a league name map for display.
    """
    mc = movement_club_df[["transfer_id", "source", "target", "season", "window"]]
    ml = movement_league_df[["transfer_id", "source", "target"]].rename(
        columns={"source": "league_source", "target": "league_target"}
    )
    joined = mc.merge(ml, on="transfer_id", how="inner")

    src_side = joined[["source", "season", "window", "league_source"]].rename(
        columns={"source": "club", "league_source": "league"}
    )
    tgt_side = joined[["target", "season", "window", "league_target"]].rename(
        columns={"target": "club", "league_target": "league"}
    )
    mapping = (
        pd.concat([src_side, tgt_side], ignore_index=True)
        .dropna(subset=["club"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return mapping


def p2_violations(p2_map: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``(club, season, window)`` that resolve to more than one league.

    The P2 invariant is that each such triple maps to exactly one league; a
    non-empty result is a data-quality flag.
    """
    counts = (
        p2_map.groupby(["club", "season", "window"])["league"]
        .nunique()
        .reset_index(name="n_leagues")
    )
    return counts[counts["n_leagues"] > 1].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streamlit-cached accessors (thin wrappers over the pure functions above)
# ---------------------------------------------------------------------------
try:
    import streamlit as st

    _cache_resource = st.cache_resource
    _cache_data = st.cache_data
except Exception:  # pragma: no cover - lets the module import without Streamlit
    def _identity(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

    _cache_resource = _identity
    _cache_data = _identity


@_cache_resource(show_spinner="Loading transfer networks…")
def get_all_graphs() -> dict[str, nx.MultiDiGraph]:
    """All four networks as graphs, loaded once per session."""
    return {name: read_graph(name) for name in NETWORKS}


def get_graph(name: str) -> nx.MultiDiGraph:
    """One network graph (shares the cached ``get_all_graphs`` objects)."""
    return get_all_graphs()[name]


@_cache_data(show_spinner="Building edge tables…")
def get_edges(name: str) -> pd.DataFrame:
    """Tidy edge DataFrame for one network (cached)."""
    return edge_frame(name, get_graph(name))


@_cache_data(show_spinner=False)
def get_league_names() -> dict[str, str]:
    """``league_id -> league name`` (e.g. ``GB1 -> Premier League``)."""
    return node_names(get_graph("finance_league"))


@_cache_data(show_spinner="Joining movement <-> finance (P1)…")
def get_p1(level: str = "club") -> pd.DataFrame:
    """P1 join at the given level (``"club"`` or ``"league"``)."""
    if level not in ("club", "league"):
        raise ValueError(f"level must be 'club' or 'league', got {level!r}")
    return build_p1(get_edges(f"movement_{level}"), get_edges(f"finance_{level}"))


@_cache_data(show_spinner="Deriving club -> league mapping (P2)…")
def get_p2() -> pd.DataFrame:
    """The derived ``(club, season, window) -> league`` mapping (cached)."""
    return build_p2(get_edges("movement_club"), get_edges("movement_league"))
