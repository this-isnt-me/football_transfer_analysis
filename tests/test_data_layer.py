"""Tests for the shared data layer — schema, edge frames, and the P1/P2 mechanisms.

These exercise the *pure* functions (no Streamlit caching) and assert against the
verified schema recorded in CLAUDE.md (data regenerated 2026-06-18).
"""
import networkx as nx
import pandas as pd
import pytest

from src.data_layer import (
    EDGE_COLUMNS,
    NETWORKS,
    build_p1,
    build_p2,
    edge_frame,
    p1_invariant_report,
    p2_violations,
    read_graph,
)

# Verified ground truth (see CLAUDE.md "Verified schema").
EXPECTED_NODES = {
    "movement_club": 5598,
    "movement_league": 11,
    "finance_club": 4437,
    "finance_league": 11,
}
EXPECTED_EDGES = {
    "movement_club": 135004,
    "movement_league": 135004,
    "finance_club": 112795,
    "finance_league": 112795,
}
EXPECTED_POSITIONS = {
    "Goalkeeper", "Centre-Back", "Full-Back", "Central Midfielder",
    "Attacking Mid", "Winger / Wide Attacker", "Striker",
}
P1_CLUB_UNMATCHED = 22209  # movement deals with no finance counterpart


# --------------------------------------------------------------------------- #
# Fixtures — load each network's graph and edge frame once for the module.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def graphs():
    return {name: read_graph(name) for name in NETWORKS}


@pytest.fixture(scope="module")
def edges(graphs):
    return {name: edge_frame(name, G) for name, G in graphs.items()}


# --------------------------------------------------------------------------- #
# Schema / graph structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(NETWORKS))
def test_graph_is_directed_multigraph(graphs, name):
    G = graphs[name]
    assert isinstance(G, nx.MultiDiGraph)
    assert G.is_directed() and G.is_multigraph()


@pytest.mark.parametrize("name", list(NETWORKS))
def test_node_counts(graphs, name):
    assert graphs[name].number_of_nodes() == EXPECTED_NODES[name]


@pytest.mark.parametrize("name", list(NETWORKS))
def test_edge_counts(graphs, name):
    assert graphs[name].number_of_edges() == EXPECTED_EDGES[name]


# --------------------------------------------------------------------------- #
# Tidy edge frames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(NETWORKS))
def test_edge_frame_schema(edges, name):
    df = edges[name]
    assert list(df.columns) == EDGE_COLUMNS
    assert len(df) == EXPECTED_EDGES[name]


@pytest.mark.parametrize("name", list(NETWORKS))
def test_transfer_id_unique(edges, name):
    """Post-regeneration, transfer_id is one-per-edge (no dedup needed)."""
    tid = edges[name]["transfer_id"]
    assert tid.notna().all()
    assert tid.is_unique


@pytest.mark.parametrize("name", list(NETWORKS))
def test_position_normalised(edges, name):
    """`position` is populated from `position` or `player_position` per file."""
    pos = set(edges[name]["position"].dropna().unique())
    # Movement files carry a single stray 'Unknown'; everything else is canonical.
    assert pos.issubset(EXPECTED_POSITIONS | {"Unknown"})
    assert EXPECTED_POSITIONS.issubset(pos)


@pytest.mark.parametrize("name", list(NETWORKS))
def test_window_lowercase(edges, name):
    assert set(edges[name]["window"].dropna().unique()) == {"summer", "winter"}


def test_movement_weight_is_one(edges):
    assert (edges["movement_club"]["weight"] == 1).all()
    assert (edges["movement_league"]["weight"] == 1).all()


def test_finance_weight_has_no_nulls(edges):
    """Missing fee never appears as an in-network null — it is a missing edge."""
    for name in ("finance_club", "finance_league"):
        assert edges[name]["weight"].notna().all()
        assert (edges[name]["weight"] > 0).all()


# --------------------------------------------------------------------------- #
# P1 — transfer_id join (LEFT join movement -> finance)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def p1_club(edges):
    return build_p1(edges["movement_club"], edges["finance_club"])


def test_p1_left_join_row_count(p1_club, edges):
    """LEFT join keeps exactly one row per movement edge."""
    assert len(p1_club) == len(edges["movement_club"]) == EXPECTED_EDGES["movement_club"]


def test_p1_unmatched_count_and_null_fee(p1_club):
    rep = p1_invariant_report(p1_club)
    assert rep["n_unmatched"] == P1_CLUB_UNMATCHED
    assert rep["n_matched"] == EXPECTED_EDGES["finance_club"]
    # Missing fee stays NULL, never 0.
    assert rep["unmatched_fee_all_null"] is True
    assert rep["n_fee_equals_zero"] == 0


def test_p1_missing_fee_is_null_not_zero(p1_club):
    unmatched = p1_club[~p1_club["matched"]]
    assert unmatched["fee"].isna().all()
    assert not (p1_club["fee"].fillna(-1) == 0).any()  # no zero fees anywhere


def test_p1_reversal_invariant(p1_club):
    """movement.source == finance.target and movement.target == finance.source."""
    rep = p1_invariant_report(p1_club)
    assert rep["endpoint_violations"] == 0
    assert rep["position_divergences"] == 0
    assert rep["matched_fee_any_null"] is False


def test_p1_league_level(edges):
    p1_lg = build_p1(edges["movement_league"], edges["finance_league"])
    rep = p1_invariant_report(p1_lg)
    assert rep["n_movement"] == EXPECTED_EDGES["movement_league"]
    assert rep["endpoint_violations"] == 0
    assert rep["n_fee_equals_zero"] == 0


# --------------------------------------------------------------------------- #
# P2 — time-dependent club -> league mapping
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def p2_map(edges):
    return build_p2(edges["movement_club"], edges["movement_league"])


def test_p2_one_league_per_club_season_window(p2_map):
    """The core P2 invariant: each (club, season, window) -> exactly one league."""
    violations = p2_violations(p2_map)
    assert violations.empty, (
        f"{len(violations)} (club, season, window) triples map to >1 league:\n"
        f"{violations.head(20)}"
    )


def test_p2_outside_system_maps_to_os_league(p2_map):
    """Club node OS1 (Outside System) maps to the OS1 league."""
    os_rows = p2_map[p2_map["club"] == "OS1"]
    assert not os_rows.empty
    assert set(os_rows["league"].unique()) == {"OS1"}


def test_p2_covers_all_clubs(p2_map, graphs):
    """Every club node that appears on an edge gets a league assignment."""
    mc = graphs["movement_club"]
    clubs_on_edges = {u for u, _ in mc.edges()} | {v for _, v in mc.edges()}
    mapped = set(p2_map["club"].dropna().unique())
    assert clubs_on_edges.issubset(mapped)


def test_p2_leagues_are_valid_ids(p2_map, graphs):
    """Every assigned league is a real node id in the league network."""
    league_ids = set(graphs["movement_league"].nodes())
    assert set(p2_map["league"].dropna().unique()).issubset(league_ids)