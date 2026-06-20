# ⚽ Football Transfer Network Analysis

A Streamlit analytics app over four directed multigraphs of football transfers —
**player movement** and **money flow**, at **club** and **league** scale. It implements
37 numbered analyses across 5 sections (2004–2024, 21 seasons).

## Quick start

```bash
# 1. create / activate the virtualenv (Python 3.12)
python -m venv .venv
.venv\Scripts\activate            # PowerShell
# source .venv/Scripts/activate   # Git Bash

# 2. install deps
pip install -r requirements.txt

# 3. run the app
streamlit run Home.py
```

Then open the printed URL (default http://localhost:8501). Use the sidebar to switch
between the **Home** health-check page and the five section pages.

Run the tests with `pytest` (39 tests covering the data layer + P1/P2 invariants).

## The data

Four GraphML files in `data/` (all directed multigraphs, one edge per transfer):

| File | Grain | Edge direction | `weight` |
|---|---|---|---|
| `movement_club_net` | club (5,598) | seller → buyer (player path) | 1 |
| `movement_league_net` | league (11) | seller → buyer | 1 |
| `finance_club_net` | club (4,437) | **buyer → seller** (money path) | fee (€) |
| `finance_league_net` | league (11) | **buyer → seller** | fee (€) |

Two mechanisms bridge the networks (implemented + tested in the data layer):

- **P1** — the `transfer_id` join (movement ⇄ finance). LEFT join movement → finance; a
  missing fee stays **NULL, never 0**.
- **P2** — the time-dependent `(club, season, window) → league` mapping, derived from the
  data (clubs change leagues across seasons).

The **finance reversal**: finance edges run buyer → seller, so finance **out-strength =
spend**, **in-strength = revenue**, net profit = revenue − spend. When layers are
combined, finance is reversed to align with the player path.

## Project layout

```
Home.py                     landing page: network overview + P1/P2 health check
pages/1..5_*.py             thin multipage entries (one ui.section_page call each)
src/
  data_layer.py             graphs, tidy edge frames, P1, P2  (single source)
  metrics.py                all cached metric computations
  ui.py                     shared controls + Plotly helpers + page boilerplate
  section1..5.py            per-section render functions (the 37 analyses)
scripts/
  explore_data.py           Phase-0 schema verifier (re-run if data/ is regenerated)
  probe_anomalies.py        Phase-0 anomaly probe
tests/test_data_layer.py    schema + P1/P2 invariant tests
```

Performance: graphs are cached with `st.cache_resource`, all derived frames with
`st.cache_data`, so reruns never recompute. First view of a heavy analysis computes once
then is instant.

## Sections

1. **Single-network analyses (#1–15)** — degree, PageRank, betweenness, reciprocity,
   strength, Gini/Lorenz, seasonal/window/position breakdowns, the #15 ego-net.
2. **Club vs League (#16–18)** — P2 reconciliation, ranking correlation, temporal
   divergence between scales.
3. **Movement vs Finance (#19–25)** — corridor flow, fee-per-player, prestige divergence,
   positional fee efficiency, window arbitrage, motifs, capital asymmetry (all via P1).
4. **All four combined (#26–32)** — multilayer centrality, community detection, dominance
   index, community drift, shock detection, position-stratified multilayer, feeders.
5. **Temporal league Sankeys (#33–37)** — state-transition Sankeys over the ordered
   (season, window) stages; movement follows the player, finance follows the money.

## Known limitations & approximations

- **#3 Betweenness** is **approximate** (Brandes with *k* pivot sources on hop-distance),
  club-level only. First load at default *k*=500 takes ~20 s on the 5,598-node graph, then
  caches; read it as ranks, not exact values.
- **#24 Motifs** at **club level are sampled** to the top-*k* most active clubs (full club
  triad census is infeasible) — labelled an estimate. At league level the 11-node movement
  graph is complete, so all triads are saturated and z ≡ 0 (flagged in-app).
- **#27/#29 Community detection**: the #27 force layout is a **top-N drawable subgraph**
  (the full 5,598-node graph is never drawn). Leiden is seed-deterministic; **Infomap is
  not** seedable, so its communities vary slightly run-to-run.
- Fees are **nominal** (no inflation adjustment) — compare trends, not absolute eras.
- Fees are heavily right-skewed → the app reports **medians/percentiles**, not means.

## Data-quality flags (verified against the 2026-06-18 data)

- The `data/` files were **regenerated and pre-deduplicated**: edge counts are lower than
  the original guide, and `transfer_id` is now **globally unique** (no dedup step needed).
- **~22,209 movement transfers have no finance counterpart** → NULL fee under P1 (free
  transfers / loans / unrecorded), never treated as €0.
- The **position attribute key is inconsistent** across files (`player_position` in three,
  `position` in one) — normalised on load. Two movement edges carry a stray `Unknown`.
- **`window` values are lowercase** `summer` / `winter`.
- **Club names are not unique** (7–8 names map to two node ids each) — all club-level
  output is keyed/labelled by **node id**, not name.
- **Catch-all pseudo-nodes**: `OS1` (external clubs), `515` "Without Club" (free agency),
  `75` "UnknownUnknown". `515` tops the raw movement degree rankings, so all club-level
  rankings exclude these three by default (toggle off to include them).
- **P1 invariant** holds exactly: 0 endpoint reversal violations and 0 position
  divergences across all matched deals.
- **P2 invariant** holds: every `(club, season, window)` resolves to exactly one league
  (0 violations); the club→league rollup reconciles to the league network exactly (#16).
- Non-ASCII club names are clean UTF-8 (any garbling is a console display artifact only).
