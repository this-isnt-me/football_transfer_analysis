# CLAUDE.md

Guidance for Claude Code when working in this repository. The standing rules below apply
on **every** turn. The phased build tasks live in `build-playbook.md` — work through them
in order, one at a time.

## What this repository is

A data-analysis project on football transfer networks. It currently contains the input
datasets (`data/`) and a detailed analytical specification (`transfer_network_analysis_guide.md`
plus its companion display spec). The goal is a **Streamlit analytics app** implementing
the 37 numbered analyses across 5 sections. Treat the guide as the authoritative
requirements document and the display spec as the authority on presentation; reference
analyses by their numbers.

## Role

You are a senior data scientist and network analyst who ships polished, performant
Streamlit applications. You verify against the data before assuming, build incrementally,
and never report a step done without running it.

## Environment

- Python 3.12, virtualenv at `.venv` (activate: `.venv\Scripts\activate` in PowerShell,
  or `source .venv/Scripts/activate` in the Bash tool).
- The venv is currently **empty** (pip only); no `requirements.txt` / `pyproject.toml`
  exists yet. Install what an analysis needs and add it to `requirements.txt` as you go.
- Expected stack: `networkx`, `pandas`, `numpy` for the core; `streamlit` for the app;
  `plotly` as the primary (interactive) charting library — Sankey diagrams and beyond —
  with `matplotlib` acceptable for simple static plots; `python-igraph` or `graph-tool`
  (C-backed) for heavy metrics (betweenness, motifs/triad census, multilayer);
  `leidenalg` / `infomap` for community detection.

## The data model (read before writing any analysis)

Four GraphML files in `data/`, all **directed multigraphs** (parallel edges allowed —
one edge per transfer):

| File | Grain | Edge direction | `weight` |
|---|---|---|---|
| `data/movement_club_net.graphml.xml` | club | seller → buyer (player path) | 1 |
| `data/movement_league_net.graphml.xml` | league (11 nodes) | seller → buyer (player path) | 1 |
| `data/finance_club_net.graphml.xml` | club | **buyer → seller** (money path) | fee / valuation |
| `data/finance_league_net.graphml.xml` | league (11 nodes) | **buyer → seller** (money path) | fee / valuation |

Load with `networkx.read_graphml(path)`. Node attribute: `name`. Edge attributes:
`transfer_id` (str), `window` (str: Summer/Winter), `season` (int), `position` (str),
`weight` (float).

### Three conventions that pervade every analysis

1. **The finance reversal.** Finance edges run **buyer → seller**, the *reverse* of
   movement (seller → buyer). Consequences: finance **out-strength = money paid (spend)**,
   **in-strength = money received (sales revenue)**; net profit = in − out. Whenever
   movement and finance are combined, one layer must be flipped to align semantics —
   always state which.

2. **P1 — the `transfer_id` join (movement ⇄ finance).** A single deal is a movement edge
   and a finance edge that are exact reverses sharing one `transfer_id`. Join on
   `transfer_id`. Invariant after join: `movement.source == finance.target` and
   `movement.target == finance.source`, and `position` agrees on both sides. Movement has
   ~166k edges vs finance's ~143k → ~23k movement edges have no financial counterpart
   (free transfers/loans/unrecorded). **Left-join movement → finance and treat a missing
   fee as NULL, never 0** (a known €0 free transfer ≠ an unrecorded fee).

3. **P2 — time-dependent club→league mapping.** Clubs change leagues across seasons
   (promotion/relegation), so **no static club→league lookup is valid**. Build it from the
   data: the same `transfer_id` appears in both the club and league network, revealing
   `(club, season, window) → league` for each endpoint. Each `(club, season, window)` must
   resolve to exactly one league; flag violations. Use this for any club→league rollup,
   and to validate that aggregating club edges reproduces the league network.

P1 and P2 are implemented and **tested** in Phase 1 and respected everywhere after.

### "Outside System" — a club-level trap

At **club level**, "Outside System" is a *name* shared by many distinct node IDs (each
external club is a separate, anonymised node). **Always key/label club-level output by
node id, not by name** — labelling by name fuses unrelated clubs into indistinguishable
rows. For rankings, keep ids separate (they self-filter out of top-X) or collapse to one
aggregate node shown as a *separate reference*, never ranked beside real clubs. At
**league level** this problem disappears (all map to the single "Outside System" league).

## Display defaults (from the companion spec)

- **Club networks (~5,600 nodes): never draw the full node-link graph.** Use ranked
  tables / top-X bars (≈15–20 for bars, ≈25–30 for tables), distributions, scatters, time
  series. Full network drawings only for ego-nets (#15) or community-coloured *filtered /
  top-X* subgraphs (#27).
- **League networks (11 nodes):** node-link graphs, chord diagrams, 11×11 heatmaps, and
  per-season small multiples all work directly; always show all 11.
- Fees are heavily right-skewed — report **median/percentiles**, not mean.
- Make top-X a UI control where useful. Follow the display spec's chart choice per
  analysis unless the data makes it impossible — if so, flag it.

## App conventions

- Multipage Streamlit app with thin page files. Separate modules: a **data layer**
  (loading + P1/P2), a **metrics/analysis** module, and **page** files.
- `st.cache_resource` for graph objects, `st.cache_data` for derived frames. Never
  recompute expensive work on rerun.

## Computational notes

Most analyses are O(E) and trivial. The expensive ones, per the guide: exact betweenness
(#3) and motif/triad census at club scale (#24), and club-level multilayer analysis (#26)
— use a C-backed library (igraph/graph-tool) and/or sampling / top-N subgraphs for these,
and label results as estimates. **Aggregate movement parallel edges to counts before
running PageRank**, especially at league level. Community detection is club-level only
(11 nodes is too small to cluster).

## Working style

- Build incrementally and **run the app to confirm each section renders before claiming it
  done.**
- Commit at each phase/section checkpoint with a clear message.
- When data contradicts the documented schema or spec, **stop and surface it** rather than
  coding around it.

## Verified schema

The data model above is the *documented* schema. **Phase 0 confirms it against the actual
GraphML files** (exact node/edge counts, attribute keys, distinct `season` / `window` /
`position` values, `transfer_id` uniqueness, "Outside System" encoding at club level, and
the count of finance edges with null weight). Record any deviations here as you find them.

_(Phase 0 findings: to be completed)_
