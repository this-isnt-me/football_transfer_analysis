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

The data model above is the *documented* schema. Phase 0 verified it against the actual
GraphML files (networkx 3.6.1, Python 3.12) via `explore_data.py` + `probe_anomalies.py`
(both throwaway). **Several documented claims are wrong — see deviations. Do NOT code
around them; design Phase 1 around the verified reality.**

> **⚠ The `data/` files were regenerated on 2026-06-18 and DIFFER from the guide's table
> and from a prior Phase-0 run.** They now appear **pre-deduplicated**: edge counts are
> lower than the guide states and `transfer_id` is now globally unique (one edge per deal).
> Edge *attributes also changed* — extra columns were added and the position key is
> inconsistent across files. The guide/display-spec numbers (166,258 / 142,853 edges, the
> "transfer_id not unique" duplicate-pair problem, weight==1 placeholder) describe the
> **old** files and no longer apply. The figures below are the current ground truth — trust
> these, not the guide's table, and re-run the two probe scripts if the files change again.

### Confirmed (matches the documented model)

- **Node counts: exact match on all four files** — movement_club 5,598; movement_league 11;
  finance_club 4,437; finance_league 11. (Edge counts changed — see deviation 1.)
- All four are **directed multigraphs** (`MultiDiGraph`). Node attr keys: `name` only.
- `season` is `int`, spans **2004–2024 (21 seasons)** in all four files.
- `position` is a clean **7-value canonical taxonomy** (no synonyms, no normalisation
  needed, contra guide #4): `Goalkeeper`, `Centre-Back`, `Full-Back`, `Central Midfielder`,
  `Attacking Mid`, `Winger / Wide Attacker`, `Striker`. There is no GK/DEF/MID/FWD grouping
  in the data — add a 4-class mapping only if wanted. (But see deviation 2 on the *key name*
  and the stray `Unknown`.)
- **`window` values are lowercase `summer` / `winter`** (not `Summer`/`Winter` as in the
  data-model table and guide). **Filter/group on lowercase.**
- **Finance reversal verified, exhaustively.** The P1 invariant
  `movement.(src,dst) == finance.(dst,src)` held with **0 violations across all 112,795
  shared `transfer_id`s** (not a sample), and `position` agreed on every one (0
  divergences).
- Movement `weight` is uniformly `1` (int). Finance `weight` is float, range now
  **450.0 – 222,000,000**, with **no nulls and no `weight==1` placeholder** (the old
  single weight-1 edge is gone post-regeneration).
- League node sets (both league files, 11 each), shown as `id → name`: `BE1`→Jupiler Pro
  League, `ES1`→LaLiga, `FR1`→Ligue 1, `GB1`→Premier League, `GB2`→Championship,
  `IT1`→Serie A, `L1`→Bundesliga, `NL1`→Eredivisie, `OS1`→Outside System, `PL1`→PKO BP
  Ekstraklasa, `PO1`→Liga Portugal.

### Deviations from the documented model — surfaced, NOT coded around

1. **Edge counts are LOWER than the guide and `transfer_id` IS now unique.** Current
   counts: **movement (club & league) 135,004 edges; finance (club & league) 112,795
   edges** (guide says 166,258 / 142,853). On all four files `transfer_id` is present on
   every edge, **globally unique, max multiplicity 1** — the old "duplicate parallel edge /
   31k ids on 2 edges" problem is **gone**. The data was deduplicated upstream. **P1 needs
   no dedup step now** (a prior plan to "decide a dedup policy" is obsolete) — but P1 still
   joins on `transfer_id` and must still treat a missing finance row as NULL fee.

2. **The position attribute key is INCONSISTENT across files, and is mostly NOT `position`.**
   - `movement_club`, `movement_league`, `finance_club` → key is **`player_position`**.
   - `finance_league` → key is **`position`**.
   Documented model says every file uses `position`; only one file does. **Read position
   with a fallback** (`d.get("position") or d.get("player_position")`) or normalise on load.
   Also: the two **movement** files contain **one edge with position `'Unknown'`** (same
   deal in both); finance has none. So movement effectively has 8 distinct values (7 + 1
   `Unknown`); decide whether to keep, drop, or relabel that single edge.

3. **Edge attribute sets are richer than documented and differ per file.** Documented set
   was exactly `transfer_id, window, season, position, weight`. Actual keys:
   - `movement_club`: `transfer_id, window, season, player_position, weight, player_id,`
     `player_home_country, player_value`
   - `movement_league`: same minus `player_value`
   - `finance_club`: `transfer_id, window, season, player_position, weight, player_id`
   - `finance_league`: `transfer_id, window, season, position, weight, player_id,`
     `player_value`
   New attributes available: **`player_id`** (every edge, all files — enables true
   per-player tracing, not just per-deal), **`player_home_country`** (movement files only),
   and **`player_value`** (movement_club + finance_league only). `player_value` is an `int`
   valuation (range 0–180,000,000) **distinct from the fee in `weight`**; it has many zeros
   (23,648 in movement_club, 1,439 in finance_league) — treat 0 as "no valuation", and do
   **not** conflate it with `weight`/fee.

4. **"Outside System" at club level is a SINGLE node, id `OS1`** (name `Outside System`) —
   not many anonymised ids as display-spec D0 claims, and not the literal id
   `"Outside System"`. External clubs **keep their real names** (e.g. `Nashville SC`,
   `Al-Arabi SC`). → **D0's premise and its "keep ids separate / collapse" guidance do not
   apply.** Treat `OS1` as one ordinary catch-all node. *However,* "label by node id, not
   name" still holds for a different reason: **7–8 real club names map to 2 node ids each**
   (e.g. `Arsenal FC` → ids `11` & `4673`), so name is not a unique key — key/label
   club-level output by node id.

### Clarifications (refine, don't contradict, the documented model)

- **Some names contain non-ASCII chars stored as clean UTF-8** (e.g. `CD Universidad
  Católica`, `América FC`). The garbled `Cat�lica` seen in a console is a Windows codepage
  *display* artifact only — raw bytes are valid UTF-8 (`\xc3\xb3` = ó). **No file
  corruption; read/display with `encoding="utf-8"`.**
- **No finance edges have null/missing `weight`** (0 of 112,795). "Missing fee" never
  appears as an in-network null — it manifests only as **movement transfers with no finance
  edge**. Finance `transfer_id`s are a **clean subset** of movement's (0 finance ids absent
  from movement); **22,209 movement ids have no finance counterpart**. So P1 is a **left
  join movement → finance with fee = NULL** for those 22,209 — there are no in-network
  nulls to filter, contrary to what guide #20 implies.
- **`season` spans 2004–2024 (21 seasons), all four files.** The Sankey section (#33)
  assumes "Summer 2010 → … ~14 seasons / ~28 stages"; the real range is **~42 stages**
  (21 × 2 windows). Plan Sankey scalability/filtering (#35) for 42 stages, and make the
  start season a UI control rather than hard-coding 2010.
