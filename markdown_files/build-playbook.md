# Build Playbook — Phased Prompts

Paste each phase into Claude Code **one at a time**, in order. Don't start the next
phase until the current one's checkpoint passes. `CLAUDE.md` holds the standing rules;
these prompts assume it's already in the repo.

---

## Phase 0 — Explore the data (no app code yet)

```
Read transfer_network_analysis_guide.md and the display spec in full, then explore the
data before writing any app code.

Write a throwaway exploration script that loads each of the four GraphML files in data/
with networkx and reports, per network:
- node count and edge count (compare against the spec's table; flag any mismatch),
- exact node attribute keys and edge attribute keys present,
- a few sample edges with their attributes,
- distinct values of season, window, and position,
- whether transfer_id exists on edges and is unique,
- how "Outside System" is encoded at club level (verify the claim that many distinct
  NodeIds share the name "Outside System"),
- weight semantics, and how many finance edges have null/missing weight.

Print a concise findings summary. Then update the "Verified schema" section of CLAUDE.md
with what you found. If anything contradicts the spec, surface it clearly and do NOT
code around it. Stop after reporting — no app code yet.

```

**Checkpoint:** you've seen the real schema, CLAUDE.md is updated, discrepancies flagged.

---

## Phase 1 — Scaffold + data layer + P1/P2 (tested)

```
Using the verified schema in CLAUDE.md, set up the project and the shared data layer.

1. Create a venv, requirements.txt (streamlit, networkx, pandas, numpy, plotly,
   python-igraph, scipy), and a multipage Streamlit app with a home page.
2. Build a cached data layer that loads the four networks once and exposes them as both
   networkx graphs (st.cache_resource) and tidy edge DataFrames (st.cache_data).
3. Implement and TEST the two core mechanisms:
   - P1 (transfer_id join, movement <-> finance) with the invariant assertions and a
     LEFT join movement->finance; missing fee stays NULL, never 0.
   - P2 (time-dependent club->league mapping) derived from data, asserting one league
     per (club, season, window).
   Write assertions/tests for both and run them; report results.

Do not implement any of the 37 analyses yet. The app must launch with `streamlit run`
showing a home page and stub navigation. Commit when the data layer and its tests pass.
```

**Checkpoint:** app launches; P1 and P2 tests pass; data layer is the single source for
graphs and edge frames.

---

## Phase 2 — Section 1: Single-network analyses (#1–15)

```
Implement Section 1 (analyses #1-15) from the guide as Streamlit pages, following the
display spec for chart type, top-X, and Outside-System handling for each one.

Reuse the Phase 1 data layer and metrics module; don't re-load graphs per page. Respect
all CLAUDE.md invariants (finance reversal for strength/PageRank, median for fee
distributions, never draw the full club graph, NodeId labelling for Outside System).
Use #15 (squad rebuilding ego-net) as the one place a real node-link drawing appears.

Run the app and confirm every Section 1 page renders with real output before finishing.
Commit. Tell me anything in the data that complicated an analysis.
```

**Checkpoint:** all Section 1 pages render correctly with live data.

---

## Phase 3 — Section 2: Cross-network, same type (#16–18)

```
Implement Section 2 (analyses #16-18) — club vs league, all using the P2 mapping from
the data layer. Keep aggregation within one type so direction stays consistent.

#16 is a reconciliation check: rolling club edges up to leagues via P2 must reproduce
the league network exactly; display the diff and highlight mismatches. Follow the
display spec for #17 and #18. Run the app, verify the pages render, commit, and report
whether the #16 reconciliation is clean (it validates P2).
```

**Checkpoint:** pages render; #16 reconciliation reported (clean or with localised flags).

---

## Phase 4 — Section 3: Cross-network, same granularity (#19–25)

```
Implement Section 3 (analyses #19-25) — movement vs finance, all using P1. Remember
finance endpoints are swapped: to align a movement corridor (sell, buy) with its money,
flip the finance pair to (buy, sell).

Exclude the ~23k unmatched moves (NULL fee) from fee means; never zero them. Use
medians for fee stats. For motif analysis (#24), full enumeration is fine at league
level (11 nodes) but sample / restrict to top-k active clubs at club level. Follow the
display spec per analysis. Run, verify, commit.
```

**Checkpoint:** all Section 3 pages render; #24 uses sampling/top-k at club scale.

---

## Phase 5 — Section 4: All four networks combined (#26–32)

```
Implement Section 4 (analyses #26-32). Align finance to movement by reversing one
finance layer; use P1 to know which edges correspond and P2 for any club->league rollup.

For heavy computations (multilayer #26, community detection #27/#29) use igraph and
leidenalg/infomap; compute expensive measures on top-N subgraphs where needed and label
estimates as such. Community detection is club-level only (11 nodes is too small).
Follow the display spec (cross-layer scatters, alluvial/streamgraph for drift, etc.).
Run, verify each page, commit, and note any computation you approximated or sampled.
```

**Checkpoint:** Section 4 pages render; heavy analyses are cached and labelled as
estimates where approximated.

---

## Phase 6 — Section 5: Temporal league-level Sankeys (#33–37)

```
Implement Section 5 (analyses #33-37) — the temporal Sankey diagrams.

- Node per (league, stage); stages ordered left->right by (season, window). Aggregate
  by groupby(season, window, source_league, target_league) summing weight (player
  counts for movement, fees for finance). transfer_id is not needed here.
- Render every league at every stage even when inactive so lanes persist; label the
  diagram as flow, not stock, unless retention edges are added.
- Make Outside System pinned to one edge of each stage and toggleable; provide a
  with-OS view (true totals) and a without-OS view. Threshold tiny flows into "other".
- Finance Sankey displays as-is (paying league -> receiving league); keep the two
  Sankeys deliberately mirrored and annotate "left = origin of the thing flowing".
- Provide the comparative view (#37): aligned stage axes, plus a small-multiple net-flow
  chart per league per window.

Run, verify both Sankeys and the toggles work, commit.
```

**Checkpoint:** both Sankeys render with stage ordering, persistent lanes, OS toggle, and
aligned comparative view.

---

## Phase 7 — Refactor & consolidate

```
The app was built section-by-section over six phases, so expect accumulated duplication:
repeated aggregation/groupby logic, near-identical page boilerplate, copy-pasted chart
styling, the throwaway Phase 0 exploration script, and overlapping metric helpers. Clean
this up without changing any analysis output.

Do a behaviour-preserving refactor of the whole app. Outputs must stay identical — verify
by re-running, not by assuming.

First, INVENTORY before changing anything:
- List every file and its responsibility. Flag dead/throwaway files (e.g. the Phase 0
  exploration script), unused modules, and unused imports/dependencies.
- Identify duplicated logic across pages: aggregation/groupby patterns, the P1 join, P2
  rollups, top-X selection, Outside-System handling, finance-reversal direction flips, and
  repeated chart styling (ranked bars, scatters, heatmaps, box/violin, Sankey).

Then propose a consolidation plan and show it to me BEFORE deleting or moving files:
- Single-source the shared logic: all P1/P2 and aggregation in the data layer; all metrics
  in the metrics module; a small shared UI/plotting helper module for the recurring chart
  types and the club/league + top-X controls.
- Combine trivially small or redundant page files within a section where it reads better;
  keep pages thin (fetch from cached layers, call a helper, render).
- Move any keep-worthy scripts (e.g. data exploration) into a scripts/ folder; delete true
  throwaways.
- Prune requirements.txt to what's actually imported.

After I approve, apply the plan, then RE-RUN the app and the P1/P2 tests and confirm every
page renders the same output as before. Commit as a single "refactor" commit and give me a
before/after file tree with a one-line note on what each change consolidated.
```


**Checkpoint:** consolidation plan approved before changes; post-refactor app and tests
pass with identical output; before/after file tree delivered.


## Phase 8 — Polish & QA pass
```
Do a final QA and polish pass across the whole app:
- Click through every page; confirm nothing errors and all caches behave.
- Check performance: no expensive recompute on rerun; heavy pages load acceptably.
- Verify Outside-System handling and the finance reversal are consistent everywhere.
- Tidy navigation, page titles, and any obvious UX rough edges per the display spec.
- Update CLAUDE.md / a short README with how to run the app and any known limitations.

Give me a final summary: what's implemented, what's approximate or sampled, and every
data-quality flag you hit during the build.

Checkpoint: full click-through is clean; summary of coverage, approximations, and
data flags delivered.
```

