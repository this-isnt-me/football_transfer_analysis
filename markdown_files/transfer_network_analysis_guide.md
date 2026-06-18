# Analytical Methodology Guide — Football Transfer Networks

A structured set of analytical approaches for four directed, weighted transfer networks:

| Network | Grain | Edge direction | Weight | Size |
|---|---|---|---|---|
| `movement_club_net` | Club | seller → buyer (player path) | 1 | 5,598 nodes / 166,258 edges |
| `movement_league_net` | League | seller → buyer (player path) | 1 | 11 nodes / 166,258 edges |
| `finance_club_net` | Club | **buyer → seller** (money path) | fee / valuation | 4,437 nodes / 142,853 edges |
| `finance_league_net` | League | **buyer → seller** (money path) | fee / valuation | 11 nodes / 142,853 edges |

---

## Preliminaries — two mechanisms used throughout

Two technical patterns recur in almost every cross-network or aggregation step. Define them once and reuse.

### P1. The `transfer_id` join (movement ⇄ finance)

For one real-world deal, the movement edge runs `sell_club → buy_club` and the finance edge runs `buy_club → sell_club` — they are **exact reverses sharing the same `transfer_id`**. To join:

- Key on `transfer_id` (unique per deal).
- **Validation invariant:** after the join, `movement.source == finance.target` and `movement.target == finance.source`; `position` should be identical on both sides. Rows that violate this flag data errors.
- **Cardinality:** movement has 166,258 edges vs finance's 142,853 — a gap of **~23,405 transfers with no financial counterpart** (free transfers, loans, or deals with neither a fee nor a valuation). Always use a **left join from movement → finance**, and treat the missing fee as **NULL, not zero** — a free transfer (known €0) and an unrecorded fee are different and should never be averaged together.

### P2. Time-dependent club → league mapping

Clubs change leagues across seasons (promotion/relegation), so **no static club→league lookup is valid**. Build the mapping *from the data*:

- Each `transfer_id` appears in both the club network and the league network. The club edge gives `club_source → club_target`; the league edge gives `league_source → league_target`. Therefore each transfer reveals **`(club, season, window) → league`** for both endpoints at that moment.
- Construct the lookup by joining the club and league networks of the *same type* on `transfer_id`, then collecting `(club, season, window, league)` tuples from both source and target sides.
- Within a single `(season, window)` a club should resolve to exactly one league; if it resolves to two, that is a data-quality flag.
- "Outside System" collapses all clubs outside the ten leagues into one league label. Confirm at the outset whether club-level *nodes* for external clubs are kept individually or also collapsed — this changes node counts and several interpretations below.

---

# Section 1 — Single Network Analyses

### 1. In-Degree / Out-Degree
**Measures:** count of incoming/outgoing transfer edges per node. In movement, out-degree = players sold/released, in-degree = players recruited. **Insight:** net exporters vs importers of talent; squad churn. **Method:** plain edge count (`groupby` source/target), or `in_degree`/`out_degree`. **Direction:** in finance the reversal inverts meaning — finance out-degree counts *purchases* (you were the payer), in-degree counts *sales*; degree is far less informative on finance than strength (#9). **Level:** both. **Feasibility:** trivial, O(E).

### 2. Weighted PageRank
**Measures:** recursive prestige — a node gains standing by being pointed to by important nodes. On movement (`A→B` = player leaves A, joins B), PageRank rewards **attractive destinations** that pull players from other attractive clubs. **Insight:** the talent-pull hierarchy / "top of the food chain." **Method:** `networkx.pagerank(weight='weight')`; since movement weight is 1, aggregate parallel edges to counts first (essential at league level). **Direction:** reverse the graph to score *selling* prestige instead. **Level:** both. **Feasibility:** O(E)/iteration, converges in seconds even at 5,598 nodes.

### 3. Betweenness Centrality
**Measures:** how often a node lies on shortest transfer paths — brokers / stepping-stones. **Insight:** "conduit" clubs and leagues in talent pipelines (e.g. mid-tier leagues bridging development clubs to elite buyers). **Method:** Brandes algorithm. **Direction:** run on **movement** with hop-distance (weight=1); avoid finance, where treating a large fee as a "long distance" is meaningless. **Level:** club-level only — at 11 nodes betweenness is degenerate and dominated by Outside System. **Feasibility:** exact is ~O(V·E) ≈ 9×10⁸, heavy. Use a C-backed library (`igraph`, `graph-tool`) and/or **approximate with k pivot sources** (`networkx.betweenness_centrality(k=500–1000)`); report it as an estimate.

### 4. Position-Filtered Subgraph Analysis
**Measures:** filter edges by `position`, rebuild the subgraph, then re-run any node metric. **Insight:** clubs/leagues that specialise in producing or buying a position (a league as a net striker-exporter). **Method:** edge subset on `position` → recompute degree/PageRank. **Caveat:** normalise the `position` taxonomy first (collapse synonyms to GK/DEF/MID/FWD or a fixed granular set). **Level:** both. **Feasibility:** trivial.

### 5. Reciprocity Analysis
**Measures:** share of edges with a reciprocal partner — do `A→B` deals coincide with `B→A`? **Insight:** standing trading partnerships and corridors (repeat buyer–seller pairs, loan relationships, swap deals). **Method:** collapse the multigraph to a simple directed graph (`A→B` count) first, then `overall_reciprocity`; for weighted balance compare dyadic flows. **Direction:** in finance, mutual edges mean two clubs paying each other (rare) — more interesting is comparing movement reciprocity (player swaps) against finance. **Level:** both; club-level is the more revealing scale. **Feasibility:** trivial after aggregation.

### 6. Seasonal Degree Evolution
**Measures:** in/out-degree per node across `season`. **Insight:** clubs rising (takeover-driven expansion) or contracting (distress selling). **Method:** `groupby(season)` → per-node degree time series; plot trajectories for top-N or cluster them. **Level:** both. **Feasibility:** trivial.

### 7. Transfer Window Comparison (Summer vs Winter)
**Measures:** structure and volume split by `window`. **Insight:** summer squad-building vs winter reactive/emergency signings; who is active in January (relegation battlers, injury crises). **Method:** partition edges by `window`; compare degree distributions, volumes, `position` mix. **Level:** both. **Feasibility:** trivial.

### 8. Position Supply Trends Over Time
**Measures:** transfer volume per `position` per `season`. **Insight:** tactical/market shifts (rising demand for specific roles). **Method:** `groupby(season, position)` count on movement (weight=1 = supply). **Level:** both; league-level cleaner for macro trends. **Feasibility:** trivial.

### 9. Weighted Out-Strength / In-Strength
**Measures:** sum of edge `weight` per node. **This is the analysis most exposed to the finance reversal.** Because finance edges are buyer→seller: **out-strength = total fees *paid* = gross spend**; **in-strength = total fees *received* = sales revenue**. Net transfer profit = in-strength − out-strength. In movement, weight=1, so strength = degree. **Insight:** biggest spenders vs biggest sellers; net-spend leaderboards. **Method:** `groupby(source).weight.sum()` (out), `groupby(target).weight.sum()` (in). **Level:** both; most meaningful on finance. **Feasibility:** trivial.

### 10. Flow Concentration (Gini)
**Measures:** inequality of flow across nodes — Gini/Lorenz of in-strength, out-strength, or edge weights. **Insight:** market domination by a few super-clubs/leagues; expect movement Gini < finance Gini (money is far more concentrated than bodies). **Method:** Gini on the strength vector. **Level:** both — across 11 leagues it quantifies league dominance, across clubs intra-market concentration. **Feasibility:** trivial.

### 11. Weighted PageRank on Financial Network
**Measures:** PageRank with `weight`=fee on finance. Given buyer→seller direction, importance accrues to **sellers that extract large fees from wealthy buyers** — the prestige-selling hierarchy. **Insight:** clubs/leagues that command premium fees from elite buyers (feeders to the giants). **Method:** `pagerank(weight='weight')`. **Direction:** as-is = selling power; **reverse the graph** to measure buying/spending prestige — offer both. **Level:** both. **Feasibility:** trivial-to-moderate.

### 12. Seasonal Spending Trajectory
**Measures:** per-node spend (finance out-strength) and revenue (in-strength) across `season`. **Insight:** spending ramps (TV-deal inflation, takeover eras), austerity dips (e.g. 2020–21). **Method:** `groupby(node, season)` summing `weight` on outgoing (spend) and incoming (revenue) finance edges. **Direction:** spend = source side (reversal). **Caveat:** fees are nominal — deflate by a transfer-market index or convert to share-of-season-total before comparing eras. **Level:** both. **Feasibility:** trivial.

### 13. Winter vs Summer Financial Comparison
**Measures:** fee volume/distribution split by `window`. **Insight:** winter "desperation premium"; clubs that systematically overpay in January. **Method:** `groupby(window)` on `weight`; compare median fee per deal (ties to #20). **Level:** both. **Feasibility:** trivial.

### 14. Position-Based Valuation Analysis
**Measures:** distribution of fee (`weight`) by `position`, optionally × `season`. **Insight:** which positions command premiums and how that repricing moves over time. **Method:** `groupby(position)` fee stats — use **median/percentiles**, not mean (heavy right skew). **Direction:** edge-attribute analysis, but note you are on finance weights. **Level:** both; essentially edge-level/market-level. **Feasibility:** trivial.

### 15. Squad Rebuilding Visualisation
**Measures:** a single club's ego-network for one `season`/`window` — who it bought from and sold to, by `position`, with fees. **Insight:** reconstructs a club's strategy in a rebuild (newly promoted side, post-takeover overhaul). **Method:** `ego_graph(radius=1)` filtered on `season`/`window`; colour edges by `position`, width by fee (attach fee via the **P1 `transfer_id` join** to the movement ego-net). Tools: Gephi, networkx+matplotlib, chord/ego layouts. **Direction:** combine movement (who) + finance (how much, reversed) via P1. **Level:** club-level by definition. **Feasibility:** trivial.

---

# Section 2 — Cross-Network (Same Type: Club vs League)

All three require the **P2** time-dependent mapping; perform aggregation within one type so direction stays consistent.

### 16. Aggregation Consistency Check
**Measures:** does rolling club edges up to leagues reproduce the league network? **Insight:** validates P2 and surfaces mapping errors, ambiguous clubs, and Outside-System handling. **Method:** using P2, relabel each club edge's endpoints with their `(season, window)` league, then `groupby(source_league, target_league, season, window, position)` and sum; compare to the league network's counts/weights — they should match exactly. Discrepancies localise data issues. **Level:** cross. **Feasibility:** trivial groupby; the P2 join (O(E)) is the real work.

### 17. Node Ranking Correlation
**Measures:** rank correlation (Spearman/Kendall) between a league-level metric and the aggregate of its member clubs' metric. **Insight:** is a league's standing broad-based or carried by one or two mega-clubs (LaLiga's PageRank dominated by two clubs vs a flatter Premier League)? **Method:** compute the club metric, aggregate at the `(club, season, window)` grain via P2, roll up to league, correlate against the league-net metric. **Level:** cross. **Feasibility:** trivial post-metrics.

### 18. Temporal Divergence Between Scales
**Measures:** compare a metric's `season` trajectory at club vs league scale. **Insight:** e.g. flat league spend but rising intra-league concentration (rich-get-richer within a league), or league growth driven purely by a big-spender's promotion. **Method:** compute per-season at both scales (P2 for the rollup); correlate trajectories and flag divergence windows. **Level:** cross. **Feasibility:** trivial.

---

# Section 3 — Cross-Network (Same Granularity: Movement vs Finance)

Every analysis here uses **P1**. Remember finance endpoints are swapped: to align a movement corridor `(sell, buy)` with its money, flip the finance pair to `(buy, sell)`.

### 19. Player Flow vs Money Flow Correlation
**Measures:** per node or directed pair, players moved vs money moved. **Insight:** high-volume/low-value corridors (youth feeder pipelines) vs low-volume/high-value ones (occasional marquee sales). **Method:** aggregate movement by `(sell, buy)` → player count; aggregate finance by `(buy, sell)` → fee sum; align by flipping the finance pair; correlate. **Direction:** the flip is mandatory. **Level:** both. **Feasibility:** trivial.

### 20. Fee Per Player (Edge-Level Efficiency)
**Measures:** join each movement edge to its finance edge (P1) to get fee per individual deal, then average per corridor/node. **Insight:** corridors/clubs commanding the highest average fee (quality over quantity) vs feeders (low fee/player). **Method:** P1 inner-join attaches `weight` to each move; the ~23k unmatched moves carry NULL fee — exclude from the mean, don't zero them. **Direction:** the join is direction-agnostic (keyed on `transfer_id`), but the source/target swap is a useful validation. **Level:** both. **Feasibility:** trivial.

### 21. Prestige Divergence
**Measures:** gap between a node's movement-PageRank rank and finance-PageRank rank. **Insight:** "talent magnets that aren't cash magnets" — destinations players choose for sporting reasons at modest fees — vs pure cash extractors. **Method:** compute both PageRanks, **align their semantics first** (e.g. for *selling* prestige use reversed movement vs as-is finance; for *buying* prestige use as-is movement vs reversed finance), rank-correlate, flag large gaps. **Direction:** semantic alignment is the crux. **Level:** both. **Feasibility:** moderate (two PageRanks).

### 22. Positional Fee Efficiency Over Time
**Measures:** fee-per-player by `position` × `season` (P1 join, then group). **Insight:** positional inflation cycles (e.g. fullbacks/keepers repricing in certain eras). **Method:** P1 join → `groupby(position, season)` median fee; cross-check that `position` agrees on both sides. **Level:** both. **Feasibility:** trivial.

### 23. Window Arbitrage Analysis
**Measures:** systematic fee-per-player and volume differences by `window` for comparable moves. **Insight:** the January premium/discount and clubs that time the market. **Method:** P1 join → `groupby(window)` median fee, ideally controlling for `position` (and corridor) to compare like with like. **Level:** both. **Feasibility:** trivial.

### 24. Motif Analysis
**Measures:** frequencies of small subgraph patterns — reciprocal pairs, chains `A→B→C`, triangles — optionally typed by combining movement + finance. **Insight:** talent escalators (3-tier feeder→mid→elite pipelines), swap deals (reciprocal move + offsetting fees), money-recycling loops. **Method:** triad census (`igraph.triad_census`, FANMOD/ESU). **Direction:** directed motifs; a movement chain `A→B→C` maps to finance `C→B→A` — read as money flowing back up the pipeline. **Level:** league trivial (full enumeration on 11 nodes); club **expensive** — restrict to top-k active clubs, or use **ESU/RAND-ESU sampling** or a C-backed library. **Feasibility:** the main computational caution in this section — sample at club level.

### 25. Capital Flow Asymmetry by Position
**Measures:** per node × `position`, players-in vs players-out (movement) against money-in vs money-out (finance). **Insight:** "sell expensive strikers, buy cheap ones" trading patterns; leagues exporting pricey attackers while importing cheap defenders. **Method:** per `(node, position)` compute movement in/out counts and finance in/out-strength; form asymmetry ratios; roll to league via P2 if needed. **Direction:** money-out = spend = finance out-strength (reversal). **Level:** both. **Feasibility:** trivial.

---

# Section 4 — All Four Networks Combined

Align finance to movement by reversing one finance layer so layers are comparable; use P1 to know which edges correspond and P2 for any club→league rollup.

### 26. Multi-Layer Network Analysis
**Measures:** model clubs (and leagues) as nodes with **movement and finance as layers** (and the two scales as further layers). Compute multilayer centrality / versatility and inter-layer correlation. **Insight:** integrated importance — nodes central in *both* talent and money flow vs only one. **Method:** `muxViz`, `pymnet`, or aligned `igraph`/`graph-tool` layers; reverse the finance layer first; multilayer PageRank / eigenvector versatility. **Direction:** state which layer was reversed. **Level:** both scales as layers (true multi-scale). **Feasibility:** club-level multilayer is heavy — use C-backed tools or compute expensive measures on a top-N subgraph.

### 27. Community Detection & Cross-Layer Comparison
**Measures:** clusters of clubs that trade together, per layer, compared across layers and scales. **Insight:** trading ecosystems/blocs (e.g. a Portugal↔Premier-League pipeline); do money-communities match player-communities? **Method:** **Leiden** (weighted/directed) or, better for directed flows, **Infomap** (flow-based); compare partitions with NMI/ARI. **Direction:** Infomap respects direction; reverse finance to align semantics or interpret money-flow communities directly. **Level:** club-level (11 nodes is too small to cluster meaningfully). **Feasibility:** scales to 5,598 nodes/166k edges in seconds.

### 28. Dominance Index
**Measures:** a composite per-node score blending metrics across all four nets — net talent gain (movement in−out), net spend (finance out−in), and prestige (movement + finance PageRank). **Insight:** who truly dominates = sporting pull × financial muscle. **Method:** z-score each component across nodes (direction-correcting finance inputs), then weighted sum or PCA first component; report weighting sensitivity. **Level:** both. **Feasibility:** trivial once components exist.

### 29. Temporal Community Drift
**Measures:** run community detection per `season` (or `window`) and track formation/merging/splitting. **Insight:** emergence of new trading blocs or realignment after rule/financial changes. **Method:** per-period Leiden/Infomap; align communities across time by Jaccard overlap; quantify drift. **Level:** club-level. **Feasibility:** many runs, each cheap; no sampling needed.

### 30. Shock Detection & Propagation
**Measures:** detect anomalous `season`/`window` spikes or drops and trace propagation through the network. **Insight:** the 2020–21 dip, mega-sale → reinvestment cascades, rule/FFP shocks, promotion windfalls. **Method:** change-point/anomaly detection on per-node and aggregate series (STL, PELT, Bayesian change points); trace cascades by following money via P1 across ordered windows (in-strength at window *t* → out-strength at *t*/*t*+1). **Direction:** money in then re-spent — respect the finance reversal in the trace. **Level:** both (league for macro shocks, club for specific cascades). **Feasibility:** moderate; per-node change-points and graph traversal are cheap.

### 31. Position-Stratified Multi-Layer Analysis
**Measures:** build the full four-net multilayer **per `position`** and compare structure across positions. **Insight:** position-specific market topologies — the striker market global, concentrated, expensive; the goalkeeper market structurally different; which leagues own which positional submarket by volume vs by value. **Method:** `position` filter → four sub-nets per position → centrality/community/dominance per position. **Level:** both. **Feasibility:** trivial subsetting; metrics cheap on subsets.

### 32. Feeder Club Specialisation
**Measures:** identify feeders — high movement out-degree to a *concentrated* set of higher-prestige destinations, with **net money in** (finance in-strength > out-strength). **Insight:** the selling/development tier of the pyramid (Eredivisie, Liga Portugal, Championship→PL) and how specialised each is. **Method:** per node: movement out-degree; destination concentration (Herfindahl/entropy of out-targets weighted by their movement PageRank); finance net (in−out). Classify by league via P2; rank destinations by movement PageRank. **Direction:** feeder signature = players out (movement) + money in (finance in-strength) — be explicit about the reversal. **Level:** both (feeder clubs and feeder leagues). **Feasibility:** trivial.

---

# Section 5 — Sankey Diagrams (League-Level Temporal Flow)

Stages run chronologically: Summer 2010 → Winter 2010 → Summer 2011 → … Eleven possible league nodes (the ten leagues + Outside System).

### 33. Node and Flow Construction
Use **one node per league per stage** ("LaLiga @ Summer 2010"), *not* a single persistent node — a temporal Sankey needs distinct nodes at each stage for flows to span stage *t* → *t*+1. With ~2 windows/year over ~14 seasons that is ~28 stages × 11 ≈ 300 nodes.

Model it as a **state-transition Sankey**: a flow from `LeagueA @ t → LeagueB @ t+1` represents players (or money) whose league-state changed between consecutive windows, i.e. transfers occurring in that window. Aggregate by `groupby(season, window, source_league, target_league)` summing `weight` (player counts for movement, fees for finance). `transfer_id` is not needed here — simple aggregation of parallel edges suffices. Define the stage order explicitly as an ordered categorical of `(season, window)`.

### 34. Self-Loops and Retention
A pure transfer dataset records only *movers*, not *stayers*, so a strict conservation Sankey isn't directly available. Two honest options:

- **Flow-only Sankey (recommended default):** each stage's height = that window's transfer volume; clearly label it as *flow*, not *stock*. No retention edges.
- **Conservation Sankey:** add explicit self-retention edges `LeagueA @ t → LeagueA @ t+1`, but computing "players who stayed" requires a squad-size baseline **outside these four networks** — flag that dependency.

For a league with **no activity in a window**, still render the node (epsilon height or a carried-forward retention edge) so its lane persists across stages; dropping it would falsely imply the league vanished. Tools: Plotly Sankey, `d3-sankey`, R `networkD3`.

### 35. Scalability and Filtering
~300 nodes is manageable, but up to 121 flows per transition × ~27 transitions clutters fast. Mitigate by: thresholding small flows into an "other" band, interactive league toggles, colour-by-league, and a selectable stage range.

**Outside System** is almost certainly the highest-volume node (all external deals). Make it **toggleable/collapsible and visually parked** (e.g. pinned to one edge of each stage): provide one view with it shown (true total volume) and one with it filtered (so intra-system inter-league patterns aren't drowned out).

### 36. Direction Handling for the Finance Sankey
Money flows buyer → seller, which is already the finance edge direction. Display **as-is**: source side = paying league (money out), target side = receiving league (money in); flow *into* a league node = revenue received, flow *out of* it = spend.

Deliberately keep the two Sankeys **mirrored**: the movement Sankey follows the *player* (selling league → buying league), the finance Sankey follows the *money* (buying league → selling league). So for the same deal a league sits on opposite sides of the two diagrams — annotate clearly ("left = origin of the thing flowing"). That mirroring *is* the comparative signal in #37. (If you prefer visual parallelism over semantic honesty, reverse finance so both put the same league on the same side — but label it as reversed.)

### 37. Comparative Reading
Place the two Sankeys with aligned stages and read each league per window:

- Big movement **outflow** + big finance **inflow** = a profitable selling/feeder league.
- Big movement **inflow** + disproportionate finance **outflow** = a premium-paying buyer league.
- Where *volume* and *value* diverge, you've found market (in)efficiency.

Because finance is mirrored, the same league appears on opposite sides — guide the reader explicitly. **Outside System** must be read with care: it is not a peer league but a heterogeneous aggregate of many external divisions, so flows to/from it represent the ten leagues' *net interaction with the rest of world football*, not a single market. Interpret high Outside-System volume as the system's role as a global net buyer or seller, not as trade with one comparable entity.

# Display & Visualisation Spec — Companion to the Analysis Guide

How to present the output of each of the 37 analyses: display type, whether to focus on a top-X subset, and how to handle "Outside System". Numbering matches `transfer_network_analysis_guide.md`.

---

## D0. Handling "Outside System" at club level (important)

At **club level**, "Outside System" is a **placeholder *name* shared by many distinct `NodeId`s** — each external club is kept as its own node but anonymised under the same name. This has direct display consequences:

- **Always label club-level outputs by `NodeId` (or `NodeId` + name), never by name alone.** Labelling by name produces multiple indistinguishable "Outside System" rows.
- **Two ranking modes — choose per analysis:**
  - *Keep IDs separate (default for rankings):* each external club is individually low-volume and falls outside any top-X naturally, so your leaderboards stay populated by real in-system clubs. Cleanest.
  - *Collapse all "Outside System" IDs into one aggregate node:* useful to measure total interaction with the rest of world, but the aggregate is huge — **exclude it from top-X rankings and show it as a separate reference bar/row**, never ranked beside individual clubs.
- **Network/ego visualisations:** either hide external nodes or merge them into one "rest of world" node to declutter; kept separate they form a low-degree halo.
- **Caveat on collapsing:** merging by name fuses genuinely different external clubs (they're anonymised, so you can't tell them apart anyway) — fine for a "rest of world" framing, never use it to distinguish specific external clubs.
- **At league level this problem disappears:** every "Outside System" club `NodeId` maps (via P2) to the single "Outside System" league, so the multi-ID issue exists only at club scale.

## D1. General display principles

- **Club networks (≈5,600 nodes): never draw the full node-link graph.** Default outputs are **ranked tables / top-X bars**, **distributions** (histogram/box/violin), **scatters**, and **time series**. True network drawings only for ego-nets (#15) or community-coloured layouts on a *filtered or top-X* subgraph (#27).
- **League networks (11 nodes): node-link graphs, chord diagrams, and 11×11 heatmaps all work directly** — and small-multiples over `season`/`window`.
- **Top-X guidance:** ~15–20 for readable bar charts, ~25–30 for tables. Distributions use all nodes. League charts always show all 11.

---

## Sections 1–4 — per-analysis display

| # | Analysis | Primary display | Focus / Top-X | Outside System (club) |
|---|---|---|---|---|
| 1 | In/Out-Degree | Two ranked bars (in, out) **+ in-vs-out scatter** with parity diagonal (net importers/exporters) | Top-20 each bar; scatter = all | Exclude OS aggregate from bars, or keep IDs separate so they self-filter |
| 2 | Weighted PageRank | Ranked bar / table; league: node-link with node size = score | Top-20–30 | Exclude/aggregate; show separately |
| 3 | Betweenness | Ranked **table of brokers** (short list); optional force layout of top subgraph, size = score | Top-15; show as ranks (values approximate) | Exclude OS aggregate |
| 4 | Position-filtered subgraph | **Heatmap** club × position (value = degree/strength), or small-multiple bars per position | Top-X clubs as rows | Exclude OS row |
| 5 | Reciprocity | Single **scalar/gauge** per network + table of top reciprocal dyads; league: chord / 11×11 matrix | Top dyads | Drop OS dyads or label clearly |
| 6 | Seasonal Degree Evolution | **Heatmap** club × season (avoid spaghetti); league: 11 lines | Top-X clubs only | OS as its own row/line |
| 7 | Window comparison | Paired/grouped bar (Summer vs Winter); per-club slope chart | Top-X for slope | Separate OS bar |
| 8 | Position supply trends | **Stacked area or multi-line** over season, series = position | All positions | n/a (edge attr) |
| 9 | Out/In-Strength | Spend & revenue ranked bars **+ spend-vs-revenue scatter** + diverging net-spend bar | Top-20; net-spend top & bottom 15 | Exclude OS aggregate from leaderboards |
| 10 | Gini | **Scalar + Lorenz curve** (overlay movement vs finance, or by season) | n/a | Decide in/out of population, state it |
| 11 | Finance PageRank | Ranked bar/table (selling prestige); league: node-link, size = score | Top-20–30 | Exclude/aggregate |
| 12 | Seasonal spending trajectory | Multi-line net-spend over season (top-X) or cumulative lines; league: 11 lines | Top-X spenders | OS own line; note deflation |
| 13 | Winter vs Summer financial | Paired bar of totals **+ box/violin** of fee distribution by window | All / both levels | Separate OS |
| 14 | Position-based valuation | **Box/violin** of fee by position (sorted by median); facet by season for trend | All positions | n/a (edge attr) |
| 15 | Squad rebuilding | **Ego node-link diagram** (edge colour = position, width = fee) or single-club chord/Sankey | One club, one window | Merge external counterparties into one "rest of world" node |
| 16 | Aggregation consistency | **Reconciliation/diff table** (or discrepancy heatmap; should be all-zero), highlight mismatches | Flagged rows only | Check OS mapping reconciles |
| 17 | Node ranking correlation | **Scatter** (league-net vs club-aggregate metric), 11 labelled points, ρ annotated | All 11 | OS as one labelled point |
| 18 | Temporal divergence (scales) | Two-panel / dual-axis line over season, divergence windows shaded | n/a | OS contribution notable — keep visible |
| 19 | Flow vs money correlation | **Scatter** (players vs fees per corridor), log-log, quadrant labels | All pairs; annotate outliers | Flag OS-involved corridors distinctly |
| 20 | Fee per player | Ranked bar of corridors/clubs by mean fee **+ per-deal histogram**; or bubble (volume vs avg fee) | Top-X corridors | Exclude/flag OS corridors |
| 21 | Prestige divergence | **Slope / dumbbell chart** (movement rank ↔ finance rank); or rank-rank scatter, off-diagonal = divergent | Top-X divergent | Exclude OS aggregate |
| 22 | Positional fee efficiency / time | **Heatmap** position × season (median fee), or multi-line | All positions | n/a |
| 23 | Window arbitrage | Paired box (Summer vs Winter median fee/player), faceted by position | All positions | n/a |
| 24 | Motif analysis | **Motif-profile bar** with motif glyphs; z-scores vs null model (error bars if sampled) | Standard triad set | n/a (structural) |
| 25 | Capital flow asymmetry / position | **Diverging heatmap** node × position (centred at parity); or diverging bars for selected clubs | Top-X clubs | Exclude OS rows |
| 26 | Multi-layer network | **Cross-layer scatter** (movement vs finance centrality) + ranked versatility table; league: layered/multiplex node-link | Top-X table; all 11 league viz | Exclude OS from club scatter |
| 27 | Community detection & cross-layer | **Community-coloured force layout** (filtered/aggregated graph) **+ alluvial** comparing partitions across layers + NMI/ARI | Drawable subgraph | Merge external into one node before layout |
| 28 | Dominance index | Ranked bar **+ radar / parallel-coordinates** of component breakdown for leaders | Top-15 | Exclude OS aggregate |
| 29 | Temporal community drift | **Alluvial / streamgraph** of community membership over season (split/merge bands) | Major communities | n/a |
| 30 | Shock detection & propagation | **Time series with change-points flagged** (shaded/marked) + cascade graph/Sankey for a specific shock | Anomalous windows | OS flows can mask shocks — view with/without |
| 31 | Position-stratified multilayer | **Small multiples** per position (chord or cross-layer scatter); or league × position dominance heatmap | Per position | Consistent OS treatment across facets |
| 32 | Feeder club specialisation | **Scatter** (players sold vs net money in; bubble = destination concentration; colour = league) **+ ranked feeder table** with top destinations | Top-X feeders | Exclude OS aggregate from feeder ranking |

---

## Section 5 — Sankey display (encoding notes)

The display *is* the Sankey, so this is about encoding rather than chart choice.

- **#33 Layout:** node-per-(league, stage); stages ordered left→right by `(season, window)`. Colour bands by league (consistent palette across both diagrams); flow width = aggregated `weight` (player counts for movement, fees for finance).
- **#34 Persistence:** render every league at every stage even when inactive (epsilon height or carried retention edge) so lanes don't disappear; clearly label the diagram as *flow* unless you've added explicit retention edges.
- **#35 Outside System:** **pin it to one edge of each stage and make it toggleable.** Provide two saved views — *with* OS (true totals) and *without* OS (intra-system inter-league structure becomes legible). Thresholding tiny flows into an "other" band also helps.
- **#36 Direction:** display finance as-is (paying → receiving): flow *into* a league band = revenue, *out of* it = spend. Keep the two Sankeys deliberately mirrored (movement follows the player, finance follows the money) and annotate "left = origin of the thing flowing."
- **#37 Comparative reading:** stack the two Sankeys vertically (or side-by-side) with **aligned stage axes** so each window lines up. A useful companion view is a small-multiple **net-flow chart per league per window** (net players vs net money) to read divergence numerically alongside the visual.
