"""THROWAWAY Phase-0 exploration. Verifies the documented schema against the
actual GraphML files. Safe to delete after reporting."""
import collections
import itertools
import networkx as nx

DATA = "data"
FILES = {
    "movement_club":   f"{DATA}/movement_club_net.graphml.xml",
    "movement_league": f"{DATA}/movement_league_net.graphml.xml",
    "finance_club":    f"{DATA}/finance_club_net.graphml.xml",
    "finance_league":  f"{DATA}/finance_league_net.graphml.xml",
}
EXPECTED = {  # from the guide's table
    "movement_club":   (5598, 166258),
    "movement_league": (11,   166258),
    "finance_club":    (4437, 142853),
    "finance_league":  (11,   142853),
}


def attr_keys(elem_attr_iter):
    keys = set()
    for _, d in elem_attr_iter:
        keys |= set(d.keys())
    return keys


def edge_attr_keys(G):
    keys = set()
    for _, _, d in G.edges(data=True):
        keys |= set(d.keys())
    return keys


for name, path in FILES.items():
    print("=" * 78)
    print(f"NETWORK: {name}  ({path})")
    print("=" * 78)
    G = nx.read_graphml(path)
    nn, ne = G.number_of_nodes(), G.number_of_edges()
    exp_n, exp_e = EXPECTED[name]
    flag_n = "" if nn == exp_n else f"  <<< MISMATCH expected {exp_n}"
    flag_e = "" if ne == exp_e else f"  <<< MISMATCH expected {exp_e}"
    print(f"  type            : {type(G).__name__}  (multigraph={G.is_multigraph()}, directed={G.is_directed()})")
    print(f"  nodes           : {nn}{flag_n}")
    print(f"  edges           : {ne}{flag_e}")

    nkeys = attr_keys(G.nodes(data=True))
    ekeys = edge_attr_keys(G)
    print(f"  node attr keys  : {sorted(nkeys)}")
    print(f"  edge attr keys  : {sorted(ekeys)}")

    # sample edges
    print("  sample edges    :")
    for u, v, d in itertools.islice(G.edges(data=True), 3):
        print(f"     {u!r} -> {v!r}  {d}")

    # distinct season / window / position
    seasons, windows, positions = set(), set(), set()
    tids = []
    weights = []
    wt_missing = 0
    for _, _, d in G.edges(data=True):
        if "season" in d:
            seasons.add((type(d["season"]).__name__, d["season"]))
        if "window" in d:
            windows.add(d["window"])
        if "position" in d:
            positions.add(d["position"])
        if "transfer_id" in d:
            tids.append(d["transfer_id"])
        if "weight" in d and d["weight"] is not None:
            weights.append(d["weight"])
        else:
            wt_missing += 1

    season_vals = sorted({s[1] for s in seasons})
    season_types = sorted({s[0] for s in seasons})
    print(f"  season          : types={season_types} range={min(season_vals)}..{max(season_vals)} n={len(season_vals)}")
    print(f"  window distinct : {sorted(windows)}")
    print(f"  position distinct ({len(positions)}): {sorted(positions)}")

    # transfer_id presence/uniqueness
    if tids:
        ctr = collections.Counter(tids)
        ndist = len(ctr)
        maxmult = max(ctr.values())
        ndup = sum(1 for c in ctr.values() if c > 1)
        mult_hist = collections.Counter(ctr.values())
        print(f"  transfer_id     : present on {len(tids)}/{ne} edges; distinct={ndist}; "
              f"unique={'YES' if ndist == len(tids) else 'NO'}; max_multiplicity={maxmult}; "
              f"ids_with_dupes={ndup}; multiplicity_hist={dict(sorted(mult_hist.items()))}")
        print(f"  sample tids     : {tids[:3]}  (type={type(tids[0]).__name__})")
    else:
        print("  transfer_id     : ABSENT on edges")

    # weight semantics
    if weights:
        wmin, wmax = min(weights), max(weights)
        wtypes = sorted({type(w).__name__ for w in weights})
        n_one = sum(1 for w in weights if w == 1 or w == 1.0)
        allone = all(w == 1 or w == 1.0 for w in weights)
        print(f"  weight          : types={wtypes} range={wmin}..{wmax} all==1?{allone} n(weight==1)={n_one}")
    print(f"  edges w/ missing/null weight : {wt_missing}")

    # Outside System encoding (club level)
    if "club" in name:
        os_ids = [nid for nid, d in G.nodes(data=True) if d.get("name") == "Outside System"]
        os_named_id = [nid for nid in G.nodes() if nid == "Outside System"]
        print(f"  'Outside System' nodes by name attr : {len(os_ids)} node id(s)")
        print(f"     sample ids: {os_ids[:5]}")
        print(f"  node id literally 'Outside System' exists? {bool(os_named_id)}")
        # name -> count of distinct ids, find names mapping to multiple ids
        name_to_ids = collections.defaultdict(list)
        for nid, d in G.nodes(data=True):
            name_to_ids[d.get("name")].append(nid)
        dup_names = {nm: ids for nm, ids in name_to_ids.items() if len(ids) > 1}
        print(f"  names mapping to >1 node id : {len(dup_names)}")
        for nm, ids in itertools.islice(dup_names.items(), 6):
            print(f"     {nm!r} -> {len(ids)} ids: {ids}")
    else:
        # league: list nodes with names
        print("  league nodes (id -> name):")
        for nid, d in sorted(G.nodes(data=True)):
            print(f"     {nid!r} -> {d.get('name')!r}")
    print()