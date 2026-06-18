"""THROWAWAY follow-up probe. Resolves the anomalies explore_data.py surfaced:
inconsistent position key, extra attrs, P1 subset relationship, mojibake."""
import collections
import networkx as nx

DATA = "data"
FILES = {
    "movement_club":   f"{DATA}/movement_club_net.graphml.xml",
    "movement_league": f"{DATA}/movement_league_net.graphml.xml",
    "finance_club":    f"{DATA}/finance_club_net.graphml.xml",
    "finance_league":  f"{DATA}/finance_league_net.graphml.xml",
}
G = {k: nx.read_graphml(p) for k, p in FILES.items()}

print("### POSITION values per file (probing BOTH 'position' and 'player_position') ###")
for name, g in G.items():
    pos = collections.Counter()
    key_used = collections.Counter()
    for _, _, d in g.edges(data=True):
        for k in ("position", "player_position"):
            if k in d:
                pos[d[k]] += 1
                key_used[k] += 1
    print(f"\n{name}: position-key usage = {dict(key_used)}")
    print(f"   distinct positions ({len(set(p for p in pos))}):")
    for p, c in sorted(pos.items()):
        print(f"      {p!r}: {c}")

print("\n### player_value semantics (where present) ###")
for name in ("movement_club", "finance_league"):
    g = G[name]
    vals = [d["player_value"] for _, _, d in g.edges(data=True) if "player_value" in d]
    if vals:
        types = sorted({type(v).__name__ for v in vals})
        nz = sum(1 for v in vals if v == 0)
        print(f"{name}: n={len(vals)} types={types} min={min(vals)} max={max(vals)} n(==0)={nz}")

print("\n### player_id presence ###")
for name, g in G.items():
    n = sum(1 for _, _, d in g.edges(data=True) if "player_id" in d)
    print(f"{name}: player_id on {n}/{g.number_of_edges()} edges")

print("\n### P1 subset: are finance transfer_ids a subset of movement's? ###")
mv_ids = {d["transfer_id"] for _, _, d in G["movement_club"].edges(data=True)}
fn_ids = {d["transfer_id"] for _, _, d in G["finance_club"].edges(data=True)}
print(f"movement distinct ids : {len(mv_ids)}")
print(f"finance  distinct ids : {len(fn_ids)}")
print(f"finance ids NOT in movement : {len(fn_ids - mv_ids)}")
print(f"movement ids with NO finance (the P1 gap) : {len(mv_ids - fn_ids)}")

print("\n### P1 invariant check on shared ids (movement.src==finance.tgt & movement.tgt==finance.src) ###")
mv_edge = {}
for u, v, d in G["movement_club"].edges(data=True):
    mv_edge[d["transfer_id"]] = (u, v, d.get("player_position"))
violations, checked, pos_div = 0, 0, 0
for u, v, d in G["finance_club"].edges(data=True):
    tid = d["transfer_id"]
    if tid in mv_edge:
        mu, mv_, mpos = mv_edge[tid]
        checked += 1
        if not (mu == v and mv_ == u):
            violations += 1
        fpos = d.get("player_position")
        if mpos != fpos:
            pos_div += 1
print(f"checked={checked} invariant_violations={violations} position_divergences={pos_div}")

print("\n### Mojibake: read raw bytes around a suspect name to confirm file is clean UTF-8 ###")
import re
with open(FILES["movement_club"], "rb") as f:
    blob = f.read(4_000_000)
idx = blob.find(b"Universidad Cat")
if idx != -1:
    snippet = blob[idx-5:idx+30]
    print(f"   raw bytes: {snippet!r}")
    try:
        print(f"   decoded utf-8: {snippet.decode('utf-8', errors='replace')!r}")
    except Exception as e:
        print("   utf-8 decode error:", e)

print("\n### Outside System node id at club level ###")
for name in ("movement_club", "finance_club"):
    g = G[name]
    os_ids = [nid for nid, d in g.nodes(data=True) if d.get("name") == "Outside System"]
    print(f"{name}: 'Outside System' -> node id(s) {os_ids}")