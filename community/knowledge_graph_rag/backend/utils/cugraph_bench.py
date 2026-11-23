"""NetworkX vs nx-cugraph (backend="cugraph") on the knowledge graph and on a larger synthetic graph.
Usage: python cugraph_bench.py backend/data/knowledge_graph.graphml [report.txt]   (needs networkx and nx-cugraph-cu13, e.g.
  pip install networkx nx-cugraph-cu13 --extra-index-url=https://pypi.nvidia.com). Same calls, different backend= argument."""
import sys, time, platform
import networkx as nx
import nx_cugraph  # noqa: F401  (registers the backend)

GRAPHML = sys.argv[1]
L = []
p = lambda s: (L.append(s), print(s, flush=True))


def timeit(fn, reps=3):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return out, ts


p(f"python {platform.python_version()}, networkx {nx.__version__}, nx-cugraph {nx_cugraph.__version__}, backends {sorted(nx.utils.backends.backends)}")
try:
    import cupy
    p(f"GPU: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception as e:
    p(f"GPU name n/a ({e})")

G = nx.read_graphml(GRAPHML)
p(f"\nOur graph: knowledge_graph.graphml, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (DiGraph)")
S = nx.gnm_random_graph(200_000, 2_000_000, seed=1, directed=True)
p(f"Synthetic: gnm_random_graph(200,000 nodes, 2,000,000 edges, directed)")

cases = [
    ("pagerank", lambda g, b: nx.pagerank(g, backend=b)),
    ("betweenness_centrality (k=500 ours, k=20 synthetic)", lambda g, b: nx.betweenness_centrality(g, k=min(500 if g.number_of_nodes() < 10000 else 20, g.number_of_nodes()), seed=7, backend=b)),
]
for gname, g in [("ours", G), ("synthetic", S)]:
    # convert once, as a real app would keep the GPU graph
    t = time.perf_counter(); gg = nx_cugraph.from_networkx(g); tconv = time.perf_counter() - t
    p(f"\n[{gname}] nx -> nx_cugraph conversion (host->GPU copy): {tconv:.3f}s")
    for cname, fn in cases:
        if gname == "synthetic" and cname.startswith("betweenness"):
            reps_nx = 1
        else:
            reps_nx = 3
        r_nx, t_nx = timeit(lambda: fn(g, "networkx"), reps=reps_nx)
        r_cu, t_cu = timeit(lambda: fn(gg, "cugraph"), reps=3)
        top_nx = sorted(r_nx, key=r_nx.get, reverse=True)[:5]
        top_cu = sorted(r_cu, key=r_cu.get, reverse=True)[:5]
        p(f"  {cname:40s} networkx {min(t_nx):8.3f}s | cugraph {min(t_cu):8.3f}s (first call {t_cu[0]:.3f}s) | "
          f"speed-up x{min(t_nx) / min(t_cu):.1f}")
        if gname == "ours":
            p(f"      top-5 networkx: {top_nx}")
            p(f"      top-5 cugraph : {top_cu}")
p("\nNOTE: the example itself never uses cuGraph: ui_backend.py builds the graph with nx.from_pandas_edgelist and chat.py "
  "walks it with NetworkxEntityGraph (pure NetworkX). With NX_CUGRAPH_AUTOCONFIG=True (or backend='cugraph') the SAME "
  "networkx calls can dispatch to the GPU; dfs_edges (what get_entity_knowledge uses) is a traversal on a 3k-node graph "
  "where the GPU has nothing to win.")
if len(sys.argv) > 2:  # optional: save the report
    open(sys.argv[2], "w").write("\n".join(L) + "\n")
