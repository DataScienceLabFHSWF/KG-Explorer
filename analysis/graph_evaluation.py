"""
Graph Evolution Evaluation
==========================
Computes a structural snapshot of the Fusion KG — the same metrics tracked
in agentic iterative KG expansion papers (e.g. arXiv 2502.13025) to measure
how a knowledge graph evolves and what structural properties it exhibits.

Metrics (all computed on the CO_OCCURS_WITH backbone):
  - nodes, edges
  - average degree, max degree
  - Louvain modularity, number of communities
  - global transitivity (clustering coefficient)
  - degree assortativity
  - k-core: max k and size of the k-core at max k
  - sampled average shortest path length and diameter
  - betweenness centrality: top-10 bridge nodes
  - articulation points count
  - typed-IE edge statistics (counts per relation type)

The snapshot is saved to ``output/graph_snapshot.json``.

Usage
-----
    python -m analysis.graph_evaluation                # full graph
    python -m analysis.graph_evaluation --year 2020    # temporal slice
    python -m analysis.graph_evaluation --sample 5000  # limit to top-5000 nodes by degree
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    build_networkx_graph,
    fetch_co_occurrence_edges,
    get_database,
    get_driver,
)

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = OUTPUT_DIR / "graph_snapshot.json"

# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _modularity_and_communities(G: nx.Graph) -> tuple[float, int]:
    """Louvain modularity and community count (falls back to greedy if leidenalg absent)."""
    try:
        import community as cm  # python-louvain
        partition = cm.best_partition(G)
        Q = cm.modularity(partition, G)
        n_communities = len(set(partition.values()))
        return round(Q, 4), n_communities
    except ImportError:
        pass
    # fallback: networkx greedy modularity
    comms = nx.community.greedy_modularity_communities(G)
    Q = nx.community.modularity(G, comms)
    return round(Q, 4), len(comms)


def _kcore(G: nx.Graph) -> tuple[int, int]:
    """Return (max_k, size_of_max_kcore)."""
    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values()) if core_numbers else 0
    size = sum(1 for v in core_numbers.values() if v == max_k)
    return max_k, size


def _sampled_path_stats(G: nx.Graph, n_samples: int = 500) -> tuple[float | None, int | None]:
    """Approximate average shortest path and diameter by sampling pairs.

    Returns (avg_path, diameter) or (None, None) if graph is disconnected /
    too small.  Pairs are sampled directly to avoid O(n²) memory overhead.
    """
    lcc_nodes = max(nx.connected_components(G), key=len)
    lcc = G.subgraph(lcc_nodes)
    if len(lcc) < 3:
        return None, None

    nodes = list(lcc.nodes())
    n = len(nodes)
    # Sample without materialising the full O(n²) pair list.
    max_pairs = n * (n - 1) // 2
    actual_k = min(n_samples, max_pairs)
    sampled: set[tuple] = set()
    attempts = 0
    max_attempts = actual_k * 20
    while len(sampled) < actual_k and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        sampled.add((min(u, v), max(u, v)))
        attempts += 1

    lengths = []
    diam = 0
    for u, v in sampled:
        try:
            sp = nx.shortest_path_length(lcc, u, v)
            lengths.append(sp)
            if sp > diam:
                diam = sp
        except nx.NetworkXNoPath:
            pass

    if not lengths:
        return None, None
    return round(sum(lengths) / len(lengths), 3), diam


def _betweenness_top(G: nx.Graph, k: int = 10, n_samples: int = 500) -> list[dict]:
    """Return the top-k nodes by approximate betweenness centrality."""
    bc = nx.betweenness_centrality(G, k=min(n_samples, len(G)), normalized=True,
                                   weight="weight")
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"node_idx": int(idx), "betweenness": round(v, 6)} for idx, v in top]


def _articulation_points(G: nx.Graph) -> int:
    return sum(1 for _ in nx.articulation_points(G))


def _typed_edge_counts(driver, db: str) -> dict[str, int]:
    """Count edges per typed-IE relation type."""
    rel_types = ["USES", "ACHIEVES", "CONTAINS", "REQUIRES",
                 "PRODUCES", "IMPROVES", "COMPARES_TO", "IS_TYPE_OF"]
    counts: dict[str, int] = {}
    with driver.session(database=db) as s:
        for rt in rel_types:
            r = s.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS n").single()
            counts[rt] = r["n"] if r else 0
    return counts


# ---------------------------------------------------------------------------
# Main snapshot function
# ---------------------------------------------------------------------------

def compute_snapshot(
    driver=None,
    db: str | None = None,
    year: int | None = None,
    sample: int | None = None,
) -> dict:
    """Compute all metrics and return a structured snapshot dict."""
    t0 = time.time()
    if driver is None:
        driver = get_driver()
    if db is None:
        db = get_database()

    print("  Fetching graph from Neo4j ...")
    nodes, idx, edges = fetch_co_occurrence_edges(driver, database=db, year=year)
    G = build_networkx_graph(nodes, edges)
    node_labels = {v: k for k, v in idx.items()}

    # Optional degree-based sampling (for speed on very large graphs)
    if sample and len(G) > sample:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:sample]
        G = G.subgraph([n for n, _ in top_nodes]).copy()
        print(f"  Sampled to {sample} highest-degree nodes.")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = round(sum(degrees) / n_nodes, 3) if n_nodes else 0.0
    max_degree = max(degrees) if degrees else 0

    print(f"  Graph: {n_nodes} nodes, {n_edges} edges -- computing metrics ...")

    mod, n_comms = _modularity_and_communities(G)
    transitivity = round(nx.transitivity(G), 4)
    assortativity = round(nx.degree_assortativity_coefficient(G), 4)
    max_k, kcore_size = _kcore(G)
    avg_path, diameter = _sampled_path_stats(G)
    ap_count = _articulation_points(G)
    top_betweenness = _betweenness_top(G, k=10)

    # Annotate betweenness entries with entity names
    for entry in top_betweenness:
        entry["entity"] = node_labels.get(entry["node_idx"], "?")

    typed = _typed_edge_counts(driver, db)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year_filter": year,
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "louvain_modularity": mod,
        "n_communities": n_comms,
        "global_transitivity": transitivity,
        "degree_assortativity": assortativity,
        "max_k_core": max_k,
        "max_k_core_size": kcore_size,
        "avg_shortest_path": avg_path,
        "diameter_estimate": diameter,
        "articulation_points": ap_count,
        "top_betweenness": top_betweenness,
        "typed_edge_counts": typed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return snapshot


def save_snapshot(snapshot: dict, path: Path = SNAPSHOT_PATH) -> Path:
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"  Saved snapshot -> {path}")
    return path


# ---------------------------------------------------------------------------
# run() for the pipeline orchestrator
# ---------------------------------------------------------------------------

def run(driver=None, year: int | None = None, **kwargs) -> dict:
    """Entry point called by run_analysis.py."""
    snap = compute_snapshot(driver=driver, year=year)
    save_snapshot(snap)

    print(f"\n  Graph Snapshot ({snap['generated_at'][:10]})")
    print(f"    Nodes              : {snap['nodes']:,}")
    print(f"    Edges              : {snap['edges']:,}")
    print(f"    Avg degree         : {snap['avg_degree']}")
    print(f"    Max degree         : {snap['max_degree']}")
    print(f"    Louvain modularity : {snap['louvain_modularity']}")
    print(f"    Communities        : {snap['n_communities']}")
    print(f"    Transitivity       : {snap['global_transitivity']}")
    print(f"    Assortativity      : {snap['degree_assortativity']}")
    print(f"    Max k-core         : {snap['max_k_core']}  (size={snap['max_k_core_size']})")
    print(f"    Avg shortest path  : {snap['avg_shortest_path']}")
    print(f"    Diameter (est.)    : {snap['diameter_estimate']}")
    print(f"    Articulation pts   : {snap['articulation_points']}")
    print(f"    Typed IE edges     : {snap['typed_edge_counts']}")
    print(f"    Elapsed            : {snap['elapsed_s']}s")

    return snap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Compute graph structural snapshot.")
    p.add_argument("--year", type=int, default=None, help="Restrict to papers from this year.")
    p.add_argument("--sample", type=int, default=None,
                   help="Limit to top-N nodes by degree (faster for huge graphs).")
    args = p.parse_args()

    d = get_driver()
    snap = compute_snapshot(driver=d, year=args.year, sample=args.sample)
    save_snapshot(snap)
    d.close()
    d.close()
