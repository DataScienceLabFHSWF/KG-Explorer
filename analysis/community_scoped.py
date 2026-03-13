"""
Community-Scoped Analysis of the Fusion KG
===========================================
Runs computationally expensive methods (TDA, spectral, FCA) on individual
Louvain community subgraphs rather than the full 50K-node graph.

Strategy
--------
  1. Detect communities with Louvain on the full co-occurrence graph.
  2. For each community (above a minimum size threshold):
       a. Extract the induced subgraph.
       b. Run TDA persistent homology  → per-community H1/H2 gaps.
       c. Run spectral analysis         → per-community Fiedler value.
       d. Run FCA implication mining    → per-community category rules.
  3. Aggregate results into a unified report.

Why do it this way?
-------------------
  - TDA on the full graph: O(n³) in distance matrix → infeasible at 50K nodes.
  - Per-community TDA:     many small O(k³) problems, k ~ 50–500 nodes.
  - Results are more interpretable: a void inside "plasma diagnostics" has
    clear domain meaning vs a void in the global graph.

Outputs
-------
  - 40_community_fiedler_values.png  — Fiedler value per community (bar chart)
  - 41_community_tda_gaps.png        — Number of H1/H2 gaps per community
  - 42_community_gap_scatter.png     — persistence vs community size scatter
  - community_tda_gaps.json          — All detected gaps with persistence/community
  - community_spectral.csv           — Per-community spectral metrics
  - community_fca_implications.json  — Per-community FCA implication rules

References
----------
  Blondel, V. D. et al. (2008). J. Stat. Mech., P10008.
  Bauer, U. (2021). J. Appl. Comput. Topology, 5, 391–423.
  Chung, F. R. K. (1997). Spectral Graph Theory. AMS.
  Ganter, B. & Wille, R. (1999). Formal Concept Analysis. Springer.
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    fetch_co_occurrence_edges,
    fetch_entity_categories,
    get_driver,
    save_figure,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("viridis", 10)

# Minimum community size for expensive analyses
MIN_COMMUNITY_SIZE_TDA = 15
MIN_COMMUNITY_SIZE_SPECTRAL = 10
MIN_COMMUNITY_SIZE_FCA = 8
# Hard cap: trim to top-N nodes inside a community before TDA
TDA_SUBGRAPH_CAP = 300


# ---------------------------------------------------------------------------
# Helper: detect communities on the full graph
# ---------------------------------------------------------------------------
def _detect_communities(G: nx.Graph) -> dict[int, list]:
    """
    Run Louvain community detection.  Returns {community_id: [node, ...]}.
    Falls back to connected-components if python-louvain is not installed.
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight")
    except ImportError:
        print("  [WARN] python-louvain not installed – falling back to "
              "connected-components as community proxy.")
        partition = {}
        for cid, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = cid

    communities: dict[int, list] = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    return dict(communities), partition


# ---------------------------------------------------------------------------
# Per-community: TDA (persistent homology)
# ---------------------------------------------------------------------------
def _tda_community(subgraph: nx.Graph,
                   nodes: list,
                   max_nodes: int = TDA_SUBGRAPH_CAP) -> dict:
    """
    Run ripser persistent homology on a community subgraph.

    Large communities are trimmed to ``max_nodes`` highest-weighted-degree
    entities before computing homology — preserving the densest core.
    Returns a dict with persistence diagrams and structured gap list.
    """
    try:
        import ripser
    except ImportError:
        return {"error": "ripser not installed", "gaps": []}

    # Trim to densest core if needed
    if len(nodes) > max_nodes:
        degrees = dict(subgraph.degree(weight="weight"))
        nodes = sorted(nodes, key=lambda n: degrees.get(n, 0),
                       reverse=True)[:max_nodes]
        subgraph = subgraph.subgraph(nodes).copy()
        nodes = list(subgraph.nodes())

    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Distance matrix: d(u,v) = 1/w; infinity for missing edges
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for u, v, data in subgraph.edges(data=True):
        w = float(data.get("weight", 1.0))
        d = 1.0 / max(w, 1e-10)
        i, j = node_idx[u], node_idx[v]
        D[i, j] = d
        D[j, i] = d

    finite_vals = D[D < np.inf]
    if len(finite_vals) == 0:
        return {"error": "no edges in subgraph", "gaps": []}

    finite_max = float(finite_vals.max())
    D[D == np.inf] = finite_max * 2.0

    try:
        diagrams = ripser.ripser(D, maxdim=2,
                                 distance_matrix=True)["dgms"]
    except Exception as exc:
        return {"error": str(exc), "gaps": []}

    threshold = 0.05 * finite_max
    gaps = []
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            pers = (float(death) - float(birth)) if np.isfinite(death) else None
            if pers is None or pers > threshold:
                gaps.append({
                    "dim": dim,
                    "birth": float(birth),
                    "death": float(death) if np.isfinite(death) else None,
                    "persistence": pers,
                })

    return {
        "n_nodes": n,
        "gaps": gaps,
        "h1_count": sum(1 for g in gaps if g["dim"] == 1),
        "h2_count": sum(1 for g in gaps if g["dim"] == 2),
        "finite_max_dist": finite_max,
    }


# ---------------------------------------------------------------------------
# Per-community: Spectral (Fiedler value)
# ---------------------------------------------------------------------------
def _spectral_community(subgraph: nx.Graph, nodes: list) -> dict:
    """
    Compute the Fiedler value (λ₂) and spectral gap for a community subgraph.

    A small Fiedler value indicates the community itself is internally
    near-disconnected — a candidate for further splitting.
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse import csgraph
        from scipy.sparse.linalg import eigsh
    except ImportError:
        return {"error": "scipy not installed"}

    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    rows, cols, data = [], [], []
    for u, v, d in subgraph.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = float(d.get("weight", 1.0))
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    if not rows:
        return {"error": "no edges", "fiedler_value": 0.0}

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    L = csgraph.laplacian(A, normed=True)

    k = min(6, n - 1)
    if k < 2:
        return {"fiedler_value": 0.0, "spectral_gap": 0.0, "n_nodes": n}

    try:
        vals, vecs = eigsh(L, k=k, which="SM", tol=1e-5, maxiter=5000)
    except Exception as exc:
        return {"error": str(exc), "n_nodes": n}

    vals = np.sort(np.abs(vals))
    fiedler = float(vals[1]) if len(vals) > 1 else 0.0
    gap = float(vals[2] - vals[1]) if len(vals) > 2 else 0.0

    return {
        "n_nodes": n,
        "fiedler_value": fiedler,
        "spectral_gap": gap,
        "eigenvalues": vals.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-community: FCA (attribute implications)
# ---------------------------------------------------------------------------
def _fca_community(subgraph: nx.Graph,
                   entity_categories: dict,
                   all_categories: list) -> dict:
    """
    Build a formal context restricted to subgraph entities and compute
    the Duquenne-Guigues implication basis.
    """
    try:
        from concepts import Context
    except ImportError:
        return {"error": "concepts not installed", "implications": []}

    members = [n for n in subgraph.nodes() if n in entity_categories]
    if len(members) < 3:
        return {"implications": [], "n_concepts": 0, "n_entities": len(members)}

    booleans = [
        [cat in entity_categories[ent] for cat in all_categories]
        for ent in members
    ]
    # Skip columns (categories) that are all-False or all-True — they carry
    # no information and slow down lattice computation on large contexts.
    col_sums = np.array(booleans).sum(axis=0)
    active_cols = [i for i, s in enumerate(col_sums) if 0 < s < len(members)]
    active_cats = [all_categories[i] for i in active_cols]
    filtered = [[row[i] for i in active_cols] for row in booleans]

    try:
        ctx = Context(members, active_cats, filtered)
        lattice = ctx.lattice
        implications = [
            {"premise": list(impl.premise), "conclusion": list(impl.conclusion)}
            for impl in lattice.implications()
        ]
        n_concepts = len(lattice)
    except Exception as exc:
        return {"error": str(exc), "implications": [], "n_entities": len(members)}

    return {
        "n_entities": len(members),
        "n_concepts": n_concepts,
        "n_implications": len(implications),
        "implications": implications,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _plot_fiedler_values(spectral_results: dict):
    """Bar chart of Fiedler value per community, sorted descending."""
    records = [
        (cid, r["fiedler_value"], r["n_nodes"])
        for cid, r in spectral_results.items()
        if "fiedler_value" in r and r["n_nodes"] >= MIN_COMMUNITY_SIZE_SPECTRAL
    ]
    if not records:
        return

    records.sort(key=lambda x: x[1])  # sort by Fiedler ascending
    cids = [f"C{r[0]}\n(n={r[2]})" for r in records[:30]]  # top 30 for legibility
    vals = [r[1] for r in records[:30]]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#C62828" if v < 0.05 else "#1565C0" for v in vals]
    ax.barh(cids, vals, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0.05, color="red", linestyle="--", lw=1.5,
               label="Fragmentation threshold (λ₂ < 0.05)")
    ax.set_xlabel("Fiedler Value λ₂ (algebraic connectivity)")
    ax.set_title("Per-Community Fiedler Values\n"
                 "(Red = community may need further splitting)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "40_community_fiedler_values")
    plt.close(fig)


def _plot_tda_gap_counts(tda_results: dict):
    """Bar chart of H1/H2 gap counts per community."""
    records = [
        (cid, r.get("h1_count", 0), r.get("h2_count", 0), r.get("n_nodes", 0))
        for cid, r in tda_results.items()
        if "error" not in r
    ]
    if not records:
        return

    records.sort(key=lambda x: x[2] + x[1], reverse=True)
    records = records[:25]

    cids = [f"C{r[0]}" for r in records]
    h1 = [r[1] for r in records]
    h2 = [r[2] for r in records]
    x = np.arange(len(cids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, h1, width=w, label="H₁ loops", color="#1976D2", alpha=0.85)
    ax.bar(x + w / 2, h2, width=w, label="H₂ voids (knowledge gaps)",
           color="#E65100", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=45, ha="right")
    ax.set_ylabel("Number of persistent features")
    ax.set_title("Persistent Homology Features per Community\n"
                 "(H₂ voids = unrealised research connections)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "41_community_tda_gaps")
    plt.close(fig)


def _plot_gap_scatter(all_gaps: list):
    """Scatter plot: persistence vs community size for H2 gaps."""
    h2 = [g for g in all_gaps if g["dim"] == 2 and g["persistence"] is not None]
    if not h2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [g["community_size"] for g in h2]
    pers = [g["persistence"] for g in h2]
    ax.scatter(sizes, pers, alpha=0.6, color="#E65100", edgecolors="white",
               linewidths=0.5, s=60)
    ax.set_xlabel("Community size (nodes)")
    ax.set_ylabel("Gap persistence")
    ax.set_title("H₂ Knowledge-Gap Persistence vs Community Size\n"
                 "(Upper-right = large community, structurally significant gap)")
    fig.tight_layout()
    save_figure(fig, "42_community_gap_scatter")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run entry-point
# ---------------------------------------------------------------------------
def run(driver, year=None, max_nodes: int = TDA_SUBGRAPH_CAP):
    """
    Execute community-scoped TDA, spectral, and FCA analyses.

    Parameters
    ----------
    max_nodes : int
        Per-community node cap passed to _tda_community() before ripser.
        Defaults to TDA_SUBGRAPH_CAP (300).  Increase for more thorough
        but slower analysis.  Pass via ``--max-nodes`` on the CLI.
    """
    print("  Fetching co-occurrence graph…")
    nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
    print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")

    # Build networkx graph
    G = nx.Graph()
    for name in nodes:
        G.add_node(name)
    for si, ti, w in edges:
        G.add_edge(nodes[si], nodes[ti], weight=w)

    # Fetch entity→category mapping for FCA
    print("  Fetching entity categories…")
    entity_categories, all_categories = fetch_entity_categories(driver)

    # ── Community detection ──────────────────────────────────────────
    print("  Detecting communities (Louvain)…")
    communities, partition = _detect_communities(G)
    n_communities = len(communities)
    sizes = sorted([len(m) for m in communities.values()], reverse=True)
    print(f"  Found {n_communities} communities. "
          f"Top sizes: {sizes[:10]}")

    # Sort communities by size descending
    sorted_communities = sorted(communities.items(),
                                key=lambda x: len(x[1]), reverse=True)

    tda_results: dict = {}
    spectral_results: dict = {}
    fca_results: dict = {}

    for cid, members in tqdm(sorted_communities,
                              desc="  Analysing communities", unit="comm"):
        subgraph = G.subgraph(members).copy()
        n = subgraph.number_of_nodes()

        # TDA
        if n >= MIN_COMMUNITY_SIZE_TDA:
            tda_results[cid] = _tda_community(subgraph, list(members),
                                               max_nodes=max_nodes)

        # Spectral
        if n >= MIN_COMMUNITY_SIZE_SPECTRAL:
            spectral_results[cid] = _spectral_community(subgraph, list(members))
            spectral_results[cid]["community_id"] = cid

        # FCA
        if n >= MIN_COMMUNITY_SIZE_FCA:
            fca_results[cid] = _fca_community(subgraph, entity_categories,
                                               all_categories)

    # ── Aggregate gaps ───────────────────────────────────────────────
    all_gaps = []
    for cid, result in tda_results.items():
        if "error" in result:
            continue
        for gap in result.get("gaps", []):
            gap["community"] = cid
            gap["community_size"] = result["n_nodes"]
            all_gaps.append(gap)

    # Sort by persistence descending (None = infinite persistence → most significant)
    all_gaps.sort(
        key=lambda g: g["persistence"] if g["persistence"] is not None else 1e9,
        reverse=True,
    )

    h1_gaps = [g for g in all_gaps if g["dim"] == 1]
    h2_gaps = [g for g in all_gaps if g["dim"] == 2]
    print(f"\n  TDA summary:  {len(h1_gaps)} H₁ loops, "
          f"{len(h2_gaps)} H₂ knowledge-gap voids detected across communities")

    # ── Plots ────────────────────────────────────────────────────────
    print("  Generating community-scoped plots…")
    _plot_fiedler_values(spectral_results)
    _plot_tda_gap_counts(tda_results)
    _plot_gap_scatter(all_gaps)

    # ── Save outputs ─────────────────────────────────────────────────
    gaps_path = OUTPUT_DIR / "community_tda_gaps.json"
    with open(gaps_path, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {gaps_path}")

    # Spectral CSV
    spectral_rows = []
    for cid, r in spectral_results.items():
        if "fiedler_value" in r:
            spectral_rows.append({
                "community_id": cid,
                "n_nodes": r["n_nodes"],
                "fiedler_value": r["fiedler_value"],
                "spectral_gap": r.get("spectral_gap", None),
                "fragmented": r["fiedler_value"] < 0.05,
            })
    if spectral_rows:
        spec_path = OUTPUT_DIR / "community_spectral.csv"
        pd.DataFrame(spectral_rows).sort_values(
            "fiedler_value").to_csv(spec_path, index=False)
        print(f"  Saved: {spec_path}")

    # FCA JSON
    fca_path = OUTPUT_DIR / "community_fca_implications.json"
    with open(fca_path, "w", encoding="utf-8") as f:
        json.dump(fca_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: list(o) if isinstance(o, set) else str(o))
    print(f"  Saved: {fca_path}")

    # ── Print top fragmented communities ─────────────────────────────
    fragmented = [
        (cid, r["fiedler_value"], r["n_nodes"])
        for cid, r in spectral_results.items()
        if "fiedler_value" in r and r["fiedler_value"] < 0.05
    ]
    fragmented.sort(key=lambda x: x[1])
    if fragmented:
        print(f"\n  Internally fragmented communities (λ₂ < 0.05):")
        for cid, lam, n in fragmented[:10]:
            print(f"    Community {cid:4d}: λ₂ = {lam:.4f}, {n} nodes")

    return {
        "n_communities": n_communities,
        "tda_results": tda_results,
        "spectral_results": spectral_results,
        "fca_results": fca_results,
        "all_gaps": all_gaps,
    }


if __name__ == "__main__":
    _driver = get_driver()
    try:
        run(_driver)
    finally:
        _driver.close()
