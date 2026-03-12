"""
Void / Knowledge-Gap Extraction from Persistent Homology
==========================================================
Maps H1 (loops) and H2 (voids) features back to concrete entity names.

A void in H2 at threshold ε between entities {A, B, C, …} means:
  - All pairwise co-occurrences exist at weight ≥ 1/ε
  - But the full simplex never appears — they were never studied together.
  → Potential research opportunity.

Sampling
--------
  Uses the same top-800 subgraph as TDA.  Cocycle representatives
  from ripser identify the simplices (and thus entities) bounding
  each persistent feature.

Computational Complexity
-----------------------
  Same as TDA: O(n³) typical for VR persistence via ripser.
  Cocycle extraction is O(|cocycle|) per feature.

Outputs
-------
  - 23_void_entity_network.png   — subgraph of entities forming persistent voids
  - 24_loop_entity_network.png   — subgraph of entities forming persistent loops
  - 25_gap_persistence_vs_degree.png — persistence vs hub-ness
  - knowledge_gaps.json          — structured gap report for downstream use

References
----------
  Carlsson, G. (2009). Bull. AMS, 46(2), 255-308.
  Edelsbrunner, H. & Harer, J. (2010). Computational Topology. AMS.
  Bauer, U. (2021). J. Appl. Comput. Topology, 5, 391-423.
"""

import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from analysis.neo4j_utils import (
    fetch_co_occurrence_edges,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("Set2", 8)


def _select_subgraph(nodes, edges, max_nodes=800):
    """Select densest subgraph by weighted degree (same as TDA)."""
    n = len(nodes)
    if n <= max_nodes:
        return nodes, edges, {i: i for i in range(n)}

    deg = np.zeros(n)
    for si, ti, w in edges:
        deg[si] += w
        deg[ti] += w

    top_indices = set(np.argsort(deg)[-max_nodes:])
    old_to_new = {}
    new_nodes = []
    for old_idx in sorted(top_indices):
        old_to_new[old_idx] = len(new_nodes)
        new_nodes.append(nodes[old_idx])

    new_edges = []
    for si, ti, w in edges:
        if si in old_to_new and ti in old_to_new:
            new_edges.append((old_to_new[si], old_to_new[ti], w))

    return new_nodes, new_edges, old_to_new


def _build_distance_matrix(edges, n):
    """Convert co-occurrence weights to distances: d = 1/w."""
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    for si, ti, w in edges:
        d = 1.0 / max(w, 1e-10)
        dist[si][ti] = d
        dist[ti][si] = d
    return dist


# ---------------------------------------------------------------------------
# 1.  Compute persistence with cocycle tracking
# ---------------------------------------------------------------------------
def compute_persistence_with_cocycles(dist_matrix, max_dim=2):
    """Compute persistent homology with representative cocycles."""
    from ripser import ripser

    result = ripser(
        dist_matrix, maxdim=max_dim,
        distance_matrix=True, do_cocycles=True,
    )
    return result["dgms"], result["cocycles"]


# ---------------------------------------------------------------------------
# 2.  Map topological features to entity names
# ---------------------------------------------------------------------------
def extract_feature_entities(dgms, cocycles, nodes, dim, persistence_threshold=0.05, top_k=30):
    """Extract entities participating in persistent H_dim features.

    For each persistent feature, the cocycle gives the edges (dim=1) or
    triangles (dim=2) involved.  We collect the vertex indices from these
    simplices and map them to entity names.

    Returns list of dicts:
      {entities, birth, death, persistence, cocycle_size}
    """
    if dim >= len(dgms) or len(dgms[dim]) == 0:
        return []

    dgm = dgms[dim]
    coc = cocycles[dim] if dim < len(cocycles) else []

    finite_mask = np.isfinite(dgm[:, 1])
    features = []

    for i, (birth, death) in enumerate(dgm):
        if not np.isfinite(death):
            persistence = float("inf")
        else:
            persistence = death - birth

        if persistence < persistence_threshold:
            continue

        # Get vertex indices from the cocycle
        vertex_set = set()
        if i < len(coc) and len(coc[i]) > 0:
            cocycle_data = np.array(coc[i])
            # Cocycle rows are [v0, v1, ..., v_dim, coefficient]
            # For dim=1: [v0, v1, coeff] → edges
            # For dim=2: [v0, v1, v2, coeff] → triangles
            simplex_cols = dim + 1  # number of vertex columns
            if cocycle_data.ndim == 2 and cocycle_data.shape[1] > simplex_cols:
                for row in cocycle_data:
                    for vi in range(simplex_cols):
                        vertex_set.add(int(row[vi]))
            elif cocycle_data.ndim == 1 and len(cocycle_data) > simplex_cols:
                for vi in range(simplex_cols):
                    vertex_set.add(int(cocycle_data[vi]))

        entity_names = [nodes[v] for v in sorted(vertex_set) if v < len(nodes)]

        features.append({
            "entities": entity_names,
            "birth": float(birth),
            "death": float(death) if np.isfinite(death) else None,
            "persistence": float(persistence) if np.isfinite(persistence) else None,
            "cocycle_size": len(vertex_set),
        })

    # Sort by persistence (longest first)
    features.sort(key=lambda f: -(f["persistence"] or float("inf")))
    return features[:top_k]


# ---------------------------------------------------------------------------
# 3.  Visualise void entities as a network
# ---------------------------------------------------------------------------
def plot_feature_network(features, all_edges, all_nodes, dim, filename, title):
    """Draw a subgraph of entities involved in persistent features."""
    if not features:
        print(f"  No H{dim} features to visualise.")
        return

    # Collect all entities from features
    entity_set = set()
    for f in features:
        entity_set.update(f["entities"])

    if not entity_set:
        print(f"  No entity mappings available for H{dim} features.")
        return

    # Build a subgraph of just these entities
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    sub_indices = {node_idx[e] for e in entity_set if e in node_idx}

    G = nx.Graph()
    for e in entity_set:
        G.add_node(e)

    idx_to_name = {i: n for n, i in node_idx.items()}
    for si, ti, w in all_edges:
        if si in sub_indices and ti in sub_indices:
            G.add_edge(idx_to_name[si], idx_to_name[ti], weight=w)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Node colour by feature count
    feature_counts = Counter()
    for f in features:
        for e in f["entities"]:
            feature_counts[e] += 1

    node_colors = [feature_counts.get(n, 0) for n in G.nodes()]
    node_sizes = [300 + 150 * feature_counts.get(n, 0) for n in G.nodes()]

    pos = nx.spring_layout(G, k=2.0 / max(1, len(G) ** 0.5), iterations=80, seed=42)

    edges_data = G.edges(data=True)
    weights = [d.get("weight", 1) for _, _, d in edges_data]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 2.0 * w / max_w for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color="gray", alpha=0.4)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, cmap="YlOrRd",
                                alpha=0.85, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label=f"# H{dim} features involving this entity")

    # Label top entities
    top_entities = sorted(feature_counts, key=feature_counts.get, reverse=True)[:25]
    labels = {n: n for n in top_entities}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  Persistence vs degree scatter
# ---------------------------------------------------------------------------
def plot_persistence_vs_degree(features, all_edges, all_nodes, dim=2):
    """Scatter: mean weighted degree of void entities vs persistence."""
    if not features or not any(f["entities"] for f in features):
        print("  No features with entity mappings for scatter plot.")
        return

    # Compute weighted degrees
    n = len(all_nodes)
    wdeg = np.zeros(n)
    for si, ti, w in all_edges:
        wdeg[si] += w
        wdeg[ti] += w
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    xs, ys, sizes = [], [], []
    for f in features:
        if not f["entities"] or f["persistence"] is None:
            continue
        degrees = [wdeg[node_idx[e]] for e in f["entities"] if e in node_idx]
        if not degrees:
            continue
        xs.append(np.mean(degrees))
        ys.append(f["persistence"])
        sizes.append(len(f["entities"]))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(xs, ys, s=[40 + 20 * s for s in sizes],
                         c=sizes, cmap="viridis", alpha=0.7, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="# entities in feature")
    ax.set_xlabel("Mean weighted degree of participating entities")
    ax.set_ylabel(f"H{dim} persistence")
    ax.set_title(f"Knowledge Gaps: Persistence vs Entity Hub-ness (H{dim})")
    fig.tight_layout()
    save_figure(fig, "25_gap_persistence_vs_degree")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, max_nodes=800, year: int | None = None):
    """Execute void/gap extraction pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    max_nodes
        Subgraph size limit for cocycle computation.
    year
        Optional year filter for the graph.
    """
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GAP EXTRACTION (VOID ANALYSIS)")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        sub_nodes, sub_edges, old_to_new = _select_subgraph(nodes, edges, max_nodes)
        print(f"  Subgraph for TDA: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        # Distance matrix
        print("  Building distance matrix...")
        dist = _build_distance_matrix(sub_edges, len(sub_nodes))

        # Persistent homology with cocycles
        print("  Computing persistent homology with cocycle tracking...")
        dgms, cocycles = compute_persistence_with_cocycles(dist, max_dim=2)

        for dim, dgm in enumerate(dgms):
            finite = np.sum(np.isfinite(dgm[:, 1]))
            infinite = len(dgm) - finite
            print(f"    H{dim}: {len(dgm)} features ({finite} finite, {infinite} infinite)")

        # Extract entity mappings for H1 (loops) and H2 (voids)
        print("\n--- H1 Knowledge Loops ---")
        h1_features = extract_feature_entities(
            dgms, cocycles, sub_nodes, dim=1, persistence_threshold=0.05, top_k=30,
        )
        print(f"  Persistent H1 features mapped: {len(h1_features)}")
        for f in h1_features[:5]:
            ents = ", ".join(f["entities"][:6])
            if len(f["entities"]) > 6:
                ents += f" … (+{len(f['entities']) - 6} more)"
            print(f"    pers={f['persistence']:.4f}  [{f['cocycle_size']} entities]: {ents}")

        print("\n--- H2 Knowledge Voids (Research Gaps) ---")
        h2_features = extract_feature_entities(
            dgms, cocycles, sub_nodes, dim=2, persistence_threshold=0.05, top_k=30,
        )
        print(f"  Persistent H2 features mapped: {len(h2_features)}")
        for f in h2_features[:5]:
            ents = ", ".join(f["entities"][:6])
            if len(f["entities"]) > 6:
                ents += f" … (+{len(f['entities']) - 6} more)"
            p_str = f"{f['persistence']:.4f}" if f["persistence"] else "∞"
            print(f"    pers={p_str}  [{f['cocycle_size']} entities]: {ents}")

        # Visualisations
        print("\n--- Visualisations ---")
        if h2_features:
            plot_feature_network(
                h2_features, sub_edges, sub_nodes, dim=2,
                filename="23_void_entity_network",
                title="H₂ Void Entities — Knowledge Gaps\n(pairwise co-occurring but never jointly studied)",
            )
        if h1_features:
            plot_feature_network(
                h1_features, sub_edges, sub_nodes, dim=1,
                filename="24_loop_entity_network",
                title="H₁ Loop Entities — Redundant Knowledge Circuits",
            )

        plot_persistence_vs_degree(h2_features, sub_edges, sub_nodes, dim=2)

        # Save structured report
        report = {
            "subgraph_nodes": len(sub_nodes),
            "subgraph_edges": len(sub_edges),
            "h1_loops": [
                {**f, "type": "loop"} for f in h1_features
            ],
            "h2_voids": [
                {**f, "type": "void"} for f in h2_features
            ],
        }
        with open(OUTPUT_DIR / "knowledge_gaps.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved: {OUTPUT_DIR / 'knowledge_gaps.json'}")

        return report

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
