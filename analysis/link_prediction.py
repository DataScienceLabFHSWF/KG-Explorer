"""
Link Prediction — Missing Co-Occurrence Detection
===================================================
Scores non-edges to find entity pairs that *should* co-occur
based on their neighbourhood structure but currently don't.

Methods
-------
  - Common Neighbours:  |Γ(u) ∩ Γ(v)|
  - Jaccard Coefficient: |Γ(u) ∩ Γ(v)| / |Γ(u) ∪ Γ(v)|
  - Adamic-Adar Index:   Σ_{w ∈ Γ(u)∩Γ(v)} 1/log|Γ(w)|
  - Resource Allocation:  Σ_{w ∈ Γ(u)∩Γ(v)} 1/|Γ(w)|

Sampling
--------
  Subsamples to 3,000 nodes by weighted degree.  Scores all non-edges
  reachable via 2-hop neighbourhood scan: O(n·d²) where d = avg degree.

Computational Complexity
-----------------------
  - 2-hop neighbour scan: O(n·d²) ≈ O(n·d_avg²)
  - Scoring per candidate pair: O(d) for set intersection
  - Total: O(n·d²) ≈ 3.27M candidates at n=3000

Outputs
-------
  - 29_link_prediction_scores.png     — top predicted links
  - 30_predicted_links_network.png    — network of top predictions
  - predicted_links.csv               — ranked list of missing edges

References
----------
  Liben-Nowell, D. & Kleinberg, J. (2007). JASIST, 58(7), 1019-1031.
  Adamic, L. A. & Adar, E. (2003). Social Networks, 25(3), 211-230.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from tqdm import tqdm

from analysis.neo4j_utils import (
    fetch_co_occurrence_edges,
    build_networkx_graph,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("Set2", 8)

MAX_NODES = 3000  # link prediction is O(n²) on non-edges


def _select_subgraph(nodes, edges, max_nodes=MAX_NODES):
    """Take top-N nodes by weighted degree."""
    n = len(nodes)
    if n <= max_nodes:
        return nodes, edges

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

    return new_nodes, new_edges


# ---------------------------------------------------------------------------
# 1.  Score non-edges using multiple methods
# ---------------------------------------------------------------------------
def predict_links(G, nodes, top_k=200):
    """Score non-edges with multiple link prediction heuristics.

    Only considers pairs where both nodes have degree ≥ 2 and
    share at least one common neighbour (for efficiency).
    """
    n = len(G)
    print(f"  Graph has {n} nodes, {G.number_of_edges()} edges")

    # Pre-compute neighbor sets
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    degrees = dict(G.degree())

    # Only score pairs with ≥1 common neighbour (avoids O(n²) over all non-edges)
    # We iterate high-degree nodes and check their 2-hop non-neighbours
    print("  Scoring non-edges via 2-hop neighbourhood scan...")
    candidates = {}

    # Focus on nodes with degree ≥ 3 for meaningful predictions
    active_nodes = [v for v in G.nodes() if degrees[v] >= 3]
    print(f"  Active nodes (degree ≥ 3): {len(active_nodes)}")

    for v in tqdm(active_nodes, desc="  Link prediction", unit="node", leave=False):
        two_hop = set()
        for nb in neighbors[v]:
            two_hop.update(neighbors[nb])
        two_hop -= neighbors[v]
        two_hop.discard(v)

        for u in two_hop:
            if u <= v:
                continue  # avoid duplicates
            if degrees[u] < 3:
                continue

            pair = (v, u)
            if pair in candidates:
                continue

            cn = neighbors[v] & neighbors[u]
            if not cn:
                continue

            cn_score = len(cn)
            union = neighbors[v] | neighbors[u]
            jaccard = cn_score / len(union) if union else 0

            aa_score = sum(
                1.0 / np.log(max(degrees[w], 2))
                for w in cn
            )
            ra_score = sum(
                1.0 / max(degrees[w], 1)
                for w in cn
            )

            candidates[pair] = {
                "common_neighbours": cn_score,
                "jaccard": jaccard,
                "adamic_adar": aa_score,
                "resource_allocation": ra_score,
            }

    print(f"  Candidate non-edges scored: {len(candidates)}")

    # Rank by Adamic-Adar (generally best predictor)
    ranked = sorted(candidates.items(), key=lambda x: -x[1]["adamic_adar"])[:top_k]

    rows = []
    for (v, u), scores in ranked:
        rows.append({
            "entity_a": nodes[v],
            "entity_b": nodes[u],
            **scores,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2.  Visualisations
# ---------------------------------------------------------------------------
def plot_prediction_scores(df, top_k=50):
    """Horizontal bar chart of top predicted missing links."""
    show = df.head(top_k).copy()
    show["pair"] = show["entity_a"] + " — " + show["entity_b"]
    show = show.iloc[::-1]  # reverse for horizontal bar

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_k * 0.3)))

    # Adamic-Adar
    ax = axes[0]
    ax.barh(show["pair"], show["adamic_adar"], color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("Adamic-Adar Score")
    ax.set_title("Top Predicted Missing Links\n(Adamic-Adar Index)")
    ax.tick_params(axis="y", labelsize=7)

    # Common Neighbours
    ax = axes[1]
    ax.barh(show["pair"], show["common_neighbours"], color=PALETTE[2], edgecolor="white")
    ax.set_xlabel("Common Neighbours")
    ax.set_title("Top Predicted Missing Links\n(Common Neighbours)")
    ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Link Prediction: Entity Pairs That Should Co-Occur", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, "29_link_prediction_scores")
    plt.close(fig)


def plot_predicted_links_network(df, G, nodes, top_k=40):
    """Network showing existing edges + predicted missing links."""
    top = df.head(top_k)
    entity_set = set(top["entity_a"]) | set(top["entity_b"])

    node_idx = {n: i for i, n in enumerate(nodes)}

    subG = nx.Graph()
    for ent in entity_set:
        subG.add_node(ent)

    # Add existing edges between these entities
    for ent_a in entity_set:
        ia = node_idx.get(ent_a)
        if ia is None:
            continue
        for nb in G.neighbors(ia):
            nb_name = nodes[nb]
            if nb_name in entity_set:
                w = G[ia][nb].get("weight", 1)
                subG.add_edge(ent_a, nb_name, weight=w, predicted=False)

    # Add predicted edges
    predicted_edges = []
    for _, row in top.iterrows():
        if not subG.has_edge(row["entity_a"], row["entity_b"]):
            subG.add_edge(row["entity_a"], row["entity_b"],
                          weight=row["adamic_adar"], predicted=True)
            predicted_edges.append((row["entity_a"], row["entity_b"]))

    if subG.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(subG, k=2.5, iterations=80, seed=42)

    # Draw existing edges
    existing = [(u, v) for u, v, d in subG.edges(data=True) if not d.get("predicted")]
    nx.draw_networkx_edges(subG, pos, edgelist=existing, ax=ax,
                           edge_color="gray", alpha=0.3, width=0.8)

    # Draw predicted edges (highlighted)
    nx.draw_networkx_edges(subG, pos, edgelist=predicted_edges, ax=ax,
                           edge_color="red", alpha=0.7, width=2.0,
                           style="dashed", label="Predicted missing links")

    nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=300,
                           node_color=PALETTE[0], alpha=0.8,
                           edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(subG, pos, ax=ax, font_size=7)

    ax.set_title(f"Predicted Missing Co-Occurrences (top-{top_k})\n"
                 "Red dashed = predicted link; gray = existing co-occurrence")
    ax.legend(loc="upper left")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "30_predicted_links_network")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Execute link prediction pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    year
        Optional year filter for the underlying graph.
    """
    print("\n" + "=" * 60)
    print("  LINK PREDICTION (MISSING CO-OCCURRENCES)")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        sub_nodes, sub_edges = _select_subgraph(nodes, edges)
        print(f"  Subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        G = build_networkx_graph(sub_nodes, sub_edges)

        df = predict_links(G, sub_nodes, top_k=200)

        # Report top predictions
        print(f"\n  Top-15 predicted missing co-occurrences:")
        print(f"  {'Entity A':<25} {'Entity B':<25} {'AA':>8} {'CN':>4} {'Jaccard':>8}")
        for _, row in df.head(15).iterrows():
            print(f"  {row['entity_a']:<25} {row['entity_b']:<25} "
                  f"{row['adamic_adar']:8.3f} {row['common_neighbours']:4.0f} "
                  f"{row['jaccard']:8.4f}")

        # Save CSV
        csv_path = OUTPUT_DIR / "predicted_links.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({len(df)} predicted links)")

        # Visualisations
        print("\n--- Visualisations ---")
        plot_prediction_scores(df)
        plot_predicted_links_network(df, G, sub_nodes)

        return {"predicted_links": df}

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
