"""
Structural Hole and Bridge Concept Detection
==============================================
Identifies concepts that bridge disconnected communities using
Burt's structural-hole theory.

High betweenness + low clustering = bridge concept (structural hole filler).
High effective size = node's neighbours are not redundant (Burt, 1992).

Effective size is approximated as  degree - 2·triangles/degree,
which avoids the prohibitively expensive O(n·d²) full constraint
computation (Burt, 1992, ch. 2; Borgatti, 1997).

Outputs
-------
  - 26_structural_holes_scatter.png  — betweenness vs clustering scatter
  - 27_effective_size_distribution.png — distribution of effective size
  - 28_bridge_concepts_network.png  — network of top bridge concepts
  - structural_holes.csv             — ranked list of structural hole entities

References
----------
  Burt, R. S. (1992). Structural Holes. Harvard University Press.
  Borgatti, S. P. (1997). "Structural Holes: Unpacking Burt's Redundancy
      Measures." Connections, 20(1), 35-38.
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

MAX_NODES = 2000  # constraint computation is O(n*d²)


def _select_subgraph(nodes, edges, max_nodes=MAX_NODES):
    """Take top-N nodes by weighted degree."""
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


# ---------------------------------------------------------------------------
# 1.  Effective size (fast Burt approximation)
# ---------------------------------------------------------------------------
def compute_effective_size(G):
    """Fast approximation of Burt's effective size for each node.

    effective_size(v) = deg(v) - 2·T(v) / deg(v)

    where T(v) is the number of triangles through v.  High effective
    size means the node's neighbours are non-redundant — i.e. they
    don't know each other — which is Burt's structural hole.

    Complexity: O(m·sqrt(m)) via nx.triangles, far cheaper than the
    full constraint computation which is O(n·d²).
    """
    triangles = nx.triangles(G)
    result = {}
    for v in G.nodes():
        d = G.degree(v)
        if d < 2:
            result[v] = float(d)       # no redundancy possible
        else:
            result[v] = d - 2.0 * triangles[v] / d
    return result


# ---------------------------------------------------------------------------
# 2.  Identify bridge concepts
# ---------------------------------------------------------------------------
def find_bridge_concepts(G, nodes, top_k=50):
    """Find concepts with high betweenness and low clustering."""
    print("  Computing betweenness centrality (k=500 sample)...")
    betweenness = nx.betweenness_centrality(G, k=min(500, len(G)), weight="weight")

    print("  Computing clustering coefficients...")
    clustering = nx.clustering(G, weight="weight")

    print("  Computing effective size (fast Burt approximation)...")
    eff_size = compute_effective_size(G)

    df = pd.DataFrame([
        {
            "entity": nodes[i],
            "betweenness": betweenness.get(i, 0),
            "clustering": clustering.get(i, 0),
            "effective_size": eff_size.get(i, 0),
            "degree": G.degree(i),
            "weighted_degree": sum(d.get("weight", 1) for _, _, d in G.edges(i, data=True)),
        }
        for i in tqdm(G.nodes(), desc="  Building bridge table", unit="node", leave=False)
    ])

    # Bridge score: high betweenness * (1 - clustering) * normalised effective size
    max_es = df["effective_size"].max()
    if max_es > 0:
        df["bridge_score"] = (
            df["betweenness"] *
            (1 - df["clustering"]) *
            (df["effective_size"] / max_es)
        )
    else:
        df["bridge_score"] = df["betweenness"] * (1 - df["clustering"])

    df = df.sort_values("bridge_score", ascending=False)
    return df


# ---------------------------------------------------------------------------
# 3.  Visualisations
# ---------------------------------------------------------------------------
def plot_betweenness_vs_clustering(df, top_k=20):
    """Scatter: betweenness vs clustering with bridge concepts highlighted."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(df["clustering"], df["betweenness"],
               s=10, alpha=0.3, color="gray", label="All entities")

    top = df.head(top_k)
    ax.scatter(top["clustering"], top["betweenness"],
               s=80, alpha=0.85, color=PALETTE[3], edgecolors="black",
               linewidths=0.5, label=f"Top-{top_k} bridge concepts", zorder=5)

    for _, row in top.head(10).iterrows():
        ax.annotate(
            row["entity"], (row["clustering"], row["betweenness"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("Clustering coefficient")
    ax.set_ylabel("Betweenness centrality")
    ax.set_title("Structural Holes: High Betweenness + Low Clustering = Bridge Concepts")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "26_structural_holes_scatter")
    plt.close(fig)


def plot_effective_size_distribution(df):
    """Distribution of effective size values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = df["effective_size"].dropna()
    ax.hist(vals, bins=50, color=PALETTE[1], edgecolor="white", alpha=0.8)
    ax.axvline(vals.median(), color="red", linestyle="--", label=f"Median: {vals.median():.1f}")
    ax.set_xlabel("Effective Size (Burt approximation)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Effective Size\n(higher = more non-redundant contacts = structural hole)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "27_effective_size_distribution")
    plt.close(fig)


def plot_bridge_network(df, G, nodes, top_k=30):
    """Visualise the interconnection pattern of top bridge concepts."""
    top_entities = set(df.head(top_k)["entity"])
    node_name_to_idx = {n: i for i, n in enumerate(nodes)}

    subG = nx.Graph()
    for ent in top_entities:
        if ent in node_name_to_idx:
            subG.add_node(ent)

    for ent in top_entities:
        idx = node_name_to_idx.get(ent)
        if idx is None:
            continue
        for neighbor in G.neighbors(idx):
            neighbor_name = nodes[neighbor]
            if neighbor_name in top_entities:
                w = G[idx][neighbor].get("weight", 1)
                subG.add_edge(ent, neighbor_name, weight=w)

    if subG.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(subG, k=2.0, iterations=80, seed=42)

    bridge_scores = dict(zip(df["entity"], df["bridge_score"]))
    node_colors = [bridge_scores.get(n, 0) for n in subG.nodes()]
    node_sizes = [300 + 500 * bridge_scores.get(n, 0) / max(bridge_scores.values())
                  for n in subG.nodes()]

    edges_data = subG.edges(data=True)
    weights = [d.get("weight", 1) for _, _, d in edges_data]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 3.0 * w / max_w for w in weights]

    nx.draw_networkx_edges(subG, pos, ax=ax, width=edge_widths,
                           edge_color="gray", alpha=0.4)
    sc = nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, cmap="RdYlGn_r",
                                alpha=0.85, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Bridge score")
    nx.draw_networkx_labels(subG, pos, ax=ax, font_size=7)

    ax.set_title(f"Top-{top_k} Bridge Concepts (Structural Hole Spanners)")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "28_bridge_concepts_network")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Execute structural hole analysis pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    year
        Optional publication year filter for the co-occurrence graph.
    """
    print("\n" + "=" * 60)
    print("  STRUCTURAL HOLE DETECTION")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        sub_nodes, sub_edges, _ = _select_subgraph(nodes, edges)
        print(f"  Subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        G = build_networkx_graph(sub_nodes, sub_edges)

        df = find_bridge_concepts(G, sub_nodes)

        # Report top bridges
        print(f"\n  Top-15 bridge concepts (structural hole spanners):")
        print(f"  {'Entity':<30} {'Betw':>8} {'Clust':>8} {'EffSz':>8} {'Bridge':>8}")
        for _, row in df.head(15).iterrows():
            print(f"  {row['entity']:<30} {row['betweenness']:8.4f} "
                  f"{row['clustering']:8.4f} {row['effective_size']:8.1f} "
                  f"{row['bridge_score']:8.6f}")

        # Save CSV
        csv_path = OUTPUT_DIR / "structural_holes.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({len(df)} entities)")

        # Visualisations
        print("\n--- Visualisations ---")
        plot_betweenness_vs_clustering(df)
        plot_effective_size_distribution(df)
        plot_bridge_network(df, G, sub_nodes)

        return {"bridge_concepts": df, "graph": G}

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
