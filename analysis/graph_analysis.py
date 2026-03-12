"""
Graph-Theoretic Analysis of the Fusion KG
==========================================
Computes and visualises:
  1. Degree distribution + power-law fit
  2. Centrality rankings (PageRank, betweenness, closeness, eigenvector)
  3. Community detection (Louvain)
  4. k-Core decomposition
  5. Clustering coefficient distribution

Computational Complexity
-----------------------
  - PageRank: O(m·i), i = power-iteration steps (full graph)
  - Betweenness: O(k·(m+n)) with k=500 random shortest-path samples
    (Brandes, 2001; exact: O(n·m), infeasible at 50K nodes)
  - Closeness: O(s·(m+n)) with s=2000 random SSSP (sampled for n>10K)
  - Louvain communities: O(m) (Blondel et al., 2008)
  - k-Core: O(m) (Batagelj & Zaversnik, 2003)

References
----------
  Barabási, A.-L. & Albert, R. (1999). Science, 286(5439), 509.
  Blondel, V. D. et al. (2008). J. Stat. Mech., P10008.
  Brandes, U. (2001). J. Math. Sociology, 25(2), 163-177.
  Batagelj, V. & Zaversnik, M. (2003). arXiv:cs/0310049.
  Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). SIAM Rev., 51(4), 661.
  Newman, M. E. J. (2010). Networks: An Introduction. OUP.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns

from analysis.neo4j_utils import (
    build_networkx_graph,
    fetch_co_occurrence_edges,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("viridis", 8)


# ---------------------------------------------------------------------------
# 1.  Degree distribution + power-law fit
# ---------------------------------------------------------------------------
def analyse_degree_distribution(G: nx.Graph, nodes: list[str]):
    """Plot degree distribution on log-log scale and fit a power law."""
    degrees = np.array([d for _, d in G.degree()])
    weighted_degrees = np.array([d for _, d in G.degree(weight="weight")])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Unweighted
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(degrees) + 1), 50)
    ax.hist(degrees, bins=bins, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Count")
    ax.set_title("(a) Unweighted Degree Distribution")

    fit = powerlaw.Fit(degrees[degrees > 0], discrete=True, verbose=False)
    ax.text(
        0.95, 0.95,
        f"$\\alpha = {fit.alpha:.2f}$\n$k_{{\\min}} = {fit.xmin}$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # Weighted
    ax = axes[1]
    bins_w = np.logspace(0, np.log10(max(weighted_degrees) + 1), 50)
    ax.hist(weighted_degrees, bins=bins_w, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Strength $s$ (weighted degree)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Weighted Degree (Strength) Distribution")

    fig.suptitle("Degree Distributions of the Fusion Co-Occurrence Graph", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "01_degree_distribution")
    plt.close(fig)

    # Power-law comparison table
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
    print("\n  Power-law fit comparison:")
    print(f"    α = {fit.alpha:.3f}, x_min = {fit.xmin}")
    print(f"    vs lognormal:   R = {R_ln:+.3f}, p = {p_ln:.4f}")
    print(f"    vs exponential: R = {R_exp:+.3f}, p = {p_exp:.4f}")

    return degrees, weighted_degrees, fit


# ---------------------------------------------------------------------------
# 2.  Centrality rankings
# ---------------------------------------------------------------------------
def analyse_centralities(G: nx.Graph, nodes: list[str]):
    """Compute four centrality measures and create ranking + comparison plots."""
    n = G.number_of_nodes()

    print("  Computing PageRank...")
    pr = nx.pagerank(G, weight="weight")

    print("  Computing betweenness centrality (may take a moment)...")
    bc = nx.betweenness_centrality(G, weight="weight", k=min(500, n))

    # Closeness is O(V*(V+E)) — approximate on large graphs
    if n > 10_000:
        import random
        sample_nodes = random.sample(list(G.nodes()), min(2000, n))
        print(f"  Computing closeness centrality (sampled on {len(sample_nodes)} nodes)...")
        cc = {}
        for node in G.nodes():
            cc[node] = 0.0
        for sn in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G, sn)
            for target, dist in lengths.items():
                if dist > 0:
                    cc[target] += 1.0 / dist
        # normalise
        max_cc = max(cc.values()) if cc else 1.0
        cc = {k: v / max_cc for k, v in cc.items()}
    else:
        print("  Computing closeness centrality...")
        cc = nx.closeness_centrality(G)

    print("  Computing eigenvector centrality...")
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        ec = {n_: 0.0 for n_ in G.nodes()}

    df = pd.DataFrame({
        "entity": [nodes[i] for i in G.nodes()],
        "pagerank": [pr[i] for i in G.nodes()],
        "betweenness": [bc[i] for i in G.nodes()],
        "closeness": [cc[i] for i in G.nodes()],
        "eigenvector": [ec[i] for i in G.nodes()],
    })
    df.to_csv(OUTPUT_DIR / "centralities.csv", index=False)

    # Top-20 bar charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    measures = [
        ("pagerank", "PageRank", PALETTE[0]),
        ("betweenness", "Betweenness Centrality", PALETTE[1]),
        ("closeness", "Closeness Centrality", PALETTE[2]),
        ("eigenvector", "Eigenvector Centrality", PALETTE[4]),
    ]
    for ax, (col, title, color) in zip(axes.flat, measures):
        top = df.nlargest(20, col)
        ax.barh(range(len(top)), top[col].values, color=color, edgecolor="white")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["entity"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Score")

    fig.suptitle("Top-20 Entities by Centrality Measure", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, "02_centrality_rankings")
    plt.close(fig)

    # Scatter: PageRank vs Betweenness (bridge concepts)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["pagerank"], df["betweenness"], s=10, alpha=0.4, c=PALETTE[0])

    # Label outliers
    for _, row in df.nlargest(10, "betweenness").iterrows():
        ax.annotate(
            row["entity"], (row["pagerank"], row["betweenness"]),
            fontsize=7, alpha=0.85, textcoords="offset points", xytext=(5, 3),
        )
    for _, row in df.nlargest(5, "pagerank").iterrows():
        if row["entity"] not in df.nlargest(10, "betweenness")["entity"].values:
            ax.annotate(
                row["entity"], (row["pagerank"], row["betweenness"]),
                fontsize=7, alpha=0.85, textcoords="offset points", xytext=(5, 3),
            )

    ax.set_xlabel("PageRank")
    ax.set_ylabel("Betweenness Centrality")
    ax.set_title("PageRank vs Betweenness — Bridge Concept Identification")
    fig.tight_layout()
    save_figure(fig, "03_pagerank_vs_betweenness")
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# 3.  Community detection (Louvain)
# ---------------------------------------------------------------------------
def analyse_communities(G: nx.Graph, nodes: list[str]):
    """Run Louvain community detection and visualise."""
    from networkx.algorithms.community import louvain_communities

    communities = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
    communities = sorted(communities, key=len, reverse=True)

    # Assign community labels
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for ci, comm in enumerate(communities):
        for node in comm:
            labels[node] = ci

    # Community size bar chart
    sizes = [len(c) for c in communities]
    fig, ax = plt.subplots(figsize=(12, 5))
    top_n = min(25, len(sizes))
    colours = sns.color_palette("tab20", top_n)
    ax.bar(range(top_n), sizes[:top_n], color=colours, edgecolor="white")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Number of Entities")
    ax.set_title(f"Community Sizes (Louvain, {len(communities)} communities detected)")

    # Annotate top-3 with sample entities
    for ci in range(min(3, len(communities))):
        sample = sorted(communities[ci], key=lambda n: G.degree(n, weight="weight"), reverse=True)[:5]
        sample_names = [nodes[s] for s in sample]
        ax.text(
            ci, sizes[ci] + max(sizes) * 0.02,
            "\n".join(sample_names),
            ha="center", va="bottom", fontsize=6,
        )

    fig.tight_layout()
    save_figure(fig, "04_community_sizes")
    plt.close(fig)

    # Save community assignments
    comm_df = pd.DataFrame({
        "entity": [nodes[i] for i in range(len(nodes))],
        "community": labels[:len(nodes)],
    })
    comm_df.to_csv(OUTPUT_DIR / "communities.csv", index=False)
    print(f"  Found {len(communities)} communities; largest has {sizes[0]} entities")

    return communities, labels


# ---------------------------------------------------------------------------
# 4.  k-Core decomposition
# ---------------------------------------------------------------------------
def analyse_kcore(G: nx.Graph, nodes: list[str]):
    """Compute k-core decomposition and visualise the shell distribution."""
    coreness = nx.core_number(G)
    max_core = max(coreness.values()) if coreness else 0

    # Shell distribution
    shell_counts = {}
    for k in sorted(set(coreness.values())):
        shell_counts[k] = sum(1 for v in coreness.values() if v == k)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of coreness values
    ax = axes[0]
    core_vals = list(coreness.values())
    ax.hist(core_vals, bins=range(0, max_core + 2), color=PALETTE[5], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Coreness $k$")
    ax.set_ylabel("Number of Entities")
    ax.set_title(f"(a) k-Core Distribution (max core = {max_core})")

    # Cumulative: entities in k-core or higher
    ax = axes[1]
    ks = sorted(shell_counts.keys())
    cumulative = [sum(shell_counts.get(j, 0) for j in ks if j >= k) for k in ks]
    ax.plot(ks, cumulative, "o-", color=PALETTE[6], markersize=4)
    ax.set_xlabel("Core number $k$")
    ax.set_ylabel("Entities in $k$-core or higher")
    ax.set_title("(b) Cumulative k-Core Membership")
    ax.set_yscale("log")

    fig.suptitle("k-Core Decomposition of the Co-Occurrence Graph", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "05_kcore_decomposition")
    plt.close(fig)

    # Inner core entities
    inner_core = [nodes[n] for n, k in coreness.items() if k == max_core]
    print(f"  Max core = {max_core}, inner-core entities: {len(inner_core)}")
    if inner_core:
        print(f"    Sample: {inner_core[:10]}")

    return coreness


# ---------------------------------------------------------------------------
# 5.  Clustering coefficient
# ---------------------------------------------------------------------------
def analyse_clustering(G: nx.Graph, nodes: list[str]):
    """Compute and plot local clustering coefficients."""
    clustering = nx.clustering(G, weight="weight")
    vals = np.array(list(clustering.values()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, color=PALETTE[2], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Clustering Coefficient Distribution "
        f"(mean = {vals.mean():.3f}, median = {np.median(vals):.3f})"
    )
    ax.axvline(vals.mean(), color="red", linestyle="--", label=f"Mean = {vals.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "06_clustering_coefficients")
    plt.close(fig)

    return clustering


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Execute the full graph-theoretic analysis pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    year
        If set, restrict the entity co-occurrence graph to papers
        published in that year (see ``neo4j_utils.fetch_co_occurrence_edges``).
    """
    print("\n" + "=" * 60)
    print("  GRAPH-THEORETIC ANALYSIS")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        G = build_networkx_graph(nodes, edges)

        print("\n--- Degree Distribution ---")
        degrees, weighted_degrees, fit = analyse_degree_distribution(G, nodes)

        print("\n--- Centrality Analysis ---")
        centrality_df = analyse_centralities(G, nodes)

        print("\n--- Community Detection ---")
        communities, labels = analyse_communities(G, nodes)

        print("\n--- k-Core Decomposition ---")
        coreness = analyse_kcore(G, nodes)

        print("\n--- Clustering Coefficients ---")
        clustering = analyse_clustering(G, nodes)

        return {
            "nodes": nodes,
            "graph": G,
            "degrees": degrees,
            "centralities": centrality_df,
            "communities": communities,
            "community_labels": labels,
            "coreness": coreness,
            "clustering": clustering,
        }
    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
