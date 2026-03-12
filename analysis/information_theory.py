"""
Information-Theoretic Analysis of the Fusion KG
=================================================
Computes and visualises:
  1. Von Neumann graph entropy
  2. Mutual information between NER categories
  3. Category-level entropy (diversity of co-occurrence partners)
  4. Temporal analysis: category popularity over time

Sampling
--------
  Von Neumann entropy subsamples to 5,000 nodes (same as spectral)
  to make eigenvalue computation feasible.

Computational Complexity
-----------------------
  - Von Neumann entropy: O(n² + k·n)  (eigenvalue computation)
  - Category MI: O(|C|² · n)  where |C| = number of categories
  - Shannon entropy: O(|C| · n)

References
----------
  Braunstein, S. L., Gharibian, S. & Severini, S. (2006).
      Annals of Combinatorics, 10(3), 291-317.
  Cover, T. M. & Thomas, J. A. (2006). Elements of Information Theory.
      Wiley, 2nd ed.
"""

from collections import defaultdict
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from analysis.neo4j_utils import (
    build_adjacency_matrix,
    fetch_co_occurrence_edges,
    fetch_entity_categories,
    fetch_paper_metadata,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("coolwarm", 8)


def _eigsh_with_progress(matrix, k, label="eigenvalues", **kwargs):
    """Run eigsh in a background thread while printing elapsed-time updates."""
    result = [None, None]
    done = threading.Event()

    def _worker():
        try:
            vals, vecs = eigsh(matrix, k=k, **kwargs)
            result[0] = vals
            result[1] = vecs
        except Exception as exc:
            result[0] = exc
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.perf_counter()
    t.start()

    interval = 10
    while not done.is_set():
        done.wait(timeout=interval)
        if not done.is_set():
            elapsed = time.perf_counter() - t0
            print(f"    … still computing {label} ({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"    ✓ {label} finished in {elapsed:.1f}s", flush=True)

    if isinstance(result[0], Exception):
        raise result[0]
    return result[0], result[1]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("coolwarm", 8)


# ---------------------------------------------------------------------------
# 1.  Von Neumann graph entropy
# ---------------------------------------------------------------------------
def von_neumann_entropy(A, k=100, max_nodes=5000):
    """Compute the Von Neumann entropy of the graph.

    S(G) = -tr(L̃ log L̃)  where L̃ = L / tr(L)
    Approximated from the first k eigenvalues.
    Uses subsampling for large graphs (eigendecomp is O(n²k)).
    """
    n = A.shape[0]
    if n > max_nodes:
        # Subsample: keep rows/cols of top-max_nodes nodes by degree
        degs = np.array(A.sum(axis=1)).flatten()
        top = np.argsort(degs)[-max_nodes:]
        top.sort()
        A = A[top][:, top]
        print(f"  Subsampled adjacency to {A.shape[0]} nodes for Von Neumann entropy")

    L_norm = laplacian(A, normed=False)
    trace_L = L_norm.diagonal().sum()

    k = min(k, A.shape[0] - 2)
    try:
        eigenvalues, _ = _eigsh_with_progress(
            L_norm, k=k, label=f"{k} eigenvalues for Von Neumann entropy (shift-invert)",
            sigma=0.0, which="LM",
        )
    except Exception:
        eigenvalues, _ = _eigsh_with_progress(
            L_norm, k=k, label=f"{k} eigenvalues for Von Neumann entropy (direct)",
            which="SM",
        )
    eigenvalues = eigenvalues / trace_L

    # Remove zeros to avoid log(0)
    pos = eigenvalues[eigenvalues > 1e-15]
    entropy = -np.sum(pos * np.log2(pos))

    return entropy


# ---------------------------------------------------------------------------
# 2.  Category mutual information
# ---------------------------------------------------------------------------
def category_mutual_information(
    nodes: list[str],
    edges: list[tuple[int, int, float]],
    entity_cats: dict[str, set[str]],
    all_categories: list[str],
):
    """Compute pairwise mutual information between categories via co-occurrence edges."""
    cat_idx = {c: i for i, c in enumerate(all_categories)}
    nc = len(all_categories)

    # Build joint distribution:
    # P(c1, c2) = sum of weights on edges where one end is in c1 and other in c2
    joint = np.zeros((nc, nc))
    for si, ti, w in edges:
        src_name = nodes[si]
        tgt_name = nodes[ti]
        src_cats = entity_cats.get(src_name, set())
        tgt_cats = entity_cats.get(tgt_name, set())
        for sc in src_cats:
            for tc in tgt_cats:
                joint[cat_idx[sc]][cat_idx[tc]] += w
                if sc != tc:
                    joint[cat_idx[tc]][cat_idx[sc]] += w

    total = joint.sum()
    if total == 0:
        return np.zeros((nc, nc))

    P = joint / total
    P_row = P.sum(axis=1)
    P_col = P.sum(axis=0)

    mi = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            if P[i, j] > 0 and P_row[i] > 0 and P_col[j] > 0:
                mi[i, j] = P[i, j] * np.log2(P[i, j] / (P_row[i] * P_col[j]))

    return mi


def plot_mutual_information(mi, all_categories):
    """Heatmap of pairwise mutual information between categories."""
    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.zeros_like(mi, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(
        mi, mask=mask,
        xticklabels=all_categories, yticklabels=all_categories,
        cmap="YlGnBu", annot=True, fmt=".3f", linewidths=0.5,
        ax=ax, square=True,
        cbar_kws={"label": "Mutual Information (bits)"},
        annot_kws={"size": 5},
    )
    ax.set_title("Pairwise Mutual Information Between NER Categories\n"
                 "(via co-occurrence edge weights)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "20_category_mutual_information")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.  Category-level entropy (diversity)
# ---------------------------------------------------------------------------
def category_entropy(
    nodes: list[str],
    edges: list[tuple[int, int, float]],
    entity_cats: dict[str, set[str]],
    all_categories: list[str],
):
    """For each category, compute the Shannon entropy of its co-occurrence partners' categories."""
    cat_idx = {c: i for i, c in enumerate(all_categories)}
    nc = len(all_categories)

    # For each category, count weights to each partner category
    partner_weights = defaultdict(lambda: np.zeros(nc))
    for si, ti, w in edges:
        src_cats = entity_cats.get(nodes[si], set())
        tgt_cats = entity_cats.get(nodes[ti], set())
        for sc in src_cats:
            for tc in tgt_cats:
                partner_weights[sc][cat_idx[tc]] += w
            for tc in src_cats:
                if tc != sc:
                    partner_weights[sc][cat_idx[tc]] += w

    # Compute entropy
    entropies = {}
    for cat in all_categories:
        dist = partner_weights[cat]
        total = dist.sum()
        if total == 0:
            entropies[cat] = 0.0
            continue
        probs = dist / total
        probs = probs[probs > 0]
        entropies[cat] = -np.sum(probs * np.log2(probs))

    # Plot
    df = pd.DataFrame([
        {"category": c, "entropy": entropies[c]}
        for c in all_categories
    ]).sort_values("entropy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(all_categories) * 0.35)))
    colors = plt.cm.RdYlGn(df["entropy"] / df["entropy"].max())
    ax.barh(df["category"], df["entropy"], color=colors, edgecolor="white")
    ax.set_xlabel("Shannon Entropy (bits)")
    ax.set_title("Category Diversity — Entropy of Co-Occurrence Partner Categories\n"
                 "(high = diverse partners, low = concentrated)")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["entropy"] + 0.02, i, f"{row['entropy']:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "21_category_entropy")
    plt.close(fig)

    return entropies


# ---------------------------------------------------------------------------
# 4.  Temporal analysis
# ---------------------------------------------------------------------------
def temporal_category_analysis(driver, database=None):
    """Analyse how category popularity evolves over publication year."""
    from analysis.neo4j_utils import get_database

    db = database or get_database()
    query = """
    MATCH (p:Paper)-[:MENTIONS]->(e:Entity)-[:IN_CATEGORY]->(c:Category)
    WHERE p.year_published IS NOT NULL
    RETURN p.year_published AS year, c.name AS category, count(DISTINCT e) AS entity_count
    ORDER BY year
    """
    with driver.session(database=db) as session:
        rows = [dict(r) for r in session.run(query)]

    if not rows:
        print("  No temporal data available.")
        return

    df = pd.DataFrame(rows)
    # Filter to years with enough data
    year_counts = df.groupby("year")["entity_count"].sum()
    valid_years = year_counts[year_counts > 5].index
    df = df[df["year"].isin(valid_years)]

    if df.empty:
        print("  Not enough temporal data for meaningful analysis.")
        return

    # Top categories by total count
    total_by_cat = df.groupby("category")["entity_count"].sum().nlargest(10)
    top_cats = total_by_cat.index.tolist()

    pivot = df[df["category"].isin(top_cats)].pivot_table(
        index="year", columns="category", values="entity_count",
        aggfunc="sum", fill_value=0,
    )

    # Normalise per year (relative importance)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Absolute
    ax = axes[0]
    pivot.plot(ax=ax, linewidth=2, marker="o", markersize=3)
    ax.set_ylabel("Number of Unique Entities")
    ax.set_title("(a) Category Popularity Over Time (absolute)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    # Relative (stacked area)
    ax = axes[1]
    pivot_norm.plot.area(ax=ax, stacked=True, alpha=0.7)
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Publication Year")
    ax.set_title("(b) Category Share Over Time (normalised)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(0, 100)

    fig.suptitle("Temporal Evolution of NER Categories", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, "22_temporal_category_evolution")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Execute the information-theoretic analysis pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    year
        Optional year filter for the graph.
    """
    print("\n" + "=" * 60)
    print("  INFORMATION-THEORETIC ANALYSIS")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        entity_cats, all_categories = fetch_entity_categories(driver)
        print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")
        print(f"  Categories: {len(all_categories)}")

        # Von Neumann entropy
        print("\n--- Von Neumann Graph Entropy ---")
        A = build_adjacency_matrix(nodes, edges)
        entropy = von_neumann_entropy(A)
        print(f"  Graph entropy S(G) ≈ {entropy:.4f} bits")

        # Mutual information
        print("\n--- Category Mutual Information ---")
        mi = category_mutual_information(nodes, edges, entity_cats, all_categories)
        plot_mutual_information(mi, all_categories)

        # Category entropy (diversity)
        print("\n--- Category Diversity (Shannon Entropy) ---")
        entropies = category_entropy(nodes, edges, entity_cats, all_categories)

        # Temporal analysis
        print("\n--- Temporal Category Evolution ---")
        temporal_category_analysis(driver)

        return {
            "von_neumann_entropy": entropy,
            "mutual_information": mi,
            "category_entropies": entropies,
        }

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
