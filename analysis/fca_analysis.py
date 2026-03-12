"""
Formal Concept Analysis (FCA) of the Fusion KG
================================================
Computes and visualises:
  1. Formal context cross-tabulation (entity × category)
  2. Concept lattice statistics
  3. Attribute implications (Duquenne-Guigues basis)
  4. Category co-occurrence heatmap
  5. Category hierarchy from implications

Sampling
--------
  Subsamples to 200 entities (most category-diverse).  The concept
  lattice can grow exponentially in min(|G|, |M|); at 200 × 90 the
  lattice has ~865 concepts and is tractable via Next Closure.

Computational Complexity
-----------------------
  - Next Closure (Ganter): O(|G|·|M|·|L|) where |L| = number of concepts
  - Implication extraction: O(|L|²)
  - Worst case: |L| = 2^min(|G|, |M|)

References
----------
  Ganter, B. & Wille, R. (1999). Formal Concept Analysis. Springer.
  Ganter, B. (2010). Two basic algorithms in concept analysis. ICFCA.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.neo4j_utils import (
    fetch_entity_categories,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("Set2", 8)


# ---------------------------------------------------------------------------
# 1.  Category statistics overview
# ---------------------------------------------------------------------------
def category_statistics(entity_cats: dict[str, set[str]], all_categories: list[str]):
    """Compute and visualise basic category statistics."""
    # Entities per category
    cat_counts = defaultdict(int)
    for cats in entity_cats.values():
        for c in cats:
            cat_counts[c] += 1

    df = pd.DataFrame([
        {"category": c, "entity_count": cat_counts[c]}
        for c in all_categories
    ]).sort_values("entity_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(all_categories) * 0.35)))
    ax.barh(df["category"], df["entity_count"], color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("Number of Entities")
    ax.set_title(f"Entity Counts per NER Category ({len(all_categories)} categories)")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["entity_count"] + max(df["entity_count"]) * 0.01, i,
                str(row["entity_count"]), va="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "16_category_entity_counts")
    plt.close(fig)

    # Multi-category entities
    multi_counts = [len(cats) for cats in entity_cats.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    max_cats = max(multi_counts)
    bins = range(1, max_cats + 2)
    ax.hist(multi_counts, bins=bins, color=PALETTE[1], edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Number of Categories per Entity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Category Assignments per Entity")
    ax.set_xticks(range(1, max_cats + 1))
    fig.tight_layout()
    save_figure(fig, "17_categories_per_entity")
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# 2.  Category co-occurrence matrix
# ---------------------------------------------------------------------------
def category_cooccurrence_heatmap(entity_cats: dict[str, set[str]], all_categories: list[str]):
    """Build and plot the category co-occurrence matrix."""
    n = len(all_categories)
    cat_idx = {c: i for i, c in enumerate(all_categories)}
    matrix = np.zeros((n, n), dtype=int)

    for cats in entity_cats.values():
        cat_list = sorted(cats)
        for c in cat_list:
            matrix[cat_idx[c]][cat_idx[c]] += 1  # diagonal: count
        for i in range(len(cat_list)):
            for j in range(i + 1, len(cat_list)):
                ci, cj = cat_idx[cat_list[i]], cat_idx[cat_list[j]]
                matrix[ci][cj] += 1
                matrix[cj][ci] += 1

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(
        matrix, mask=mask,
        xticklabels=all_categories, yticklabels=all_categories,
        cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5,
        ax=ax, square=True, cbar_kws={"label": "Co-occurrence count"},
        annot_kws={"size": 6},
    )
    ax.set_title("Category Co-Occurrence Matrix\n(number of entities assigned to both categories)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "18_category_cooccurrence")
    plt.close(fig)

    return matrix


# ---------------------------------------------------------------------------
# 3.  Formal Concept Analysis with the `concepts` library
# ---------------------------------------------------------------------------
def run_fca(entity_cats: dict[str, set[str]], all_categories: list[str], max_entities=200):
    """Run FCA and extract the concept lattice + implications.

    Note: FCA lattice computation is exponential in the worst case.
    We limit to ~200 entities (the most category-diverse) to keep
    runtime under a minute.
    """
    try:
        from concepts import Context
    except ImportError:
        print("  WARNING: `concepts` library not installed; skipping FCA lattice.")
        print("  Install with: pip install concepts")
        return None, None

    # Subsample — FCA is combinatorially expensive
    entities = sorted(entity_cats.keys())
    if len(entities) > max_entities:
        # Take entities with highest category diversity
        entities = sorted(
            entities,
            key=lambda e: len(entity_cats[e]),
            reverse=True,
        )[:max_entities]
        print(f"  Subsampled to {max_entities} most category-diverse entities for FCA")

    # The concepts library forbids object/property name overlap.
    # Prefix entity names to guarantee uniqueness.
    overlap = set(entities) & set(all_categories)
    if overlap:
        print(f"  Resolving {len(overlap)} entity/category name overlaps")
    ent_labels = [f"e:{e}" for e in entities]

    booleans = [
        [cat in entity_cats.get(ent, set()) for cat in all_categories]
        for ent in entities
    ]

    ctx = Context(ent_labels, all_categories, booleans)
    lattice = ctx.lattice

    print(f"  Formal concepts in lattice: {len(lattice)}")

    # Extract implications from concept intents:
    # For each concept, its intent lists categories that ALWAYS co-occur
    # for the entities in its extent.  We derive rules A → B where
    # a single category A appearing implies the full intent.
    implications = []
    for concept in lattice:
        intent = set(concept.intent)
        extent = set(concept.extent)
        if len(intent) >= 2 and len(extent) >= 2:
            # Each pair within the intent is a mutual implication
            # for the entities in this concept's extent
            for cat in intent:
                implied = intent - {cat}
                implications.append((frozenset([cat]), frozenset(implied), len(extent)))

    # Deduplicate and keep strongest (largest support)
    seen = {}
    for premise, conclusion, support in implications:
        key = (premise, conclusion)
        if key not in seen or seen[key] < support:
            seen[key] = support
    implications = [
        (set(p), set(c), s)
        for (p, c), s in sorted(seen.items(), key=lambda x: -x[1])
    ]

    print(f"  Attribute implications found: {len(implications)}")
    if implications:
        print("  Top implications (by entity support):")
        for premise, conclusion, support in implications[:15]:
            p_str = " ∧ ".join(sorted(premise)) if premise else "∅"
            c_str = " ∧ ".join(sorted(conclusion))
            print(f"    {p_str}  →  {c_str}  (support: {support} entities)")

    # Save implications
    impl_data = [
        {"premise": sorted(p), "conclusion": sorted(c), "support": s}
        for p, c, s in implications
    ]
    with open(OUTPUT_DIR / "fca_implications.json", "w") as f:
        json.dump(impl_data, f, indent=2)

    return lattice, implications


# ---------------------------------------------------------------------------
# 4.  Implication graph (category hierarchy)
# ---------------------------------------------------------------------------
def plot_implication_graph(implications, all_categories):
    """Visualise implications as a directed graph (premise → conclusion)."""
    if not implications:
        print("  No implications to visualise.")
        return

    import networkx as nx

    G = nx.DiGraph()
    for premise, conclusion, support in implications:
        for p in premise:
            for c in conclusion:
                if G.has_edge(p, c):
                    G[p][c]["weight"] = max(G[p][c]["weight"], support)
                else:
                    G.add_edge(p, c, weight=support)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    # Node sizes based on number of edges
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=PALETTE[2], alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray",
                           arrows=True, arrowsize=15, alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.set_title("Attribute Implication Graph\n(edge: premise category → implied category)")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "19_implication_graph")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None):
    """Execute the FCA analysis pipeline."""
    print("\n" + "=" * 60)
    print("  FORMAL CONCEPT ANALYSIS")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        entity_cats, all_categories = fetch_entity_categories(driver)
        print(f"  Entities with categories: {len(entity_cats)}")
        print(f"  Unique categories: {len(all_categories)}")

        # Category statistics
        print("\n--- Category Statistics ---")
        category_statistics(entity_cats, all_categories)

        # Co-occurrence heatmap
        print("\n--- Category Co-occurrence ---")
        category_cooccurrence_heatmap(entity_cats, all_categories)

        # FCA lattice and implications
        print("\n--- Formal Concept Analysis ---")
        lattice, implications = run_fca(entity_cats, all_categories)

        # Implication graph
        if implications:
            print("\n--- Implication Graph ---")
            plot_implication_graph(implications, all_categories)

        return {
            "entity_categories": entity_cats,
            "all_categories": all_categories,
            "lattice": lattice,
            "implications": implications,
        }

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
