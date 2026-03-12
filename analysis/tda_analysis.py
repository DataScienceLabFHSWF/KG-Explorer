"""
Persistent Homology (TDA) Analysis of the Fusion KG
=====================================================
Computes and visualises:
  1. Persistence diagrams for H0, H1, H2
  2. Persistence barcodes
  3. Betti curves (features alive vs filtration parameter)
  4. Most persistent features mapped back to entities

Sampling
--------
  Subsamples to top-800 nodes by weighted degree.  The Vietoris-Rips
  complex grows as O(2^n) in the worst case; at 800 nodes the distance
  matrix is 800×800 and ripser runs in ~15s.

Computational Complexity
-----------------------
  - Distance matrix: O(n²)
  - Ripser (VR persistence up to dim 2): O(n³) typical for dense matrices
    (Bauer, 2021)

References
----------
  Bauer, U. (2021). J. Appl. Comput. Topology, 5, 391-423.
  Carlsson, G. (2009). Bull. AMS, 46(2), 255-308.
  Edelsbrunner, H. & Harer, J. (2010). Computational Topology. AMS.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from analysis.neo4j_utils import (
    fetch_co_occurrence_edges,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
H_COLORS = ["#1976D2", "#E65100", "#6A1B9A"]
H_LABELS = ["$H_0$ (components)", "$H_1$ (loops)", "$H_2$ (voids)"]


def _weight_to_distance(edges, n):
    """Convert co-occurrence weights to distances: d = 1/w.
    Returns a dense distance matrix (for small sub-graphs)."""
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    for si, ti, w in edges:
        d = 1.0 / max(w, 1e-10)
        dist[si][ti] = d
        dist[ti][si] = d
    return dist


def _select_subgraph(nodes, edges, max_nodes=800):
    """Select the densest subgraph for TDA (full graph is too large).

    Strategy: take the top-N nodes by weighted degree, keep edges between them.
    """
    n = len(nodes)
    if n <= max_nodes:
        return nodes, edges, list(range(n))

    # Compute weighted degree
    deg = np.zeros(n)
    for si, ti, w in edges:
        deg[si] += w
        deg[ti] += w

    top_indices = set(np.argsort(deg)[-max_nodes:])
    # Re-index
    old_to_new = {}
    new_nodes = []
    for old_idx in sorted(top_indices):
        old_to_new[old_idx] = len(new_nodes)
        new_nodes.append(nodes[old_idx])

    new_edges = []
    for si, ti, w in edges:
        if si in old_to_new and ti in old_to_new:
            new_edges.append((old_to_new[si], old_to_new[ti], w))

    original_indices = sorted(top_indices)
    return new_nodes, new_edges, original_indices


# ---------------------------------------------------------------------------
# 1.  Persistence diagrams
# ---------------------------------------------------------------------------
def compute_persistence(dist_matrix, max_dim=2):
    """Compute persistent homology using ripser."""
    from ripser import ripser

    result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    return result["dgms"]


def plot_persistence_diagrams(dgms):
    """Plot persistence diagrams for each homology dimension."""
    n_dims = len(dgms)
    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for dim, (dgm, ax) in enumerate(zip(dgms, axes)):
        if len(dgm) == 0:
            ax.set_title(f"{H_LABELS[dim]} — no features")
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite_pts = dgm[finite_mask]
        infinite_pts = dgm[~finite_mask]

        if len(finite_pts) > 0:
            ax.scatter(
                finite_pts[:, 0], finite_pts[:, 1],
                s=20, alpha=0.6, c=H_COLORS[dim], zorder=3,
            )
        if len(infinite_pts) > 0:
            max_val = finite_pts[:, 1].max() if len(finite_pts) > 0 else 1.0
            ax.scatter(
                infinite_pts[:, 0],
                np.full(len(infinite_pts), max_val * 1.15),
                s=40, marker="^", c=H_COLORS[dim], zorder=3,
                label="∞ (never dies)",
            )
            ax.legend(fontsize=8)

        # Diagonal
        all_finite = dgm[finite_mask]
        if len(all_finite) > 0:
            lo = min(all_finite.min(), 0)
            hi = all_finite.max() * 1.2
        else:
            lo, hi = 0, 1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"{H_LABELS[dim]}  ({len(dgm)} features)")

    fig.suptitle("Persistence Diagrams of the Co-Occurrence Complex", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "07_persistence_diagrams")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2.  Barcodes
# ---------------------------------------------------------------------------
def plot_barcodes(dgms):
    """Plot persistence barcodes."""
    n_dims = len(dgms)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims))
    if n_dims == 1:
        axes = [axes]

    for dim, (dgm, ax) in enumerate(zip(dgms, axes)):
        if len(dgm) == 0:
            ax.set_title(f"{H_LABELS[dim]} — no features")
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite_pts = dgm[finite_mask]

        # Sort by persistence (longest first)
        if len(finite_pts) > 0:
            persistence = finite_pts[:, 1] - finite_pts[:, 0]
            order = np.argsort(-persistence)
            show = order[:min(50, len(order))]  # top 50

            for yi, idx in enumerate(show):
                b, d = finite_pts[idx]
                ax.plot([b, d], [yi, yi], color=H_COLORS[dim], linewidth=2, solid_capstyle="butt")

        ax.set_xlabel("Filtration parameter (1 / co-occurrence weight)")
        ax.set_ylabel("Feature")
        ax.set_title(f"{H_LABELS[dim]} Barcode (top-50 by persistence)")
        ax.invert_yaxis()

    fig.suptitle("Persistence Barcodes", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "08_persistence_barcodes")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.  Betti curves
# ---------------------------------------------------------------------------
def plot_betti_curves(dgms):
    """Plot Betti number as a function of filtration parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        pts = dgm[finite_mask]
        if len(pts) == 0:
            continue

        # Sample filtration values
        all_vals = np.concatenate([pts[:, 0], pts[:, 1]])
        filt_range = np.linspace(all_vals.min(), all_vals.max(), 500)

        betti = np.zeros(len(filt_range))
        for b, d in pts:
            betti += (filt_range >= b) & (filt_range < d)

        ax.plot(filt_range, betti, color=H_COLORS[dim], linewidth=2, label=H_LABELS[dim])

    ax.set_xlabel("Filtration parameter $\\epsilon$ (1 / co-occurrence weight)")
    ax.set_ylabel("Betti number $\\beta_k$")
    ax.set_title("Betti Curves — Topological Features Alive at Each Scale")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "09_betti_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  Most persistent features
# ---------------------------------------------------------------------------
def report_persistent_features(dgms, nodes, dim=1, top_k=10):
    """Report the most persistent H1/H2 features (knowledge gaps / loops)."""
    if dim >= len(dgms) or len(dgms[dim]) == 0:
        print(f"  No H{dim} features found.")
        return

    dgm = dgms[dim]
    finite_mask = np.isfinite(dgm[:, 1])
    pts = dgm[finite_mask]
    if len(pts) == 0:
        print(f"  No finite H{dim} features found.")
        return

    persistence = pts[:, 1] - pts[:, 0]
    order = np.argsort(-persistence)

    label = "loops (redundant knowledge circuits)" if dim == 1 else "voids (knowledge gaps)"
    print(f"\n  Top-{top_k} most persistent H{dim} features ({label}):")
    print(f"  {'Birth':>10} {'Death':>10} {'Persistence':>12}")
    for i in range(min(top_k, len(order))):
        idx = order[i]
        b, d = pts[idx]
        p = persistence[idx]
        print(f"  {b:10.4f} {d:10.4f} {p:12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, max_nodes=800, year: int | None = None):
    """Execute the persistent homology analysis pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    max_nodes
        Size of subgraph for TDA.
    year
        Year filter for the underlying graph (see ``neo4j_utils``).
    """
    print("\n" + "=" * 60)
    print("  PERSISTENT HOMOLOGY (TDA) ANALYSIS")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        # Select subgraph
        sub_nodes, sub_edges, original_indices = _select_subgraph(nodes, edges, max_nodes)
        print(f"  Subgraph for TDA: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        # Build distance matrix
        print("  Building distance matrix...")
        dist = _weight_to_distance(sub_edges, len(sub_nodes))

        # Compute persistent homology
        print("  Computing persistent homology (up to dim 2)...")
        dgms = compute_persistence(dist, max_dim=2)

        for dim, dgm in enumerate(dgms):
            finite = np.sum(np.isfinite(dgm[:, 1]))
            infinite = len(dgm) - finite
            print(f"    H{dim}: {len(dgm)} total features ({finite} finite, {infinite} infinite)")

        # Visualise
        print("  Generating visualisations...")
        plot_persistence_diagrams(dgms)
        plot_barcodes(dgms)
        plot_betti_curves(dgms)

        # Report
        report_persistent_features(dgms, sub_nodes, dim=1)
        report_persistent_features(dgms, sub_nodes, dim=2)

        return {"dgms": dgms, "sub_nodes": sub_nodes}

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
