"""
Spectral Analysis of the Fusion KG
====================================
Computes and visualises:
  1. Eigenvalue spectrum of the normalised Laplacian
  2. Fiedler vector (algebraic connectivity / bipartition)
  3. Spectral clustering
  4. Graph Fourier Transform on citation and year signals
  5. Heat kernel trace (multiscale structure)

Sampling
--------
  Subsamples to top-5,000 nodes by weighted degree (MAX_SPECTRAL_NODES).
  The shift-invert eigsh requires an LU factorisation of the Laplacian:
  O(n²) memory for sparse LU on a dense-block graph.  At 5,000 nodes
  this takes ~9s; at 50,000 it is infeasible.

Computational Complexity
-----------------------
  - Normalised Laplacian: O(m)
  - Shift-invert eigsh (k eigenvalues): O(n² + k·n)  (Lehoucq et al., 1998)
  - Spectral clustering (k-means on eigenvectors): O(n·k·i)
  - GFT: O(k·n)  (Shuman et al., 2013)

References
----------
  Chung, F. R. K. (1997). Spectral Graph Theory. AMS.
  Fiedler, M. (1973). Czech. Math. J., 23(2), 298-305.
  Lehoucq, R. B., Sorensen, D. C. & Yang, C. (1998). ARPACK Users' Guide.
  Ng, A. Y., Jordan, M. I. & Weiss, Y. (2002). On Spectral Clustering. NIPS.
  Shuman, D. I. et al. (2013). IEEE Signal Proc. Mag., 30(3), 83-98.
"""

import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from analysis.neo4j_utils import (
    build_adjacency_matrix,
    fetch_co_occurrence_edges,
    get_driver,
    save_figure,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("magma", 8)


def _eigsh_with_progress(matrix, k, label="eigenvalues", **kwargs):
    """Run eigsh in a background thread while printing elapsed-time updates."""
    result = [None, None]  # [eigenvalues, eigenvectors] or exception
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

    interval = 10  # seconds between updates
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


# ---------------------------------------------------------------------------
# 1.  Eigenvalue spectrum
# ---------------------------------------------------------------------------
def compute_spectrum(A, k=50):
    """Compute the smallest k eigenvalues of the normalised Laplacian.

    Uses shift-invert mode (sigma=0) for efficient computation of smallest eigenvalues.
    Prints periodic progress updates so large graphs don't appear frozen.
    """
    L_norm = laplacian(A, normed=True)
    k = min(k, A.shape[0] - 2)
    try:
        eigenvalues, eigenvectors = _eigsh_with_progress(
            L_norm, k=k, label=f"{k} smallest eigenvalues (shift-invert)",
            sigma=0.0, which="LM",
        )
    except Exception:
        # Fallback: if shift-invert fails (e.g. singular matrix), try direct
        eigenvalues, eigenvectors = _eigsh_with_progress(
            L_norm, k=k, label=f"{k} smallest eigenvalues (direct)",
            which="SM",
        )

    # Sort ascending
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def plot_spectrum(eigenvalues):
    """Plot the eigenvalue spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full spectrum
    ax = axes[0]
    ax.plot(range(len(eigenvalues)), eigenvalues, "o-", color=PALETTE[1], markersize=4)
    ax.set_xlabel("Index $i$")
    ax.set_ylabel("$\\lambda_i$")
    ax.set_title("(a) Smallest Eigenvalues of $\\mathcal{L}$")
    ax.axhline(eigenvalues[1], color="red", ls="--", alpha=0.5, label=f"$\\lambda_2 = {eigenvalues[1]:.4f}$")
    ax.legend()

    # Spectral gap
    ax = axes[1]
    gaps = np.diff(eigenvalues)
    ax.bar(range(len(gaps)), gaps, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.set_xlabel("$i$")
    ax.set_ylabel("$\\lambda_{i+1} - \\lambda_i$")
    ax.set_title("(b) Spectral Gaps (eigenvalue differences)")

    fig.suptitle("Spectral Analysis of the Normalised Graph Laplacian", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "10_laplacian_spectrum")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2.  Fiedler vector
# ---------------------------------------------------------------------------
def plot_fiedler_vector(eigenvectors, nodes):
    """Visualise the Fiedler vector (second eigenvector)."""
    fiedler = eigenvectors[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(fiedler, bins=80, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", ls="--", linewidth=1.5, label="Bipartition boundary")
    ax.set_xlabel("Fiedler vector component $v_{2,i}$")
    ax.set_ylabel("Count")
    ax.set_title("(a) Fiedler Vector Distribution")
    ax.legend()

    # Sorted values
    ax = axes[1]
    sorted_vals = np.sort(fiedler)
    ax.plot(sorted_vals, color=PALETTE[4], linewidth=1)
    ax.axhline(0, color="red", ls="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Node (sorted by Fiedler value)")
    ax.set_ylabel("$v_{2,i}$")
    ax.set_title("(b) Sorted Fiedler Vector — Optimal Bipartition")

    pos_count = np.sum(fiedler >= 0)
    neg_count = np.sum(fiedler < 0)
    ax.text(
        0.05, 0.95,
        f"Partition: {neg_count} | {pos_count}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    fig.suptitle("Fiedler Vector — Algebraic Connectivity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "11_fiedler_vector")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.  Spectral clustering
# ---------------------------------------------------------------------------
def spectral_clustering(eigenvectors, nodes, n_clusters=8):
    """Perform spectral clustering and visualise in 2D embedding."""
    X = eigenvectors[:, 1:n_clusters + 1]  # skip constant eigenvector
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X / norms

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(X_normed)

    # 2D scatter using first two spectral coordinates
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_normed[:, 0], X_normed[:, 1],
        c=labels, cmap="tab20", s=8, alpha=0.6,
    )
    ax.set_xlabel("Spectral coordinate 1 ($v_2$)")
    ax.set_ylabel("Spectral coordinate 2 ($v_3$)")
    ax.set_title(f"Spectral Clustering ({n_clusters} clusters)")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    save_figure(fig, "12_spectral_clustering")
    plt.close(fig)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [plt.cm.tab20(i / n_clusters) for i in unique]
    ax.bar(unique, counts, color=colours, edgecolor="white")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Entities")
    ax.set_title("Spectral Cluster Sizes")
    fig.tight_layout()
    save_figure(fig, "13_spectral_cluster_sizes")
    plt.close(fig)

    # Save assignments
    df = pd.DataFrame({"entity": nodes[:len(labels)], "spectral_cluster": labels})
    df.to_csv(OUTPUT_DIR / "spectral_clusters.csv", index=False)

    return labels


# ---------------------------------------------------------------------------
# 4.  Graph Fourier Transform
# ---------------------------------------------------------------------------
def graph_fourier_analysis(eigenvalues, eigenvectors, nodes, signal, signal_name="Signal"):
    """Decompose a graph signal using the GFT and visualise."""
    # Truncate signal to match eigenvector size
    n = eigenvectors.shape[0]
    sig = np.zeros(n)
    sig[:len(signal)] = signal[:n]

    # Replace NaN with 0
    sig = np.nan_to_num(sig, nan=0.0)

    # GFT: project onto eigenbasis
    gft_coeffs = eigenvectors.T @ sig

    # Energy per frequency
    energy = gft_coeffs ** 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GFT coefficients
    ax = axes[0]
    ax.stem(eigenvalues, gft_coeffs, linefmt="-", markerfmt="o", basefmt="k-")
    ax.set_xlabel("Eigenvalue $\\lambda_k$ (graph frequency)")
    ax.set_ylabel("GFT coefficient $\\hat{f}(\\lambda_k)$")
    ax.set_title(f"(a) Graph Fourier Transform of {signal_name}")

    # Energy spectrum
    ax = axes[1]
    n_low = len(eigenvalues) // 3
    low_energy = np.sum(energy[:n_low])
    high_energy = np.sum(energy[n_low:])
    total = low_energy + high_energy
    ax.bar(
        ["Low frequency\n(smooth, global)", "High frequency\n(local, anomalous)"],
        [low_energy / total * 100, high_energy / total * 100],
        color=[PALETTE[1], PALETTE[5]], edgecolor="white",
    )
    ax.set_ylabel("Energy (%)")
    ax.set_title(f"(b) {signal_name} — Spectral Energy Distribution")

    fig.suptitle(f"Graph Fourier Analysis — {signal_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"14_gft_{signal_name.lower().replace(' ', '_')}")
    plt.close(fig)

    return gft_coeffs


# ---------------------------------------------------------------------------
# 5.  Heat kernel trace
# ---------------------------------------------------------------------------
def heat_kernel_analysis(eigenvalues):
    """Compute and plot the heat kernel trace at multiple time scales."""
    t_values = np.logspace(-2, 3, 200)
    traces = np.array([np.sum(np.exp(-t * eigenvalues)) for t in t_values])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_values, traces, color=PALETTE[2], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time $t$")
    ax.set_ylabel("Heat kernel trace $\\mathrm{tr}(e^{-tL})$")
    ax.set_title("Heat Kernel Trace — Multiscale Structure Summary")

    # Annotate characteristic scales
    # The derivative helps identify transitions
    d_traces = np.gradient(np.log(traces + 1e-10), np.log(t_values))
    inflection_pts = np.where(np.diff(np.sign(np.gradient(d_traces))))[0]
    for pt in inflection_pts[:3]:  # annotate first 3
        ax.axvline(t_values[pt], color="gray", ls=":", alpha=0.5)
        ax.text(t_values[pt], traces[pt], f" t={t_values[pt]:.2f}", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "15_heat_kernel_trace")
    plt.close(fig)

    return t_values, traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MAX_SPECTRAL_NODES = 5000  # subsample for tractable eigendecomposition


def _select_subgraph(nodes, edges, max_nodes=MAX_SPECTRAL_NODES):
    """Select densest subgraph for spectral analysis (full graph is too large)."""
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


def run(driver=None, n_eigenvalues=50, n_clusters=8, max_nodes=MAX_SPECTRAL_NODES, year: int | None = None):
    """Execute the full spectral analysis pipeline.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    n_eigenvalues
        Number of smallest Laplacian eigenvalues to compute.
    n_clusters
        k for spectral clustering.
    max_nodes
        Maximum nodes for subsampling.
    year
        Optional publication year filter applied to the graph.
    """
    print("\n" + "=" * 60)
    print("  SPECTRAL GRAPH ANALYSIS")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        # Subsample for tractability
        sub_nodes, sub_edges = _select_subgraph(nodes, edges, max_nodes)
        if len(sub_nodes) < len(nodes):
            print(f"  Subsampled to top-{len(sub_nodes)} nodes by weighted degree")
            print(f"  Subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        A = build_adjacency_matrix(sub_nodes, sub_edges)

        # Eigendecomposition
        k = min(n_eigenvalues, len(nodes) - 2)
        print(f"  Computing {k} smallest eigenvalues of normalised Laplacian...")
        eigenvalues, eigenvectors = compute_spectrum(A, k=k)

        print(f"  Fiedler value (algebraic connectivity): λ₂ = {eigenvalues[1]:.6f}")

        # Visualise spectrum
        print("  Plotting eigenvalue spectrum...")
        plot_spectrum(eigenvalues)

        # Fiedler vector
        print("  Plotting Fiedler vector...")
        plot_fiedler_vector(eigenvectors, sub_nodes)

        # Spectral clustering
        nc = min(n_clusters, k - 1)
        print(f"  Spectral clustering into {nc} clusters...")
        sc_labels = spectral_clustering(eigenvectors, sub_nodes, n_clusters=nc)

        # GFT: weighted degree as signal
        deg_signal = np.array(A.sum(axis=1)).flatten()
        print("  Graph Fourier Transform on weighted degree signal...")
        graph_fourier_analysis(eigenvalues, eigenvectors, sub_nodes, deg_signal, "Weighted Degree")

        # Heat kernel
        print("  Heat kernel analysis...")
        heat_kernel_analysis(eigenvalues)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "spectral_labels": sc_labels,
        }

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
