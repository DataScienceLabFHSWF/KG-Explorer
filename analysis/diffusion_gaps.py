"""
Diffusion-Based Gap Detection
=============================
Combines graph topology with semantic embeddings to find entities that
are conceptually close to a hub but structurally hard to reach from it.

Two diffusion processes are used:

1. **Personalised PageRank (PPR)**
   For a hub entity ``h`` we run PPR with the restart distribution
   concentrated on ``h``. The resulting score vector ``π_h(·)`` is the
   stationary distribution of a random walker who teleports back to
   ``h`` with probability ``α``. Entities with low ``π_h`` are
   diffusion-blocked from ``h``.

2. **Heat-kernel diffusion**
   The matrix exponential ``H_t = exp(-tL)`` of the (normalised)
   graph Laplacian ``L`` measures multi-scale connectivity: short ``t``
   captures local neighbourhood, long ``t`` reveals global structure.
   We use a Krylov-subspace expm-vector product to avoid forming the
   dense matrix.

For each hub we then compare diffusion mass against semantic similarity
(sentence-transformer cosine, reusing the entity_linker cache). Entities
that are *semantically very similar* but *diffusion-cold* are flagged as
**reachability gaps**: the literature treats them as related concepts,
yet there is no efficient citation-style path between them.

Outputs
-------
- ``output/ppr_reachability_gaps.csv``  — per-hub list of cold-but-similar entities
- ``output/heat_kernel_gaps.csv``       — same but using heat kernel at t=3

References
----------
- Page, L. et al. (1999). The PageRank Citation Ranking. Stanford TR.
- Kondor, R. & Lafferty, J. (2002). Diffusion Kernels on Graphs and
  Other Discrete Input Spaces. ICML.
- Buehler, M.J. (2025). Agentic Deep Graph Reasoning. arXiv:2502.13025.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    build_adjacency_matrix,
    fetch_co_occurrence_edges,
)

logger = logging.getLogger(__name__)

_LINKER_CACHE = OUTPUT_DIR / "entity_linker_cache.pkl"


def _load_embeddings():
    if not _LINKER_CACHE.exists():
        raise FileNotFoundError(
            f"{_LINKER_CACHE} not found. Build the entity_linker cache first."
        )
    with _LINKER_CACHE.open("rb") as f:
        cache = pickle.load(f)
    return cache["names"], cache["embeds"]


def _build_subgraph(driver, max_nodes: int):
    """Return (nodes, idx, A) for the top-N entities by weighted degree."""
    nodes, idx, edges = fetch_co_occurrence_edges(driver)
    deg = np.zeros(len(nodes))
    for s, t, w in edges:
        deg[s] += w
        deg[t] += w
    keep = sorted(np.argsort(deg)[-max_nodes:].tolist())
    remap = {old: new for new, old in enumerate(keep)}
    new_nodes = [nodes[i] for i in keep]
    new_edges = [(remap[s], remap[t], w) for s, t, w in edges
                 if s in remap and t in remap]
    A = build_adjacency_matrix(new_nodes, new_edges)
    return new_nodes, A


# ---------------------------------------------------------------------------
# 1. Personalised PageRank
# ---------------------------------------------------------------------------
def _ppr_vector(A_csr: sp.csr_matrix, source: int, alpha: float = 0.15,
                tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Compute PPR with restart at ``source`` via power iteration.

    π_{k+1} = (1-α) · P · π_k + α · e_source
    where P is the row-normalised adjacency matrix.
    """
    n = A_csr.shape[0]
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    Dinv = sp.diags(1.0 / deg)
    P = (Dinv @ A_csr).T.tocsr()  # column-stochastic transpose

    e = np.zeros(n)
    e[source] = 1.0
    pi = e.copy()
    for _ in range(max_iter):
        pi_new = (1 - alpha) * (P @ pi) + alpha * e
        if np.linalg.norm(pi_new - pi, 1) < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi


# ---------------------------------------------------------------------------
# 2. Heat kernel
# ---------------------------------------------------------------------------
def _heat_diffusion(A_csr: sp.csr_matrix, source: int, t: float = 3.0
                    ) -> np.ndarray:
    """Compute exp(-tL) · e_source where L is the symmetric normalised Laplacian.

    Uses ``scipy.sparse.linalg.expm_multiply`` so we never form the dense
    matrix exponential.
    """
    n = A_csr.shape[0]
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(n) - D_inv_sqrt @ A_csr @ D_inv_sqrt
    e = np.zeros(n)
    e[source] = 1.0
    return expm_multiply(-t * L, e)


# ---------------------------------------------------------------------------
# Reachability-gap detection
# ---------------------------------------------------------------------------
def _reachability_gaps_for_hub(
    hub_idx: int,
    nodes: list[str],
    diffusion: np.ndarray,
    embed_lookup: dict[str, np.ndarray],
    *,
    sim_threshold: float = 0.55,
    diffusion_quantile: float = 0.10,
    top_k: int = 10,
):
    """For a single hub, list nodes that are semantically similar but
    diffusion-cold (in the bottom ``diffusion_quantile`` of mass)."""
    hub_name = nodes[hub_idx]
    if hub_name not in embed_lookup:
        return []
    hub_vec = embed_lookup[hub_name]

    # Restrict to nodes we have embeddings for
    in_emb = [(i, n) for i, n in enumerate(nodes) if n in embed_lookup and i != hub_idx]
    if not in_emb:
        return []
    idxs = np.array([i for i, _ in in_emb])
    names = [n for _, n in in_emb]
    M = np.stack([embed_lookup[n] for n in names])

    sims = M @ hub_vec  # vectors are L2-normalised
    diff = diffusion[idxs]

    # diffusion-cold cut-off
    if diff.size == 0:
        return []
    cold = np.quantile(diff, diffusion_quantile)

    mask = (sims >= sim_threshold) & (diff <= cold)
    if not mask.any():
        return []

    sel_sims = sims[mask]
    sel_diff = diff[mask]
    sel_names = [names[i] for i in np.where(mask)[0]]
    # Score = similarity rewarded, low diffusion penalised more for hubs
    score = sel_sims * (1.0 - sel_diff / (sel_diff.max() + 1e-12))
    order = np.argsort(score)[::-1][:top_k]

    return [{
        "hub": hub_name,
        "candidate": sel_names[i],
        "cosine": float(round(sel_sims[i], 4)),
        "diffusion_mass": float(round(sel_diff[i], 8)),
        "score": float(round(score[i], 4)),
    } for i in order]


def detect_reachability_gaps(
    driver,
    *,
    max_nodes: int = 4000,
    n_hubs: int = 25,
    diffusion: str = "ppr",
    sim_threshold: float = 0.55,
    diffusion_quantile: float = 0.10,
    top_k_per_hub: int = 8,
):
    """Run PPR or heat-kernel diffusion from each hub and surface
    semantically-close nodes that are diffusion-cold.

    Parameters
    ----------
    diffusion
        ``"ppr"`` or ``"heat"``.
    n_hubs
        Number of hub seeds (top entities by weighted degree in the
        sub-graph).
    """
    print("  Loading entity embeddings…")
    emb_names, emb_mat = _load_embeddings()
    embed_lookup = {n: emb_mat[i] for i, n in enumerate(emb_names)}

    print("  Building subgraph…")
    nodes, A = _build_subgraph(driver, max_nodes=max_nodes)
    print(f"    {A.shape[0]} nodes, {A.nnz // 2} edges")

    deg = np.asarray(A.sum(axis=1)).flatten()
    hubs = np.argsort(deg)[::-1][:n_hubs]

    rows = []
    for h in hubs:
        if diffusion == "ppr":
            pi = _ppr_vector(A, int(h))
        elif diffusion == "heat":
            pi = _heat_diffusion(A, int(h))
        else:
            raise ValueError(f"unknown diffusion mode: {diffusion}")
        rows.extend(_reachability_gaps_for_hub(
            int(h), nodes, pi, embed_lookup,
            sim_threshold=sim_threshold,
            diffusion_quantile=diffusion_quantile,
            top_k=top_k_per_hub,
        ))

    if not rows:
        print("  No reachability gaps found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    suffix = "ppr" if diffusion == "ppr" else "heat_kernel"
    out = OUTPUT_DIR / f"{suffix}_reachability_gaps.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} reachability-gap pairs → {out}")
    return df


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------
def run(driver, year: int | None = None, **kwargs):
    print("\n" + "=" * 60)
    print("  DIFFUSION-BASED GAP DETECTION (PPR + heat kernel)")
    print("=" * 60)

    print("\n--- 1. Personalised PageRank reachability ---")
    try:
        ppr_df = detect_reachability_gaps(driver, diffusion="ppr")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        ppr_df = None

    print("\n--- 2. Heat-kernel reachability (t = 3) ---")
    try:
        heat_df = detect_reachability_gaps(driver, diffusion="heat")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        heat_df = None

    return {"ppr_gaps": ppr_df, "heat_gaps": heat_df}
