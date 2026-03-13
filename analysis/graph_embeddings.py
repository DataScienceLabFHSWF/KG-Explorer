"""
Graph Embeddings for the Fusion Knowledge Graph
================================================
Learns dense vector representations of entities and papers using:

  1. Node2Vec structural embeddings (random-walk based, no ML framework needed)
  2. GraphSAGE inductive embeddings (GNN — requires torch + torch-geometric)
  3. Hybrid embeddings  = Node2Vec/GraphSAGE  ⊕  entity-name semantic encoding
     (sentence-transformers required for the semantic component)
  4. UMAP 2D projection → interactive Plotly knowledge map
  5. Downstream applications:
       – Nearest-neighbour "related entity" lookup (cosine similarity)
       – Cluster quality evaluation (silhouette)
       – Embedding-based link prediction baseline

Why embeddings?
---------------
  Graph metrics (centrality, spectral) capture *structural* position but lose
  *content* semantics encoded in entity names.  Embeddings fuse both:
  entities that co-occur often AND share semantic meaning land close together
  in the latent space, enabling downstream retrieval and recommendation.

Node2Vec vs GraphSAGE
---------------------
  Node2Vec (Grover & Leskovec 2016):
    - Random-walk-based, no GPU required.
    - Fixed transductive embeddings — re-run if the graph changes.
    - Fast to train; good baseline for community-structure capture.
    - "Skip-gram" objective: nodes appearing in the same walk are similar.

  GraphSAGE (Hamilton et al. 2017):
    - Inductive: learns an aggregation function, not just lookup table.
    - Can embed previously unseen nodes using their neighbourhood features.
    - Requires PyTorch and torch-geometric (heavy deps).
    - Trained with a link-prediction (edge-existence) unsupervised loss.
    - Better for downstream ML tasks that require generalisation.

  Recommendation:  Start with Node2Vec (always available).  Use GraphSAGE
  for production deployments where new entities arrive frequently.

Outputs
-------
  - knowledge_map.html         — interactive 2D UMAP Plotly scatter map
  - embeddings.npz             — array of node names + embedding matrix
  - node2vec_similar.csv       — Top-K nearest entities per entity
  - embedding_cluster_eval.csv — Silhouette score per community

Dependencies
------------
  Required:  networkx, scikit-learn, numpy, umap-learn
  Optional:  node2vec           (pip install node2vec)
             sentence-transformers  (pip install sentence-transformers)
             plotly             (pip install plotly)
             torch + torch-geometric (pip install torch torch-geometric)
  The module degrades gracefully when optional packages are absent.

References
----------
  Grover, A. & Leskovec, J. (2016). node2vec: Scalable Feature Learning
    for Networks. KDD 2016. arXiv:1607.00653.
  Hamilton, W. et al. (2017). Inductive Representation Learning on Large
    Graphs. NeurIPS 2017. arXiv:1706.02216.
  McInnes, L. et al. (2018). UMAP: Uniform Manifold Approximation and
    Projection. arXiv:1802.03426.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    fetch_co_occurrence_edges,
    get_driver,
    save_figure,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

MAX_NODES = 5000        # Trim graph for embedding; keep highest-degree nodes
TOP_K_SIMILAR = 10      # Nearest neighbours to report per entity
EMBEDDING_DIM = 64      # Node2Vec / combined embedding dimension
UMAP_MIN_DIST = 0.1
UMAP_N_NEIGHBORS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _select_subgraph(G: nx.Graph, max_nodes: int = MAX_NODES):
    """Trim G to top-max_nodes highest weighted-degree nodes."""
    if G.number_of_nodes() <= max_nodes:
        return G

    degrees = {n: sum(d.get("weight", 1.0)
                      for _, _, d in G.edges(n, data=True))
               for n in G.nodes()}
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    return G.subgraph(top_nodes).copy()


def _cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of a 2D embedding matrix."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = M / norms
    return normed @ normed.T


# ---------------------------------------------------------------------------
# 1. Node2Vec Structural Embeddings
# ---------------------------------------------------------------------------
def _build_node2vec_embeddings(G: nx.Graph,
                                dim: int = EMBEDDING_DIM,
                                walk_length: int = 30,
                                num_walks: int = 200,
                                p: float = 1.0,
                                q: float = 0.5,
                                workers: int = 4) -> tuple[list, np.ndarray] | None:
    """
    Learn Node2Vec embeddings.  Returns (node_list, embedding_matrix) or None
    if node2vec is not installed.

    Hyperparameter guidance:
      p (return parameter)  — q < 1 encourages DFS-like walks (structural roles).
                               q > 1 encourages BFS-like walks (local proximity).
                               Default q=0.5 biases toward community structure,
                               which is appropriate for a co-occurrence KG.
      p                     — Controls likelihood of revisiting a node.
                               p=1 is neutral; p<1 → more local; p>1 → more global.

    For the Fusion KG (dense co-occurrence clusters):
      p=1.0, q=0.5 is recommended — rewards connectivity within clusters.
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        print("  [INFO] node2vec not installed. "
              "Run: pip install node2vec\n"
              "  Skipping Node2Vec structural embeddings.")
        return None

    print("  Training Node2Vec (this may take a few minutes)…")
    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        weight_key="weight",
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    node_list = [n for n in G.nodes() if str(n) in model.wv]
    if not node_list:
        return None

    matrix = np.array([model.wv[str(n)] for n in node_list])
    print(f"  Node2Vec matrix: {matrix.shape}")
    return node_list, matrix


# ---------------------------------------------------------------------------
# 2. GraphSAGE Inductive Embeddings (optional — requires torch + pyg)
# ---------------------------------------------------------------------------
def _build_graphsage_embeddings(G: nx.Graph,
                                 node_list: list,
                                 dim: int = EMBEDDING_DIM,
                                 hidden_dim: int = 128,
                                 num_layers: int = 2,
                                 epochs: int = 20) -> np.ndarray | None:
    """
    Train a GraphSAGE model with an unsupervised link-prediction loss and
    return the resulting node embeddings.

    Architecture
    ------------
    Input:  degree-based node feature (1D) — we have no attribute features,
            so we use normalised weighted degree as the sole input feature.
            For richer features, concatenate with entity-name embeddings.

    Model:  SAGEConv × num_layers → ReLU → SAGEConv → L2-normed output.

    Loss:   Positive pairs  = existing edges (sampled).
            Negative pairs  = random non-edges (2× positive count).
            Binary cross-entropy on dot-product similarity.

    Why num_layers=2?  Each layer aggregates one hop of neighbourhood.
    Two layers → each entity's embedding captures its 2-hop neighbourhood,
    which is sufficient for most KG community structure.

    Returns
    -------
    numpy array of shape (len(node_list), dim), or None if torch/pyg absent.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import from_networkx, negative_sampling
    except ImportError:
        print("  [INFO] torch / torch-geometric not installed.\n"
              "  Run: pip install torch torch-geometric\n"
              "  Skipping GraphSAGE embeddings.")
        return None

    print(f"  Training GraphSAGE ({num_layers} layers, {epochs} epochs)…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build PyG Data object ───────────────────────────────────────
    node_idx = {n: i for i, n in enumerate(node_list)}
    edge_index_list = []
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])

    n = len(node_list)
    if not edge_index_list:
        print("  [WARN] No edges in node_list — skipping GraphSAGE.")
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Use normalised weighted degree as a single input feature per node
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1.0
    x = torch.tensor(
        [[degrees.get(n, 0.0) / max_deg] for n in node_list],
        dtype=torch.float,
    )
    x = x.to(device)
    edge_index = edge_index.to(device)

    # ── GraphSAGE model ─────────────────────────────────────────────
    class _GraphSAGE(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, layers):
            super().__init__()
            self.convs = nn.ModuleList()
            dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
            for i in range(layers):
                self.convs.append(SAGEConv(dims[i], dims[i + 1]))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
            return F.normalize(x, dim=-1)

    model = _GraphSAGE(
        in_dim=x.size(1),
        hidden=hidden_dim,
        out_dim=dim,
        layers=num_layers,
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── Training loop (unsupervised link-prediction) ─────────────────
    pos_edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # undirected once
    model.train()
    for epoch in range(epochs):
        optimiser.zero_grad()
        z = model(x, edge_index)

        # Positive scores
        src, dst = pos_edge_index
        pos_scores = (z[src] * z[dst]).sum(dim=-1)

        # Negative samples
        neg_ei = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=n,
            num_neg_samples=pos_scores.size(0),
        )
        neg_scores = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=-1)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=device),
            torch.zeros(neg_scores.size(0), device=device),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimiser.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
    matrix = z.cpu().numpy()
    print(f"  GraphSAGE matrix: {matrix.shape}")
    return matrix


# ---------------------------------------------------------------------------
# 3. Semantic Name Embeddings (optional)
# ---------------------------------------------------------------------------
def _build_semantic_embeddings(node_list: list,
                                dim: int = EMBEDDING_DIM) -> np.ndarray | None:
    """
    Encode entity *names* with a lightweight sentence-transformer model
    and PCA-reduce to ``dim`` dimensions.

    Falls back gracefully if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [INFO] sentence-transformers not installed. "
              "Run: pip install sentence-transformers\n"
              "  Skipping semantic name embeddings.")
        return None

    print("  Encoding entity names with sentence-transformers…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [str(n) for n in node_list],
        batch_size=256,
        show_progress_bar=True,
    )

    # PCA to target dim
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(dim, vectors.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectors_reduced = pca.fit_transform(vectors)
    print(f"  Semantic matrix: {vectors_reduced.shape}")
    return vectors_reduced.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Hybrid Fusion
# ---------------------------------------------------------------------------
def _build_hybrid_embeddings(structural: np.ndarray,
                              semantic: np.ndarray | None,
                              alpha: float = 0.6) -> np.ndarray:
    """
    Concatenate L2-normalised structural and semantic embeddings.
    ``alpha`` weights the structural component in the l2-normed blend.

    If semantic embeddings are absent, returns the structural embeddings.
    """
    def l2_norm(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(norms == 0, 1e-10, norms)

    if semantic is None:
        return l2_norm(structural)

    # Align dimensions if needed (PCA may have reduced to fewer)
    d_struct = structural.shape[1]
    d_sem = semantic.shape[1]
    if d_struct != d_sem:
        # Project to the smaller dimension via truncation or zero-padding
        d = min(d_struct, d_sem)
        structural = structural[:, :d]
        semantic = semantic[:, :d]

    combined = np.concatenate(
        [alpha * l2_norm(structural),
         (1 - alpha) * l2_norm(semantic)],
        axis=1,
    )
    return combined


# ---------------------------------------------------------------------------
# 4. UMAP projection + interactive Plotly map
# ---------------------------------------------------------------------------
def _build_umap_projection(embeddings: np.ndarray,
                            node_list: list,
                            partition: dict | None = None) -> np.ndarray | None:
    """
    Project embeddings to 2D with UMAP and save an interactive Plotly map.
    ``partition`` maps node → community ID (optional colour-coding).
    Returns the 2D coordinates or None if umap-learn is absent.
    """
    try:
        import umap
    except ImportError:
        print("  [INFO] umap-learn not installed. "
              "Run: pip install umap-learn\n"
              "  Skipping UMAP knowledge map.")
        return None

    print("  Computing UMAP 2D projection…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    # Save static matplotlib version (always available)
    fig, ax = plt.subplots(figsize=(12, 10))
    if partition:
        communities = [partition.get(n, -1) for n in node_list]
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=communities, cmap="tab20", alpha=0.55,
            s=12, linewidths=0, rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Community ID")
    else:
        ax.scatter(coords[:, 0], coords[:, 1],
                   alpha=0.55, s=12, color="#1565C0",
                   linewidths=0, rasterized=True)

    ax.set_title("Knowledge Graph Embedding Map (UMAP)\n"
                 "Each point = one entity, proximity = structural+semantic similarity")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    fig.tight_layout()
    save_figure(fig, "50_knowledge_map_umap")
    plt.close(fig)

    # Interactive Plotly version (optional)
    try:
        import plotly.express as px

        df_umap = pd.DataFrame({
            "entity": [str(n) for n in node_list],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "community": [str(partition.get(n, "?")) for n in node_list]
            if partition else ["all"] * len(node_list),
        })
        fig_plotly = px.scatter(
            df_umap,
            x="x", y="y",
            color="community",
            hover_data={"entity": True, "x": False, "y": False},
            title="Fusion KG · Entity Embedding Map (UMAP)",
            template="plotly_white",
            opacity=0.6,
        )
        fig_plotly.update_traces(marker_size=4)
        out_html = OUTPUT_DIR / "knowledge_map.html"
        fig_plotly.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"  Saved: {out_html}")
    except ImportError:
        print("  [INFO] plotly not installed. "
              "Run: pip install plotly  for the interactive map.")

    return coords


# ---------------------------------------------------------------------------
# 5. Nearest-neighbour similarity table
# ---------------------------------------------------------------------------
def _save_nearest_neighbours(node_list: list,
                              embeddings: np.ndarray,
                              k: int = TOP_K_SIMILAR) -> None:
    """Compute pairwise cosine similarity and save top-K per entity."""
    print(f"  Computing top-{k} nearest neighbours…")
    n = len(node_list)
    if n > 10_000:
        print(f"  [WARN] {n} nodes — approximate NN via batched cosine.")

    sim = _cosine_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -1.0)  # exclude self

    records = []
    for i in tqdm(range(n), desc="  NN lookup", unit="entity", leave=False):
        top_idx = np.argpartition(sim[i], -k)[-k:]
        top_idx = top_idx[np.argsort(sim[i][top_idx])[::-1]]
        for j in top_idx:
            records.append({
                "entity": node_list[i],
                "similar_to": node_list[j],
                "cosine_similarity": float(sim[i][j]),
            })

    out_path = OUTPUT_DIR / "node2vec_similar.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. Embedding-based cluster evaluation
# ---------------------------------------------------------------------------
def _evaluate_cluster_quality(node_list: list,
                               embeddings: np.ndarray,
                               partition: dict | None) -> None:
    """
    Compute per-community silhouette scores to assess whether the embedding
    space separates communities well.
    """
    if partition is None:
        return
    from sklearn.metrics import silhouette_samples, silhouette_score

    labels = np.array([partition.get(n, -1) for n in node_list])
    unique, counts = np.unique(labels, return_counts=True)
    # Keep only communities with ≥ 2 members for silhouette
    valid_mask = np.isin(labels, unique[counts >= 2])
    if valid_mask.sum() < 10:
        return

    X = embeddings[valid_mask]
    y = labels[valid_mask]

    try:
        overall = silhouette_score(X, y, metric="cosine",
                                   sample_size=min(5000, len(y)))
        sample_scores = silhouette_samples(X, y, metric="cosine")

        rows = []
        for cid in np.unique(y):
            mask_c = y == cid
            rows.append({
                "community_id": int(cid),
                "n_members": int(mask_c.sum()),
                "mean_silhouette": float(sample_scores[mask_c].mean()),
            })

        df = pd.DataFrame(rows).sort_values("mean_silhouette", ascending=False)
        out_path = OUTPUT_DIR / "embedding_cluster_eval.csv"
        df.to_csv(out_path, index=False)
        print(f"  Silhouette (overall, cosine): {overall:.4f}")
        print(f"  Saved: {out_path}")
    except Exception as exc:
        print(f"  [WARN] Silhouette computation failed: {exc}")


# ---------------------------------------------------------------------------
# Main run entry-point
# ---------------------------------------------------------------------------
def run(driver, year: int | None = None):
    """
    Build, project, and evaluate graph embeddings for the Fusion KG.
    """
    print("  Fetching co-occurrence graph…")
    nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
    print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")

    G = nx.Graph()
    for name in nodes:
        G.add_node(name)
    for si, ti, w in edges:
        G.add_edge(nodes[si], nodes[ti], weight=w)

    # Trim if needed
    G = _select_subgraph(G, max_nodes=MAX_NODES)
    active_nodes = list(G.nodes())
    print(f"  Working subgraph: {len(active_nodes)} nodes, "
          f"{G.number_of_edges()} edges")

    # Optional community labels for colouring
    partition = None
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight")
        print(f"  Community partition: "
              f"{len(set(partition.values()))} communities")
    except ImportError:
        print("  [INFO] python-louvain not installed — community colouring disabled.")

    # ── 1. Node2Vec ────────────────────────────────────────────────────
    n2v_result = _build_node2vec_embeddings(G, dim=EMBEDDING_DIM)

    if n2v_result is None:
        # ── 1b. GraphSAGE (if torch-geometric available) ───────────────
        sage_matrix = _build_graphsage_embeddings(
            G, active_nodes, dim=EMBEDDING_DIM)

        if sage_matrix is not None:
            struct_matrix = sage_matrix
            node_list = active_nodes
            print("  Using GraphSAGE embeddings as structural component.")
        else:
            # ── 1c. Fallback: Laplacian spectral embedding ─────────────
            print("  Falling back to Laplacian spectral embedding…")
            from sklearn.manifold import SpectralEmbedding
            A = nx.to_numpy_array(G, nodelist=active_nodes, weight="weight")
            try:
                se = SpectralEmbedding(n_components=EMBEDDING_DIM,
                                       affinity="precomputed",
                                       random_state=42)
                struct_matrix = se.fit_transform(A).astype(np.float32)
            except Exception as exc:
                print(f"  [WARN] Spectral embedding failed: {exc}")
                return
            node_list = active_nodes
    else:
        node_list, struct_matrix = n2v_result
        # Re-align partition to node_list order (node2vec may drop isolated nodes)
        if partition:
            partition = {n: partition[n] for n in node_list if n in partition}

    # ── 2. Semantic embeddings (optional) ─────────────────────────────
    sem_matrix = _build_semantic_embeddings(node_list, dim=EMBEDDING_DIM)

    # ── 3. Hybrid fusion ──────────────────────────────────────────────
    embeddings = _build_hybrid_embeddings(struct_matrix, sem_matrix, alpha=0.6)
    print(f"  Final embedding matrix: {embeddings.shape}")

    # ── 4. Save raw embeddings ─────────────────────────────────────────
    npz_path = OUTPUT_DIR / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        nodes=np.array(node_list, dtype=object),
        embeddings=embeddings,
    )
    print(f"  Saved: {npz_path}")

    # ── 5. UMAP projection ────────────────────────────────────────────
    _build_umap_projection(embeddings, node_list, partition=partition)

    # ── 6. Nearest neighbours ─────────────────────────────────────────
    _save_nearest_neighbours(node_list, embeddings, k=TOP_K_SIMILAR)

    # ── 7. Cluster quality ────────────────────────────────────────────
    _evaluate_cluster_quality(node_list, embeddings, partition)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  Graph Embeddings summary:")
    print(f"    Entities embedded : {len(node_list)}")
    print(f"    Embedding dim     : {embeddings.shape[1]}")
    print(f"    Semantic fusion   : {'yes' if sem_matrix is not None else 'no'}")
    print(f"    UMAP map          : {'knowledge_map.html'}")

    return {
        "node_list": node_list,
        "embeddings": embeddings,
        "partition": partition,
    }


if __name__ == "__main__":
    _driver = get_driver()
    try:
        run(_driver)
    finally:
        _driver.close()
