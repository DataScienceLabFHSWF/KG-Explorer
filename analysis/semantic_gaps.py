"""
Semantic Gap Detection
======================
Finds knowledge gaps that purely structural methods miss.

Two complementary detectors:

1. **Semantic non-co-occurrence**
   Pairs of entities whose *names* are semantically very similar
   (cosine on sentence-transformer embeddings) but that have NEVER
   co-occurred in any paper. Existing ``link_prediction.py`` only finds
   missing links between nodes that already share neighbours; this
   module catches conceptually-related entities living in *different*
   regions of the graph — the analogue of the "bridge node" gaps
   discussed in arXiv:2502.13025 (Agentic Deep Graph Reasoning).

2. **Cross-community sparse bridges**
   Pairs of Louvain communities that are connected by far fewer
   edges than their sizes would predict (vs. a configuration-model
   null). Low-bridge community pairs mark unexplored interdisciplinary
   territory in the fusion literature.

Reuses
------
- ``output/entity_linker_cache.pkl`` (sentence-transformer embeddings)
- ``output/communities.csv`` (Louvain assignment)
- Neo4j CO_OCCURS_WITH edges via ``fetch_co_occurrence_edges``

Outputs
-------
- ``output/semantic_gaps.csv``         — top semantic-but-not-structural pairs
- ``output/community_bridge_gaps.csv`` — under-connected community pairs

References
----------
- Buehler M.J. (2025). Agentic Deep Graph Reasoning. arXiv:2502.13025.
- Liben-Nowell & Kleinberg (2007). The link-prediction problem for
  social networks. JASIST 58(7), 1019-1031.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    fetch_co_occurrence_edges,
    get_database,
)

logger = logging.getLogger(__name__)

_LINKER_CACHE = OUTPUT_DIR / "entity_linker_cache.pkl"


# ---------------------------------------------------------------------------
# 1.  Semantic non-co-occurrence
# ---------------------------------------------------------------------------
def _load_entity_embeddings():
    """Load cached entity embeddings produced by ``analysis.entity_linker``."""
    if not _LINKER_CACHE.exists():
        raise FileNotFoundError(
            f"{_LINKER_CACHE} not found. Run the chat app once (or instantiate "
            "EntityLinker manually) to build the embedding cache first."
        )
    with _LINKER_CACHE.open("rb") as f:
        cache = pickle.load(f)
    return cache["names"], cache["embeds"]


def _fetch_existing_pairs(driver, names_subset):
    """Return the set of frozenset({a, b}) for every existing CO_OCCURS_WITH edge
    whose both endpoints are in ``names_subset``."""
    db = get_database()
    name_set = set(names_subset)
    cypher = """
    MATCH (a:Entity)-[:CO_OCCURS_WITH]-(b:Entity)
    WHERE elementId(a) < elementId(b)
      AND a.name_norm IN $names AND b.name_norm IN $names
    RETURN a.name_norm AS a, b.name_norm AS b
    """
    with driver.session(database=db) as sess:
        rows = sess.run(cypher, names=list(name_set))
        return {frozenset((r["a"], r["b"])) for r in rows}


def detect_semantic_gaps(
    driver,
    *,
    top_n_entities: int = 2000,
    similarity_threshold: float = 0.55,
    similarity_max: float = 0.80,
    max_pairs: int = 200,
):
    """Find entity pairs with high semantic similarity that never co-occur.

    Parameters
    ----------
    top_n_entities
        Restrict search to the top-N most-mentioned entities (the ones the
        entity_linker indexed first). Smaller = faster, larger = wider net.
    similarity_threshold
        Minimum cosine score (0..1) to consider a pair "semantically related".
    similarity_max
        Maximum cosine score. Above this, pairs are almost always synonyms,
        spelling variants, or plurals — not genuine research opportunities.
    max_pairs
        Maximum rows to return / write to CSV.
    """
    names, embeds = _load_entity_embeddings()
    if len(names) > top_n_entities:
        names = names[:top_n_entities]
        embeds = embeds[:top_n_entities]
    n = len(names)
    print(f"  Considering top-{n} entities for semantic gap search")

    # Cosine similarity (embeddings are L2-normalised by EntityLinker)
    print("  Computing pairwise cosine similarities…")
    sim = embeds @ embeds.T
    # Only upper triangle, exclude diagonal
    iu = np.triu_indices(n, k=1)
    scores = sim[iu]
    mask = (scores >= similarity_threshold) & (scores <= similarity_max)
    candidate_i = iu[0][mask]
    candidate_j = iu[1][mask]
    candidate_s = scores[mask]
    print(
        f"  {len(candidate_s)} candidate pairs in cosine band "
        f"[{similarity_threshold}, {similarity_max}]"
    )

    if len(candidate_s) == 0:
        return pd.DataFrame(columns=["entity_a", "entity_b", "cosine"])

    # Look up which of these already co-occur in Neo4j
    candidate_names = {names[i] for i in candidate_i} | {names[j] for j in candidate_j}
    existing = _fetch_existing_pairs(driver, candidate_names)
    print(f"  Of those, {len(existing)} pairs already co-occur in the graph")

    rows = []
    for i, j, s in zip(candidate_i, candidate_j, candidate_s):
        a, b = names[int(i)], names[int(j)]
        if frozenset((a, b)) in existing:
            continue
        # Exclude trivial near-duplicates ("tokamak" vs "tokamaks")
        if _looks_like_duplicate(a, b):
            continue
        rows.append({"entity_a": a, "entity_b": b, "cosine": float(s)})

    df = pd.DataFrame(rows).sort_values("cosine", ascending=False).head(max_pairs)
    out = OUTPUT_DIR / "semantic_gaps.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} semantic gaps → {out}")
    return df


def _looks_like_duplicate(a: str, b: str) -> bool:
    """Heuristic to drop pairs that are basically the same word.

    Catches: identical strings, substring matches (singular/plural,
    modifier variants), and pairs that become identical after stripping
    hyphens/spaces and folding common unicode (é → e, ü → u, etc.).
    """
    import unicodedata

    GREEK = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "ε": "epsilon", "η": "eta", "θ": "theta", "λ": "lambda",
        "μ": "mu", "ν": "nu", "π": "pi", "ρ": "rho",
        "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi",
        "ψ": "psi", "ω": "omega",
    }

    def norm(s: str) -> str:
        s = s.lower().strip()
        for g, lat in GREEK.items():
            s = s.replace(g, lat)
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.replace("-", "").replace(" ", "").replace("_", "")

    sa, sb = a.lower().strip(), b.lower().strip()
    if sa == sb:
        return True
    if sa in sb or sb in sa:
        return True
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    # Word-order permutation: "flow shear" vs "shear flow"
    ta = sorted(sa.replace("-", " ").split())
    tb = sorted(sb.replace("-", " ").split())
    if ta == tb and len(ta) >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# 2.  Cross-community sparse bridges
# ---------------------------------------------------------------------------
def detect_community_bridge_gaps(
    driver,
    *,
    min_community_size: int = 20,
    max_pairs: int = 100,
    year: int | None = None,
):
    """Find pairs of communities with anomalously few connecting edges.

    For every pair (C_i, C_j) of communities (both above min size), we
    compute the observed number of inter-community edges and compare it
    to the expectation under a configuration-model null:

        expected = (deg_i * deg_j) / (2 * m)

    where deg_i is the total degree of community i and m is the total
    number of edges in the graph. The 'gap score' is

        gap = expected - observed

    so large positive gap = the two communities are *less* connected
    than chance — an under-explored interdisciplinary territory.
    """
    comm_path = OUTPUT_DIR / "communities.csv"
    if not comm_path.exists():
        raise FileNotFoundError(
            f"{comm_path} not found. Run the 'graph' analysis module first."
        )
    comm_df = pd.read_csv(comm_path)
    # Expected columns: entity, community
    name_to_comm = dict(zip(comm_df["entity"], comm_df["community"]))

    print("  Fetching CO_OCCURS_WITH edges…")
    nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
    print(f"    {len(nodes)} nodes, {len(edges)} edges")

    # Sum degrees and inter-community edge counts
    from collections import defaultdict
    comm_deg: dict[int, float] = defaultdict(float)
    pair_obs: dict[tuple[int, int], float] = defaultdict(float)
    total_w = 0.0

    for si, ti, w in edges:
        a, b = nodes[si], nodes[ti]
        ca, cb = name_to_comm.get(a), name_to_comm.get(b)
        if ca is None or cb is None:
            continue
        comm_deg[ca] += w
        comm_deg[cb] += w
        total_w += w
        if ca != cb:
            key = (min(ca, cb), max(ca, cb))
            pair_obs[key] += w

    if total_w == 0:
        print("  No weighted edges found.")
        return pd.DataFrame()

    # Community sizes
    sizes = comm_df.groupby("community").size().to_dict()
    big = {c for c, s in sizes.items() if s >= min_community_size}
    print(f"  {len(big)} communities ≥ {min_community_size} entities")

    # Score every pair of large communities
    rows = []
    big_list = sorted(big)
    for ii, ci in enumerate(big_list):
        for cj in big_list[ii + 1:]:
            obs = pair_obs.get((ci, cj), 0.0)
            exp = (comm_deg[ci] * comm_deg[cj]) / (2.0 * total_w)
            gap = exp - obs
            if gap <= 0:
                continue
            rows.append({
                "community_a": int(ci),
                "community_b": int(cj),
                "size_a": int(sizes[ci]),
                "size_b": int(sizes[cj]),
                "observed_edges": float(obs),
                "expected_edges": float(round(exp, 3)),
                "gap_score": float(round(gap, 3)),
                "ratio": float(round(obs / exp, 4)) if exp > 0 else 0.0,
            })

    df = pd.DataFrame(rows).sort_values("gap_score", ascending=False).head(max_pairs)

    # Annotate each community with up to 3 high-signal labels.
    # Prefer entities ranked by mention count (read from centralities.csv if
    # available); otherwise fall back to the longest non-numeric names.
    rep_by_comm = _community_labels(comm_df)
    df["label_a"] = df["community_a"].map(rep_by_comm)
    df["label_b"] = df["community_b"].map(rep_by_comm)

    out = OUTPUT_DIR / "community_bridge_gaps.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} community bridge gaps → {out}")
    return df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def _community_labels(comm_df: pd.DataFrame, k: int = 3) -> dict:
    """Return {community_id: 'a, b, c'} using the most informative entities.

    Tries (in order):
      1. ranking by ``pagerank`` from ``output/centralities.csv``
      2. ranking by entity name length, excluding pure-numeric strings
    """
    cent_path = OUTPUT_DIR / "centralities.csv"
    rank: dict[str, float] = {}
    if cent_path.exists():
        try:
            cent = pd.read_csv(cent_path)
            if "pagerank" in cent.columns and "entity" in cent.columns:
                rank = dict(zip(cent["entity"], cent["pagerank"]))
        except Exception:
            rank = {}

    def is_meaningful(name: str) -> bool:
        s = str(name).strip()
        if not s or s.replace(".", "").replace(",", "").isdigit():
            return False
        if len(s) < 3:
            return False
        return True

    out: dict = {}
    for cid, grp in comm_df.groupby("community"):
        cands = [e for e in grp["entity"] if is_meaningful(e)]
        if rank:
            cands.sort(key=lambda e: -rank.get(e, 0.0))
        else:
            cands.sort(key=lambda e: -len(str(e)))
        out[cid] = ", ".join(cands[:k]) if cands else ""
    return out


def run(driver, year: int | None = None, **kwargs):
    """Pipeline entry — runs both detectors and writes CSV outputs."""
    print("\n" + "=" * 60)
    print("  SEMANTIC GAP DETECTION")
    print("=" * 60)

    print("\n--- Semantic non-co-occurrence ---")
    try:
        sem_df = detect_semantic_gaps(driver)
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        sem_df = None

    print("\n--- Cross-community sparse bridges ---")
    try:
        comm_df = detect_community_bridge_gaps(driver, year=year)
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        comm_df = None

    return {"semantic_gaps": sem_df, "community_bridge_gaps": comm_df}
