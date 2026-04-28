"""
Advanced Gap Detection — structural & temporal signals
======================================================

Implements four mathematically distinct gap-detection methods that
complement the existing pipeline:

1. **Longest shortest paths**  (Buehler 2025, §4)
   Sample BFS to find diameter-defining paths in the largest connected
   component. These are the longest reasoning chains the LLM can build
   from the KG and are the natural seed for compositional ("Step A → D")
   hypothesis generation.

2. **Forman–Ricci edge curvature**  (Sreejith et al. 2016)
   For an unweighted edge (u,v):
       κ_F(u,v) = 4 − deg(u) − deg(v) + 3·|N(u)∩N(v)|
   Strongly negative curvature marks bottleneck edges whose removal
   would disconnect dense regions. Such edges are *fragile bridges*
   between communities — exactly the structural-hole analogue at the
   edge level (vs. the existing node-level structural_holes module).

3. **Temporal hub trajectories**  (Buehler 2025, §5)
   Per-year weighted degree per entity. We classify each well-mentioned
   entity as ``emerging`` (recent ↑), ``stable``, or ``stalled`` (recent ↓)
   using the slope of a linear fit to the last few years. Stalled hubs
   = saturated topics; emerging hubs = research fronts.

4. **Articulation points**  (Buehler 2025, §6, after Tarjan 1972)
   Single nodes whose removal increases the number of connected
   components. In a research KG these are *single-point-of-failure*
   interdisciplinary connectors — losing them would split the literature.

Outputs
-------
- ``output/longest_paths.json``      — diameter chains for LLM seeding
- ``output/edge_curvature.csv``      — top negative-curvature edges
- ``output/entity_trajectories.csv`` — per-entity yearly degree summary
- ``output/articulation_points.csv`` — fragile single-node connectors

References
----------
- Buehler, M.J. (2025). Agentic Deep Graph Reasoning. arXiv:2502.13025.
- Sreejith, R.P. et al. (2016). Forman curvature for complex networks.
  J. Stat. Mech. 063206.
- Tarjan, R.E. (1972). Depth-first search and linear graph algorithms.
  SIAM J. Computing 1(2).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from analysis.neo4j_utils import (
    OUTPUT_DIR,
    fetch_co_occurrence_edges,
    get_database,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import re

_YEAR_RE = re.compile(r"^\d{4}([-/.]\d{1,2}([-/.]\d{1,2})?)?$")
_PURE_NUM_RE = re.compile(r"^[\d.,\s\-/]+$")


def _is_noisy_entity(name: str) -> bool:
    """True for purely numeric, date-like, or otherwise low-information names.

    These slip into NER output but do not represent fusion concepts. They
    pollute path / curvature / articulation analyses, so we filter them
    at graph-build time.
    """
    s = str(name).strip()
    if len(s) < 3:
        return True
    if _YEAR_RE.match(s):
        return True
    if _PURE_NUM_RE.match(s):
        return True
    return False


def _build_graph(driver, year: int | None = None, max_nodes: int | None = None,
                 drop_noisy: bool = True):
    """Fetch CO_OCCURS_WITH edges and return (nodes, networkx graph).

    If ``max_nodes`` is set we restrict to the top-N entities by weighted
    degree to keep BFS / curvature computations tractable.
    If ``drop_noisy`` is set, year-like and purely-numeric entity names
    are removed before subgraph selection.
    """
    nodes, _idx, edges = fetch_co_occurrence_edges(driver, year=year)

    if drop_noisy:
        keep_idx = {i for i, n in enumerate(nodes) if not _is_noisy_entity(n)}
        remap = {old: new for new, old in enumerate(sorted(keep_idx))}
        edges = [(remap[s], remap[t], w) for s, t, w in edges
                 if s in remap and t in remap]
        nodes = [nodes[i] for i in sorted(keep_idx)]

    if max_nodes and len(nodes) > max_nodes:
        deg = np.zeros(len(nodes))
        for s, t, w in edges:
            deg[s] += w
            deg[t] += w
        keep = set(np.argsort(deg)[-max_nodes:])
        remap = {old: new for new, old in enumerate(sorted(keep))}
        edges = [(remap[s], remap[t], w) for s, t, w in edges
                 if s in remap and t in remap]
        nodes = [nodes[i] for i in sorted(keep)]

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for s, t, w in edges:
        if s != t:
            G.add_edge(s, t, weight=float(w))
    return nodes, G


# ---------------------------------------------------------------------------
# 1.  Longest shortest paths  (paper §4)
# ---------------------------------------------------------------------------
def detect_longest_paths(
    driver,
    *,
    max_nodes: int = 5000,
    sample_sources: int = 200,
    top_k: int = 25,
    seed: int = 42,
):
    """Find the longest shortest paths in the largest connected component.

    Pure-BFS sample (no edge weights) is used; this matches the paper's
    "compositional reasoning chain" framing where each hop is a single
    inference step. We sample ``sample_sources`` random starting nodes,
    run BFS from each, and keep paths whose length equals the maximum
    eccentricity seen so far. The top-K longest-and-most-diverse paths
    become hypothesis seeds.
    """
    print("  Building graph…")
    nodes, G = _build_graph(driver, max_nodes=max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Largest component
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        return []
    H = G.subgraph(comps[0]).copy()
    print(f"    Largest component: {H.number_of_nodes()} nodes")

    rng = np.random.default_rng(seed)
    sources = rng.choice(list(H.nodes()), size=min(sample_sources, H.number_of_nodes()),
                         replace=False)

    best: list[tuple[int, int, int]] = []  # (length, src, tgt)
    for src in sources:
        lengths = nx.single_source_shortest_path_length(H, int(src))
        far_node, far_len = max(lengths.items(), key=lambda kv: kv[1])
        best.append((far_len, int(src), int(far_node)))

    # Keep distinct longest pairs
    best.sort(reverse=True)
    seen_pairs: set[frozenset] = set()
    chains = []
    for length, s, t in best:
        key = frozenset((s, t))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        try:
            path = nx.shortest_path(H, source=s, target=t)
        except nx.NetworkXNoPath:
            continue
        chains.append({
            "length": int(length),
            "source": nodes[s],
            "target": nodes[t],
            "path": [nodes[i] for i in path],
        })
        if len(chains) >= top_k:
            break

    out = OUTPUT_DIR / "longest_paths.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(chains, f, indent=2, ensure_ascii=False)
    print(f"  {len(chains)} longest-path chains → {out}")
    return chains


# ---------------------------------------------------------------------------
# 2.  Forman–Ricci edge curvature
# ---------------------------------------------------------------------------
def detect_edge_curvature(
    driver,
    *,
    max_nodes: int = 5000,
    top_k: int = 200,
    min_degree: int = 5,
    max_degree_ratio: float = 5.0,
):
    """Compute Forman–Ricci curvature for each edge; return the most
    negative ones (bottleneck / fragile bridges).

    κ_F(u,v) = 4 − deg(u) − deg(v) + 3·|N(u) ∩ N(v)|

    Filters
    -------
    - Both endpoints must have degree ≥ ``min_degree`` (drop pure leaves).
    - max(deg)/min(deg) ≤ ``max_degree_ratio`` (drop pure hub–spoke edges,
      where curvature is dominated by the hub's degree alone and offers no
      structural insight).
    """
    print("  Building graph…")
    nodes, G = _build_graph(driver, max_nodes=max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    deg = dict(G.degree())
    rows = []
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        if du < min_degree or dv < min_degree:
            continue
        if max(du, dv) / max(min(du, dv), 1) > max_degree_ratio:
            continue
        common = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        kappa = 4 - du - dv + 3 * common
        rows.append((nodes[u], nodes[v], int(du), int(dv),
                     int(common), int(kappa)))

    df = pd.DataFrame(rows, columns=[
        "entity_a", "entity_b", "deg_a", "deg_b", "common_neighbours", "forman_curvature"
    ])
    df = df.sort_values("forman_curvature").head(top_k)
    out = OUTPUT_DIR / "edge_curvature.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} balanced bottleneck edges → {out}")
    return df


# ---------------------------------------------------------------------------
# 3.  Temporal hub trajectories
# ---------------------------------------------------------------------------
def detect_entity_trajectories(
    driver,
    *,
    min_total_mentions: int = 30,
    recent_window: int = 5,
    top_k_per_class: int = 30,
    drop_incomplete_last_year: bool = True,
):
    """Per-year weighted degree per entity → emerging / stable / stalled.

    For every entity with ≥ ``min_total_mentions`` co-occurrences over
    its lifetime, we compute the linear-regression slope of yearly
    weighted degree across the last ``recent_window`` years (relative to
    the median year so units are comparable). Classification:

        slope ≥ +0.5σ  →  emerging
        slope ≤ -0.5σ  →  stalled
        otherwise       →  stable

    The last calendar year is *dropped* by default because partial-year
    crawls always look like a sharp decline.
    """
    db = get_database()
    cypher = """
    MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
    WHERE elementId(a) < elementId(b)
    UNWIND coalesce(r.papers, []) AS pid
    MATCH (p:Paper {paper_id: pid})
    WHERE p.year_published IS NOT NULL
    RETURN a.name_norm AS a, b.name_norm AS b,
           toInteger(p.year_published) AS year, count(*) AS w
    """
    print("  Aggregating per-year edge weights from Neo4j…")
    per_year = defaultdict(lambda: defaultdict(float))  # entity -> {year: degree}
    all_years: set[int] = set()
    with driver.session(database=db) as sess:
        for rec in sess.run(cypher):
            year = int(rec["year"])
            w = float(rec["w"])
            per_year[rec["a"]][year] += w
            per_year[rec["b"]][year] += w
            all_years.add(year)

    print(f"    {len(per_year)} entities have year-resolved co-occurrences")

    drop_year = max(all_years) if (drop_incomplete_last_year and all_years) else None
    if drop_year is not None:
        print(f"    Dropping last year ({drop_year}) as likely-incomplete")

    def is_year_string(s: str) -> bool:
        # Defer to the shared noisy-entity filter so behaviour matches
        # the structural analyses above.
        return _is_noisy_entity(s)

    rows = []
    for ent, years in per_year.items():
        if is_year_string(ent):
            continue
        if drop_year is not None:
            years = {y: w for y, w in years.items() if y != drop_year}
        total = sum(years.values())
        if total < min_total_mentions:
            continue
        ys = sorted(years.keys())
        recent = ys[-recent_window:]
        if len(recent) < 3:
            continue
        x = np.array(recent, dtype=float)
        y = np.array([years[yr] for yr in recent], dtype=float)
        y_log = np.log1p(y)
        if x.std() == 0:
            continue
        slope = float(np.polyfit(x - x.mean(), y_log, 1)[0])
        rows.append({
            "entity": ent,
            "total_degree": float(total),
            "recent_years": ",".join(str(int(v)) for v in recent),
            "recent_degrees": ",".join(f"{v:.0f}" for v in y),
            "log_slope": round(slope, 4),
            "first_year": int(ys[0]),
            "last_year": int(ys[-1]),
        })

    if not rows:
        print("  No qualifying entities.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sigma = df["log_slope"].std() or 1.0
    def _classify(s):
        if s >= 0.5 * sigma:
            return "emerging"
        if s <= -0.5 * sigma:
            return "stalled"
        return "stable"
    df["trajectory"] = df["log_slope"].apply(_classify)

    # Keep top entities per class to bound output size
    parts = []
    for cls in ("emerging", "stalled", "stable"):
        sub = df[df["trajectory"] == cls]
        if cls == "emerging":
            sub = sub.nlargest(top_k_per_class, "log_slope")
        elif cls == "stalled":
            sub = sub.nsmallest(top_k_per_class, "log_slope")
        else:
            sub = sub.nlargest(top_k_per_class, "total_degree")
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["trajectory", "log_slope"], ascending=[True, False]
    )

    out = OUTPUT_DIR / "entity_trajectories.csv"
    df.to_csv(out, index=False)
    counts = df["trajectory"].value_counts().to_dict()
    print(f"  Trajectories: {counts} → {out}")
    return df


# ---------------------------------------------------------------------------
# 4.  Articulation points
# ---------------------------------------------------------------------------
def detect_articulation_points(
    driver,
    *,
    max_nodes: int = 5000,
    top_k: int = 100,
    weight_threshold: float = 2.0,
):
    """Find articulation points (Tarjan): nodes whose removal increases
    the number of connected components.

    The raw co-occurrence graph is too dense for many cuts to exist, so
    we first **prune weak edges** (weight < ``weight_threshold``). On the
    resulting backbone graph, articulation points correspond to genuine
    structural single-points-of-failure.

    Each point is ranked by ``removal_impact`` = size of the second-
    largest fragment that appears after removal.
    """
    print("  Building graph…")
    nodes, G = _build_graph(driver, max_nodes=max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Keep only sufficiently strong edges → backbone
    weak = [(u, v) for u, v, d in G.edges(data=True)
            if d.get("weight", 1.0) < weight_threshold]
    G.remove_edges_from(weak)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"    Backbone graph (w >= {weight_threshold}): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"    Working on giant component: {H.number_of_nodes()} nodes")

    arts = list(nx.articulation_points(H))
    print(f"    Found {len(arts)} articulation points")

    rows = []
    for v in arts:
        H_minus = H.copy()
        H_minus.remove_node(v)
        comps = sorted((len(c) for c in nx.connected_components(H_minus)),
                       reverse=True)
        if len(comps) < 2:
            continue
        rows.append({
            "entity": nodes[v],
            "degree": int(H.degree(v)),
            "n_fragments": len(comps),
            "largest_fragment": int(comps[0]),
            "second_fragment": int(comps[1]),
            "removal_impact": int(comps[1]),
        })

    df = pd.DataFrame(rows).sort_values("removal_impact", ascending=False).head(top_k)
    out = OUTPUT_DIR / "articulation_points.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} fragile bridges → {out}")
    return df


# ---------------------------------------------------------------------------
# 5.  Triadic-closure deficit
# ---------------------------------------------------------------------------
def detect_triadic_closure_deficit(
    driver,
    *,
    max_nodes: int = 4000,
    min_common_neighbours: int = 4,
    top_k: int = 200,
):
    """Open triangles whose closure probability is anomalously low.

    For a non-edge (u, v) with shared-neighbour set S = N(u) ∩ N(v),
    the configuration-model expected closure probability is roughly

        p_exp(u, v) = (deg(u) · deg(v)) / (2m)

    The *observed-vs-expected* score is

        deficit(u, v) = |S| · log( |S| / (1 + p_exp · |S|) )

    Pairs with high ``deficit`` have many bridging neighbours (so the
    pair is in the same neighbourhood) but the actual edge is missing
    far more than chance would predict — a sharper version of
    Adamic-Adar that prices the gap relative to a null model.
    """
    print("  Building graph…")
    nodes, G = _build_graph(driver, max_nodes=max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    deg = dict(G.degree())
    m = G.number_of_edges()
    if m == 0:
        return pd.DataFrame()
    two_m = 2.0 * m

    # Only score non-edges (u, v) reachable via a 2-hop scan
    seen: set[frozenset] = set()
    rows = []
    for u in G.nodes():
        nu = set(G.neighbors(u))
        # Candidate v: 2-hop neighbours, not direct neighbour, not self
        candidates = set()
        for w in nu:
            candidates.update(G.neighbors(w))
        candidates.discard(u)
        candidates -= nu
        for v in candidates:
            key = frozenset((u, v))
            if key in seen:
                continue
            seen.add(key)
            common = nu & set(G.neighbors(v))
            k = len(common)
            if k < min_common_neighbours:
                continue
            p_exp = (deg[u] * deg[v]) / two_m
            # Penalise pairs whose endpoints are themselves super-hubs
            denom = 1.0 + p_exp * k
            if denom <= 0:
                continue
            deficit = k * float(np.log(max(k / denom, 1e-12)))
            if deficit <= 0:
                continue
            rows.append((nodes[u], nodes[v], int(deg[u]), int(deg[v]),
                         int(k), float(round(p_exp, 4)),
                         float(round(deficit, 4))))

    df = pd.DataFrame(rows, columns=[
        "entity_a", "entity_b", "deg_a", "deg_b",
        "common_neighbours", "p_expected", "deficit"
    ]).sort_values("deficit", ascending=False).head(top_k)

    out = OUTPUT_DIR / "triadic_deficit.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} triadic-deficit pairs → {out}")
    return df


# ---------------------------------------------------------------------------
# 6.  2-edge-cuts (bridge edges)
# ---------------------------------------------------------------------------
def detect_bridge_edges(
    driver,
    *,
    max_nodes: int = 5000,
    weight_threshold: float = 2.0,
    top_k: int = 100,
):
    """Find bridge edges (2-edge-cuts): edges whose removal disconnects
    the graph.

    Run on the same backbone graph as ``detect_articulation_points`` so
    bridges reflect genuine interdisciplinary connections, not random
    weak co-occurrences. Each bridge is annotated with the size of the
    fragment that detaches when the edge is removed.
    """
    print("  Building graph…")
    nodes, G = _build_graph(driver, max_nodes=max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    weak = [(u, v) for u, v, d in G.edges(data=True)
            if d.get("weight", 1.0) < weight_threshold]
    G.remove_edges_from(weak)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"    Backbone graph (w >= {weight_threshold}): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() == 0:
        return pd.DataFrame()

    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    bridges = list(nx.bridges(H))
    print(f"    Found {len(bridges)} bridge edges")

    n_total = H.number_of_nodes()
    rows = []
    for u, v in bridges:
        H_minus = H.copy()
        H_minus.remove_edge(u, v)
        comps = sorted(nx.connected_components(H_minus), key=len, reverse=True)
        if len(comps) < 2:
            continue
        # Smaller fragment = isolated branch
        smaller = min(len(comps[0]), len(comps[1]))
        rows.append({
            "entity_a": nodes[u],
            "entity_b": nodes[v],
            "weight": float(H[u][v].get("weight", 1.0)),
            "fragment_size": int(smaller),
            "fraction_isolated": float(round(smaller / n_total, 4)),
        })

    df = pd.DataFrame(rows).sort_values("fragment_size", ascending=False).head(top_k)
    out = OUTPUT_DIR / "bridge_edges.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} bridge edges → {out}")
    return df


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------
def run(driver, year: int | None = None, **kwargs):
    print("\n" + "=" * 60)
    print("  ADVANCED GAP DETECTION")
    print("=" * 60)

    print("\n--- 1. Longest shortest paths ---")
    paths = detect_longest_paths(driver)

    print("\n--- 2. Forman–Ricci edge curvature ---")
    curv = detect_edge_curvature(driver)

    print("\n--- 3. Temporal hub trajectories ---")
    traj = detect_entity_trajectories(driver)

    print("\n--- 4. Articulation points ---")
    arts = detect_articulation_points(driver)

    print("\n--- 5. Triadic-closure deficit ---")
    triadic = detect_triadic_closure_deficit(driver)

    print("\n--- 6. Bridge edges (2-edge-cuts) ---")
    bridges = detect_bridge_edges(driver)

    return {
        "longest_paths": paths,
        "edge_curvature": curv,
        "entity_trajectories": traj,
        "articulation_points": arts,
        "triadic_deficit": triadic,
        "bridge_edges": bridges,
    }
