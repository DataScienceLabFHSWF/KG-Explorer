"""Shared Neo4j connection helper and graph export utilities."""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_driver():
    """Return a Neo4j driver using .env credentials."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "fusion2026")
    return GraphDatabase.driver(uri, auth=(user, password))


def get_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


def fetch_co_occurrence_edges(driver, database: str | None = None, year: int | None = None):
    """Fetch co-occurrence edges as (src, tgt, weight) tuples.

    Parameters
    ----------
    driver
        Neo4j driver instance.
    database
        Optional database name (defaults to value of NEO4J_DATABASE).
    year
        If provided, restrict edges to papers published in that year
        and recompute weights based on the filtered paper set. This makes
        it easy to perform temporal slices of the graph.

    Returns
    -------
    nodes : list[str]
        Sorted list of unique entity names (name_norm).
    idx : dict[str, int]
        Mapping from entity name to integer index.
    edges : list[tuple[int, int, float]]
        (src_idx, tgt_idx, weight) for every CO_OCCURS_WITH edge. When
        ``year`` is specified the weight equals the number of papers in
        that year in which the two entities co-occurred.

    Notes
    -----
    The underlying Neo4j edges carry a ``papers`` list property which
    records all paper_ids that contributed to the co-occurrence. When a
    year filter is applied we fetch metadata for all papers and then
    compute the intersection in Python. This avoids expensive Cypher
    pattern matches and keeps the API simple.
    """
    db = database or get_database()
    query = """
    MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
    WHERE elementId(a) < elementId(b)
    RETURN a.name_norm AS src, b.name_norm AS tgt,
           r.weight AS weight, r.papers AS papers
    """
    with driver.session(database=db) as session:
        result = session.run(query)
        raw = [(r["src"], r["tgt"], float(r["weight"]), r.get("papers", []))
               for r in result]

    if year is not None:
        # build map of paper_id -> year
        paper_meta = fetch_paper_metadata(driver, database=db)
        valid = {p["paper_id"] for p in paper_meta if p.get("year") == year}
        filtered = []
        for src, tgt, w, papers in raw:
            if not papers:
                continue
            keep = [pid for pid in papers if pid in valid]
            if keep:
                filtered.append((src, tgt, float(len(keep))))
        raw_edges = filtered
    else:
        raw_edges = [(src, tgt, w) for src, tgt, w, _ in raw]

    node_set = set()
    for s, t, _ in raw_edges:
        node_set.add(s)
        node_set.add(t)

    nodes = sorted(node_set)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[s], idx[t], float(w)) for s, t, w in raw_edges]
    return nodes, idx, edges


def fetch_entity_categories(driver, database: str | None = None):
    """Fetch entity → set-of-categories mapping.

    Returns
    -------
    entity_cats : dict[str, set[str]]
        Mapping from entity name_norm to its category names.
    all_categories : list[str]
        Sorted list of unique category names.
    """
    db = database or get_database()
    query = """
    MATCH (e:Entity)-[:IN_CATEGORY]->(c:Category)
    RETURN e.name_norm AS entity, c.name AS category
    """
    from collections import defaultdict
    entity_cats: dict[str, set[str]] = defaultdict(set)
    with driver.session(database=db) as session:
        for r in session.run(query):
            entity_cats[r["entity"]].add(r["category"])

    all_cats = sorted(set(c for cats in entity_cats.values() for c in cats))
    return dict(entity_cats), all_cats


def fetch_paper_metadata(driver, database: str | None = None):
    """Fetch paper nodes with metadata."""
    db = database or get_database()
    query = """
    MATCH (p:Paper)
    RETURN p.paper_id AS paper_id,
           p.title AS title,
           p.year_published AS year,
           p.first_author AS author,
           p.scholarly_citations_count AS citations,
           p.dataset AS dataset
    """
    with driver.session(database=db) as session:
        return [dict(r) for r in session.run(query)]


def build_adjacency_matrix(nodes, edges):
    """Build a sparse adjacency matrix from edge list.

    Returns a scipy CSR matrix of shape (n, n).
    """
    from scipy.sparse import csr_matrix

    n = len(nodes)
    row, col, data = [], [], []
    for si, ti, w in edges:
        row.extend([si, ti])
        col.extend([ti, si])
        data.extend([w, w])
    return csr_matrix((data, (row, col)), shape=(n, n))


def build_networkx_graph(nodes, edges):
    """Build a weighted networkx graph."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for si, ti, w in edges:
        G.add_edge(si, ti, weight=w)
    return G


def save_figure(fig, name: str, dpi: int = 150):
    """Save a matplotlib figure to the output directory."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    return path
