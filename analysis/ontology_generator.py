"""
OWL Ontology Generator for the Fusion Knowledge Graph
======================================================
Builds a proper OWL 2 ontology from the Neo4j knowledge graph by combining:

1. **Entity categories** → ``owl:Class`` hierarchy under ``fusionkg:Entity``
2. **Co-occurrence patterns** → ``owl:ObjectProperty`` declarations for
   frequent inter-category relationships
3. **FCA implications** → ``rdfs:subClassOf`` axioms (if fca_implications.json exists)
4. **Top entities per category** → ``owl:NamedIndividual`` as exemplars

The result is a proper domain ontology suitable for:
  - Guiding KnowledgeGraphBuilder extractions (``kgbuilder run --ontology``)
  - Aligning to upper ontologies (BFO, EMMO)
  - Serving as a SKOS-compatible vocabulary

Output
------
  - output/fusion_ontology.owl  — Full OWL 2 (RDF/XML)
  - output/fusion_ontology.ttl  — Turtle serialisation (human-readable)

Dependencies
------------
  Requires ``rdflib`` (pip install rdflib).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rdflib import Graph, Namespace, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS
except ImportError:
    raise ImportError("rdflib is required: pip install rdflib")

from analysis.neo4j_utils import (
    get_driver, get_database, fetch_entity_categories, OUTPUT_DIR,
)

# ── Namespaces ────────────────────────────────────────────────────────────────

FUSIONKG = Namespace("http://fusionkg.2026/ontology#")
FUSIONKB = Namespace("http://fusionkg.2026/kb#")  # knowledge base (individuals)
BFO = Namespace("http://purl.obolibrary.org/obo/")
SCHEMA = Namespace("http://schema.org/")


def _safe_local(name: str) -> str:
    """Turn a string into a valid IRI local name."""
    # Replace problematic characters
    safe = name.replace(" ", "_").replace("-", "_").replace("/", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    if safe and safe[0].isdigit():
        safe = "C_" + safe
    return safe


# ── Neo4j data fetching ──────────────────────────────────────────────────────

def _fetch_co_occurrence_categories(driver, min_weight: int = 5):
    """Fetch inter-category co-occurrence patterns.

    Returns dict mapping (cat_a, cat_b) → total co-occurrence weight.
    """
    db = get_database()
    query = """
    MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
    WHERE elementId(a) < elementId(b) AND r.weight >= $min_weight
    MATCH (a)-[:IN_CATEGORY]->(ca:Category)
    MATCH (b)-[:IN_CATEGORY]->(cb:Category)
    WHERE ca.name <> cb.name
    RETURN ca.name AS cat_a, cb.name AS cat_b, sum(r.weight) AS total_weight
    ORDER BY total_weight DESC
    """
    pairs: dict[tuple[str, str], int] = {}
    with driver.session(database=db) as session:
        for r in session.run(query, min_weight=min_weight):
            key = tuple(sorted([r["cat_a"], r["cat_b"]]))
            pairs[key] = pairs.get(key, 0) + int(r["total_weight"])
    return pairs


def _fetch_top_entities_per_category(driver, top_n: int = 10):
    """Fetch the top-N entities by mention count per category."""
    db = get_database()
    query = """
    MATCH (e:Entity)-[:IN_CATEGORY]->(c:Category)
    MATCH (p:Paper)-[m:MENTIONS]->(e)
    WITH c.name AS category, e.name_norm AS entity, sum(m.count) AS mentions
    ORDER BY category, mentions DESC
    WITH category, collect({entity: entity, mentions: mentions})[..10] AS top
    RETURN category, top
    """
    result: dict[str, list[dict]] = {}
    with driver.session(database=db) as session:
        for r in session.run(query):
            result[r["category"]] = list(r["top"])
    return result


# ── Ontology construction ────────────────────────────────────────────────────

def build_ontology(
    driver,
    include_individuals: bool = True,
    include_properties: bool = True,
    fca_implications_path: Path | None = None,
) -> Graph:
    """Build an OWL 2 ontology from the fusion KG.

    Parameters
    ----------
    driver
        Neo4j driver.
    include_individuals
        Include top entities as NamedIndividuals.
    include_properties
        Derive ObjectProperties from inter-category co-occurrences.
    fca_implications_path
        Path to fca_implications.json for SubClassOf axioms.
    """
    g = Graph()
    g.bind("fusionkg", FUSIONKG)
    g.bind("fusionkb", FUSIONKB)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("bfo", BFO)
    g.bind("schema", SCHEMA)

    # ── Ontology metadata ─────────────────────────────────────────
    ont = FUSIONKG[""]
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal("Fusion Energy Knowledge Graph Ontology")))
    g.add((ont, RDFS.comment, Literal(
        "Auto-generated from the Fusion KG (Loreti et al., 2025). "
        "Contains entity categories as classes, inter-category relationships "
        "as object properties, and top entities as named individuals."
    )))
    g.add((ont, OWL.versionInfo, Literal("1.0.0-auto")))
    g.add((ont, DCTERMS.creator, Literal("FusionKG Analysis Pipeline")))

    # ── Superclass ────────────────────────────────────────────────
    entity_cls = FUSIONKG["Entity"]
    g.add((entity_cls, RDF.type, OWL.Class))
    g.add((entity_cls, RDFS.label, Literal("Entity")))
    g.add((entity_cls, RDFS.comment, Literal(
        "Top-level class for all entities extracted from fusion energy literature."
    )))
    # Align to BFO:Entity if desired
    g.add((entity_cls, RDFS.seeAlso, BFO["BFO_0000001"]))

    # ── Classes from categories ───────────────────────────────────
    print("  Fetching entity categories …")
    entity_cats, all_categories = fetch_entity_categories(driver)
    print(f"  {len(all_categories)} categories, {len(entity_cats)} entities")

    cat_uris: dict[str, URIRef] = {}
    for cat in all_categories:
        local = _safe_local(cat)
        uri = FUSIONKG[local]
        cat_uris[cat] = uri
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.subClassOf, entity_cls))
        g.add((uri, RDFS.label, Literal(cat)))
        g.add((uri, SKOS.prefLabel, Literal(cat, lang="en")))

    # ── Object properties from co-occurrence ──────────────────────
    if include_properties:
        print("  Deriving inter-category relationships …")
        co_occ = _fetch_co_occurrence_categories(driver, min_weight=5)
        # Take top-30 most common inter-category patterns
        top_pairs = sorted(co_occ.items(), key=lambda x: x[1], reverse=True)[:30]

        # Generic co-occurrence property
        co_occurs_prop = FUSIONKG["coOccursWith"]
        g.add((co_occurs_prop, RDF.type, OWL.ObjectProperty))
        g.add((co_occurs_prop, RDF.type, OWL.SymmetricProperty))
        g.add((co_occurs_prop, RDFS.label, Literal("co-occurs with")))
        g.add((co_occurs_prop, RDFS.domain, entity_cls))
        g.add((co_occurs_prop, RDFS.range, entity_cls))

        for (cat_a, cat_b), weight in top_pairs:
            # Create a specific sub-property for strong inter-category links
            local = f"coOccurs_{_safe_local(cat_a)}_{_safe_local(cat_b)}"
            prop_uri = FUSIONKG[local]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.subPropertyOf, co_occurs_prop))
            g.add((prop_uri, RDFS.label,
                   Literal(f"co-occurs: {cat_a} <-> {cat_b}")))
            if cat_a in cat_uris:
                g.add((prop_uri, RDFS.domain, cat_uris[cat_a]))
            if cat_b in cat_uris:
                g.add((prop_uri, RDFS.range, cat_uris[cat_b]))
            g.add((prop_uri, RDFS.comment,
                   Literal(f"Total co-occurrence weight: {weight}")))

        print(f"  {len(top_pairs)} inter-category relationships")

    # ── FCA implications as SubClassOf ────────────────────────────
    fca_path = fca_implications_path or OUTPUT_DIR / "fca_implications.json"
    if fca_path.exists():
        print("  Loading FCA implications …")
        with open(fca_path) as f:
            implications = json.load(f)
        n_axioms = 0
        for impl in implications:
            premise_cats = impl.get("premise", [])
            conclusion_cats = impl.get("conclusion", [])
            # Simple 1-to-1 implications → SubClassOf
            for p in premise_cats:
                for c in conclusion_cats:
                    if p in cat_uris and c in cat_uris and p != c:
                        g.add((cat_uris[p], RDFS.subClassOf, cat_uris[c]))
                        n_axioms += 1
        print(f"  {n_axioms} SubClassOf axioms from FCA")

    # ── Named individuals (top entities) ──────────────────────────
    if include_individuals:
        print("  Adding top entities as individuals …")
        top_ents = _fetch_top_entities_per_category(driver, top_n=10)
        n_individuals = 0
        for cat, entities in top_ents.items():
            if cat not in cat_uris:
                continue
            for ent_info in entities:
                name = ent_info["entity"]
                local = _safe_local(name)
                ind_uri = FUSIONKB[local]
                g.add((ind_uri, RDF.type, OWL.NamedIndividual))
                g.add((ind_uri, RDF.type, cat_uris[cat]))
                g.add((ind_uri, RDFS.label, Literal(name)))
                g.add((ind_uri, SKOS.prefLabel, Literal(name, lang="en")))
                mentions = ent_info.get("mentions")
                if mentions is not None:
                    g.add((ind_uri, FUSIONKG["mentionCount"],
                           Literal(mentions, datatype=XSD.integer)))
                n_individuals += 1
        print(f"  {n_individuals} named individuals")

    # ── Annotation properties ─────────────────────────────────────
    mc_prop = FUSIONKG["mentionCount"]
    g.add((mc_prop, RDF.type, OWL.DatatypeProperty))
    g.add((mc_prop, RDFS.label, Literal("mention count")))
    g.add((mc_prop, RDFS.domain, entity_cls))
    g.add((mc_prop, RDFS.range, XSD.integer))

    return g


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_ontology(g: Graph, output_dir: Path | None = None) -> dict[str, Path]:
    """Save the ontology in RDF/XML and Turtle formats."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    owl_path = out / "fusion_ontology.owl"
    g.serialize(str(owl_path), format="xml")
    paths["owl"] = owl_path
    print(f"  Saved: {owl_path}")

    ttl_path = out / "fusion_ontology.ttl"
    g.serialize(str(ttl_path), format="turtle")
    paths["ttl"] = ttl_path
    print(f"  Saved: {ttl_path}")

    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(driver, year: int | None = None):
    """Entry point compatible with run_analysis.py."""
    g = build_ontology(driver)
    paths = save_ontology(g)
    n_triples = len(g)
    print(f"  Ontology: {n_triples} triples")
    return {"triples": n_triples, "paths": {k: str(v) for k, v in paths.items()}}


if __name__ == "__main__":
    driver = get_driver()
    try:
        run(driver)
    finally:
        driver.close()
