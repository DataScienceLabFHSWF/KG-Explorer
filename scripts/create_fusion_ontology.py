"""
Generate the Fusion Energy domain OWL ontology for use with KnowledgeGraphBuilder.

This script creates a rich OWL 2 DL ontology covering all entity categories
discovered in the Loreti et al. NER data, with:
  - A three-level class hierarchy (top > domain group > leaf category)
  - Object properties derived from known fusion physics relationships
  - Datatype properties for provenance and metrics
  - Normalisation of messy/duplicate raw category names
  - Annotation-rich output for use with KGBuilder and Fuseki

Output
------
  output/fusion_ontology.owl  — OWL/XML  (for KGBuilder, Protégé, etc.)
  output/fusion_ontology.ttl  — Turtle   (human-readable)

Usage
-----
  python scripts/create_fusion_ontology.py
  python scripts/create_fusion_ontology.py --no-neo4j   # skip Neo4j enrichment

The ontology is written without a Neo4j connection so it can be generated
before the KG is loaded — making it available as input to KnowledgeGraphBuilder
which produces the KG.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS
except ImportError:
    sys.exit("rdflib is required: pip install rdflib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Namespaces ────────────────────────────────────────────────────────────────

FONTO = Namespace("http://fusionkg.2026/ontology#")
FKBASE = Namespace("http://fusionkg.2026/kb#")
BFO = Namespace("http://purl.obolibrary.org/obo/")
SCHEMA = Namespace("http://schema.org/")

# ── Normalisation map ─────────────────────────────────────────────────────────
# Maps messy raw category names → canonical category names

CANONICAL = {
    # Typo / whitespace variants
    " Detection and Monitoring Systems":    "Detection and Monitoring Systems",
    " Experimental Apparatus":              "Experimental Apparatus",
    " Physics Entity":                      "Physics Entity",
    "Detection and Monitoring systems":     "Detection and Monitoring Systems",
    "Nuclear Fusion SystemComponent":       "Nuclear Fusion System Component",
    "Nuclear Fusion SystemComponent":       "Nuclear Fusion System Component",
    "Nuclear FusionSystemComponent":        "Nuclear Fusion System Component",
    "Plasma Event":                         "Plasma Event",
    "Plasma event":                         "Plasma Event",
    "Plasma Property":                      "Plasma Property",
    "Plasma property":                      "Plasma Property",
    "Plasma property.":                     "Plasma Property",
    "Plasma propertyy":                     "Plasma Property",
    "chemical element or compound":         "Chemical Element or Compound",
    "particle":                             "Particle",
    "magnetic pressure":                    "Plasma Property",   # absorb stray value
    "influence quantities":                 "Physics Entity",    # absorb stray value

    # Merge near-duplicates
    "Astrophysical setting":                "Astrophysical Object",
    "Celestial Body":                       "Astrophysical Object",
    "Celestial object":                     "Astrophysical Object",
    "Nuclear Fusion Campaign":              "Nuclear Fusion Experimental Facility Phase",
    "Nuclear Fusion Experimental Facility phase": "Nuclear Fusion Experimental Facility Phase",
    "Experimental Campaign":               "Nuclear Fusion Experimental Facility Phase",
    "Heating and Current Drive":            "Heating Method",
    "Plasma description":                   "Plasma Property",
    "Plasma type":                          "Plasma dynamic and behavior",
    "Plasma dynamic and behavior":          "Plasma Dynamics and Behavior",
    "Research Project":                     "Research field",
    "Research field":                       "Field of Study",
    "Field of Study":                       "Field of Study",
    "Stability analysis":                   "Theory and Calculation",
    "Isotopic effects":                     "Physical Process",
    "Database":                             "Software and Simulation",
    "Software and simulation":              "Software and Simulation",
    "Unit of Measurement":                  "Unit",
    "Workshop":                             "Event",
    "Coordinate System":                    "Field Configuration",
    "Control Systems":                      "Nuclear Fusion System Component",
    "Program":                              "Nuclear Fusion Program",
    "Project":                              "Research field",
    "Publisher":                            "Facility or Institution",
    "Organization":                         "Facility or Institution",
    "Location":                             "Country and location",
    "Propulsion Method":                    "Nuclear Fusion Technique",
    "Nuclear Fusion Device":                "Nuclear Fusion Device Type",
    "Nuclear Fusion Device Line":           "Nuclear Fusion Device Type",
    "Nuclear Fusion Reactor Type":          "Nuclear Fusion Device Type",
    "Nuclear Fusion Diagnostic":            "Detection and Monitoring Systems",
    "Nuclear Fusion Program":               "Nuclear Fusion Program",
    "Acronym":                              "Concept",
    "Ion":                                  "Particle",
    "Geometry":                             "Field Configuration",
    "Application":                          "Technology",
    "Safety Feature and Regulatory Standard": "Technology",
    "Target":                               "Nuclear Fusion System Component",
    "Radiation":                            "Particle",
    "Scientific Publication and citation":  "Concept",
}

# ── Canonical category set ────────────────────────────────────────────────────

CANONICAL_CATEGORIES = sorted({
    "Chemical Element or Compound",
    "Concept",
    "Country and location",
    "Detection and Monitoring Systems",
    "Event",
    "Experimental Apparatus",
    "Facility or Institution",
    "Field Configuration",
    "Field of Study",
    "Heating Method",
    "Material",
    "Nuclear Fusion Device Type",
    "Nuclear Fusion Experimental Facility",
    "Nuclear Fusion Experimental Facility Phase",
    "Nuclear Fusion Program",
    "Nuclear Fusion System Component",
    "Nuclear Fusion System Configuration",
    "Nuclear Fusion Technique",
    "Particle",
    "Person",
    "Physical Process",
    "Physics Entity",
    "Plasma Dynamics and Behavior",
    "Plasma Event",
    "Plasma Property",
    "Plasma region",
    "Software and Simulation",
    "State of Matter",
    "Technology",
    "Theory and Calculation",
    "Time reference",
    "Unit",
    "Astrophysical Object",
})

# ── Domain group hierarchy ────────────────────────────────────────────────────
# Mid-level grouping classes that sit between Entity and the leaf categories.

DOMAIN_GROUPS: dict[str, list[str]] = {
    "FusionDevice": [
        "Nuclear Fusion Device Type",
        "Nuclear Fusion Experimental Facility",
        "Nuclear Fusion Experimental Facility Phase",
        "Nuclear Fusion System Component",
        "Nuclear Fusion System Configuration",
        "Experimental Apparatus",
        "Detection and Monitoring Systems",
    ],
    "PlasmaPhysics": [
        "Plasma Property",
        "Plasma Dynamics and Behavior",
        "Plasma Event",
        "Plasma region",
        "Physical Process",
        "Field Configuration",
        "State of Matter",
    ],
    "FusionScience": [
        "Nuclear Fusion Technique",
        "Heating Method",
        "Theory and Calculation",
        "Physics Entity",
        "Concept",
        "Field of Study",
    ],
    "ResearchContext": [
        "Nuclear Fusion Program",
        "Facility or Institution",
        "Person",
        "Event",
        "Time reference",
        "Software and Simulation",
        "Technology",
    ],
    "Matter": [
        "Chemical Element or Compound",
        "Material",
        "Particle",
        "State of Matter",
        "Astrophysical Object",
    ],
    "Location": [
        "Country and location",
    ],
    "Measurement": [
        "Unit",
    ],
}

# ── Object properties ─────────────────────────────────────────────────────────

OBJECT_PROPERTIES: list[dict] = [
    {
        "id": "isConfinedIn",
        "label": "is confined in",
        "comment": "Relates a plasma to the device in which it is confined.",
        "domain": "PlasmaPhysics",
        "range":  "FusionDevice",
        "inverse": "confines",
    },
    {
        "id": "confines",
        "label": "confines",
        "comment": "A fusion device that provides confinement for a plasma.",
        "domain": "FusionDevice",
        "range":  "PlasmaPhysics",
        "inverse": "isConfinedIn",
    },
    {
        "id": "isComponentOf",
        "label": "is component of",
        "comment": "Relates a system component to its parent device or facility.",
        "domain": "Nuclear Fusion System Component",
        "range":  "Nuclear Fusion Device Type",
        "characteristics": ["Transitive"],
    },
    {
        "id": "usedInFacility",
        "label": "used in facility",
        "comment": "Relates a device, technique, or measurement system to the experimental facility using it.",
        "domain": "FusionDevice",
        "range":  "Nuclear Fusion Experimental Facility",
    },
    {
        "id": "appliesHeatingMethod",
        "label": "applies heating method",
        "comment": "A device or experiment applies a particular plasma heating technique.",
        "domain": "Nuclear Fusion Device Type",
        "range":  "Heating Method",
    },
    {
        "id": "exhibits",
        "label": "exhibits",
        "comment": "A plasma exhibits a physical process or event.",
        "domain": "PlasmaPhysics",
        "range":  "Plasma Event",
    },
    {
        "id": "hasPlasmaProperty",
        "label": "has plasma property",
        "comment": "Associates a plasma regime or event with a measurable property.",
        "domain": "PlasmaPhysics",
        "range":  "Plasma Property",
    },
    {
        "id": "measuredBy",
        "label": "measured by",
        "comment": "A plasma property is measured by a diagnostic or monitoring system.",
        "domain": "Plasma Property",
        "range":  "Detection and Monitoring Systems",
    },
    {
        "id": "producedBy",
        "label": "produced by",
        "comment": "A particle or radiation is produced by a physical process.",
        "domain": "Particle",
        "range":  "Physical Process",
    },
    {
        "id": "partOfProgram",
        "label": "part of program",
        "comment": "An experimental facility or campaign belongs to a fusion programme.",
        "domain": "Nuclear Fusion Experimental Facility",
        "range":  "Nuclear Fusion Program",
    },
    {
        "id": "simulatedBy",
        "label": "simulated by",
        "comment": "A physical process or configuration is modelled by a simulation or software code.",
        "domain": "PlasmaPhysics",
        "range":  "Software and Simulation",
    },
    {
        "id": "basedOnTheory",
        "label": "based on theory",
        "comment": "A technique, simulation, or calculation is grounded in a theoretical framework.",
        "domain": "FusionScience",
        "range":  "Theory and Calculation",
    },
    {
        "id": "conductedBy",
        "label": "conducted by",
        "comment": "An experiment or programme is conducted by an institution.",
        "domain": "FusionDevice",
        "range":  "Facility or Institution",
    },
    {
        "id": "locatedIn",
        "label": "located in",
        "comment": "A facility is located in a country or geographical region.",
        "domain": "Facility or Institution",
        "range":  "Country and location",
    },
    {
        "id": "composedOf",
        "label": "composed of",
        "comment": "A material or compound is composed of chemical elements.",
        "domain": "Material",
        "range":  "Chemical Element or Compound",
    },
    {
        "id": "coOccursWith",
        "label": "co-occurs with",
        "comment": "Two entities co-occur in the same sentence in the fusion literature. "
                   "Symmetric edge derived from NER co-occurrence analysis.",
        "characteristics": ["Symmetric"],
        "domain": "Entity",
        "range":  "Entity",
    },
]

# ── Datatype properties ───────────────────────────────────────────────────────

DATATYPE_PROPERTIES: list[dict] = [
    {
        "id": "mentionCount",
        "label": "mention count",
        "comment": "Number of times this entity was mentioned across the literature corpus.",
        "range": XSD.nonNegativeInteger,
        "domain": "Entity",
    },
    {
        "id": "confidenceScore",
        "label": "confidence score",
        "comment": "KGBuilder extraction confidence score in [0, 1].",
        "range": XSD.decimal,
        "domain": "Entity",
    },
    {
        "id": "paperYear",
        "label": "paper year",
        "comment": "Year the source paper was published.",
        "range": XSD.integer,
        "domain": "Entity",
    },
    {
        "id": "doi",
        "label": "DOI",
        "comment": "Digital Object Identifier of the source paper.",
        "range": XSD.string,
    },
    {
        "id": "sourceText",
        "label": "source text",
        "comment": "The verbatim sentence from which this entity was extracted.",
        "range": XSD.string,
        "domain": "Entity",
    },
    {
        "id": "coOccurrenceWeight",
        "label": "co-occurrence weight",
        "comment": "Number of papers in which two entities co-occur.",
        "range": XSD.nonNegativeInteger,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_local(name: str) -> str:
    safe = name.replace(" ", "").replace("-", "").replace("/", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    if safe and safe[0].isdigit():
        safe = "C_" + safe
    return safe


def _cat_uri(cat: str) -> URIRef:
    return FONTO[_safe_local(cat)]


def _group_uri(group: str) -> URIRef:
    return FONTO[group]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_fusion_ontology() -> Graph:
    g = Graph()
    g.bind("fonto", FONTO)
    g.bind("fkbase", FKBASE)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("bfo", BFO)
    g.bind("schema", SCHEMA)
    g.bind("xsd", XSD)

    # ── Ontology metadata ──────────────────────────────────────────
    ont = FONTO[""]
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal(
        "Nuclear Fusion Energy Knowledge Graph Ontology")))
    g.add((ont, RDFS.comment, Literal(
        "Domain ontology for the nuclear fusion energy knowledge graph. "
        "Derived from the NER schema of Loreti et al. (arXiv:2504.07738) and "
        "extended with a structured class hierarchy and object properties for "
        "use with KnowledgeGraphBuilder (DataScienceLabFHSWF/KnowledgeGraphBuilder, "
        "branch fast-api)."
    )))
    g.add((ont, OWL.versionInfo, Literal("1.0.0")))
    g.add((ont, DCTERMS.creator, Literal(
        "Fusion KG Analysis Pipeline — KG-Explorer")))
    g.add((ont, DCTERMS.subject, Literal("Nuclear Fusion Energy")))

    # ── Top-level superclass ───────────────────────────────────────
    entity_cls = FONTO["Entity"]
    g.add((entity_cls, RDF.type, OWL.Class))
    g.add((entity_cls, RDFS.label, Literal("Entity")))
    g.add((entity_cls, RDFS.comment, Literal(
        "Top-level class for all named entities extracted from "
        "nuclear fusion energy literature."
    )))
    g.add((entity_cls, RDFS.seeAlso, BFO["BFO_0000001"]))

    # ── Domain group (mid-level) classes ───────────────────────────
    group_categories: dict[str, set[str]] = {}
    for group_name, cats in DOMAIN_GROUPS.items():
        guri = _group_uri(group_name)
        g.add((guri, RDF.type, OWL.Class))
        g.add((guri, RDFS.subClassOf, entity_cls))
        g.add((guri, RDFS.label, Literal(group_name)))
        g.add((guri, SKOS.prefLabel, Literal(group_name, lang="en")))
        group_categories[group_name] = set(cats)

    # Category → its group(s)
    cat_to_groups: dict[str, list[str]] = {}
    for group_name, cats in DOMAIN_GROUPS.items():
        for cat in cats:
            cat_to_groups.setdefault(cat, []).append(group_name)

    # ── Leaf category classes ──────────────────────────────────────
    cat_uris: dict[str, URIRef] = {}
    for cat in CANONICAL_CATEGORIES:
        uri = _cat_uri(cat)
        cat_uris[cat] = uri
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.label, Literal(cat)))
        g.add((uri, SKOS.prefLabel, Literal(cat, lang="en")))
        # Subclass of domain group(s) or directly of Entity
        parents = cat_to_groups.get(cat, [])
        if parents:
            for parent in parents:
                g.add((uri, RDFS.subClassOf, _group_uri(parent)))
        else:
            g.add((uri, RDFS.subClassOf, entity_cls))

    # Also ensure "State of Matter" is only in one group
    # (it appears in both Matter and PlasmaPhysics groups — fine for OWL)

    # ── Object properties ──────────────────────────────────────────
    def _resolve_class(name: str) -> URIRef:
        """Resolve either a canonical category name or a group name to a URI."""
        if name in cat_uris:
            return cat_uris[name]
        if name in [g_name for g_name in DOMAIN_GROUPS]:
            return _group_uri(name)
        return entity_cls

    for prop in OBJECT_PROPERTIES:
        pid = FONTO[prop["id"]]
        g.add((pid, RDF.type, OWL.ObjectProperty))
        g.add((pid, RDFS.label, Literal(prop["label"])))
        if prop.get("comment"):
            g.add((pid, RDFS.comment, Literal(prop["comment"])))
        domain_cls = _resolve_class(prop.get("domain", "Entity"))
        range_cls = _resolve_class(prop.get("range", "Entity"))
        g.add((pid, RDFS.domain, domain_cls))
        g.add((pid, RDFS.range, range_cls))
        for char in prop.get("characteristics", []):
            if char == "Symmetric":
                g.add((pid, RDF.type, OWL.SymmetricProperty))
            elif char == "Transitive":
                g.add((pid, RDF.type, OWL.TransitiveProperty))
        if prop.get("inverse"):
            inv_uri = FONTO[prop["inverse"]]
            g.add((pid, OWL.inverseOf, inv_uri))

    # ── Datatype properties ────────────────────────────────────────
    for dprop in DATATYPE_PROPERTIES:
        pid = FONTO[dprop["id"]]
        g.add((pid, RDF.type, OWL.DatatypeProperty))
        g.add((pid, RDFS.label, Literal(dprop["label"])))
        if dprop.get("comment"):
            g.add((pid, RDFS.comment, Literal(dprop["comment"])))
        if "range" in dprop:
            g.add((pid, RDFS.range, dprop["range"]))
        if "domain" in dprop:
            dom = _resolve_class(dprop["domain"])
            g.add((pid, RDFS.domain, dom))

    # ── Annotation property: nameNorm ─────────────────────────────
    nn = FONTO["nameNorm"]
    g.add((nn, RDF.type, OWL.AnnotationProperty))
    g.add((nn, RDFS.label, Literal("normalised name")))
    g.add((nn, RDFS.comment, Literal(
        "Lowercase, whitespace-normalised form of the entity name used as "
        "the Neo4j node key."
    )))

    # ── Paper class ───────────────────────────────────────────────
    paper_cls = FONTO["Paper"]
    g.add((paper_cls, RDF.type, OWL.Class))
    g.add((paper_cls, RDFS.label, Literal("Paper")))
    g.add((paper_cls, RDFS.comment, Literal(
        "A scientific paper from the fusion energy literature corpus."
    )))
    g.add((paper_cls, RDFS.seeAlso, SCHEMA["ScholarlyArticle"]))

    # ── mentions property ─────────────────────────────────────────
    mentions_prop = FONTO["mentions"]
    g.add((mentions_prop, RDF.type, OWL.ObjectProperty))
    g.add((mentions_prop, RDFS.label, Literal("mentions")))
    g.add((mentions_prop, RDFS.comment, Literal(
        "A paper mentions an entity."
    )))
    g.add((mentions_prop, RDFS.domain, paper_cls))
    g.add((mentions_prop, RDFS.range, entity_cls))

    # ── Normalisation annotations ──────────────────────────────────
    # Record which raw category labels map to each canonical class
    # using SKOS altLabel
    reverse_canonical: dict[str, list[str]] = {}
    for raw, canon in CANONICAL.items():
        if canon in cat_uris:
            reverse_canonical.setdefault(canon, []).append(raw)

    for canon, aliases in reverse_canonical.items():
        uri = cat_uris.get(canon)
        if uri is None:
            continue
        for alias in aliases:
            g.add((uri, SKOS.altLabel, Literal(alias.strip(), lang="en")))

    return g


def save_ontology(g: Graph) -> dict[str, Path]:
    owl_path = OUTPUT_DIR / "fusion_ontology.owl"
    ttl_path = OUTPUT_DIR / "fusion_ontology.ttl"
    g.serialize(str(owl_path), format="xml")
    g.serialize(str(ttl_path), format="turtle")
    return {"owl": owl_path, "ttl": ttl_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Fusion Energy domain OWL ontology.")
    parser.parse_args()

    print("Building fusion ontology …")
    g = build_fusion_ontology()
    paths = save_ontology(g)
    print(f"  Triples      : {len(g)}")
    print(f"  Classes      : {len(CANONICAL_CATEGORIES)} leaf + {len(DOMAIN_GROUPS)} groups + 1 top + 1 Paper")
    print(f"  Properties   : {len(OBJECT_PROPERTIES)} object + {len(DATATYPE_PROPERTIES)} datatype")
    print(f"  OWL/XML      : {paths['owl']}")
    print(f"  Turtle       : {paths['ttl']}")
    print("\nDone. Use this ontology with KnowledgeGraphBuilder:")
    print(f"  python scripts/quickstart.py \\")
    print(f"    --ontology {paths['owl']} \\")
    print(f"    --documents output/kgbuilder_input/docs/")


if __name__ == "__main__":
    main()
