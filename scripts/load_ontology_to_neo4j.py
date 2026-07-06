"""
Load Fusion Ontology into Neo4j via neosemantics (n10s)
=======================================================
Uses the n10s plugin installed in the Docker container to natively import
output/fusion_ontology.ttl as RDF/OWL into Neo4j.

Prerequisites
-------------
1. The docker-compose.yml already has "n10s" in NEO4J_PLUGINS and
   ./output:/var/lib/neo4j/import bind-mount.
   Restart Neo4j if the container is already running:
       docker-compose down && docker-compose up -d
   Wait ~60 s for the container to be healthy and n10s to load.

2. The TTL file must exist:  output/fusion_ontology.ttl
   If not, generate it first:  python -m analysis.ontology_generator

What this script does
---------------------
1. Polls until n10s is available (waits up to 90 s after container start).
2. Creates the uniqueness constraint on Resource.uri (required by n10s).
3. Initialises the n10s graphconfig (idempotent).
4. Optionally wipes previous n10s :Resource nodes (--wipe).
5. Imports the TTL via  n10s.rdf.import.fetch  using a file:// URI that
   resolves inside the container as /var/lib/neo4j/import/<filename>.
6. Prints a labelled summary of what was imported.

Usage
-----
  python scripts/load_ontology_to_neo4j.py
  python scripts/load_ontology_to_neo4j.py --wipe
  python scripts/load_ontology_to_neo4j.py --ttl output/my_ontology.ttl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.neo4j_utils import get_driver, get_database  # type: ignore

_DEFAULT_TTL = Path(__file__).parent.parent / "output" / "fusion_ontology.ttl"

# Inside the container ./output is mounted at this path:
_CONTAINER_IMPORT_DIR = "/var/lib/neo4j/import"

# ── n10s graphconfig ──────────────────────────────────────────────────────────
#   handleVocabUris  SHORTEN  – use namespace prefixes (e.g. fusionkg__uses)
#   handleRDFTypes   LABELS_AND_NODES – each rdf:type becomes a Neo4j label
#                    AND a :Resource node in the ontology graph
_GRAPHCONFIG = {
    "handleVocabUris": "SHORTEN",
    "handleMultival": "ARRAY",
    "handleRDFTypes": "LABELS_AND_NODES",
    "keepLangTag": False,
    "keepCustomDataTypes": False,
}

# ── Cypher ─────────────────────────────────────────────────────────────────────

_CONSTRAINT = """\
CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
FOR (r:Resource) REQUIRE r.uri IS UNIQUE
"""

_GRAPHCONFIG_CHECK = """\
CALL n10s.graphconfig.show() YIELD param
RETURN count(*) AS n
"""

_GRAPHCONFIG_INIT = "CALL n10s.graphconfig.init($cfg)"

_WIPE_RESOURCES = """\
MATCH (r:Resource) DETACH DELETE r
"""

_IMPORT_TTL = """\
CALL n10s.rdf.import.fetch($uri, "Turtle")
YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
RETURN terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
"""

_SUMMARY = """\
MATCH (r:Resource)
WITH labels(r) AS lbls, count(*) AS n
RETURN lbls, n
ORDER BY n DESC
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wait_for_n10s(driver, db: str, retries: int = 18, delay: float = 5.0) -> None:
    """Poll until n10s procedures are available (plugin may still be loading)."""
    for attempt in range(1, retries + 1):
        try:
            with driver.session(database=db) as s:
                s.run("CALL n10s.graphconfig.show() YIELD param RETURN 1 LIMIT 1").consume()
            return
        except Exception as exc:
            if attempt == retries:
                sys.exit(
                    f"\n[ERROR] n10s is not available after {retries} attempts.\n"
                    "Make sure the container is running with the n10s plugin:\n"
                    "  docker-compose down && docker-compose up -d\n"
                    f"Last error: {exc}"
                )
            print(
                f"  n10s not ready yet (attempt {attempt}/{retries}), "
                f"retrying in {delay:.0f} s …"
            )
            time.sleep(delay)


# ── Core logic ─────────────────────────────────────────────────────────────────

def load(ttl_path: Path, wipe: bool, db: str, driver) -> None:
    # The filename maps to the container's import directory
    container_uri = f"file:///{_CONTAINER_IMPORT_DIR.lstrip('/')}/{ttl_path.name}"
    print(f"  Host TTL path  : {ttl_path}")
    print(f"  Container URI  : {container_uri}")

    print("\n[1/5] Checking n10s availability …")
    _wait_for_n10s(driver, db)
    print("      n10s is available.")

    with driver.session(database=db) as session:
        # ── 1. Uniqueness constraint (required before any n10s import) ────────
        print("\n[2/5] Creating Resource.uri uniqueness constraint …")
        session.run(_CONSTRAINT)

        # ── 2. Graphconfig init ───────────────────────────────────────────────
        n = session.run(_GRAPHCONFIG_CHECK).single()["n"]
        if n > 0 and not wipe:
            print("\n[3/5] n10s graphconfig already exists — skipping init.")
        else:
            print("\n[3/5] Initialising n10s graphconfig …")
            session.run(_GRAPHCONFIG_INIT, cfg=_GRAPHCONFIG)

        # ── 3. Optional wipe ──────────────────────────────────────────────────
        if wipe:
            print("\n[4/5] Wiping existing :Resource nodes …")
            session.run(_WIPE_RESOURCES)
            # Re-init graphconfig after wipe (n10s clears config with Resources)
            session.run(_GRAPHCONFIG_INIT, cfg=_GRAPHCONFIG)
        else:
            print("\n[4/5] Skipping wipe (use --wipe to replace previous import).")

        # ── 4. Import ─────────────────────────────────────────────────────────
        print(f"\n[5/5] Importing {ttl_path.name} …")
        result = session.run(_IMPORT_TTL, uri=container_uri)
        row = result.single()
        if row is None:
            sys.exit(
                "[ERROR] n10s.rdf.import.fetch returned no result.\n"
                "Check that the file is in output/ and the container\n"
                "has the ./output:/var/lib/neo4j/import mount."
            )

        status = row["terminationStatus"]
        loaded = row["triplesLoaded"]
        parsed = row["triplesParsed"]
        ns     = row["namespaces"] or {}
        extra  = row["extraInfo"] or ""

        print(f"\n  Termination status : {status}")
        print(f"  Triples parsed     : {parsed}")
        print(f"  Triples loaded     : {loaded}")
        print(f"  Namespaces found   : {len(ns)}")
        for prefix, ns_uri in sorted(ns.items()):
            print(f"    {prefix:<15} {ns_uri}")
        if extra:
            print(f"  Extra info         : {extra}")

        if status != "OK":
            sys.exit(f"\n[ERROR] Import did not finish cleanly — status: {status}\n{extra}")

        # ── 5. Summary ────────────────────────────────────────────────────────
        print("\n  :Resource node label breakdown:")
        for r in session.run(_SUMMARY):
            lbl_str = " | ".join(sorted(r["lbls"]))
            print(f"    {lbl_str:<70}  {r['n']:>6}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="load_ontology_to_neo4j",
        description="Import fusion_ontology.ttl into Neo4j using n10s (neosemantics).",
    )
    parser.add_argument(
        "--ttl", type=Path, default=_DEFAULT_TTL,
        help=f"Path to the Turtle file on the HOST (default: {_DEFAULT_TTL})",
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="Delete all existing :Resource nodes before importing (full re-import)",
    )
    args = parser.parse_args()

    if not args.ttl.exists():
        sys.exit(
            f"[ERROR] TTL file not found: {args.ttl}\n"
            "Generate it first:  python -m analysis.ontology_generator"
        )

    driver = get_driver()
    db = get_database()
    try:
        load(args.ttl, wipe=args.wipe, db=db, driver=driver)
    finally:
        driver.close()

    print(
        "\nDone — ontology loaded via n10s.\n"
        "\nSample queries to explore:\n"
        "  // All OWL classes\n"
        "  MATCH (c:Resource:owl__Class)\n"
        "  RETURN c.uri, c.rdfs__label ORDER BY c.rdfs__label\n"
        "\n"
        "  // All typed-IE object properties with domain/range\n"
        "  MATCH (p:Resource:owl__ObjectProperty)-[:rdfs__domain]->(d)\n"
        "        ,(p)-[:rdfs__range]->(r)\n"
        "  RETURN p.rdfs__label AS property, d.rdfs__label AS domain,\n"
        "         r.rdfs__label AS range, p.fusionkg__neo4jRelType AS neo4j_type\n"
        "\n"
        "  // Property hierarchy\n"
        "  MATCH (child:owl__ObjectProperty)-[:rdfs__subPropertyOf]->(parent)\n"
        "  RETURN child.rdfs__label, parent.rdfs__label\n"
    )


if __name__ == "__main__":
    main()
