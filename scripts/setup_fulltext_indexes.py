"""Create Neo4j full-text indexes that back the GraphRAG retrieval pass.

Indexes
-------
- ``paper_text`` — fulltext over ``Paper.title`` and ``Paper.abstract``
- ``entity_name`` — fulltext over ``Entity.name_norm`` (cheap synonym lookup)

Idempotent: dropping + recreating is safe; existing indexes are reused.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from analysis.neo4j_utils import get_driver, get_database


INDEXES = {
    "paper_text": (
        "CREATE FULLTEXT INDEX paper_text IF NOT EXISTS "
        "FOR (p:Paper) ON EACH [p.title, p.abstract]"
    ),
    "entity_name": (
        "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name_norm]"
    ),
}


def main() -> None:
    driver = get_driver()
    db = get_database()
    with driver.session(database=db) as s:
        for name, ddl in INDEXES.items():
            print(f"  Creating index '{name}' …")
            s.run(ddl)
        print("\n  Index status:")
        for r in s.run("SHOW FULLTEXT INDEXES"):
            print(f"    {r['name']:20s}  state={r['state']:10s}  "
                  f"on={r.get('labelsOrTypes', '?')}/{r.get('properties', '?')}")
    driver.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
