"""Quick schema introspection for evaluating ontology coverage."""
from dotenv import load_dotenv
load_dotenv()
from analysis.neo4j_utils import get_driver

d = get_driver()
with d.session() as s:
    print("=== Node label counts ===")
    for r in s.run("CALL db.labels() YIELD label RETURN label"):
        lbl = r["label"]
        cnt = s.run(f"MATCH (n:{lbl}) RETURN count(n) AS c").single()["c"]
        print(f"  {cnt:8d}  :{lbl}")

    print("\n=== Relationship type counts ===")
    for r in s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"):
        rt = r["relationshipType"]
        cnt = s.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS c").single()["c"]
        print(f"  {cnt:8d}  -[:{rt}]->")

    print("\n=== Categories (top 50) ===")
    for r in s.run(
        "MATCH (c:Category)<-[:IN_CATEGORY]-(e) "
        "RETURN c.name AS name, count(e) AS cnt ORDER BY cnt DESC LIMIT 50"
    ):
        print(f"  {r['cnt']:6d}  {r['name']}")

    print("\n=== Paper properties (sample) ===")
    rec = s.run("MATCH (p:Paper) RETURN keys(p) AS k LIMIT 1").single()
    print("  ", rec["k"] if rec else "no papers")
    yr = s.run(
        "MATCH (p:Paper) WHERE p.year_published IS NOT NULL "
        "RETURN min(p.year_published) AS lo, max(p.year_published) AS hi, "
        "count(p) AS n_with_year"
    ).single()
    print(f"   year range: {yr['lo']}–{yr['hi']}  ({yr['n_with_year']} papers with year)")

    print("\n=== Sample 'tokamak' lookup ===")
    for r in s.run(
        "MATCH (e:Entity) WHERE e.name_norm CONTAINS 'tokamak' "
        "OPTIONAL MATCH (e)-[:IN_CATEGORY]->(c:Category) "
        "OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e) "
        "WITH e.name_norm AS entity, collect(DISTINCT c.name) AS cats, "
        "     count(DISTINCT p) AS papers, sum(m.count) AS mentions "
        "RETURN entity, cats, papers, mentions ORDER BY mentions DESC LIMIT 10"
    ):
        print(f"  {r['entity']:30s}  papers={r['papers']:4d}  cats={r['cats']}")

    print("\n=== Does Paper.abstract exist? ===")
    rec = s.run(
        "MATCH (p:Paper) WHERE p.abstract IS NOT NULL "
        "RETURN count(p) AS n LIMIT 1"
    ).single()
    print(f"   {rec['n']} papers with abstract")
    rec = s.run(
        "MATCH (p:Paper) WHERE p.title IS NOT NULL "
        "RETURN count(p) AS n LIMIT 1"
    ).single()
    print(f"   {rec['n']} papers with title")

d.close()
