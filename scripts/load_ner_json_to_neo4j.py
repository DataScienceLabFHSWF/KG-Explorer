import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


def normalize_entity(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def chunked(items: list[dict[str, Any]], size: int = 1000):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "nan":
        return None
    return stripped


def load_json_records(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def extract_graph_rows(data_dir: Path, max_files: int | None = None):
    json_files = sorted(data_dir.glob("*_NER.json"))
    if max_files is not None:
        json_files = json_files[:max_files]

    papers: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    entity_categories: list[dict[str, str]] = []
    co_occurrences: dict[tuple[str, str], dict[str, Any]] = {}

    for json_file in json_files:
        dataset = json_file.stem.replace("_NER", "")
        records = load_json_records(json_file)

        for record_index, record in enumerate(records):
            paper_id = f"{json_file.name}:{record_index}"
            title = record.get("title")
            papers.append(
                {
                    "paper_id": paper_id,
                    "dataset": dataset,
                    "source_file": json_file.name,
                    "title": title if isinstance(title, str) else None,
                    "abstract": record.get("abstract") if isinstance(record.get("abstract"), str) else None,
                    "url": clean_url(record.get("URL")),
                    "year_published": record.get("year_published") if isinstance(record.get("year_published"), int) else None,
                    "first_author": record.get("first_author") if isinstance(record.get("first_author"), str) else None,
                    "scholarly_citations_count": record.get("scholarly_citations_count")
                    if isinstance(record.get("scholarly_citations_count"), int)
                    else None,
                }
            )

            for field_name in record.get("fields_of_study", []) or []:
                if isinstance(field_name, str) and field_name.strip():
                    fields.append({"paper_id": paper_id, "field": field_name.strip()})

            mention_counter: Counter[tuple[str, str]] = Counter()
            entity_name_by_norm: dict[str, str] = {}
            categories_by_entity: defaultdict[str, set[str]] = defaultdict(set)

            ner_entries = record.get("NER-RE", []) or []
            for sentence_block in ner_entries:
                if not isinstance(sentence_block, dict):
                    continue

                unique_entities_in_sentence: set[str] = set()
                for entity_item in sentence_block.get("entities", []) or []:
                    if not isinstance(entity_item, dict):
                        continue
                    entity_name = entity_item.get("entity")
                    category = entity_item.get("category")
                    if not isinstance(entity_name, str) or not entity_name.strip():
                        continue

                    entity_name = entity_name.strip()
                    normalized = normalize_entity(entity_name)
                    if not normalized:
                        continue

                    if not isinstance(category, str) or not category.strip():
                        category = "Uncategorized"
                    category = category.strip()

                    mention_counter[(normalized, category)] += 1
                    entity_name_by_norm.setdefault(normalized, entity_name)
                    categories_by_entity[normalized].add(category)
                    unique_entities_in_sentence.add(normalized)

                for first, second in combinations(sorted(unique_entities_in_sentence), 2):
                    key = (first, second)
                    entry = co_occurrences.get(key)
                    if entry is None:
                        co_occurrences[key] = {"left": first, "right": second, "weight": 1, "papers": {paper_id}}
                    else:
                        entry["weight"] += 1
                        entry["papers"].add(paper_id)

            for (entity_norm, category), count in mention_counter.items():
                mentions.append(
                    {
                        "paper_id": paper_id,
                        "entity_norm": entity_norm,
                        "entity_name": entity_name_by_norm.get(entity_norm, entity_norm),
                        "category": category,
                        "count": count,
                    }
                )

            for entity_norm, categories in categories_by_entity.items():
                for category in categories:
                    entity_categories.append(
                        {
                            "entity_norm": entity_norm,
                            "category": category,
                        }
                    )

    co_occurrence_rows = [
        {
            "left": row["left"],
            "right": row["right"],
            "weight": row["weight"],
            "papers": sorted(row["papers"]),
        }
        for row in co_occurrences.values()
    ]

    return papers, fields, mentions, entity_categories, co_occurrence_rows


def ensure_schema(driver, database: str):
    statements = [
        "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT field_name_unique IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT entity_norm_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name_norm IS UNIQUE",
        "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for statement in statements:
            session.run(statement)


def wipe_graph(driver, database: str):
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def insert_papers(driver, database: str, rows: list[dict[str, Any]]):
    query = """
    UNWIND $rows AS row
    MERGE (p:Paper {paper_id: row.paper_id})
    SET p.dataset = row.dataset,
        p.source_file = row.source_file,
        p.title = row.title,
        p.abstract = row.abstract,
        p.url = row.url,
        p.year_published = row.year_published,
        p.first_author = row.first_author,
        p.scholarly_citations_count = row.scholarly_citations_count
    """
    with driver.session(database=database) as session:
        for batch in chunked(rows):
            session.run(query, rows=batch)


def insert_fields(driver, database: str, rows: list[dict[str, Any]]):
    query = """
    UNWIND $rows AS row
    MATCH (p:Paper {paper_id: row.paper_id})
    MERGE (f:Field {name: row.field})
    MERGE (p)-[:HAS_FIELD]->(f)
    """
    with driver.session(database=database) as session:
        for batch in chunked(rows):
            session.run(query, rows=batch)


def insert_mentions(driver, database: str, rows: list[dict[str, Any]]):
    query = """
    UNWIND $rows AS row
    MATCH (p:Paper {paper_id: row.paper_id})
    MERGE (e:Entity {name_norm: row.entity_norm})
      ON CREATE SET e.name = row.entity_name
      ON MATCH SET e.name = coalesce(e.name, row.entity_name)
    MERGE (p)-[r:MENTIONS]->(e)
    SET r.count = row.count
    """
    with driver.session(database=database) as session:
        for batch in chunked(rows):
            session.run(query, rows=batch)


def insert_entity_categories(driver, database: str, rows: list[dict[str, Any]]):
    query = """
    UNWIND $rows AS row
    MATCH (e:Entity {name_norm: row.entity_norm})
    MERGE (c:Category {name: row.category})
    MERGE (e)-[:IN_CATEGORY]->(c)
    """
    with driver.session(database=database) as session:
        for batch in chunked(rows):
            session.run(query, rows=batch)


def insert_co_occurrences(driver, database: str, rows: list[dict[str, Any]]):
    query = """
    UNWIND $rows AS row
    MATCH (left:Entity {name_norm: row.left})
    MATCH (right:Entity {name_norm: row.right})
    MERGE (left)-[r:CO_OCCURS_WITH]-(right)
    SET r.weight = row.weight,
        r.papers = row.papers
    """
    with driver.session(database=database) as session:
        for batch in chunked(rows):
            session.run(query, rows=batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Load Fusion NER JSON data into Neo4j.")
    parser.add_argument("--uri", required=True, help="Neo4j URI, e.g. bolt://localhost:7687")
    parser.add_argument("--user", required=True, help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument("--data-dir", default="data", help="Directory containing *_NER.json files")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick test runs")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all existing nodes/relationships in the target database before import",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    papers, fields, mentions, entity_categories, co_occurrences = extract_graph_rows(
        data_dir=data_dir,
        max_files=args.max_files,
    )

    if not papers:
        raise RuntimeError(f"No records found in {data_dir} (expected *_NER.json with a 'data' list).")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        if args.wipe:
            wipe_graph(driver, args.database)

        ensure_schema(driver, args.database)
        insert_papers(driver, args.database, papers)
        if fields:
            insert_fields(driver, args.database, fields)
        if mentions:
            insert_mentions(driver, args.database, mentions)
        if entity_categories:
            insert_entity_categories(driver, args.database, entity_categories)
        if co_occurrences:
            insert_co_occurrences(driver, args.database, co_occurrences)

        print("Import complete")
        print(f"  Papers: {len(papers)}")
        print(f"  Field links: {len(fields)}")
        print(f"  Mentions: {len(mentions)}")
        print(f"  Entity-category links: {len(entity_categories)}")
        print(f"  Co-occurrence pairs: {len(co_occurrences)}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
