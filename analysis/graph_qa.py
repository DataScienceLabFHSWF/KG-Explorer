"""
Graph Query Agent for the Fusion Knowledge Graph
==================================================
Provides structured query capabilities over the fusion KG via Neo4j Cypher.
This is the foundation for a hybrid GraphRAG system — it generates precise
subgraph answers that can later be combined with vector search + LLM synthesis.

Query strategies
----------------
1. **Entity lookup**     — properties, categories, mention count
2. **Neighbourhood**     — direct co-occurring entities
3. **Path search**       — shortest paths between two concepts
4. **Bridge concepts**   — entities connecting disjoint communities
5. **Trend query**       — entity mention frequency by year
6. **Community context** — entities in the same Louvain community
7. **Gap-aware**         — query the gap report for research opportunities

Usage
-----
    python -m analysis.graph_qa "What connects tokamak and stellarator?"
    python -m analysis.graph_qa --mode bridge "plasma"

Or as a library:

    from analysis.graph_qa import GraphQA
    qa = GraphQA(driver)
    result = qa.query("tokamak")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.neo4j_utils import get_driver, get_database, OUTPUT_DIR


@dataclass
class QueryResult:
    """Structured result from a graph query."""
    query: str
    mode: str
    data: list[dict[str, Any]]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "data": self.data,
            "summary": self.summary,
        }

    def __str__(self) -> str:
        lines = [f"[{self.mode}] {self.query}", f"  {self.summary}"]
        for item in self.data[:10]:
            lines.append(f"  - {item}")
        if len(self.data) > 10:
            lines.append(f"  … and {len(self.data) - 10} more")
        return "\n".join(lines)


class GraphQA:
    """Query interface for the fusion knowledge graph.

    Parameters
    ----------
    driver
        Neo4j driver instance.
    database
        Neo4j database name (defaults to env).
    """

    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.db = database or get_database()

    def _run(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self.driver.session(database=self.db) as session:
            result = session.run(cypher, **(params or {}))
            return [dict(r) for r in result]

    # ── 1. Entity lookup ──────────────────────────────────────────

    def entity_lookup(self, name: str) -> QueryResult:
        """Look up an entity by name (case-insensitive, fuzzy)."""
        # Try exact match first, then CONTAINS
        rows = self._run("""
            MATCH (e:Entity)
            WHERE toLower(e.name_norm) = toLower($name)
            OPTIONAL MATCH (e)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e)
            WITH e, collect(DISTINCT c.name) AS categories,
                 count(DISTINCT p) AS paper_count,
                 sum(m.count) AS total_mentions
            RETURN e.name_norm AS entity, categories, paper_count, total_mentions
        """, {"name": name})

        if not rows:
            rows = self._run("""
                MATCH (e:Entity)
                WHERE toLower(e.name_norm) CONTAINS toLower($name)
                OPTIONAL MATCH (e)-[:IN_CATEGORY]->(c:Category)
                OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e)
                WITH e, collect(DISTINCT c.name) AS categories,
                     count(DISTINCT p) AS paper_count,
                     sum(m.count) AS total_mentions
                RETURN e.name_norm AS entity, categories, paper_count, total_mentions
                ORDER BY total_mentions DESC
                LIMIT 10
            """, {"name": name})

        summary = f"Found {len(rows)} matching entities"
        return QueryResult(query=name, mode="entity_lookup", data=rows,
                           summary=summary)

    # ── 2. Neighbourhood ─────────────────────────────────────────

    def neighbours(self, name: str, top_n: int = 20) -> QueryResult:
        """Find top co-occurring entities."""
        rows = self._run("""
            MATCH (e:Entity)-[r:CO_OCCURS_WITH]-(other:Entity)
            WHERE toLower(e.name_norm) = toLower($name)
            OPTIONAL MATCH (other)-[:IN_CATEGORY]->(c:Category)
            WITH other.name_norm AS neighbour,
                 r.weight AS weight,
                 collect(DISTINCT c.name) AS categories
            RETURN neighbour, weight, categories
            ORDER BY weight DESC
            LIMIT $top_n
        """, {"name": name, "top_n": top_n})

        summary = f"{len(rows)} neighbours of '{name}'"
        return QueryResult(query=name, mode="neighbours", data=rows,
                           summary=summary)

    # ── 3. Path search ────────────────────────────────────────────

    def shortest_path(self, source: str, target: str,
                      max_hops: int = 5) -> QueryResult:
        """Find the shortest co-occurrence path between two entities."""
        rows = self._run("""
            MATCH (a:Entity), (b:Entity)
            WHERE toLower(a.name_norm) = toLower($src)
              AND toLower(b.name_norm) = toLower($tgt)
            MATCH path = shortestPath((a)-[:CO_OCCURS_WITH*1..5]-(b))
            RETURN [n IN nodes(path) | n.name_norm] AS path_nodes,
                   length(path) AS hops,
                   [r IN relationships(path) | r.weight] AS weights
            LIMIT 3
        """, {"src": source, "tgt": target})

        if rows:
            path_str = " -> ".join(rows[0]["path_nodes"])
            summary = f"Path ({rows[0]['hops']} hops): {path_str}"
        else:
            summary = f"No path found between '{source}' and '{target}'"

        return QueryResult(query=f"{source} -> {target}", mode="shortest_path",
                           data=rows, summary=summary)

    # ── 4. Bridge concepts ────────────────────────────────────────

    def bridge_concepts(self, category_a: str, category_b: str,
                        top_n: int = 10) -> QueryResult:
        """Find entities that bridge two categories."""
        rows = self._run("""
            MATCH (a:Entity)-[:IN_CATEGORY]->(ca:Category {name: $cat_a})
            MATCH (b:Entity)-[:IN_CATEGORY]->(cb:Category {name: $cat_b})
            MATCH (a)-[r1:CO_OCCURS_WITH]-(bridge:Entity)-[r2:CO_OCCURS_WITH]-(b)
            WHERE a <> b AND bridge <> a AND bridge <> b
            OPTIONAL MATCH (bridge)-[:IN_CATEGORY]->(bc:Category)
            WITH bridge.name_norm AS bridge_entity,
                 collect(DISTINCT bc.name) AS bridge_categories,
                 count(DISTINCT a) + count(DISTINCT b) AS connections,
                 sum(r1.weight) + sum(r2.weight) AS total_weight
            RETURN bridge_entity, bridge_categories, connections, total_weight
            ORDER BY total_weight DESC
            LIMIT $top_n
        """, {"cat_a": category_a, "cat_b": category_b, "top_n": top_n})

        summary = f"{len(rows)} bridge concepts between '{category_a}' and '{category_b}'"
        return QueryResult(query=f"{category_a} <-> {category_b}",
                           mode="bridge_concepts", data=rows, summary=summary)

    # ── 5. Trend query ────────────────────────────────────────────

    def entity_trend(self, name: str) -> QueryResult:
        """Get yearly mention frequency for an entity."""
        rows = self._run("""
            MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
            WHERE toLower(e.name_norm) = toLower($name)
              AND p.year_published IS NOT NULL
            RETURN p.year_published AS year, sum(m.count) AS mentions
            ORDER BY year
        """, {"name": name})

        if rows:
            years = [r["year"] for r in rows]
            summary = f"'{name}' mentioned across {len(rows)} years ({min(years)}-{max(years)})"
        else:
            summary = f"No temporal data for '{name}'"

        return QueryResult(query=name, mode="entity_trend", data=rows,
                           summary=summary)

    # ── 6. Community context ──────────────────────────────────────

    def community_members(self, name: str, top_n: int = 20) -> QueryResult:
        """Find entities in the same community (from communities.csv)."""
        csv_path = OUTPUT_DIR / "communities.csv"
        if not csv_path.exists():
            return QueryResult(
                query=name, mode="community",
                data=[],
                summary="communities.csv not found — run graph analysis first",
            )

        import pandas as pd
        df = pd.read_csv(csv_path)

        # Find target community
        name_lower = name.lower()
        match = df[df["entity"].str.lower() == name_lower]
        if match.empty:
            match = df[df["entity"].str.lower().str.contains(name_lower, na=False)]

        if match.empty:
            return QueryResult(query=name, mode="community", data=[],
                               summary=f"Entity '{name}' not found in communities")

        community_id = match.iloc[0]["community"]
        members = df[df["community"] == community_id].head(top_n)

        rows = members.to_dict("records")
        summary = (f"Community {community_id}: {len(df[df['community'] == community_id])} "
                   f"members (showing top {len(rows)})")

        return QueryResult(query=name, mode="community", data=rows,
                           summary=summary)

    # ── 7. Gap-aware query ────────────────────────────────────────

    def research_gaps(self, entity: str | None = None) -> QueryResult:
        """Query the gap report for research opportunities."""
        gap_path = OUTPUT_DIR / "gap_report.json"
        if not gap_path.exists():
            return QueryResult(
                query=entity or "all", mode="gaps", data=[],
                summary="gap_report.json not found — run gap analysis first",
            )

        with open(gap_path) as f:
            report = json.load(f)

        results = []

        # Hypotheses (main content of the gap report)
        for hyp in report.get("hypotheses", []):
            if entity:
                entity_lower = entity.lower()
                entities_str = " ".join(hyp.get("entities", [])).lower()
                hyp_str = hyp.get("hypothesis", "").lower()
                if entity_lower not in entities_str and entity_lower not in hyp_str:
                    continue
            results.append(hyp)

        summary = f"{len(results)} research gaps" + (f" related to '{entity}'" if entity else "")
        return QueryResult(query=entity or "all", mode="gaps", data=results,
                           summary=summary)

    # ── Auto-dispatch ─────────────────────────────────────────────

    def query(self, text: str) -> QueryResult:
        """Parse a natural-language-ish query and dispatch to the right method.

        Heuristics:
          - "A -> B" or "path from A to B" → shortest_path
          - "bridge A B" → bridge_concepts
          - "trend X" or "over time" → entity_trend
          - "community X" or "cluster" → community_members
          - "gaps" or "opportunities" → research_gaps
          - "neighbours of X" → neighbours
          - plain entity name → entity_lookup
        """
        text = text.strip()

        # Path query: "A -> B" or "path from A to B"
        path_match = re.match(r"(.+?)\s*->\s*(.+)", text)
        if path_match:
            return self.shortest_path(path_match.group(1).strip(),
                                      path_match.group(2).strip())

        text_lower = text.lower()

        if text_lower.startswith("path "):
            parts = re.split(r"\s+(?:to|and)\s+", text[5:], maxsplit=1)
            if len(parts) == 2:
                return self.shortest_path(parts[0].strip(), parts[1].strip())

        if text_lower.startswith("bridge "):
            parts = text[7:].split(",", 1)
            if len(parts) == 2:
                return self.bridge_concepts(parts[0].strip(), parts[1].strip())

        if "trend" in text_lower or "over time" in text_lower:
            entity = re.sub(r"(trend|over time|of)\s*", "", text, flags=re.I).strip()
            return self.entity_trend(entity)

        if "communit" in text_lower or "cluster" in text_lower:
            entity = re.sub(r"(community|cluster|of|in)\s*", "", text, flags=re.I).strip()
            return self.community_members(entity)

        if "gap" in text_lower or "opportunit" in text_lower:
            entity = re.sub(r"\b(gaps?|opportunit\w+|research|for)\b\s*", "", text, flags=re.I).strip()
            return self.research_gaps(entity or None)

        if "neighbour" in text_lower or "neighbor" in text_lower or "co-occur" in text_lower:
            entity = re.sub(r"(neighbours?|neighbors?|co-occur\w*|of)\s*", "", text, flags=re.I).strip()
            return self.neighbours(entity)

        # Default: entity lookup
        return self.entity_lookup(text)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query the Fusion Knowledge Graph")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--mode", choices=[
        "lookup", "neighbours", "path", "bridge", "trend", "community", "gaps"
    ], default=None, help="Force a specific query mode")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    driver = get_driver()
    qa = GraphQA(driver)

    try:
        if args.mode == "lookup":
            result = qa.entity_lookup(query_text)
        elif args.mode == "neighbours":
            result = qa.neighbours(query_text)
        elif args.mode == "path":
            parts = query_text.split(",", 1)
            result = qa.shortest_path(parts[0].strip(),
                                      parts[1].strip() if len(parts) > 1 else "")
        elif args.mode == "bridge":
            parts = query_text.split(",", 1)
            result = qa.bridge_concepts(parts[0].strip(),
                                        parts[1].strip() if len(parts) > 1 else "")
        elif args.mode == "trend":
            result = qa.entity_trend(query_text)
        elif args.mode == "community":
            result = qa.community_members(query_text)
        elif args.mode == "gaps":
            result = qa.research_gaps(query_text)
        else:
            result = qa.query(query_text)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(result)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
