"""Long-term Memory Graph for the Fusion KG Chat Agent
======================================================
Persists conversation history, learned facts, and explored topics as a
sub-graph inside the same Neo4j database.  Memory nodes are kept separate
from the main KG entities via distinct labels.

Node types
----------
  MemorySession   One per chat session.
  Memory          An individual remembered item (Q/A pair, fact, note).
  MemoryFact      A structured claim extracted from an answer
                  (subject, predicate, object, confidence).

Relationship types
------------------
  (Memory)-[:IN_SESSION]->(MemorySession)
  (Memory)-[:MENTIONS]->(Entity)        # links to main KG entities
  (MemoryFact)-[:DERIVED_FROM]->(Memory)

Embedding
---------
When ``sentence-transformers`` is available the module stores compact 384-d
embeddings on Memory nodes so semantic recall works even without exact matches.

Usage
-----
    from analysis.memory_graph import MemoryGraph
    mem = MemoryGraph()

    sid = mem.new_session(topic="plasma confinement")

    mem.remember(
        session_id=sid,
        question="How does plasma confinement work in a stellarator?",
        answer="Stellarators use twisted magnetic coils...",
        entities=["stellarator", "plasma confinement", "magnetic field"],
        coverage_score=0.82,
    )

    past = mem.recall(query="stellarator coils", top_k=3)
    for m in past:
        print(m["question"], "→", m["answer"][:80])

CLI
---
    python -m analysis.memory_graph --recall "plasma temperature"
    python -m analysis.memory_graph --stats
    python -m analysis.memory_graph --export output/memory_export.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Schema init queries ───────────────────────────────────────────────────────

_CONSTRAINTS = [
    "CREATE CONSTRAINT memory_session_id IF NOT EXISTS FOR (s:MemorySession) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT memory_fact_id IF NOT EXISTS FOR (f:MemoryFact) REQUIRE f.id IS UNIQUE",
]

_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX memory_text IF NOT EXISTS
FOR (m:Memory) ON EACH [m.question, m.answer, m.topic]
"""

# ── Cypher queries ────────────────────────────────────────────────────────────

_UPSERT_SESSION = """
MERGE (s:MemorySession {id: $id})
ON CREATE SET s.topic = $topic,
              s.created_at = $ts,
              s.turn_count = 0
ON MATCH  SET s.last_active = $ts
RETURN s.id AS id
"""

_INSERT_MEMORY = """
CREATE (m:Memory {
  id:             $id,
  question:       $question,
  answer:         $answer,
  topic:          $topic,
  coverage_score: $coverage_score,
  entities_json:  $entities_json,
  timestamp:      $timestamp,
  session_id:     $session_id
})
WITH m
MATCH (s:MemorySession {id: $session_id})
CREATE (m)-[:IN_SESSION]->(s)
SET s.turn_count = coalesce(s.turn_count, 0) + 1
RETURN m.id AS id
"""

_LINK_ENTITIES = """
UNWIND $names AS name
MATCH (e:Entity)
WHERE toLower(e.name) = toLower(name) OR toLower(e.name_norm) = toLower(name)
WITH e LIMIT 1
MATCH (m:Memory {id: $memory_id})
MERGE (m)-[:MENTIONS]->(e)
"""

_FULLTEXT_RECALL = """
CALL db.index.fulltext.queryNodes('memory_text', $query)
YIELD node, score
WHERE node:Memory
RETURN node.id AS id,
       node.question AS question,
       node.answer AS answer,
       node.topic AS topic,
       node.coverage_score AS coverage_score,
       node.timestamp AS timestamp,
       node.session_id AS session_id,
       score
ORDER BY score DESC
LIMIT $top_k
"""

_RECENT_MEMORIES = """
MATCH (m:Memory)
OPTIONAL MATCH (m)-[:IN_SESSION]->(s:MemorySession)
RETURN m.id AS id,
       m.question AS question,
       m.answer AS answer,
       m.topic AS topic,
       m.coverage_score AS coverage_score,
       m.timestamp AS timestamp,
       m.session_id AS session_id
ORDER BY m.timestamp DESC
LIMIT $top_k
"""

_SESSION_MEMORIES = """
MATCH (m:Memory)-[:IN_SESSION]->(s:MemorySession {id: $session_id})
RETURN m.id AS id,
       m.question AS question,
       m.answer AS answer,
       m.coverage_score AS coverage_score,
       m.timestamp AS timestamp
ORDER BY m.timestamp ASC
"""

_UPSERT_FACT = """
MERGE (f:MemoryFact {id: $id})
ON CREATE SET f.subject    = $subject,
              f.predicate  = $predicate,
              f.object     = $object,
              f.confidence = $confidence,
              f.timestamp  = $timestamp
WITH f
MATCH (m:Memory {id: $memory_id})
MERGE (f)-[:DERIVED_FROM]->(m)
"""

_STATS = """
MATCH (s:MemorySession)
WITH count(s) AS sessions
MATCH (m:Memory)
WITH sessions, count(m) AS memories
OPTIONAL MATCH (f:MemoryFact)
RETURN sessions, memories, count(f) AS facts
"""


# ── MemoryGraph ───────────────────────────────────────────────────────────────

class MemoryGraph:
    """Long-term memory backed by Neo4j.

    Parameters
    ----------
    driver
        Optional neo4j.GraphDatabase.driver instance.  If not provided,
        one is created using ``NEO4J_URI`` / ``NEO4J_USERNAME`` /
        ``NEO4J_PASSWORD`` environment variables (same as the rest of the app).
    enable_embeddings
        Whether to attempt to compute and store sentence embeddings.
        Requires ``sentence-transformers``; silently disabled if not installed.
    """

    def __init__(self, driver=None, enable_embeddings: bool = True):
        if driver is None:
            driver = self._default_driver()
        self._driver = driver
        self._embedder = None
        if enable_embeddings:
            self._init_embedder()
        self._init_schema()

    # ── Setup ──────────────────────────────────────────────────────────────

    @staticmethod
    def _default_driver():
        import os
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "password")
        return GraphDatabase.driver(uri, auth=(user, pwd))

    def _init_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("Memory embedder loaded (all-MiniLM-L6-v2)")
        except Exception as exc:
            logger.debug("sentence-transformers not available for memory: %s", exc)
            self._embedder = None

    def _init_schema(self) -> None:
        with self._driver.session() as session:
            for stmt in _CONSTRAINTS:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug("Constraint already exists or failed: %s", exc)
            try:
                session.run(_FULLTEXT_INDEX)
            except Exception as exc:
                logger.debug("Fulltext index already exists or failed: %s", exc)

    # ── Session management ─────────────────────────────────────────────────

    def new_session(self, topic: str = "general") -> str:
        """Create a new MemorySession and return its ID."""
        ts = datetime.now(timezone.utc).isoformat()
        sid = hashlib.sha1(f"{topic}-{ts}".encode()).hexdigest()[:16]
        with self._driver.session() as session:
            session.run(_UPSERT_SESSION, id=sid, topic=topic, ts=ts)
        logger.debug("New memory session: %s (topic=%s)", sid, topic)
        return sid

    # ── Remember ───────────────────────────────────────────────────────────

    def remember(
        self,
        session_id: str,
        question: str,
        answer: str,
        entities: list[str] | None = None,
        topic: str | None = None,
        coverage_score: float = 0.0,
        facts: list[dict] | None = None,
    ) -> str:
        """Store a Q/A turn in the memory graph.

        Parameters
        ----------
        session_id
            ID from ``new_session()``.
        question, answer
            The turn content.
        entities
            KG entity names mentioned in the answer.
        topic
            Optional topic label (defaults to first 60 chars of question).
        coverage_score
            Self-reported quality (0–1) from the agent.
        facts
            Optional list of ``{subject, predicate, object, confidence}`` dicts
            to store as MemoryFact nodes.

        Returns
        -------
        str
            The memory node ID.
        """
        ts = datetime.now(timezone.utc).isoformat()
        topic = topic or question[:60]
        mem_id = hashlib.sha1(f"{session_id}-{ts}-{question[:40]}".encode()).hexdigest()[:20]

        with self._driver.session() as db_session:
            db_session.run(
                _INSERT_MEMORY,
                id=mem_id,
                question=question,
                answer=answer,
                topic=topic,
                coverage_score=coverage_score,
                entities_json=json.dumps(entities or []),
                timestamp=ts,
                session_id=session_id,
            )

            if entities:
                db_session.run(_LINK_ENTITIES, names=entities, memory_id=mem_id)

            for fact in (facts or []):
                fid = hashlib.sha1(
                    f"{mem_id}-{fact.get('predicate','')}-{fact.get('object','')}".encode()
                ).hexdigest()[:20]
                db_session.run(
                    _UPSERT_FACT,
                    id=fid,
                    subject=fact.get("subject", ""),
                    predicate=fact.get("predicate", ""),
                    object=fact.get("object", ""),
                    confidence=fact.get("confidence", 0.5),
                    timestamp=ts,
                    memory_id=mem_id,
                )

        logger.debug("Remembered turn %s (session=%s)", mem_id, session_id)
        return mem_id

    # ── Recall ─────────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve past memories relevant to *query* via fulltext search.

        Falls back to recency-ordered recall when the fulltext index is not
        available or returns no results.
        """
        rows: list[dict] = []

        # 1. Fulltext search
        try:
            with self._driver.session() as session:
                result = session.run(_FULLTEXT_RECALL, query=query, top_k=top_k)
                rows = [dict(r) for r in result]
        except Exception as exc:
            logger.debug("Fulltext memory recall failed: %s", exc)

        # 2. Embedding-based re-ranking (if available and >1 result)
        if self._embedder and len(rows) > 1:
            rows = self._rerank_by_embedding(query, rows)

        # 3. Fallback to recency
        if not rows:
            try:
                with self._driver.session() as session:
                    result = session.run(_RECENT_MEMORIES, top_k=top_k)
                    rows = [dict(r) for r in result]
            except Exception as exc:
                logger.debug("Recency fallback failed: %s", exc)

        return rows[:top_k]

    def recall_session(self, session_id: str) -> list[dict]:
        """Return all memories from a specific session (chronological)."""
        with self._driver.session() as session:
            result = session.run(_SESSION_MEMORIES, session_id=session_id)
            return [dict(r) for r in result]

    def _rerank_by_embedding(self, query: str, rows: list[dict]) -> list[dict]:
        """Re-rank *rows* by cosine similarity to the query embedding."""
        import numpy as np
        try:
            texts = [f"{r.get('question','')} {r.get('answer',''[:100])}" for r in rows]
            embs = self._embedder.encode([query] + texts, normalize_embeddings=True)
            q_emb = embs[0]
            sims = embs[1:] @ q_emb
            order = np.argsort(-sims)
            return [rows[i] for i in order]
        except Exception as exc:
            logger.debug("Embedding re-rank failed: %s", exc)
            return rows

    # ── Stats & export ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return counts of sessions, memories, and facts."""
        with self._driver.session() as session:
            result = session.run(_STATS)
            row = result.single()
            if row:
                return dict(row)
        return {"sessions": 0, "memories": 0, "facts": 0}

    def export(self, path: str | Path) -> None:
        """Dump all Memory nodes to a JSON file for offline inspection."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (m:Memory) RETURN m {.*} AS props ORDER BY m.timestamp ASC"
            )
            records = [dict(r["props"]) for r in result]
        Path(path).write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("Exported %d memory records → %s", len(records), path)

    def close(self) -> None:
        self._driver.close()


# ── Fact extraction helper ────────────────────────────────────────────────────

def extract_facts_from_answer(llm, question: str, answer: str) -> list[dict]:
    """Ask the LLM to extract structured (subject, predicate, object) triples.

    Returns a list of dicts with keys: subject, predicate, object, confidence.
    Used to populate MemoryFact nodes with richer semantics than co-occurrences.
    """
    prompt = (
        "Extract up to 5 factual claims from the following fusion-energy answer as "
        "JSON triples. Each triple must have:\n"
        "  subject   – the entity making or described by the claim\n"
        "  predicate – a short verb phrase (e.g. 'achieves', 'requires', 'produces')\n"
        "  object    – what the subject relates to\n"
        "  confidence – your confidence 0.0–1.0\n\n"
        "Return ONLY a JSON array of objects, no prose.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer[:800]}\n\n"
        "Facts (JSON array):"
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        # Strip markdown fences
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = raw.strip().strip('`')
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [
                {k: f.get(k, "") for k in ("subject", "predicate", "object", "confidence")}
                for f in facts if isinstance(f, dict)
            ][:5]
    except Exception as exc:
        logger.debug("Fact extraction failed: %s", exc)
    return []


import re  # noqa: E402 – needed by extract_facts_from_answer


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> None:
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(prog="analysis.memory_graph")
    parser.add_argument("--recall", metavar="QUERY", help="Recall memories matching QUERY")
    parser.add_argument("--stats", action="store_true", help="Print memory graph statistics")
    parser.add_argument("--export", metavar="PATH", help="Export all memories to JSON file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    mem = MemoryGraph()

    if args.stats:
        s = mem.stats()
        print(f"Memory graph: {s['sessions']} sessions | {s['memories']} memories | {s['facts']} facts")

    if args.recall:
        rows = mem.recall(args.recall, top_k=args.top_k)
        if not rows:
            print("No memories found.")
        for r in rows:
            print(f"\n[{r.get('timestamp','?')[:19]}] {r.get('question','?')}")
            print(f"  → {str(r.get('answer',''))[:120]}")
            print(f"     coverage={r.get('coverage_score',0):.2f}")

    if args.export:
        mem.export(args.export)
        print(f"Exported to {args.export}")

    mem.close()


if __name__ == "__main__":
    _main()
