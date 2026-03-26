"""
LLM-powered GraphCypher QA for the Fusion Knowledge Graph
==========================================================
Pipeline:
  1. Semantic entity linking  — sentence-transformers finds exact KG entity names
  2. Cypher generation        — LLM writes query using injected entity names
  3. Neo4j execution          — via LangChain GraphCypherQAChain
  4. Fallback                 — direct lookup when LLM Cypher returns 0 rows
  5. Answer synthesis         — LLM writes natural language answer from results

Requires:
  - Ollama running locally (ollama serve)  → default model: nemotron-3-nano:4b
  - langchain-neo4j, langchain-ollama, langchain-core packages
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import ChatOllama

load_dotenv()

logger = logging.getLogger(__name__)

# ── Cypher generation prompt ───────────────────────────────────────────────
# Variables: {schema} (auto-injected by LangChain), {question}
# The question string may be pre-augmented with [Verified KG entities: ...] hints.

CYPHER_GENERATION_TEMPLATE = """\
You are a Neo4j Cypher expert for a Fusion Energy Knowledge Graph.
Generate ONLY valid Cypher. No markdown fences, no explanation, no comments.

=== GRAPH SCHEMA ===
{schema}

=== NODE LABELS ===
(:Entity   {{name_norm: STRING}})   — physics/engineering concepts; all values are LOWERCASE
(:Paper    {{title: STRING, year_published: INTEGER, abstract: STRING}})
(:Category {{name: STRING}})        — concept category buckets
(:Field    {{name: STRING}})        — raw NER outputs per paper

=== RELATIONSHIP TYPES ===
(e1:Entity)-[:CO_OCCURS_WITH {{weight: INTEGER}}]-(e2:Entity)
(p:Paper)-[:MENTIONS {{count: INTEGER}}]->(e:Entity)
(e:Entity)-[:IN_CATEGORY]->(c:Category)
(p:Paper)-[:HAS_FIELD]->(f:Field)

=== CATEGORY VALUES (exact strings) ===
"Device Type", "Plasma Property", "Diagnostic Method",
"Material", "Process", "Organization", "Location"

=== ENTITY NAME NOTE ===
Entity.name_norm values are stored in LOWERCASE (e.g., "tokamak", "iter", "tritium").
Always match with:  e.name_norm CONTAINS toLower('search term')

=== SIX CYPHER PATTERNS — memorise and reuse ===

# 1 — Entity info + mention stats
MATCH (e:Entity)
WHERE e.name_norm CONTAINS toLower('tokamak')
OPTIONAL MATCH (e)-[:IN_CATEGORY]->(c:Category)
OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e)
WITH e.name_norm AS entity,
     collect(DISTINCT c.name) AS categories,
     count(DISTINCT p) AS papers,
     sum(m.count) AS mentions
RETURN entity, categories, papers, mentions
ORDER BY mentions DESC LIMIT 10

# 2 — Top co-occurrence neighbours
MATCH (e:Entity)-[r:CO_OCCURS_WITH]-(other:Entity)
WHERE e.name_norm CONTAINS toLower('plasma confinement')
OPTIONAL MATCH (other)-[:IN_CATEGORY]->(c:Category)
RETURN other.name_norm AS neighbour,
       r.weight AS weight,
       collect(DISTINCT c.name) AS categories
ORDER BY weight DESC LIMIT 20

# 3 — Trend over time (yearly mentions)
MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
WHERE e.name_norm CONTAINS toLower('tritium')
  AND p.year_published IS NOT NULL
RETURN p.year_published AS year, sum(m.count) AS mentions
ORDER BY year

# 4 — Shortest path between two concepts
MATCH (a:Entity), (b:Entity)
WHERE a.name_norm CONTAINS toLower('divertor')
  AND b.name_norm CONTAINS toLower('stellarator')
MATCH path = shortestPath((a)-[:CO_OCCURS_WITH*1..6]-(b))
RETURN [n IN nodes(path) | n.name_norm] AS path_nodes,
       [r IN relationships(path) | r.weight] AS weights,
       length(path) AS hops
LIMIT 3

# 5 — Top entities within a category
MATCH (e:Entity)-[:IN_CATEGORY]->(c:Category)
WHERE c.name = 'Diagnostic Method'
OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e)
WITH e.name_norm AS method,
     count(DISTINCT p) AS papers,
     sum(m.count) AS mentions
RETURN method, papers, mentions
ORDER BY mentions DESC LIMIT 15

# 6 — Papers mentioning two concepts together
MATCH (p:Paper)-[:MENTIONS]->(e1:Entity),
      (p)-[:MENTIONS]->(e2:Entity)
WHERE e1.name_norm CONTAINS toLower('tokamak')
  AND e2.name_norm CONTAINS toLower('tritium')
RETURN DISTINCT p.title AS title, p.year_published AS year
ORDER BY year DESC LIMIT 10

=== RULES ===
- ALWAYS use e.name_norm CONTAINS toLower('term') — NEVER exact equality for entity names
- ALWAYS add LIMIT (max 25 rows)
- For aggregations: use WITH + RETURN, not RETURN *
- For trends: group by year_published, ORDER BY year
- For paths: CO_OCCURS_WITH*1..6
- If the question asks for "most", "top", "most common" → ORDER BY + LIMIT

Question: {question}

Cypher:"""

# ── QA answer synthesis prompt ────────────────────────────────────────────

QA_TEMPLATE = """\
You are a scientific assistant specialising in nuclear fusion research.
Answer the question using ONLY the knowledge graph data below.
- Cite specific entity names, numbers, and years from the data.
- If the data is empty or insufficient, say so clearly.
- Structure longer answers with short bullet points.
- Answer in the same language as the question.

Knowledge graph results:
{context}

Question: {question}

Answer:"""


# ── Agent class ────────────────────────────────────────────────────────────

class FusionCypherAgent:
    """LLM-to-Cypher-to-Answer agent with semantic entity linking.

    Pipeline per question:
      1. EntityLinker finds verified entity name_norm strings from the KG.
      2. Those names are injected into the question before Cypher generation.
      3. GraphCypherQAChain generates + executes Cypher + synthesises answer.
      4. If Cypher returns 0 rows, a direct fallback lookup runs automatically.
    """

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        neo4j_db: str | None = None,
        ollama_url: str | None = None,
        model: str | None = None,
        use_entity_linker: bool = True,
    ) -> None:
        uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        pwd = neo4j_password or os.getenv("NEO4J_PASSWORD", "fusion2026")
        db = neo4j_db or os.getenv("NEO4J_DATABASE", "neo4j")
        url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        mdl = model or os.getenv("OLLAMA_MODEL", "nemotron-3-nano:4b")

        logger.info("FusionCypherAgent init | neo4j=%s  ollama=%s  model=%s", uri, url, mdl)

        self._db = db

        # LangChain Neo4j wrapper (for schema introspection + chain)
        self._graph = Neo4jGraph(
            url=uri, username=user, password=pwd,
            database=db, enhanced_schema=False,
        )

        # Raw driver for fallback direct queries
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

        # LLM
        self._llm = ChatOllama(base_url=url, model=mdl, temperature=0)

        # LangChain GraphCypherQAChain
        self._chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            cypher_prompt=PromptTemplate(
                input_variables=["schema", "question"],
                template=CYPHER_GENERATION_TEMPLATE,
            ),
            qa_prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=QA_TEMPLATE,
            ),
            verbose=False,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
        )

        # Semantic entity linker (built once, then cached to disk)
        self._linker = None
        if use_entity_linker:
            try:
                from analysis.entity_linker import EntityLinker
                logger.info("FusionCypherAgent: initialising entity linker…")
                self._linker = EntityLinker(self._driver)
                logger.info(
                    "FusionCypherAgent: entity linker ready (%d entities indexed)",
                    self._linker.entity_count,
                )
            except Exception as exc:
                logger.warning("EntityLinker init failed (non-fatal): %s", exc)

    # ── Public ───────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Ask a natural language question about the fusion knowledge graph.

        Parameters
        ----------
        question
            Free-text question in any language.
        history
            Previous conversation turns for follow-up context.
            Each entry: {"role": "user"|"assistant", "content": str}

        Returns
        -------
        dict with:
            answer          — LLM-synthesised natural language answer
            cypher          — Cypher generated by the LLM
            context         — raw Neo4j rows (list[dict])
            linked_entities — [(name_norm, score), ...] from entity linker
            fallback_used   — True if fallback direct query was triggered
            error           — error string or None
        """
        # Step 1: semantic entity linking
        linked: list[tuple[str, float]] = []
        if self._linker:
            try:
                linked = self._linker.link(question)
                logger.info("EntityLinker found: %s", [n for n, _ in linked])
            except Exception as exc:
                logger.warning("Entity linking error: %s", exc)

        # Step 2: build augmented question (entity hints + conversation history)
        augmented = _build_augmented_question(question, linked, history)

        # Step 3: run LangChain chain
        cypher, context, answer, error = "", [], "", None
        try:
            result = self._chain.invoke({"query": augmented})
            steps = result.get("intermediate_steps", [])
            cypher = steps[0].get("query", "").strip() if steps else ""
            context = steps[1].get("context", []) if len(steps) > 1 else []
            answer = result.get("result", "").strip()
            logger.info("Chain ok | cypher=%.80s | rows=%d", cypher, len(context))
        except Exception as exc:
            error = str(exc)
            logger.error("Chain error: %s", exc)

        # Step 4: fallback when Cypher returns nothing
        fallback_used = False
        if not error and not context and linked:
            logger.info("No rows returned — running fallback direct lookup")
            fallback_rows = self._fallback_lookup(linked)
            if fallback_rows:
                context = fallback_rows
                fallback_used = True
                # Re-synthesise answer from fallback rows
                try:
                    ctx_str = "\n".join(str(r) for r in fallback_rows[:15])
                    qa_text = QA_TEMPLATE.replace("{context}", ctx_str).replace("{question}", question)
                    resp = self._llm.invoke(qa_text)
                    answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
                except Exception as exc2:
                    logger.warning("Fallback answer synthesis failed: %s", exc2)

        return {
            "answer": answer,
            "cypher": cypher,
            "context": context,
            "linked_entities": linked,
            "fallback_used": fallback_used,
            "error": error,
        }

    def schema_summary(self) -> str:
        """Return the Neo4j schema string loaded by LangChain."""
        return getattr(self._graph, "schema", None) or "(schema not loaded)"

    # ── Fallback ─────────────────────────────────────────────────────────

    def _fallback_lookup(
        self,
        linked: list[tuple[str, float]],
        top_n: int = 4,
    ) -> list[dict]:
        """Direct entity + neighbour lookup when LLM Cypher returns nothing."""
        names = [name for name, _ in linked[:top_n]]
        cypher = """
            MATCH (e:Entity)
            WHERE ANY(n IN $names WHERE e.name_norm CONTAINS n)
            OPTIONAL MATCH (e)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (:Paper)-[m:MENTIONS]->(e)
            WITH e.name_norm AS entity,
                 collect(DISTINCT c.name) AS categories,
                 sum(m.count) AS mentions
            RETURN entity, categories, mentions
            ORDER BY mentions DESC LIMIT 15
        """
        try:
            with self._driver.session(database=self._db) as sess:
                return [dict(r) for r in sess.run(cypher, names=names)]
        except Exception as exc:
            logger.warning("Fallback lookup failed: %s", exc)
            return []


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_augmented_question(
    question: str,
    linked: list[tuple[str, float]],
    history: list[dict[str, str]] | None,
    max_turns: int = 3,
) -> str:
    """Build the full question string passed to the Cypher generation prompt.

    Injects:
    - Verified entity names from semantic linking (highest-confidence first)
    - Recent conversation turns for follow-up coreference resolution
    """
    parts: list[str] = []

    # Entity hints — only high-confidence matches
    strong = [(n, s) for n, s in linked if s >= 0.35]
    if strong:
        names_str = ", ".join(f'"{n}"' for n, _ in strong)
        parts.append(
            f"[Verified entity names in this knowledge graph: {names_str}]\n"
            f"[Use these exact values in your CONTAINS clauses.]\n"
        )

    # Conversation history
    if history:
        recent: list[str] = []
        for turn in history[-(max_turns * 2):]:
            role = turn.get("role", "")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                recent.append(f"User: {content}")
            elif role == "assistant":
                # Truncate long answers so we don't bloat the prompt
                recent.append(f"Assistant: {content[:350]}")
        if recent:
            parts.append("Previous conversation:\n" + "\n".join(recent) + "\n")

    parts.append(f"Question: {question}")
    return "\n".join(parts)
