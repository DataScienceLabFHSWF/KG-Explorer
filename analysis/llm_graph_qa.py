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
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import ChatOllama

from analysis.answer_gap_logger import log_answer_event

load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel string that QA prompt asks the LLM to emit when grounded answer
# is impossible. The agent intercepts this and routes the event to the
# answer-gap logger so the gap-detection pipeline can later surface
# systematic blind spots in the KG.
MISSING_DATA_SENTINEL = "[MISSING_FROM_GRAPH]"

# ── Cypher generation prompt ───────────────────────────────────────────────
# Variables: {schema} (auto-injected by LangChain), {question}
# The question string may be pre-augmented with [Verified KG entities: ...] hints.

CYPHER_GENERATION_TEMPLATE = """\
You are a Neo4j Cypher expert for a Fusion Energy Knowledge Graph.
Generate ONLY valid **Cypher 5** for Neo4j. NO SQL keywords (no SELECT, FROM,
WHERE x AS y, GROUP BY, JOIN). No markdown fences, no explanation, no comments.

=== GRAPH SCHEMA ===
{schema}

=== NODE LABELS ===
(:Entity   {{name_norm: STRING}})   — physics/engineering concepts; all LOWERCASE
(:Paper    {{title: STRING, year_published: INTEGER, abstract: STRING,
             first_author: STRING, scholarly_citations_count: INTEGER}})
(:Category {{name: STRING}})        — 90 NER concept buckets
(:Field    {{name: STRING}})        — raw NER outputs per paper

=== RELATIONSHIP TYPES ===
(e1:Entity)-[:CO_OCCURS_WITH {{weight: INTEGER, papers: LIST}}]-(e2:Entity)
(p:Paper)-[:MENTIONS {{count: INTEGER}}]->(e:Entity)
(e:Entity)-[:IN_CATEGORY]->(c:Category)
(p:Paper)-[:HAS_FIELD]->(f:Field)

=== ACTUAL CATEGORY NAMES (use these strings exactly) ===
{categories}

=== ENTITY NAME NOTE ===
Entity.name_norm values are stored in LOWERCASE (e.g., "tokamak", "iter", "tritium").
ALWAYS match with:  e.name_norm CONTAINS toLower('search term')

=== EIGHT CYPHER PATTERNS — pick the one that best fits the question ===

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

# 5 — Top entities within a category (use category names from list above)
MATCH (e:Entity)-[:IN_CATEGORY]->(c:Category)
WHERE c.name = 'Nuclear Fusion Device Type'
OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(e)
WITH e.name_norm AS device,
     count(DISTINCT p) AS papers,
     sum(m.count) AS mentions
RETURN device, papers, mentions
ORDER BY mentions DESC LIMIT 15

# 6 — Latest / most-cited paper that mentions a concept
MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
WHERE e.name_norm CONTAINS toLower('iter')
  AND p.year_published IS NOT NULL
RETURN p.title AS title,
       p.year_published AS year,
       p.first_author AS author,
       p.scholarly_citations_count AS citations,
       left(p.abstract, 600) AS abstract_excerpt
ORDER BY p.year_published DESC, p.scholarly_citations_count DESC
LIMIT 5

# 7 — Definition / explanation: pull abstract excerpts that ground a concept
MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
WHERE e.name_norm CONTAINS toLower('lawson criterion')
  AND p.abstract IS NOT NULL
RETURN p.title AS title,
       p.year_published AS year,
       left(p.abstract, 800) AS abstract_excerpt,
       m.count AS local_mentions
ORDER BY m.count DESC, p.scholarly_citations_count DESC
LIMIT 5

# 8 — Free-text fulltext search across abstracts (true GraphRAG fallback)
CALL db.index.fulltext.queryNodes('paper_text', 'lawson criterion ignition')
YIELD node, score
RETURN node.title AS title,
       node.year_published AS year,
       left(node.abstract, 800) AS abstract_excerpt,
       score
ORDER BY score DESC LIMIT 5

=== CHOOSING A PATTERN ===
- "What is X?" / "Define X" / "Why X?" / "How does X work?" → pattern 7 (or 8 if no entity match)
- "Most/top/which X are most ..." → pattern 1, 2, or 5
- "Trend / over time / since 1990" → pattern 3
- "Connection / relate / link between A and B" → pattern 4
- "Latest / newest / most cited paper on X" → pattern 6
- Anything where the user wants prose understanding → ALWAYS prefer 7 or 8
  so abstract text reaches the answer LLM.

=== HARD RULES ===
- Cypher only. NEVER write SELECT, FROM, GROUP BY, or SQL subqueries.
- For "latest year" use:  ORDER BY p.year_published DESC LIMIT 1
  Do NOT write things like  WHERE p.year = MAX(...).
- ALWAYS use e.name_norm CONTAINS toLower('term') — never exact equality.
- ALWAYS add LIMIT (≤ 25 rows; ≤ 5 when returning abstracts).
- For aggregations: WITH + RETURN.
- For paths: CO_OCCURS_WITH*1..6.

Question: {question}

Cypher:"""

# ── QA answer synthesis prompt ────────────────────────────────────────────

QA_TEMPLATE = """\
You are a scientific assistant specialising in nuclear fusion research,
answering on top of a knowledge graph (KG) extracted by named entity recognition
from 8,358 fusion-energy papers (1958–2024). Two kinds of evidence reach you:

  (a) Structured KG rows  — entity statistics, neighbours, paths, communities.
  (b) Paper abstract excerpts — verbatim text from the underlying papers.

Always prefer (b) for definitions, mechanisms, motivations, and physics
reasoning. Use (a) for prevalence, trends, structural relations.

Writing rules
-------------
- Cite specific entity names, numbers, paper titles, and years from the
  evidence. Never invent facts that are not in the evidence.
- For "what / why / how" questions, build the explanation from the abstract
  excerpts. Quote short phrases inline using single quotes when useful.
- Distinguish observed evidence from background reasoning explicitly. If the
  abstracts cover (e.g.) the Lawson criterion, the D-T reaction-rate curve, the
  ICF / MCF distinction, or the tritium fuel cycle, explain it in plain
  language and ground it in the cited papers.
- If the evidence does NOT contain enough to answer (no abstract excerpts and
  only thin co-occurrence rows), state this explicitly and emit the literal
  marker {sentinel} on its own line at the end of your answer. The marker is
  used to log knowledge gaps; never include it when the answer is complete.
- Structure longer answers with short bullet points. Answer in the same
  language as the question.

Evidence
--------
{context}

Question: {question}

Answer:""".replace("{sentinel}", MISSING_DATA_SENTINEL)


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

        # Pull the actual category names from the graph and inject into the
        # Cypher prompt so the LLM never invents non-existent category labels.
        cypher_template_filled = CYPHER_GENERATION_TEMPLATE.replace(
            "{categories}", self._fetch_category_block()
        )

        # LangChain GraphCypherQAChain
        self._chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            cypher_prompt=PromptTemplate(
                input_variables=["schema", "question"],
                template=cypher_template_filled,
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

        # Step 2: GraphRAG abstract retrieval (always runs in parallel with
        # the LLM-generated Cypher). This guarantees the answer prompt sees
        # paper text whenever the question is answerable from the corpus.
        abstract_hits = self._fetch_abstract_context(question, linked)
        logger.info("GraphRAG retrieved %d abstract excerpts", len(abstract_hits))

        # Step 3: build augmented question (entity hints + conversation history)
        augmented = _build_augmented_question(question, linked, history)

        # Step 4: run LangChain chain (LLM-generated Cypher)
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

        # Step 5: fallback when Cypher returns nothing
        fallback_used = False
        if not error and not context and linked:
            logger.info("No rows returned — running fallback direct lookup")
            fallback_rows = self._fallback_lookup(linked)
            if fallback_rows:
                context = fallback_rows
                fallback_used = True

        # Step 6: final answer synthesis using BOTH structured rows and
        # abstract excerpts. We re-synthesise whenever GraphRAG produced
        # abstracts (because LangChain only saw the structured rows) or when
        # the fallback ran. This is the "real GraphRAG" merge step.
        if abstract_hits or fallback_used or not answer:
            try:
                merged = _format_evidence(context, abstract_hits)
                qa_text = (
                    QA_TEMPLATE
                    .replace("{context}", merged)
                    .replace("{question}", question)
                )
                resp = self._llm.invoke(qa_text)
                answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            except Exception as exc2:
                logger.warning("Final answer synthesis failed: %s", exc2)

        # Step 7: detect missing-data sentinel and log to the answer-gap store.
        missing_from_graph = MISSING_DATA_SENTINEL in (answer or "")
        if missing_from_graph:
            answer = answer.replace(MISSING_DATA_SENTINEL, "").strip()
        # Heuristic: also log if no abstracts AND no structured rows reached us.
        no_evidence = (not abstract_hits) and (not context)
        if missing_from_graph or no_evidence:
            try:
                log_answer_event(
                    question=question,
                    linked_entities=[n for n, _ in linked],
                    n_rows=len(context),
                    n_abstracts=len(abstract_hits),
                    cypher=cypher,
                    sentinel=missing_from_graph,
                )
            except Exception as exc3:
                logger.warning("Could not log answer-gap event: %s", exc3)

        return {
            "answer": answer,
            "cypher": cypher,
            "context": context,
            "abstracts": abstract_hits,
            "linked_entities": linked,
            "fallback_used": fallback_used,
            "missing_from_graph": missing_from_graph,
            "error": error,
        }

    def schema_summary(self) -> str:
        """Return the Neo4j schema string loaded by LangChain."""
        return getattr(self._graph, "schema", None) or "(schema not loaded)"

    # ── Category / abstract retrieval helpers ────────────────────────────

    def _fetch_category_block(self, top_n: int = 30) -> str:
        """Read the top-N category names from the live KG so the Cypher prompt
        always reflects what actually exists."""
        try:
            with self._driver.session(database=self._db) as sess:
                rows = list(sess.run(
                    "MATCH (c:Category)<-[:IN_CATEGORY]-(e) "
                    "RETURN c.name AS name, count(e) AS cnt "
                    "ORDER BY cnt DESC LIMIT $top_n",
                    top_n=top_n,
                ))
            names = [f'"{r["name"]}"  (n={r["cnt"]})' for r in rows]
            return "\n".join(f"  {n}" for n in names)
        except Exception as exc:
            logger.warning("Could not fetch categories from KG: %s", exc)
            return '  (failed to load \u2014 use any plausible NER category name)'

    def _fetch_abstract_context(
        self,
        question: str,
        linked: list[tuple[str, float]],
        per_entity: int = 2,
        fulltext_top: int = 3,
        max_chars: int = 700,
    ) -> list[dict]:
        """Real GraphRAG retrieval. Two strategies are merged:

        1. **Entity-anchored**: for each linked entity (top-3), fetch the few
           most-mentioning abstracts. Guarantees grounding when the linker
           found at least one entity.
        2. **Fulltext fallback**: if no entities link, run the question through
           the ``paper_text`` Lucene index for free-text retrieval.

        Returned dicts have keys ``title``, ``year``, ``excerpt``, ``source``.
        """
        out: list[dict] = []
        seen_titles: set[str] = set()

        # 1) Entity-anchored
        if linked:
            try:
                names = [n for n, _ in linked[:3]]
                with self._driver.session(database=self._db) as sess:
                    rows = sess.run(
                        """
                        UNWIND $names AS qname
                        MATCH (e:Entity) WHERE e.name_norm = qname
                        MATCH (p:Paper)-[m:MENTIONS]->(e)
                        WHERE p.abstract IS NOT NULL
                        WITH qname, p, m
                        ORDER BY m.count DESC,
                                 coalesce(p.scholarly_citations_count, 0) DESC
                        WITH qname, collect({title: p.title, year: p.year_published,
                                             abstract: p.abstract, mentions: m.count})[..$per_entity] AS docs
                        UNWIND docs AS d
                        RETURN qname AS entity, d.title AS title, d.year AS year,
                               d.abstract AS abstract, d.mentions AS mentions
                        """,
                        names=names, per_entity=per_entity,
                    )
                    for r in rows:
                        title = (r["title"] or "")
                        if title in seen_titles:
                            continue
                        seen_titles.add(title)
                        out.append({
                            "title": title,
                            "year": r["year"],
                            "excerpt": (r["abstract"] or "")[:max_chars],
                            "source": f"entity:{r['entity']}",
                        })
            except Exception as exc:
                logger.warning("Entity-anchored retrieval failed: %s", exc)

        # 2) Fulltext fallback (always run; cheap and complementary)
        try:
            ft_query = _sanitize_for_lucene(question)
            if ft_query:
                with self._driver.session(database=self._db) as sess:
                    rows = sess.run(
                        "CALL db.index.fulltext.queryNodes('paper_text', $q) "
                        "YIELD node, score "
                        "WHERE node.abstract IS NOT NULL "
                        "RETURN node.title AS title, node.year_published AS year, "
                        "       node.abstract AS abstract, score "
                        "ORDER BY score DESC LIMIT $top",
                        q=ft_query, top=fulltext_top,
                    )
                    for r in rows:
                        title = (r["title"] or "")
                        if title in seen_titles:
                            continue
                        seen_titles.add(title)
                        out.append({
                            "title": title,
                            "year": r["year"],
                            "excerpt": (r["abstract"] or "")[:max_chars],
                            "source": f"fulltext:{r['score']:.2f}",
                        })
        except Exception as exc:
            logger.warning("Fulltext retrieval failed: %s", exc)

        return out

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


# Lucene reserved characters that must be stripped/escaped before being
# passed to db.index.fulltext.queryNodes.
_LUCENE_BAD = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]|&&|\|\|')
def _sanitize_for_lucene(text: str) -> str:
    """Strip Lucene reserved characters and very short tokens from a question
    so it can be passed safely as a fulltext query."""
    cleaned = _LUCENE_BAD.sub(" ", text)
    tokens = [t for t in cleaned.split() if len(t) > 2]
    # Cap to avoid pathological recall; most questions are short anyway
    return " ".join(tokens[:12])


def _format_evidence(rows: list[dict], abstracts: list[dict]) -> str:
    """Render the joint context (structured rows + abstract excerpts) for the
    answer-synthesis prompt."""
    parts: list[str] = []

    if rows:
        parts.append("STRUCTURED KG ROWS")
        parts.append("------------------")
        for r in rows[:15]:
            parts.append(f"- {r}")
        parts.append("")

    if abstracts:
        parts.append("ABSTRACT EXCERPTS (verbatim from the underlying papers)")
        parts.append("-------------------------------------------------------")
        for i, doc in enumerate(abstracts[:6], 1):
            year = f" ({doc['year']})" if doc.get("year") else ""
            src = doc.get("source", "")
            parts.append(f"[{i}] {doc.get('title','(untitled)')}{year}  <{src}>")
            excerpt = (doc.get("excerpt") or "").strip().replace("\n", " ")
            parts.append(f"    {excerpt}")
            parts.append("")

    if not parts:
        return "(no evidence retrieved)"
    return "\n".join(parts)

