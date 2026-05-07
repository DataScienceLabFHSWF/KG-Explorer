"""Typed-Relation Information Extraction (IE) for the Fusion Knowledge Graph.

The current graph has only generic ``CO_OCCURS_WITH`` edges between entities.
This module upgrades them to typed, directional relations by asking a local
LLM to classify each entity pair given the sentence(s) where they co-occur.

Relation taxonomy
-----------------
USES          entity A uses/employs/utilises entity B
ACHIEVES      entity A achieves/reaches/attains entity B (a result or value)
CONTAINS      entity A contains/consists of entity B (structural part-of)
REQUIRES      entity A requires/needs entity B (precondition)
PRODUCES      entity A produces/generates entity B
IMPROVES      entity A improves/enhances entity B
COMPARES_TO   entity A is compared to / benchmarked against entity B
IS_TYPE_OF    entity A is a type / variant / subclass of entity B
NONE          no extractable directed relation

Workflow
--------
1. Sample the highest-weight ``CO_OCCURS_WITH`` pairs from Neo4j.
2. For each pair fetch up to ``max_sentences`` context sentences from abstracts
   where both entity names appear.
3. Batch several pairs into a single LLM call (faster than one call per pair).
4. Parse the structured LLM output into (subj, rel, obj) triples.
5. Write results to ``output/typed_relations.csv`` for human review.
6. Optionally commit triples to Neo4j as new edges (``--commit``).

Usage
-----
    python -m analysis.typed_relation_ie --pairs 100 --dry-run
    python -m analysis.typed_relation_ie --pairs 100 --commit
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Relation taxonomy ─────────────────────────────────────────────────────────

RELATIONS = [
    "USES",
    "ACHIEVES",
    "CONTAINS",
    "REQUIRES",
    "PRODUCES",
    "IMPROVES",
    "COMPARES_TO",
    "IS_TYPE_OF",
    "NONE",
]

# ── Neo4j queries ─────────────────────────────────────────────────────────────

# Fetch high-weight co-occurrence pairs not yet typed
_QUERY_PAIRS = """
MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
WHERE id(a) < id(b)
  AND NOT EXISTS { (a)-[:USES|ACHIEVES|CONTAINS|REQUIRES|PRODUCES|IMPROVES|COMPARES_TO|IS_TYPE_OF]->(b) }
  AND NOT EXISTS { (b)-[:USES|ACHIEVES|CONTAINS|REQUIRES|PRODUCES|IMPROVES|COMPARES_TO|IS_TYPE_OF]->(a) }
RETURN a.name AS subj, b.name AS obj, r.weight AS weight
ORDER BY r.weight DESC
LIMIT $limit
"""

# Find sentences in abstracts where both entity names appear
_QUERY_CONTEXT = """
CALL db.index.fulltext.queryNodes('paper_text', $query)
YIELD node
WHERE node.abstract IS NOT NULL
WITH node
LIMIT 5
UNWIND split(node.abstract, '. ') AS sent
WITH sent
WHERE toLower(sent) CONTAINS toLower($subj)
  AND toLower(sent) CONTAINS toLower($obj)
RETURN sent
LIMIT $max_sentences
"""

# Write a single typed relation edge
_QUERY_WRITE = """
MATCH (a:Entity {name: $subj})
MATCH (b:Entity {name: $obj})
MERGE (a)-[r:{rel_type}]->(b)
ON CREATE SET r.source = 'typed_ie', r.confidence = $confidence
"""

# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
You are a relation-extraction assistant for nuclear fusion research.
Given pairs of named entities and the sentence(s) in which they co-occur,
classify the directed relation from entity A to entity B.

Allowed relation types:
  USES        — A uses / employs / utilises B
  ACHIEVES    — A achieves / reaches / attains B (a result, temperature, value)
  CONTAINS    — A contains / consists of / includes B
  REQUIRES    — A requires / needs B as a precondition
  PRODUCES    — A produces / generates B
  IMPROVES    — A improves / enhances / optimises B
  COMPARES_TO — A is explicitly compared or benchmarked against B
  IS_TYPE_OF  — A is a type, variant, or subclass of B
  NONE        — no clear directed relation can be extracted

Respond ONLY with a JSON array. Each element must have exactly these keys:
  "subj"       : the A entity (copy exactly as given)
  "rel"        : one of the relation types above (uppercase)
  "obj"        : the B entity (copy exactly as given)
  "confidence" : a float 0.0–1.0 reflecting how certain you are
  "rationale"  : one short sentence explaining the choice

Do not add any text outside the JSON array.
""")

_USER_TEMPLATE = textwrap.dedent("""\
Classify the relations for the following entity pairs.
Each pair is followed by the sentences where they co-occur.

{pairs_block}
""")


def _build_pairs_block(pairs: list[dict]) -> str:
    """Format a list of (subj, obj, sentences) dicts for the LLM prompt."""
    parts = []
    for i, p in enumerate(pairs, 1):
        sentences = " | ".join(p["sentences"]) if p["sentences"] else "(no context found)"
        parts.append(
            f"Pair {i}:\n"
            f"  A = {p['subj']}\n"
            f"  B = {p['obj']}\n"
            f"  Context: {sentences}"
        )
    return "\n\n".join(parts)


# ── LLM call (Ollama) ─────────────────────────────────────────────────────────

def _call_llm(pairs_block: str, model: str, base_url: str) -> str:
    """Call the Ollama /api/chat endpoint and return the raw response text."""
    import requests  # local import — only needed when actually calling the LLM

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(pairs_block=pairs_block)},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_llm_output(raw: str) -> list[dict]:
    """Extract the JSON array from LLM output, tolerating minor prose noise."""
    m = _JSON_RE.search(raw)
    if not m:
        logger.warning("LLM returned no JSON array; raw output: %s", raw[:300])
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s; raw: %s", exc, m.group(0)[:300])
        return []
    # Validate / normalise
    valid = []
    for item in items:
        rel = (item.get("rel") or "NONE").strip().upper()
        if rel not in RELATIONS:
            rel = "NONE"
        valid.append(
            {
                "subj": str(item.get("subj", "")),
                "rel": rel,
                "obj": str(item.get("obj", "")),
                "confidence": float(item.get("confidence", 0.5)),
                "rationale": str(item.get("rationale", "")),
            }
        )
    return valid


# ── Core logic ────────────────────────────────────────────────────────────────

def _fetch_pairs(driver: Any, limit: int) -> list[dict]:
    """Fetch high-weight untyped entity pairs from Neo4j."""
    with driver.session() as session:
        rows = session.run(_QUERY_PAIRS, limit=limit).data()
    return rows  # list of {subj, obj, weight}


def _fetch_context(driver: Any, subj: str, obj: str, max_sentences: int = 3) -> list[str]:
    """Retrieve context sentences from abstracts where both entities appear."""
    # Build a Lucene query for the fulltext index
    # Use both entity names; OR is fine — we filter in Cypher anyway
    lucene_query = f'"{_escape_lucene(subj)}" "{_escape_lucene(obj)}"'
    with driver.session() as session:
        rows = session.run(
            _QUERY_CONTEXT,
            {
                "query": lucene_query,
                "subj": subj,
                "obj": obj,
                "max_sentences": max_sentences,
            },
        ).data()
    return [r["sent"] for r in rows]


_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]|&&|\|\|')


def _escape_lucene(text: str) -> str:
    return _LUCENE_SPECIAL.sub(lambda m: "\\" + m.group(0), text)


def run(
    n_pairs: int = 100,
    batch_size: int = 5,
    max_sentences: int = 3,
    min_confidence: float = 0.6,
    model: str = "gemma4:e2b",
    ollama_url: str = "http://localhost:11434",
    output_csv: Path = Path("output/typed_relations.csv"),
    commit: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the typed-relation IE pipeline.

    Parameters
    ----------
    n_pairs       : Total entity pairs to classify.
    batch_size    : Pairs per LLM call (5–10 is a good balance).
    max_sentences : Context sentences retrieved per pair from abstracts.
    min_confidence: Minimum confidence to include in CSV / commit to graph.
    model         : Ollama model name.
    ollama_url    : Ollama base URL.
    neo4j_url     : Neo4j Bolt URL.
    neo4j_user    : Neo4j username.
    neo4j_password: Neo4j password.
    output_csv    : Path for the output CSV.
    commit        : If True, write edges to Neo4j.
    dry_run       : If True, skip all LLM + write operations; just fetch pairs.

    Returns a summary dict.
    """
    from dotenv import load_dotenv
    load_dotenv()
    from analysis.neo4j_utils import get_driver  # lazy import

    driver = get_driver()

    logger.info("Fetching up to %d untyped CO_OCCURS_WITH pairs …", n_pairs)
    raw_pairs = _fetch_pairs(driver, limit=n_pairs)
    logger.info("Got %d pairs", len(raw_pairs))

    if dry_run:
        driver.close()
        return {"pairs_fetched": len(raw_pairs), "triples": [], "committed": 0}

    # Enrich pairs with context sentences
    logger.info("Fetching context sentences …")
    pairs_with_ctx: list[dict] = []
    for p in raw_pairs:
        sentences = _fetch_context(driver, p["subj"], p["obj"], max_sentences)
        pairs_with_ctx.append({"subj": p["subj"], "obj": p["obj"], "sentences": sentences})
        if not sentences:
            logger.debug("  No context for: %s — %s", p["subj"], p["obj"])

    # Classify in batches
    all_triples: list[dict] = []
    batches = [
        pairs_with_ctx[i : i + batch_size]
        for i in range(0, len(pairs_with_ctx), batch_size)
    ]
    logger.info("Classifying %d pairs in %d batches …", len(pairs_with_ctx), len(batches))

    for batch_idx, batch in enumerate(batches, 1):
        logger.debug("Batch %d/%d", batch_idx, len(batches))
        block = _build_pairs_block(batch)
        try:
            raw = _call_llm(block, model=model, base_url=ollama_url)
        except Exception as exc:
            logger.warning("LLM call failed for batch %d: %s", batch_idx, exc)
            continue
        triples = _parse_llm_output(raw)
        for t in triples:
            if t["rel"] != "NONE" and t["confidence"] >= min_confidence:
                all_triples.append(t)

    logger.info(
        "Extracted %d typed triples (confidence ≥ %.2f)", len(all_triples), min_confidence
    )

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(all_triples, output_csv)
    logger.info("Wrote %d triples to %s", len(all_triples), output_csv)

    # Commit to Neo4j
    committed = 0
    if commit:
        committed = _commit_to_neo4j(driver, all_triples)
        logger.info("Committed %d typed edges to Neo4j", committed)

    driver.close()
    return {
        "pairs_fetched": len(raw_pairs),
        "triples": len(all_triples),
        "committed": committed,
        "output_csv": str(output_csv),
    }


# ── CSV output ────────────────────────────────────────────────────────────────

def _write_csv(triples: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["subj", "rel", "obj", "confidence", "rationale"]
        )
        writer.writeheader()
        writer.writerows(triples)


# ── Neo4j commit ──────────────────────────────────────────────────────────────

def _commit_to_neo4j(driver: Any, triples: list[dict]) -> int:
    """Write typed edges to Neo4j. Returns the number of edges created/merged."""
    committed = 0
    with driver.session() as session:
        for t in triples:
            rel_type = t["rel"]
            if rel_type not in RELATIONS or rel_type == "NONE":
                continue
            # Dynamic relationship type requires string-interpolation in Cypher
            # (APOC-free; rel_type is validated against the allow-list above)
            cypher = _QUERY_WRITE.format(rel_type=rel_type)
            try:
                session.run(
                    cypher,
                    subj=t["subj"],
                    obj=t["obj"],
                    confidence=t["confidence"],
                )
                committed += 1
            except Exception as exc:
                logger.debug("Write failed for (%s)-[%s]->(%s): %s", t["subj"], rel_type, t["obj"], exc)
    return committed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="analysis.typed_relation_ie",
        description="Classify CO_OCCURS_WITH edges into typed relations using an LLM.",
    )
    parser.add_argument("--pairs", type=int, default=100,
                        help="Max entity pairs to classify (default: 100)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Pairs per LLM call (default: 5)")
    parser.add_argument("--max-sentences", type=int, default=3,
                        help="Context sentences per pair (default: 3)")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Min confidence to include triple (default: 0.6)")
    parser.add_argument("--model", default="gemma4:e2b",
                        help="Ollama model (default: gemma4:e2b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--output", default="output/typed_relations.csv",
                        help="Output CSV path (default: output/typed_relations.csv)")
    parser.add_argument("--commit", action="store_true",
                        help="Write typed edges to Neo4j (default: CSV only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch pairs only, no LLM calls, no writes")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    result = run(
        n_pairs=args.pairs,
        batch_size=args.batch_size,
        max_sentences=args.max_sentences,
        min_confidence=args.min_confidence,
        model=args.model,
        ollama_url=args.ollama_url,
        output_csv=Path(args.output),
        commit=args.commit,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
