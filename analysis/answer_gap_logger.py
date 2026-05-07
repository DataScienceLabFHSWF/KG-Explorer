"""
Answer-Gap Logger
=================
Persistent log of chat questions that the GraphRAG agent could not answer
from the knowledge graph + abstracts. Two failure signals are recorded:

* **sentinel** \u2014 the QA LLM emitted ``[MISSING_FROM_GRAPH]`` because the
  evidence it was given was insufficient.
* **no_evidence** \u2014 the retrieval layer (Cypher rows + fulltext abstracts)
  returned nothing at all.

Each event is appended to ``output/answer_gap_log.jsonl`` (one JSON object
per line, append-only). The log is consumed by
``analysis.answer_gap_report`` to produce the systematic gap report
described in ``docs/NEXT_STEPS.md``.

The logger is intentionally lightweight: no external service, no DB, just
a JSONL file so the chat path stays fast and the gap pipeline can read it
batch-style.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from analysis.neo4j_utils import OUTPUT_DIR

LOG_PATH = Path(os.getenv("ANSWER_GAP_LOG", OUTPUT_DIR / "answer_gap_log.jsonl"))


def log_answer_event(
    *,
    question: str,
    linked_entities: list[str],
    n_rows: int,
    n_abstracts: int,
    cypher: str = "",
    sentinel: bool = False,
) -> None:
    """Append one event line to the answer-gap log.

    Parameters
    ----------
    question
        The user's natural-language query.
    linked_entities
        Entity ``name_norm`` values found by the semantic linker.
    n_rows
        Number of Cypher result rows that reached the QA prompt.
    n_abstracts
        Number of abstract excerpts that reached the QA prompt.
    cypher
        Generated Cypher (truncated for storage).
    sentinel
        ``True`` if the LLM emitted the missing-data sentinel.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "linked_entities": linked_entities,
        "n_rows": n_rows,
        "n_abstracts": n_abstracts,
        "cypher": (cypher or "")[:300],
        "sentinel": bool(sentinel),
        "no_evidence": (n_rows == 0 and n_abstracts == 0),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
