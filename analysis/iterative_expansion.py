"""
Iterative Graph Expansion — Agentic Multi-Hop Reasoning
========================================================
Implements the agentic iterative reasoning loop described in
arXiv 2502.13025 ("Deep Graph Reasoning"), adapted for our Fusion KG.

The loop:
  1. Start from a seed question.
  2. Ask the KG-RAG agent → get answer + linked entities + KG context.
  3. Extract *frontier entities* (newly mentioned concepts not yet explored).
  4. Pick the most relevant frontier entity and generate the next question.
  5. Repeat for N hops, logging each step.
  6. Write a JSONL trace and a Markdown summary to ``output/``.

This is an **exploration tool**, not a production pipeline.  It can surface
multi-hop reasoning chains and hypothesis paths that a single-turn query misses.

Usage
-----
    python -m analysis.iterative_expansion "What causes plasma instabilities?"
    python -m analysis.iterative_expansion "How does ITER use superconducting magnets?" --hops 6
    python -m analysis.iterative_expansion --seed-file questions.txt --hops 4

Output
------
    output/expansion_trace_<timestamp>.jsonl   — one JSON object per hop
    output/expansion_summary_<timestamp>.md    — human-readable chain
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.neo4j_utils import OUTPUT_DIR, get_database, get_driver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_entities(result: dict) -> list[str]:
    """Return entity names found by the linker or mentioned in KG context."""
    linked = [name for name, _ in (result.get("linked_entities") or [])]
    ctx_names: list[str] = []
    for row in (result.get("context") or []):
        for key in ("entity", "name_norm", "subject", "object", "src", "tgt"):
            val = row.get(key)
            if val and isinstance(val, str):
                ctx_names.append(val.lower())
    return list(dict.fromkeys(linked + ctx_names))  # deduplicated, order preserved


def _next_question(
    llm,
    question: str,
    answer: str,
    frontier_entity: str,
    hop: int,
) -> str:
    """Ask the LLM to generate one follow-up question about *frontier_entity*."""
    prompt = (
        "You are a nuclear-fusion research assistant helping to explore a knowledge graph. "
        "Based on the previous question and answer, generate ONE concise follow-up question "
        "that probes the role or properties of the entity in angle-brackets. "
        "The question should be answerable from a fusion-energy knowledge graph.\n\n"
        f"Previous question (hop {hop}): {question}\n"
        f"Summary of answer: {answer[:400]}\n"
        f"Entity to probe: <{frontier_entity}>\n\n"
        "Follow-up question (one sentence, no preamble):"
    )
    resp = llm.invoke(prompt)
    raw = (resp.content.strip() if hasattr(resp, "content") else str(resp).strip())
    # strip any "Follow-up question:" prefix the model might add
    raw = re.sub(r'^(follow[\-\s]up question[:\s]*)', '', raw, flags=re.IGNORECASE).strip()
    return raw or f"What is the role of {frontier_entity} in nuclear fusion?"


def _pick_frontier(
    all_seen: set[str],
    current_entities: list[str],
    answer: str,
) -> str | None:
    """Pick the most interesting unseen entity from the current answer."""
    answer_lower = answer.lower()
    # prefer entities that appear in the answer text (most salient)
    for ent in current_entities:
        if ent not in all_seen and ent.lower() in answer_lower:
            return ent
    # fallback: first unseen entity in the linked/context list
    for ent in current_entities:
        if ent not in all_seen:
            return ent
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_expansion(
    seed_question: str,
    n_hops: int = 5,
    driver=None,
    db: str | None = None,
) -> list[dict]:
    """Run the iterative expansion loop.

    Returns a list of hop records (one dict per hop).
    """
    from analysis.llm_graph_qa import FusionCypherAgent

    if driver is None:
        driver = get_driver()
    if db is None:
        db = get_database()

    agent = FusionCypherAgent()
    llm = agent._llm  # reuse the same LLM instance

    hops: list[dict] = []
    all_seen_entities: set[str] = set()
    question = seed_question
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Iterative Expansion  ({n_hops} hops)")
    print(f"  Seed: {seed_question}")
    print(f"{'='*60}")

    for hop_n in range(1, n_hops + 1):
        t0 = time.time()
        print(f"\n  [Hop {hop_n}/{n_hops}] {question[:80]}")

        result = agent.ask(question)
        elapsed = round(time.time() - t0, 1)

        answer = result.get("answer") or ""
        coverage = result.get("coverage_score")
        linked = _extract_entities(result)

        # Compute graph_utilization: fraction of linked entities whose names
        # appear in the answer text.
        answer_lower = answer.lower()
        used = [e for e in linked if e.lower() in answer_lower]
        graph_util = round(len(used) / len(linked), 3) if linked else 0.0

        hop_record = {
            "hop": hop_n,
            "question": question,
            "answer": answer[:800],
            "linked_entities": linked[:15],
            "n_context_rows": len(result.get("context") or []),
            "n_abstracts": len(result.get("abstracts") or []),
            "coverage_score": coverage,
            "graph_utilization": graph_util,
            "missing_from_graph": result.get("missing_from_graph"),
            "elapsed_s": elapsed,
            "cypher": result.get("cypher", "")[:300],
        }
        hops.append(hop_record)
        all_seen_entities.update(e.lower() for e in linked)

        print(f"    coverage={coverage}  util={graph_util}  rows={hop_record['n_context_rows']}  abstracts={hop_record['n_abstracts']}  ({elapsed}s)")
        if answer:
            print(f"    answer: {answer[:150]} …")

        if hop_n < n_hops:
            frontier = _pick_frontier(all_seen_entities, linked, answer)
            if frontier is None:
                print("    No new frontier entity found — stopping early.")
                break
            print(f"    → frontier entity: «{frontier}»")
            question = _next_question(llm, question, answer, frontier, hop_n)

    return hops


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_trace(hops: list[dict], ts: str) -> Path:
    path = OUTPUT_DIR / f"expansion_trace_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for h in hops:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"\n  Trace → {path}")
    return path


def _write_summary(hops: list[dict], seed: str, ts: str) -> Path:
    path = OUTPUT_DIR / f"expansion_summary_{ts}.md"
    lines = [
        f"# Iterative Expansion Summary",
        f"",
        f"**Seed question**: {seed}  ",
        f"**Generated**: {ts}  ",
        f"**Hops**: {len(hops)}",
        f"",
        f"---",
        f"",
    ]
    for h in hops:
        lines += [
            f"## Hop {h['hop']}",
            f"",
            f"**Question**: {h['question']}",
            f"",
            f"**Answer** (truncated):",
            f"> {h['answer'][:500]}",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Coverage | {h['coverage_score']} |",
            f"| Graph utilization | {h['graph_utilization']} |",
            f"| KG rows | {h['n_context_rows']} |",
            f"| Abstracts | {h['n_abstracts']} |",
            f"| Missing from graph | {h['missing_from_graph']} |",
            f"| Elapsed | {h['elapsed_s']}s |",
            f"",
        ]
        if h["linked_entities"]:
            ents = ", ".join(f"`{e}`" for e in h["linked_entities"][:8])
            lines.append(f"**Linked entities**: {ents}")
            lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary → {path}")
    return path


# ---------------------------------------------------------------------------
# run() for the orchestrator (optional; not called by default)
# ---------------------------------------------------------------------------

def run(driver=None, year: int | None = None, **kwargs) -> dict:
    """Placeholder — iterative expansion is not part of the default pipeline."""
    return {"status": "skipped (use CLI to run interactively)"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Run iterative multi-hop graph expansion from a seed question."
    )
    p.add_argument("question", nargs="?", default=None,
                   help="Seed question. Required unless --seed-file is given.")
    p.add_argument("--hops", type=int, default=5, help="Number of reasoning hops (default: 5).")
    p.add_argument("--seed-file", type=str, default=None,
                   help="Text file with one seed question per line (runs each independently).")
    args = p.parse_args()

    seeds: list[str] = []
    if args.seed_file:
        seeds = [l.strip() for l in Path(args.seed_file).read_text().splitlines() if l.strip()]
    elif args.question:
        seeds = [args.question]
    else:
        p.error("Provide a question or --seed-file.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    driver = get_driver()

    for seed in seeds:
        hops = run_expansion(seed, n_hops=args.hops, driver=driver)
        _write_trace(hops, ts)
        _write_summary(hops, seed, ts)

    driver.close()
