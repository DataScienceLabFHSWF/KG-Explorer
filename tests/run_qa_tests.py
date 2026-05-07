"""
QA Behavioural Test Runner
==========================
Drives the full chat agent (`FusionCypherAgent`) over a fixed set of
questions defined in ``tests/qa_test_questions.json`` and reports
per-question pass/fail.

This script is the **wiring point** between the chat path and the
answer-gap pipeline:

    qa_test_questions.json
        |
        v
    run_qa_tests.py
        |
        +--> FusionCypherAgent.ask(q)              # GraphRAG: Cypher + abstracts
        |        |
        |        +--> answer_gap_logger.log_answer_event(...)
        |             (auto-fires on missing-data sentinel or zero evidence)
        |
        +--> per-question score vs `expect` block in the dataset
        |
        +--> writes output/qa_test_results.{json,md}
        |
        +--> finally: invokes analysis.answer_gap_report.main()
             which aggregates output/answer_gap_log.jsonl into
             output/answer_gap_report.{json,md}

Run with:
    python -m tests.run_qa_tests              # full dataset
    python -m tests.run_qa_tests feedback-01  # single question by id
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from analysis.answer_gap_report import main as build_gap_report
from analysis.llm_graph_qa import FusionCypherAgent
from analysis.neo4j_utils import OUTPUT_DIR

DATASET_PATH = Path(__file__).with_name("qa_test_questions.json")
RESULTS_JSON = OUTPUT_DIR / "qa_test_results.json"
RESULTS_MD = OUTPUT_DIR / "qa_test_results.md"


def _check_expectations(result: dict, expect: dict) -> tuple[bool, list[str]]:
    """Return ``(passed, failure_reasons)`` for one question."""
    reasons: list[str] = []
    answer = (result.get("answer") or "").lower()
    linked = {n for n, _ in (result.get("linked_entities") or [])}
    rows = result.get("context") or []
    abstracts = result.get("abstracts") or []
    missing = bool(result.get("missing_from_graph"))

    for ent in expect.get("must_link_entities", []):
        if not any(ent.lower() in li.lower() for li in linked):
            reasons.append(f"entity '{ent}' not linked")

    if expect.get("must_have_rows") and not rows:
        reasons.append("no KG rows returned")
    min_rows = expect.get("min_rows")
    if min_rows is not None and len(rows) < min_rows:
        reasons.append(f"only {len(rows)} rows (<{min_rows})")

    if expect.get("must_have_abstracts") and not abstracts:
        reasons.append("no abstract excerpts retrieved")

    min_chars = expect.get("min_answer_chars")
    if min_chars and len(answer) < min_chars:
        reasons.append(f"answer too short ({len(answer)}<{min_chars} chars)")

    for term in expect.get("must_not_contain", []):
        if term.lower() in answer:
            reasons.append(f"answer contains forbidden token '{term}'")

    for term in expect.get("should_mention", []):
        if term.lower() not in answer:
            reasons.append(f"answer does not mention '{term}'")

    if "missing_from_graph" in expect:
        if expect["missing_from_graph"] != missing:
            reasons.append(
                f"missing_from_graph={missing}, expected {expect['missing_from_graph']}"
            )

    return (len(reasons) == 0, reasons)


def _run_one(agent: FusionCypherAgent, item: dict) -> dict:
    print(f"\n[{item['id']}] {item['question']}")
    t0 = time.time()
    try:
        result = agent.ask(item["question"])
    except Exception as exc:
        result = {"error": str(exc), "answer": "", "context": [], "abstracts": [],
                  "linked_entities": [], "missing_from_graph": False}
    dt = time.time() - t0

    passed, reasons = _check_expectations(result, item.get("expect", {}))
    status = "PASS" if passed else "FAIL"
    print(f"  -> {status}  ({dt:.1f}s)  rows={len(result.get('context') or [])}  "
          f"abstracts={len(result.get('abstracts') or [])}  "
          f"linked={len(result.get('linked_entities') or [])}  "
          f"missing={result.get('missing_from_graph')}")
    if reasons:
        for r in reasons:
            print(f"     - {r}")

    return {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "passed": passed,
        "reasons": reasons,
        "elapsed_s": round(dt, 2),
        "answer": result.get("answer", ""),
        "cypher": result.get("cypher", ""),
        "n_rows": len(result.get("context") or []),
        "n_abstracts": len(result.get("abstracts") or []),
        "linked_entities": [n for n, _ in (result.get("linked_entities") or [])],
        "fallback_used": result.get("fallback_used", False),
        "missing_from_graph": result.get("missing_from_graph", False),
        "error": result.get("error"),
    }


def _render_markdown(results: list[dict]) -> str:
    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"] or "uncategorised", []).append(r)

    lines = [
        "# QA Behavioural Test Results",
        "",
        f"**Pass rate**: {n_pass}/{n_total}  ({100*n_pass/max(n_total,1):.0f}%)",
        "",
        "## Per-category summary",
        "",
        "| Category | Pass | Total |",
        "|---|---|---|",
    ]
    for cat, rs in by_cat.items():
        lines.append(f"| {cat} | {sum(1 for r in rs if r['passed'])} | {len(rs)} |")
    lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    for r in results:
        flag = "PASS" if r["passed"] else "FAIL"
        lines.append(f"### `[{r['id']}]` {flag}  *(category: {r['category']})*")
        lines.append("")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        lines.append(
            f"- linked: `{', '.join(r['linked_entities']) or '(none)'}`  "
            f"- rows: **{r['n_rows']}**  - abstracts: **{r['n_abstracts']}**  "
            f"- missing-from-graph: **{r['missing_from_graph']}**  "
            f"- elapsed: {r['elapsed_s']}s"
        )
        if r["reasons"]:
            lines.append("")
            lines.append("**Failures:**")
            for reason in r["reasons"]:
                lines.append(f"  - {reason}")
        if r.get("error"):
            lines.append(f"  - error: `{r['error']}`")
        if r.get("cypher"):
            lines.append("")
            lines.append("```cypher")
            lines.append(r["cypher"][:400])
            lines.append("```")
        if r.get("answer"):
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append("> " + r["answer"].replace("\n", "\n> "))
        lines.append("")
    return "\n".join(lines)


def main(only_id: str | None = None) -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    items = dataset["questions"]
    if only_id:
        items = [it for it in items if it["id"] == only_id]
        if not items:
            print(f"No question with id '{only_id}'")
            sys.exit(1)

    print(f"Running {len(items)} questions...")
    agent = FusionCypherAgent()

    results: list[dict] = []
    for item in items:
        results.append(_run_one(agent, item))

    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    RESULTS_MD.write_text(_render_markdown(results), encoding="utf-8")

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"Pass rate: {n_pass}/{len(results)}")
    print(f"Results : {RESULTS_JSON}")
    print(f"Markdown: {RESULTS_MD}")

    # Aggregate the gap log this run produced (and any prior runs).
    print(f"\nBuilding gap report from logged failures...")
    build_gap_report()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    main(only)
