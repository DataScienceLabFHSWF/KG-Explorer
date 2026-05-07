"""
Answer-Gap Report Generator
===========================
Aggregates ``output/answer_gap_log.jsonl`` (produced by the chat agent)
into a structured report of the questions the system could not answer.

Two complementary views are produced:

1. **Per-question table** \u2014 every failed question with the entities it
   linked to, the cypher attempted, and the failure mode.
2. **Concept frequency** \u2014 which entities show up most often in failed
   questions, signalling systematic blind spots in either the KG or the
   prompt template. These map directly back to candidate gap-detection
   targets (e.g.\ \"every question about ``lawson criterion`` fails because
   the abstracts cover it but no entity exists yet\").

Outputs
-------
- ``output/answer_gap_report.json``  \u2014 machine-readable summary
- ``output/answer_gap_report.md``    \u2014 human-readable summary

Run with:
    python -m analysis.answer_gap_report
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from analysis.neo4j_utils import OUTPUT_DIR
from analysis.answer_gap_logger import LOG_PATH


def _load_events() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    events = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def build_report(events: list[dict]) -> dict:
    """Aggregate raw events into a structured report."""
    by_mode: dict[str, list[dict]] = {
        "no_evidence": [],
        "sentinel_only": [],
        "unlinked": [],
    }
    entity_freq: Counter[str] = Counter()
    unlinked_questions: list[str] = []

    for ev in events:
        if not ev.get("linked_entities"):
            unlinked_questions.append(ev.get("question", ""))
            by_mode["unlinked"].append(ev)
        for ent in ev.get("linked_entities") or []:
            entity_freq[ent] += 1

        if ev.get("no_evidence"):
            by_mode["no_evidence"].append(ev)
        elif ev.get("sentinel"):
            by_mode["sentinel_only"].append(ev)

    return {
        "total_failed_questions": len(events),
        "by_failure_mode": {k: len(v) for k, v in by_mode.items()},
        "top_entities_in_failures": entity_freq.most_common(20),
        "examples": {k: v[:5] for k, v in by_mode.items()},
        "unlinked_question_samples": unlinked_questions[:20],
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Answer-Gap Report")
    lines.append("")
    lines.append(
        f"Total failed questions logged: **{report['total_failed_questions']}**"
    )
    lines.append("")

    lines.append("## Failure modes")
    lines.append("")
    lines.append("| Mode | Count |")
    lines.append("|---|---|")
    for k, v in report["by_failure_mode"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    if report["top_entities_in_failures"]:
        lines.append("## Entities most often present in failed questions")
        lines.append("")
        lines.append("These concepts are reachable by the entity linker, yet")
        lines.append("the agent still could not produce a grounded answer.")
        lines.append("Strong candidates for ontology enrichment or prompt-")
        lines.append("template additions.")
        lines.append("")
        lines.append("| Entity | Failed questions |")
        lines.append("|---|---|")
        for name, n in report["top_entities_in_failures"]:
            lines.append(f"| `{name}` | {n} |")
        lines.append("")

    if report["unlinked_question_samples"]:
        lines.append("## Questions where no entity could be linked")
        lines.append("")
        lines.append(
            "These are the strongest signal for **missing concepts**: the user "
            "asked about something the KG has no representation of."
        )
        lines.append("")
        for q in report["unlinked_question_samples"]:
            lines.append(f"- {q}")
        lines.append("")

    for mode, examples in report["examples"].items():
        if not examples:
            continue
        lines.append(f"## Examples \u2014 `{mode}`")
        lines.append("")
        for ex in examples:
            lines.append(f"- **Q:** {ex.get('question', '')}")
            ents = ex.get("linked_entities") or []
            if ents:
                lines.append(f"  - linked entities: {', '.join(ents)}")
            lines.append(
                f"  - rows={ex.get('n_rows', 0)}, abstracts={ex.get('n_abstracts', 0)}, sentinel={ex.get('sentinel')}"
            )
            cy = (ex.get("cypher") or "").strip()
            if cy:
                lines.append(f"  - cypher: `{cy[:160]}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    events = _load_events()
    report = build_report(events)

    json_out = OUTPUT_DIR / "answer_gap_report.json"
    md_out = OUTPUT_DIR / "answer_gap_report.md"
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"  Logged events    : {report['total_failed_questions']}")
    print(f"  Failure modes    : {report['by_failure_mode']}")
    print(f"  JSON report      : {json_out}")
    print(f"  Markdown report  : {md_out}")


if __name__ == "__main__":
    main()
