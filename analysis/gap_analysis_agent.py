"""
LLM-Powered Gap Analysis Agent
================================
Aggregates detected knowledge gaps from all modules and generates
structured research hypotheses.

This module works in two modes:
  1. **Offline** (default): Aggregates and summarises gaps without an LLM,
     producing a structured gap report with suggested research questions.
  2. **LLM-enhanced** (when OPENAI_API_KEY is set): Sends the gap summary
     to an LLM for richer hypothesis generation.

Inputs (from other modules)
---------------------------
  - knowledge_gaps.json      ← void_extraction (H1 loops, H2 voids)
  - structural_holes.csv     ← structural_holes (bridge concepts)
  - predicted_links.csv      ← link_prediction (missing co-occurrences)
  - fca_implications.json    ← fca_analysis (category implications)
  - communities.csv          ← graph_analysis (community structure)
  - centralities.csv         ← graph_analysis (node importance)

Outputs
-------
  - gap_report.json           — structured gap report
  - gap_report.md             — human-readable gap report
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from analysis.neo4j_utils import OUTPUT_DIR


def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# 1.  Aggregate all gap data
# ---------------------------------------------------------------------------
def aggregate_gaps():
    """Load and aggregate gap data from all analysis modules."""
    gaps = {
        "topological": {"loops": [], "voids": []},
        "structural_holes": [],
        "predicted_links": [],
        "fca_implications": [],
        "summary_stats": {},
    }

    # Topological gaps
    kg = _load_json(OUTPUT_DIR / "knowledge_gaps.json")
    if kg:
        gaps["topological"]["loops"] = kg.get("h1_loops", [])
        gaps["topological"]["voids"] = kg.get("h2_voids", [])
        gaps["summary_stats"]["h1_loops"] = len(gaps["topological"]["loops"])
        gaps["summary_stats"]["h2_voids"] = len(gaps["topological"]["voids"])
        print(f"  Loaded {len(gaps['topological']['loops'])} H1 loops, "
              f"{len(gaps['topological']['voids'])} H2 voids")

    # Structural holes
    sh = _load_csv(OUTPUT_DIR / "structural_holes.csv")
    if sh is not None:
        top = sh.head(30)
        gaps["structural_holes"] = top.to_dict("records")
        gaps["summary_stats"]["structural_holes"] = len(top)
        print(f"  Loaded top-{len(top)} structural hole entities")

    # Predicted links
    pl = _load_csv(OUTPUT_DIR / "predicted_links.csv")
    if pl is not None:
        top = pl.head(50)
        gaps["predicted_links"] = top.to_dict("records")
        gaps["summary_stats"]["predicted_links"] = len(top)
        print(f"  Loaded top-{len(top)} predicted missing links")

    # FCA implications
    fca = _load_json(OUTPUT_DIR / "fca_implications.json")
    if fca:
        gaps["fca_implications"] = fca[:30]
        gaps["summary_stats"]["fca_implications"] = len(fca)
        print(f"  Loaded {len(fca)} FCA implications")

    # Community structure
    comm = _load_csv(OUTPUT_DIR / "communities.csv")
    if comm is not None:
        gaps["summary_stats"]["communities"] = comm["community"].nunique()

    # Centralities
    cent = _load_csv(OUTPUT_DIR / "centralities.csv")
    if cent is not None:
        gaps["summary_stats"]["total_entities"] = len(cent)
        top_pr = cent.nlargest(5, "pagerank")[["entity", "pagerank"]].to_dict("records")
        gaps["summary_stats"]["top_pagerank"] = top_pr

    return gaps


# ---------------------------------------------------------------------------
# 2.  Generate research hypotheses (offline mode)
# ---------------------------------------------------------------------------
def generate_hypotheses(gaps):
    """Generate structured research hypotheses from gap data."""
    hypotheses = []

    # From topological voids (highest value gaps)
    for void in gaps["topological"]["voids"][:10]:
        entities = void.get("entities", [])
        if len(entities) >= 3:
            hypotheses.append({
                "type": "topological_void",
                "confidence": "high",
                "entities": entities,
                "hypothesis": (
                    f"The entities {', '.join(entities[:5])} co-occur pairwise "
                    f"in fusion literature but have never been studied together. "
                    f"A joint investigation may reveal novel interactions."
                ),
                "persistence": void.get("persistence"),
                "suggested_action": "Literature review of joint occurrences; "
                                    "propose experimental or computational study.",
            })

    # From structural holes
    for sh in gaps["structural_holes"][:10]:
        if sh.get("bridge_score", 0) > 0:
            hypotheses.append({
                "type": "structural_hole",
                "confidence": "medium",
                "entities": [sh["entity"]],
                "hypothesis": (
                    f"'{sh['entity']}' bridges disconnected research communities "
                    f"(betweenness={sh['betweenness']:.4f}, clustering={sh['clustering']:.4f}). "
                    f"This concept may be under-explored as a connector."
                ),
                "bridge_score": sh["bridge_score"],
                "suggested_action": "Investigate how this concept links different "
                                    "fusion sub-fields; potential review paper topic.",
            })

    # From predicted missing links
    for pl in gaps["predicted_links"][:15]:
        hypotheses.append({
            "type": "missing_link",
            "confidence": "medium",
            "entities": [pl["entity_a"], pl["entity_b"]],
            "hypothesis": (
                f"'{pl['entity_a']}' and '{pl['entity_b']}' share "
                f"{pl['common_neighbours']:.0f} common neighbours but never "
                f"co-occur directly. This may indicate an unexplored relationship."
            ),
            "adamic_adar": pl["adamic_adar"],
            "suggested_action": "Search for papers at the intersection; "
                                "if none exist, consider a study proposal.",
        })

    # From H1 loops (redundancy)
    for loop in gaps["topological"]["loops"][:5]:
        entities = loop.get("entities", [])
        if len(entities) >= 3:
            hypotheses.append({
                "type": "knowledge_loop",
                "confidence": "low",
                "entities": entities,
                "hypothesis": (
                    f"A redundant knowledge circuit exists among "
                    f"{', '.join(entities[:5])}. These concepts form a loop "
                    f"in the co-occurrence complex, suggesting well-established "
                    f"but potentially over-studied connections."
                ),
                "persistence": loop.get("persistence"),
                "suggested_action": "Verify if this represents consensus knowledge "
                                    "or an echo chamber in the literature.",
            })

    return hypotheses


# ---------------------------------------------------------------------------
# 3.  Generate markdown report
# ---------------------------------------------------------------------------
def generate_report(gaps, hypotheses):
    """Generate a human-readable markdown gap report."""
    lines = [
        "# Fusion Knowledge Graph — Gap Analysis Report",
        "",
        "## Executive Summary",
        "",
    ]

    stats = gaps["summary_stats"]
    lines.append(f"- **Total entities analysed**: {stats.get('total_entities', '?')}")
    lines.append(f"- **Communities detected**: {stats.get('communities', '?')}")
    lines.append(f"- **H1 knowledge loops**: {stats.get('h1_loops', 0)}")
    lines.append(f"- **H2 knowledge voids**: {stats.get('h2_voids', 0)}")
    lines.append(f"- **Structural holes**: {stats.get('structural_holes', 0)} bridge concepts")
    lines.append(f"- **Predicted missing links**: {stats.get('predicted_links', 0)}")
    lines.append(f"- **FCA implications**: {stats.get('fca_implications', 0)}")
    lines.append("")

    # Research Hypotheses
    lines.append("## Research Hypotheses")
    lines.append("")

    type_labels = {
        "topological_void": "🔴 Topological Void (Knowledge Gap)",
        "structural_hole": "🟡 Structural Hole (Bridge Concept)",
        "missing_link": "🟠 Missing Link (Predicted Co-Occurrence)",
        "knowledge_loop": "🔵 Knowledge Loop (Redundancy)",
    }

    for i, h in enumerate(hypotheses, 1):
        tl = type_labels.get(h["type"], h["type"])
        lines.append(f"### {i}. {tl}")
        lines.append("")
        lines.append(f"**Entities**: {', '.join(h['entities'][:8])}")
        lines.append("")
        lines.append(f"**Hypothesis**: {h['hypothesis']}")
        lines.append("")
        lines.append(f"**Suggested Action**: {h['suggested_action']}")
        lines.append("")

    # Top Structural Holes
    if gaps["structural_holes"]:
        lines.append("## Top Bridge Concepts (Structural Holes)")
        lines.append("")
        lines.append("| Entity | Betweenness | Clustering | Eff. Size | Bridge Score |")
        lines.append("|--------|-------------|------------|-----------|--------------|")
        for sh in gaps["structural_holes"][:15]:
            lines.append(
                f"| {sh['entity']} | {sh['betweenness']:.4f} | "
                f"{sh['clustering']:.4f} | {sh.get('effective_size', 0):.1f} | "
                f"{sh['bridge_score']:.6f} |"
            )
        lines.append("")

    # Top Predicted Links
    if gaps["predicted_links"]:
        lines.append("## Top Predicted Missing Co-Occurrences")
        lines.append("")
        lines.append("| Entity A | Entity B | Adamic-Adar | Common Neighbours | Jaccard |")
        lines.append("|----------|----------|-------------|-------------------|---------|")
        for pl in gaps["predicted_links"][:20]:
            lines.append(
                f"| {pl['entity_a']} | {pl['entity_b']} | "
                f"{pl['adamic_adar']:.3f} | {pl['common_neighbours']:.0f} | "
                f"{pl['jaccard']:.4f} |"
            )
        lines.append("")

    # FCA Implications
    if gaps["fca_implications"]:
        lines.append("## Category Implications (FCA)")
        lines.append("")
        lines.append("These rules describe which NER categories systematically co-occur:")
        lines.append("")
        for impl in gaps["fca_implications"][:15]:
            p = " ∧ ".join(impl["premise"]) if impl["premise"] else "∅"
            c = " ∧ ".join(impl["conclusion"])
            s = impl.get("support", "?")
            lines.append(f"- {p} → {c}  *(support: {s})*")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by the Fusion KG Analysis Pipeline*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4.  LLM-enhanced hypothesis generation (optional)
# ---------------------------------------------------------------------------
def enhance_with_llm(gaps, hypotheses):
    """If an OpenAI API key is available, enrich hypotheses with LLM reasoning."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  OPENAI_API_KEY not set — skipping LLM enhancement.")
        print("  Set it in .env to enable LLM-powered hypothesis generation.")
        return hypotheses

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("  openai package not installed — skipping LLM enhancement.")
        return hypotheses

    # Build a concise summary for the LLM
    summary_parts = [
        "You are analysing a knowledge graph of nuclear fusion energy research.",
        f"The graph has {gaps['summary_stats'].get('total_entities', '?')} entities "
        f"and {gaps['summary_stats'].get('communities', '?')} communities.",
        "",
        "Key findings from mathematical analysis:",
    ]

    for h in hypotheses[:10]:
        summary_parts.append(f"- {h['hypothesis']}")

    prompt = "\n".join(summary_parts)
    prompt += (
        "\n\nBased on these structural findings, generate 5 specific, actionable "
        "research hypotheses for nuclear fusion researchers. For each hypothesis, "
        "explain the scientific rationale and suggest concrete next steps. "
        "Format as numbered items."
    )

    print("  Querying LLM for enhanced hypotheses...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a nuclear fusion research advisor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        llm_text = response.choices[0].message.content
        print("  LLM response received.")

        # Add LLM hypotheses to the list
        hypotheses.append({
            "type": "llm_enhanced",
            "confidence": "medium",
            "entities": [],
            "hypothesis": llm_text,
            "suggested_action": "Review and validate with domain experts.",
        })
    except Exception as e:
        print(f"  LLM query failed: {e}")

    return hypotheses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Execute gap analysis and hypothesis generation.

    The ``year`` argument is accepted for API uniformity but is ignored; the
    agent simply reads whatever output files are present. If earlier stages
    were run with a year filter the files will already reflect that.
    """
    print("\n" + "=" * 60)
    print("  GAP ANALYSIS AGENT")
    print("=" * 60)

    # Aggregate data from all modules
    print("\n--- Loading analysis results ---")
    gaps = aggregate_gaps()

    if not any([
        gaps["topological"]["loops"],
        gaps["topological"]["voids"],
        gaps["structural_holes"],
        gaps["predicted_links"],
    ]):
        print("  No gap data found. Run void_extraction, structural_holes, "
              "and link_prediction first.")
        return None

    # Generate hypotheses
    print("\n--- Generating research hypotheses ---")
    hypotheses = generate_hypotheses(gaps)
    print(f"  Generated {len(hypotheses)} hypotheses")

    # Try LLM enhancement
    hypotheses = enhance_with_llm(gaps, hypotheses)

    # Generate report
    print("\n--- Generating gap report ---")
    report_md = generate_report(gaps, hypotheses)

    md_path = OUTPUT_DIR / "gap_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Saved: {md_path}")

    # Save structured JSON
    json_path = OUTPUT_DIR / "gap_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": gaps["summary_stats"],
            "hypotheses": hypotheses,
            "topological_gaps": {
                "loops": len(gaps["topological"]["loops"]),
                "voids": len(gaps["topological"]["voids"]),
            },
            "structural_holes": len(gaps["structural_holes"]),
            "predicted_links": len(gaps["predicted_links"]),
        }, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Print summary
    print(f"\n  === GAP ANALYSIS SUMMARY ===")
    print(f"  Hypotheses generated: {len(hypotheses)}")
    type_counts = Counter(h["type"] for h in hypotheses)
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    return {"hypotheses": hypotheses, "gaps": gaps}


if __name__ == "__main__":
    run()
