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

    # Semantic gaps (sentence-transformer cosine over entity names that never co-occur)
    sem = _load_csv(OUTPUT_DIR / "semantic_gaps.csv")
    if sem is not None:
        top = sem.head(30)
        gaps["semantic_gaps"] = top.to_dict("records")
        gaps["summary_stats"]["semantic_gaps"] = len(top)
        print(f"  Loaded top-{len(top)} semantic gaps")
    else:
        gaps["semantic_gaps"] = []

    # Cross-community sparse bridges
    cbg = _load_csv(OUTPUT_DIR / "community_bridge_gaps.csv")
    if cbg is not None:
        top = cbg.head(20)
        gaps["community_bridge_gaps"] = top.to_dict("records")
        gaps["summary_stats"]["community_bridge_gaps"] = len(top)
        print(f"  Loaded top-{len(top)} community bridge gaps")
    else:
        gaps["community_bridge_gaps"] = []

    # Longest shortest paths (compositional reasoning seeds)
    lp = _load_json(OUTPUT_DIR / "longest_paths.json")
    if lp:
        gaps["longest_paths"] = lp[:15]
        gaps["summary_stats"]["longest_paths"] = len(lp)
        print(f"  Loaded {len(lp)} longest-path chains")
    else:
        gaps["longest_paths"] = []

    # Forman–Ricci edge curvature (bottleneck edges)
    ec = _load_csv(OUTPUT_DIR / "edge_curvature.csv")
    if ec is not None:
        gaps["edge_curvature"] = ec.head(30).to_dict("records")
        gaps["summary_stats"]["edge_curvature"] = len(ec)
        print(f"  Loaded top-{len(gaps['edge_curvature'])} bottleneck edges")
    else:
        gaps["edge_curvature"] = []

    # Temporal trajectories
    tr = _load_csv(OUTPUT_DIR / "entity_trajectories.csv")
    if tr is not None:
        gaps["entity_trajectories"] = tr.to_dict("records")
        emerging = tr[tr["trajectory"] == "emerging"].nlargest(10, "log_slope").to_dict("records")
        stalled = tr[tr["trajectory"] == "stalled"].nsmallest(10, "log_slope").to_dict("records")
        gaps["emerging_fronts"] = emerging
        gaps["stalled_hubs"] = stalled
        gaps["summary_stats"]["emerging_fronts"] = len(emerging)
        gaps["summary_stats"]["stalled_hubs"] = len(stalled)
        print(f"  Loaded trajectories: {len(emerging)} emerging, {len(stalled)} stalled")
    else:
        gaps["emerging_fronts"] = []
        gaps["stalled_hubs"] = []

    # Articulation points (fragile bridges)
    ap = _load_csv(OUTPUT_DIR / "articulation_points.csv")
    if ap is not None:
        gaps["articulation_points"] = ap.head(20).to_dict("records")
        gaps["summary_stats"]["articulation_points"] = len(ap)
        print(f"  Loaded top-{len(gaps['articulation_points'])} articulation points")
    else:
        gaps["articulation_points"] = []

    # Triadic-closure deficit (sharper Adamic-Adar with null-model correction)
    td = _load_csv(OUTPUT_DIR / "triadic_deficit.csv")
    if td is not None:
        gaps["triadic_deficit"] = td.head(25).to_dict("records")
        gaps["summary_stats"]["triadic_deficit"] = len(td)
        print(f"  Loaded top-{len(gaps['triadic_deficit'])} triadic-deficit pairs")
    else:
        gaps["triadic_deficit"] = []

    # Bridge edges (2-edge-cuts)
    be = _load_csv(OUTPUT_DIR / "bridge_edges.csv")
    if be is not None:
        gaps["bridge_edges"] = be.head(15).to_dict("records")
        gaps["summary_stats"]["bridge_edges"] = len(be)
        print(f"  Loaded top-{len(gaps['bridge_edges'])} bridge edges")
    else:
        gaps["bridge_edges"] = []

    # PPR reachability gaps (semantically related but diffusion-cold)
    ppr = _load_csv(OUTPUT_DIR / "ppr_reachability_gaps.csv")
    if ppr is not None:
        gaps["ppr_gaps"] = ppr.head(30).to_dict("records")
        gaps["summary_stats"]["ppr_gaps"] = len(ppr)
        print(f"  Loaded top-{len(gaps['ppr_gaps'])} PPR reachability gaps")
    else:
        gaps["ppr_gaps"] = []

    # Heat-kernel reachability gaps
    hk = _load_csv(OUTPUT_DIR / "heat_kernel_reachability_gaps.csv")
    if hk is not None:
        gaps["heat_gaps"] = hk.head(30).to_dict("records")
        gaps["summary_stats"]["heat_gaps"] = len(hk)
        print(f"  Loaded top-{len(gaps['heat_gaps'])} heat-kernel reachability gaps")
    else:
        gaps["heat_gaps"] = []

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

    # Answer-gap report (questions the chat agent could not answer).
    # This is the *behavioural* gap signal, complementing all the
    # structural ones above. Produced by analysis/answer_gap_report.py from
    # the live JSONL log written by the chat path.
    ag = _load_json(OUTPUT_DIR / "answer_gap_report.json")
    if ag:
        gaps["answer_gaps"] = ag
        gaps["summary_stats"]["answer_gap_questions"] = ag.get("total_failed_questions", 0)
        gaps["summary_stats"]["answer_gap_top_entities"] = ag.get("top_entities_in_failures", [])[:10]
        n_top = len(ag.get("top_entities_in_failures", []))
        n_unlinked = len(ag.get("unlinked_question_samples", []))
        print(f"  Loaded answer-gap report: {ag.get('total_failed_questions', 0)} failed questions, "
              f"{n_top} repeat-offender entities, {n_unlinked} unlinked-question samples")
    else:
        gaps["answer_gaps"] = None

    return gaps


# ---------------------------------------------------------------------------
# 2.  Generate research hypotheses (offline mode)
# ---------------------------------------------------------------------------
def generate_hypotheses(gaps):
    """Generate structured research hypotheses from gap data."""
    hypotheses = []

    # From answer-gap (behavioural) report. These are the strongest signal
    # because they come from real user questions the system could not
    # answer. Each repeat-offender entity becomes a candidate for typed-
    # relation enrichment or focused PDF ingestion.
    ag = gaps.get("answer_gaps")
    if ag:
        top_ents = ag.get("top_entities_in_failures") or []
        for name, n in top_ents[:10]:
            hypotheses.append({
                "type": "answer_gap_entity",
                "confidence": "high" if n >= 3 else "medium",
                "entities": [name],
                "hypothesis": (
                    f"The entity '{name}' is repeatedly linked by the chat agent "
                    f"({n} failed questions) but the surrounding KG context is "
                    f"insufficient to ground an answer. Likely missing typed "
                    f"relations (e.g. USES_FUEL, HAS_TEMPERATURE, IS_TYPE_OF) "
                    f"or insufficient abstract coverage."
                ),
                "suggested_action": (
                    f"(1) Run a focused IE pass on the top papers mentioning '{name}'. "
                    f"(2) Ingest 5\u201310 highly-cited recent reviews via daq/. "
                    f"(3) Add typed Reaction / Quantity / ConfinementClass relations."
                ),
                "source": "answer_gap_report",
                "failed_questions": n,
            })

        # Unlinked question samples = concepts MISSING from the KG entirely.
        for q in (ag.get("unlinked_question_samples") or [])[:5]:
            hypotheses.append({
                "type": "answer_gap_unlinked",
                "confidence": "high",
                "entities": [],
                "hypothesis": (
                    f"User question '{q}' could not be linked to any entity in the KG. "
                    f"This indicates a concept the NER pipeline did not extract or that "
                    f"is genuinely absent from the corpus."
                ),
                "suggested_action": (
                    "Add the missing concept(s) to the NER schema or ingest papers "
                    "covering this topic."
                ),
                "source": "answer_gap_report",
            })

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

    # From semantic gaps (entities semantically related but never co-occur)
    for sg in gaps.get("semantic_gaps", [])[:15]:
        hypotheses.append({
            "type": "semantic_gap",
            "confidence": "medium",
            "entities": [sg["entity_a"], sg["entity_b"]],
            "hypothesis": (
                f"'{sg['entity_a']}' and '{sg['entity_b']}' are semantically "
                f"very similar (cosine={sg['cosine']:.3f}) but never co-occur in "
                f"any paper. They likely belong to disconnected research threads "
                f"that should be cross-referenced."
            ),
            "cosine": sg["cosine"],
            "suggested_action": "Search for papers that mention either concept "
                                "to determine whether the gap reflects a true "
                                "research opportunity or merely vocabulary drift.",
        })

    # From cross-community sparse bridges (interdisciplinary opportunities)
    for cbg in gaps.get("community_bridge_gaps", [])[:8]:
        hypotheses.append({
            "type": "community_bridge_gap",
            "confidence": "medium",
            "entities": [
                *str(cbg.get("label_a", "")).split(", "),
                *str(cbg.get("label_b", "")).split(", "),
            ],
            "hypothesis": (
                f"Communities {cbg['community_a']} ({cbg.get('label_a','?')}) and "
                f"{cbg['community_b']} ({cbg.get('label_b','?')}) share only "
                f"{cbg['observed_edges']:.0f} co-occurrence edges, vs. an expected "
                f"{cbg['expected_edges']:.0f} under a configuration-model null "
                f"(ratio={cbg['ratio']:.3f}). This points to an under-explored "
                f"interdisciplinary connection."
            ),
            "gap_score": cbg["gap_score"],
            "suggested_action": "Identify representative entities from each "
                                "community and look for natural points of overlap "
                                "(e.g. shared methods, instruments, or boundary "
                                "phenomena).",
        })

    # From longest shortest paths (compositional reasoning seeds, paper §4)
    for lp in gaps.get("longest_paths", [])[:6]:
        path = lp.get("path", [])
        if len(path) < 4:
            continue
        hypotheses.append({
            "type": "reasoning_chain",
            "confidence": "low",
            "entities": path,
            "hypothesis": (
                f"A {lp['length']}-hop reasoning chain connects "
                f"'{lp['source']}' to '{lp['target']}' via: "
                f"{' → '.join(path)}. Such long chains are candidate "
                f"compositional inference paths (Step A→D in Buehler 2025)."
            ),
            "length": lp["length"],
            "suggested_action": "Walk the path with a domain expert (or LLM) to "
                                "see whether each consecutive hop is mechanistic, "
                                "associative, or coincidental — mechanistic chains "
                                "often imply testable hypotheses.",
        })

    # From Forman–Ricci edge curvature (bottleneck edges)
    for ec in gaps.get("edge_curvature", [])[:8]:
        hypotheses.append({
            "type": "bottleneck_edge",
            "confidence": "medium",
            "entities": [ec["entity_a"], ec["entity_b"]],
            "hypothesis": (
                f"The edge ('{ec['entity_a']}', '{ec['entity_b']}') has Forman–Ricci "
                f"curvature {ec['forman_curvature']} (degrees {ec['deg_a']}, "
                f"{ec['deg_b']}; only {ec['common_neighbours']} shared neighbours). "
                f"Strongly negative curvature indicates a fragile bottleneck: the "
                f"two endpoints draw on largely disjoint communities."
            ),
            "forman_curvature": ec["forman_curvature"],
            "suggested_action": "Identify which other entities could plausibly "
                                "close triangles around this edge — they would "
                                "strengthen the bridge and surface adjacent gaps.",
        })

    # From emerging fronts (research opportunities at the frontier)
    for ef in gaps.get("emerging_fronts", [])[:6]:
        hypotheses.append({
            "type": "emerging_front",
            "confidence": "medium",
            "entities": [ef["entity"]],
            "hypothesis": (
                f"'{ef['entity']}' shows fast multiplicative growth in recent "
                f"co-occurrence degree (log-slope={ef['log_slope']:+.3f}, "
                f"recent years {ef['recent_years']}: {ef['recent_degrees']}). "
                f"This is an active research front whose neighbouring concepts "
                f"are likely under-developed."
            ),
            "log_slope": ef["log_slope"],
            "suggested_action": "Inspect the entity's current k-hop neighbourhood; "
                                "missing co-occurrences with classical fusion "
                                "concepts often indicate first-mover opportunities.",
        })

    # From stalled hubs (saturated topics)
    for sh in gaps.get("stalled_hubs", [])[:4]:
        hypotheses.append({
            "type": "stalled_hub",
            "confidence": "low",
            "entities": [sh["entity"]],
            "hypothesis": (
                f"'{sh['entity']}' is a high-mention hub whose recent activity is "
                f"declining (log-slope={sh['log_slope']:+.3f}). It may represent "
                f"saturated knowledge — worth probing for unresolved sub-questions "
                f"or methodological refresh."
            ),
            "log_slope": sh["log_slope"],
            "suggested_action": "Search for review papers explaining the slowdown; "
                                "open questions in such reviews are good gap targets.",
        })

    # From articulation points (fragile single-node connectors)
    for ap in gaps.get("articulation_points", [])[:6]:
        hypotheses.append({
            "type": "fragile_bridge",
            "confidence": "medium",
            "entities": [ap["entity"]],
            "hypothesis": (
                f"'{ap['entity']}' is an articulation point: removing it would "
                f"split the giant component into {ap['n_fragments']} fragments "
                f"(second-largest size: {ap['second_fragment']}). The literature "
                f"depends on this single connector — alternative bridging "
                f"concepts deserve exploration."
            ),
            "removal_impact": ap["removal_impact"],
            "suggested_action": "Use semantic search to propose redundant bridge "
                                "concepts that would make this connection more "
                                "robust; absence of such concepts is the gap.",
        })

    # From triadic-closure deficit (null-model corrected missing links)
    for td in gaps.get("triadic_deficit", [])[:10]:
        hypotheses.append({
            "type": "triadic_deficit",
            "confidence": "medium",
            "entities": [td["entity_a"], td["entity_b"]],
            "hypothesis": (
                f"'{td['entity_a']}' and '{td['entity_b']}' share "
                f"{td['common_neighbours']:.0f} neighbours but never directly "
                f"co-occur, with deficit score {td['deficit']:.2f} (configuration-"
                f"model expectation {td['p_expected']:.4f}). Their absence is far "
                f"larger than chance — a strong missing-link candidate."
            ),
            "deficit": td["deficit"],
            "suggested_action": "Inspect the shared neighbourhood; the bridging "
                                "concepts often hint at the implicit relationship.",
        })

    # From bridge edges (2-edge-cuts: single edges that hold communities together)
    for be in gaps.get("bridge_edges", [])[:6]:
        hypotheses.append({
            "type": "bridge_edge",
            "confidence": "medium",
            "entities": [be["entity_a"], be["entity_b"]],
            "hypothesis": (
                f"The single edge ('{be['entity_a']}', '{be['entity_b']}', "
                f"weight={be['weight']:.0f}) is a backbone bridge: removing it "
                f"would isolate a fragment of {be['fragment_size']} entities "
                f"({be['fraction_isolated']*100:.1f}% of the giant component). "
                f"The connection rests on a single citation pattern."
            ),
            "fragment_size": be["fragment_size"],
            "suggested_action": "Look for alternative concept pairs that should "
                                "co-occur to reinforce this bridge — their absence "
                                "is the literature gap.",
        })

    # From PPR reachability gaps (semantic vs. diffusion mismatch)
    for ppr in gaps.get("ppr_gaps", [])[:10]:
        hypotheses.append({
            "type": "reachability_gap_ppr",
            "confidence": "medium",
            "entities": [ppr["hub"], ppr["candidate"]],
            "hypothesis": (
                f"'{ppr['candidate']}' is semantically very close to the hub "
                f"'{ppr['hub']}' (cosine={ppr['cosine']:.3f}) yet its "
                f"Personalised PageRank mass from that hub is in the bottom "
                f"diffusion decile (π={ppr['diffusion_mass']:.2e}). The "
                f"literature treats them as related concepts, but no efficient "
                f"citation path connects them."
            ),
            "score": ppr["score"],
            "suggested_action": "Bridge papers covering both concepts would have "
                                "an outsized structural effect on the KG.",
        })

    # From heat-kernel reachability gaps (multi-scale diffusion analogue)
    for hk in gaps.get("heat_gaps", [])[:6]:
        hypotheses.append({
            "type": "reachability_gap_heat",
            "confidence": "low",
            "entities": [hk["hub"], hk["candidate"]],
            "hypothesis": (
                f"At heat-kernel scale t=3, '{hk['candidate']}' receives almost "
                f"no diffusion mass from '{hk['hub']}' yet remains semantically "
                f"similar (cosine={hk['cosine']:.3f}). Multi-scale analysis "
                f"confirms the gap is structural, not just local."
            ),
            "score": hk["score"],
            "suggested_action": "Cross-check against PPR results; consistent "
                                "hub–candidate pairs across scales are the most "
                                "actionable gaps.",
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
        "topological_void": "Topological Void (Knowledge Gap)",
        "structural_hole": "Structural Hole (Bridge Concept)",
        "missing_link": "Missing Link (Predicted Co-Occurrence)",
        "knowledge_loop": "Knowledge Loop (Redundancy)",
        "semantic_gap": "Semantic Gap (Related but Never Co-occur)",
        "community_bridge_gap": "Community Bridge Gap (Under-connected Communities)",
        "reasoning_chain": "Reasoning Chain (Long Compositional Path)",
        "bottleneck_edge": "Bottleneck Edge (Negative Forman Curvature)",
        "emerging_front": "Emerging Research Front",
        "stalled_hub": "Stalled Hub (Saturated Topic)",
        "fragile_bridge": "Fragile Bridge (Articulation Point)",
        "triadic_deficit": "Triadic-Closure Deficit (Null-Corrected Missing Link)",
        "bridge_edge": "Bridge Edge (2-Edge-Cut)",
        "reachability_gap_ppr": "Reachability Gap (PPR-Cold)",
        "reachability_gap_heat": "Reachability Gap (Heat-Kernel-Cold)",
        "llm_enhanced": "LLM-Enhanced Hypotheses",
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
def _build_llm_prompt(gaps, hypotheses):
    summary_parts = [
        "You are analysing a knowledge graph of nuclear fusion energy research.",
        f"The graph has {gaps['summary_stats'].get('total_entities', '?')} entities "
        f"and {gaps['summary_stats'].get('communities', '?')} communities.",
        "",
        "Key findings from mathematical analysis:",
    ]
    for h in hypotheses[:12]:
        summary_parts.append(f"- [{h['type']}] {h['hypothesis']}")
    prompt = "\n".join(summary_parts)
    prompt += (
        "\n\nBased on these structural findings, generate 5 specific, actionable "
        "research hypotheses for nuclear fusion researchers. For each hypothesis, "
        "explain the scientific rationale and suggest concrete next steps. "
        "Format as numbered items."
    )
    return prompt


def enhance_with_llm(gaps, hypotheses):
    """Enrich hypotheses with LLM reasoning.

    Tries Ollama first (local, free; configured via OLLAMA_BASE_URL and
    OLLAMA_MODEL env vars), then falls back to OpenAI if OPENAI_API_KEY
    is set. Silently no-ops if neither is available.
    """
    prompt = _build_llm_prompt(gaps, hypotheses)
    system = "You are a nuclear fusion research advisor."
    llm_text = None
    backend = None

    # 1. Try Ollama (local)
    if os.getenv("GAP_AGENT_USE_OLLAMA", "1") != "0":
        try:
            import requests
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "nemotron-3-nano:4b")
            print(f"  Querying Ollama ({model} at {base}) for enhanced hypotheses...")
            r = requests.post(
                f"{base}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1500},
                },
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()
            llm_text = (data.get("message") or {}).get("content")
            if llm_text:
                backend = f"ollama:{model}"
        except Exception as e:
            print(f"  Ollama unavailable ({e}) — will try OpenAI fallback.")

    # 2. Fall back to OpenAI
    if not llm_text:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                print("  Querying OpenAI for enhanced hypotheses...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                )
                llm_text = response.choices[0].message.content
                backend = "openai:gpt-4o-mini"
            except Exception as e:
                print(f"  OpenAI query failed: {e}")
        else:
            if backend is None:
                print("  No LLM backend available — skipping enhancement.")
                print("  (Run a local Ollama server or set OPENAI_API_KEY.)")

    if llm_text:
        print(f"  LLM response received via {backend}.")
        hypotheses.append({
            "type": "llm_enhanced",
            "confidence": "medium",
            "entities": [],
            "hypothesis": llm_text,
            "backend": backend,
            "suggested_action": "Review and validate with domain experts.",
        })

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
        gaps.get("semantic_gaps"),
        gaps.get("community_bridge_gaps"),
        gaps.get("longest_paths"),
        gaps.get("edge_curvature"),
        gaps.get("emerging_fronts"),
        gaps.get("articulation_points"),
    ]):
        print("  No gap data found. Run void_extraction, structural_holes, "
              "link_prediction, semantic_gaps, and advanced_gaps first.")
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
