# Fusion Knowledge Graph — Next Steps & Handoff

**Date**: March 2026  
**Status**: Phase 1 complete, Phase 2 partially complete  
**Project**: `D:\Documents\FusionData`

---

## Current State Summary

The project is a mathematical analysis and data acquisition layer built on a Knowledge Graph of nuclear fusion energy (~50,600 entities, ~249,500 co-occurrence edges, ~8,400 papers).

**What works today:**
- 12 analysis modules running end-to-end via `python run_analysis.py`
- Data acquisition pipeline (`python -m daq`) with OpenAlex enrichment, arXiv fallback for closed-access papers, and OA landscape analysis
- OWL 2 ontology generation from KG structure + FCA implications (2,487 triples)
- Graph query interface (`python -m analysis.graph_qa "tokamak -> stellarator"`) with 7 query modes
- Zipf's law validation for entity frequency distributions
- 32 publication-quality plots, 5 CSV exports, 4 JSON reports, OWL/TTL ontology files
- Interactive HTML graph explorers (Pyvis)

---

## Priority 1 — Validation & Quality (Low Risk, High Impact)

### 1.1  Add test coverage

There are **zero unit tests**. This is the single biggest risk for maintainability.

**Recommended scope:**
- `tests/test_doi_extraction.py` — parse a sample NER JSON, verify PaperRecord fields
- `tests/test_openalex_client.py` — mock API responses, verify enrichment logic
- `tests/test_graph_qa.py` — test each query mode against a small fixture graph
- `tests/test_ontology_generator.py` — verify OWL output is valid RDF (parse with rdflib)

**Framework**: pytest. No CI exists yet — consider adding GitHub Actions later.

### 1.2  Entity normalisation

Link prediction (§8 of FINDINGS_REPORT.md) reveals ~30–50 singular/plural pairs (tokamak↔tokamaks, stellarator↔stellarators, ion↔ions) that should be merged.

**Existing helper**: `scripts/suggest_entity_merges.py` scans `output/predicted_links.csv` and prints candidates.

**What's needed:**
- Review the suggestions with a domain expert
- Write a merge script that updates Neo4j (MERGE nodes, recompute edge weights)
- Re-run all analyses on the cleaned graph to see if gap findings change

### 1.3  Validate OWL ontology with domain expert

`output/fusion_ontology.owl` contains 90 classes, 30 object properties, 9,984 SubClassOf axioms (from FCA), and 413 named individuals. It needs a domain expert review:
- Are the FCA-derived SubClassOf relationships semantically correct?
- Should some of the 90 NER categories be collapsed? (FCA analysis shows Concept ≈ Physical Process ≈ Physics Entity are equivalent on the data)
- Are the inter-category object properties meaningful?

### 1.4  Fix plot numbering collision

Plots 15 and 16 collide between the original spectral/FCA modules and the new Zipf module:
- `15_heat_kernel_trace.png` vs `15_zipf_law.png`
- `16_category_entity_counts.png` vs `16_zipf_deviation.png`

Renumber the Zipf plots to 31/32 or renumber all plots sequentially.

---

## Priority 2 — Full-Text Enrichment Pipeline (Medium Risk, High Impact)

This is the core of Phase 2: move from abstract-only data to full-text extraction.

### 2.1  Scale up DAQ downloads

The pipeline has been tested on a 30-paper batch. To run on all 8,176 papers:

```powershell
python -m daq --data-dir data/ --output data/fulltexts/ --email you@uni.de
```

**Expected results** (from the test batch): ~25% of papers are downloadable (OA gold/green/bronze/diamond + arXiv preprints). That gives ~2,000 full-text PDFs.

**Considerations:**
- Rate limiting is built in (1 req/s to OpenAlex, politeness delays on downloads)
- Some publishers (IOP, Elsevier) have bot protection — arXiv fallback handles this
- Resume is built in — the pipeline skips already-downloaded PDFs
- Allow ~4–6 hours for the full run

### 2.2  Feed PDFs into KGBuilder

Once PDFs are downloaded, the `kgbuilder_bridge.py` module produces the right directory structure:

```
output/kgbuilder_input/
├── docs/               # PDFs
├── manifest.json       # Metadata mapping
└── ontology/
    └── fusion_ner.owl  # OWL stub
```

Next steps:
1. Use the generated `fusion_ontology.ttl` (from the ontology generator, not the stub) as the KGBuilder ontology
2. Run: `kgbuilder run --docs output/kgbuilder_input/docs/ --ontology output/fusion_ontology.ttl`
3. This will produce chunked text, entity extraction at full-text level, and embeddings
4. The extracted entities extend the existing KG significantly

**Blocker**: KGBuilder's public API/CLI needs to be available. The bridge module generates the right format but hasn't been tested end-to-end with KGBuilder.

### 2.3  Embed chunks into Qdrant

KGBuilder uses Qdrant for vector storage. After full-text processing:
- Chunks are embedded with `qwen3-embedding` (1024-dim)
- Store in Qdrant alongside Neo4j for hybrid retrieval
- This enables the vector leg of the GraphRAG pipeline

**Status**: Qdrant is configured in the KGBuilder Docker stack but not yet in this project's `docker-compose.yml`.

---

## Priority 3 — Hybrid RAG / GraphQA Enhancement (Medium Risk, Medium Impact)

### 3.1  Upgrade GraphQA with LLM synthesis

The current `analysis/graph_qa.py` returns structured data (entities, paths, communities). The next step is to wrap query results with an LLM for natural-language answers:

1. User asks: *"What are the main challenges with stellarator confinement?"*
2. `GraphQA` dispatches to `neighbours("stellarator")` + `research_gaps("stellarator")`
3. Assembled context (entities, co-occurrences, gap hypotheses) is sent to an LLM
4. LLM synthesises a grounded answer with citations to the KG

**Architecture** (from Fusion_KG_Concept.md §7):
```
Query → Entity extraction → Cypher traversal → Vector retrieval → Rank fusion → LLM synthesis
```

**Dependencies**: An LLM API (OpenAI, Anthropic, or local via Ollama). The `gap_analysis_agent.py` already has the pattern for optional LLM usage with offline fallback.

### 3.2  Benchmark QA strategies

The concept document (§7) outlines 4 QA strategies:
1. **Vector-only** — standard RAG on embedded chunks
2. **Graph-only** — Cypher traversal from the current GraphQA
3. **Naive hybrid** — concatenate KG context + vector context
4. **Hybrid SOTA** — re-rank with reciprocal rank fusion

Benchmark these with a Ragas-style evaluation framework (faithfulness, relevance, context precision). This requires:
- A ground-truth QA dataset (20–50 hand-written questions with expected answers)
- The vector store populated (see §2.3)
- Evaluation framework (ragas or equivalent)

### 3.3  Add interactive QA frontend

Currently GraphQA is CLI-only. Options for a lightweight frontend:
- **Streamlit** — fastest to build, single-file app
- **Gradio** — good for demo/sharing
- **FastAPI + HTMX** — more production-ready

Scope: text input → GraphQA dispatch → formatted result display with entity links.

---

## Priority 4 — Ontology & Semantic Improvements (Lower Risk, Long-Term Value)

### 4.1  Link to upper ontologies

The generated OWL ontology is standalone. For interoperability:
- Map top-level classes to **BFO** (Basic Formal Ontology) or **EMMO** (European Materials Modelling Ontology)
- Example: `fusion:NuclearFusionDeviceType rdfs:subClassOf bfo:MaterialEntity`
- This enables cross-domain queries and linked data integration

### 4.2  SHACL validation shapes

Write SHACL shapes to validate incoming KG data:
- Every Entity must have at least one Category
- Every Paper must have a title and year
- CO_OCCURS_WITH weight must be positive integer
- This catches data quality issues early in the pipeline

### 4.3  OWL reasoner pass

Run a DL reasoner (HermiT, Pellet, or ELK) on the generated ontology to:
- Discover inferred subclass relationships
- Detect inconsistencies in the FCA-derived axioms
- Materialise inferred triples and add them back to the KG

---

## Priority 5 — Advanced Analysis Extensions (Exploratory)

### 5.1  Temporal decomposition

The `--year` flag exists but hasn't been used systematically. Run analyses for each decade or 5-year window:

```powershell
foreach ($y in 2000, 2005, 2010, 2015, 2020) {
    python run_analysis.py --year $y --only graph --skip-plots
}
```

Compare: How have communities evolved? When did stellarator research grow? Are knowledge gaps closing over time?

### 5.2  GNN-based link prediction

The current link prediction uses neighbourhood heuristics (Adamic-Adar, Jaccard). Upgrade to:
- Graph Neural Networks (PyTorch Geometric or DGL)
- Node2Vec embeddings + logistic regression
- Compare AUC with the heuristic baselines

### 5.3  Cross-domain generalization

The pipeline is domain-agnostic. Test on:
- Semiconductor physics papers
- Quantum computing abstracts
- Battery/energy storage literature

This validates the tool as a general-purpose KG analysis framework.

### 5.4  Publication

The `docs/internal_working_paper.tex` has a draft structure. To turn this into a publishable paper:
- Add the Zipf analysis and OA landscape results
- Include the ontology generation methodology
- Write a "reproducibility" section documenting the full pipeline
- Target: Journal of Informetrics, Scientometrics, or a domain-specific fusion journal

---

## Quick Reference: How to Run Everything

```powershell
# Start infrastructure
docker compose up -d

# Run all 12 analysis modules
python run_analysis.py

# Run specific modules
python run_analysis.py --only zipf ontology

# Query the graph
python -m analysis.graph_qa "tokamak"
python -m analysis.graph_qa "tokamak -> stellarator"
python -m analysis.graph_qa "trend deuterium"
python -m analysis.graph_qa "gaps plasma"

# Run DAQ pipeline (download OA papers)
python -m daq --data-dir data/ --output data/fulltexts/ --email you@uni.de --limit 50

# OA landscape analysis
python scripts/oa_landscape.py

# Entity merge suggestions
python scripts/suggest_entity_merges.py
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_analysis.py` | Main orchestrator — runs all 12 analysis modules |
| `analysis/neo4j_utils.py` | Shared Neo4j driver, queries, helpers |
| `analysis/graph_qa.py` | Graph query interface (7 modes, CLI + library) |
| `analysis/ontology_generator.py` | OWL 2 ontology construction from KG |
| `analysis/zipf_analysis.py` | Zipf's law validation |
| `daq/pipeline.py` | DAQ pipeline orchestrator |
| `daq/kgbuilder_bridge.py` | KGBuilder integration (manifest + OWL stub) |
| `scripts/oa_landscape.py` | Open-access landscape analysis |
| `Fusion_KG_Concept.md` | Full theoretical concept document (~1,300 lines) |
| `docs/FINDINGS_REPORT.md` | Detailed analysis results with interpretations |
| `.env` | Neo4j credentials (neo4j/fusion2026) |
| `docker-compose.yml` | Neo4j 5.26 with GDS + APOC |

---

## Dependencies

All in `requirements.txt`. Key additions since the original setup:
- `rdflib>=7.0.0` — OWL ontology generation
- `requests>=2.31.0` — DAQ pipeline HTTP
- `tqdm>=4.66.0` — progress bars
- `powerlaw>=1.5` — Zipf's law fitting

## Known Issues

1. **Eigenvector centrality returns zeros** — ARPACK convergence issue on sparse matrices. Use PageRank instead.
2. **Windows cp1252 encoding** — Unicode characters in console output cause crashes. ASCII fallbacks have been applied to `run_analysis.py` and `daq/pipeline.py`, but other modules may still have Unicode in print statements.
3. **IOP publisher bot protection** — Downloads from iopscience.iop.org are blocked by Radware. The arXiv fallback covers many of these papers.
4. **No CI/CD** — No automated testing or deployment pipeline.

---

## Roadmap — Answer-Gap Reports (post Raúl review, 2026)

The chat agent now performs real GraphRAG: it merges Cypher-extracted KG rows
with abstract excerpts retrieved by an entity-anchored + Lucene fulltext
search (`paper_text` index over title+abstract). Whenever the synthesis LLM
cannot ground an answer it emits a `[MISSING_FROM_GRAPH]` sentinel, which is
captured by [analysis/answer_gap_logger.py](../analysis/answer_gap_logger.py)
and aggregated by
[analysis/answer_gap_report.py](../analysis/answer_gap_report.py).

This produces a *new* class of gap evidence — **questions the system was
asked but could not answer** — orthogonal to the structural / triadic /
diffusion gaps already implemented. The roadmap below ties them together.

### R1 — Auto-publish answer-gap reports

Run `python -m analysis.answer_gap_report` weekly (or on each Streamlit
shutdown) and write `output/answer_gap_report.{json,md}`. Add a third tab to
the chat UI surfacing the top failed-question entities so a human curator
sees, at a glance, which concepts the KG keeps tripping over.

### R2 — Cluster failed questions by linked entities

Two failure modes deserve different treatments:

- **Unlinked questions** (entity linker returned nothing) → the user asked
  about a concept the KG does not represent. Candidate for adding a new
  entity / category, or re-running NER on a focused sub-corpus.
- **Linked-but-unanswerable** (linker found e.g.\ `tokamak` but the answer
  was insufficient) → the entity exists but the relations needed to answer
  ("What fuel does it use?", "Why D-T?") are missing from the typed schema.

Cluster the JSONL log by `linked_entities`-tuple and rank clusters by size:
the largest clusters point at the next typed-relation to add (e.g.\
`:Reaction(name)` linked to devices via `:USES_FUEL`, with reaction
properties `Q_value`, `peak_T`, etc.).

### R3 — Feed gap clusters into `gap_analysis_agent.py`

Add a new hypothesis type `answer_gap` to
[analysis/gap_analysis_agent.py](../analysis/gap_analysis_agent.py) so the
existing gap pipeline (triadic-closure deficit + diffusion + bridge-edge)
can be cross-referenced with the *behavioural* gaps from chat. A concept
that scores high on all four signals (low triadic closure, low PPR
coverage, weak bridge support, AND surfaces in many failed questions)
becomes a top-priority enrichment target.

### R4 — Targeted PDF ingestion driven by gap reports

The `daq/` pipeline already supports DOI-driven ingestion. A future
extension reads the top-K entities from the answer-gap report, queries
OpenAlex for the most-cited papers mentioning each, downloads the PDFs,
re-runs NER on them, and re-loads the KG. This closes the loop:
chat-failure → gap report → focused acquisition → KG enrichment → fewer
chat failures.

### R5 — Typed relations from focused IE

The expert-feedback images (D-T fusion-energy schematic with 80 % in the
neutron, reaction-rate-vs-temperature curve with the D-T peak near
60 keV, ICF vs MCF schematic) all expose a structural blind spot: the
current schema has only `:MENTIONS` and `:CO_OCCURS_WITH`. To answer
"why deuterium-tritium?" the KG needs typed edges such as
`(:Device)-[:USES_FUEL]->(:Reaction)`,
`(:Reaction)-[:HAS_OPTIMAL_TEMPERATURE]->(:Quantity)`, and
`(:Concept)-[:IS_TYPE_OF]->(:ConfinementClass)`. R5 is a focused IE pass
restricted to the highest-frequency entities surfaced by the answer-gap
report, so the labour cost is bounded by the gap evidence rather than
applied to all 51 k entities at once.

---

## Status snapshot — May 2026

After the May 2026 rewrite the chat path looks like this:

```
question
  ├─ EntityLinker (semantic, junk-filtered)
  ├─ _fetch_abstract_context  ── GraphRAG  ──┐
  ├─ GraphCypherQAChain (LLM Cypher) ────────┤
  ├─ _format_evidence(rows + abstracts) ─────┘
  ├─ QA_TEMPLATE final synthesis (LLM)
  └─ if [MISSING_FROM_GRAPH] or zero-evidence
       → answer_gap_logger.log_answer_event
       → answer_gap_report (aggregated)
       → gap_analysis_agent  (new `answer_gap_*` hypotheses)
```

Behavioural QA test set: [tests/qa_test_questions.json](../tests/qa_test_questions.json)
(17 items; current pass rate 13/17 ≈ 76 % with `gemma4:e2b`).

Sentinel pipeline verified end-to-end: 3 sentinel events flowed through to
`output/answer_gap_report.json` and surfaced as 10 `answer_gap_entity`
hypotheses + N `answer_gap_unlinked` hypotheses inside
`output/gap_report.{md,json}`.

### Concrete next directions

**D1 — Fix the `Which X appear most often?` Cypher pattern.**
The runner caught `feedback-02` (devices) failing because the LLM matches
on `"fusion devices"` substring instead of enumerating the
`Nuclear Fusion Device Type` category. Add a new Cypher pattern in
[analysis/llm_graph_qa.py](../analysis/llm_graph_qa.py): when the
question contains `most/top/common/frequent` AND a category name,
generate `MATCH (e)-[:IN_CATEGORY]->(c {name:$cat}) ... ORDER BY mentions DESC`.

**D2 — Quantitative facts in abstracts (`feedback-06`).**
The 80 % neutron / 20 % alpha energy split is a *sentence* in many
abstracts. Add a sentence-level retriever (split abstracts on `.`,
embed sentences, run the question against sentence embeddings) instead
of returning whole-abstract excerpts. This unlocks numeric questions
without any schema changes.

**D3 — Run the QA test set in CI.**
[tests/run_qa_tests.py](../tests/run_qa_tests.py) already produces
`output/qa_test_results.{json,md}`. Wrap it in pytest so a regression
in the Cypher template (e.g. SQL bleed-through) fails the build.

**D4 — Cluster failed questions before reporting.**
Today `answer_gap_report` ranks by raw entity frequency. Group
questions by canonical concept first (e.g. all "cost"-flavoured
questions → one cluster), then surface clusters. This makes the report
actionable: each cluster is one PDF-ingestion / NER-extension job.

**D5 — Promote sentinel decisions to a confidence score.**
Right now the agent emits a binary `[MISSING_FROM_GRAPH]` marker.
Have the QA prompt emit a self-graded coverage score (0–1) per answer
and log it. Then the gap report can rank by *low-confidence* answers
in addition to outright sentinel hits.

**D6 — Loop the report back into `daq/`.**
Add a small driver (`daq/from_gap_report.py`) that reads
`answer_gap_report.json`, takes the top-K offending entities, queries
OpenAlex for the most-cited recent papers mentioning them, and feeds
DOIs into the existing `daq/pipeline.py`. This is the first step that
makes the system self-improving.

**D7 — Typed-relation extraction prompt for the same model.**
Once D6 brings new abstracts in, run a second Ollama prompt over each
abstract that asks specifically for typed triples
`(device, USES_FUEL, fuel)`, `(reaction, HAS_OPTIMAL_TEMPERATURE, T)`,
etc. Store those as new edges in Neo4j with provenance back to the
paper. The chat agent already has Cypher patterns ready for typed
relations once they exist.

**D8 — Surface the gap report in the Streamlit UI.**
Add a tab to [chat_app.py](../chat_app.py) that renders
`answer_gap_report.md` so users (and reviewers) can see in real time
what the system is failing on. Closes the human-in-the-loop on
ontology curation.
