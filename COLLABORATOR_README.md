# Fusion Energy Knowledge Graph — Project Overview

## What Is This?

We are building a **mathematical analysis and extension layer** on top of a Knowledge Graph (KG) of nuclear fusion energy research, originally constructed by Loreti et al. ([arXiv:2504.07738](https://arxiv.org/abs/2504.07738)).

The KG contains ~50,000 entities and ~250,000 co-occurrence relationships extracted via LLM-based Named Entity Recognition from ~8,400 fusion energy paper abstracts.

## Why Does It Matter?

Current scientific literature databases (Scopus, Web of Science, Google Scholar) provide search and citation links — but no **structured, domain-ontology-grounded knowledge graph** with entity-level relationships, mathematical analysis of knowledge structure, or automated gap detection.

We aim to build infrastructure that can:
1. **Reveal hidden structure** in a scientific domain using graph theory, topology, spectral analysis, and information theory.
2. **Detect knowledge gaps** — combinations of concepts that have never been studied together but probably should be.
3. **Answer domain questions** via a hybrid Graph-RAG chatbot grounded in both the KG and full-text embeddings.
4. **Generalise** the pipeline to any scientific domain (semiconductors, quantum computing, etc.).

## What's Implemented So Far

| Module | Status | What It Does |
|---|---|---|
| **Data loading** | Done | NER JSON → Neo4j (Docker) |
| **Graph analysis** | Done | Degree distribution, power-law fit, centralities, Louvain communities, k-core |
| **Persistent homology (TDA)** | Done | Simplicial complex filtration, persistence diagrams, Betti curves, knowledge gap detection |
| **Spectral analysis** | Done | Laplacian eigenvalues, Fiedler vector, spectral clustering, GFT, heat kernel |
| **Formal Concept Analysis (FCA)** | Done | Category lattice, co-occurrence heatmap, Duquenne–Guigues implications |
| **Information theory** | Done | Von Neumann entropy, mutual information, category entropy, temporal evolution |
| **Void extraction** | Done | Maps H1 loops and H2 voids back to concrete entity names |
| **Structural holes** | Done | Burt's structural holes, effective size, bridge concept ranking |
| **Link prediction** | Done | Missing edge detection via Adamic-Adar, Jaccard, common neighbours |
| **Interactive explorer** | Done | Pyvis HTML visualisation with community colouring and gap highlights |
| **Gap analysis agent** | Done | Integrated gap synthesis with optional LLM enhancement |
| **Zipf analysis** | Done | Power-law validation of entity mention frequencies |
| **Ontology generator** | Done | OWL 2 ontology from KG structure + FCA implications (2,487 triples) |
| **Graph query interface** | Done | 7 query modes: lookup, neighbours, path, bridge, trend, community, gaps |
| **DAQ pipeline** | Done | OpenAlex enrichment, arXiv fallback, PDF download, KGBuilder packaging |
| **OA landscape analysis** | Done | Open-access status breakdown, yearly trends, publisher/repository analysis |

## How to Run

```bash
# 1. Start Neo4j
docker compose up -d

# 2. Load data
.venv\Scripts\python scripts\load_ner_json_to_neo4j.py

# 3. Run analysis (all modules)
.venv\Scripts\python run_analysis.py

# 4. Results appear in output/
```

## Key Documents

- **[Fusion_KG_Concept.md](Fusion_KG_Concept.md)** — Full theoretical concept document (~1,300 lines) covering all mathematical frameworks, gap detection strategies, ontology engineering, and long-term vision.
- **[docs/implementation_guides.md](docs/implementation_guides.md)** — Practical code examples and step-by-step procedures for each analysis framework.
- **[README.md](README.md)** — Detailed technical setup and module documentation.
- **[presentation/](presentation/)** — LaTeX Beamer presentation of the concept and results.

## What's Next

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the full handoff/roadmap document. Key priorities:

1. **Add test coverage** — zero unit tests exist currently
2. **Entity normalisation** — merge singular/plural pairs identified by link prediction
3. **Scale up DAQ** — download all ~2,000 open-access PDFs
4. **KGBuilder integration** — feed PDFs + ontology into full-text extraction
5. **Hybrid RAG chatbot** — add LLM synthesis layer to the graph query interface
6. **Upper ontology linking** — map OWL classes to BFO/EMMO

## Tech Stack

- **Neo4j 5.26** (Docker, with GDS + APOC plugins)
- **Python 3.10** — NetworkX, SciPy, ripser, persim, concepts, scikit-learn, matplotlib
- **LaTeX** — Beamer (fhswf theme) for presentations

## Contact

This project is part of ongoing research at FH Südwestfalen.
