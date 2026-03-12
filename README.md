# Fusion Knowledge Graph — Mathematical Analysis Suite

A complete pipeline for loading Named Entity Recognition (NER) data from nuclear fusion energy papers into **Neo4j**, then running a suite of mathematical analyses with automated visualisation.

> **Note:** The original paper source files and any presentation materials are purposely **not** included in this repository.  They are available from the STFC data archive:
> https://edata.stfc.ac.uk/items/61b38908-2523-4cc5-bc4b-51bad6edb7f4


Based on the methodology described in *Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information* (Loreti et al., [arXiv:2504.07738](https://arxiv.org/abs/2504.07738)).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Data Loading](#data-loading)
5. [Analysis Modules](#analysis-modules)
6. [Output Reference](#output-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement        | Version   |
|--------------------|-----------|
| Python             | ≥ 3.12    |
| Docker + Compose   | ≥ 24.x   |
| pip                | ≥ 23.x   |

---

## Quick Start

```powershell
# 1. Start Neo4j (runs in background)
docker compose up -d

# 2. Wait for Neo4j to become healthy (~30 s)
docker compose ps          # STATE should be "healthy"

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Load the NER data into Neo4j
python scripts/load_ner_json_to_neo4j.py --data-dir data

# 5. Run the full analysis pipeline
python run_analysis.py

# 6. Open the Neo4j browser (optional)
#    http://localhost:7474  (user: neo4j, password: fusion2026)
```

All visualisations and exports are written to the `output/` directory.

---

## Project Structure

```
FusionData/
├── docker-compose.yml          # Neo4j 5.26 + GDS plugin
├── .env                        # Neo4j connection credentials
├── requirements.txt            # Python dependencies
├── run_analysis.py             # Main orchestrator (CLI)
│
├── scripts/
│   ├── load_ner_json_to_neo4j.py   # JSON → Neo4j loader
│   ├── suggest_entity_merges.py    # Singular/plural merge candidates
│   └── oa_landscape.py             # Open-access status analysis
│
├── analysis/                   # Analysis package
│   ├── __init__.py
│   ├── neo4j_utils.py          # Shared driver, queries, helpers
│   ├── graph_analysis.py       # Graph-theoretic metrics
│   ├── tda_analysis.py         # Persistent homology (TDA)
│   ├── spectral_analysis.py    # Spectral graph theory
│   ├── fca_analysis.py         # Formal Concept Analysis
│   ├── information_theory.py   # Information-theoretic measures
│   ├── void_extraction.py      # Map TDA features to entity names
│   ├── structural_holes.py     # Burt's structural holes
│   ├── link_prediction.py      # Missing edge detection
│   ├── interactive_explorer.py # Pyvis HTML visualisation
│   ├── gap_analysis_agent.py   # Integrated gap synthesis (LLM-optional)
│   ├── zipf_analysis.py        # Zipf's law validation
│   ├── ontology_generator.py   # OWL 2 ontology from KG structure
│   └── graph_qa.py             # Graph query interface (7 modes)
│
├── data/                       # NER JSON source files
│   ├── Tokamak_1_NER.json
│   ├── Stellarator_1_NER.json
│   └── ...  (15 files total)
│
├── daq/                        # Data Acquisition pipeline
│   ├── __init__.py
│   ├── __main__.py             # CLI entry-point (python -m daq)
│   ├── doi_extraction.py       # NER JSON → PaperRecord catalogue
│   ├── openalex_client.py      # OpenAlex API client for OA metadata
│   ├── downloader.py           # Rate-limited PDF downloader
│   ├── kgbuilder_bridge.py     # Package output for KGBuilder
│   └── pipeline.py             # End-to-end orchestrator
│
├── output/                     # Generated plots, CSVs, JSONs, OWL
│   └── (created at runtime)
│
├── docs/                       # Documentation
│   ├── FINDINGS_REPORT.md      # Detailed analysis results
│   ├── NEXT_STEPS.md           # Handoff / roadmap for next developer
│   ├── WHAT_IS_THIS.md         # Quick explainer (English)
│   ├── WAS_IST_DAS.md          # Quick explainer (German)
│   ├── implementation_guides.md
│   └── internal_working_paper.tex
│
├── notebooks/
│   └── exploration.ipynb       # Ad-hoc exploration notebook
│
└── Fusion_KG_Concept.md        # Full theoretical concept document
```

---

## Data Loading

The loader script reads all `*_NER.json` files from the `data/` directory and creates the following Neo4j graph model:

```
(:Paper)  -[:HAS_FIELD]->     (:Field)
(:Paper)  -[:MENTIONS {count}]->  (:Entity)
(:Entity) -[:IN_CATEGORY]->   (:Category)
(:Entity) -[:CO_OCCURS_WITH {weight, papers}]- (:Entity)
```

### Loader CLI Options

In addition to the loader above, a small helper script is provided to find
likely singular/plural entity pairs by inspecting the link prediction output:

```powershell
python scripts/suggest_entity_merges.py      # prints suggestions
python scripts/suggest_entity_merges.py --output candidates.csv
```

It scans `output/predicted_links.csv` for pairs where the names differ only
by an "s"/"es" suffix, e.g. `tokamak <-> tokamaks`.

### Loader CLI Options

```
python scripts/load_ner_json_to_neo4j.py [options]

  --data-dir DIR       Path to JSON directory (default: data)
  --uri URI            Neo4j bolt URI (default: from .env)
  --user USER          Neo4j username (default: from .env)
  --password PASS      Neo4j password (default: from .env)
  --drop               Drop existing data before loading
```

---

## Data Acquisition Pipeline (`daq/`)

The `daq` package provides a full-text acquisition pipeline that scans the existing NER JSON data, resolves open-access metadata via [OpenAlex](https://openalex.org), downloads available PDFs, and packages them for the [KnowledgeGraphBuilder](https://github.com/your-org/KnowledgeGraphBuilder) pipeline.

### Quick Start

```powershell
# Run on a small test batch
python -m daq --data-dir data/ --output data/fulltexts/ --limit 10 --email you@uni.de

# Full run (8,176 unique papers)
python -m daq --data-dir data/ --output data/fulltexts/ --email you@uni.de
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `doi_extraction.py` | Parse NER JSONs → `PaperRecord` catalogue, deduplicate by DOI/title |
| 2 | `openalex_client.py` | Resolve DOIs + title search → OA status, PDF URLs, licenses |
| 3 | `downloader.py` | Rate-limited, polite PDF download with resume support |
| 4 | `kgbuilder_bridge.py` | Copy PDFs + generate `manifest.json` + OWL ontology stub |

### Output

```
data/fulltexts/
├── catalogue.json              # Full paper catalogue (8,176 records)
├── catalogue_enriched.json     # After OpenAlex enrichment
├── pdfs/                       # Downloaded PDFs
└── kgbuilder_input/
    ├── docs/                   # PDFs ready for kgbuilder run --docs
    ├── manifest.json           # paper_id → PDF mapping for KG linkage
    └── ontology/
        └── fusion_ner.owl      # Auto-generated OWL stub from NER categories
```

### KGBuilder Integration

```powershell
# After DAQ completes, feed into KGBuilder:
kgbuilder run --docs data/fulltexts/kgbuilder_input/docs/ \
              --ontology data/fulltexts/kgbuilder_input/ontology/fusion_ner.owl
```

### API Usage

```python
from daq import DAQPipeline

pipe = DAQPipeline(data_dir="data/", output_dir="data/fulltexts/", email="you@uni.de")
stats = pipe.run()
print(stats)
```

---

## Analysis Modules

### 1. Graph Analysis (`analysis/graph_analysis.py`)

Classical graph-theoretic metrics on the entity co-occurrence network.

| Metric | Method |
|--------|--------|
| Degree distribution | Log-log histogram + power-law fit (Clauset-Shalizi-Newman) |
| Centrality rankings | PageRank, betweenness (sampled), closeness, eigenvector |
| Community detection | Louvain algorithm (resolution = 1.0) |
| k-core decomposition | Coreness distribution + cumulative membership |
| Clustering coefficients | Local clustering coefficient distribution |

### 2. Topological Data Analysis (`analysis/tda_analysis.py`)

Persistent homology of the co-occurrence network via [Ripser](https://ripser.scikit-tda.org/).

| Output | Description |
|--------|-------------|
| Persistence diagrams | Birth-death scatter for H0, H1, H2 |
| Barcodes | Top-50 most persistent features per dimension |
| Betti curves | Betti numbers as functions of the filtration parameter |

The subgraph is trimmed to the top-N nodes by weighted degree (configurable via `--max-nodes`).

### 3. Spectral Analysis (`analysis/spectral_analysis.py`)

Spectral graph theory on the normalised Laplacian.

| Output | Description |
|--------|-------------|
| Eigenvalue spectrum | Smallest k eigenvalues + spectral gap |
| Fiedler vector | Histogram and sorted-values plot of the second eigenvector |
| Spectral clustering | k-means on spectral embedding (2D scatter + cluster sizes) |
| Graph Fourier Transform | GFT coefficient magnitudes + energy distribution |
| Heat kernel trace | tr(e^{-tL}) across logarithmic time scales |

### 4. Formal Concept Analysis (`analysis/fca_analysis.py`)

Category structure analysis using the [concepts](https://github.com/xflr6/concepts) library.

| Output | Description |
|--------|-------------|
| Category statistics | Entity counts per category, multi-category distribution |
| Co-occurrence heatmap | Annotated NER-category co-occurrence matrix |
| FCA lattice | Duquenne-Guigues basis of implications |
| Implication graph | Directed graph of category-to-category implications |

### 5. Information Theory (`analysis/information_theory.py`)

Information-theoretic measures on the graph and category structure.

| Output | Description |
|--------|-------------|
| Von Neumann entropy | S(G) approximated from Laplacian eigenvalues |
| Mutual information | Pairwise MI between NER categories via edge weights |
| Category entropy | Shannon entropy of each category's co-occurrence partners |
| Temporal evolution | Category popularity and share over publication years |

### 6. Void Extraction (`analysis/void_extraction.py`)

Maps persistent homology features (H1 loops, H2 voids) back to concrete entity names.

### 7. Structural Holes (`analysis/structural_holes.py`)

Burt's structural hole analysis: betweenness, clustering, effective size, bridge concept ranking.

### 8. Link Prediction (`analysis/link_prediction.py`)

Missing edge detection via Adamic-Adar, Jaccard coefficient, common neighbours.

### 9. Interactive Explorer (`analysis/interactive_explorer.py`)

Pyvis HTML visualisations of the graph (top-500 entities) with community colouring and gap highlights.

### 10. Gap Analysis Agent (`analysis/gap_analysis_agent.py`)

Integrated gap synthesis combining all detected gaps into structured research hypotheses. Supports optional LLM enhancement.

### 11. Zipf Analysis (`analysis/zipf_analysis.py`)

Validates that entity mention frequencies follow Zipf's law via Clauset-Shalizi-Newman MLE fitting.

### 12. Ontology Generator (`analysis/ontology_generator.py`)

Builds an OWL 2 ontology from Neo4j categories, co-occurrence patterns, FCA implications, and exemplar individuals. Outputs `fusion_ontology.owl` (RDF/XML) and `fusion_ontology.ttl` (Turtle).

### Graph Query Interface (`analysis/graph_qa.py`)

Interactive query tool with 7 modes: entity lookup, neighbours, shortest path, bridge concepts, trend, community, and gap search.

```powershell
python -m analysis.graph_qa "tokamak"                    # entity lookup
python -m analysis.graph_qa "tokamak -> stellarator"      # shortest path
python -m analysis.graph_qa "neighbours of plasma"        # co-occurrences
python -m analysis.graph_qa "trend deuterium"             # yearly mentions
python -m analysis.graph_qa "gaps plasma"                 # research gaps
python -m analysis.graph_qa --mode community plasma       # community members
python -m analysis.graph_qa --json "tokamak"              # JSON output
```

---

## Output Reference

After a full run, `output/` contains:

### Visualisations (PNG)

| File | Module | Description |
|------|--------|-------------|
| `01_degree_distribution.png` | graph | Log-log degree distribution + power-law fit |
| `02_centrality_rankings.png` | graph | Top-20 entities by four centralities |
| `03_pagerank_vs_betweenness.png` | graph | Scatter of PageRank vs betweenness |
| `04_community_sizes.png` | graph | Louvain community sizes |
| `05_kcore_decomposition.png` | graph | k-core coreness distribution |
| `06_clustering_coefficients.png` | graph | Local clustering coefficient histogram |
| `07_persistence_diagrams.png` | tda | Birth–death persistence diagrams |
| `08_persistence_barcodes.png` | tda | Top-50 persistence barcodes |
| `09_betti_curves.png` | tda | Betti number curves |
| `10_laplacian_spectrum.png` | spectral | Eigenvalue spectrum |
| `11_fiedler_vector.png` | spectral | Fiedler vector (2nd eigenvector) |
| `12_spectral_clustering.png` | spectral | 2D spectral clustering scatter |
| `13_spectral_cluster_sizes.png` | spectral | Spectral cluster membership |
| `14_gft_coefficients.png` | spectral | Graph Fourier Transform magnitudes |
| `14_gft_energy.png` | spectral | GFT cumulative energy distribution |
| `15_heat_kernel_trace.png` | spectral | Heat kernel trace over time |
| `15_zipf_law.png` | zipf | Rank-frequency log-log plot with power-law fit |
| `16_category_entity_counts.png` | fca | Entities per NER category |
| `16_zipf_deviation.png` | zipf | Over/under-represented entity heatmap |
| `17_categories_per_entity.png` | fca | Multi-category entity distribution |
| `18_category_cooccurrence.png` | fca | Category co-occurrence heatmap |
| `19_implication_graph.png` | fca | FCA implication graph |
| `20_category_mutual_information.png` | info | Category pairwise MI heatmap |
| `21_category_entropy.png` | info | Category diversity (Shannon entropy) |
| `22_temporal_category_evolution.png` | info | Category popularity over time |
| `23_void_entity_network.png` | voids | Subgraph of entities forming persistent voids |
| `24_loop_entity_network.png` | voids | Subgraph of entities forming persistent loops |
| `25_gap_persistence_vs_degree.png` | voids | Persistence vs hub-ness scatter |
| `26_structural_holes_scatter.png` | holes | Betweenness vs clustering scatter |
| `27_effective_size_distribution.png` | holes | Distribution of effective size |
| `28_bridge_concepts_network.png` | holes | Network of top bridge concepts |
| `29_link_prediction_scores.png` | links | Top predicted missing links |
| `30_predicted_links_network.png` | links | Subgraph of high-confidence predictions |

### Data Exports

| File | Format | Description |
|------|--------|-------------|
| `centralities.csv` | CSV | All centrality scores for every entity |
| `communities.csv` | CSV | Entity to community mapping |
| `spectral_clusters.csv` | CSV | Entity to spectral cluster mapping |
| `structural_holes.csv` | CSV | Betweenness, clustering, effective size |
| `predicted_links.csv` | CSV | Predicted missing edges with scores |
| `fca_implications.json` | JSON | Duquenne-Guigues implication basis |
| `knowledge_gaps.json` | JSON | H1 loops and H2 voids with entity names |
| `gap_report.json` | JSON | Integrated gap synthesis (38 hypotheses) |
| `zipf_stats.json` | JSON | Power-law fit statistics |
| `fusion_ontology.owl` | RDF/XML | OWL 2 ontology (2,487 triples) |
| `fusion_ontology.ttl` | Turtle | Human-readable ontology serialisation |
| `interactive_graph.html` | HTML | Pyvis interactive graph explorer |
| `interactive_gaps.html` | HTML | Pyvis explorer with gap highlights |
| `gap_report.md` | Markdown | Human-readable gap summary |

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt endpoint |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | `fusion2026` | Password |
| `NEO4J_DATABASE` | `neo4j` | Database name |

### Orchestrator CLI

```
python run_analysis.py [options]

  --skip MODULE [MODULE ...]    Skip specific modules
  --only MODULE [MODULE ...]    Run only specific modules
  --max-nodes N                 TDA subgraph size limit (default: 800)
  --year Y                      Restrict analyses to papers published in year Y

Available modules: graph, tda, spectral, fca, information, voids, holes,
                   links, explorer, gaps, zipf, ontology
```

Examples:

```powershell
# Skip TDA (slow on large graphs)
python run_analysis.py --skip tda

# Only run graph and spectral analysis
python run_analysis.py --only graph spectral

# Increase TDA subgraph to 1200 nodes
python run_analysis.py --max-nodes 1200

# Analyse only papers from 2022
python run_analysis.py --year 2022

# Run the new modules only
python run_analysis.py --only zipf ontology

# Combine filters
python run_analysis.py --year 2020 --only graph tda
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ServiceUnavailable` from Neo4j driver | Ensure Docker is running: `docker compose ps`. Wait for `healthy` status. |
| `AuthenticationError` | Check `.env` credentials match `docker-compose.yml` `NEO4J_AUTH` value. |
| `No module named 'ripser'` | `pip install -r requirements.txt` — ripser requires a C compiler (MSVC on Windows). |
| Empty plots | Run the loader first: `python scripts/load_ner_json_to_neo4j.py --data-dir data` |
| Docker volume reset | `docker compose down -v` removes data volumes. Re-run the loader afterwards. |

### Resetting Everything

```powershell
docker compose down -v          # remove Neo4j data
Remove-Item -Recurse output     # remove analysis outputs
docker compose up -d             # fresh start
python scripts/load_ner_json_to_neo4j.py --data-dir data
python run_analysis.py
```

---

## License

Research code — see the source paper for citation details.
