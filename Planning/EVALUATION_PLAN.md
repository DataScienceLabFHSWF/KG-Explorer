# Evaluation Plan — Paper D (Fusion KG / KG-Explorer)

**Goal**: Produce quantitative results for the journal paper on multi-source knowledge graph fusion and analysis.  
**Owner**: Felix (writing lead + fusion analysis), Ferdinand (entity normalization + DAQ), Ole (visualization)  
**Timeline**: Journal paper — no April conference deadline, but results should be ready by May.  
**Note**: This is a private repo.

---

## Prerequisites

### 1. Services (Docker Compose)

```bash
docker compose up -d
```

| Service | Port | Check |
|---------|------|-------|
| Neo4j 5.26.0-community | 7474 (browser), 7687 (bolt) | `curl http://localhost:7474` |

Neo4j is configured with **GDS** (Graph Data Science) and **APOC** plugins.

**Default credentials**: `neo4j` / `fusion2026`

### 2. Data

Current graph statistics:
- ~50,600 entities
- ~249,500 edges

Verify the graph is loaded:
```bash
docker exec -it neo4j cypher-shell -u neo4j -p fusion2026 \
  "MATCH (n) RETURN count(n) AS nodes UNION ALL MATCH ()-[r]->() RETURN count(r) AS edges"
```

### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Known Issue

**ARPACK convergence bug**: Some spectral graph algorithms (e.g., spectral clustering, certain centrality computations) may fail with ARPACK convergence errors on the full graph. Workaround: use `--max-nodes` to subsample, or switch to approximate algorithms.

---

## Phase 1: Entity Normalization (Critical First Step)

**Who**: Ferdinand  
**Time**: ~1 day  
**Script**: `scripts/suggest_entity_merges.py`  
**Priority**: Do this BEFORE any analysis — duplicate entities skew all metrics.

### Step 1: Generate merge suggestions

```bash
python scripts/suggest_entity_merges.py > results/merge_suggestions.json
```

This uses string similarity and embedding-based matching to find potential duplicates.

### Step 2: Review merge suggestions

The output is a JSON list of `(entity_a, entity_b, confidence, reason)` tuples. Review manually — **do not auto-merge without review** (false positives will corrupt the graph).

```python
import json
suggestions = json.load(open("results/merge_suggestions.json"))
# Sort by confidence, review high-confidence merges first
for s in sorted(suggestions, key=lambda x: -x["confidence"]):
    print(f'{s["confidence"]:.2f}  {s["entity_a"]}  ←→  {s["entity_b"]}  ({s["reason"]})')
```

### Step 3: Execute approved merges

```cypher
-- Example merge in Cypher (run via cypher-shell or browser)
MATCH (a:Entity {name: "duplicate_name"}), (b:Entity {name: "canonical_name"})
CALL apoc.refactor.mergeNodes([a, b], {properties: "combine"}) YIELD node
RETURN node
```

### Step 4: Record merge statistics
- Number of entities before/after
- Number of merges approved vs rejected
- Categories of duplicates (acronyms, spelling variants, etc.)

---

## Phase 2: Full Analysis Pipeline

**Who**: Felix  
**Time**: ~4–8 hours (depends on graph size after normalization)  
**Script**: `scripts/run_analysis.py`

### Run complete analysis

```bash
python scripts/run_analysis.py \
  --output results/full_analysis/ \
  2>&1 | tee logs/analysis_full.log
```

### Run specific modules only

```bash
# Skip slow modules during development
python scripts/run_analysis.py --skip spectral clustering

# Run only specific modules
python scripts/run_analysis.py --only "degree_distribution community_detection centrality"
```

### 14 Analysis Modules

| # | Module | Output | Paper Use |
|---|--------|--------|-----------|
| 1 | Degree distribution | Power-law fit, plots | 4.1 Structural |
| 2 | Community detection | Louvain/Leiden communities | 4.2 Communities |
| 3 | Centrality analysis | Betweenness, PageRank, closeness | 4.1 Structural |
| 4 | Clustering coefficients | Local + global clustering | 4.1 Structural |
| 5 | Connected components | Component sizes, giant component ratio | 4.1 Structural |
| 6 | Path analysis | Diameter, avg shortest path | 4.1 Structural |
| 7 | Spectral analysis | Eigenvalue spectrum | 4.3 Spectral ⚠ |
| 8 | Node classification | Type distribution, label propagation | 4.2 Communities |
| 9 | Edge type analysis | Relation type distribution | 4.1 Structural |
| 10 | Temporal analysis | Growth over time (if timestamps) | 4.4 Temporal |
| 11 | Motif analysis | Common subgraph patterns | 4.3 Patterns |
| 12 | Robustness analysis | Attack tolerance, percolation | 4.3 Robustness |
| 13 | Embedding analysis | Node2Vec/GDS embeddings | 4.2 Embeddings |
| 14 | Comparison metrics | Cross-source statistics | 4.4 Fusion Quality |

> ⚠ Module 7 (Spectral) may trigger ARPACK bug — use `--max-nodes 10000` if it fails on the full graph.

---

## Phase 3: Data Quality Assessment (DAQ Pipeline)

**Who**: Ferdinand  
**Time**: ~2 hours  
**Pipeline**: DAQ modules (already exist in codebase)

Run the data quality assessment pipeline:

```bash
python scripts/run_analysis.py --only "daq"
```

Metrics to capture:
- **Completeness**: % of entities with all expected properties filled
- **Consistency**: % of edges with valid source + target types
- **Timeliness**: Age distribution of data sources
- **Accuracy**: Cross-reference sample against authoritative sources
- **Uniqueness**: Post-normalization duplicate rate (should be near 0)

---

## Phase 4: Cross-Source Fusion Analysis

**Who**: Felix  
**Time**: ~1 day

This is unique to Paper D — analyzing how well data from multiple sources (NER extractions, manual entries, imported datasets) fuse into a coherent graph.

### Source attribution analysis

```cypher
-- Count entities by source
MATCH (n) RETURN n.source AS source, count(n) AS count ORDER BY count DESC

-- Count cross-source edges
MATCH (a)-[r]->(b) WHERE a.source <> b.source
RETURN a.source, b.source, count(r) AS cross_edges ORDER BY cross_edges DESC
```

### NER-to-KG quality

```bash
# Analyze NER extraction quality
python scripts/load_ner_json_to_neo4j.py --analyze-only
```

### Open Access landscape

```bash
python scripts/oa_landscape.py > results/oa_landscape.json
```

---

## Phase 5: Year-Based Analysis (If Temporal Data Exists)

**Who**: Ole  
**Time**: ~4 hours

```bash
python scripts/run_analysis.py --year 2024 --output results/analysis_2024/
python scripts/run_analysis.py --year 2025 --output results/analysis_2025/
```

Compare structural metrics across years to show KG evolution.

---

## Phase 6: Visualization (For Paper Figures)

**Who**: Ole  
**Time**: ~1 day

Generate publication-quality figures:

1. **Network overview** — Full graph layout (use `--max-nodes 5000` for readability)
2. **Community structure** — Color-coded by Louvain community
3. **Degree distribution** — Log-log plot with power-law fit line
4. **Source attribution** — Sankey diagram showing entity flow from sources
5. **Fusion quality** — Heatmap of cross-source edge density
6. **Growth timeline** — If temporal data exists

Use Neo4j Browser or Bloom for interactive exploration, export SVG for paper.

---

## Output Checklist (What Goes Into the Paper)

| Result | Script/Source | Paper Section |
|--------|--------------|---------------|
| Entity normalization stats | `suggest_entity_merges.py` | 3.2 Data Cleaning |
| Graph structural metrics (14 modules) | `run_analysis.py` | 4.1 Structural Analysis |
| Community detection results | `run_analysis.py` (Louvain/Leiden) | 4.2 Community Analysis |
| Power-law degree distribution | `run_analysis.py` (module 1) | 4.1 Structural Analysis |
| DAQ quality scores | DAQ pipeline | 4.3 Quality Assessment |
| Cross-source fusion metrics | Cypher queries + analysis | 4.4 Fusion Analysis |
| NER extraction quality | `load_ner_json_to_neo4j.py` | 3.1 Data Collection |
| OA landscape analysis | `oa_landscape.py` | 4.4 Fusion Analysis |
| Network visualization figures | Neo4j Browser / matplotlib | Figures 2-5 |
| Before/after normalization comparison | Manual recording | 3.2 Data Cleaning |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| ARPACK convergence error | Use `--max-nodes 10000` or skip spectral module: `--skip spectral` |
| Neo4j OOM | Increase heap in `docker-compose.yml`: `NEO4J_server_memory_heap_max__size=4G` |
| GDS plugin not found | Verify `NEO4J_PLUGINS=["graph-data-science"]` in Docker config |
| APOC not available | Verify `NEO4J_PLUGINS=["apoc"]` in Docker config |
| Slow community detection | Use `--max-nodes` to subsample or use approximate Leiden |
| `suggest_entity_merges.py` slow | Expected — O(n²) comparison. Run overnight for full graph |
| No temporal data | Skip Phase 5, note as limitation in paper |
| Neo4j password wrong | Default is `fusion2026` — check `docker-compose.yml` |

---

## Important: No Unit Tests

This repo currently has **zero unit tests**. For any new analysis scripts or modifications:
1. Test on a subsampled graph first (`--max-nodes 1000`)
2. Verify outputs manually before running on full graph
3. Consider adding basic smoke tests for each module

---

*Created: 2026-03-16*
