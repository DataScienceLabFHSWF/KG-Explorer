# Agentic Deep Graph Reasoning (arXiv 2502.13025v1)

## What this paper is about

This paper describes an autonomous, iterative graph expansion framework in which an LLM is used not only to answer questions, but to generate, integrate, and refine a growing knowledge graph over many iterations.

Key elements:
- recursive graph construction with feedback loops
- graph-native reasoning tokens and graph extraction from model output
- emergence of scale-free structure, hubs, bridge nodes, and modularity
- evaluation using graph evolution metrics (degree growth, modularity, shortest path distribution, betweenness, k-core, articulation points)
- agentic compositional reasoning over longest shortest paths to generate novel hypotheses
- explicit use of graph structure to improve LLM answers and reduce hallucination

## How it relates to our KG evaluation methods

Our repository already contains several of the same building blocks:
- a Neo4j-based KG with `analysis.graph_qa.py` for structured query modes
- a Streamlit frontend and LLM graph QA wrapper in `analysis/llm_graph_qa.py`
- semantic entity linking with `analysis/entity_linker.py`

The paper extends that type of KG work in two major directions:
1. **Graph expansion as itself the task**: the KG is grown iteratively by an agent, rather than just queried.
2. **Graph topology as a primary evaluation signal**: emergent structure is measured continuously, not just query accuracy.

That means our current setup is well-aligned for this paper’s ideas, but we can improve by treating the graph itself as an evolving object and tracking how it changes in response to LLM-guided reasoning.

## What we can learn from it

### Evaluation lessons

The paper shows that the following metrics are useful for KG evaluation beyond simple retrieval:
- number of nodes and edges over iterations
- average degree and maximum degree growth
- Louvain modularity and number of communities
- average shortest path length and graph diameter
- degree assortativity
- global transitivity (clustering)
- k-core index and size of dense core
- betweenness centrality and articulation points
- persistence of bridge nodes / interdisciplinary connectors
- distribution of node centralities and shortest path lengths

These are exactly the kinds of metrics we can build for our fusion KG evaluation pipeline.

### Architectural lessons

The paper’s agentic workflow suggests several practical improvements:
- use the current KG to seed subsequent queries, rather than only answering a single query
- extract a local graph from LLM reasoning output and merge it into the global KG
- generate follow-up prompts based on new graph structure
- explicitly prefer entity names and verified graph concepts to reduce hallucination
- use path-based reasoning, especially longest-shortest-path analysis, to identify richer inference chains
- compare LLM answers produced with graph context vs without, using graph-utilization and reasoning quality metrics

### What is most relevant for us

The most immediately transferable ideas are:
- build a `graph evolution evaluation` module that computes the same structural metrics used in the paper
- add a lightweight recursive graph expansion loop as an experimental mode in our LLM KG agent
- implement a “graph-enhanced answer scoring” routine to compare pure LLM output vs graph-guided answers
- add bridge-node and hub-node analysis to our evaluation reports
- use longest-shortest-path extraction as a debugging / hypothesis-generation tool

## Can we implement some of it?

Yes. The most feasible near-term implementations are:

1. **Graph evolution metrics**
   - add functions to compute modularity, average degree, degree distribution, shortest path length, diameter, betweenness, articulation points, and k-core statistics
   - include those metrics in our existing `analysis/graph_qa.py` or a new `analysis/graph_evaluation.py`

2. **Graph-guided answer evaluation**
   - extend `analysis/llm_graph_qa.py` to log whether answers came from graph context, and compare against a fallback without graph evidence
   - record `graph_utilization`, `reasoning_depth`, and `novelty` markers in debug logs or evaluation reports

3. **Recursive LLM-to-graph expansion experiment**
   - add an optional mode that runs iterative question generation and graph merge loops
   - even a short 10-20 iteration loop would let us study structural changes and hub emergence
   - this can be experimental, separate from the main KG QA pipeline

4. **Bridge/hub analysis and longest-path reasoning**
   - enable our system to identify high-betweenness connectors and stable hubs
   - use longest shortest-path computations as a way to generate richer downstream queries

## Suggested docs placement and next step

I created this new note in `docs/agentic_deep_graph_reasoning.md` so we can keep the paper summary and implementation ideas together.

If we want to turn this into code next, the most useful next step is to add a new evaluation module such as `analysis/graph_evolution.py` and wire it into `run_analysis.py` so structural metrics are automatically generated.

## Concrete proposal for our project

Add a new evaluation workflow that computes:
- `node_count`, `edge_count`
- `avg_degree`, `max_degree`
- `louvain_modularity`, `community_count`
- `avg_shortest_path`, `diameter`
- `betweenness_mean`, `articulation_count`
- `k_core_index`, `k_core_size`
- `bridge_node_persistence`

Then add a small experimental recursive run mode in `analysis/llm_graph_qa.py` that:
- uses the LLM to generate a follow-up query from the last graph state
- extracts new graph triples from the response
- merges them into the KG
- repeats for N iterations

That would bring our KG evaluation much closer to the paper’s agentic reasoning approach.

---

## Implementation status — gap-detection extensions

We have implemented two new analysis modules that operationalise several of the ideas above. The full pipeline now produces eleven distinct gap types, aggregated into a single hypothesis report.

### Module: `analysis/semantic_gaps.py`

Two complementary detectors:

1. **Semantic non-co-occurrence** — sentence-transformer cosine over the existing entity-linker cache. Pairs in the cosine band `[0.55, 0.80]` that never appear together in a paper become candidate gaps. The upper bound discards near-synonyms (plurals, hyphenation, unicode and Greek-letter variants, word-order permutations) which are aggressively filtered. Output: `output/semantic_gaps.csv`.
2. **Cross-community sparse bridges** — for every pair of Louvain communities ≥ 20 entities, we compare observed inter-community edge weight against the configuration-model expectation `(deg_i · deg_j) / 2m` and flag pairs with large positive `expected − observed`. Communities are labelled by their three highest-PageRank entities. Output: `output/community_bridge_gaps.csv`.

### Module: `analysis/advanced_gaps.py`

Four mathematically distinct detectors, each producing its own CSV/JSON output and feeding hypothesis cards to the gap agent.

| # | Method | Mathematics | Output | Hypothesis type |
|---|---|---|---|---|
| 1 | Longest shortest paths (Buehler 2025 §4) | BFS sample of `O(sample · V)` from random sources in the giant component; longest pairs become Step A→D reasoning seeds | `longest_paths.json` | `reasoning_chain` |
| 2 | Forman–Ricci edge curvature | `κ_F(u,v) = 4 − deg(u) − deg(v) + 3·|N(u) ∩ N(v)|`; we filter to balanced edges (max-deg/min-deg ≤ 5) so the result is not just hub-spoke noise | `edge_curvature.csv` | `bottleneck_edge` |
| 3 | Temporal hub trajectories (Buehler 2025 §5) | Per-year weighted degree pulled from `CO_OCCURS_WITH.papers`, log-slope over the last 5 years (last calendar year dropped to avoid partial-crawl bias), classified `emerging` / `stable` / `stalled` at ±0.5 σ | `entity_trajectories.csv` | `emerging_front`, `stalled_hub` |
| 4 | Articulation points (Tarjan 1972, Buehler 2025 §6) | Run on the *backbone* graph (edges with weight ≥ 2) so we get genuine cut vertices, not random hub artefacts; ranked by size of the second-largest fragment that appears after removal | `articulation_points.csv` | `fragile_bridge` |

A shared `_is_noisy_entity()` helper drops year-strings, dates, and pure-numeric NER artefacts before any structural analysis to keep results interpretable.

### Updated module: `analysis/gap_analysis_agent.py`

- Aggregator now loads all six new outputs alongside the existing TDA / structural-hole / link-prediction / FCA results.
- `generate_hypotheses()` emits six new hypothesis types (`semantic_gap`, `community_bridge_gap`, `reasoning_chain`, `bottleneck_edge`, `emerging_front`, `stalled_hub`, `fragile_bridge`).
- LLM enrichment now tries **local Ollama first** (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, defaults match the chat agent), then falls back to OpenAI, then no-ops cleanly. The chosen `backend` is recorded on the LLM hypothesis card.
- `gap_report.md` rendering knows the new emoji labels.

### Pipeline order

`run_analysis.py` registers two new modules in dependency order:

```
graph → tda → spectral → fca → information → voids → holes → links →
explorer → semgaps → advgaps → gaps → zipf → ontology → communities → embeddings
```

The first run-through can be done with:

```
python run_analysis.py --only graph holes links semgaps advgaps gaps
```

This requires:
- `output/communities.csv` (from `graph`) for community-bridge gaps
- `output/centralities.csv` (from `graph`) for community labels
- `output/entity_linker_cache.pkl` (built lazily by the chat app or `EntityLinker(...)` constructor) for semantic gaps

### Current run statistics

After running the new modules end-to-end against the 50,595-node KG:

| Detector | Count |
|---|---|
| Semantic gaps (cosine ∈ [0.55, 0.80]) | 200 |
| Community bridge gaps | 100 |
| Longest reasoning chains | 25 (length 5 hops) |
| Forman bottleneck edges (balanced) | 200 |
| Emerging entities | 30 |
| Stalled entities | 30 |
| Articulation points (backbone w ≥ 2) | 391 found, top 100 reported |
| Triadic-closure deficit pairs | 200 |
| Bridge edges (2-edge-cuts on backbone) | 933 found, top 100 reported |
| PPR reachability gaps (25 hubs) | 33 |
| Heat-kernel reachability gaps (25 hubs, t=3) | 11 |

Total hypothesis cards emitted by the gap agent: **123 across 15 categories**, persisted to `output/gap_report.json` and `output/gap_report.md`.

### Mathematical methods *not yet* implemented but considered

These rank next in usefulness vs. cost:

- **Degree-corrected SBM residuals** — strictly stronger than the configuration-null `community_bridge_gaps` but requires `graspologic` and several minutes of compute.
- **Multi-scale heat-kernel sweeps** — currently only `t=3` is reported; sweeping `t ∈ {1, 5, 20}` would expose local-vs-global gap regimes but multiplies runtime per hub.
- **Ollivier–Ricci curvature** — complementary to the cheap Forman variant we already use, but each edge requires solving an optimal-transport sub-problem (`GraphRicciCurvature` package).
- **Hyperbolic embedding distortion** — maps the KG into hyperbolic space and flags entities whose embedding distortion exceeds chance; useful for hierarchical fields but requires a fresh embedding pipeline.

#### Implemented since the previous draft

- ✅ **Personalised PageRank reachability gaps** — `analysis/diffusion_gaps.py::detect_reachability_gaps(diffusion="ppr")` → `output/ppr_reachability_gaps.csv`
- ✅ **Heat-kernel diffusion** — same module, `diffusion="heat"` (uses `scipy.sparse.linalg.expm_multiply` against the symmetric normalised Laplacian) → `output/heat_kernel_reachability_gaps.csv`
- ✅ **Triadic-closure deficit** — `analysis/advanced_gaps.py::detect_triadic_closure_deficit()` → `output/triadic_deficit.csv`
- ✅ **2-edge-cuts (bridges)** — `analysis/advanced_gaps.py::detect_bridge_edges()` (uses `nx.bridges()` on the backbone graph) → `output/bridge_edges.csv`
