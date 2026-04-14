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
