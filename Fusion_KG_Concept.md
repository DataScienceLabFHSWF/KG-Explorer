# Analysing, Extending, and Opening the Fusion Energy Knowledge Graph

**Version 2 — March 2026**
**DataScienceLab FH-SWF / Gaia Lab**

---

## Table of Contents

1. [Context and Motivation](#1-context-and-motivation)
2. [The Fusion KG as TBox / ABox](#2-the-fusion-kg-as-tbox--abox)
3. [Full-Text Enrichment Strategy](#3-full-text-enrichment-strategy)
4. [Mathematical Structures in the Graph](#4-mathematical-structures-in-the-graph)
   - 4.1 Graph-Theoretic Analysis
   - 4.2 Algebraic Topology — Persistent Homology
   - 4.3 Spectral Graph Theory
   - 4.4 Category Theory for Ontology Engineering
   - 4.5 Formal Concept Analysis
   - 4.6 Information-Theoretic Measures
5. [Gap Detection from Graph Structure](#5-gap-detection-from-graph-structure)
6. [Domain Decomposition — Formal Submodule Design](#6-domain-decomposition--formal-submodule-design)
7. [Building a QA / Chatbot Layer (GraphRAG)](#7-building-a-qa--chatbot-layer-graphrag)
8. [Generalisation to Other Domains](#8-generalisation-to-other-domains)
9. [Authors' Pipeline — Availability and Reconstruction](#9-authors-pipeline--availability-and-reconstruction)
10. [Modular Ontology Extension — Linking Fusion to the World](#10-modular-ontology-extension--linking-fusion-to-the-world)
11. [Long-Term Vision: Open Scientific Knowledge Infrastructure](#11-long-term-vision-open-scientific-knowledge-infrastructure)
12. [Concrete Implementation Roadmap](#12-concrete-implementation-roadmap)
13. [Summary](#13-summary)
14. [References](#14-references)

> **Note:** Practical code examples and step-by-step guides have been moved to [docs/implementation_guides.md](docs/implementation_guides.md).

---

## 1  Context and Motivation

Loreti et al. (arXiv:2504.07738) constructed a Knowledge Graph (KG) of nuclear fusion energy from ~8,400 scientific abstracts (sourced via lens.org, spanning 1958–2024), yielding **108,811 nodes** and **718,335 co-occurrence edges** across **28 entity categories**.  Their schema (node labels `Abstract`, `Sentence`, `Entity`, `Person`, `TimeReference`, `KeyWord`; relationships `HAS_SENTENCE`, `CONTAINS`, `CC`, `HAS_FIRST_AUTHOR`, `WAS_PUBLISHED_IN`, `HAS_KEYWORD`) functions as a lightweight **TBox** (terminological box), while the concrete entities and edges constitute the **ABox** (assertional box).

We have access to the authors' NER data (15 JSON files covering the major fusion sub-topics: Tokamak, Stellarator, Inertial Confinement, Deuterium-Tritium, Nuclear Fusion, Fusion Energy).

**Goals:**

1. Load this data into Neo4j and analyse the graph's mathematical structure.
2. Enrich the KG with full-text documents and combine KG-RAG with vector/hybrid RAG.
3. Build a QA / chatbot layer on top (GraphRAG).
4. Treat the 28-category schema as the **seed ontology module** for a larger, modular ontology network linking fusion research to foundational physics, engineering, and beyond — using the KGPlatform ecosystem (KnowledgeGraphBuilder, GraphQAAgent, OntologyExtender).
5. Generalise the pipeline to other scientific/engineering domains (physics, semiconductors, materials science).
6. Integrate graph-structural gap detection into the KGPlatform full-stack.
7. Make the resulting knowledge infrastructure accessible to the wider public.

---

## 2  The Fusion KG as TBox / ABox

### 2.1  TBox — The Categorical Schema

The paper defines implicit terminological commitments:

| TBox Element | Realisation in the Fusion KG |
|---|---|
| **Classes** | The 28 NER category types (e.g., *Nuclear Fusion Device Type*, *Plasma Property*, *Physical Process*) |
| **Object properties** | `HAS_SENTENCE`, `CONTAINS`, `CC`, `HAS_FIRST_AUTHOR`, `WAS_PUBLISHED_IN`, `HAS_KEYWORD` |
| **Data properties** | `weight` on CC edges; `embeddings` (768-dim vectors) on Sentence nodes |
| **Axioms** | Implicit: every `CC` edge connects two `Entity` nodes; every `Sentence` belongs to exactly one `Abstract`; categories are intended to be mutually exclusive |

This is *not* a formal OWL ontology — it is a **pragmatic schema** defined by the NER prompt design.  To use it as a proper TBox we must:

- Formalise the 28 categories as OWL classes with disjointness axioms where appropriate.
- Declare domain/range constraints on the properties.
- Add SHACL shapes for validation (as KnowledgeGraphBuilder already does for its domain).

### 2.2  ABox — The Instantiated Graph

The JSON data we hold represents the ABox:
- Each paper record → an `Abstract` node with metadata (title, URL, year, author, citations).
- Each extracted entity → an `Entity` node typed by one or more categories.
- Sentence-level co-occurrence → `CC` edges with weights.
- Fields of study → a lightweight classification layer.

The distinction matters because it separates **what can be said** (TBox) from **what is said** (ABox), enabling:
- **Validation**: Does the ABox conform to the TBox? (SHACL)
- **Extension**: Can the TBox grow to accommodate new knowledge? (OntologyExtender)
- **Reasoning**: Can we infer new facts from TBox axioms applied to the ABox? (OWL reasoners, SHACL2FOL)

---

## 3  Full-Text Enrichment Strategy

### 3.1  Problem: Abstracts Are Lossy Compression

The current KG is built from abstracts only.  Abstracts are a highly condensed representation of paper content — they typically lose:
- Detailed methodology descriptions
- Specific numerical results and measurements
- Literature review cross-references
- Negative results and failed approaches
- Detailed component descriptions and experimental setups

Moving to full texts would dramatically increase entity coverage and enable richer relationship extraction.

### 3.2  Full-Text Acquisition Pipeline

```
lens.org / Semantic Scholar / OpenAlex / Unpaywall
         ↓
  DOI / URL Resolution
         ↓
  ┌──────────────────────────────────────┐
  │  Open Access Check                   │
  │  (Unpaywall API → OA status + URL)   │
  │  Green OA: preprint from arXiv/repo  │
  │  Gold OA: publisher open access      │
  │  Bronze OA: free-to-read on pub site │
  │  Closed: skip or institutional proxy │
  └──────────────────────────────────────┘
         ↓
  PDF/HTML Download (respect robots.txt, rate limits)
         ↓
  Document Processing (KnowledgeGraphBuilder)
    • PDF → text (pdfminer, PyMuPDF)
    • Section segmentation (GROBID or custom)
    • Chunking with metadata preservation
         ↓
  Embedding + Indexing (Qdrant)
    • chunk_id, paper_id, section, page
    • 1024-dim qwen3-embedding vectors
```

**Key considerations:**
- **Legal**: Only acquire open-access papers.  Of the ~8,400 abstracts, a significant fraction of fusion papers are from publicly funded research and available via arXiv, institutional repositories, or publisher OA (particularly IAEA technical documents, JET reports, ITER publications).
- **Metadata preservation**: Each chunk must retain `paper_id`, `section_type` (introduction, methodology, results, etc.), `page_number`, `sentence_index` so that the vector store links back to the KG.
- **Incremental**: Start with the papers that have the highest citation count or that appear most central in the KG (highest PageRank entities → their source papers first).

### 3.3  Linking KG and Vector Store

The enrichment creates a **dual-index architecture**:

```
┌──────────────────────┐     ┌──────────────────────┐
│   Neo4j (KG)         │     │   Qdrant (Vectors)    │
│                      │     │                       │
│  (Paper)─MENTIONS──→ │     │  chunk_id             │
│     │    (Entity)     │◄───►│  paper_id  ──────────►│
│     │                 │     │  section              │
│  (Entity)─CC──→      │     │  embedding [1024]     │
│     (Entity)          │     │  text                 │
└──────────────────────┘     └───────────────────────┘
         ↕                              ↕
    Cypher queries               Vector similarity
         ↕                              ↕
    ┌──────────────────────────────────────┐
    │          Hybrid RAG Fusion           │
    │  rank_fusion(kg_results, vec_results)│
    │  → LLM synthesis with provenance    │
    └──────────────────────────────────────┘
```

**Linking mechanism:**
1. Every chunk in Qdrant carries a `paper_id` that matches a `Paper` node in Neo4j.
2. Entities extracted from full-text chunks get `MENTIONS` edges back to the same `Paper` node (and new `EXTRACTED_FROM` edges to specific chunks if granularity is needed).
3. At query time, the hybrid strategy:
   - **KG path**: Extract query entities → Cypher traversal → return connected entities, papers, co-occurrence paths.
   - **Vector path**: Embed query → Qdrant top-k → return relevant text chunks with metadata.
   - **Fusion**: Reciprocal Rank Fusion (RRF) or weighted combination → deduplicate by paper_id → LLM synthesis.

### 3.4  What Full-Text Enrichment Enables

| Capability | Abstract-Only KG | Full-Text Enriched KG |
|---|---|---|
| Entity coverage | High-frequency concepts only | Long-tail entities, specific measurements, component details |
| Relationship quality | Co-occurrence only | Explicit relations from methodology sections ("X was heated using Y") |
| Numerical data | Rare (only headline results) | Detailed: temperatures, pressures, confinement times, power levels |
| Cross-references | Not captured | Section-level citation analysis, method attribution |
| QA depth | "What is X?" type answers | "How was X measured in experiment Y?" type answers |
| Provenance | Paper-level | Sentence-level with section context |

---

## 4  Mathematical Structures in the Graph

The fusion KG is a large, weighted, typed graph.  Several mathematical frameworks reveal structure in it.  This section provides the theoretical foundations, concrete definitions, and implementation paths for each.

### 4.1  Graph-Theoretic Analysis

#### 4.1.1  Formal Definitions

Let $G = (V, E, w)$ be the co-occurrence graph where:
- $V$ is the set of entity nodes (|V| ≈ 108,811)
- $E \subseteq V \times V$ is the set of co-occurrence edges (|E| ≈ 718,335)
- $w: E \to \mathbb{R}^+$ assigns co-occurrence weights

Each node $v \in V$ carries a type label $\tau(v) \in \mathcal{C}$ where $\mathcal{C}$ is the set of 28 categories.

#### 4.1.2  Degree Distribution and Power Laws

The **degree** of node $v$ is $\deg(v) = \sum_{(v,u) \in E} 1$ (unweighted) or $\deg_w(v) = \sum_{(v,u) \in E} w(v,u)$ (weighted).

The paper validates Zipf's law on entity frequency: $f(r) = C / r^\alpha$ where $r$ is the rank and $\alpha \approx 1$.  If the degree distribution follows a power law $P(k) \propto k^{-\gamma}$, then $\gamma$ characterises the graph:
- $\gamma \in (2, 3)$: scale-free with hub-dominated structure (expected for a co-occurrence graph of scientific terminology).
- $\gamma > 3$: more homogeneous, hubs less dominant.

**Implementation**: Fit the degree sequence to a power-law distribution using the Clauset-Shalizi-Newman method (`powerlaw` Python package), compare against log-normal and exponential alternatives via likelihood ratio test.

#### 4.1.3  Centrality Measures

| Centrality | Definition | Interpretation for Fusion KG |
|---|---|---|
| **Degree centrality** | $C_D(v) = \deg(v) / (|V| - 1)$ | Raw connectivity — which concepts appear with the most others? |
| **Weighted degree** (strength) | $s(v) = \sum_{u} w(v,u)$ | Total co-occurrence weight — which concepts are most frequently discussed together with others? |
| **Betweenness centrality** | $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$ and $\sigma_{st}(v)$ those passing through $v$ | **Bridge concepts** — entities that connect otherwise separate sub-communities (e.g., *divertor* bridging *plasma physics* and *materials engineering*).  High betweenness = interdisciplinary concept. |
| **Closeness centrality** | $C_C(v) = \frac{|V| - 1}{\sum_u d(v, u)}$ | How quickly can you reach all other concepts from this one? Central concepts have short paths to everything. |
| **Eigenvector centrality** | $C_E(v) = \frac{1}{\lambda} \sum_u A_{vu} C_E(u)$ (eigenvector of adjacency matrix for largest eigenvalue $\lambda$) | Importance by association: a concept is important if it co-occurs with other important concepts.  Similar to PageRank but without damping. |
| **PageRank** | $\text{PR}(v) = \frac{1-d}{|V|} + d \sum_{u \to v} \frac{\text{PR}(u)}{\deg^{\text{out}}(u)}$ with damping $d \approx 0.85$ | Recursive importance — identifies the "canonical" concepts.  On an undirected co-occurrence graph, this converges to a weighted degree-like measure but with propagation effects. |

**Implementation**: Neo4j GDS (Graph Data Science) library provides all of these as built-in algorithms callable via Cypher.  For the co-occurrence graph projected in-memory:

```cypher
-- Project the graph
CALL gds.graph.project('fusion-cooc', 'Entity',
  {CO_OCCURS_WITH: {properties: 'weight'}})

-- PageRank
CALL gds.pageRank.stream('fusion-cooc',
  {relationshipWeightProperty: 'weight'})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS entity, score
ORDER BY score DESC LIMIT 50

-- Betweenness
CALL gds.betweenness.stream('fusion-cooc')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS entity, score
ORDER BY score DESC LIMIT 50
```

#### 4.1.4  Community Detection

**Louvain algorithm** optimises modularity $Q$:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where $m$ is the total edge weight, $k_i$ is the weighted degree of node $i$, and $\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ are in the same community.

The **Leiden algorithm** is an improvement that guarantees well-connected communities (no disconnected sub-communities within a detected community).

**Expected communities in the fusion KG**:
1. Magnetic confinement core (tokamak, stellarator, plasma confinement)
2. Inertial confinement / laser fusion
3. Plasma physics fundamentals (MHD, transport, instabilities)
4. Fusion materials and engineering
5. Diagnostics and measurement
6. Fuel cycle (deuterium, tritium, breeding)
7. Reactor design and power plant concepts

Community detection generates the **empirical domain decomposition** that feeds into the formal ontology submodule design (§6).

#### 4.1.5  k-Core Decomposition

The $k$-core of $G$ is the maximal subgraph where every node has degree $\geq k$.  The **coreness** $c(v)$ is the maximum $k$ for which $v$ belongs to a $k$-core.

- **High-coreness entities** = the dense consensual core of fusion knowledge (fundamental concepts everyone uses).
- **Low-coreness entities** = peripheral, specialised, or emerging topics.
- The **corona** $C_k = k\text{-core} \setminus (k+1)\text{-core}$ forms "shells" of decreasing centrality.

This decomposition identifies the **knowledge frontier**: entities on the outer shells are candidates for ontology extension because they represent areas where the knowledge graph is structurally thin.

#### 4.1.6  Motif Analysis

A **network motif** is a recurring subgraph pattern.  In the co-occurrence graph:
- **Triangles** (3-cliques): concept triads that systematically appear together (e.g., `tokamak–plasma–confinement`).  The clustering coefficient $C(v) = \frac{2 |\text{triangles through } v|}{\deg(v)(\deg(v)-1)}$ quantifies this locally.
- **4-cliques and higher**: increasingly rare but represent deeply interconnected concept clusters.
- **Stars**: one hub connected to many peripheral nodes — indicates a broad concept (e.g., *plasma* connecting to dozens of specific properties).

The **motif spectrum** (relative frequency of each motif type compared to a random graph) characterises the graph's organisational principle.

### 4.2  Algebraic Topology — Persistent Homology

#### 4.2.1  Intuition

Algebraic topology studies shapes through their "holes" at various dimensions.  Applied to a knowledge graph, these holes correspond to structural features of the knowledge landscape.

Imagine lowering a threshold and watching entities gradually "link up" as weaker and weaker co-occurrence edges are included.  At first only the strongest connections exist (highly studied pairs); as the threshold drops, looser associations appear, filling in the graph.  During this process, structures **form** (are born) and **collapse** (die) — and it is precisely the structures that survive across a wide range of thresholds that tell us something real about the domain.

#### 4.2.2  Simplicial Complex Construction

From the co-occurrence graph, we build a **Vietoris-Rips complex** (or more precisely, a **clique complex**):

**Definition.** Given a threshold $\epsilon \geq 0$, define the graph $G_\epsilon = (V, E_\epsilon)$ where $E_\epsilon = \{(u,v) \in E : w(u,v) \geq \epsilon\}$.  The **clique complex** $X_\epsilon$ has:
- A vertex (0-simplex) for each $v \in V$
- An edge (1-simplex) for each $(u,v) \in E_\epsilon$
- A triangle (2-simplex) $[u,v,w]$ for each 3-clique $\{u,v,w\}$ in $G_\epsilon$
- A $k$-simplex $[v_0, \ldots, v_k]$ for each $(k+1)$-clique in $G_\epsilon$

#### 4.2.3  Filtration and Persistence

As $\epsilon$ decreases from $\max(w)$ to $0$, we include progressively weaker co-occurrence edges, obtaining a nested sequence of simplicial complexes:

$$X_{\epsilon_{\max}} \subseteq X_{\epsilon_{\max}-1} \subseteq \cdots \subseteq X_0$$

This is a **filtration**.  At each step, topological features (connected components, loops, voids) are **born** or **die**.

#### 4.2.3a  What "Born" and "Die" Mean — Detailed Explanation

These terms describe the **lifecycle of a topological feature** as the co-occurrence threshold changes.

**Born (birth):**
A feature is *born* at threshold $\epsilon_b$ when it **first appears** in the simplicial complex at that step of the filtration.  Concretely:

- **$H_0$ (connected component) is born** when a previously isolated entity first becomes reachable at co-occurrence weight $\epsilon_b$.  At the start of the filtration (highest threshold, only very strong co-occurrences), every entity is isolated — so one component is "born" per node.  As the threshold drops, components start merging.
- **$H_1$ (loop / cycle) is born** when a new edge at threshold $\epsilon_b$ creates a closed loop that **cannot be contracted** to a point.  For example, adding the edge `stellarator–H-mode` might create the loop `stellarator → plasma → confinement → H-mode → stellarator` — a circular chain of pairwise co-occurrences that has no "shortcut" filling in the middle.
- **$H_2$ (void / cavity) is born** when triangular faces appear around a hollow region.  Imagine four concepts A, B, C, D where all six pairwise edges exist, but the interior is empty — like a hollow tetrahedron.  The void is "born" when the last bounding triangle forms.

**Die (death):**
A feature *dies* at threshold $\epsilon_d$ when a later step in the filtration **destroys** it by filling it in or collapsing it:

- **$H_0$ component dies** when it **merges** with another component.  If `tokamak` and `stellarator` were separate components and an edge between them is added at threshold $\epsilon_d$, the younger component dies (by convention, the one born later).
- **$H_1$ loop dies** when a new edge or triangle **fills in** the interior of the loop, making it contractible.  For example, if a direct `stellarator–confinement` edge appears at $\epsilon_d$, the loop `stellarator → plasma → confinement → stellarator` can now be "shrunk" through the new shortcut — the loop dies.
- **$H_2$ void dies** when a higher-dimensional simplex **fills** the cavity.  If the interior of the hollow tetrahedron above gets filled (all four concepts appear together in the same context), the void collapses.

**Persistence = $\epsilon_b - \epsilon_d$:**
The difference between birth and death thresholds measures how **robust** a feature is.  A feature born at high $\epsilon_b$ (strong co-occurrence) and dying at low $\epsilon_d$ (or never dying) survives across many threshold levels and reflects genuine structure.  Features with small persistence ($\epsilon_b \approx \epsilon_d$) are **noise** — they appear and vanish within a small threshold window.

**Plain-language summary:**

| Event | What Happens | Fusion KG Analogy |
|---|---|---|
| **Birth** of $H_0$ | An entity appears | A research topic enters the literature |
| **Death** of $H_0$ | Two clusters merge into one | Two sub-fields are recognised as connected |
| **Birth** of $H_1$ | A loop forms with no shortcut | A circular chain of related concepts emerges (e.g., `tokamak → plasma → instability → ELM → divertor → tokamak`) |
| **Death** of $H_1$ | A triangle fills the loop | A direct relationship is established, closing the gap |
| **Birth** of $H_2$ | A hollow cavity forms | Several concepts co-occur pairwise but never jointly — a knowledge gap |
| **Death** of $H_2$ | The cavity is filled | Research explicitly connecting all those concepts together appears |

**Example with actual numbers:**
In our analysis of the top-800 entity subgraph (by weighted degree), the most persistent $H_1$ feature has birth $= 0.333$ and death $= 1.0$, giving persistence $= 0.667$.  Since we use distance $d = 1/w$ (where $w$ is co-occurrence weight), this translates to:
- The loop **appeared** (was born) when we included co-occurrences with weight $\geq 1/0.333 = 3$ (strong co-occurrence).
- The loop **was destroyed** (died) when we included co-occurrences with weight $\geq 1/1.0 = 1$ (weak co-occurrence).
- It **survived** across a significant range of thresholds, meaning this circular relationship is a genuine structural feature, not noise.

| Homology Group | Topological Feature | Interpretation for KG |
|---|---|---|
| $H_0$ | Connected components | **Research sub-fields** at a given co-occurrence strength.  A component that persists across many thresholds is a fundamental sub-domain. |
| $H_1$ | Loops / 1-cycles | **Redundant knowledge circuits** — e.g., `tokamak → plasma → confinement → H-mode → ELMs → tokamak`.  Persistent loops = well-established, multiply-confirmed conceptual chains. |
| $H_2$ | Voids / cavities | **Knowledge gaps** — regions where many pairwise co-occurrences exist but certain higher-order relationships are missing.  A persistent 2-void between concepts A, B, C means they co-occur pairwise but never all three together. |

#### 4.2.4  Persistence Diagrams and Barcodes

Each topological feature is represented as a point $(b_i, d_i)$ in a **persistence diagram**, where $b_i$ is the birth threshold and $d_i$ is the death threshold.  Features far from the diagonal $d = b$ are **persistent** (structurally significant); those near the diagonal are noise.

**Barcodes** present the same information as horizontal bars: long bars = robust structure, short bars = noise.

#### 4.2.5  Concrete Application: Knowledge Gap Detection

A void in $H_2$ detected at threshold $\epsilon$ between entities $\{A, B, C, D\}$ means:
- $A$-$B$, $A$-$C$, $A$-$D$, $B$-$C$, $B$-$D$, $C$-$D$ all co-occur at weight $\geq \epsilon$
- But the full 4-clique $\{A, B, C, D\}$ never appears — they never all occur in the same sentence/context.

This signals a **potential research opportunity**: the four concepts are related pairwise but have never been studied together.  This could be:
- A genuine gap (no one has investigated the combination).
- An artefact of the abstract-only data (the combination exists in full texts).

Either way, it generates hypotheses for domain experts to evaluate.

#### 4.2.6  Implementation

```python
import numpy as np
from ripser import ripser
from persim import plot_diagrams

# Build distance matrix from co-occurrence weights
# Convert weights to distances: d(u,v) = 1/w(u,v) for connected pairs
# d(u,v) = infinity for disconnected pairs
distance_matrix = np.full((n, n), np.inf)
for (i, j), weight in co_occurrence_weights.items():
    distance_matrix[i][j] = 1.0 / weight
    distance_matrix[j][i] = 1.0 / weight
np.fill_diagonal(distance_matrix, 0)

# Compute persistent homology up to dimension 2
result = ripser(distance_matrix, maxdim=2,
                distance_matrix=True)

# Plot persistence diagrams
plot_diagrams(result['dgms'], show=True)

# Extract persistent H2 features (knowledge gaps)
h2_features = result['dgms'][2]
persistent_voids = h2_features[
    h2_features[:, 1] - h2_features[:, 0] > threshold
]
```

**Scalability note**: For 108K nodes, the full distance matrix is infeasible (O(n²) memory).  Solutions:
- Work on subgraphs per community (post Louvain/Leiden).
- Use sparse Rips complex implementations (`gudhi.SimplexTree`).
- Apply dimensionality reduction first (UMAP on node embeddings → TDA on reduced space).

### 4.3  Spectral Graph Theory

#### 4.3.1  The Graph Laplacian

**Definition.** For a weighted graph $G = (V, E, w)$, define:
- **Adjacency matrix** $A$: $A_{ij} = w(i,j)$ if $(i,j) \in E$, else $0$.
- **Degree matrix** $D$: $D_{ii} = \sum_j A_{ij}$ (diagonal).
- **Unnormalised Laplacian**: $L = D - A$
- **Normalised Laplacian**: $\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$
- **Random-walk Laplacian**: $L_{\text{rw}} = D^{-1} L = I - D^{-1} A$

$L$ is positive semi-definite with eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{|V|}$.

#### 4.3.2  The Fiedler Value (Algebraic Connectivity)

$\lambda_2$ (the smallest non-zero eigenvalue) is the **Fiedler value**.  It measures how well-connected the graph is:

- $\lambda_2 = 0$: graph is disconnected.
- Small $\lambda_2$: near a bottleneck — there's a weak link between two large sub-communities.
- Large $\lambda_2$: the graph is robust and well-connected.

The corresponding **Fiedler vector** $\mathbf{v}_2$ gives the optimal bipartition of the graph: nodes with $v_{2,i} > 0$ vs. $v_{2,i} < 0$ form the two sides of the weakest cut.  This is the spectral version of community detection.

**For the fusion KG**: $\lambda_2$ reveals how tightly connected the overall fusion research landscape is.  If small, there are distinct sub-fields with weak links between them — important for understanding knowledge transfer barriers.

#### 4.3.3  Spectral Clustering

Given the first $k$ eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ of $\mathcal{L}$, embed each node as a point in $\mathbb{R}^k$:

$$\phi(v_i) = \left( \frac{v_{1,i}}{\sqrt{d_i}}, \ldots, \frac{v_{k,i}}{\sqrt{d_i}} \right)$$

Then apply k-means clustering in this spectral space.  This is the **Ng-Jordan-Weiss spectral clustering** algorithm.

Compared to Louvain/Leiden:
- Spectral clustering has theoretical guarantees (minimises the normalised cut).
- Louvain/Leiden is faster and more practical for large graphs.
- Both should produce similar communities — discrepancies are informative.

#### 4.3.4  Graph Fourier Transform (GFT)

Any function $f: V \to \mathbb{R}$ (e.g., citation count per entity, year of first appearance) can be decomposed in the eigenbasis of $L$:

$$\hat{f}(\lambda_k) = \langle \mathbf{v}_k, f \rangle = \sum_i v_{k,i} \cdot f(v_i)$$

- **Low-frequency components** ($\hat{f}(\lambda_k)$ for small $\lambda_k$): smooth, global trends that vary slowly across the graph.  For citations: concepts that are uniformly important.
- **High-frequency components** ($\hat{f}(\lambda_k)$ for large $\lambda_k$): local anomalies — concepts whose importance differs sharply from their neighbours.

**Application**: Apply GFT to the "year of first mention" signal to identify concepts that appeared unusually early or late relative to their graph neighbourhood (temporal anomaly detection).

#### 4.3.5  Heat Kernel Diffusion

The **heat kernel** $H_t = e^{-tL}$ models diffusion on the graph at time scale $t$:

$$(H_t)_{ij} = \sum_k e^{-t\lambda_k} v_{k,i} v_{k,j}$$

**Interpretation**: $(H_t)_{ij}$ is the amount of "heat" (or information, influence) that flows from node $i$ to node $j$ after time $t$.

- Small $t$: local neighbourhood effects — which concepts directly influence each other?
- Large $t$: global equilibrium — how does a breakthrough in one area eventually affect the whole field?

**Application**:  Model how a new result in *divertor design* propagates through the knowledge graph to affect *reactor economics*.  The heat kernel trace $\text{tr}(H_t) = \sum_k e^{-t\lambda_k}$ is related to the graph's **zeta function** and gives a multiscale summary of graph structure.

#### 4.3.6  Plain-Language Summary of Spectral Methods

**What is the Laplacian eigenvector decomposition doing?**  Think of the entities in the graph as masses connected by springs (the co-occurrence edges).  The eigenvalues are the **natural vibration frequencies** of this spring system:
- The **lowest frequency** ($\lambda_1 = 0$) is the constant mode: every entity moves together.
- The **second-lowest** ($\lambda_2$, the Fiedler value) is the mode that separates the graph into two halves with the least effort.  If $\lambda_2$ is very small, only a thin "bridge" holds the two halves together.
- **Higher frequencies** capture finer substructure: communities within communities.

**Why does this matter practically?**
- Spectral clustering produces communities with a formal optimality guarantee (minimised normalised-cut), complementing the heuristic Louvain/Leiden approach.
- The GFT decomposes any numerical attribute (e.g., publication year, citation count) into global trends vs. local anomalies — useful for spotting outlier entities.
- The heat kernel answers "what-if" questions: *if new research energy entered concept X, where would it diffuse to over time?*

**Computational challenge:** Computing eigenvalues of a 50,000-node sparse Laplacian requires iterative solvers (`scipy.sparse.linalg.eigsh`).  The shift-invert strategy (setting $\sigma = 0$) converts the problem into solving linear systems with $L - 0 \cdot I = L$, which is much faster for the *smallest* eigenvalues.  Even so, this computation may take several minutes on large graphs; we provide timed progress feedback so runs are not mistaken for hangs.

#### 4.3.7  Implementation

```python
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Build sparse adjacency matrix from Neo4j export
# A[i,j] = co-occurrence weight

# Compute Laplacian
L = csgraph.laplacian(A, normed=True)

# First k eigenvalues/vectors
k = 20
eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

# Fiedler value
fiedler_value = eigenvalues[1]
fiedler_vector = eigenvectors[:, 1]

# Spectral clustering
from sklearn.cluster import KMeans
embedding = eigenvectors[:, 1:k]     # skip constant eigenvector
embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
kmeans = KMeans(n_clusters=k-1)
labels = kmeans.fit_predict(embedding)

# Heat kernel at time t
t = 1.0
heat_kernel_diag = np.sum(np.exp(-t * eigenvalues)
                          * eigenvectors**2, axis=1)
```

### 4.4  Category Theory for Ontology Engineering

#### 4.4.1  Why Category Theory?

Category theory is the mathematics of **structure-preserving mappings**.  In ontology engineering, we constantly deal with:
- Mapping one ontology into another (alignment).
- Merging two ontologies that share some concepts (integration).
- Selecting a view of a large ontology for a specific task (restriction).
- Composing multiple mappings (transitivity of alignment).

Category theory provides the **right level of abstraction** for rigorously discussing these operations.

#### 4.4.2  Ontologies as Categories

**Definition.** An ontology $\mathcal{O}$ can be modelled as a category where:
- **Objects** = classes (types) in the ontology.  In the fusion KG: the 28 NER categories plus any superclasses we add.
- **Morphisms** = relationships (properties) between classes.  In the fusion KG: `CONTAINS`, `CC`, `HAS_FIELD`, etc.
- **Composition**: if there's a morphism $f: A \to B$ and $g: B \to C$, then there's a composed morphism $g \circ f: A \to C$.  In the KG: if `Abstract` → `HAS_SENTENCE` → `Sentence` and `Sentence` → `CONTAINS` → `Entity`, then there's an implicit composed relation `Abstract` → `MENTIONS` → `Entity`.
- **Identity**: every class has an identity morphism (trivial self-relation).

**Example**: The fusion TBox as a category:

```
Abstract ──HAS_SENTENCE──► Sentence ──CONTAINS──► Entity
    │                                                │
    ├──HAS_FIRST_AUTHOR──► Person                    │
    │                                                │
    ├──WAS_PUBLISHED_IN──► TimeReference             │
    │                                                │
    └──HAS_KEYWORD──► KeyWord                 CC (weighted)
                                                     │
                                              Entity ◄─┘
```

Each path through this diagram is a composed morphism.

#### 4.4.3  Functors — Ontology Mappings

**Definition.** A **functor** $F: \mathcal{O}_1 \to \mathcal{O}_2$ between ontology-categories assigns:
- To each class $A$ in $\mathcal{O}_1$, a class $F(A)$ in $\mathcal{O}_2$.
- To each relationship $f: A \to B$ in $\mathcal{O}_1$, a relationship $F(f): F(A) \to F(B)$ in $\mathcal{O}_2$.
- Such that composition and identities are preserved: $F(g \circ f) = F(g) \circ F(f)$ and $F(\text{id}_A) = \text{id}_{F(A)}$.

**Concrete example**: Mapping the fusion ontology to BFO:

$$F_{\text{BFO}}: \mathcal{O}_{\text{fusion}} \to \mathcal{O}_{\text{BFO}}$$

| $A$ in $\mathcal{O}_{\text{fusion}}$ | $F(A)$ in $\mathcal{O}_{\text{BFO}}$ |
|---|---|
| Nuclear Fusion Device Type | Material Entity |
| Physical Process | Process |
| Plasma Property | Quality |
| Nuclear Fusion Experimental Facility | Site |
| Person | not mapped (BFO doesn't model persons) |

When $F$ doesn't map everything, it's a **partial functor** — signalling that the target ontology has a different scope.  This is precisely the information we need for modular ontology design.

**Types of functors and their ontological meaning:**

| Functor Type | Mathematical Definition | Ontological Meaning |
|---|---|---|
| **Faithful** | $F$ is injective on each morphism set $\text{Hom}(A,B)$ | No relationship conflation — distinct source relations map to distinct target relations. |
| **Full** | $F$ is surjective on each $\text{Hom}(A,B)$ | Every relationship in the target between mapped classes has a pre-image — target doesn't add new relationships between things the source already knows about. |
| **Fully faithful** | Both full and faithful | The mapping is a perfect "embedding" of one ontology's relational structure into another. |
| **Essentially surjective** | Every target object is isomorphic to $F(A)$ for some $A$ | Every concept in the target has a counterpart in the source (up to equivalence). |
| **Equivalence** | Fully faithful + essentially surjective | The two ontologies are structurally identical (from a category-theoretic perspective). |

#### 4.4.4  Natural Transformations — Comparing Mappings

**Definition.** Given two functors $F, G: \mathcal{O}_1 \to \mathcal{O}_2$, a **natural transformation** $\eta: F \Rightarrow G$ assigns to each class $A$ in $\mathcal{O}_1$ a morphism $\eta_A: F(A) \to G(A)$ in $\mathcal{O}_2$ such that for every relationship $f: A \to B$:

$$\eta_B \circ F(f) = G(f) \circ \eta_A$$

**Ontological meaning**: If two experts independently align the fusion ontology to EMMO ($F$ and $G$), a natural transformation captures the *systematic differences* between their alignments.  Finding a natural transformation means the two alignments are "coherently different" — they disagree in a structured, reconcilable way.

If no natural transformation exists, the alignments are *incommensurable* — they make fundamentally different structural choices.

#### 4.4.5  Pushouts — Ontology Merging

Given two ontology modules $\mathcal{A}$ and $\mathcal{B}$ that share a common core $\mathcal{C}$ (with inclusion functors $i: \mathcal{C} \to \mathcal{A}$ and $j: \mathcal{C} \to \mathcal{B}$), the **pushout** $\mathcal{A} \sqcup_{\mathcal{C}} \mathcal{B}$ is the minimal merged ontology that contains both $\mathcal{A}$ and $\mathcal{B}$ with shared concepts identified:

```
        C
       / \
      i   j
     /     \
    A       B
     \     /
      \   /
       ↘ ↙
    A ⊔_C B   (pushout = merged ontology)
```

**Concrete example**: Merging a Plasma Physics module ($\mathcal{A}$) with a Fusion Materials module ($\mathcal{B}$) that share a common core of basic physics concepts ($\mathcal{C}$ — e.g., temperature, pressure, particle).  The pushout gives the minimal integrated ontology where `temperature` is only defined once but inherits all relationships from both modules.

**Why this matters**: Without pushouts, merging ontologies naively creates duplicates and inconsistencies.  The pushout construction guarantees that shared concepts are identified and that no information is lost.

#### 4.4.6  Ologs — Ontology Logs

Spivak (2012) introduced **ologs** as a category-theoretic framework specifically for knowledge representation.  An olog is a category where:
- Objects are labelled with natural-language noun phrases ("a tokamak", "a plasma temperature").
- Morphisms are labelled with functional relationships ("has", "is measured in", "confines").
- The labels must satisfy a **readability condition**: following a path of morphisms should read as a grammatically correct English sentence.

**Example olog for a fragment of the fusion KG:**

```
[a tokamak] ──produces──► [a plasma]
     │                        │
     │                        ├──has──► [a temperature]
     │                        │              │
     │                        │              └──is measured in──► [keV]
     │                        │
     │                        └──exhibits──► [an instability]
     │
     └──has component──► [a divertor]
                              │
                              └──is made of──► [a material]
```

Ologs provide a **bridge** between formal category theory and domain-expert-friendly knowledge modelling.  They can be automatically checked for consistency (no path should lead to contradictory conclusions).

### 4.5  Formal Concept Analysis (FCA)

#### 4.5.1  Formal Context

**Definition.** A **formal context** is a triple $\mathbb{K} = (G, M, I)$ where:
- $G$ is a set of **objects** (in our case: entities in the KG).
- $M$ is a set of **attributes** (in our case: the 28 NER categories).
- $I \subseteq G \times M$ is an **incidence relation** ($g I m$ means "entity $g$ belongs to category $m$").

**Example** (small subset):

|  | Device Type | Plasma Property | Chemical Element | Physical Process | Facility |
|---|---|---|---|---|---|
| Tokamak | × |  |  |  |  |
| H-mode |  | × |  |  |  |
| Deuterium |  |  | × |  |  |
| Confinement |  |  |  | × |  |
| JET |  |  |  |  | × |
| Plasma |  | × |  |  |  |

#### 4.5.2  Formal Concepts

For a set of objects $A \subseteq G$, define:
$$A' = \{m \in M : \forall g \in A, \; g I m\}$$
(the attributes shared by all objects in $A$).

For a set of attributes $B \subseteq M$, define:
$$B' = \{g \in G : \forall m \in B, \; g I m\}$$
(the objects having all attributes in $B$).

A **formal concept** is a pair $(A, B)$ where $A' = B$ and $B' = A$.  $A$ is the **extent** (objects), $B$ is the **intent** (shared attributes).

#### 4.5.3  Concept Lattice

The set of all formal concepts, ordered by extent inclusion $(A_1, B_1) \leq (A_2, B_2) \iff A_1 \subseteq A_2$, forms a **complete lattice** — the **concept lattice** $\underline{\mathfrak{B}}(\mathbb{K})$.

**What the lattice reveals for the fusion KG:**
- **Subcategory structure**: If all entities typed as *Plasma Event* are also typed as *Physical Process*, the lattice shows *Plasma Event* below *Physical Process* — suggesting a formal subclass relationship that the original 28-type flat taxonomy doesn't capture.
- **Category interdependencies**: Categories that always co-occur on the same entities are grouped together.
- **Potential splits**: If a category contains two clearly separate clusters of entities in the lattice, it should be split into sub-categories.

#### 4.5.4  Attribute Implications

From the formal context, we can derive **implications** $B_1 \to B_2$ meaning "every entity with all attributes in $B_1$ also has all attributes in $B_2$".  The **Duquenne-Guigues basis** is the minimal set of implications from which all others follow.

**Example implications we might discover:**
- `{Plasma Event} → {Physical Process}` (every plasma event is a physical process)
- `{Control Systems} → {Experimental Apparatus}` (every control system is experimental apparatus)
- `{Nuclear Fusion Experimental Facility} → {Facility or Institution}` (hierarchy)

These implications can be directly translated into OWL subclass axioms for the formalised ontology.

#### 4.5.5  Plain-Language Summary of FCA

**What is FCA really doing?**  It builds a rigorous "is-a" hierarchy from raw data, without anyone defining upfront that "Plasma Event is a subclass of Physical Process".  Instead, FCA observes that every entity labelled *Plasma Event* is also labelled *Physical Process*, and automatically derives the subclass relationship.

Think of FCA as reading the data and producing a **rule book**:
- "If an entity is a Control System, then it is also always classified as Experimental Apparatus."
- "If an entity is both a Chemical Element and a Particle, then it is always also a Physical Process."

These rules (called the Duquenne-Guigues basis) form the **minimal, complete axiom set** — every other rule that holds in the data can be logically derived from this basis.

**Why does this matter?**
- The original NER taxonomy from Loreti et al. is **flat** (28 independent categories).  FCA reveals the hidden hierarchical structure.
- Each discovered implication becomes a candidate OWL `SubClassOf` axiom.
- Contradictions or unexpected implications reveal NER labelling errors or conceptual ambiguities.

#### 4.5.6  Implementation

```python
from concepts import Context

# Build formal context from KG
objects = list(entity_categories.keys())  # entity names
attributes = list(all_categories)         # 28 category names
booleans = [[cat in entity_categories[ent]
             for cat in attributes]
            for ent in objects]

ctx = Context(objects, attributes, booleans)
lattice = ctx.lattice

# Get implications
for impl in lattice.implications():
    print(f"{impl.premise} → {impl.conclusion}")

# Visualise lattice (for subsets)
lattice.graphviz()
```

### 4.6  Information-Theoretic Measures

#### 4.6.1  Graph Entropy

**Von Neumann entropy** of the graph:

$$S(G) = -\text{tr}(\tilde{L} \log \tilde{L}) = -\sum_i \tilde{\lambda}_i \log \tilde{\lambda}_i$$

where $\tilde{L} = L / \text{tr}(L)$ is the normalised Laplacian scaled to be a density matrix, and $\tilde{\lambda}_i$ are its eigenvalues.

- Low entropy: regular, homogeneous structure.
- High entropy: complex, heterogeneous structure.

#### 4.6.2  Mutual Information Between Categories

For two categories $c_1, c_2 \in \mathcal{C}$, define the mutual information across co-occurrence edges:

$$I(c_1; c_2) = \sum_{e=(u,v) \in E} p(e) \log \frac{p(e)}{p_1(u) \cdot p_2(v)}$$

where $p(e)$ is the normalised weight of edge $e$ between an entity of type $c_1$ and one of type $c_2$.

**Interpretation**: Which category pairs share the most information through co-occurrence?  High MI between *Chemical Element* and *Physical Process* means knowing which element is discussed tells you a lot about which process is discussed (and vice versa).

#### 4.6.3  Transfer Entropy for Temporal Analysis

Partition papers by year and build a time series of category popularities $X_t^{c}$ (number of entities of category $c$ in papers from year $t$).

**Transfer entropy** from category $c_1$ to $c_2$:

$$T_{c_1 \to c_2} = \sum p(x_{t+1}^{c_2}, x_t^{c_2}, x_t^{c_1}) \log \frac{p(x_{t+1}^{c_2} | x_t^{c_2}, x_t^{c_1})}{p(x_{t+1}^{c_2} | x_t^{c_2})}$$

**Interpretation**: Does knowing the past of *Plasma Physics* research help predict the future of *Fusion Materials* research beyond what *Fusion Materials*' own past already tells us?  If $T_{\text{plasma} \to \text{materials}} > 0$, then plasma physics developments *drive* materials research.

#### 4.6.4  Plain-Language Summary of Information-Theoretic Measures

**Von Neumann Entropy — "How disordered is the graph?"**
Think of a perfectly regular lattice (like a checkerboard): it has low entropy because its structure is simple and repetitive.  A random network has high entropy because there is no pattern.  The fusion KG's Von Neumann entropy $S(G)$ tells us where it falls on this spectrum.  A high value suggests the knowledge landscape is genuinely complex (many different sub-communities, heterogeneous connectivity), while a low value would suggest a more uniform, orderly structure.

**Mutual Information — "Which categories are informationally coupled?"**
If knowing that a paper mentions "Chemical Element" entities automatically tells you it also mentions "Physical Process" entities, those two categories have high mutual information.  If two categories appear totally independently of each other, MI ≈ 0.  The MI heatmap reveals which NER categories are **informationally redundant** (high MI → possible merge candidates) and which are **informationally independent** (low MI → truly orthogonal dimensions of the taxonomy).

**Category Entropy — "How diverse are a category's partners?"**
A category with high Shannon entropy has co-occurrence partners spread evenly across many other categories.  A category with low entropy tends to co-occur with only one or two specific other categories.  For example, *Research Field* entities might co-occur evenly with everything (high entropy), while *Regulatory Standard* entities might almost exclusively co-occur with *Safety Feature* entities (low entropy).  This reveals how **specialised** vs. **broadly connected** each category is.

**Transfer Entropy — "Does category X drive category Y over time?"**
Transfer entropy measures **directed causal influence** between time series.  If a surge in *Plasma Physics* papers in year $t$ is predictive of a surge in *Fusion Materials* papers in year $t+1$ (beyond what Materials papers' own history predicts), the transfer entropy from Plasma Physics to Fusion Materials is positive.  This reveals which sub-fields are **leaders** (drivers of future research) and which are **followers**.

---

## 5  Gap Detection from Graph Structure

### 5.1  Types of Gaps

| Gap Type | How to Detect | What It Means |
|---|---|---|
| **Topological voids** ($H_2$ in persistent homology) | Persistent cavities in the clique complex | Concepts that co-occur pairwise but are never studied together — potential research opportunities |
| **Structural holes** (Burt, 1992) | Nodes with high betweenness but low clustering coefficient | Concepts that bridge disconnected communities but are themselves poorly integrated — fragile knowledge links |
| **Missing edges** (link prediction) | Low-rank matrix completion or GNN-based link prediction on the adjacency matrix | Entity pairs that *should* co-occur based on their neighbourhood structure but don't — predicted but unobserved relationships |
| **Ontological gaps** | FCA implication analysis + SHACL validation | Categories that should exist but don't; entities that violate expected type constraints — schema incompleteness |
| **Temporal gaps** | Compare community structure across time slices | Topics that were active in the past but have no recent publications — abandoned or completed research areas |
| **Coverage gaps** | Compare KG categories against external ontology (e.g., EMMO) | Entire concept areas present in the reference ontology but absent from the KG — blind spots |

### 5.2  Integration into KGPlatform

The gap detection pipeline fits naturally into the existing HITL feedback loop:

```
GraphQAAgent
  │  detects low-confidence answers
  │  or unanswerable queries
  ▼
Gap Analyser (NEW MODULE)
  ├── Topological: ripser/gudhi on co-occurrence subgraph
  ├── Structural: betweenness + clustering coefficient
  ├── Link prediction: GNN or matrix factorisation
  ├── Ontological: FCA implications vs. current OWL
  └── Coverage: diff against reference ontologies
  │
  ▼  produces gap report
KnowledgeGraphBuilder
  │  identifies missing documents/entities
  ▼
OntologyExtender
  │  proposes schema changes via multi-agent debate
  ▼
Human Reviewer (Streamlit UI)
  │  accepts/rejects/modifies
  ▼
Updated Ontology + Re-extraction
```

**Implementation plan for the Gap Analyser module:**

```python
# gap_analyser/topological_gaps.py
class TopologicalGapDetector:
    """Detect H1 and H2 features in the co-occurrence complex."""
    
    def __init__(self, neo4j_driver, community_label=None):
        self.driver = neo4j_driver
        self.community = community_label
    
    def extract_subgraph(self):
        """Pull co-occurrence edges from Neo4j,
        optionally filtered by community."""
        # Cypher query → sparse adjacency matrix
        ...
    
    def compute_persistence(self, max_dim=2):
        """Compute persistent homology via ripser."""
        # Returns persistence diagrams for H0, H1, H2
        ...
    
    def identify_voids(self, persistence_threshold=0.1):
        """Extract persistent H2 features
        and map back to entity names."""
        # Returns list of (entities, birth, death, persistence)
        ...
    
    def generate_gap_report(self):
        """Produce structured report for OntologyExtender."""
        ...
```

```python
# gap_analyser/structural_gaps.py
class StructuralGapDetector:
    """Detect structural holes and bridge concepts."""
    
    def compute_burt_constraint(self):
        """Burt's constraint measure for each node.
        Low constraint = structural hole."""
        ...
    
    def find_bridge_concepts(self, top_k=20):
        """High betweenness + low clustering."""
        ...
```

```python
# gap_analyser/link_predictor.py
class LinkPredictor:
    """Predict missing co-occurrence edges."""
    
    def common_neighbours_score(self, u, v):
        return len(set(neighbours[u]) & set(neighbours[v]))
    
    def adamic_adar_score(self, u, v):
        return sum(1 / np.log(deg[w])
                   for w in set(neighbours[u]) & set(neighbours[v]))
    
    def predict_missing_edges(self, top_k=100):
        """Score all non-edges and return top-k candidates."""
        ...
```

---

## 6  Domain Decomposition — Formal Submodule Design

### 6.1  The Problem

Given a large, flat ontology (like the 28 categories), how do we formally decompose it into coherent sub-modules?  This matters because:
- Monolithic ontologies are hard to maintain, version, and validate.
- Different stakeholders need different subsets.
- Modular ontologies enable independent evolution and selective import.

### 6.2  Formal Approaches to Domain Splitting

#### 6.2.1  Graph-Based Decomposition (Bottom-Up)

Use the **knowledge graph structure** itself to discover natural module boundaries:

1. **Community detection** (Louvain/Leiden) on the entity co-occurrence graph → identifies clusters of strongly co-occurring concepts.
2. **Spectral clustering** on the graph Laplacian → gives optimal cuts with theoretical guarantees.
3. **Correlation clustering** on the entity-category co-occurrence matrix → groups categories that tend to type the same entities or appear in the same papers.

**Procedure:**
```
Step 1: Run Leiden community detection on co-occurrence graph
        → k communities C_1, ..., C_k

Step 2: For each community C_i, compute the dominant categories:
        dominant_cats(C_i) = {c ∈ C : P(τ(v) = c | v ∈ C_i) > threshold}

Step 3: Define module M_i = OWL ontology restricted to dominant_cats(C_i)
        with all inter-category relationships observed within C_i

Step 4: Define inter-module links from edges spanning communities:
        If entities in C_i and C_j frequently co-occur,
        create alignment axioms between M_i and M_j

Step 5: Validate each module independently (SHACL)
        and the composition (SHACL on merged graph)
```

This is a **data-driven decomposition**: the modules reflect the actual structure of scientific discourse, not an a priori design.

#### 6.2.2  FCA-Based Decomposition (Top-Down)

Use the concept lattice (§4.5) to identify natural sub-lattices:

1. Compute the concept lattice $\underline{\mathfrak{B}}(\mathbb{K})$ for the entity–category context.
2. Look for **lattice decompositions**: if the lattice factors into a product $L_1 \times L_2$, then the two factors correspond to independent sub-ontologies.
3. More commonly, look for **sublattices** that are approximately independent (share few concepts with the rest).
4. Each sublattice defines a module.

#### 6.2.3  Modular Ontology Modeling (MOM) Methodology

Shimizu et al. (2022) propose a systematic methodology:

1. **Scope definition**: For each candidate module, define its **competency questions** (CQs) — the questions the module should be able to answer.
2. **Pattern selection**: Choose **Ontology Design Patterns** (ODPs) from existing pattern libraries (e.g., the ODP repository at ontologydesignpatterns.org).
3. **Module axiomatisation**: Write OWL axioms for each module, referencing imports from other modules.
4. **Interface specification**: For each module, explicitly define:
   - **Required imports**: which external classes/properties are used.
   - **Exported classes**: which classes are available for other modules to import.
   - **Alignment axioms**: how exported classes relate to the upper ontology and to other module exports.

#### 6.2.4  Coupling and Cohesion Metrics

Borrowed from software engineering, apply module quality metrics:

- **Cohesion** of module $M$: proportion of intra-module edges vs. total edges from $M$'s entities.
  $$\text{cohesion}(M) = \frac{|\{(u,v) \in E : u \in M, v \in M\}|}{|\{(u,v) \in E : u \in M \text{ or } v \in M\}|}$$
  High cohesion = module covers a tightly related concept cluster.

- **Coupling** between modules $M_i$ and $M_j$: number of cross-module edges.
  $$\text{coupling}(M_i, M_j) = |\{(u,v) \in E : u \in M_i, v \in M_j\}|$$
  Low coupling = modules are relatively independent.

- **Modularity** $Q$ (Newman): optimised by Louvain/Leiden, measures the quality of the overall decomposition.

**Good decomposition**: high cohesion within each module + low coupling between modules + high overall modularity $Q$.

#### 6.2.5  Proposed Fusion Ontology Module Structure

Based on the 28 categories and expected community detection results:

| Module | Categories Covered | Expected CQs |
|---|---|---|
| **Fusion Devices** | Device Type, System Component, System Configuration, Experimental Apparatus | "What components does a tokamak have?" "What is the difference between a stellarator and a tokamak?" |
| **Plasma Physics** | Plasma Property, Plasma Event, Plasma Region, Plasma Dynamics, Field Configuration | "What instabilities occur in H-mode?" "What is the plasma beta?" |
| **Nuclear Reactions** | Physical Process, Particle, Chemical Element/Compound | "What products does D-T fusion produce?" "What is the cross-section for D-T at 10 keV?" |
| **Diagnostics & Control** | Detection/Monitoring Systems, Control Systems, Software/Simulation | "How is plasma temperature measured?" "What feedback systems control ELMs?" |
| **Theory & Computation** | Theory/Calculation, Concept, Research Field | "What models describe tokamak transport?" "What codes simulate MHD?" |
| **Facilities & People** | Facility/Institution, Person, Country/Location, Time Reference | "Which facilities run tokamak experiments?" "Who are the leading researchers in stellarator design?" |
| **Safety & Standards** | Safety Feature/Regulatory Standard, Database, Scientific Publication | "What safety standards apply to tritium handling?" |
| **Fusion Materials** | (new sub-module, entities from Chemical Element + System Component overlap) | "What materials withstand plasma-facing conditions?" |

---

## 7  Building a QA / Chatbot Layer (GraphRAG)

### 7.1  Architecture with Full-Text Enrichment

```
                  User Query
                      │
              ┌───────▼───────┐
              │ Query Analyser │
              │ (entity + intent│
              │  extraction)   │
              └───┬───────┬───┘
                  │       │
        ┌─────────▼──┐  ┌▼──────────┐
        │  KG Path   │  │ Vector    │
        │            │  │ Path      │
        │ Cypher     │  │ Qdrant    │
        │ generation │  │ top-k     │
        │ (ontology- │  │ on full-  │
        │  informed) │  │ text      │
        │            │  │ chunks    │
        └─────┬──────┘  └──┬───────┘
              │             │
              ▼             ▼
        ┌─────────────────────┐
        │  Reciprocal Rank    │
        │  Fusion (RRF)       │
        │  + Deduplication     │
        │  by paper_id        │
        └────────┬────────────┘
                 │
        ┌────────▼────────────┐
        │  Context Assembly   │
        │  • KG subgraph      │
        │  • Text chunks      │
        │  • Co-occurrence     │
        │    triplets          │
        │  • Metadata          │
        └────────┬────────────┘
                 │
        ┌────────▼────────────┐
        │  LLM Synthesis      │
        │  (grounded answer   │
        │   + source refs)    │
        └─────────────────────┘
```

### 7.2  Analytical Queries Enabled by Mathematical Analysis

| Query Type | Method | Example |
|---|---|---|
| Bridge concept discovery | Betweenness centrality | "What concepts connect tokamak and stellarator research?" |
| Knowledge gap identification | Persistent homology H2 | "Where are promising unexplored combinations of concepts?" |
| Trend analysis | GFT temporal signal decomposition | "Which research areas are growing fastest?" |
| Community comparison | Louvain + inter-community edge analysis | "How do magnetic and inertial confinement communities overlap?" |
| Concept importance ranking | PageRank / eigenvector centrality | "What are the most influential concepts in fusion research?" |
| Knowledge diffusion tracing | Heat kernel | "How did the concept of 'high-β' propagate through the research landscape?" |

---

## 8  Generalisation to Other Domains

### 8.1  The Generic Pipeline

The Loreti et al. approach is **domain-agnostic** in its core steps.  Here is the generalised pipeline:

```
┌──────────────────────────────────────────────────────────┐
│  STAGE 0: Domain Scoping                                 │
│  • Define search queries for lens.org / Semantic Scholar │
│  • Define category taxonomy (adaptable per domain)       │
│  • Set quality thresholds                                │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 1: Data Acquisition (DAQ)                         │
│  • API queries → abstract + metadata retrieval           │
│  • Deduplication                                         │
│  • (Optional) Full-text acquisition via Unpaywall/OA     │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 2: Named Entity Recognition (NER)                 │
│  • LLM-based zero-shot NER guided by category taxonomy   │
│  • Sentence-level extraction                             │
│  • Category assignment                                   │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 3: Entity Resolution                              │
│  • Normalisation (case, singular, acronym expansion)     │
│  • LLM-assisted synonym/acronym detection                │
│  • Zipf's law validation as quality check                │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 4: KG Construction                                │
│  • Entity → node, co-occurrence → weighted edge          │
│  • SHACL validation against ontology                     │
│  • Semantic RE for high-weight pairs (LLM)               │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 5: KG-RAG Application                             │
│  • Cypher query generation                               │
│  • Hybrid retrieval (KG + vector)                        │
│  • LLM answer synthesis with provenance                  │
└──────────────────────────────────────────────────────────┘
```

### 8.2  Domain Adaptation Examples

| Domain | Search Scope | Category Taxonomy (seed) | Special Considerations |
|---|---|---|---|
| **Semiconductor Technology** | "semiconductor fabrication", "CMOS", "EUV lithography", "MOSFET", "FinFET" | Device Types, Materials, Fabrication Processes, Measurement Techniques, Physical Properties, Design Rules, Failure Modes | Strong patent literature; include patent APIs |
| **Quantum Computing** | "quantum computing", "qubit", "quantum error correction" | Qubit Types, Gate Operations, Algorithms, Error Models, Hardware Platforms, Materials | Rapidly evolving; temporal analysis critical |
| **Renewable Energy** | "solar cell", "wind turbine", "battery storage", "grid integration" | Energy Sources, Conversion Devices, Storage Technologies, Grid Components, Materials, Efficiency Metrics | Cross-links to policy/economics ontologies |
| **General Physics** | "classical mechanics", "quantum mechanics", "thermodynamics", "electromagnetism" | Physical Laws, Constants, Particles, Fields, Mathematical Structures, Experiments | Foundational; serves as upper-mid ontology for all above |

### 8.3  Cross-Domain Linking

Once multiple domain KGs exist, they can be linked via:
1. **Shared entities**: "plasma" appears in both fusion and semiconductor KGs.
2. **Upper ontology alignment**: both map to BFO/EMMO → shared concepts automatically identified.
3. **SKOS mappings**: `fusion:PlasmaProperty skos:related semiconductor:PlasmaProcessing`.
4. **Embedding similarity**: entities with similar vector embeddings across domains are candidate matches.

---

## 9  Authors' Pipeline — Availability and Reconstruction

### 9.1  Current Status

The Loreti et al. pipeline code is **not publicly available**.  The paper (arXiv:2504.07738v2, June 2025) does not include a code/data availability statement, and no GitHub repository was found linked to the paper or its authors at UKAEA/STFC.

The NER data (JSON files) we have was obtained directly from the authors.

### 9.2  Reconstruction via KGPlatform

Our KGPlatform already implements a superset of the Loreti et al. pipeline:

| Loreti et al. Step | KGPlatform Equivalent | Status |
|---|---|---|
| Data acquisition (lens.org API) | Not yet implemented (KGPlatform processes local documents) | Gap: need API integration |
| LLM-based NER (Llama 3.1/3.3) | KnowledgeGraphBuilder extraction (qwen3:8b, ontology-guided) | Working; uses OWL constraints instead of category list |
| Entity resolution (normalisation + LLM) | KnowledgeGraphBuilder assembly (EntityResolver, ERRunner) | Working |
| Neo4j KG construction | KnowledgeGraphBuilder storage (Neo4j + provenance) | Working |
| Zipf's law validation | Not yet implemented | Gap: add to quality metrics |
| KG-RAG (multi-prompt) | GraphQAAgent (4 strategies including hybrid_sota) | Working |
| Semantic RE (high-weight pairs) | KnowledgeGraphBuilder extraction (LLM-guided RE) | Working; ontology-constrained |

**Gaps to close:**
1. Add lens.org / Semantic Scholar / OpenAlex API integration for data acquisition.
2. Add Zipf's law validation as a quality metric.
3. Add the category-list-guided NER mode (complementary to ontology-guided).

---

## 10  Modular Ontology Extension — Linking Fusion to the World

### 10.1  The Modular Ontology Design Principle

Rather than building one monolithic fusion ontology, we follow **Modular Ontology Modeling (MOM)** (Shimizu et al., 2022):

- Each module covers a *coherent sub-domain* with clear boundaries.
- Modules are linked via **ontology design patterns** (ODPs) and explicit inter-module axioms.
- Modules can be independently versioned, validated, and extended.

The 28-category fusion schema is our **seed module**.  It needs to be:
1. Formalised as OWL.
2. Aligned to upper/mid-level ontologies.
3. Extended with modules for adjacent domains.

### 10.2  Alignment to Upper and Mid-Level Ontologies

See previous version (§4.2–4.3 in v1) for the full alignment tables (BFO, QUDT, EMMO, PROV-O, SKOS, schema.org).

### 10.3  The Rhizomatic Ontology Network

```
                    ┌─────────────────────────────────────────────┐
                    │            Upper Ontology (BFO)             │
                    └──────┬──────────┬──────────┬───────────┬────┘
                           │          │          │           │
                    ┌──────▼───┐ ┌────▼────┐ ┌──▼────┐ ┌────▼────┐
                    │  QUDT    │ │  EMMO   │ │ PROV-O│ │  SKOS   │
                    │ (units)  │ │(matls.) │ │(prov.)│ │(mapping)│
                    └──────┬───┘ └────┬────┘ └──┬────┘ └────┬────┘
                           │          │          │           │
              ┌────────────┼──────────┼──────────┼───────────┤
              │            │          │          │           │
        ┌─────▼──────┐ ┌──▼──────────▼──┐  ┌────▼───┐ ┌────▼─────────┐
        │  Fusion    │ │   Plasma       │  │ Nuclear│ │  Fusion      │
        │  Devices   │ │   Physics      │  │ Decom. │ │  Materials   │
        │  Module    │ │   Module       │  │ Module │ │  Module      │
        └─────┬──────┘ └──┬─────────┬──┘  └────┬───┘ └────┬─────────┘
              │            │         │          │           │
              └────────────┴─────────┴──────────┴───────────┘
                           │
                    ┌──────▼──────────────────────────────────┐
                    │  Fusion Energy KG (ABox)                │
                    │  108K entities, 718K co-occurrence edges│
                    │  + NER data from Loreti et al.          │
                    └─────────────────────────────────────────┘
```

### 10.4  Extension Workflow via OntologyExtender

1. **Gap detection** (§5): Graph structure analysis identifies missing concepts, under-connected areas, and schema incompleteness.
2. **Proposal generation**: Multi-agent debate → candidate new classes, properties, axioms.
3. **Expert review**: Human reviewer via Streamlit UI.
4. **Validation**: SHACL + SHACL2FOL → satisfiability check before deployment.
5. **Re-extraction**: KnowledgeGraphBuilder re-runs with updated ontology.

---

## 11  Long-Term Vision: Open Scientific Knowledge Infrastructure

### 11.1  The Problem with Current Systems

| System | Limitation |
|---|---|
| **Web of Science** | Proprietary (Clarivate); subscription-only; no KG structure, only metadata + citation links |
| **Scopus** | Proprietary (Elsevier); similar limitations |
| **Google Scholar** | Free but no API, no structured data, no graph queries |
| **Semantic Scholar** | Open API, good metadata, but no domain-specific KG structure or ontology alignment |
| **OpenAlex** | Fully open, rich metadata, but still flat (no entity-level KG, no domain ontology) |
| **Wikidata** | Open, graph-structured, but shallow coverage of specialised scientific concepts |

None of these provides **structured, domain-ontology-grounded knowledge graphs** with entity-level relationships and QA capabilities.

### 11.2  Proposed Open Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Public Interface Layer                       │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Web UI   │  │  REST API    │  │  SPARQL    │  │ GraphQL  │ │
│  │ (search, │  │  (entities,  │  │  Endpoint  │  │ Endpoint │ │
│  │  browse, │  │  relations,  │  │  (RDF/OWL  │  │ (custom  │ │
│  │  QA chat)│  │  papers)     │  │   queries) │  │  queries)│ │
│  └──────────┘  └──────────────┘  └────────────┘  └──────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     Authentication & Rate Limiting              │
│                     (API keys, tiered access)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  KGPlatform Core                          │  │
│  │  KnowledgeGraphBuilder │ GraphQAAgent │ OntologyExtender │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │ Neo4j   │  │ Qdrant  │  │ Fuseki  │  │ Ollama / cloud  │  │
│  │ (KG)    │  │ (vecs)  │  │ (RDF)   │  │ LLM inference   │  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3  Making It Public: Concrete Steps

1. **Data licensing**: Ensure all ingested data is OA-compatible.  Use OpenAlex (CC0) + arXiv (non-exclusive distribution) as primary sources.
2. **FAIR principles**: Make the KG Findable (DOI for dataset releases), Accessible (public API + SPARQL), Interoperable (BFO-aligned OWL, standard vocabularies), Reusable (CC-BY license on derived KG).
3. **RDF/Linked Data export**: Publish the KG as Linked Open Data (LOD) with persistent URIs.  Register in LOD Cloud.
4. **Web UI**: Public Streamlit or React frontend with:
   - Entity search and browse (typeahead, category filtering)
   - Graph visualisation (Cytoscape.js / Sigma.js)
   - QA chatbot (GraphQAAgent)
   - Ontology browser (class hierarchy, properties)
5. **API**: REST + GraphQL + SPARQL endpoints with:
   - Entity lookup, neighbourhood queries, path queries
   - Batch export (JSON-LD, Turtle, CSV)
   - Rate-limited public tier + authenticated researcher tier
6. **Community contribution**: Allow researchers to submit new papers, propose entity corrections, suggest ontology extensions via the OntologyExtender HITL interface.
7. **Sustainability**: Host on institutional infrastructure (university HPC / NFDI / EOSC) with long-term funding commitment.  Alternatively, deploy on community cloud (e.g., Hugging Face Spaces for the demo, cloud VPS for the full stack).

### 11.4  Comparison to Existing Open Initiatives

| Feature | Our Vision | OpenAlex | Wikidata | Domain KGs (e.g., BioPortal) |
|---|---|---|---|---|
| Structured KG with domain ontology | Yes | No (flat metadata) | Partially (generic schema) | Yes (per domain) |
| Entity-level relationships | Yes (co-occurrence + semantic) | No | Yes (but generic) | Yes |
| QA chatbot | Yes (GraphRAG) | No | No | Rarely |
| Full-text grounded | Yes (via Qdrant) | No | No | Rarely |
| Multi-domain | Yes (modular) | Yes (all of science) | Yes (all of knowledge) | No (single domain) |
| Ontology evolution | Yes (HITL OntologyExtender) | No | Manual editing | Manual curation |
| Mathematical analysis | Yes (TDA, spectral, FCA) | No | No | No |
| Open access | Yes | Yes | Yes | Varies |

---

---

## 12  Concrete Implementation Roadmap

### Phase 1: Load and Analyse (Current → 2 weeks)

| Task | Implementation Guide | KGPlatform Component | Priority |
|---|---|---|---|
| Load NER JSON into Neo4j | Script exists: `load_ner_json_to_neo4j.py` | Standalone | Done |
| Install Neo4j GDS plugin | Add to docker-compose: `NEO4J_PLUGINS: '["graph-data-science"]'` | Infrastructure | High |
| Run centrality analysis (PageRank, betweenness) | §12.1 | Neo4j GDS | High |
| Community detection (Leiden) | §12.1 | Neo4j GDS | High |
| Degree distribution + power law fit | §12.1 + Python `powerlaw` package | Python script | High |
| Zipf's law validation | Compare entity frequency rank-frequency to $f = C/r$ | Python script | Medium |
| FCA concept lattice on entity-category | §12.3 | Python `concepts` library | Medium |
| Persistent homology (per community) | §12.2 | Python `ripser` + `persim` | Medium |
| Spectral analysis (Fiedler, clustering, GFT) | §12.4 | Python `scipy` | Medium |

### Phase 2: Full-Text Enrichment + QA Layer (4–6 weeks)

| Task | Implementation Guide | KGPlatform Component | Priority |
|---|---|---|---|
| Unpaywall/OpenAlex API integration | §12.6 | New: `data_acquisition` module | High |
| Full-text PDF download (OA papers) | §12.6 | New: downloader script | High |
| PDF → text conversion + chunking | Existing KGPlatform document loaders | KnowledgeGraphBuilder | High |
| Embed chunks into Qdrant with paper_id metadata | Existing embedding pipeline | KnowledgeGraphBuilder | High |
| Formalise 28-category schema as OWL | Export FCA results → Protégé → OWL file | Manual + OntologyExtender | High |
| Configure GraphQAAgent with fusion ontology | Set ontology path in GraphQA config | GraphQAAgent | High |
| Implement hybrid RAG (KG + vector fusion) | Configure `hybrid_sota` strategy | GraphQAAgent | High |
| Benchmark QA strategies | Ragas-style evaluation on fusion questions | GraphQAAgent | Medium |

### Phase 3: Gap Detection + Ontology Network (6–10 weeks)

| Task | Implementation Guide | KGPlatform Component | Priority |
|---|---|---|---|
| Implement GapAnalyser module | §5 (topological, structural, coverage) | New: `gap_analyser` | High |
| Wire GapAnalyser → OntologyExtender feedback | REST endpoint: POST /gaps → proposal generation | Platform orchestration | High |
| Align fusion categories to BFO | §12.5 | OntologyExtender | Medium |
| Link to QUDT for physical quantities | OWL imports + property mapping | OntologyExtender | Medium |
| Create SKOS mappings to broader vocabularies | SKOS vocabulary file | Manual | Medium |
| Connect to Nuclear Decommissioning ontology | Cross-module owl:imports | OntologyExtender | Medium |
| Domain decomposition into sub-modules | §6.2 (Leiden + FCA + MOM) | Analysis scripts | Medium |

### Phase 4: Multi-Domain + Public Access (10–16 weeks)

| Task | Implementation Guide | KGPlatform Component | Priority |
|---|---|---|---|
| Generalise DAQ pipeline to new domains | §8 (semiconductor, quantum, etc.) | New: `daq` module | Medium |
| Build first cross-domain KG (fusion + physics) | §8.3 | KnowledgeGraphBuilder | Medium |
| Public REST API design + implementation | §11.3 | New: API gateway | Medium |
| SPARQL endpoint via Fuseki | Existing KGPlatform Fuseki | Infrastructure | Medium |
| Web UI for public browsing + QA | §11.3 (Streamlit → React if needed) | Frontend redesign | Low |
| RDF/Linked Data export + LOD registration | n10s export + W3C standards | Infrastructure | Low |
| Community contribution workflow | HITL OntologyExtender with public access | OntologyExtender | Low |

---

## 13  Summary

The Loreti et al. fusion KG provides a rich ABox grounded in a practical 28-category schema.  By formalising this schema into an OWL TBox, enriching it with full-text documents, and applying rigorous mathematical analysis, we gain:

1. **Full-text enrichment** → dramatic increase in entity coverage, relationship quality, and QA depth through dual-index KG + vector architecture.
2. **Mathematical analysis** (TDA, spectral theory, FCA, category theory, information theory) → structural insight into the knowledge landscape, knowledge gap detection, and quality metrics.
3. **Gap detection** → automated identification of topological voids, structural holes, missing relationships, and ontological incompleteness, integrated into the KGPlatform HITL feedback loop.
4. **Formal domain decomposition** → principled methods (community detection, FCA, coupling/cohesion metrics, MOM methodology) for splitting monolithic ontologies into versioned, composable sub-modules.
5. **Intelligent QA** (GraphRAG via KGPlatform) → grounded, explainable answers combining graph traversal and vector retrieval with provenance.
6. **Generalisation** → the pipeline is domain-agnostic; applicable to semiconductors, quantum computing, general physics, or any scientific/engineering domain.
7. **Modular extension** (BFO alignment, QUDT, EMMO, domain modules) → interoperability with the broader scientific knowledge ecosystem.
8. **Public access** → open API, SPARQL endpoint, web UI, community contribution — a next-generation alternative to closed systems like Web of Science.
9. **Continuous evolution** (OntologyExtender HITL loop) → a living ontology that grows with the research frontier.

This positions the fusion KG not as a static dataset, but as the **seed crystal** for a growing, modular, rhizomatic knowledge infrastructure — exactly the vision articulated in the GAIA Lab framework.

---

## 14  References

- Loreti, A. et al. (2025). "Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information." arXiv:2504.07738.
- Arp, R., Smith, B. & Spear, A. (2015). *Building Ontologies with Basic Formal Ontology*. MIT Press.
- Shimizu, C. et al. (2022). "Modular Ontology Modeling." *Semantic Web Journal* 14(3).
- Spivak, D. I. (2012). "Ologs: A Categorical Framework for Knowledge Representation." *PLoS ONE* 7(1).
- Ganter, B. & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer.
- Carlsson, G. (2009). "Topology and Data." *Bulletin of the AMS* 46(2), 255–308.
- Burt, R. S. (1992). *Structural Holes: The Social Structure of Competition*. Harvard University Press.
- Newman, M. E. J. (2006). "Modularity and community structure in networks." *PNAS* 103(23).
- Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
- Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). "Power-law distributions in empirical data." *SIAM Review* 51(4).
- Traag, V. A., Waltman, L. & van Eck, N. J. (2019). "From Louvain to Leiden." *Scientific Reports* 9.
- QUDT ontology: https://qudt.org
- EMMO: https://emmc.info/emmo-info/
- KGPlatform: https://github.com/DataScienceLabFHSWF/KGPlatform
- OpenAlex: https://openalex.org
- Unpaywall: https://unpaywall.org
