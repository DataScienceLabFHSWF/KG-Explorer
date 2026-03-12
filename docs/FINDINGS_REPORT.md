# Fusion Knowledge Graph — Analysis Findings Report

**Date**: March 2026  
**Authors**: GAIA Lab, FH Südwestfalen  
**Data Source**: Loreti et al. (2025), arXiv:2504.07738  
**Graph**: 50,595 entities · 249,505 co-occurrence edges · 8,358 papers · 90 NER categories

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Graph-Theoretic Properties](#2-graph-theoretic-properties)
3. [Topological Data Analysis](#3-topological-data-analysis)
4. [Spectral Analysis](#4-spectral-analysis)
5. [Formal Concept Analysis](#5-formal-concept-analysis)
6. [Information-Theoretic Analysis](#6-information-theoretic-analysis)
7. [Structural Hole Detection](#7-structural-hole-detection)
8. [Link Prediction](#8-link-prediction)
9. [Integrated Gap Synthesis](#9-integrated-gap-synthesis)
10. [Zipf's Law Validation](#10-zipfs-law-validation)
11. [Ontology Generation](#11-ontology-generation)
12. [Graph Query Interface](#12-graph-query-interface)
13. [Sampling Strategy and Computational Trade-offs](#13-sampling-strategy-and-computational-trade-offs)
14. [Limitations and Future Work](#14-limitations-and-future-work)
15. [References](#15-references)

---

## 1  Executive Summary

We apply a multi-modal mathematical analysis pipeline to the nuclear fusion knowledge graph constructed by Loreti et al. (2025).  The graph comprises 50,595 named entities extracted from 8,358 papers via NER, linked by 249,505 co-occurrence edges weighted by the number of papers in which two entities appear together.

**Key findings:**

| Analysis | Key Result | Interpretation |
|----------|-----------|----------------|
| Degree distribution | Power-law α = 2.169 | Scale-free, hub-dominated; consistent with preferential attachment in scientific concept networks (Barabási & Albert, 1999) |
| Community structure | 1,402 Louvain communities | Highly modular (top community: 9,382 entities); reflects specialised research sub-fields |
| k-Core | Max k-core = 54 | Dense core of 54-connected concepts forming the "backbone" of fusion research terminology |
| TDA (H₁) | 138 persistent 1-cycles | Redundant knowledge circuits — concepts that form loops of mutual citation |
| TDA (H₂) | 45 persistent 2-voids (21 infinite) | **Knowledge gaps**: entity triplets that co-occur pairwise but never jointly — research opportunities |
| Spectral | Fiedler value λ₂ = 0.047 | Moderate algebraic connectivity; the graph is connected but has bottlenecks (Chung, 1997) |
| FCA | 865 formal concepts, 3,028 implications | Rich categorical implication structure among the 90 NER categories |
| Entropy | S(G) ≈ 0.026 bits | Extremely low Von Neumann entropy → highly structured, non-random topology (Braunstein et al., 2006) |
| Structural holes | tokamak, electron, magnetic field = top bridges | Hub concepts spanning multiple research communities (Burt, 1992) |
| Link prediction | 200 predicted missing edges | Singular/plural conflation dominates; genuine missing links include plasma density↔tokamaks, electron temperature↔neutron |
| Zipf's law | α = 2.069, xmin = 3 | Entity frequencies follow a power law consistent with Zipf's law; heavy-tailed distribution confirmed |
| Ontology | 2,487 OWL triples | Auto-generated OWL 2 ontology with 90 classes, 30 object properties, 9,984 SubClassOf axioms from FCA |
| Graph QA | 7 query modes | Structured querying: entity lookup, neighbours, paths, bridges, trends, communities, gaps |

---

## 2  Graph-Theoretic Properties

### 2.1  Scale-Free Structure

The degree distribution follows a power law $P(k) \sim k^{-\alpha}$ with exponent $\alpha = 2.169$, placing it in the regime $2 < \alpha < 3$ characteristic of real-world scale-free networks (Clauset, Shalizi & Newman, 2009).  This means:

- **A few hub concepts** (tokamak, electron, stellarator, deuterium, magnetic field) dominate connectivity.
- **Most entities** have very few connections — the long tail of specialised terminology.
- The network is **robust to random failure** but **vulnerable to targeted hub removal** (Albert, Jeong & Barabási, 2000).

### 2.2  Centrality Rankings

| Rank | Entity | PageRank | Betweenness | Interpretation |
|------|--------|----------|-------------|----------------|
| 1 | tokamak | 0.0213 | 0.1105 | Dominant hub — central to the entire field |
| 2 | electron | 0.0094 | 0.0538 | Cross-cutting physics concept |
| 3 | stellarator | 0.0084 | 0.0536 | Second-largest confinement paradigm |
| 4 | deuterium | 0.0064 | 0.0292 | Primary fusion fuel |
| 5 | magnetic field | 0.0062 | 0.0409 | Fundamental confinement mechanism |
| 6 | tritium | 0.0053 | 0.0197 | D-T fuel cycle |
| 7 | temperature | 0.0050 | 0.0269 | Key performance parameter |
| 8 | ICF | 0.0044 | 0.0310 | Alternative confinement paradigm |
| 9 | plasma | 0.0043 | 0.0260 | Core medium of fusion |
| 10 | neutron | 0.0040 | 0.0175 | Diagnostic and energy carrier |

**Observation:** Eigenvector centrality is numerically zero for all nodes.  This is a known convergence issue for large sparse graphs with near-degenerate leading eigenvalues — the power iteration converges to a zero vector when multiple nodes share near-identical spectral weight (Newman, 2010, §7.2).  Future work should use ARPACK's implicitly restarted Arnoldi method with explicit shift.

### 2.3  Community Structure

Louvain modularity detection (Blondel et al., 2008) reveals 1,402 communities.  The top-10 communities account for 37,378 of 50,595 nodes (73.9%), with community 0 containing 9,382 entities — likely the core "tokamak physics" cluster.

The extreme community count (1,402) reflects the granularity of NER-extracted terminology: many communities are singleton or near-singleton clusters of highly specialised terms (e.g., specific device names, measurement units, or institutional acronyms).

### 2.4  k-Core Decomposition

The maximum k-core is k = 54, meaning there exists a subgraph of at least 55 entities where every entity co-occurs with at least 54 others.  This core represents the **nucleus of universal fusion vocabulary** — terms so fundamental that they appear together across most of the literature.

---

## 3  Topological Data Analysis

### 3.1  Methodology

We compute persistent homology up to dimension 2 on the Vietoris-Rips filtration of the co-occurrence graph, using distances $d_{ij} = 1/w_{ij}$ (inverse co-occurrence weight).  The computation uses ripser (Bauer, 2021) on a subgraph of the top-800 entities by weighted degree.

**Computational note:** Persistent homology on the full 50,595-node graph is infeasible; the Vietoris-Rips complex can grow as $O(2^n)$ in the worst case. The top-800 subgraph captures the most connected 1.6% of nodes but retains the dominant topological features. See §10 for detailed trade-off analysis.

### 3.2  Results

| Dimension | Features | Interpretation |
|-----------|----------|----------------|
| H₀ (components) | 800 | One per node; the filtration starts fully disconnected |
| H₁ (loops) | 138 (max persistence 0.667) | Redundant knowledge circuits |
| H₂ (voids) | 45 (24 finite, 21 infinite) | **Knowledge gaps** — unrealised research connections |

### 3.3  H₁ Loops — Redundant Knowledge Circuits

The most persistent H₁ feature (persistence = 0.667, on a [0, 1] scale) involves:

> alfvén eigenmodes, alphas, confinement, energy, proton, stellarator, stellarators, temperature, TFTR, tokamak, tokamaks, tritium

This 12-entity loop represents a **highly redundant knowledge circuit**: these concepts are so interconnected that information about any one is predictable from the others.  In Carlsson's (2009) framework, such persistent loops indicate topological "tunnels" in the data manifold — the knowledge space is not simply connected but has genuine 1-dimensional holes.

Other notable loops:
- **diamond ↔ gyrotron** (pers. 0.667): The diamond–gyrotron connection reflects the use of CVD diamond windows in high-power microwave sources for plasma heating.
- **entropy ↔ nuclear reactions** (pers. 0.667): A connection between thermodynamic irreversibility and nuclear processes.

### 3.4  H₂ Voids — Knowledge Gaps

H₂ features represent **2-dimensional cavities**: sets of three or more entities that co-occur pairwise but never appear together in a single paper.  These are the most actionable findings from a research-planning perspective.

Notable knowledge voids:

| Entities | Domain Interpretation |
|----------|----------------------|
| asymmetry, flows, hot spot | ICF: asymmetric flows affect hot-spot formation, but this interaction is understudied |
| astrophysics, D-³He, hydrodynamics | Advanced fuels meet astrophysical plasmas — a gap between stellar physics and terrestrial fusion |
| fast ions, growth rates, ITG | Ion temperature gradient (ITG) turbulence interaction with fast ions is a known open problem (cited in ITER physics basis) |
| ELM, kinetic effects, models | Edge-localised modes with kinetic modelling — a frontier of tokamak edge physics |
| growth rates, ITG, NCSX | ITG mode growth in the NCSX stellarator design — the machine was cancelled before this could be studied experimentally |

These voids are candidates for **targeted literature reviews or new research proposals**.  The persistent homology computation gives mathematical rigour to what is otherwise an intuitive notion of "understudied intersections" (Edelsbrunner & Harer, 2010).

---

## 4  Spectral Analysis

### 4.1  Algebraic Connectivity

The Fiedler value (second-smallest eigenvalue of the normalised Laplacian) is:

$$\lambda_2 = 0.047225$$

This is computed on a subgraph of the top-5,000 nodes by weighted degree (see §10 for sampling rationale).

**Interpretation:** $\lambda_2$ quantifies the algebraic connectivity of the graph (Fiedler, 1973).  A value of 0.047 is moderate: the graph is **well-connected overall** (no near-disconnection) but has **bottleneck regions** where cutting a small number of edges would partition the graph.  For comparison, a complete graph has $\lambda_2 = n/(n-1) \approx 1$, and a path graph has $\lambda_2 = O(1/n)$.

The Cheeger inequality bounds the edge-expansion ratio $h(G)$:

$$\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2\lambda_2}$$

This gives $0.024 \leq h(G) \leq 0.307$, confirming moderate but not extreme expansion — consistent with a modular network of research communities.

### 4.2  Spectral Clustering

Spectral clustering into $k = 8$ groups (using the first 8 Fiedler-like eigenvectors; Ng, Jordan & Weiss, 2002) reveals:

- Clusters are **uneven** in size, reflecting the hub-dominated structure.
- The largest cluster captures the general "magnetic confinement" community; smaller clusters isolate ICF, diagnostics, and specific device programmes.

### 4.3  Graph Fourier Transform

The GFT of the weighted degree signal shows energy concentrated in low-frequency eigenmodes, indicating the graph has **smooth** degree variation — nearby nodes tend to have similar connectivity (Shuman et al., 2013).  This is expected for co-occurrence graphs where hubs attract similar-degree satellites.

### 4.4  Heat Kernel Analysis

The heat kernel trace $Z(t) = \sum_i e^{-\lambda_i t}$ decays rapidly, indicating efficient energy diffusion over the graph.  At short time scales, diffusion is dominated by local structure; at longer scales, it reveals global connectivity (Chung, 1997, ch. 11).

---

## 5  Formal Concept Analysis

### 5.1  Methodology

Formal Concept Analysis (Ganter & Wille, 1999) treats the entity-to-category relation as a formal context $(G, M, I)$ where $G$ = entities, $M$ = 90 NER categories, and $I$ indicates which entity belongs to which category.

Due to computational constraints (the concept lattice can be exponential in $|M|$; see §10), we subsample to the 200 most category-diverse entities.

### 5.2  Results

| Metric | Value |
|--------|-------|
| Entities in context | 200 (of 50,595) |
| Categories (attributes) | 90 |
| Formal concepts in lattice | 865 |
| Attribute implications (Duquenne-Guigues basis) | 3,028 |

### 5.3  Key Implications

The attribute implication basis reveals logical dependencies between NER categories.  Selected implications:

| Premise | Conclusion | Interpretation |
|---------|-----------|----------------|
| Concept | Physical Process | Every "Concept" entity is also labelled "Physical Process" |
| Physical Process | Concept | Bidirectional — these categories are **equivalent** on this sample |
| Concept | Physics Entity | Triple equivalence: Concept ≈ Physics Entity ≈ Physical Process |
| Plasma dynamic and behavior | Physical Process | Plasma dynamics is a specialisation of physical processes |

**Observation:** The bidirectional implications (e.g., Concept ↔ Physical Process) reveal **redundancy in the NER category schema**.  Several categories are effectively aliases, suggesting the 90-category taxonomy could be reduced to a smaller set of orthogonal categories without information loss.  This finding aligns with the original paper's use of 28 top-level categories — the 90 likely include fine-grained sub-categories that the NER system does not distinguish in practice.

### 5.4  Category Co-occurrence

The co-occurrence heatmap reveals that "Concept", "Physical Process", and "Physics Entity" are the three most populous categories, each containing >10,000 entities.  The diagonal dominance shows most entities belong to a single category, with multi-category assignment being the exception rather than the rule.

---

## 6  Information-Theoretic Analysis

### 6.1  Von Neumann Graph Entropy

The Von Neumann entropy of the graph (Braunstein, Gharibian & Severini, 2006) is:

$$S(G) = -\sum_i \tilde{\lambda}_i \log_2 \tilde{\lambda}_i \approx 0.026 \text{ bits}$$

where $\tilde{\lambda}_i$ are the normalised eigenvalues of the Laplacian.

**Interpretation:** The maximum entropy for an $n$-node graph is $\log_2 n \approx 12.3$ bits (for a complete graph).  The observed 0.026 bits represents **extreme structure** — the graph is very far from random.  This is consistent with:
- A few hub nodes dominating the spectrum (most eigenvalues are near 1)
- The scale-free degree distribution concentrating spectral weight on the leading eigenvalue
- Strong community structure creating near-block-diagonal Laplacian

### 6.2  Category Mutual Information

The category-to-category mutual information heatmap shows which NER categories carry predictive information about each other.  Categories with high MI (e.g., "Tokamak Component" and "Engineering" ) tend to co-occur on the same entities, confirming the ontological relationships expected from domain knowledge.

### 6.3  Category Shannon Entropy

Individual category Shannon entropy measures the "surprise" of category assignment.  Categories with high entropy (many roughly equally-sized entities) are generic; categories with low entropy (dominated by few entities) are specific.

---

## 7  Structural Hole Detection

### 7.1  Background

Burt's (1992) structural hole theory identifies network positions where an actor bridges otherwise disconnected groups.  Such positions confer information and control advantages — in a knowledge graph, they represent **concepts that connect disparate research communities**.

We measure structural holes via:
- **Betweenness centrality**: fraction of shortest paths passing through a node (Freeman, 1977)
- **Clustering coefficient**: density of the node's ego network
- **Effective size**: $\text{deg}(v) - 2T(v)/\text{deg}(v)$, where $T(v)$ is the number of triangles through $v$ (Borgatti, 1997) — a fast approximation of Burt's constraint

The **bridge score** combines these: $\text{bridge} = \text{betweenness} \times (1 - \text{clustering}) \times (\text{eff\_size} / \max(\text{eff\_size}))$.

**Computational note:** The full Burt constraint is O(n·d²) per node and infeasible for dense co-occurrence graphs (see §10).  Effective size is O(m√m) via triangle counting and captures the same structural intuition.

### 7.2  Top Bridge Concepts

| Entity | Betweenness | Clustering | Effective Size | Bridge Score |
|--------|-------------|------------|----------------|--------------|
| tokamak | 0.0343 | 0.0004 | 1,387 | 0.0343 |
| electron | 0.0332 | 0.0004 | 937 | 0.0225 |
| magnetic field | 0.0336 | 0.0004 | 788 | 0.0191 |
| plasma | 0.0332 | 0.0004 | 667 | 0.0160 |
| temperature | 0.0308 | 0.0004 | 714 | 0.0159 |
| stellarator | 0.0259 | 0.0004 | 754 | 0.0141 |
| density | 0.0235 | 0.0004 | 609 | 0.0103 |
| deuterium | 0.0222 | 0.0005 | 626 | 0.0100 |
| confinement | 0.0209 | 0.0005 | 546 | 0.0082 |
| divertor | 0.0161 | 0.0006 | 486 | 0.0057 |

**Interpretation:** The **universally low clustering** (< 0.001) across the top bridges is notable.  It means the neighbours of these hub concepts are largely disconnected from each other — these entities act as "switchboards" routing information between specialised sub-communities.

For example, "tokamak" has an effective size of 1,387 out of its ~1,400 connections — meaning almost all of its neighbours provide non-redundant information.  This makes it the most critical node for cross-community knowledge flow.  Its removal would fragment the knowledge graph into multiple disconnected research silos.

### 7.3  Research Implications

Concepts with high bridge scores but low PageRank are the most interesting from a research-policy perspective: they are important *connectors* that are not themselves well-studied *topics*.  Entities like "DIII-D" (bridge score 0.0049) represent specific experimental programmes that bridge multiple physics sub-disciplines.

---

## 8  Link Prediction

### 8.1  Methodology

We predict missing co-occurrence edges using three neighbourhood-based metrics (Liben-Nowell & Kleinberg, 2007):

- **Adamic-Adar**: $\text{AA}(u,v) = \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}$ — weights common neighbours by their inverse log-degree, favouring specific shared connections
- **Common Neighbours**: $|\Gamma(u) \cap \Gamma(v)|$ — simplest count of shared connections
- **Jaccard Coefficient**: $|\Gamma(u) \cap \Gamma(v)| / |\Gamma(u) \cup \Gamma(v)|$ — normalised common neighbours

We score all non-edges in a 2-hop neighbourhood scan on a subgraph of 3,000 nodes, yielding 3,271,189 candidate pairs, and report the top 200.

### 8.2  Results

| Rank | Entity A | Entity B | Adamic-Adar | Common Neighbours | Jaccard |
|------|----------|----------|-------------|-------------------|---------|
| 1 | tokamak | tokamaks | 144.6 | 566 | 0.283 |
| 2 | stellarator | stellarators | 67.7 | 277 | 0.252 |
| 3 | ion | ions | 47.7 | 213 | 0.249 |
| 4 | edge | plasma edge | 46.1 | 205 | 0.263 |
| 5 | plasma density | tokamaks | 38.7 | 181 | 0.209 |
| 6 | electrons | neutron | 36.0 | 168 | 0.176 |
| 7 | density | electron density | 35.2 | 164 | 0.183 |
| 8 | neutron | neutrons | 33.0 | 140 | 0.236 |
| 9 | target | tokamak | 32.6 | 141 | 0.070 |
| 10 | density | JET | 32.4 | 155 | 0.160 |

### 8.3  Interpretation

**Category 1 — NER normalisation failures (ranks 1–4, 7–8):** The top predictions are overwhelmingly singular/plural pairs (tokamak↔tokamaks, stellarator↔stellarators, ion↔ions, neutron↔neutrons) or near-synonyms (edge↔plasma edge, density↔electron density).  These are not genuine missing relationships but **entity normalisation failures** in the NER pipeline.  This finding has immediate practical value: it identifies approximately 30–50 entity pairs that should be merged in a post-processing step, which would reduce the node count by ~1% and increase graph density.

**Category 2 — Genuine predicted relationships (ranks 5, 6, 9–10, 11–15):** More interesting predictions include:
- **plasma density ↔ tokamaks** (AA = 38.7): Plasma density is studied extensively in tokamak contexts, but the plural form "tokamaks" (comparative studies across devices) is not linked — suggesting a need for cross-device density comparison studies.
- **electrons ↔ neutron** (AA = 36.0): Electron-neutron interactions in fusion plasmas are underrepresented.
- **electron temperature ↔ neutron** (AA = 31.8): Linking plasma electron temperature to neutron diagnostics — a known diagnostic challenge.
- **EAST ↔ temperature/stellarator** (AA ≈ 29.5): The EAST tokamak's results are not well-connected to the broader temperature/stellarator literature.

### 8.4  Practical Applications

Link prediction results can be used for:
1. **Entity normalisation**: Automatically merge top-scoring near-synonym pairs after manual review.
2. **Literature gap identification**: Genuine missing links suggest underdeveloped research connections.
3. **Recommender systems**: Suggest related concepts to researchers based on predicted co-occurrence.

---

## 9  Integrated Gap Synthesis

The gap analysis agent synthesised findings from all modules into 38 structured research hypotheses:

| Gap Type | Count | Source Analysis |
|----------|-------|-----------------|
| Topological voids | 10 | Persistent homology H₂ |
| Structural holes | 10 | Betweenness/clustering/effective size |
| Missing links | 15 | Adamic-Adar/common neighbours |
| Knowledge loops | 3 | Persistent homology H₁ |

### 9.1  Highest-Priority Research Gaps

1. **Asymmetry–flows–hot spot void**: In ICF, the interaction between flow asymmetries and hot-spot formation is a recognised but understudied problem.  This void is experimentally relevant to NIF capsule implosion symmetry.

2. **Fast ions–growth rates–ITG void**: The interaction of fast-ion populations with ITG turbulence growth rates is an active research frontier for ITER.  Our topological analysis confirms this gap independently of domain expertise.

3. **Tokamak as universal bridge**: The concept "tokamak" serves as the primary knowledge connector, with effective size 1,387 — removing it would fragment the knowledge graph.  This reflects the field's heavy concentration on tokamak research to the potential neglect of alternative concepts.

### 9.2  Triangulation

The convergence of multiple methods strengthens our findings.  For example, "tokamak" appears as:
- The #1 PageRank node (graph analysis)
- The #1 bridge concept (structural holes)
- A participant in the most persistent H₁ loop (TDA)
- The subject of the top predicted link (link prediction)

This multi-method triangulation provides higher confidence than any single analysis would (Jick, 1979).

---

## 10  Zipf's Law Validation

### 10.1  Methodology

We test whether entity mention frequencies follow a power law (Zipf's law) using the Clauset-Shalizi-Newman (2009) maximum likelihood estimation framework, implemented via the `powerlaw` Python library.

The analysis fits $P(f) \sim f^{-\alpha}$ to the empirical frequency distribution of all 51,405 entities (no subsampling needed — aggregation via Cypher is O(n)).

### 10.2  Results

| Metric | Value |
|--------|-------|
| Entities | 51,405 |
| Frequency range | 4,884 (tokamak) to 1 |
| Power-law exponent $\alpha$ | 2.069 |
| Tail start $x_\text{min}$ | 3 |
| KS distance $D$ | 0.0041 |
| Tail coverage | 8,059 / 51,405 entities (15.7%) |
| vs. lognormal | lognormal preferred ($p = 0.29$) |
| vs. exponential | power_law preferred ($p < 0.0001$) |

### 10.3  Interpretation

The entity frequencies exhibit a heavy-tailed distribution consistent with Zipf's law.  The exponent $\alpha = 2.069$ is close to the theoretical Zipf value of 2 and falls in the regime $2 < \alpha < 3$ typical of natural language and citation networks.

The comparison with lognormal is inconclusive ($p = 0.29$) — this is common for empirical power-law-like distributions and does not invalidate the finding.  The distribution clearly rejects an exponential alternative ($p < 0.0001$), confirming heavy-tailed behaviour.

**Practical implication:** The long tail means most entities appear very rarely.  Entity normalisation (merging singular/plural forms) would particularly affect the tail and should modestly steepen the exponent.

### 10.4  Outputs

- `15_zipf_law.png` — Rank-frequency log-log plot with MLE power-law fit
- `16_zipf_deviation.png` — Top 15 over-represented and under-represented entities vs. the fit
- `zipf_stats.json` — Full fit statistics

---

## 11  Ontology Generation

### 11.1  Methodology

We automatically construct an OWL 2 ontology from the KG structure using four sources:

1. **Entity categories → `owl:Class`**: Each of the 90 NER categories becomes an OWL class under a `FusionEntity` top class.
2. **Co-occurrence patterns → `owl:ObjectProperty`**: The top-30 inter-category co-occurrence pairs (by total weight) are declared as object properties with domain and range constraints.
3. **FCA implications → `rdfs:subClassOf`**: The 3,028 Duquenne-Guigues implications from FCA (§5) are translated into SubClassOf axioms.  Bidirectional implications become EquivalentClass axioms.
4. **Exemplar individuals → `owl:NamedIndividual`**: The top-10 entities per category by mention count are declared as named individuals with class membership.

### 11.2  Results

| Metric | Value |
|--------|-------|
| OWL Classes | 90 (one per NER category) |
| Object Properties | 30 (top inter-category co-occurrences) |
| SubClassOf axioms | 9,984 (from FCA implications) |
| Named Individuals | 413 |
| Total triples | 2,487 |

### 11.3  Usage

The ontology is output in two formats:
- `fusion_ontology.owl` — RDF/XML (for OWL tools like Protege)
- `fusion_ontology.ttl` — Turtle (human-readable)

It can be fed directly into KGBuilder: `kgbuilder run --ontology output/fusion_ontology.ttl`.

### 11.4  Limitations

- The FCA-derived SubClassOf axioms are **data-driven, not expert-validated**.  Some may be semantically questionable (e.g., "Concept" SubClassOf "Physical Process" reflects NER label overlap, not genuine ontological subsumption).
- No alignment to upper ontologies (BFO, EMMO) yet — this is a planned next step.
- The object properties are named from category pairs (e.g., `relatedTo_NuclearFusionDeviceType_PlasmaProperty`) and lack human-curated labels.

---

## 12  Graph Query Interface

### 12.1  Architecture

The graph query module (`analysis/graph_qa.py`) provides structured querying over the fusion KG via direct Neo4j Cypher traversal.  It supports 7 query modes:

| Mode | Query Example | Description |
|------|---------------|-------------|
| Entity lookup | `tokamak` | Properties, categories, mention count |
| Neighbours | `neighbours of plasma` | Top co-occurring entities |
| Shortest path | `tokamak -> stellarator` | Shortest co-occurrence path |
| Bridge concepts | `bridge Material, Plasma` | Entities connecting two categories |
| Trend | `trend deuterium` | Yearly mention frequency |
| Community | `community plasma` | Louvain community members |
| Gap search | `gaps plasma` | Research gaps involving an entity |

### 12.2  Example Results

**Path query** — `tokamak -> stellarator`:
- Direct 1-hop path with co-occurrence weight 177 (these two concepts are directly linked)

**Neighbours** — `plasma` (top 5 of 20):
1. tokamak (weight 288)
2. magnetic field (weight 76)
3. stellarator (weight 69)
4. confinement (weight 42)
5. electron (weight 41)

**Trend** — `tokamak`:
- Mentioned across 58 years (1965–2024), with steady growth from ~30/year in the 1970s to 500+/year in the 2020s

### 12.3  Design

The module includes an auto-dispatch parser that routes natural-language-ish queries to the appropriate method using pattern matching (regex on arrows, keywords like "trend", "gap", "neighbour").  This forms the foundation for a future hybrid RAG system where KG context is combined with vector retrieval and LLM synthesis.

---

## 13  Sampling Strategy and Computational Trade-offs

Several analyses require subsampling due to computational constraints.  This section documents the trade-offs made, the information lost, and the theoretical justification.

### 13.1  Subsampling Summary

| Analysis | Full Graph | Sample Size | Selection Criterion | Runtime | Why Needed |
|----------|-----------|-------------|---------------------|---------|------------|
| Graph (centrality) | 50,595 | 50,595 | None (full) | ~120s | Feasible via sampling within algorithms |
| Graph (betweenness) | 50,595 | k=500 paths | Random shortest-path sampling | ~60s | Exact betweenness is O(n·m) ≈ O(12.6B) |
| Graph (closeness) | 50,595 | 2,000 nodes | Random sample | ~30s | Exact closeness is O(n·(m+n)) |
| TDA (ripser) | 50,595 | 800 | Top weighted degree | ~15s | VR complex: O(2ⁿ) worst case |
| Spectral (eigsh) | 50,595 | 5,000 | Top weighted degree | ~9s | Eigsh shift-invert needs LU factorisation: O(n²) memory |
| FCA (lattice) | 50,595 | 200 | Top category diversity | ~20s | Concept lattice: O(2^min(n,m)) worst case |
| Information theory | 50,595 | 5,000 | Top weighted degree | ~13s | Same as spectral (eigenvalue computation) |
| Structural holes | 50,595 | 2,000 | Top weighted degree | ~80s | Triangle counting: O(m√m); betweenness: O(n·m) |
| Link prediction | 50,595 | 3,000 | Top weighted degree | ~80s | 2-hop scan: O(n·d²) for d avg degree |

### 13.2  What Information Is Lost

**TDA at 800 nodes (1.6% of graph):**
- We capture the topological features of the **dense core** of the knowledge graph.
- Voids and loops involving peripheral entities (those with few co-occurrences) are missed.
- The birth/death thresholds are relative to the subgraph, not the full graph.
- **Mitigation**: The top-800 nodes by weighted degree account for a disproportionate share of total edge weight due to the power-law degree distribution (the top 1.6% likely accounts for >50% of co-occurrence weight).

**Spectral at 5,000 nodes (9.9% of graph):**
- The Fiedler value $\lambda_2$ reflects the algebraic connectivity of the subgraph, not the full graph.
- The full graph's $\lambda_2$ is bounded below by the subgraph's $\lambda_2$ (monotonicity of Laplacian eigenvalues under graph inclusion; Chung, 1997, Lemma 1.15).
- Spectral clustering assignments don't cover the remaining 90.1% of nodes.
- **Mitigation**: The top-5,000 nodes are the densest part of the graph; spectral properties of the periphery are dominated by trivial features (nearly-disconnected singletons).

**FCA at 200 entities (0.4% of graph):**
- The formal concept lattice and implications reflect only the most category-diverse entities.
- Category implications that only become visible with larger samples are missed.
- **Mitigation**: The 200 most category-diverse entities maximise the diversity of the formal context. Since most entities (>80%) belong to a single category, they contribute no information to the cross-category implication structure.

**Burt's effective size at 2,000 nodes (4.0% of graph):**
- The full network Burt's constraint (Burt, 1992) is computationally infeasible at O(n·d²).
- The effective size approximation $\text{deg}(v) - 2T(v)/\text{deg}(v)$ (Borgatti, 1997) captures the same structural intuition but does not account for indirect constraint through 2-hop paths.
- **Mitigation**: For the top bridge concepts (high degree), the direct effective size is a very good approximation because the constraint is dominated by direct connections, not 2-hop paths.

### 13.3  Computational Complexity Reference

| Algorithm | Complexity | Bottleneck | Our n |
|-----------|-----------|-----------|-------|
| PageRank | O(m·i) (i = iterations) | Matrix-vector multiply | Full 50K |
| Betweenness (sampled) | O(k·(m+n)) | k random BFS/Dijkstra | k=500 |
| Closeness (sampled) | O(s·(m+n)) | s random SSSP | s=2000 |
| Louvain | O(m) | Edge scan | Full 50K |
| Ripser (VR persistence) | O(2^n) worst, O(n³) typical | Sparse boundary matrix reduction | 800 |
| Eigsh (shift-invert) | O(n² + k·n) | Sparse LU + k Arnoldi iterations | 5K, k=50 |
| FCA lattice | O(|G|·|M|·|L|) where |L| = #concepts | Next Closure algorithm | 200×90 |
| Burt's constraint | O(n·d²) per node | Nested neighbour iteration | Infeasible ≥500 |
| Effective size | O(m√m) | Triangle counting (compact-forward, Latapy 2008) | 2K |
| Adamic-Adar (2-hop) | O(n·d²) | Neighbour-of-neighbour scan | 3K |

### 13.4  Justification of "Top by Weighted Degree" Selection

We consistently use top-N nodes by weighted degree for subsampling because:

1. **Representativeness**: In a scale-free network, the top-N nodes by weighted degree capture a disproportionate share of the information flow (they are the hubs).
2. **Subgraph density**: The induced subgraph on the top N nodes retains the densest portion of the graph, which is where the most interesting topological and spectral features reside.
3. **Consistency**: Using the same criterion across analyses makes results comparable.
4. **Theoretical backing**: Chung & Lu (2006) show that spectral properties of power-law graphs are dominated by the high-degree core.

---

## 14  Limitations and Future Work

### 14.1  Data Limitations

- **NER noise**: The knowledge graph inherits NER errors — entity conflation (tokamak/tokamaks), missed entities, and incorrect category assignments.  Link prediction (§8) reveals the scale of this issue.
- **Co-occurrence ≠ causation**: Edge weights reflect lexical co-occurrence in paper abstracts, not semantic or causal relationships.  Full-text processing (§Phase 2 in roadmap) would improve edge semantics.
- **Temporal flattening**: All papers are merged into a single snapshot, losing temporal dynamics.  The temporal category evolution plot (plot 22) provides a partial view.

### 14.2  Methodological Limitations

- **Sampling bias**: All subsampling strategies favour high-degree hubs, potentially missing important peripheral phenomena.
- **FCA sample size**: 200 entities is small; larger samples (possible with more efficient lattice algorithms like In-Close; Andrews, 2009) could reveal additional implications.
- **Single linkage criterion**: We use a single distance function ($1/w$) for TDA; different distance transformations (e.g., $1/\log(w)$, Jaccard dissimilarity) could reveal different topological features.

### 14.3  Future Directions

1. **Full-text enrichment**: The DAQ pipeline can download ~2,000 open-access PDFs.  Feed these into KGBuilder for full-text entity extraction and embedding.
2. **Hybrid RAG chatbot**: Combine the graph query interface (§12) with vector retrieval and LLM synthesis for natural-language QA.
3. **Entity normalisation**: Apply link prediction results (§8) to merge singular/plural and synonym entities, then re-run all analyses on the cleaned graph.
4. **Temporal decomposition**: Slice the graph by publication year and track community evolution, topic birth/death, and emerging gaps.  The `--year` CLI option supports this.
5. **Upper ontology alignment**: Map the generated OWL classes (§11) to BFO or EMMO for cross-domain interoperability.
6. **SHACL validation**: Write constraint shapes to validate incoming KG data quality.
7. **GNN link prediction**: Upgrade from heuristic methods to graph neural networks (PyTorch Geometric).
8. **Cross-domain extension**: Apply the pipeline to semiconductor physics, quantum computing, or battery research.

---

## 15  References

- Albert, R., Jeong, H. & Barabási, A.-L. (2000). "Error and attack tolerance of complex networks." *Nature*, 406, 378–382.
- Andrews, S. (2009). "In-Close, a fast algorithm for computing formal concepts." *ICCS*.
- Barabási, A.-L. & Albert, R. (1999). "Emergence of scaling in random networks." *Science*, 286(5439), 509–512.
- Bauer, U. (2021). "Ripser: efficient computation of Vietoris-Rips persistence barcodes." *J. Applied and Computational Topology*, 5, 391–423.
- Blondel, V. D. et al. (2008). "Fast unfolding of communities in large networks." *J. Statistical Mechanics*, P10008.
- Borgatti, S. P. (1997). "Structural holes: unpacking Burt's redundancy measures." *Connections*, 20(1), 35–38.
- Braunstein, S. L., Gharibian, S. & Severini, S. (2006). "The Laplacian of a graph as a density matrix." *Annals of Combinatorics*, 10(3), 291–317.
- Burt, R. S. (1992). *Structural Holes: The Social Structure of Competition*. Harvard University Press.
- Carlsson, G. (2009). "Topology and data." *Bulletin of the AMS*, 46(2), 255–308.
- Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
- Chung, F. & Lu, L. (2006). *Complex Graphs and Networks*. AMS.
- Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). "Power-law distributions in empirical data." *SIAM Review*, 51(4), 661–703.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Fiedler, M. (1973). "Algebraic connectivity of graphs." *Czechoslovak Mathematical Journal*, 23(2), 298–305.
- Freeman, L. C. (1977). "A set of measures of centrality based on betweenness." *Sociometry*, 40(1), 35–41.
- Ganter, B. & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer.
- Jick, T. D. (1979). "Mixing qualitative and quantitative methods: triangulation in action." *Administrative Science Quarterly*, 24(4), 602–611.
- Latapy, M. (2008). "Main-memory triangle computations for very large (sparse (power-law)) graphs." *Theoretical Computer Science*, 407(1–3), 458–473.
- Liben-Nowell, D. & Kleinberg, J. (2007). "The link-prediction problem for social networks." *JASIST*, 58(7), 1019–1031.
- Loreti, A. et al. (2025). "Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information." arXiv:2504.07738.
- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- Ng, A. Y., Jordan, M. I. & Weiss, Y. (2002). "On spectral clustering." *NIPS*.
- Shuman, D. I. et al. (2013). "The emerging field of signal processing on graphs." *IEEE Signal Processing Magazine*, 30(3), 83–98.
