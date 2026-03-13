# Implementation Guides

*Extracted from the Concept Document — practical code examples and step-by-step procedures for each analysis framework.*

---

## 1  Guide: Graph-Theoretic Analysis in Neo4j GDS

**Prerequisites**: Neo4j 5.x with GDS plugin installed.

```cypher
-- Step 1: Project the co-occurrence graph into GDS memory
CALL gds.graph.project(
  'fusion-entities',
  'Entity',
  {CO_OCCURS_WITH: {
    type: 'CO_OCCURS_WITH',
    properties: {weight: {defaultValue: 1.0}}
  }}
)

-- Step 2: Degree distribution
CALL gds.degree.stream('fusion-entities',
  {relationshipWeightProperty: 'weight'})
YIELD nodeId, score AS weightedDegree
WITH gds.util.asNode(nodeId) AS node, weightedDegree
RETURN node.name AS entity, weightedDegree
ORDER BY weightedDegree DESC
LIMIT 100

-- Step 3: PageRank
CALL gds.pageRank.stream('fusion-entities',
  {maxIterations: 100, dampingFactor: 0.85,
   relationshipWeightProperty: 'weight'})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.name AS entity, score
ORDER BY score DESC LIMIT 50

-- Step 4: Betweenness centrality
CALL gds.betweenness.stream('fusion-entities')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.name AS entity, score AS betweenness
ORDER BY betweenness DESC LIMIT 50

-- Step 5: Community detection (Leiden)
CALL gds.leiden.stream('fusion-entities',
  {relationshipWeightProperty: 'weight'})
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
RETURN communityId,
       collect(node.name)[..10] AS sampleEntities,
       count(*) AS size
ORDER BY size DESC

-- Step 6: k-Core decomposition (via GDS or custom)
-- GDS doesn't have built-in k-core;
-- export and compute in Python:
CALL gds.graph.nodeProperties.stream('fusion-entities',
  ['degree'])
YIELD nodeId, propertyValue
-- Then use networkx.core_number() in Python

-- Step 7: Cleanup
CALL gds.graph.drop('fusion-entities')
```

## 2  Guide: Persistent Homology with ripser

```python
import numpy as np
from neo4j import GraphDatabase
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

driver = GraphDatabase.driver("bolt://localhost:7687",
                              auth=("neo4j", "password"))

def get_community_subgraph(community_id, max_nodes=500):
    query = """
    MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
    WHERE a.community = $cid AND b.community = $cid
    RETURN a.name_norm AS src, b.name_norm AS tgt,
           r.weight AS weight
    """
    with driver.session() as session:
        result = session.run(query, cid=community_id)
        return [(r["src"], r["tgt"], r["weight"])
                for r in result]

edges = get_community_subgraph(community_id=0)
nodes = sorted(set(s for s, _, _ in edges)
               | set(t for _, t, _ in edges))
idx = {n: i for i, n in enumerate(nodes)}
n = len(nodes)

dist = np.full((n, n), np.inf)
np.fill_diagonal(dist, 0)
for src, tgt, w in edges:
    d = 1.0 / max(w, 1e-10)
    dist[idx[src]][idx[tgt]] = d
    dist[idx[tgt]][idx[src]] = d

result = ripser(dist, maxdim=2, distance_matrix=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for dim in range(3):
    ax = axes[dim]
    dgm = result['dgms'][dim]
    ax.set_title(f'H{dim} ({len(dgm)} features)')
    plot_diagrams(dgm, ax=ax, show=False)
plt.tight_layout()
plt.savefig('persistence_diagrams.png', dpi=150)

h2 = result['dgms'][2]
if len(h2) > 0:
    persistence = h2[:, 1] - h2[:, 0]
    significant = h2[persistence > np.median(persistence)]
    print(f"Found {len(significant)} significant H2 voids")
```

## 3  Guide: Formal Concept Analysis

```python
import json
from pathlib import Path
from collections import defaultdict
from concepts import Context

entity_categories = defaultdict(set)
for json_file in Path("data").glob("*_NER.json"):
    with open(json_file) as f:
        data = json.load(f)["data"]
    for paper in data:
        for sentence_block in paper.get("NER-RE", []):
            for ent in sentence_block.get("entities", []):
                name = ent.get("entity", "").strip().lower()
                cat = ent.get("category", "").strip()
                if name and cat:
                    entity_categories[name].add(cat)

all_categories = sorted(set(
    c for cats in entity_categories.values() for c in cats
))
entities = sorted(entity_categories.keys())

booleans = [[cat in entity_categories[ent]
             for cat in all_categories]
            for ent in entities]

ctx = Context(entities, all_categories, booleans)
lattice = ctx.lattice

print(f"Objects: {len(entities)}")
print(f"Attributes: {len(all_categories)}")
print(f"Formal concepts: {len(lattice)}")

for impl in lattice.implications():
    if impl.conclusion:
        print(f"  {set(impl.premise)} → {set(impl.conclusion)}")
```

## 4  Guide: Spectral Analysis

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

n = len(nodes)
row, col, data = [], [], []
for src_idx, tgt_idx, weight in edges:
    row.extend([src_idx, tgt_idx])
    col.extend([tgt_idx, src_idx])
    data.extend([weight, weight])

A = csr_matrix((data, (row, col)), shape=(n, n))
L_norm = laplacian(A, normed=True)

k = 20
eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM')

print(f"Fiedler value: {eigenvalues[1]:.6f}")

X = eigenvectors[:, 1:k]
X /= np.linalg.norm(X, axis=1, keepdims=True)
kmeans = KMeans(n_clusters=8, n_init=20)
labels = kmeans.fit_predict(X)

# Heat kernel
for t in [0.1, 1.0, 10.0, 100.0]:
    heat_trace = np.sum(np.exp(-t * eigenvalues))
    print(f"Heat kernel trace at t={t}: {heat_trace:.4f}")
```

## 5  Guide: Category-Theoretic Ontology Alignment

This is a **design methodology**, not runnable code.

**Step-by-step procedure:**

1. **Model source ontology as a category** $\mathcal{O}_{\text{fusion}}$:
   - Objects: {Abstract, Sentence, Entity, Person, TimeReference, KeyWord, Category, Field}
   - Morphisms: {HAS_SENTENCE, CONTAINS, CC, HAS_FIRST_AUTHOR, WAS_PUBLISHED_IN, HAS_KEYWORD, IN_CATEGORY, HAS_FIELD}
   - Verify composition: HAS_SENTENCE ∘ CONTAINS = MENTIONS (check this holds in the data).

2. **Model target ontology as a category** $\mathcal{O}_{\text{BFO}}$:
   - Objects: {MaterialEntity, Process, Quality, Role, Site, TemporalRegion, ...}
   - Morphisms: {partOf, hasPart, participatesIn, bearerOf, realizesRole, ...}

3. **Define the functor** $F: \mathcal{O}_{\text{fusion}} \to \mathcal{O}_{\text{BFO}}$:
   - Map each fusion class to a BFO class.
   - Map each fusion relationship to a BFO relationship (or mark as unmapped).
   - Verify: does F preserve composition?

4. **Check functor type**: Faithful? Full? The missing BFO relationships suggest **extension opportunities**.

5. **For merging** (pushout): List shared concepts, verify compatibility, identify or flag for resolution.

**Software support**: CatLab.jl (Julia) or Protégé + OWL alignment plugins.

## 6  Guide: Full-Text Acquisition and Enrichment

```python
import requests
import time
from pathlib import Path

UNPAYWALL_EMAIL = "your.email@institution.edu"
OUTPUT_DIR = Path("data/fulltexts")
OUTPUT_DIR.mkdir(exist_ok=True)

def check_open_access(doi: str) -> dict | None:
    url = f"https://api.unpaywall.org/v2/{doi}"
    resp = requests.get(url,
                        params={"email": UNPAYWALL_EMAIL},
                        timeout=10)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("is_oa"):
        best = data.get("best_oa_location", {})
        return {
            "pdf_url": best.get("url_for_pdf"),
            "html_url": best.get("url_for_landing_page"),
            "license": best.get("license"),
            "version": best.get("version"),
        }
    return None

def download_pdf(url: str, output_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent":
                                "KGPlatform/1.0 (research)"})
        if resp.status_code == 200 \
                and 'pdf' in resp.headers.get(
                    'content-type', '').lower():
            output_path.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False
```

---

## 7  Guide: Community-Scoped Analysis

*Implementation of the pattern described in §4.7 of the Concept Document.*  
**Module**: `analysis/community_scoped.py` — `run(driver, year=None, max_nodes=300)`

### 7.1  General Scaffold

```python
import networkx as nx
import community as community_louvain  # python-louvain

def community_scoped_analysis(G: nx.Graph, method_fn, **method_kwargs):
    """
    Run method_fn on every community subgraph of G.

    Parameters
    ----------
    G           : full weighted co-occurrence graph
    method_fn   : callable(subgraph, **kwargs) → result
    method_kwargs: passed through to method_fn

    Returns
    -------
    dict mapping community_id -> result
    """
    partition = community_louvain.best_partition(G, weight='weight')

    from collections import defaultdict
    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    results = {}
    for cid, members in sorted(communities.items(),
                                key=lambda x: len(x[1]),
                                reverse=True):
        subgraph = G.subgraph(members).copy()
        if subgraph.number_of_nodes() < 5:
            continue
        print(f"  Community {cid}: {subgraph.number_of_nodes()} nodes, "
              f"{subgraph.number_of_edges()} edges")
        results[cid] = method_fn(subgraph, **method_kwargs)

    return results, partition
```

### 7.2  TDA Per Community

```python
import numpy as np
import ripser

def tda_on_subgraph(subgraph: nx.Graph, max_dim: int = 2,
                    max_nodes: int = 500) -> dict:
    """
    Run persistent homology on a community subgraph.

    Large communities are trimmed to the top-max_nodes entities
    by weighted degree before computing homology, preserving the
    densest (most information-rich) core of the community.
    """
    nodes = list(subgraph.nodes())

    if len(nodes) > max_nodes:
        degrees = dict(subgraph.degree(weight='weight'))
        nodes = sorted(nodes, key=lambda n: degrees[n],
                       reverse=True)[:max_nodes]
        subgraph = subgraph.subgraph(nodes).copy()
        nodes = list(subgraph.nodes())

    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for u, v, data in subgraph.edges(data=True):
        w = data.get('weight', 1.0)
        d = 1.0 / w if w > 0 else np.inf
        D[node_idx[u], node_idx[v]] = d
        D[node_idx[v], node_idx[u]] = d

    finite_max = D[D < np.inf].max() * 2
    D[D == np.inf] = finite_max

    diagrams = ripser.ripser(D, maxdim=max_dim,
                             distance_matrix=True)['dgms']

    threshold = 0.1 * (finite_max / 2)
    gaps = []
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            persistence = (death - birth) if death < np.inf else np.inf
            if persistence > threshold:
                gaps.append({
                    'dim': dim,
                    'birth': float(birth),
                    'death': float(death) if death < np.inf else None,
                    'persistence': float(persistence) if death < np.inf else None,
                })

    return {
        'diagrams': diagrams,
        'gaps': gaps,
        'nodes': nodes,
        'n_nodes': n,
    }

# Usage
tda_results, partition = community_scoped_analysis(G, tda_on_subgraph,
                                                   max_dim=2,
                                                   max_nodes=300)
```

### 7.3  Spectral Analysis Per Community

```python
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

def spectral_on_subgraph(subgraph: nx.Graph, k: int = 10) -> dict:
    """
    Compute Fiedler value + spectral clustering for a community subgraph.
    """
    nodes = list(subgraph.nodes())
    n = len(nodes)
    if n < k + 2:
        k = max(2, n - 2)

    node_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, d in subgraph.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = d.get('weight', 1.0)
        rows += [i, j]; cols += [j, i]; data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    L = csgraph.laplacian(A, normed=True)

    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM',
                                      tol=1e-6, maxiter=5000)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    fiedler_value = float(eigenvalues[1]) if n > 1 else 0.0
    fiedler_vector = eigenvectors[:, 1].tolist()

    return {
        'nodes': nodes,
        'fiedler_value': fiedler_value,
        'fiedler_vector': fiedler_vector,
        'eigenvalues': eigenvalues.tolist(),
        'spectral_gap': float(eigenvalues[2] - eigenvalues[1])
                        if n > 2 else 0.0,
    }

spectral_results, _ = community_scoped_analysis(G, spectral_on_subgraph, k=8)

# Communities with low Fiedler value are internally fragmented
fragmented = {cid: r['fiedler_value']
              for cid, r in spectral_results.items()
              if r['fiedler_value'] < 0.05}
print("Internally fragmented communities:", fragmented)
```

### 7.4  FCA Per Community

```python
from concepts import Context

def fca_on_subgraph(subgraph: nx.Graph,
                    entity_categories: dict,
                    all_categories: list) -> dict:
    """
    Build formal context and derive implications for entities in subgraph.
    """
    members = [n for n in subgraph.nodes()
               if n in entity_categories]
    if len(members) < 3:
        return {'implications': [], 'n_concepts': 0}

    booleans = [[cat in entity_categories[ent]
                 for cat in all_categories]
                for ent in members]

    ctx = Context(members, all_categories, booleans)
    lattice = ctx.lattice

    implications = []
    for impl in lattice.implications():
        implications.append({
            'premise': list(impl.premise),
            'conclusion': list(impl.conclusion),
        })

    return {
        'n_entities': len(members),
        'n_concepts': len(lattice),
        'implications': implications,
    }
```

### 7.5  Aggregating Cross-Community Results

```python
def summarise_tda_across_communities(tda_results: dict,
                                     partition: dict,
                                     nodes: list) -> list:
    """
    Collect all H2 knowledge-gap features across communities,
    sorted by persistence (most significant first).
    """
    all_gaps = []
    for cid, result in tda_results.items():
        for gap in result['gaps']:
            if gap['dim'] == 2:   # H2 = knowledge voids
                gap['community'] = cid
                gap['community_size'] = result['n_nodes']
                all_gaps.append(gap)

    all_gaps.sort(key=lambda g: g.get('persistence') or 0,
                  reverse=True)
    return all_gaps
```

---

## 8  Guide: Graph Embeddings

*Implementation of the pattern described in §4.8 of the Concept Document.*  
**Module**: `analysis/graph_embeddings.py` — `run(driver, year=None)`

### 8.1  Node2Vec (Shallow Structural Embeddings)

```python
from node2vec import Node2Vec

def train_node2vec(G: nx.Graph,
                   dimensions: int = 128,
                   walk_length: int = 30,
                   num_walks: int = 200,
                   p: float = 1.0,
                   q: float = 0.5,
                   workers: int = 4) -> dict:
    """
    Train Node2Vec embeddings on the co-occurrence graph.

    p=1, q=0.5 biases walks toward DFS (global structure).
    p=1, q=2   biases walks toward BFS (local neighbourhood).
    """
    G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    node2vec = Node2Vec(
        G_str,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p, q=q,
        weight_key='weight',
        workers=workers,
        quiet=False,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {node: model.wv[str(node)]
                  for node in G.nodes()}
    return embeddings, model

# After training, find similar entities:
# model.wv.most_similar('tokamak', topn=10)
```

**Practical notes**: With 50K nodes and 200 walks of length 30, Node2Vec generates ~10M training sentences — feasible on a single machine in 10–30 minutes.  Use `weight_key='weight'` so stronger co-occurrence edges are more likely to be traversed.

### 8.2  GraphSAGE (Message-Passing GNN)

```python
# Using PyTorch Geometric (PyG)
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class FusionGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # shape: (n_nodes, out_channels)

def build_pyg_data(G: nx.Graph,
                   node_features: dict,  # node -> np.array
                   ) -> Data:
    """Convert networkx graph to a PyG Data object."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    x = torch.tensor(
        np.stack([node_features[n] for n in nodes]),
        dtype=torch.float
    )
    edge_index = torch.tensor(
        [[node_idx[u], node_idx[v]]
         for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()
    edge_weight = torch.tensor(
        [G[u][v].get('weight', 1.0) for u, v in G.edges()],
        dtype=torch.float
    )
    return Data(x=x, edge_index=edge_index,
                edge_attr=edge_weight,
                num_nodes=len(nodes))
```

**When to use GNNs over Node2Vec**: When you have meaningful node features (category labels, citation counts, publication years) or want to train on a supervised signal.  Node2Vec is faster and unsupervised; GNNs are slower but richer.

### 8.3  Semantic + Structural Hybrid Embeddings

```python
from sentence_transformers import SentenceTransformer

def build_hybrid_embeddings(G: nx.Graph,
                             entity_names: list[str],
                             node2vec_embeddings: dict,
                             text_model_name: str = 'all-mpnet-base-v2',
                             ) -> np.ndarray:
    """
    Build hybrid embeddings by concatenating structural and semantic vectors.
    Returns array of shape (n_entities, structural_dim + text_dim).
    """
    model = SentenceTransformer(text_model_name)
    text_vecs = model.encode(entity_names, batch_size=512,
                             show_progress_bar=True,
                             normalize_embeddings=True)

    struct_vecs = np.stack([node2vec_embeddings[name]
                            for name in entity_names])
    struct_vecs /= np.linalg.norm(struct_vecs, axis=1, keepdims=True)

    hybrid = np.concatenate([text_vecs, struct_vecs], axis=1)
    return hybrid
```

### 8.4  Knowledge Map (UMAP Visualisation)

```python
import umap
import pandas as pd
import plotly.express as px

def build_knowledge_map(embeddings: np.ndarray,
                        entity_names: list[str],
                        categories: list[str],
                        communities: list[int],
                        pageranks: list[float]) -> px.Figure:
    """
    Project entity embeddings to 2D with UMAP and render an
    interactive scatter plot coloured by community.
    """
    reducer = umap.UMAP(n_components=2,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=42)
    coords = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'entity': entity_names,
        'category': categories,
        'community': [str(c) for c in communities],
        'pagerank': pageranks,
    })

    fig = px.scatter(
        df, x='x', y='y',
        color='community',
        size='pagerank',
        size_max=20,
        hover_data=['entity', 'category', 'pagerank'],
        title='Fusion Knowledge Graph — Entity Embedding Map',
        template='plotly_white',
    )
    fig.update_traces(marker=dict(opacity=0.7))
    return fig

# Export to interactive HTML
# fig.write_html('output/knowledge_map.html')
```

---

## 9  Guide: Gap Analysis Module Structure

*Code stubs for the Gap Analyser module described in §5.2 of the Concept Document.*

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
        ...

    def compute_persistence(self, max_dim=2):
        """Compute persistent homology via ripser.
        Returns persistence diagrams for H0, H1, H2."""
        ...

    def identify_voids(self, persistence_threshold=0.1):
        """Extract persistent H2 features
        and map back to entity names."""
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

## 10  Guide: Coupling and Cohesion Metrics

*Computes the coupling/cohesion metrics described in §6.2.4 of the Concept Document from the communities.csv output.*

```python
import pandas as pd, networkx as nx
from analysis.neo4j_utils import build_networkx_graph, fetch_co_occurrence_edges

comms = pd.read_csv("output/communities.csv")  # entity, community_id
node_to_comm = dict(zip(comms.entity, comms.community_id))

intra, total = 0, 0
for u, v, d in G.edges(data=True):
    w = d.get("weight", 1)
    total += w
    if node_to_comm.get(u) == node_to_comm.get(v):
        intra += w

modularity_Q = intra / total  # simplified; use community_louvain.modularity() for exact
```

---

## 11  Guide: From Analysis Outputs to OWL Modules

*Actionable pipeline described in §6.3 of the Concept Document.*

```
1. community_scoped.py output:
     community_spectral.csv        → flag communities with λ₂ < 0.05 for further splitting
     community_fca_implications.json → per-community implication base = module CQs

2. Manual review (OntologyExtender Streamlit UI or Protégé):
     For each community C_i with high cohesion:
       - Select dominant categories → becomes the module's class vocabulary
       - Convert top implications → OWL SubClassOf / ObjectSomeValuesFrom axioms
       - Identify bridge entities (structural_holes.csv) → inter-module ObjectProperty

3. graph_embeddings.py output (knowledge_map.html):
     Visual inspection of the UMAP map reveals entity clusters that
     don't align to any community → candidates for new sub-modules

4. Validation:
     - SHACL validation of each proposed module
     - Consistency check: OWL reasoner (HermiT/Pellet) on merged modules
     - Re-run coupling/cohesion metrics to verify decomposition quality
```

---

## 12  Guide: Fusion Domain Configuration for GraphQAAgent

*Domain configuration file described in §7.4 of the Concept Document.  Place at `config/domain_fusion.yaml` in the GraphQAAgent repository.*

```yaml
# config/domain_fusion.yaml
domain:
  name: Fusion Energy Research
  language: en

neo4j:
  entity_label: Entity
  paper_label: Paper
  relation_types:
    - CO_OCCURS_WITH
    - IN_CATEGORY
    - MENTIONS

cypher_templates:
  entity_neighbourhood: |
    MATCH (e:Entity {name: $name})-[:CO_OCCURS_WITH]-(n:Entity)
    RETURN n.name AS entity, n.category AS category
    ORDER BY n.mentions DESC LIMIT 20

entity_types:
  - Device Type
  - Plasma Property
  - Physical Process
  - Facility/Institution
  # ... (all 28 categories)

demo_questions:
  - "What connects tokamak and stellarator research?"
  - "Which plasma instabilities are most discussed in recent papers?"
  - "What are the key differences between ITER and W7-X?"
```

**CLI bridge** (`analysis/graph_qa.py`) — lightweight standalone Cypher interface without the full GraphQAAgent stack:

```bash
python -m analysis.graph_qa "What connects tokamak and stellarator?"
python -m analysis.graph_qa --mode bridge "plasma"
python -m analysis.graph_qa --mode trend "tritium"
```

---

## 13  Guide: Quick Start — Full Fusion Stack

*Deployment steps described in §9.4 of the Concept Document.*

```bash
# 1. Start the full GraphQAAgent stack
cd GraphQAAgent
cp .env.example .env        # set NEO4J_URI to your running instance
docker compose up -d --build
# API: http://localhost:8002/docs

# 2. Point it at the fusion KG Neo4j instance
#    (edit .env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 3. Configure fusion domain
cp config/domain.yaml config/domain_fusion.yaml
# Edit domain_fusion.yaml with fusion entity types and Cypher templates
# (see Guide 12 above)

# 4. (Optional) Load fusion OWL into Fuseki for ontology-expanded retrieval
#    Upload output/fusion_ontology.owl via Fuseki UI at http://localhost:3030

# 5. Ask the first question
curl -X POST http://localhost:8002/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What connects tokamak and stellarator research?",
       "strategy": "hybrid"}'
```
