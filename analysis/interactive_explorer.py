"""
Interactive Graph Explorer
===========================
Generates an interactive HTML visualisation of the fusion KG
using pyvis, with gap highlighting from other analysis modules.

Features
--------
  - Nodes coloured by category
  - Node size by PageRank / degree
  - Edge thickness by co-occurrence weight
  - Highlighted structural holes (red border)
  - Highlighted predicted missing links (dashed red)
  - Community colouring
  - Search & filter in the browser

Outputs
-------
  - output/interactive_graph.html     — full interactive explorer
  - output/interactive_gaps.html      — focused on detected gaps
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.neo4j_utils import (
    fetch_co_occurrence_edges,
    fetch_entity_categories,
    get_driver,
    OUTPUT_DIR,
)

MAX_NODES = 500  # browser can handle ~500 nodes smoothly


def _select_subgraph(nodes, edges, max_nodes=MAX_NODES):
    """Top-N by weighted degree."""
    n = len(nodes)
    if n <= max_nodes:
        return nodes, edges

    deg = np.zeros(n)
    for si, ti, w in edges:
        deg[si] += w
        deg[ti] += w

    top_indices = set(np.argsort(deg)[-max_nodes:])
    old_to_new = {}
    new_nodes = []
    for old_idx in sorted(top_indices):
        old_to_new[old_idx] = len(new_nodes)
        new_nodes.append(nodes[old_idx])

    new_edges = []
    for si, ti, w in edges:
        if si in old_to_new and ti in old_to_new:
            new_edges.append((old_to_new[si], old_to_new[ti], w))

    return new_nodes, new_edges


# Category → colour mapping
CATEGORY_COLORS = {
    "Concept": "#4CAF50",
    "Physical Process": "#2196F3",
    "Physics Entity": "#FF9800",
    "Plasma dynamic and behavior": "#E91E63",
    "Material": "#9C27B0",
    "Device": "#00BCD4",
    "Measurement": "#FF5722",
    "Technology": "#3F51B5",
    "Chemical Element": "#CDDC39",
    "Model": "#795548",
}
DEFAULT_COLOR = "#90A4AE"


def _get_node_color(categories):
    """Pick colour based on first matching category."""
    for cat in categories:
        if cat in CATEGORY_COLORS:
            return CATEGORY_COLORS[cat]
    return DEFAULT_COLOR


def build_interactive_graph(
    nodes, edges, entity_cats,
    structural_holes_df=None,
    predicted_links_df=None,
    communities_df=None,
    title="Fusion KG Explorer",
    filename="interactive_graph.html",
):
    """Build a pyvis interactive network."""
    from pyvis.network import Network

    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        notebook=False,
        cdn_resources="in_line",
    )
    net.heading = title

    # Load structural holes if available
    bridge_entities = set()
    if structural_holes_df is not None:
        bridge_entities = set(structural_holes_df.head(30)["entity"])

    # Load communities if available
    community_map = {}
    if communities_df is not None:
        for _, row in communities_df.iterrows():
            community_map[row["entity"]] = row["community"]

    # Compute weighted degree for sizing
    n = len(nodes)
    wdeg = np.zeros(n)
    for si, ti, w in edges:
        wdeg[si] += w
        wdeg[ti] += w
    max_wdeg = wdeg.max() if len(wdeg) > 0 else 1

    # Add nodes
    for i, name in enumerate(nodes):
        cats = entity_cats.get(name, set())
        color = _get_node_color(cats)
        size = 8 + 30 * (wdeg[i] / max_wdeg)

        border_color = "red" if name in bridge_entities else "#222"
        border_width = 3 if name in bridge_entities else 1

        cat_str = ", ".join(sorted(cats)) if cats else "uncategorized"
        comm = community_map.get(name, "?")
        tooltip = (
            f"<b>{name}</b><br>"
            f"Categories: {cat_str}<br>"
            f"Weighted degree: {wdeg[i]:.0f}<br>"
            f"Community: {comm}"
        )
        if name in bridge_entities:
            tooltip += "<br><b>⚠ Structural hole spanner</b>"

        net.add_node(
            i, label=name if wdeg[i] > max_wdeg * 0.05 else "",
            title=tooltip,
            size=size, color=color,
            borderWidth=border_width,
            borderWidthSelected=4,
            font={"size": 10, "color": "white"},
        )

    # Add existing edges
    max_w = max(w for _, _, w in edges) if edges else 1
    for si, ti, w in edges:
        width = 0.5 + 3.0 * (w / max_w)
        net.add_edge(si, ti, value=w, width=width,
                     color={"color": "rgba(150,150,150,0.3)"})

    # Add predicted missing links
    if predicted_links_df is not None:
        node_idx = {name: i for i, name in enumerate(nodes)}
        for _, row in predicted_links_df.head(30).iterrows():
            a_idx = node_idx.get(row["entity_a"])
            b_idx = node_idx.get(row["entity_b"])
            if a_idx is not None and b_idx is not None:
                net.add_edge(
                    a_idx, b_idx,
                    dashes=True, color={"color": "rgba(255,50,50,0.7)"},
                    width=2, title=f"Predicted: AA={row['adamic_adar']:.2f}",
                )

    # Physics settings
    net.set_options("""{
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 120,
                "springConstant": 0.02,
                "damping": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }""")

    out_path = str(OUTPUT_DIR / filename)
    # pyvis save_graph uses system encoding; force UTF-8 on Windows
    html = net.generate_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(driver=None, year: int | None = None):
    """Generate interactive graph explorer.

    Parameters
    ----------
    driver
        Neo4j driver (optional).
    year
        Optional year filter for the graph.
    """
    print("\n" + "=" * 60)
    print("  INTERACTIVE GRAPH EXPLORER")
    print("=" * 60)

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        nodes, idx, edges = fetch_co_occurrence_edges(driver, year=year)
        entity_cats, _ = fetch_entity_categories(driver)
        print(f"  Full graph: {len(nodes)} nodes, {len(edges)} edges")
        if year is not None:
            print(f"  (filtered to papers published in {year})")

        sub_nodes, sub_edges = _select_subgraph(nodes, edges, MAX_NODES)
        print(f"  Subgraph for explorer: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

        # Load auxiliary data if available
        structural_holes_df = None
        sh_path = OUTPUT_DIR / "structural_holes.csv"
        if sh_path.exists():
            structural_holes_df = pd.read_csv(sh_path)
            print(f"  Loaded structural holes data ({len(structural_holes_df)} entities)")

        predicted_links_df = None
        pl_path = OUTPUT_DIR / "predicted_links.csv"
        if pl_path.exists():
            predicted_links_df = pd.read_csv(pl_path)
            print(f"  Loaded link predictions ({len(predicted_links_df)} predictions)")

        communities_df = None
        comm_path = OUTPUT_DIR / "communities.csv"
        if comm_path.exists():
            communities_df = pd.read_csv(comm_path)
            print(f"  Loaded community assignments")

        # Full explorer
        print("\n--- Building full interactive explorer ---")
        build_interactive_graph(
            sub_nodes, sub_edges, entity_cats,
            structural_holes_df=structural_holes_df,
            predicted_links_df=predicted_links_df,
            communities_df=communities_df,
            title="Fusion Knowledge Graph — Interactive Explorer",
            filename="interactive_graph.html",
        )

        # Gap-focused explorer (smaller, centred on gaps)
        print("\n--- Building gap-focused explorer ---")
        # Select entities involved in gaps
        gap_entities = set()
        if structural_holes_df is not None:
            gap_entities.update(structural_holes_df.head(50)["entity"])
        if predicted_links_df is not None:
            gap_entities.update(predicted_links_df.head(30)["entity_a"])
            gap_entities.update(predicted_links_df.head(30)["entity_b"])

        gaps_path = OUTPUT_DIR / "knowledge_gaps.json"
        if gaps_path.exists():
            with open(gaps_path) as f:
                gaps = json.load(f)
            for void in gaps.get("h2_voids", [])[:20]:
                gap_entities.update(void.get("entities", []))

        if gap_entities:
            # Build subgraph of just gap entities + their neighbours
            node_idx = {n: i for i, n in enumerate(sub_nodes)}
            gap_indices = {node_idx[e] for e in gap_entities if e in node_idx}

            gap_nodes_set = set(gap_indices)
            gap_edges = []
            for si, ti, w in sub_edges:
                if si in gap_nodes_set or ti in gap_nodes_set:
                    gap_nodes_set.add(si)
                    gap_nodes_set.add(ti)

            # Limit to 300 nodes
            if len(gap_nodes_set) > 300:
                # Keep only direct gap entities and their edges
                gap_nodes_set = gap_indices.copy()

            old_to_new = {}
            gap_nodes = []
            for old_idx in sorted(gap_nodes_set):
                old_to_new[old_idx] = len(gap_nodes)
                gap_nodes.append(sub_nodes[old_idx])

            gap_edges = []
            for si, ti, w in sub_edges:
                if si in old_to_new and ti in old_to_new:
                    gap_edges.append((old_to_new[si], old_to_new[ti], w))

            build_interactive_graph(
                gap_nodes, gap_edges, entity_cats,
                structural_holes_df=structural_holes_df,
                predicted_links_df=predicted_links_df,
                communities_df=communities_df,
                title="Fusion KG — Knowledge Gaps Focus",
                filename="interactive_gaps.html",
            )
        else:
            print("  No gap data available yet. Run void_extraction and structural_holes first.")

        return {"explorer_path": str(OUTPUT_DIR / "interactive_graph.html")}

    finally:
        if own_driver:
            driver.close()


if __name__ == "__main__":
    run()
