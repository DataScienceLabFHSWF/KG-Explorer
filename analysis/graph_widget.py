"""
graph_widget.py
===============
Builds a focused pyvis subgraph HTML string for embedding in the chat UI.

Takes a list of entity names (seed nodes from a KG query result),
queries Neo4j for their 1-hop co-occurrence neighbourhood, and
returns a self-contained HTML string renderable via st.components.v1.html().
"""
from __future__ import annotations

# Category → colour mapping (kept in sync with interactive_explorer.py)
CATEGORY_COLORS: dict[str, str] = {
    "Concept":                        "#4CAF50",
    "Physical Process":               "#2196F3",
    "Physics Entity":                 "#FF9800",
    "Plasma dynamic and behavior":    "#E91E63",
    "Material":                       "#9C27B0",
    "Device":                         "#00BCD4",
    "Measurement":                    "#FF5722",
    "Technology":                     "#3F51B5",
    "Chemical Element":               "#CDDC39",
    "Model":                          "#795548",
}
DEFAULT_COLOR = "#90A4AE"


def _node_color(categories: set) -> str:
    for cat in categories:
        if cat in CATEGORY_COLORS:
            return CATEGORY_COLORS[cat]
    return DEFAULT_COLOR


def build_context_graph_html(
    entity_names: list[str],
    driver,
    db: str = "neo4j",
    max_neighbour_nodes: int = 40,
) -> str:
    """Return a self-contained pyvis HTML string for the given seed entities.

    Seed nodes (directly from the query result) get a gold border.
    Neighbour nodes are added up to *max_neighbour_nodes* by weighted degree.
    Returns an empty string when entity_names is empty or pyvis is unavailable.
    """
    if not entity_names:
        return ""

    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    seed_set = set(entity_names)

    # ── 1. Fetch co-occurrence edges for seeds + 1-hop neighbours ──────────
    with driver.session(database=db) as sess:
        rows = [
            dict(r)
            for r in sess.run(
                """
                MATCH (e:Entity)
                WHERE e.name_norm IN $names
                OPTIONAL MATCH (e)-[r:CO_OCCURS_WITH]-(other:Entity)
                WHERE r.weight >= 2
                RETURN e.name_norm AS src, other.name_norm AS tgt, r.weight AS weight
                """,
                names=list(seed_set),
            )
        ]

    all_nodes: set[str] = set(seed_set)
    edge_list: list[tuple[str, str, int]] = []

    for row in rows:
        src, tgt, w = row["src"], row["tgt"], row["weight"]
        if tgt is None:
            continue
        all_nodes.add(tgt)
        edge_list.append((src, tgt, int(w or 1)))

    # Trim neighbours to cap by weighted degree
    if len(all_nodes) > len(seed_set) + max_neighbour_nodes:
        wdeg: dict[str, int] = {}
        for s, t, w in edge_list:
            wdeg[s] = wdeg.get(s, 0) + w
            wdeg[t] = wdeg.get(t, 0) + w
        keep = set(seed_set)
        neighbours = sorted(
            [n for n in all_nodes if n not in keep],
            key=lambda n: wdeg.get(n, 0),
            reverse=True,
        )
        keep.update(neighbours[:max_neighbour_nodes])
        all_nodes = keep
        edge_list = [(s, t, w) for s, t, w in edge_list if s in keep and t in keep]

    # ── 1b. Trim to top-10 nodes for display (seeds prioritised) ──────────
    MAX_DISPLAY_NODES = 10
    display_wdeg: dict[str, int] = {}
    for s, t, w in edge_list:
        display_wdeg[s] = display_wdeg.get(s, 0) + w
        display_wdeg[t] = display_wdeg.get(t, 0) + w

    # Sort: seeds first (boosted), then by weighted degree descending
    ranked = sorted(
        all_nodes,
        key=lambda n: (n in seed_set, display_wdeg.get(n, 0)),
        reverse=True,
    )
    all_nodes = set(ranked[:MAX_DISPLAY_NODES])
    edge_list = [(s, t, w) for s, t, w in edge_list if s in all_nodes and t in all_nodes]

    # ── 2. Fetch categories for node colouring ─────────────────────────────
    entity_cats: dict[str, set[str]] = {}
    with driver.session(database=db) as sess:
        for row in sess.run(
            """
            MATCH (e:Entity)-[:IN_CATEGORY]->(c:Category)
            WHERE e.name_norm IN $names
            RETURN e.name_norm AS entity, c.name AS category
            """,
            names=list(all_nodes),
        ):
            entity_cats.setdefault(row["entity"], set()).add(row["category"])

    # ── 3. Build pyvis network ─────────────────────────────────────────────
    node_list = list(all_nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}

    wdeg = {}
    for s, t, w in edge_list:
        wdeg[s] = wdeg.get(s, 0) + w
        wdeg[t] = wdeg.get(t, 0) + w
    max_wdeg = max(wdeg.values()) if wdeg else 1

    net = Network(
        height="460px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        notebook=False,
        cdn_resources="in_line",
    )

    for name in node_list:
        cats = entity_cats.get(name, set())
        deg = wdeg.get(name, 1)
        is_seed = name in seed_set
        cat_str = ", ".join(sorted(cats)) if cats else "uncategorized"
        tooltip = (
            f"<b>{name}</b><br>"
            f"Categories: {cat_str}<br>"
            f"Weighted degree: {deg:.0f}"
            + ("<br><b>\u2605 Query entity</b>" if is_seed else "")
        )
        net.add_node(
            node_idx[name],
            label=name if (deg >= max_wdeg * 0.05 or is_seed) else "",
            title=tooltip,
            size=10 + 25 * (deg / max_wdeg),
            color={"background": _node_color(cats), "border": "#FFD700" if is_seed else "#444"},
            borderWidth=3 if is_seed else 1,
            font={"size": 11, "color": "white"},
        )

    max_w = max((w for _, _, w in edge_list), default=1)
    for s, t, w in edge_list:
        net.add_edge(
            node_idx[s], node_idx[t],
            value=w,
            width=0.5 + 3.0 * (w / max_w),
            color={"color": "rgba(150,150,150,0.3)"},
        )

    net.set_options("""{
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.02,
                "springLength": 100,
                "springConstant": 0.03,
                "damping": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 120}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true
        }
    }""")

    return net.generate_html()
