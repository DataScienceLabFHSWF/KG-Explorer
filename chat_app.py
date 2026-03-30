"""
Fusion KG Chat — LLM-powered Streamlit Frontend
================================================
Pipeline per question:
  1. Semantic entity linking  — sentence-transformers finds real KG entity names
  2. Cypher generation        — LLM writes query using exact entity names
  3. Neo4j execution          — via LangChain GraphCypherQAChain
  4. Fallback                 — direct lookup when LLM Cypher returns 0 rows
  5. Answer synthesis         — LLM writes natural language answer from results

Run:  streamlit run chat_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── project root on sys.path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fusion KG Chat",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light CSS overrides ────────────────────────────────────────────────────
st.markdown("""
<style>
.cypher-block {
    background: #0d1117;
    color: #79c0ff;
    font-family: monospace;
    font-size: 0.85rem;
    padding: 10px 14px;
    border-radius: 6px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
.ctx-pill {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
}
.entity-pill {
    display: inline-block;
    background: #1a3a5c;
    color: #79c0ff;
    border-radius: 10px;
    padding: 1px 9px;
    font-size: 0.78rem;
    margin: 2px 3px 2px 0;
    font-family: monospace;
}
.entity-score {
    color: #58a6ff;
    font-size: 0.7rem;
    margin-left: 2px;
}
.fallback-badge {
    display: inline-block;
    background: #3d2b1a;
    color: #e3b341;
    border-radius: 10px;
    padding: 1px 9px;
    font-size: 0.75rem;
    margin-left: 6px;
}
.answer-text {
    font-size: 1.05rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar: config ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚛️ Fusion KG Chat")
    st.caption("LLM + Neo4j · Fusion energy literature · ~50k nodes · ~250k edges")
    st.divider()

    st.subheader("LLM / Ollama settings")
    ollama_url = st.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        key="ollama_url",
    )
    ollama_model = st.text_input(
        "Model",
        value=os.getenv("OLLAMA_MODEL", "nemotron:70b"),
        help="Must be pulled in Ollama first. Other options: mistral, llama3.1:8b, qwen2.5:7b",
        key="ollama_model",
    )

    qa_mode = st.selectbox(
        "QA mode",
        options=["graph_qa", "semantic_qa"],
        index=0,
        key="qa_mode",
        help="graph_qa: LLM generates Cypher and queries the KG. semantic_qa: embedding search over paper abstracts.",
    )

    if qa_mode == "semantic_qa":
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=os.getenv("QDRANT_URL", "http://localhost:6333"),
            key="qdrant_url",
        )
        qdrant_collection = st.text_input(
            "Collection",
            value=os.getenv("QDRANT_COLLECTION", "fusion_papers"),
            key="qdrant_collection",
        )

    with st.expander("Setup instructions", expanded=False):
        st.markdown("""
**1. Ollama runs locally** — make sure `ollama serve` is running.

**2. Pull the model** (once, if not already done)
```bash
ollama pull nemotron-3-nano:4b
```
**3. Reload this page** after the model is ready.

_Tip: nemotron-3-nano:4b (2.8 GB) is fast and compact._
""")

    st.divider()
    st.subheader("Example questions")

    EXAMPLES = [
        "What is a tokamak and how does it work?",
        "Which entities are most connected to stellarator?",
        "What are the top co-occurring concepts with plasma confinement?",
        "Find a path between divertor and tritium in the knowledge graph",
        "How has research on ITER evolved over the years?",
        "What materials are used as plasma-facing components?",
        "Which fusion devices appear most often in the literature?",
        "What are the main research gaps in plasma instability?",
    ]

    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q[:30]}"):
            st.session_state["pending_query"] = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── LLM agent (cached per Ollama URL + model) ─────────────────────────────

@st.cache_resource(
    show_spinner="Connecting to Neo4j, loading LLM and building entity index (first run ~30 s)…"
)
def _get_agent(ollama_url: str, model: str):
    """Build the FusionCypherAgent. Cached by (ollama_url, model)."""
    try:
        from analysis.llm_graph_qa import FusionCypherAgent
        agent = FusionCypherAgent(ollama_url=ollama_url, model=model)
        return agent, None
    except Exception as exc:
        return None, str(exc)


def get_agent():
    return _get_agent(
        st.session_state.get("ollama_url", "http://localhost:11434"),
        st.session_state.get("ollama_model", "nemotron:70b"),
    )


# ── Semantic QA agent (cached per Qdrant URL + collection + model) ─────────

@st.cache_resource(
    show_spinner="Loading embedding model and connecting to Qdrant (first run ~10 s)…"
)
def _get_semantic_agent(ollama_url: str, model: str, qdrant_url: str, collection: str):
    """Build the SemanticQAAgent. Cached by (ollama_url, model, qdrant_url, collection)."""
    try:
        from analysis.semantic_qa import SemanticQAAgent
        agent = SemanticQAAgent(
            ollama_url=ollama_url,
            model=model,
            qdrant_url=qdrant_url,
            collection=collection,
        )
        return agent, None
    except Exception as exc:
        return None, str(exc)


def get_semantic_agent():
    return _get_semantic_agent(
        st.session_state.get("ollama_url", "http://localhost:11434"),
        st.session_state.get("ollama_model", "nemotron:70b"),
        st.session_state.get("qdrant_url", "http://localhost:6333"),
        st.session_state.get("qdrant_collection", "fusion_papers"),
    )


# ── Conversation history for LLM context ──────────────────────────────────

def _history_for_llm() -> list[dict[str, str]]:
    """Return messages list in the format expected by FusionCypherAgent.ask()."""
    out: list[dict[str, str]] = []
    for m in st.session_state.get("messages", []):
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m.get("answer", "")})
    return out


# ── Message rendering ──────────────────────────────────────────────────────

def _extract_entities_from_result(
    context: list, linked: list
) -> list:
    """Collect entity name_norm strings from context rows and linked entities."""
    names: set = set()
    for name, _ in linked:
        names.add(name)
    for row in context:
        for key in ("entity", "neighbour", "method"):
            v = row.get(key)
            if isinstance(v, str) and v:
                names.add(v)
        path = row.get("path_nodes")
        if isinstance(path, list):
            names.update(n for n in path if isinstance(n, str))
    return list(names)


def _render_assistant_msg(msg: dict):
    """Render an assistant message dict to the screen."""
    answer = msg.get("answer", "")
    cypher = msg.get("cypher", "")
    context = msg.get("context", [])
    linked = msg.get("linked_entities", [])
    fallback = msg.get("fallback_used", False)
    error = msg.get("error")

    if error:
        st.error(f"**Error:** {error}")
        return

    # graph_qa: split into answer (left) + graph panel (right)
    graph_qa_mode = bool(cypher)
    if graph_qa_mode:
        col_answer, col_graph = st.columns([3, 2])
    else:
        col_answer = st
        col_graph = None

    with col_answer:
        # ── Semantic entity linking results
        if linked:
            pills_html = "".join(
                f'<span class="entity-pill">{name} '
                f'<span class="entity-score">{score:.2f}</span></span>'
                for name, score in linked
            )
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<span style="font-size:0.78rem;color:#8b949e">🔗 Linked entities: </span>'
                f'{pills_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Answer text
        st.markdown(f'<div class="answer-text">{answer}</div>', unsafe_allow_html=True)

        # ── Metadata pills row
        pills = []
        if context:
            pills.append(f"{len(context)} KG rows")
        if cypher:
            hop_hint = "multi-hop" if ("shortestPath" in cypher or "*.." in cypher) else "direct"
            pills.append(hop_hint)
        if fallback:
            pills.append("⚡ fallback query")

        if pills:
            st.markdown(
                " ".join(f'<span class="ctx-pill">{p}</span>' for p in pills),
                unsafe_allow_html=True,
            )

        # ── Collapsible Cypher section
        if cypher:
            with st.expander("Generated Cypher", expanded=False):
                st.markdown(
                    f'<div class="cypher-block">{cypher}</div>',
                    unsafe_allow_html=True,
                )

        # ── Collapsible raw results
        if context:
            with st.expander(f"Raw graph results ({len(context)} rows)", expanded=False):
                import pandas as pd
                try:
                    st.dataframe(
                        pd.DataFrame(context),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception:
                    st.json(context[:10])

    # ── Graph view panel (graph_qa only)
    if col_graph is not None:
        with col_graph:
            entity_names = _extract_entities_from_result(context, linked)
            if entity_names:
                with st.expander("🕸 Graph View", expanded=False):
                    try:
                        import streamlit.components.v1 as components
                        from analysis.graph_widget import build_context_graph_html
                        agent, err = get_agent()
                        if err:
                            st.warning(f"Graph unavailable: {err}")
                        else:
                            html = build_context_graph_html(
                                entity_names, agent.driver, agent.db
                            )
                            if html:
                                components.html(html, height=480, scrolling=False)
                            else:
                                st.caption("No edges found for these entities.")
                    except Exception as exc:
                        st.warning(f"Graph unavailable: {exc}")


# ── Session state ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Main chat area ─────────────────────────────────────────────────────────

st.header("Fusion Knowledge Graph — LLM Chat")
st.caption(
    "Ask anything in natural language. The LLM translates your question "
    "into a Cypher query, runs it on the knowledge graph, then explains the results."
)
st.divider()

# Display existing conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            _render_assistant_msg(msg)

# ── Input ──────────────────────────────────────────────────────────────────

# Support sidebar example button clicks
pending = st.session_state.pop("pending_query", None)

user_input = st.chat_input(
    "Ask anything about fusion research…",
    key="chat_input",
)

# Sidebar button overrides direct typing
query_text = (pending or user_input or "").strip()

if query_text:
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.write(query_text)

    # Run the appropriate agent
    active_mode = st.session_state.get("qa_mode", "graph_qa")

    with st.chat_message("assistant"):
        if active_mode == "semantic_qa":
            agent, init_error = get_semantic_agent()
            if init_error:
                st.error(
                    f"**Could not initialise Semantic QA.**\n\n"
                    f"Error: `{init_error}`\n\n"
                    "Make sure Qdrant is running and the collection has been imported."
                )
                msg = {"role": "assistant", "answer": "", "cypher": "", "context": [], "error": init_error}
            else:
                with st.spinner("Embedding query → Qdrant search → Answer synthesis…"):
                    llm_history = _history_for_llm()[:-1]
                    result = agent.ask(query_text, history=llm_history)
                _render_assistant_msg(result)
                msg = {"role": "assistant", **result}
        else:
            agent, init_error = get_agent()
            if init_error:
                st.error(
                    f"**Could not connect to Ollama / Neo4j.**\n\n"
                    f"Error: `{init_error}`\n\n"
                    "Make sure Neo4j and Ollama are running, then reload the page."
                )
                msg = {"role": "assistant", "answer": "", "cypher": "", "context": [], "error": init_error}
            else:
                with st.spinner("Entity linking → Cypher generation → Graph query → Answer…"):
                    llm_history = _history_for_llm()[:-1]
                    result = agent.ask(query_text, history=llm_history)
                _render_assistant_msg(result)
                msg = {"role": "assistant", **result}

    st.session_state["messages"].append(msg)
    st.rerun()

