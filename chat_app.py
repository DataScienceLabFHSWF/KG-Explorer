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
        value=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        help=(
            "Must be pulled in Ollama first. "
            "Default: gemma4:e2b (compact, good Cypher). "
            "Alternatives: qwen2.5:7b, llama3.1:8b. "
            "Avoid nemotron-3-nano:4b — tends to hallucinate SQL."
        ),
        key="ollama_model",
    )

    with st.expander("Setup instructions", expanded=False):
        st.markdown("""
**1. Ollama runs locally** — make sure `ollama serve` is running.

**2. Pull the model** (once, if not already done)
```bash
ollama pull gemma4:e2b
```
**3. Reload this page** after the model is ready.

_Tip: gemma4:e2b is the recommended default. For higher quality on
long/complex questions, switch to qwen2.5:7b or llama3.1:8b._
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
        st.session_state.get("ollama_model", "gemma4:e2b"),
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

def _render_assistant_msg(msg: dict):
    """Render an assistant message dict to the screen."""
    answer = msg.get("answer", "")
    cypher = msg.get("cypher", "")
    context = msg.get("context", [])
    abstracts = msg.get("abstracts", [])
    linked = msg.get("linked_entities", [])
    fallback = msg.get("fallback_used", False)
    missing = msg.get("missing_from_graph", False)
    error = msg.get("error")

    if error:
        st.error(f"**Error:** {error}")
        return

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
    if abstracts:
        pills.append(f"{len(abstracts)} abstracts")
    if cypher:
        hop_hint = "multi-hop" if ("shortestPath" in cypher or "*.." in cypher) else "direct"
        pills.append(hop_hint)
    if fallback:
        pills.append("⚡ fallback query")
    if missing:
        pills.append("⚠ missing-from-graph")

    if pills:
        st.markdown(
            " ".join(f'<span class="ctx-pill">{p}</span>' for p in pills),
            unsafe_allow_html=True,
        )

    # ── Abstract excerpts (true GraphRAG evidence)
    if abstracts:
        with st.expander(f"Paper abstract excerpts ({len(abstracts)} sources)", expanded=False):
            for i, doc in enumerate(abstracts, 1):
                year = f" ({doc.get('year', '?')})"
                title = doc.get('title', '(untitled)')
                src = doc.get('source', '')
                st.markdown(f"**[{i}] {title}**{year}  &nbsp;·&nbsp; *{src}*")
                excerpt = (doc.get('excerpt') or '').strip()
                st.markdown(f"<div style='font-size:0.88rem;color:#bbb;margin-bottom:10px'>{excerpt}</div>", unsafe_allow_html=True)

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


# ── Session state ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ── Gap-report tab renderer ────────────────────────────────────────────────

def _render_gap_report_tab():
    """Render the Answer-Gap Report tab from the live JSONL log."""
    import pandas as _pd
    from analysis.answer_gap_logger import LOG_PATH
    from analysis.answer_gap_report import build_report, _load_events

    st.subheader("Answer-Gap Report")
    st.caption(
        "Questions the chat agent could not answer from the knowledge graph. "
        "Refreshed each time you view this tab."
    )

    col_refresh, _col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh report"):
            st.rerun()

    events = _load_events()
    if not events:
        st.info("No answer-gap events logged yet. Ask the chat agent some questions first.")
        return

    report = build_report(events)

    total = report["total_failed_questions"]
    modes = report["by_failure_mode"]
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total gap events", total)
    with cols[1]:
        st.metric("Sentinel (missing)", modes.get("sentinel_only", 0))
    with cols[2]:
        st.metric("Low coverage", modes.get("low_coverage", 0))
    with cols[3]:
        st.metric("Unlinked", modes.get("unlinked", 0))

    st.divider()

    clusters = report.get("question_clusters") or []
    if clusters:
        st.markdown("### Question clusters — top ingestion targets")
        st.caption("Each cluster is one conceptual blind spot.")
        rows = []
        for i, cl in enumerate(clusters[:15], 1):
            ents = ", ".join(cl["entities"]) or "(unlinked)"
            qs = " | ".join(cl["questions"][:2])
            cov = f"{cl['avg_coverage_score']:.2f}" if cl["avg_coverage_score"] is not None else "—"
            rows.append({"#": i, "Entities": ents, "Count": cl["count"],
                         "Avg coverage": cov, "Questions": qs})
        st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.divider()

    top_ents = report.get("top_entities_in_failures") or []
    if top_ents:
        st.markdown("### Repeat-offender entities")
        df = _pd.DataFrame(top_ents, columns=["Entity", "Gap events"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    unlinked = report.get("unlinked_question_samples") or []
    if unlinked:
        st.markdown("### Questions where no entity was linked")
        for q in unlinked:
            st.markdown(f"- {q}")
        st.divider()

    with st.expander(f"Raw gap log ({total} events)", expanded=False):
        for ev in events[:50]:
            st.json(ev)


# ── Main chat area ─────────────────────────────────────────────────────────

st.header("Fusion Knowledge Graph — LLM Chat")
st.caption(
    "Ask anything in natural language. The LLM translates your question "
    "into a Cypher query, runs it on the knowledge graph, then explains the results."
)
st.divider()

tab_chat, tab_gaps = st.tabs(["Chat", "Answer-Gap Report"])

with tab_chat:
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

        # Run LLM agent
        agent, init_error = get_agent()

        with st.chat_message("assistant"):
            if init_error:
                st.error(
                    f"**Could not connect to Ollama / Neo4j.**\n\n"
                    f"Error: `{init_error}`\n\n"
                    "Make sure Neo4j and Ollama are running, then reload the page."
                )
                msg = {"role": "assistant", "answer": "", "cypher": "", "context": [], "error": init_error}
            else:
                with st.spinner("Entity linking → Cypher generation → Graph query → Answer…"):
                    llm_history = _history_for_llm()[:-1]  # exclude the just-added user message
                    result = agent.ask(query_text, history=llm_history)

                _render_assistant_msg(result)
                msg = {"role": "assistant", **result}

        st.session_state["messages"].append(msg)
        st.rerun()


with tab_gaps:
    _render_gap_report_tab()

