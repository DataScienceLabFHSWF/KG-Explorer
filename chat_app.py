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
    st.subheader("Agent mode")
    agent_mode = st.radio(
        "Query mode",
        ["Standard", "ReAct"],
        index=0,
        key="agent_mode",
        help=(
            "**Standard** — single-pass Cypher + GraphRAG (fast).\n\n"
            "**ReAct** — iterative Thought→Action→Observation loop; "
            "uses KG search, entity lookup, OpenAlex, and ArXiv until "
            "coverage is sufficient."
        ),
    )
    react_max_steps = st.slider("ReAct max steps", 2, 10, 5, key="react_steps",
                                disabled=(agent_mode == "Standard"))
    react_threshold = st.slider("Coverage threshold", 0.3, 1.0, 0.75, 0.05,
                                key="react_threshold",
                                disabled=(agent_mode == "Standard"))

    st.divider()
    st.subheader("Research swarm")
    swarm_on_low = st.checkbox(
        "Auto-trigger swarm on low coverage",
        value=False,
        key="swarm_on_low",
        help="When coverage < threshold, launch parallel OpenAlex + ArXiv agents.",
    )
    swarm_threshold = st.slider("Swarm trigger threshold", 0.1, 0.7, 0.35, 0.05,
                                key="swarm_threshold",
                                disabled=not swarm_on_low)

    st.divider()
    st.subheader("Memory")
    memory_on = st.checkbox(
        "Enable long-term memory",
        value=False,
        key="memory_on",
        help="Store Q/A turns in Neo4j and recall relevant past answers.",
    )
    show_memory_recall = st.checkbox(
        "Show recalled memories",
        value=True,
        key="show_memory_recall",
        disabled=not memory_on,
    )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Memory graph (cached singleton) ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_memory_graph():
    try:
        from analysis.memory_graph import MemoryGraph
        return MemoryGraph(), None
    except Exception as exc:
        return None, str(exc)


def _ensure_memory_session() -> str | None:
    """Return (or create) the memory session ID for this browser session."""
    if not st.session_state.get("memory_on", False):
        return None
    mem, err = _get_memory_graph()
    if mem is None:
        return None
    if "memory_session_id" not in st.session_state:
        st.session_state["memory_session_id"] = mem.new_session(topic="fusion-kg-chat")
    return st.session_state["memory_session_id"]


# ── ReAct agent (cached per config) ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_react_agent(ollama_url: str, model: str, max_steps: int, threshold: float):
    try:
        from analysis.llm_graph_qa import FusionCypherAgent
        from analysis.react_agent import ReActAgent
        from daq.arxiv_client import ArXivClient
        from daq.openalex_client import OpenAlexClient
        kg_agent = FusionCypherAgent(ollama_url=ollama_url, model=model)
        arxiv = ArXivClient()
        oa = OpenAlexClient()
        return ReActAgent(
            kg_agent=kg_agent,
            openalex_client=oa,
            arxiv_client=arxiv,
            max_steps=max_steps,
            coverage_threshold=threshold,
        ), None
    except Exception as exc:
        return None, str(exc)


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

tab_chat, tab_gaps, tab_memory = st.tabs(["Chat", "Answer-Gap Report", "Memory Graph"])

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
                # ── Memory recall ──────────────────────────────────────────
                mem_session_id = _ensure_memory_session()
                if mem_session_id and st.session_state.get("show_memory_recall", True):
                    mem_graph, _ = _get_memory_graph()
                    if mem_graph:
                        past = mem_graph.recall(query_text, top_k=3)
                        if past:
                            with st.expander(f"🧠 Recalled {len(past)} past memory/memories", expanded=False):
                                for pm in past:
                                    ts = str(pm.get("timestamp", ""))[:19]
                                    st.markdown(
                                        f"**[{ts}]** {pm.get('question','?')}\n\n"
                                        f"> {str(pm.get('answer',''))[:200]}…"
                                    )

                # ── Route to ReAct or Standard agent ──────────────────────
                mode = st.session_state.get("agent_mode", "Standard")

                if mode == "ReAct":
                    react_agent, react_err = _get_react_agent(
                        st.session_state.get("ollama_url", "http://localhost:11434"),
                        st.session_state.get("ollama_model", "gemma4:e2b"),
                        st.session_state.get("react_steps", 5),
                        st.session_state.get("react_threshold", 0.75),
                    )
                    if react_err or react_agent is None:
                        st.warning(f"ReAct agent unavailable: {react_err}. Falling back to standard.")
                        mode = "Standard"
                    else:
                        with st.spinner("ReAct: iterative reasoning → KG + literature search…"):
                            react_result = react_agent.run(query_text)
                        answer = react_result.answer
                        coverage = react_result.coverage_score
                        stopped = react_result.stopped_reason
                        steps = react_result.steps

                        st.markdown(f'<div class="answer-text">{answer}</div>', unsafe_allow_html=True)
                        pills = [
                            f"ReAct • {len(steps)} steps",
                            f"coverage {coverage:.0%}",
                            stopped.replace('_', ' '),
                        ]
                        st.markdown(
                            " ".join(f'<span class="ctx-pill">{p}</span>' for p in pills),
                            unsafe_allow_html=True,
                        )
                        with st.expander("ReAct reasoning trace", expanded=False):
                            for s in steps:
                                st.markdown(f"**Step {s.step} · {s.tool}**")
                                if s.thought:
                                    st.markdown(f"*Thought:* {s.thought[:300]}")
                                st.markdown(f"*Observation:* {s.observation[:300]}")
                                st.divider()

                        result = {"answer": answer, "cypher": "", "context": [],
                                  "linked_entities": [], "fallback_used": False,
                                  "coverage_score": coverage}
                        msg = {"role": "assistant", **result}

                if mode == "Standard":
                    with st.spinner("Entity linking → Cypher generation → Graph query → Answer…"):
                        llm_history = _history_for_llm()[:-1]
                        result = agent.ask(query_text, history=llm_history)
                    _render_assistant_msg(result)
                    msg = {"role": "assistant", **result}

                # ── Research swarm on low coverage ─────────────────────────
                coverage = result.get("coverage_score", 1.0) if isinstance(result, dict) else 0.5
                swarm_threshold = st.session_state.get("swarm_threshold", 0.35)
                if (
                    st.session_state.get("swarm_on_low", False)
                    and isinstance(coverage, float)
                    and coverage < swarm_threshold
                ):
                    with st.spinner(f"🔬 Coverage low ({coverage:.0%}) — launching research swarm…"):
                        try:
                            from daq.research_swarm import maybe_trigger_swarm
                            swarm_papers = maybe_trigger_swarm(
                                question=query_text,
                                coverage_score=coverage,
                                coverage_threshold=swarm_threshold,
                                n_agents=3,
                                papers_per_agent=5,
                            )
                            if swarm_papers:
                                with st.expander(
                                    f"🔬 Research swarm found {len(swarm_papers)} candidate papers",
                                    expanded=True,
                                ):
                                    for sp in swarm_papers[:8]:
                                        oa_tag = "✅ OA" if sp.is_oa or sp.pdf_url else "🔒"
                                        st.markdown(
                                            f"**{sp.title}** &nbsp; {oa_tag} &nbsp; "
                                            f"[{sp.source}] &nbsp; cited:{sp.citations} &nbsp; "
                                            f"{sp.published[:7]}"
                                        )
                                        if sp.abstract:
                                            st.caption(sp.abstract[:200] + "…")
                        except Exception as swarm_exc:
                            st.warning(f"Swarm error: {swarm_exc}")

                # ── Store turn in memory graph ─────────────────────────────
                if mem_session_id:
                    mem_graph, _ = _get_memory_graph()
                    if mem_graph and isinstance(result, dict) and result.get("answer"):
                        linked = [name for name, _ in (result.get("linked_entities") or [])]
                        try:
                            from analysis.memory_graph import extract_facts_from_answer
                            facts = extract_facts_from_answer(
                                agent.llm, query_text, result["answer"]
                            )
                        except Exception:
                            facts = []
                        mem_graph.remember(
                            session_id=mem_session_id,
                            question=query_text,
                            answer=result["answer"],
                            entities=linked,
                            coverage_score=float(result.get("coverage_score", 0)),
                            facts=facts,
                        )

        st.session_state["messages"].append(msg)
        st.rerun()


with tab_gaps:
    _render_gap_report_tab()


with tab_memory:
    st.subheader("Long-term Memory Graph")
    st.caption(
        "Each Q/A turn is stored in Neo4j as a Memory node linked to the KG entities "
        "it mentions. Enable memory in the sidebar to start recording."
    )
    mem_graph, mem_err = _get_memory_graph()
    if mem_err:
        st.warning(f"Memory graph unavailable: {mem_err}")
    elif not st.session_state.get("memory_on", False):
        st.info("Enable **Long-term memory** in the sidebar to start recording and recalling.")
    else:
        stats = mem_graph.stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", stats.get("sessions", 0))
        c2.metric("Memories", stats.get("memories", 0))
        c3.metric("Facts", stats.get("facts", 0))
        st.divider()

        recall_q = st.text_input("Recall memories about…", key="memory_recall_q")
        if recall_q:
            rows = mem_graph.recall(recall_q, top_k=10)
            if not rows:
                st.info("No memories found.")
            for r in rows:
                ts = str(r.get("timestamp", ""))[:19]
                cov = r.get("coverage_score", 0)
                st.markdown(f"**[{ts}]** *{r.get('question','?')}*")
                st.markdown(f"> {str(r.get('answer',''))[:300]}")
                st.caption(f"coverage={cov:.2f}  session={r.get('session_id','?')[:8]}")
                st.divider()

        if st.button("Export memories to JSON", key="export_mem"):
            export_path = Path("output/memory_export.json")
            try:
                mem_graph.export(export_path)
                st.success(f"Exported → {export_path}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

        st.divider()
        st.subheader("Research Swarm")
        st.caption("Manually run the research swarm to find papers for the top gap entities.")
        top_gaps = st.slider("Top gap entities", 3, 20, 8, key="swarm_top_gaps")
        n_agents = st.slider("Concurrent agents", 1, 8, 4, key="swarm_n_agents")
        if st.button("🔬 Run research swarm from gap report", key="run_swarm"):
            try:
                from daq.research_swarm import ResearchSwarm
                with st.spinner("Swarm running — searching OpenAlex + ArXiv in parallel…"):
                    swarm = ResearchSwarm.from_gap_report(top_gaps=top_gaps, n_agents=n_agents)
                    papers = swarm.run()
                    swarm.save_results()
                summary = swarm.summary()
                st.success(
                    f"Found {summary['total']} papers "
                    f"({summary['open_access']} OA, "
                    f"OA={summary['by_source']['openalex']}, "
                    f"ArXiv={summary['by_source']['arxiv']})"
                )
                import pandas as _pd
                rows_sw = [
                    {
                        "Title": p.title[:70],
                        "Source": p.source,
                        "OA": "✅" if p.is_oa or p.pdf_url else "🔒",
                        "Cited": p.citations,
                        "Date": p.published[:7],
                        "Query": p.query[:30],
                    }
                    for p in papers[:30]
                ]
                st.dataframe(_pd.DataFrame(rows_sw), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Swarm error: {exc}")

