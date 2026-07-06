"""ReAct Agent for the Fusion Knowledge Graph
=============================================
Implements the ReAct (Reasoning + Acting) pattern:
  Thought → Action → Observation → Thought → ...

The agent stops when it decides "I have enough information to answer" or
when it reaches ``max_steps``.

Available tools
---------------
search_kg           Full-text search across entity names and abstracts in Neo4j
entity_lookup       Fetch all neighbours, relationships, and properties of an entity
cypher_query        Run an arbitrary read-only Cypher query
get_graph_context   Multi-hop subgraph around an entity (configurable depth)
openalex_search     Search OpenAlex for recent papers about a topic
arxiv_search        Search ArXiv for pre-prints about a topic

Stopping criterion
------------------
After each observation the agent produces a ``coverage_score`` (0–1).  When
``coverage_score >= coverage_threshold`` (default 0.75) it produces a final
answer.  It also stops if the LLM emits a ``[FINAL ANSWER]`` sentinel or when
``max_steps`` is reached.

Usage
-----
    from analysis.react_agent import ReActAgent
    from analysis.llm_graph_qa import FusionCypherAgent

    kg_agent = FusionCypherAgent()
    react = ReActAgent(kg_agent=kg_agent)
    result = react.run("How does plasma confinement work in a stellarator?")
    print(result["answer"])
    print(result["trace"])          # list of (thought, action, observation) dicts

CLI
---
    python -m analysis.react_agent "How does plasma confinement work?"
    python -m analysis.react_agent "What are the advantages of ITER?" --max-steps 8
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
_OUTPUT_DIR = Path("output")

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS_DOC = """
Available tools (use EXACTLY one per step):

  search_kg(query: str)
      Full-text search for entities and papers in the knowledge graph.
      Returns a list of matching entity names, abstracts, and relations.

  entity_lookup(name: str)
      Fetch the direct neighbours, relation types, and properties of a
      specific entity node. Use exact KG entity names when known.

  cypher_query(cypher: str)
      Run a READ-ONLY Cypher statement against Neo4j.
      Useful for structured or aggregate questions.

  get_graph_context(name: str, depth: int)
      Return the subgraph within ``depth`` hops of the named entity.
      depth should be 1 or 2 (3 is slow).

  openalex_search(query: str)
      Search the OpenAlex database for recent open-access papers.
      Returns titles, abstracts, DOIs, and citation counts.

  arxiv_search(query: str)
      Search ArXiv for pre-prints.
      Returns titles, abstracts, ArXiv IDs, and submission dates.

  final_answer(answer: str, coverage_score: float)
      Signal that you have enough information. Provide the complete answer
      and a coverage_score between 0.0 (no info) and 1.0 (complete info).
      ONLY call this when you are satisfied with the answer quality.
"""

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a nuclear-fusion research assistant with access to a Knowledge Graph
and scientific literature search APIs.

Your task is to answer the user's question through iterative reasoning.
At each step you MUST:
1. Write a Thought explaining what you know so far and what you need next.
2. Choose exactly ONE action from the tool list.
3. After seeing the Observation, update your reasoning.

Stop (call final_answer) when you are confident you have enough information.
Do NOT stop prematurely. Do NOT call more than one tool per step.

{tools}

Format your response at each step as:
Thought: <your reasoning>
Action: <tool_name>(<arg1>, <arg2>, ...)
"""

_STEP_PROMPT = """\
Question: {question}

Previous steps:
{history}

Observation from last action:
{observation}

What is your next thought and action?
(If you have enough information, call final_answer(answer="...", coverage_score=0.X))
"""


# ── Tool execution ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    result: Any
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_observation(self) -> str:
        if self.error:
            return f"[ERROR from {self.tool}]: {self.error}"
        if isinstance(self.result, list):
            lines = [str(r) for r in self.result[:10]]
            tail = f"\n  ... ({len(self.result)-10} more)" if len(self.result) > 10 else ""
            return "\n".join(lines) + tail
        return str(self.result)[:2000]


def _run_search_kg(kg_agent, query: str) -> list[dict]:
    """Use the KG agent's entity linker + graph context."""
    try:
        linked = kg_agent.linker.link(query, top_k=5)
        rows: list[dict] = []
        for name, score in linked:
            rows.append({"entity": name, "link_score": round(score, 3)})
        # also fetch abstract snippets via graph context
        ctx = kg_agent._get_graphrag_context(linked[:3])
        rows.append({"abstracts": ctx[:500] if ctx else "none"})
        return rows
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _run_entity_lookup(driver, entity_name: str) -> list[dict]:
    """Fetch direct neighbours and relation types."""
    cypher = """
    MATCH (e:Entity {name_norm: toLower($name)})
    OPTIONAL MATCH (e)-[r]-(n:Entity)
    RETURN e.name AS entity, type(r) AS rel, n.name AS neighbour, r.weight AS weight
    ORDER BY r.weight DESC
    LIMIT 20
    """
    with driver.session() as session:
        result = session.run(cypher, name=entity_name.lower())
        rows = [dict(r) for r in result]
    if not rows:
        # try CONTAINS match
        cypher2 = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r]-(n:Entity)
        RETURN e.name AS entity, type(r) AS rel, n.name AS neighbour, r.weight AS weight
        ORDER BY r.weight DESC
        LIMIT 20
        """
        with driver.session() as session:
            result = session.run(cypher2, name=entity_name.lower())
            rows = [dict(r) for r in result]
    return rows


def _run_cypher(driver, cypher: str) -> list[dict]:
    """Execute a read-only Cypher query."""
    if re.search(r'\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b', cypher, re.IGNORECASE):
        raise ValueError("Only READ-ONLY Cypher is allowed.")
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(r) for r in result][:50]


def _run_graph_context(driver, name: str, depth: int = 1) -> dict:
    """Return subgraph within depth hops."""
    depth = max(1, min(depth, 2))
    cypher = f"""
    MATCH path = (e:Entity)-[*1..{depth}]-(n:Entity)
    WHERE toLower(e.name) CONTAINS toLower($name)
    RETURN e.name AS center,
           [node IN nodes(path) | node.name] AS path_nodes,
           [rel IN relationships(path) | type(rel)] AS path_rels
    LIMIT 30
    """
    with driver.session() as session:
        result = session.run(cypher, name=name.lower())
        rows = [dict(r) for r in result]
    return {"entity": name, "depth": depth, "paths": rows}


# ── Action parser ─────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(
    r'Action:\s*(\w+)\s*\(([^)]*)\)',
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_RE = re.compile(r'Thought:\s*(.*?)(?=Action:|$)', re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    r'final_answer\s*\(\s*answer\s*=\s*["\']?(.*?)["\']?\s*,\s*coverage_score\s*=\s*([0-9.]+)',
    re.DOTALL | re.IGNORECASE,
)


def _parse_llm_step(text: str) -> tuple[str, str, dict]:
    """
    Parse the LLM's response into (thought, tool_name, kwargs).
    Returns ("", "final_answer", {"answer": text, "coverage_score": "1.0"}) if
    no action pattern found (fallback: treat full response as final answer).
    """
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    # Check for final_answer inline
    final_m = _FINAL_ANSWER_RE.search(text)
    if final_m:
        return thought, "final_answer", {
            "answer": final_m.group(1).strip(),
            "coverage_score": final_m.group(2).strip(),
        }

    action_m = _ACTION_RE.search(text)
    if not action_m:
        # No parseable action – treat as final answer
        return thought, "final_answer", {
            "answer": text.strip(),
            "coverage_score": "0.5",
        }

    tool_name = action_m.group(1).strip()
    raw_args = action_m.group(2).strip()

    # parse key=value pairs
    kwargs: dict[str, Any] = {}
    for m in re.finditer(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,]+)', raw_args):
        key = m.group(1)
        val = m.group(2).strip().strip('"\'')
        kwargs[key] = val

    # positional-only call e.g. search_kg("plasma")
    if not kwargs and raw_args:
        kwargs["query"] = raw_args.strip().strip('"\'')

    return thought, tool_name, kwargs


# ── ReAct Agent ───────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    step: int
    thought: str
    tool: str
    args: dict[str, Any]
    observation: str
    coverage_score: float | None = None
    elapsed_ms: float = 0.0


@dataclass
class ReActResult:
    question: str
    answer: str
    coverage_score: float
    steps: list[ReActStep] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    stopped_reason: str = "max_steps"   # "sufficient_coverage" | "llm_final" | "max_steps"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "coverage_score": self.coverage_score,
            "stopped_reason": self.stopped_reason,
            "total_elapsed_ms": self.total_elapsed_ms,
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "tool": s.tool,
                    "args": s.args,
                    "observation": s.observation[:300],
                    "coverage_score": s.coverage_score,
                }
                for s in self.steps
            ],
        }


class ReActAgent:
    """ReAct agent that iteratively queries the KG and literature APIs.

    Parameters
    ----------
    kg_agent
        A ``FusionCypherAgent`` instance (provides LLM + Neo4j driver +
        entity linker).
    openalex_client
        Optional ``OpenAlexClient`` for literature search actions.
    arxiv_client
        Optional ``ArXivClient`` for ArXiv search actions.
    max_steps
        Maximum ReAct iterations before forced termination.
    coverage_threshold
        Stop early when the agent's self-reported coverage_score reaches
        this value (0–1).
    """

    def __init__(
        self,
        kg_agent,
        openalex_client=None,
        arxiv_client=None,
        max_steps: int = 6,
        coverage_threshold: float = 0.75,
        save_traces: bool = True,
    ):
        self.kg_agent = kg_agent
        self.openalex_client = openalex_client
        self.arxiv_client = arxiv_client
        self.max_steps = max_steps
        self.coverage_threshold = coverage_threshold
        self.save_traces = save_traces
        self._llm = kg_agent.llm
        self._driver = kg_agent.driver

    # ── Tool dispatch ──────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, kwargs: dict) -> ToolResult:
        t0 = time.perf_counter()
        try:
            if tool_name == "search_kg":
                result = _run_search_kg(self.kg_agent, kwargs.get("query", ""))
            elif tool_name == "entity_lookup":
                result = _run_entity_lookup(
                    self._driver, kwargs.get("name", kwargs.get("query", ""))
                )
            elif tool_name == "cypher_query":
                result = _run_cypher(self._driver, kwargs.get("cypher", kwargs.get("query", "")))
            elif tool_name == "get_graph_context":
                depth = int(kwargs.get("depth", 1))
                result = _run_graph_context(
                    self._driver, kwargs.get("name", kwargs.get("query", "")), depth
                )
            elif tool_name == "openalex_search":
                if self.openalex_client is None:
                    result = "[openalex_search not available — no client configured]"
                else:
                    result = self.openalex_client.search_works(
                        kwargs.get("query", ""), per_page=5
                    )
            elif tool_name == "arxiv_search":
                if self.arxiv_client is None:
                    result = "[arxiv_search not available — no client configured]"
                else:
                    result = self.arxiv_client.search(kwargs.get("query", ""), max_results=5)
            elif tool_name == "final_answer":
                result = kwargs.get("answer", "")
            else:
                result = f"[Unknown tool: {tool_name}]"
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ToolResult(tool=tool_name, args=kwargs, result=None, error=str(exc), elapsed_ms=elapsed)

        elapsed = (time.perf_counter() - t0) * 1000
        return ToolResult(tool=tool_name, args=kwargs, result=result, elapsed_ms=elapsed)

    # ── Main ReAct loop ────────────────────────────────────────────────────

    def run(self, question: str) -> ReActResult:
        """Run the ReAct loop for *question* and return a ``ReActResult``."""
        t_start = time.perf_counter()

        # Build initial prompt
        system_prompt = _SYSTEM_PROMPT.format(tools=TOOLS_DOC)
        history_lines: list[str] = []
        last_observation = "No observations yet. Start by searching the knowledge graph."
        steps: list[ReActStep] = []
        final_answer = ""
        final_coverage = 0.0
        stopped_reason = "max_steps"

        for step_i in range(1, self.max_steps + 1):
            history_str = "\n".join(history_lines[-8:]) or "(none)"
            user_prompt = _STEP_PROMPT.format(
                question=question,
                history=history_str,
                observation=last_observation,
            )

            # Ask the LLM
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            try:
                resp = self._llm.invoke(full_prompt)
                llm_text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step_i, exc)
                break

            thought, tool_name, kwargs = _parse_llm_step(llm_text)

            # Final answer?
            if tool_name == "final_answer":
                final_answer = kwargs.get("answer", llm_text)
                try:
                    final_coverage = float(kwargs.get("coverage_score", 0.8))
                except ValueError:
                    final_coverage = 0.8
                stopped_reason = "llm_final"
                steps.append(ReActStep(
                    step=step_i, thought=thought,
                    tool="final_answer", args=kwargs,
                    observation="[DONE]",
                    coverage_score=final_coverage,
                ))
                break

            # Execute tool
            tool_result = self._execute_tool(tool_name, kwargs)
            observation = tool_result.to_observation()

            # Try to extract coverage from observation (heuristic)
            cov_m = re.search(r'coverage[_\s]score[:\s=]+([0-9.]+)', observation, re.IGNORECASE)
            coverage = float(cov_m.group(1)) if cov_m else None

            step = ReActStep(
                step=step_i, thought=thought,
                tool=tool_name, args=kwargs,
                observation=observation,
                coverage_score=coverage,
                elapsed_ms=tool_result.elapsed_ms,
            )
            steps.append(step)

            # Update history
            history_lines.append(
                f"Step {step_i}:\n"
                f"  Thought: {thought[:200]}\n"
                f"  Action:  {tool_name}({kwargs})\n"
                f"  Result:  {observation[:300]}"
            )
            last_observation = observation

            # Early stopping
            if coverage is not None and coverage >= self.coverage_threshold:
                stopped_reason = "sufficient_coverage"
                final_coverage = coverage
                break

        # If loop ended without final_answer, synthesize one
        if not final_answer:
            synthesis_prompt = (
                f"Based on the following research steps, write a comprehensive answer "
                f"to the question: '{question}'\n\n"
                + "\n\n".join(
                    f"Step {s.step} ({s.tool}): {s.observation[:400]}"
                    for s in steps
                )
                + "\n\nAnswer:"
            )
            try:
                resp = self._llm.invoke(synthesis_prompt)
                final_answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
                final_coverage = 0.6
            except Exception:
                final_answer = " ".join(s.observation[:100] for s in steps)
                final_coverage = 0.3

        result = ReActResult(
            question=question,
            answer=final_answer,
            coverage_score=final_coverage,
            steps=steps,
            total_elapsed_ms=(time.perf_counter() - t_start) * 1000,
            stopped_reason=stopped_reason,
        )

        if self.save_traces:
            self._save_trace(result)

        return result

    def _save_trace(self, result: ReActResult) -> None:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = _OUTPUT_DIR / f"react_trace_{ts}.json"
        try:
            out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            logger.info("ReAct trace saved → %s", out_path)
        except OSError as exc:
            logger.warning("Could not save trace: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> None:
    import argparse
    import os
    from analysis.neo4j_utils import get_driver, get_database

    parser = argparse.ArgumentParser(prog="analysis.react_agent")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--model", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    # Lazy import to avoid circular dependencies at module load time
    from analysis.llm_graph_qa import FusionCypherAgent
    kg_agent = FusionCypherAgent()
    if args.model:
        from langchain_ollama import ChatOllama
        kg_agent.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=args.model,
        )

    try:
        from daq.arxiv_client import ArXivClient
        arxiv = ArXivClient()
    except ImportError:
        arxiv = None

    try:
        from daq.openalex_client import OpenAlexClient
        oa = OpenAlexClient()
    except ImportError:
        oa = None

    agent = ReActAgent(
        kg_agent=kg_agent,
        openalex_client=oa,
        arxiv_client=arxiv,
        max_steps=args.max_steps,
        coverage_threshold=args.coverage_threshold,
    )

    result = agent.run(args.question)

    print("\n" + "=" * 70)
    print(f"QUESTION: {result.question}")
    print(f"STOPPED:  {result.stopped_reason}  (coverage={result.coverage_score:.2f})")
    print(f"STEPS:    {len(result.steps)}")
    print("-" * 70)
    print("ANSWER:")
    print(textwrap.fill(result.answer, 80))
    print("=" * 70)

    print("\nTrace:")
    for s in result.steps:
        print(f"  [{s.step}] {s.tool} → {s.observation[:120]}")


if __name__ == "__main__":
    _main()
