"""Research Swarm — Parallel Literature Discovery for Knowledge Gap Filling
==========================================================================
Coordinates multiple concurrent research agents, each assigned to a specific
gap entity or topic.  Each agent searches both OpenAlex and ArXiv, collects
candidate papers, de-duplicates, and returns ranked results ready for
ingestion by the DAQ downloader.

Architecture
------------
                         ┌─ Agent: "stellarator coil optimisation" → OA + ArXiv ─┐
GapReport  →  Swarm  ─┤─ Agent: "plasma disruption mitigation"    → OA + ArXiv ─├──► Merged Results
                         └─ Agent: "tritium breeding blanket"       → OA + ArXiv ─┘

Key design decisions
--------------------
- ``concurrent.futures.ThreadPoolExecutor`` (not ``asyncio``) so agents can
  call the blocking ``requests`` library without conversion overhead.
- Each agent gets its own ``OpenAlexClient`` and ``ArXivClient`` instance to
  avoid shared rate-limit state.
- Results are ranked by a composite score: citation_count + recency + OA bonus.
- Duplicate detection by normalised DOI / ArXiv ID / title hash.
- Designed to be triggered automatically from the chat app when a question
  repeatedly fails (coverage_score < threshold) or manually from CLI.

Usage
-----
    from daq.research_swarm import ResearchSwarm
    swarm = ResearchSwarm.from_gap_report("output/answer_gap_report.json")
    new_papers = swarm.run(n_agents=5, papers_per_agent=8)
    swarm.save_results("output/swarm_results.json")

CLI
---
    python -m daq.research_swarm --top-gaps 10 --papers-per-agent 8 --dry-run
    python -m daq.research_swarm --query "plasma instabilities ITER" --agents 3
    python -m daq.research_swarm --top-gaps 20 --papers-per-agent 5
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daq.arxiv_client import ArXivClient, ArXivRecord
from daq.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
_OUTPUT_DIR = Path("output")
_DEFAULT_GAP_REPORT = Path("output/answer_gap_report.json")
_DEFAULT_SWARM_OUT = Path("output/swarm_results.json")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SwarmPaper:
    """A single candidate paper discovered by the swarm."""
    title: str
    abstract: str
    source: str                         # "openalex" | "arxiv"
    source_id: str                      # OA work ID or ArXiv ID
    doi: str | None = None
    pdf_url: str | None = None
    authors: list[str] = field(default_factory=list)
    published: str = ""
    citations: int = 0
    is_oa: bool = False
    query: str = ""                     # which gap query triggered this
    dedup_key: str = ""                 # normalised identifier for dedup
    score: float = 0.0                  # composite ranking score

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract[:600],
            "source": self.source,
            "source_id": self.source_id,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "authors": self.authors[:5],
            "published": self.published,
            "citations": self.citations,
            "is_oa": self.is_oa,
            "query": self.query,
            "score": round(self.score, 3),
        }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_paper(paper: SwarmPaper) -> float:
    """Composite score: citation weight + recency bonus + OA bonus."""
    # Citation component (log-normalised, caps at ~7 for 1000+ citations)
    import math
    cit_score = math.log1p(paper.citations) * 0.4

    # Recency bonus: full point for 2023+, decays by 0.1/year
    year = 0
    if paper.published and len(paper.published) >= 4:
        try:
            year = int(paper.published[:4])
        except ValueError:
            year = 2000
    recency = max(0.0, 1.0 - (2025 - year) * 0.1) * 0.4

    # OA bonus (we can actually download it)
    oa_bonus = 0.2 if paper.is_oa or paper.pdf_url else 0.0

    return cit_score + recency + oa_bonus


# ── Dedup key ─────────────────────────────────────────────────────────────────

def _dedup_key(doi: str | None, source_id: str, title: str) -> str:
    """Return a normalised key for duplicate detection."""
    if doi:
        return re.sub(r'[^a-z0-9]', '', doi.lower())
    if source_id:
        return re.sub(r'[^a-z0-9]', '', source_id.lower())
    # title-based fallback
    slug = re.sub(r'[^a-z0-9 ]', '', title.lower())
    slug = re.sub(r'\s+', '', slug[:60])
    return hashlib.md5(slug.encode()).hexdigest()[:12]


# ── Single-agent search ───────────────────────────────────────────────────────

def _agent_search(
    query: str,
    papers_per_source: int,
    oa_email: str | None,
) -> list[SwarmPaper]:
    """Run one agent: search OpenAlex AND ArXiv for *query*.

    Each agent creates its own client instances to avoid rate-limit contention.
    """
    results: list[SwarmPaper] = []

    # ── OpenAlex ─────────────────────────────────────────────────────────
    try:
        oa = OpenAlexClient(email=oa_email, rate_limit=1.5)
        oa_works = oa.search_works(query, per_page=papers_per_source)
        for w in oa_works:
            if not isinstance(w, dict):
                continue
            title = (w.get("title") or "").strip()
            if not title:
                continue
            oa_loc = w.get("open_access") or {}
            pdf_url = oa_loc.get("oa_url") or w.get("pdf_url")
            doi = w.get("doi")
            published = (w.get("publication_date") or "")[:10]
            citations = int(w.get("cited_by_count") or 0)
            abstract = w.get("abstract") or ""
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:5]
            ]
            is_oa = bool(oa_loc.get("is_oa"))
            source_id = w.get("id") or ""

            paper = SwarmPaper(
                title=title,
                abstract=abstract,
                source="openalex",
                source_id=source_id,
                doi=doi,
                pdf_url=pdf_url,
                authors=[a for a in authors if a],
                published=published,
                citations=citations,
                is_oa=is_oa,
                query=query,
                dedup_key=_dedup_key(doi, source_id, title),
            )
            paper.score = _score_paper(paper)
            results.append(paper)
        logger.debug("OA agent '%s' → %d results", query[:40], len(results))
    except Exception as exc:
        logger.warning("OpenAlex agent failed for '%s': %s", query[:40], exc)

    # ── ArXiv ─────────────────────────────────────────────────────────────
    try:
        arxiv = ArXivClient()
        arxiv_records = arxiv.search(query, max_results=papers_per_source)
        for rec in arxiv_records:
            paper = SwarmPaper(
                title=rec.title,
                abstract=rec.abstract,
                source="arxiv",
                source_id=rec.arxiv_id,
                doi=rec.doi,
                pdf_url=rec.pdf_url,
                authors=rec.authors[:5],
                published=rec.submitted,
                citations=0,      # ArXiv has no citation count
                is_oa=True,       # ArXiv is always OA
                query=query,
                dedup_key=_dedup_key(rec.doi, rec.arxiv_id, rec.title),
            )
            paper.score = _score_paper(paper)
            results.append(paper)
        logger.debug("ArXiv agent '%s' → %d results", query[:40], len(arxiv_records))
    except Exception as exc:
        logger.warning("ArXiv agent failed for '%s': %s", query[:40], exc)

    return results


# ── ResearchSwarm ─────────────────────────────────────────────────────────────

class ResearchSwarm:
    """Parallel research swarm for gap-driven literature discovery.

    Parameters
    ----------
    queries
        List of search queries (one per agent thread).
    n_agents
        Maximum concurrent agent threads.
    papers_per_agent
        Results requested from each source per agent.
    oa_email
        Contact email for OpenAlex polite pool.
    existing_dois
        Set of DOIs already in the catalogue (used for dedup).
    """

    def __init__(
        self,
        queries: list[str],
        n_agents: int = 5,
        papers_per_agent: int = 8,
        oa_email: str | None = None,
        existing_dois: set[str] | None = None,
    ):
        self.queries = queries
        self.n_agents = n_agents
        self.papers_per_agent = papers_per_agent
        self.oa_email = oa_email
        self._existing_dois = existing_dois or set()
        self._results: list[SwarmPaper] = []

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def from_gap_report(
        cls,
        report_path: str | Path = _DEFAULT_GAP_REPORT,
        top_gaps: int = 10,
        n_agents: int = 5,
        papers_per_agent: int = 8,
        oa_email: str | None = None,
        catalogue_path: str | Path | None = None,
    ) -> "ResearchSwarm":
        """Build a swarm from the gap report's top failure entities.

        Each top entity becomes a search query enriched with fusion-domain
        context (e.g. "pellet injection" → "pellet injection nuclear fusion plasma").
        """
        report_path = Path(report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Gap report not found: {report_path}")

        data = json.loads(report_path.read_text(encoding="utf-8"))
        top_ents = data.get("top_entities_in_failures") or []
        # top_entities_in_failures is [[name, count], ...]
        entity_names = [name for name, _count in top_ents[:top_gaps]]
        if not entity_names:
            # fallback: use unlinked gap entities
            entity_names = (data.get("unlinked_entities") or [])[:top_gaps]

        queries = [f"{name} nuclear fusion plasma" for name in entity_names]
        logger.info("Swarm built from gap report: %d queries from top entities", len(queries))

        # Load existing DOIs for dedup
        existing_dois: set[str] = set()
        cat_path = Path(catalogue_path) if catalogue_path else Path("data/fulltexts/catalogue_enriched.json")
        if cat_path.exists():
            try:
                catalogue = json.loads(cat_path.read_text(encoding="utf-8"))
                existing_dois = {r.get("doi") for r in catalogue if r.get("doi")} 
            except Exception:
                pass

        return cls(
            queries=queries,
            n_agents=n_agents,
            papers_per_agent=papers_per_agent,
            oa_email=oa_email,
            existing_dois=existing_dois,
        )

    @classmethod
    def from_queries(
        cls,
        queries: list[str],
        n_agents: int = 5,
        papers_per_agent: int = 8,
        oa_email: str | None = None,
    ) -> "ResearchSwarm":
        """Build a swarm from an explicit list of search queries."""
        return cls(queries=queries, n_agents=n_agents, papers_per_agent=papers_per_agent, oa_email=oa_email)

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self) -> list[SwarmPaper]:
        """Dispatch all agents concurrently and return deduplicated, ranked results."""
        t0 = time.time()
        raw: list[SwarmPaper] = []

        logger.info(
            "Swarm starting: %d queries × up to %d concurrent agents",
            len(self.queries), self.n_agents,
        )

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            future_to_query = {
                pool.submit(
                    _agent_search, q, self.papers_per_agent, self.oa_email
                ): q
                for q in self.queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    papers = future.result()
                    raw.extend(papers)
                    logger.info("Agent done: '%s' → %d papers", query[:50], len(papers))
                except Exception as exc:
                    logger.error("Agent failed for '%s': %s", query[:50], exc)

        # Deduplicate
        seen: set[str] = set()
        unique: list[SwarmPaper] = []
        for p in raw:
            if p.dedup_key in seen:
                continue
            if p.doi and p.doi in self._existing_dois:
                continue
            seen.add(p.dedup_key)
            unique.append(p)

        # Sort by composite score descending
        unique.sort(key=lambda p: p.score, reverse=True)
        self._results = unique

        elapsed = time.time() - t0
        logger.info(
            "Swarm complete: %d raw → %d unique papers in %.1fs",
            len(raw), len(unique), elapsed,
        )
        return unique

    # ── Results access ─────────────────────────────────────────────────────

    @property
    def results(self) -> list[SwarmPaper]:
        return self._results

    def top_downloadable(self, n: int = 20) -> list[SwarmPaper]:
        """Return the top-n papers that have a PDF URL (OA only)."""
        oa_papers = [p for p in self._results if p.pdf_url]
        return oa_papers[:n]

    def summary(self) -> dict[str, Any]:
        total = len(self._results)
        oa = sum(1 for p in self._results if p.is_oa or p.pdf_url)
        by_source = {
            "openalex": sum(1 for p in self._results if p.source == "openalex"),
            "arxiv": sum(1 for p in self._results if p.source == "arxiv"),
        }
        return {
            "total": total,
            "open_access": oa,
            "by_source": by_source,
            "queries": len(self.queries),
            "top_titles": [p.title[:80] for p in self._results[:5]],
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save_results(self, path: str | Path = _DEFAULT_SWARM_OUT) -> None:
        """Write results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "queries": self.queries,
            "summary": self.summary(),
            "papers": [p.to_dict() for p in self._results],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("Swarm results saved → %s (%d papers)", path, len(self._results))

    @staticmethod
    def load_results(path: str | Path = _DEFAULT_SWARM_OUT) -> list[SwarmPaper]:
        """Load previously saved swarm results."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        papers = []
        for d in data.get("papers", []):
            p = SwarmPaper(
                title=d.get("title", ""),
                abstract=d.get("abstract", ""),
                source=d.get("source", ""),
                source_id=d.get("source_id", ""),
                doi=d.get("doi"),
                pdf_url=d.get("pdf_url"),
                authors=d.get("authors", []),
                published=d.get("published", ""),
                citations=d.get("citations", 0),
                is_oa=d.get("is_oa", False),
                query=d.get("query", ""),
                score=d.get("score", 0.0),
            )
            p.dedup_key = _dedup_key(p.doi, p.source_id, p.title)
            papers.append(p)
        return papers


# ── Integration helper: trigger swarm on repeated failures ────────────────────

def maybe_trigger_swarm(
    question: str,
    coverage_score: float,
    coverage_threshold: float = 0.4,
    n_agents: int = 3,
    papers_per_agent: int = 5,
    oa_email: str | None = None,
    save: bool = True,
) -> list[SwarmPaper] | None:
    """Convenience function called by the chat app when coverage is too low.

    Returns a list of discovered papers, or None if the swarm was not triggered.
    """
    if coverage_score >= coverage_threshold:
        return None

    logger.info(
        "Low coverage (%.2f < %.2f) — triggering research swarm for: %s",
        coverage_score, coverage_threshold, question[:60],
    )
    swarm = ResearchSwarm.from_queries(
        queries=[f"{question} nuclear fusion", question],
        n_agents=n_agents,
        papers_per_agent=papers_per_agent,
        oa_email=oa_email,
    )
    results = swarm.run()
    if save and results:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        swarm.save_results(_OUTPUT_DIR / f"swarm_{ts}.json")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> None:
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(prog="daq.research_swarm")
    parser.add_argument("--top-gaps", type=int, default=10,
                        help="Number of top gap entities to use as queries (default: 10)")
    parser.add_argument("--query", nargs="+", metavar="QUERY",
                        help="Explicit search queries (overrides --top-gaps)")
    parser.add_argument("--agents", type=int, default=5,
                        help="Max concurrent agent threads (default: 5)")
    parser.add_argument("--papers-per-agent", type=int, default=8,
                        help="Papers to request per source per agent (default: 8)")
    parser.add_argument("--gap-report", default=str(_DEFAULT_GAP_REPORT),
                        help="Path to gap report JSON")
    parser.add_argument("--email",
                        default=os.getenv("OPENALEX_EMAIL", "fusiondag@research.local"),
                        help="Contact email for OpenAlex polite pool")
    parser.add_argument("--output", default=str(_DEFAULT_SWARM_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without saving")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    if args.query:
        swarm = ResearchSwarm.from_queries(
            queries=args.query,
            n_agents=args.agents,
            papers_per_agent=args.papers_per_agent,
            oa_email=args.email,
        )
    else:
        swarm = ResearchSwarm.from_gap_report(
            report_path=args.gap_report,
            top_gaps=args.top_gaps,
            n_agents=args.agents,
            papers_per_agent=args.papers_per_agent,
            oa_email=args.email,
        )

    results = swarm.run()
    summary = swarm.summary()

    print(f"\nSwarm Results: {summary['total']} papers ({summary['open_access']} OA)")
    print(f"  OpenAlex: {summary['by_source']['openalex']}  |  ArXiv: {summary['by_source']['arxiv']}")
    print(f"  Queries : {summary['queries']}")
    print("\nTop 10 papers:")
    for i, p in enumerate(results[:10], 1):
        oa_tag = "[OA]" if p.is_oa or p.pdf_url else "    "
        print(f"  {i:2}. {oa_tag} [{p.source}] {p.title[:70]}")
        print(f"       score={p.score:.2f}  cited={p.citations}  date={p.published[:7]}  query='{p.query[:30]}'")

    if not args.dry_run:
        swarm.save_results(args.output)
        print(f"\nSaved → {args.output}")
    else:
        print("\n[dry-run — not saved]")


if __name__ == "__main__":
    _main()
