"""ArXiv API client for the Fusion Knowledge Graph DAQ pipeline.

Queries the ArXiv public API (https://export.arxiv.org/api/query) to search
for physics and fusion-energy pre-prints.  No API key is required.

ArXiv subject categories used by default
-----------------------------------------
  physics.plasm-ph   — Plasma Physics
  physics.nucl-th    — Nuclear Theory
  physics.nucl-ex    — Nuclear Experiment
  cond-mat.supr-con  — Superconductivity (for magnet papers)

References
----------
- ArXiv API documentation: https://arxiv.org/help/api/user-manual
- arXiv identifier format: https://arxiv.org/help/arxiv_identifier

Usage
-----
    from daq.arxiv_client import ArXivClient
    client = ArXivClient()
    results = client.search("plasma confinement stellarator", max_results=10)
    for r in results:
        print(r["title"], r["arxiv_id"])
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://export.arxiv.org/api/query"
_DEFAULT_RATE_LIMIT = 3.0   # ArXiv asks for ≥3s between requests
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

# Default category filter — fusion-relevant physics
_DEFAULT_CATEGORIES = [
    "physics.plasm-ph",
    "physics.nucl-th",
    "physics.nucl-ex",
]


@dataclass
class ArXivRecord:
    """A single ArXiv search result."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    submitted: str = ""          # ISO date string
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""
    doi: str | None = None
    journal_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "submitted": self.submitted,
            "updated": self.updated,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
        }


def _parse_atom(xml_text: str) -> list[ArXivRecord]:
    """Parse ArXiv Atom XML into a list of ArXivRecord objects."""
    root = ET.fromstring(xml_text)
    ns = {"atom": _ATOM_NS, "arxiv": _ARXIV_NS}
    records: list[ArXivRecord] = []

    for entry in root.findall("atom:entry", ns):
        def _text(tag: str, default: str = "") -> str:
            el = entry.find(tag, ns)
            return (el.text or "").strip() if el is not None else default

        # ArXiv ID lives in <id>http://arxiv.org/abs/2201.12345v1</id>
        raw_id = _text("atom:id")
        arxiv_id = re.sub(r"https?://arxiv\.org/abs/", "", raw_id).split("v")[0]

        title = re.sub(r"\s+", " ", _text("atom:title"))
        abstract = re.sub(r"\s+", " ", _text("atom:summary"))
        submitted = _text("atom:published")
        updated = _text("atom:updated")

        authors = [
            (a.find("atom:name", ns).text or "").strip()
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]

        categories = [
            cat.get("term", "")
            for cat in entry.findall("atom:category", ns)
        ]

        # PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        jr_el = entry.find("arxiv:journal_ref", ns)
        journal_ref = jr_el.text.strip() if jr_el is not None and jr_el.text else None

        if arxiv_id and title:
            records.append(ArXivRecord(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                submitted=submitted[:10] if submitted else "",
                updated=updated[:10] if updated else "",
                categories=categories,
                pdf_url=pdf_url,
                doi=doi,
                journal_ref=journal_ref,
            ))

    return records


class ArXivClient:
    """Client for the ArXiv public query API.

    Parameters
    ----------
    categories
        List of ArXiv category strings to restrict searches.
        Pass ``[]`` to search all categories.
    rate_limit
        Minimum seconds between consecutive requests (ArXiv asks for ≥3s).
    session
        Optional ``requests.Session`` for custom proxies or headers.
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        session: requests.Session | None = None,
    ):
        self.categories = categories if categories is not None else _DEFAULT_CATEGORIES
        self.rate_limit = rate_limit
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = "FusionDAQ/1.0 (research project)"
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",       # "relevance" | "lastUpdatedDate" | "submittedDate"
        sort_order: str = "descending",
    ) -> list[ArXivRecord]:
        """Search ArXiv for *query* and return up to *max_results* records.

        The query is automatically scoped to the configured categories using
        ArXiv's ``cat:`` prefix syntax.  Pass ``categories=[]`` at init to
        disable category scoping.

        Parameters
        ----------
        query
            Plain-language or ArXiv query string (supports AND/OR/NOT).
        max_results
            Maximum number of results to return (ArXiv caps at 2000 per call).
        sort_by
            ArXiv sorting field.
        sort_order
            ``"ascending"`` or ``"descending"``.
        """
        self._wait()

        # Build category-scoped query
        if self.categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in self.categories)
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            resp = self._session.get(_API_URL, params=params, timeout=20)
            self._last_request = time.time()
        except requests.RequestException as exc:
            logger.error("ArXiv request failed: %s", exc)
            return []

        if resp.status_code != 200:
            logger.warning("ArXiv API returned %d for query: %s", resp.status_code, query)
            return []

        records = _parse_atom(resp.text)
        logger.info("ArXiv search '%s' → %d results", query[:60], len(records))
        return records

    def get_by_id(self, arxiv_id: str) -> ArXivRecord | None:
        """Fetch a single record by ArXiv ID (e.g. '2201.12345')."""
        self._wait()
        params = {"id_list": arxiv_id, "max_results": 1}
        try:
            resp = self._session.get(_API_URL, params=params, timeout=15)
            self._last_request = time.time()
        except requests.RequestException as exc:
            logger.error("ArXiv ID fetch failed (%s): %s", arxiv_id, exc)
            return None

        records = _parse_atom(resp.text)
        return records[0] if records else None

    def search_works(self, query: str, per_page: int = 5) -> list[dict]:
        """Compatibility shim matching OpenAlexClient.search_works signature."""
        records = self.search(query, max_results=per_page)
        return [r.to_dict() for r in records]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="daq.arxiv_client")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--all-categories", action="store_true",
                        help="Don't restrict to physics categories")
    parser.add_argument("--sort", default="relevance",
                        choices=["relevance", "lastUpdatedDate", "submittedDate"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    client = ArXivClient(categories=[] if args.all_categories else None)
    results = client.search(args.query, max_results=args.max_results, sort_by=args.sort)

    if not results:
        print("No results found.")
        return

    for r in results:
        print(f"\n[{r.arxiv_id}] {r.title}")
        print(f"  Authors  : {', '.join(r.authors[:3])}{'...' if len(r.authors)>3 else ''}")
        print(f"  Submitted: {r.submitted}")
        print(f"  PDF      : {r.pdf_url}")
        print(f"  Abstract : {r.abstract[:200]}...")


if __name__ == "__main__":
    _main()
