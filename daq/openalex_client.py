"""OpenAlex API client for resolving paper metadata and OA status.

OpenAlex (https://openalex.org) is a free, open catalogue of the global
research system.  It replaces the deprecated Microsoft Academic Graph and
covers >250M works.

This client:
1. Resolves DOIs → full metadata (title, authors, OA location, PDF URL).
2. Falls back to title search when no DOI is available.
3. Respects the OpenAlex polite pool (1 req/s with ``mailto`` parameter).
4. Returns results as updates to ``PaperRecord`` objects.

No API key is required; providing an email address
(via ``OPENALEX_EMAIL`` env var or constructor arg) routes requests through
the faster "polite" pool.

References
----------
- OpenAlex API docs: https://docs.openalex.org/
- Priem, J. et al. (2022). "OpenAlex: A fully-open index of scholarly works,
  authors, venues, institutions, and concepts." arXiv:2205.01833.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any
from urllib.parse import quote

import requests

from daq.doi_extraction import PaperRecord

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org"
_DEFAULT_RATE_LIMIT = 1.0  # seconds between requests (polite pool)


class OpenAlexClient:
    """Thin wrapper around the OpenAlex REST API.

    Parameters
    ----------
    email
        Contact email for the polite pool.  Falls back to the
        ``OPENALEX_EMAIL`` environment variable.
    rate_limit
        Minimum seconds between consecutive requests.
    session
        Optional ``requests.Session`` (useful for testing / proxies).
    """

    def __init__(
        self,
        email: str | None = None,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        session: requests.Session | None = None,
    ):
        self.email = email or os.getenv("OPENALEX_EMAIL")
        if not self.email:
            logger.warning(
                "No email set for OpenAlex polite pool. "
                "Set OPENALEX_EMAIL or pass email= to the constructor. "
                "Requests may be rate-limited more aggressively."
            )
        self.rate_limit = rate_limit
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "FusionDAQ/1.0 (research project)"
        self._last_request_time = 0.0

    # ── Low-level request ─────────────────────────────────────────────────

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict | None:
        """Rate-limited GET returning parsed JSON, or None on failure."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        params = dict(params or {})
        if self.email:
            params["mailto"] = self.email

        try:
            resp = self.session.get(url, params=params, timeout=15)
            self._last_request_time = time.time()

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited by OpenAlex, sleeping %ds", retry_after)
                time.sleep(retry_after)
                return self._get(url, params)

            if resp.status_code != 200:
                logger.debug("OpenAlex %s returned %d", url, resp.status_code)
                return None

            return resp.json()
        except requests.RequestException as exc:
            logger.warning("OpenAlex request failed: %s", exc)
            return None

    # ── Work resolution ───────────────────────────────────────────────────

    def resolve_doi(self, doi: str) -> dict | None:
        """Look up a work by DOI.

        Returns the raw OpenAlex work object or ``None``.
        """
        url = f"{_BASE_URL}/works/https://doi.org/{quote(doi, safe='')}"
        return self._get(url)

    def search_title(self, title: str) -> dict | None:
        """Search for a work by exact title match.

        Returns the best-matching work object or ``None``.
        """
        url = f"{_BASE_URL}/works"
        data = self._get(url, params={
            "filter": f"title.search:{title}",
            "per_page": "1",
        })
        if data and data.get("results"):
            return data["results"][0]
        return None

    # ── Extract OA metadata ───────────────────────────────────────────────

    # Repository sources we prefer, in priority order (arXiv is king in physics)
    _REPO_PRIORITY = ["arxiv", "osti", "zenodo", "hal", "pubmed"]

    @classmethod
    def _extract_oa_info(cls, work: dict) -> dict[str, Any]:
        """Pull OA status, PDF URL, and license from an OpenAlex work.

        For papers whose primary OA status is "closed", also scans all
        ``locations`` for green-OA repository copies (arXiv, OSTI, Zenodo,
        institutional repos, etc.).
        """
        oa = work.get("open_access", {})
        best_loc = work.get("best_oa_location") or {}

        # Try primary location if best_oa_location has no PDF
        pdf_url = best_loc.get("pdf_url")
        if not pdf_url:
            primary = work.get("primary_location") or {}
            pdf_url = primary.get("pdf_url")

        # Also try locations list for publisher PDF
        if not pdf_url:
            for loc in work.get("locations", []):
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    break

        # ── Repository / preprint fallback ────────────────────────────
        # Scan all locations for open-access repository copies
        # (arXiv preprints, OSTI, Zenodo, institutional repos).
        repo_pdf_url = None
        repo_name = None
        repo_landing = None

        # Collect all OA repo locations
        repo_candidates: list[tuple[int, dict]] = []
        for loc in work.get("locations", []):
            source = loc.get("source") or {}
            src_type = source.get("type", "")
            if src_type != "repository" or not loc.get("is_oa"):
                continue
            src_name = (source.get("display_name") or "").lower()
            # Assign priority: known repos first, others last
            priority = len(cls._REPO_PRIORITY)
            for i, keyword in enumerate(cls._REPO_PRIORITY):
                if keyword in src_name:
                    priority = i
                    break
            repo_candidates.append((priority, loc))

        # Pick the best repository location
        if repo_candidates:
            repo_candidates.sort(key=lambda x: x[0])
            best_repo = repo_candidates[0][1]
            repo_pdf_url = best_repo.get("pdf_url")
            repo_landing = best_repo.get("landing_page_url")
            repo_name = (best_repo.get("source") or {}).get("display_name")
            # If no direct PDF but there is a landing page, keep it
            # (the downloader will try the landing page as fallback)
            if not repo_pdf_url and repo_landing:
                repo_pdf_url = repo_landing

        return {
            "openalex_id": work.get("id"),
            "oa_status": oa.get("oa_status", "closed"),
            "pdf_url": pdf_url,
            "landing_url": best_loc.get("landing_page_url")
                           or work.get("primary_location", {}).get("landing_page_url"),
            "license": best_loc.get("license"),
            "doi": work.get("doi", "").replace("https://doi.org/", "")
                   if work.get("doi") else None,
            "title": work.get("title"),
            "repository_pdf_url": repo_pdf_url,
            "repository_name": repo_name,
        }

    # ── High-level: enrich a PaperRecord ──────────────────────────────────

    def enrich(self, record: PaperRecord) -> PaperRecord:
        """Resolve metadata for a single paper and update the record in place.

        Strategy:
        1. If the record has a DOI, try ``resolve_doi`` first.
        2. Fall back to ``search_title`` if DOI lookup fails or is absent.
        3. Update the record's OA fields with whatever was found.
        """
        work: dict | None = None

        if record.doi:
            work = self.resolve_doi(record.doi)

        if work is None and record.title:
            work = self.search_title(record.title)

        if work is None:
            logger.debug("Could not resolve: %s", record.paper_id)
            record.oa_status = "unknown"
            return record

        info = self._extract_oa_info(work)
        record.openalex_id = info["openalex_id"]
        record.oa_status = info["oa_status"]
        record.pdf_url = info["pdf_url"]
        record.landing_url = info["landing_url"]
        record.license = info["license"]
        record.repository_pdf_url = info["repository_pdf_url"]
        record.repository_name = info["repository_name"]

        # Back-fill DOI if we only had a title
        if not record.doi and info.get("doi"):
            record.doi = info["doi"]

        # Back-fill title from OpenAlex if ours was missing
        if not record.title and info.get("title"):
            record.title = info["title"]

        return record
