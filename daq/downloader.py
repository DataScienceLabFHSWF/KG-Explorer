"""Polite, rate-limited downloader for open-access PDFs and HTML.

Design principles:
- **Respectful**: obeys rate limits, sends a descriptive User-Agent, and
  only downloads papers flagged as open access by OpenAlex.
- **Resumable**: skips files that already exist on disk.
- **KGBuilder-compatible**: saves PDFs into a flat directory that can be
  passed directly to ``kgbuilder run --docs ./output_dir/``.

The module is domain-agnostic and can be reused for any research domain.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable

import requests

from daq.doi_extraction import PaperRecord

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = 30  # seconds per HTTP request
_DEFAULT_RATE_LIMIT = 1.5  # seconds between downloads
_MAX_FILENAME_LEN = 120

# Content-types we consider valid PDFs
_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_filename(raw: str) -> str:
    """Turn a title or DOI into a safe, filesystem-friendly basename."""
    # Replace path-unsafe chars
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw)
    # Collapse whitespace
    safe = re.sub(r"\s+", "_", safe).strip("_.")
    if len(safe) > _MAX_FILENAME_LEN:
        safe = safe[:_MAX_FILENAME_LEN]
    return safe


def _make_filename(record: PaperRecord) -> str:
    """Build a filename for a paper: prefer DOI, fall back to title."""
    base = record.doi or record.title or record.paper_id
    name = _sanitise_filename(base)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


# ── Downloader ────────────────────────────────────────────────────────────────

class PaperDownloader:
    """Download open-access PDFs for a set of resolved ``PaperRecord``s.

    Parameters
    ----------
    output_dir
        Directory where PDFs will be saved.
    rate_limit
        Minimum seconds between consecutive HTTP requests.
    timeout
        Per-request timeout in seconds.
    session
        Optional ``requests.Session`` (for proxies, custom auth, etc.).
    """

    def __init__(
        self,
        output_dir: Path,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        timeout: int = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = (
            "FusionDAQ/1.0 (academic research; KGPlatform integration)"
        )
        self._last_request_time = 0.0

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _try_download(self, url: str, dest: Path) -> bool:
        """Attempt to download a PDF from *url* to *dest*.

        Returns True on success, False on any failure.
        """
        self._wait()
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )
            self._last_request_time = time.time()

            if resp.status_code != 200:
                logger.debug("HTTP %d for %s", resp.status_code, url)
                return False

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()

            # Accept PDFs
            if content_type in _PDF_CONTENT_TYPES:
                dest.write_bytes(resp.content)
                return True

            # Some servers serve PDFs as octet-stream
            if content_type == "application/octet-stream" and len(resp.content) > 1024:
                # Check PDF magic bytes
                if resp.content[:5] == b"%PDF-":
                    dest.write_bytes(resp.content)
                    return True

            logger.debug(
                "Unexpected content-type %r (%d bytes) for %s",
                content_type, len(resp.content), url,
            )
            return False

        except requests.RequestException as exc:
            logger.debug("Download error for %s: %s", url, exc)
            return False

    def download(self, record: PaperRecord) -> Path | None:
        """Download the PDF for a single paper.

        Returns the path to the saved file, or ``None`` if unsuccessful.
        The record must have been enriched beforehand.

        Download strategy:
        1. If the paper is openly accessible, try ``pdf_url`` then ``landing_url``.
        2. For *any* paper (including closed), try ``repository_pdf_url`` —
           this covers arXiv preprints, OSTI, Zenodo, institutional repos.
        """
        filename = _make_filename(record)
        dest = self.output_dir / filename

        # Resume: skip if already downloaded
        if dest.exists() and dest.stat().st_size > 1024:
            logger.debug("Already exists: %s", dest.name)
            return dest

        # Collect candidate URLs in priority order
        urls: list[tuple[str, str]] = []

        # Publisher OA PDF (only for non-closed papers)
        if record.oa_status not in ("closed", "unknown", None):
            if record.pdf_url:
                urls.append((record.pdf_url, "publisher OA"))
            if record.landing_url and record.landing_url != record.pdf_url:
                urls.append((record.landing_url, "landing page"))

        # Repository / preprint fallback (for ALL papers, including closed)
        if record.repository_pdf_url:
            urls.append((record.repository_pdf_url,
                         f"repository ({record.repository_name or 'unknown'})"))

        if not urls:
            return None

        for url, source in urls:
            if self._try_download(url, dest):
                logger.info("Downloaded via %s: %s", source, dest.name)
                return dest

        return None

    def download_batch(
        self,
        records: Iterable[PaperRecord],
        *,
        limit: int | None = None,
    ) -> list[tuple[PaperRecord, Path]]:
        """Download PDFs for a batch of papers.

        Parameters
        ----------
        records
            Iterable of enriched ``PaperRecord`` objects.
        limit
            Maximum number of PDFs to download (for testing).

        Returns
        -------
        list of (record, path) tuples for successfully downloaded papers.
        """
        results: list[tuple[PaperRecord, Path]] = []
        count = 0

        for rec in records:
            if limit is not None and count >= limit:
                break

            path = self.download(rec)
            if path is not None:
                results.append((rec, path))
                count += 1

        return results
