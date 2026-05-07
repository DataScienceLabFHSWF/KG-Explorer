"""Autopilot ingestion: discover new papers for answer-gap entities.

Reads ``output/answer_gap_report.json``, takes the top-K repeat-offender
entities (those that appear most in failed QA queries), queries OpenAlex for
the most-cited papers mentioning each, and downloads any open-access PDFs
not already in the catalogue.

Intended use
------------
Run after building a gap report to automatically expand the corpus with the
papers most likely to fill identified knowledge holes::

    python -m daq.from_gap_report --top 10 --papers-per-entity 5 --dry-run
    python -m daq.from_gap_report --top 10 --papers-per-entity 5

The ``--dry-run`` flag prints what *would* be downloaded without touching
the file system or catalogue.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from daq.doi_extraction import PaperRecord, load_catalogue, save_catalogue
from daq.downloader import PaperDownloader
from daq.openalex_client import OpenAlexClient
from analysis.entity_linker import _is_junk_entity

logger = logging.getLogger(__name__)

_DEFAULT_GAP_REPORT = Path("output/answer_gap_report.json")
_DEFAULT_CATALOGUE = Path("data/fulltexts/catalogue_enriched.json")
_BASE_OA = "https://api.openalex.org"


# ── Gap-report parsing ────────────────────────────────────────────────────────

def _load_gap_entities(report_path: Path, top_k: int) -> list[str]:
    """Return the top-K entity names from the gap report's failure list."""
    if not report_path.exists():
        raise FileNotFoundError(
            f"Gap report not found: {report_path}. "
            "Run `python -m tests.run_qa_tests` first to generate it."
        )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    top_ents = data.get("top_entities_in_failures") or []
    # top_entities_in_failures is a list of [name, count] pairs
    return [name for name, _count in top_ents[:top_k]]


# ── OpenAlex keyword search ───────────────────────────────────────────────────

def _search_by_keyword(
    client: OpenAlexClient, keyword: str, per_page: int
) -> list[dict[str, Any]]:
    """Query OpenAlex for most-cited articles matching *keyword*.

    Uses full-text search ranked by ``cited_by_count`` descending.
    """
    data = client._get(
        f"{_BASE_OA}/works",
        params={
            "search": keyword,
            "filter": "type:article",
            "sort": "cited_by_count:desc",
            "per_page": str(min(per_page, 25)),  # OA caps at 200; 25 is plenty
        },
    )
    if not data:
        return []
    return data.get("results") or []


# ── OA work → PaperRecord ─────────────────────────────────────────────────────

def _work_to_record(work: dict, source_entity: str) -> PaperRecord | None:
    """Convert a raw OpenAlex work object to a ``PaperRecord``.

    Returns ``None`` if the work has neither a DOI nor a title.
    """
    doi_raw = (work.get("doi") or "").replace("https://doi.org/", "").strip()
    doi = doi_raw or None
    title = work.get("title") or None

    if not doi and not title:
        return None

    paper_id = doi or title  # doi is a stable unique key; title as fallback

    year: int | None = None
    py = work.get("publication_year")
    if isinstance(py, int):
        year = py

    first_author: str | None = None
    authorships = work.get("authorships") or []
    if authorships:
        first_author = (authorships[0].get("author") or {}).get("display_name")

    citations: int = work.get("cited_by_count") or 0

    oa_info = OpenAlexClient._extract_oa_info(work)

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        doi=doi,
        year_published=year,
        first_author=first_author,
        citations=citations,
        source_file=f"gap:{source_entity}",
        openalex_id=oa_info["openalex_id"],
        oa_status=oa_info["oa_status"],
        pdf_url=oa_info["pdf_url"],
        landing_url=oa_info["landing_url"],
        license=oa_info["license"],
        repository_pdf_url=oa_info["repository_pdf_url"],
        repository_name=oa_info["repository_name"],
    )


# ── Main run function ─────────────────────────────────────────────────────────

def run(
    gap_report: Path = _DEFAULT_GAP_REPORT,
    top_k: int = 10,
    papers_per_entity: int = 5,
    output_dir: Path = Path("data/fulltexts/"),
    catalogue_path: Path = _DEFAULT_CATALOGUE,
    email: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover and download papers for the top-K gap entities.

    Parameters
    ----------
    gap_report
        Path to ``answer_gap_report.json``.
    top_k
        Number of top repeat-offender entities to process.
    papers_per_entity
        Max OpenAlex results to fetch per entity.
    output_dir
        Where to write downloaded PDFs.
    catalogue_path
        Enriched catalogue to read/update (avoids re-downloading).
    email
        Email for the OpenAlex polite pool (or set ``OPENALEX_EMAIL`` env var).
    dry_run
        If ``True``, log what would happen but do not touch the filesystem.

    Returns
    -------
    dict with keys ``entities``, ``candidates``, ``downloadable``,
    ``downloaded`` (always 0 in dry-run mode).
    """
    entities = _load_gap_entities(gap_report, top_k)
    # Filter out generic/junk entity names before querying OpenAlex
    entities = [e for e in entities if not _is_junk_entity(e) and len(e) >= 4]
    logger.info("Processing %d gap entities (after filtering): %s", len(entities), entities)

    # ── Load existing DOIs to skip duplicates ─────────────────────────────
    existing_dois: set[str] = set()
    if catalogue_path.exists():
        for rec in load_catalogue(catalogue_path):
            if rec.doi:
                existing_dois.add(rec.doi.lower())
        logger.info("Loaded %d existing DOIs from catalogue", len(existing_dois))

    # ── Search OpenAlex for each entity ───────────────────────────────────
    client = OpenAlexClient(email=email)
    new_records: list[PaperRecord] = []

    for entity in entities:
        logger.info("Searching OpenAlex: %r", entity)
        works = _search_by_keyword(client, entity, papers_per_entity)
        added = 0
        for work in works:
            rec = _work_to_record(work, source_entity=entity)
            if rec is None:
                continue
            if rec.doi and rec.doi.lower() in existing_dois:
                logger.debug("  Skip (already have): %s", rec.doi)
                continue
            new_records.append(rec)
            if rec.doi:
                existing_dois.add(rec.doi.lower())
            added += 1
        logger.info("  Found %d new records for %r", added, entity)

    downloadable = sum(
        1 for r in new_records if r.pdf_url or r.repository_pdf_url
    )
    logger.info(
        "Total new candidates: %d  (%d downloadable)", len(new_records), downloadable
    )

    # ── Download open-access PDFs ─────────────────────────────────────────
    downloaded = 0
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloader = PaperDownloader(output_dir=output_dir)
        for rec in new_records:
            if not (rec.pdf_url or rec.repository_pdf_url):
                continue
            try:
                path = downloader.download(rec)
                if path:
                    downloaded += 1
                    logger.info("  Downloaded: %s → %s", rec.paper_id, path.name)
            except Exception:
                logger.debug("  Download failed for %s", rec.paper_id, exc_info=True)

        # Append new records to the catalogue
        existing = load_catalogue(catalogue_path) if catalogue_path.exists() else []
        save_catalogue(existing + new_records, catalogue_path)
        logger.info("Catalogue updated with %d new records → %s", len(new_records), catalogue_path)
    else:
        for rec in new_records:
            url = rec.pdf_url or rec.repository_pdf_url or "(no OA URL)"
            logger.info(
                "  [DRY] %s  oa=%s  url=%s",
                rec.paper_id, rec.oa_status, url,
            )

    return {
        "entities": entities,
        "candidates": len(new_records),
        "downloadable": downloadable,
        "downloaded": downloaded,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="daq.from_gap_report",
        description="Autopilot ingestion: fetch papers for answer-gap entities.",
    )
    parser.add_argument(
        "--gap-report",
        default=str(_DEFAULT_GAP_REPORT),
        help="Path to answer_gap_report.json (default: output/answer_gap_report.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="K",
        help="Number of top gap entities to process (default: 10)",
    )
    parser.add_argument(
        "--papers-per-entity",
        type=int,
        default=5,
        metavar="N",
        help="Max OpenAlex results per entity (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="data/fulltexts/",
        help="Output directory for PDFs (default: data/fulltexts/)",
    )
    parser.add_argument(
        "--catalogue",
        default=str(_DEFAULT_CATALOGUE),
        help="Path to enriched catalogue.json (default: data/fulltexts/catalogue_enriched.json)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Contact email for OpenAlex polite pool (or set OPENALEX_EMAIL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be downloaded without writing to disk",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    result = run(
        gap_report=Path(args.gap_report),
        top_k=args.top,
        papers_per_entity=args.papers_per_entity,
        output_dir=Path(args.output),
        catalogue_path=Path(args.catalogue),
        email=args.email,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
