"""End-to-end Data Acquisition pipeline.

Orchestrates:
1. ``doi_extraction``   – scan NER JSONs → PaperRecord catalogue
2. ``openalex_client``  – enrich records with OA metadata
3. ``downloader``       – fetch open-access PDFs
4. ``kgbuilder_bridge`` – package results for KGBuilder

Usage as library
~~~~~~~~~~~~~~~~
>>> from daq.pipeline import DAQPipeline
>>> pipe = DAQPipeline(data_dir="data/", output_dir="data/fulltexts/")
>>> stats = pipe.run(limit=20)
>>> print(stats)

Usage via CLI
~~~~~~~~~~~~~
    python -m daq --data-dir data/ --output data/fulltexts/ --limit 20
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from daq.doi_extraction import PaperRecord, build_catalogue, save_catalogue, load_catalogue
from daq.openalex_client import OpenAlexClient
from daq.downloader import PaperDownloader
from daq.kgbuilder_bridge import prepare_kgbuilder_input

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Summary statistics for a DAQ pipeline run."""
    total_papers: int = 0
    with_doi: int = 0
    enriched: int = 0
    # OA breakdown
    oa_gold: int = 0
    oa_green: int = 0
    oa_bronze: int = 0
    oa_hybrid: int = 0
    oa_diamond: int = 0
    oa_closed: int = 0
    oa_unknown: int = 0
    with_repo_preprint: int = 0   # have arXiv/OSTI/Zenodo version
    downloadable: int = 0         # have any PDF URL (publisher or repo)
    downloaded: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"DAQ Pipeline Stats\n"
            f"  Total papers       : {self.total_papers}\n"
            f"  With DOI           : {self.with_doi}\n"
            f"  Enriched via OA    : {self.enriched}\n"
            f"  -- OA breakdown ---------\n"
            f"    gold             : {self.oa_gold}\n"
            f"    green            : {self.oa_green}\n"
            f"    bronze           : {self.oa_bronze}\n"
            f"    hybrid           : {self.oa_hybrid}\n"
            f"    diamond          : {self.oa_diamond}\n"
            f"    closed           : {self.oa_closed}\n"
            f"    unknown          : {self.oa_unknown}\n"
            f"  With repo preprint : {self.with_repo_preprint}\n"
            f"  Downloadable       : {self.downloadable}\n"
            f"  Downloaded         : {self.downloaded}\n"
            f"  Time               : {self.elapsed_seconds:.1f}s"
        )


@dataclass
class DAQPipeline:
    """Full-text acquisition pipeline for scholarly paper corpora.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing NER JSON files (input).
    output_dir : str | Path
        Base directory for all outputs (catalogue, PDFs, KGBuilder package).
    email : str
        mailto for OpenAlex polite pool (required by their API guidelines).
    catalogue_path : str | Path | None
        If given, load a previously saved catalogue instead of rebuilding.
    limit : int | None
        Cap the number of papers to process (useful for testing).
    categories : list[str] | None
        NER categories to include in the OWL stub.  If ``None``, categories
        are inferred from the NER JSON files.
    """
    data_dir: str | Path = "data/"
    output_dir: str | Path = "data/fulltexts/"
    kgbuilder_output_dir: str | Path = "output/kgbuilder_input/"
    email: str = "fusiondaq@example.com"
    catalogue_path: str | Path | None = None
    limit: int | None = None
    categories: list[str] | None = None

    def run(self) -> PipelineStats:
        """Execute the full acquisition pipeline."""
        t0 = time.time()
        stats = PipelineStats()

        data_dir = Path(self.data_dir)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Build or load catalogue ───────────────────────────
        logger.info("Step 1/4 — Building paper catalogue …")
        if self.catalogue_path and Path(self.catalogue_path).exists():
            catalogue = load_catalogue(Path(self.catalogue_path))
            logger.info("  Loaded %d records from %s", len(catalogue), self.catalogue_path)
        else:
            catalogue = build_catalogue(data_dir)
            cat_path = output_dir / "catalogue.json"
            save_catalogue(catalogue, cat_path)
            logger.info("  Built catalogue: %d unique papers → %s", len(catalogue), cat_path)

        stats.total_papers = len(catalogue)
        stats.with_doi = sum(1 for r in catalogue if r.doi)

        # Apply limit
        records = catalogue[: self.limit] if self.limit else catalogue

        # Collect categories from NER JSONs (for OWL stub)
        if self.categories is None:
            cats: set[str] = set()
            for ner_file in data_dir.glob("*_NER.json"):
                try:
                    payload = json.load(open(ner_file, "r", encoding="utf-8"))
                    for entry in payload.get("data", []):
                        fos = entry.get("fields_of_study")
                        if isinstance(fos, list):
                            cats.update(fos)
                except Exception:
                    pass
            self.categories = sorted(cats) if cats else None

        # ── Step 2: Enrich via OpenAlex ───────────────────────────────
        logger.info("Step 2/4 — Enriching %d records via OpenAlex …", len(records))
        client = OpenAlexClient(email=self.email)
        for rec in tqdm(records, desc="OpenAlex enrichment", unit="paper"):
            try:
                client.enrich(rec)
                stats.enriched += 1
            except Exception:
                logger.debug("  OpenAlex failed for %s", rec.paper_id, exc_info=True)

        stats.oa_found = sum(1 for r in records if r.pdf_url)

        # OA breakdown
        from collections import Counter
        oa_counts = Counter(r.oa_status or "unknown" for r in records)
        stats.oa_gold = oa_counts.get("gold", 0)
        stats.oa_green = oa_counts.get("green", 0)
        stats.oa_bronze = oa_counts.get("bronze", 0)
        stats.oa_hybrid = oa_counts.get("hybrid", 0)
        stats.oa_diamond = oa_counts.get("diamond", 0)
        stats.oa_closed = oa_counts.get("closed", 0)
        stats.oa_unknown = oa_counts.get("unknown", 0)
        stats.with_repo_preprint = sum(1 for r in records if r.repository_pdf_url)
        stats.downloadable = sum(
            1 for r in records
            if r.pdf_url or r.repository_pdf_url
        )

        # Save enriched catalogue
        enriched_path = output_dir / "catalogue_enriched.json"
        save_catalogue(records, enriched_path)
        logger.info(
            "  Enriched: %d/%d | OA: gold=%d green=%d bronze=%d hybrid=%d diamond=%d | closed=%d | repo preprints=%d",
            stats.enriched, len(records),
            stats.oa_gold, stats.oa_green, stats.oa_bronze, stats.oa_hybrid, stats.oa_diamond,
            stats.oa_closed, stats.with_repo_preprint,
        )

        # ── Step 3: Download PDFs ─────────────────────────────────────
        logger.info("Step 3/4 \u2014 Downloading open-access + preprint PDFs \u2026")
        dl = PaperDownloader(output_dir=output_dir / "pdfs")
        downloaded = dl.download_batch(records)
        stats.downloaded = len(downloaded)
        logger.info("  Downloaded %d PDFs", stats.downloaded)

        # ── Step 4: Package for KGBuilder ─────────────────────────────
        logger.info("Step 4/4 — Packaging for KGBuilder …")
        kgb_dir = Path(self.kgbuilder_output_dir)
        result = prepare_kgbuilder_input(
            downloaded=downloaded,
            output_dir=kgb_dir,
            categories=self.categories,
        )
        logger.info("  Docs dir  : %s", result["docs_dir"])
        logger.info("  Manifest  : %s", result["manifest"])
        if "ontology" in result:
            logger.info("  Ontology  : %s", result["ontology"])
        logger.info(
            "  → Ready:  kgbuilder run --docs %s --ontology %s",
            result["docs_dir"],
            result.get("ontology", "<your-ontology.ttl>"),
        )

        stats.elapsed_seconds = time.time() - t0
        logger.info("\n%s", stats)
        return stats
