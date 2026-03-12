"""Extract paper identifiers (DOIs, titles, URLs) from NER JSON files.

This module reads the Loreti et al. NER data and produces a catalogue of
papers with their best-available identifiers for downstream resolution.
It is domain-agnostic: any JSON file following the same schema will work.

Output
------
A list of ``PaperRecord`` dataclasses, each carrying:
- paper_id (internal key)
- doi (if extractable from the URL)
- title
- url (original)
- year_published
- first_author
- scholarly_citations_count
- source_file
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class PaperRecord:
    """Minimal metadata for a single paper."""

    paper_id: str
    title: str | None = None
    doi: str | None = None
    url: str | None = None
    year_published: int | None = None
    first_author: str | None = None
    citations: int | None = None
    source_file: str | None = None

    # Populated during resolution
    openalex_id: str | None = None
    oa_status: str | None = None  # "gold", "green", "bronze", "hybrid", "closed"
    pdf_url: str | None = None
    landing_url: str | None = None
    license: str | None = None

    # Green OA / repository fallback (arXiv, OSTI, Zenodo, …)
    repository_pdf_url: str | None = None
    repository_name: str | None = None  # e.g. "arXiv", "OSTI", "Zenodo"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── DOI extraction ────────────────────────────────────────────────────────────

_DOI_RE = re.compile(r"(10\.\d{4,}/[^\s,;\"'>]+)")

# Suffixes commonly appended to DOIs in URLs that are not part of the DOI
_DOI_STRIP_SUFFIXES = re.compile(r"/(pdf|abstract|meta|full|html|summary|epdf)$", re.I)


def extract_doi(url: str) -> str | None:
    """Try to extract a DOI from a URL string.

    Returns the cleaned DOI or ``None`` if no DOI pattern is found.
    """
    if not url:
        return None
    m = _DOI_RE.search(url)
    if not m:
        return None
    doi = m.group(1).rstrip("/").rstrip(".")
    doi = _DOI_STRIP_SUFFIXES.sub("", doi)
    return doi


# ── NER JSON parsing ─────────────────────────────────────────────────────────

def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() == "nan":
        return None
    return s


def parse_ner_json(file_path: Path) -> list[PaperRecord]:
    """Parse a single ``*_NER.json`` file and return paper records."""
    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    data = payload.get("data", [])
    if not isinstance(data, list):
        return []

    records: list[PaperRecord] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue

        paper_id = f"{file_path.name}:{idx}"
        url = _clean_str(entry.get("URL"))
        doi = extract_doi(url) if url else None
        title = _clean_str(entry.get("title"))
        year = entry.get("year_published")
        author = _clean_str(entry.get("first_author"))
        cites = entry.get("scholarly_citations_count")

        records.append(PaperRecord(
            paper_id=paper_id,
            title=title,
            doi=doi,
            url=url,
            year_published=year if isinstance(year, int) else None,
            first_author=author,
            citations=cites if isinstance(cites, int) else None,
            source_file=file_path.name,
        ))

    return records


def build_catalogue(data_dir: Path, max_files: int | None = None) -> list[PaperRecord]:
    """Scan all ``*_NER.json`` files in *data_dir* and build a paper catalogue.

    Parameters
    ----------
    data_dir
        Directory containing the NER JSON files.
    max_files
        Optional cap for testing.

    Returns
    -------
    list[PaperRecord]
        De-duplicated by (doi OR title), keeping the record with the most
        metadata.
    """
    files = sorted(data_dir.glob("*_NER.json"))
    if max_files is not None:
        files = files[:max_files]

    all_records: list[PaperRecord] = []
    for f in files:
        all_records.extend(parse_ner_json(f))

    # De-duplicate: prefer records that have a DOI
    seen_doi: dict[str, PaperRecord] = {}
    seen_title: dict[str, PaperRecord] = {}
    unique: list[PaperRecord] = []

    for rec in all_records:
        if rec.doi:
            key = rec.doi.lower()
            if key in seen_doi:
                continue
            seen_doi[key] = rec
            unique.append(rec)
        elif rec.title:
            key = rec.title.lower()
            if key in seen_title:
                continue
            seen_title[key] = rec
            unique.append(rec)
        else:
            unique.append(rec)

    return unique


def save_catalogue(records: list[PaperRecord], output_path: Path) -> None:
    """Write the catalogue to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2, ensure_ascii=False)


def load_catalogue(path: Path) -> list[PaperRecord]:
    """Load a previously saved catalogue."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    records = []
    for entry in raw:
        records.append(PaperRecord(
            paper_id=entry["paper_id"],
            title=entry.get("title"),
            doi=entry.get("doi"),
            url=entry.get("url"),
            year_published=entry.get("year_published"),
            first_author=entry.get("first_author"),
            citations=entry.get("citations"),
            source_file=entry.get("source_file"),
            openalex_id=entry.get("openalex_id"),
            oa_status=entry.get("oa_status"),
            pdf_url=entry.get("pdf_url"),
            landing_url=entry.get("landing_url"),
            license=entry.get("license"),
        ))
    return records
