"""Prepare downloaded papers for consumption by KnowledgeGraphBuilder.

The KGBuilder CLI expects a directory of documents plus an OWL ontology::

    kgbuilder run --docs ./fulltext_docs/ --ontology ./ontology.ttl

This module:
1. Creates a ``manifest.json`` mapping each PDF to its Neo4j ``paper_id``,
   so the full-text KG nodes can be linked back to the existing abstract-based
   KG.
2. Optionally generates a minimal OWL ontology stub from the 28 NER categories
   (suitable for bootstrapping KGBuilder extractions on the fusion domain).
3. Provides a helper that copies/symlinks downloaded PDFs into the final
   output directory with clean filenames.

Integration modes
-----------------
**Mode A — Standalone (this project only)**:
    Use the ``prepare_kgbuilder_input()`` function to produce a docs directory
    and manifest that documents how each PDF maps to the existing KG.
    Then manually run ``kgbuilder run --docs ...`` from the KGPlatform repo.

**Mode B — Library import (KGBuilder calls us)**:
    The DAQ package can be imported by KGBuilder as a data-acquisition plugin.
    A future ``DataAcquisitionPlugin`` protocol in KGBuilder could call
    ``daq.pipeline.run()`` to fetch documents before extraction begins.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from daq.doi_extraction import PaperRecord


# ── Manifest generation ──────────────────────────────────────────────────────

def build_manifest(
    downloaded: list[tuple[PaperRecord, Path]],
) -> list[dict[str, Any]]:
    """Build a manifest mapping each downloaded PDF to paper metadata.

    Each entry contains the filename on disk plus all available metadata
    so that KGBuilder (or any downstream consumer) can link extracted
    entities back to their source papers in the fusion KG.
    """
    manifest = []
    for record, pdf_path in downloaded:
        entry = {
            "filename": pdf_path.name,
            "paper_id": record.paper_id,
            "doi": record.doi,
            "title": record.title,
            "year_published": record.year_published,
            "first_author": record.first_author,
            "openalex_id": record.openalex_id,
            "oa_status": record.oa_status,
            "license": record.license,
            "source_file": record.source_file,
            "repository_name": record.repository_name,
        }
        manifest.append({k: v for k, v in entry.items() if v is not None})
    return manifest


def save_manifest(manifest: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write the manifest to ``output_dir/manifest.json``."""
    path = output_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return path


# ── OWL stub generation ──────────────────────────────────────────────────────

_OWL_HEADER = """\
<?xml version="1.0"?>
<rdf:RDF xmlns="http://fusionkg.2026/ontology#"
     xml:base="http://fusionkg.2026/ontology"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22/rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:skos="http://www.w3.org/2004/02/skos/core#">
    <owl:Ontology rdf:about="http://fusionkg.2026/ontology">
        <rdfs:label>Fusion Energy NER Ontology (auto-generated stub)</rdfs:label>
        <rdfs:comment>
            Auto-generated from the 28 NER categories of Loreti et al. (2025).
            This stub is intended as a seed ontology for KnowledgeGraphBuilder
            extractions on the nuclear fusion domain.
        </rdfs:comment>
    </owl:Ontology>
"""

_OWL_FOOTER = "</rdf:RDF>\n"


def _class_iri(category_name: str) -> str:
    """Turn a category name into a valid OWL class IRI fragment."""
    return category_name.replace(" ", "").replace("-", "").replace("/", "_")


def generate_owl_stub(
    categories: list[str],
    output_path: Path,
) -> Path:
    """Generate a minimal OWL ontology from NER category names.

    Each category becomes an ``owl:Class`` under a common ``Entity``
    superclass.  This is intentionally simple — it serves as a seed
    for KGBuilder-guided extraction and can be extended with the
    OntologyExtender afterwards.
    """
    lines = [_OWL_HEADER]

    # Superclass
    lines.append('    <owl:Class rdf:about="http://fusionkg.2026/ontology#Entity">')
    lines.append('        <rdfs:label>Entity</rdfs:label>')
    lines.append("    </owl:Class>\n")

    for cat in sorted(categories):
        iri_frag = _class_iri(cat)
        lines.append(f'    <owl:Class rdf:about="http://fusionkg.2026/ontology#{iri_frag}">')
        lines.append(f'        <rdfs:subClassOf rdf:resource="http://fusionkg.2026/ontology#Entity"/>')
        lines.append(f'        <rdfs:label>{cat}</rdfs:label>')
        lines.append("    </owl:Class>\n")

    lines.append(_OWL_FOOTER)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── Prepare directory ─────────────────────────────────────────────────────────

def prepare_kgbuilder_input(
    downloaded: list[tuple[PaperRecord, Path]],
    output_dir: Path,
    categories: list[str] | None = None,
) -> dict[str, Path]:
    """Package downloaded PDFs + metadata for KGBuilder consumption.

    Parameters
    ----------
    downloaded
        List of (record, pdf_path) as returned by ``PaperDownloader.download_batch``.
    output_dir
        Target directory (will be created if absent).
    categories
        If provided, generates a seed OWL ontology from these NER categories.

    Returns
    -------
    dict with keys ``docs_dir``, ``manifest``, and optionally ``ontology``,
    pointing to the created paths.
    """
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDFs into the docs directory
    moved: list[tuple[PaperRecord, Path]] = []
    for record, src_path in downloaded:
        dest = docs_dir / src_path.name
        if not dest.exists():
            shutil.copy2(src_path, dest)
        moved.append((record, dest))

    # Write manifest
    manifest = build_manifest(moved)
    manifest_path = save_manifest(manifest, output_dir)

    result: dict[str, Path] = {
        "docs_dir": docs_dir,
        "manifest": manifest_path,
    }

    # OWL stub
    if categories:
        owl_path = output_dir / "ontology" / "fusion_ner.owl"
        generate_owl_stub(categories, owl_path)
        result["ontology"] = owl_path

    return result
