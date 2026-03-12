"""Data Acquisition (DAQ) package for scientific paper full-text retrieval.

This package provides a domain-agnostic pipeline for resolving paper metadata
via OpenAlex, checking open-access availability, and downloading full texts.

The output is structured to be directly consumable by the KnowledgeGraphBuilder
pipeline (``kgbuilder run --docs ./output_dir/``).

Modules
-------
doi_extraction
    Extract DOIs and titles from the existing NER JSON data files.
openalex_client
    Resolve metadata (DOI, OA status, PDF URLs) via the OpenAlex API.
downloader
    Rate-limited, polite PDF/HTML downloader with retry logic.
pipeline
    Orchestrator that chains extraction → resolution → download.
kgbuilder_bridge
    Package downloaded papers for KnowledgeGraphBuilder consumption.
"""

from daq.doi_extraction import PaperRecord, build_catalogue
from daq.openalex_client import OpenAlexClient
from daq.downloader import PaperDownloader
from daq.pipeline import DAQPipeline

__all__ = [
    "PaperRecord",
    "build_catalogue",
    "OpenAlexClient",
    "PaperDownloader",
    "DAQPipeline",
]
