"""
Open Access Landscape Analysis for the Fusion Corpus
=====================================================
Enriches the full paper catalogue via OpenAlex (metadata only, no downloads)
and produces an OA landscape report with visualisations.

This provides evidence for the open-research argument: how much
fusion research is openly accessible vs. locked behind paywalls,
and which repositories (arXiv, OSTI, Zenodo, etc.) provide alternative
access to closed-publisher content.

Usage
-----
    python scripts/oa_landscape.py                       # analyse all
    python scripts/oa_landscape.py --limit 500           # test batch
    python scripts/oa_landscape.py --catalogue data/fulltexts/catalogue_enriched.json

Outputs (in output/)
--------------------
  - 17_oa_status_breakdown.png      — pie/bar chart of OA status
  - 18_oa_by_year.png               — OA rate trends over time
  - 19_oa_repositories.png          — top repository sources for preprints
  - 20_oa_publisher_domains.png     — OA rate by publisher domain
  - oa_landscape_report.json        — full statistics
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np

# Resolve project root so we can import daq/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.neo4j_utils import OUTPUT_DIR, save_figure
from daq.doi_extraction import build_catalogue, load_catalogue, save_catalogue, PaperRecord
from daq.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _domain(url: str | None) -> str:
    """Extract the main domain from a URL."""
    if not url:
        return "unknown"
    try:
        host = urlparse(url).hostname or "unknown"
        # Strip common prefixes
        for prefix in ("www.", "www2.", "www3."):
            if host.startswith(prefix):
                host = host[len(prefix):]
        return host
    except Exception:
        return "unknown"


def compute_oa_statistics(records: list[PaperRecord]) -> dict:
    """Compute comprehensive OA statistics from enriched records."""
    total = len(records)
    enriched = [r for r in records if r.oa_status and r.oa_status != "unknown"]

    # Status breakdown
    status_counts = Counter(r.oa_status or "unknown" for r in records)

    # Year trends
    year_oa: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.year_published and 1950 <= r.year_published <= 2026:
            year_oa[r.year_published][r.oa_status or "unknown"] += 1

    # Repository analysis
    repo_counts = Counter()
    for r in records:
        if r.repository_name:
            # Simplify name
            name = r.repository_name
            if "arxiv" in name.lower():
                name = "arXiv"
            elif "osti" in name.lower():
                name = "OSTI"
            elif "zenodo" in name.lower():
                name = "Zenodo"
            elif "pubmed" in name.lower():
                name = "PubMed Central"
            elif "hal" in name.lower():
                name = "HAL"
            repo_counts[name] += 1

    # Publisher domain breakdown
    domain_oa: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        dom = _domain(r.url)
        domain_oa[dom][r.oa_status or "unknown"] += 1

    # Compute accessibility metric
    accessible = sum(1 for r in records
                     if r.oa_status not in ("closed", "unknown", None)
                     or r.repository_pdf_url)
    truly_closed = sum(1 for r in records
                       if r.oa_status == "closed" and not r.repository_pdf_url)

    return {
        "total_papers": total,
        "enriched": len(enriched),
        "status_breakdown": dict(status_counts.most_common()),
        "accessible_count": accessible,
        "truly_closed_count": truly_closed,
        "accessibility_rate": accessible / total if total else 0,
        "year_trends": {str(y): dict(counts) for y, counts in sorted(year_oa.items())},
        "repository_counts": dict(repo_counts.most_common()),
        "top_domains": {
            dom: dict(counts)
            for dom, counts in sorted(domain_oa.items(),
                                      key=lambda x: sum(x[1].values()),
                                      reverse=True)[:20]
        },
    }


# ── Visualisations ────────────────────────────────────────────────────────────

_OA_COLORS = {
    "gold": "#FFD700",
    "green": "#2ca02c",
    "bronze": "#CD7F32",
    "hybrid": "#9467bd",
    "diamond": "#00CED1",
    "closed": "#d62728",
    "unknown": "#999999",
}


def plot_status_breakdown(stats: dict):
    """Pie + bar chart of OA status distribution."""
    breakdown = stats["status_breakdown"]
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = [_OA_COLORS.get(l, "#888") for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title("OA Status Distribution", fontsize=13)

    # Bar
    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, values, color=colors, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Number of papers", fontsize=11)
    ax2.set_title("OA Status Counts", fontsize=13)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    # Summary text
    rate = stats["accessibility_rate"]
    fig.suptitle(
        f"Fusion Energy Literature — Open Access Landscape "
        f"({stats['total_papers']:,} papers, {rate:.1%} accessible)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_oa_by_year(stats: dict):
    """Stacked area chart of OA status trends by publication year."""
    trends = stats["year_trends"]
    years = sorted(int(y) for y in trends.keys())
    # Filter to a reasonable range
    years = [y for y in years if 1970 <= y <= 2025]
    if not years:
        return None

    statuses = ["gold", "diamond", "hybrid", "green", "bronze", "closed", "unknown"]
    data = {s: [] for s in statuses}
    for y in years:
        counts = trends.get(str(y), {})
        for s in statuses:
            data[s].append(counts.get(s, 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Stacked area: absolute
    arrays = [np.array(data[s]) for s in statuses]
    colors = [_OA_COLORS[s] for s in statuses]
    ax1.stackplot(years, *arrays, labels=statuses, colors=colors, alpha=0.8)
    ax1.set_ylabel("Number of papers", fontsize=11)
    ax1.set_title("OA Status by Publication Year (absolute)", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Line: OA rate (% not closed)
    totals = np.sum(arrays, axis=0)
    closed = np.array(data["closed"]) + np.array(data["unknown"])
    oa_rate = np.where(totals > 0, 1 - closed / totals, 0) * 100

    ax2.plot(years, oa_rate, "g-", linewidth=2, label="OA rate")
    ax2.fill_between(years, oa_rate, alpha=0.2, color="green")
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("OA rate (%)", fontsize=11)
    ax2.set_title("Open Access Rate Over Time", fontsize=13)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    return fig


def plot_repositories(stats: dict):
    """Bar chart of repository sources for preprints."""
    repos = stats["repository_counts"]
    if not repos:
        return None

    names = list(repos.keys())
    counts = list(repos.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, counts, color="steelblue", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of papers with repository copy", fontsize=11)
    ax.set_title("Repository Sources for Green OA / Preprints", fontsize=13)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_publisher_domains(stats: dict):
    """OA rate by publisher domain."""
    domains = stats["top_domains"]
    if not domains:
        return None

    dom_names = []
    dom_oa_rates = []
    dom_totals = []
    for dom, counts in domains.items():
        total = sum(counts.values())
        if total < 5:
            continue
        closed = counts.get("closed", 0) + counts.get("unknown", 0)
        rate = (1 - closed / total) * 100 if total else 0
        dom_names.append(dom)
        dom_oa_rates.append(rate)
        dom_totals.append(total)

    if not dom_names:
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(dom_names) * 0.4)))
    y_pos = np.arange(len(dom_names))
    colors = ["#2ca02c" if r > 50 else "#d62728" for r in dom_oa_rates]
    bars = ax.barh(y_pos, dom_oa_rates, color=colors, height=0.6, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n}  (n={t})" for n, t in zip(dom_names, dom_totals)],
                       fontsize=9)
    ax.set_xlabel("OA rate (%)", fontsize=11)
    ax.set_title("Open Access Rate by Publisher Domain", fontsize=13)
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, dom_oa_rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="OA Landscape Analysis")
    parser.add_argument("--data-dir", default="data/",
                        help="NER JSON directory")
    parser.add_argument("--catalogue", default=None,
                        help="Pre-enriched catalogue JSON to reuse")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to enrich (default: all)")
    parser.add_argument("--email", default="fusiondaq@example.com",
                        help="Email for OpenAlex polite pool")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load or build catalogue
    if args.catalogue and Path(args.catalogue).exists():
        print(f"Loading catalogue from {args.catalogue} …")
        records = load_catalogue(Path(args.catalogue))
    else:
        print("Building catalogue from NER JSONs …")
        records = build_catalogue(Path(args.data_dir))

    if args.limit:
        records = records[:args.limit]

    # Enrich if not already done
    not_enriched = [r for r in records if r.oa_status is None]
    if not_enriched:
        print(f"Enriching {len(not_enriched)} records via OpenAlex …")
        client = OpenAlexClient(email=args.email)
        for rec in tqdm(not_enriched, desc="OpenAlex", unit="paper"):
            try:
                client.enrich(rec)
            except Exception:
                pass

        # Save enriched catalogue
        out_path = Path(args.data_dir) / "fulltexts" / "catalogue_enriched.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_catalogue(records, out_path)
        print(f"Saved enriched catalogue to {out_path}")

    # Compute statistics
    print("Computing OA statistics …")
    stats = compute_oa_statistics(records)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  OA LANDSCAPE REPORT — {stats['total_papers']:,} papers")
    print(f"{'=' * 60}")
    print(f"  Enriched          : {stats['enriched']:,}")
    print(f"  Accessible (OA+repo): {stats['accessible_count']:,} "
          f"({stats['accessibility_rate']:.1%})")
    print(f"  Truly closed      : {stats['truly_closed_count']:,}")
    print(f"\n  Status breakdown:")
    for status, count in stats["status_breakdown"].items():
        pct = count / stats["total_papers"] * 100
        print(f"    {status:12s} : {count:5d}  ({pct:.1f}%)")
    if stats["repository_counts"]:
        print(f"\n  Repository preprints:")
        for repo, count in stats["repository_counts"].items():
            print(f"    {repo:30s} : {count}")

    # Generate plots
    print("\nGenerating visualisations …")
    fig1 = plot_status_breakdown(stats)
    save_figure(fig1, "17_oa_status_breakdown")
    plt.close(fig1)

    fig2 = plot_oa_by_year(stats)
    if fig2:
        save_figure(fig2, "18_oa_by_year")
        plt.close(fig2)

    fig3 = plot_repositories(stats)
    if fig3:
        save_figure(fig3, "19_oa_repositories")
        plt.close(fig3)

    fig4 = plot_publisher_domains(stats)
    if fig4:
        save_figure(fig4, "20_oa_publisher_domains")
        plt.close(fig4)

    # Save report
    report_path = OUTPUT_DIR / "oa_landscape_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {report_path}")


if __name__ == "__main__":
    main()
