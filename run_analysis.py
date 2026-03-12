#!/usr/bin/env python3
"""
run_analysis.py — Fusion KG Mathematical Analysis Orchestrator
===============================================================
Sequentially executes every analysis module and writes outputs to ``output/``.

Usage
-----
    python run_analysis.py                  # run everything
    python run_analysis.py --skip tda       # skip persistent homology
    python run_analysis.py --only graph     # run graph analysis only
    python run_analysis.py --max-nodes 500  # limit TDA subgraph size
"""

import argparse
import sys
import time
from pathlib import Path

# ── make sure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.neo4j_utils import get_driver, OUTPUT_DIR

# --------------- module registry ---------------
MODULES = {
    "graph":       "analysis.graph_analysis",
    "tda":         "analysis.tda_analysis",
    "spectral":    "analysis.spectral_analysis",
    "fca":         "analysis.fca_analysis",
    "information": "analysis.information_theory",
    "voids":       "analysis.void_extraction",
    "holes":       "analysis.structural_holes",
    "links":       "analysis.link_prediction",
    "explorer":    "analysis.interactive_explorer",
    "gaps":        "analysis.gap_analysis_agent",
    "zipf":        "analysis.zipf_analysis",
    "ontology":    "analysis.ontology_generator",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Fusion KG mathematical analysis pipeline.",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        choices=MODULES.keys(),
        default=[],
        help="Analysis modules to skip.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        choices=MODULES.keys(),
        default=None,
        help="Run only the specified modules.",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=800,
        help="Max nodes for TDA subgraph selection (default: 800).",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Restrict the analysis to papers published in this year.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Determine modules to run
    if args.only:
        to_run = {k: v for k, v in MODULES.items() if k in args.only}
    else:
        to_run = {k: v for k, v in MODULES.items() if k not in args.skip}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   Fusion Knowledge Graph -- Mathematical Analysis Suite  ")
    print("=" * 60)
    print(f"\n  Modules to run : {', '.join(to_run.keys())}")
    print(f"  Output directory: {OUTPUT_DIR}\n")

    driver = get_driver()
    results = {}
    total_start = time.time()

    try:
        for key, module_path in to_run.items():
            mod_start = time.time()
            print(f"\n{'-' * 60}")
            print(f"  > Starting module: {key}")
            print(f"{'-' * 60}")

            try:
                mod = __import__(module_path, fromlist=["run"])

                # TDA and void modules accept max_nodes arg
                if key in ("tda", "voids"):
                    result = mod.run(driver, max_nodes=args.max_nodes, year=args.year)
                elif key == "gaps":
                    # Gap agent reads files, doesn't need driver but signature accepts year
                    result = mod.run(year=args.year)
                else:
                    # Most modules accept year keyword
                    result = mod.run(driver, year=args.year)

                results[key] = result
                elapsed = time.time() - mod_start
                print(f"  [OK] {key} completed in {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - mod_start
                print(f"  [FAIL] {key} FAILED after {elapsed:.1f}s -- {e}")
                import traceback
                traceback.print_exc()

    finally:
        driver.close()

    # ── Summary ──
    total_elapsed = time.time() - total_start
    outputs = sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
    pngs = [f for f in outputs if f.suffix == ".png"]
    csvs = [f for f in outputs if f.suffix == ".csv"]
    jsons = [f for f in outputs if f.suffix == ".json"]

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total time     : {total_elapsed:.1f}s")
    print(f"  Modules run    : {len(results)} / {len(to_run)}")
    print(f"  Plots generated: {len(pngs)}")
    print(f"  CSV exports    : {len(csvs)}")
    print(f"  JSON exports   : {len(jsons)}")
    print(f"  Output folder  : {OUTPUT_DIR}")

    if pngs:
        print("\n  Generated plots:")
        for p in pngs:
            print(f"    • {p.name}")

    print()


if __name__ == "__main__":
    main()
