"""Utility to propose entity normalisations based on link prediction output.

Reads ``output/predicted_links.csv`` and flags pairs of entities where one
appears to be a simple plural of the other. These candidates can then be
manually reviewed and merged (e.g. "tokamak" <-> "tokamaks").

Usage::

    python scripts/suggest_entity_merges.py [--output FILE]

The script prints suggestions to stdout and optionally writes them to a
CSV file.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def is_plural_pair(a: str, b: str) -> bool:
    a = a.lower().strip()
    b = b.lower().strip()
    # naive rules: add/remove s or es
    if a + "s" == b or a + "es" == b:
        return True
    if b + "s" == a or b + "es" == a:
        return True
    # handle words ending in "ies" -> "y" (e.g. "densities"/"density")
    if a.endswith("ies") and a[:-3] + "y" == b:
        return True
    if b.endswith("ies") and b[:-3] + "y" == a:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Suggest singular/plural entity merges")
    parser.add_argument(
        "--output",
        help="Optional CSV file to write suggestions",
        default=None,
    )
    args = parser.parse_args()

    path = Path("output/predicted_links.csv")
    if not path.exists():
        print(f"Error: {path} not found. Run link_prediction first.")
        return

    df = pd.read_csv(path)
    suggestions = []
    for _, row in df.iterrows():
        a = str(row.get("entity_a", "")).strip()
        b = str(row.get("entity_b", "")).strip()
        if not a or not b:
            continue
        if is_plural_pair(a, b):
            suggestions.append({"singular": a if not a.endswith("s") else b,
                                "plural": b if b.endswith("s") else a})

    if not suggestions:
        print("No obvious singular/plural pairs found in predictions.")
    else:
        print("Suggested singular/plural pairings:")
        for s in suggestions:
            print(f"  {s['singular']} <-> {s['plural']}")
        if args.output:
            out = Path(args.output)
            pd.DataFrame(suggestions).to_csv(out, index=False)
            print(f"Wrote {len(suggestions)} suggestions to {out}")


if __name__ == "__main__":
    main()
