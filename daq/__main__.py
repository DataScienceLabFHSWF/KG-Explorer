"""CLI entry-point for the Data Acquisition pipeline.

Run with::

    python -m daq --data-dir data/ --output data/fulltexts/ --limit 5 --email you@uni.de

Add ``-v`` for debug-level logging.
"""

from __future__ import annotations

import argparse
import logging
import sys

from daq.pipeline import DAQPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="daq",
        description="Full-text Data Acquisition pipeline for scholarly papers.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing NER JSON files (default: data/)",
    )
    parser.add_argument(
        "--output",
        default="data/fulltexts/",
        help="Output directory for PDFs and metadata (default: data/fulltexts/)",
    )
    parser.add_argument(
        "--kgbuilder-output",
        default="output/kgbuilder_input/",
        help="Output directory for KGBuilder-ready docs (default: output/kgbuilder_input/)",
    )
    parser.add_argument(
        "--email",
        default="fusiondaq@example.com",
        help="Contact email for OpenAlex polite pool",
    )
    parser.add_argument(
        "--catalogue",
        default=None,
        help="Path to a previously saved catalogue.json to reuse",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to process (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    pipe = DAQPipeline(
        data_dir=args.data_dir,
        output_dir=args.output,
        kgbuilder_output_dir=args.kgbuilder_output,
        email=args.email,
        catalogue_path=args.catalogue,
        limit=args.limit,
    )
    stats = pipe.run()
    print(f"\n{stats}")


if __name__ == "__main__":
    main()
