"""Fusion KG → KnowledgeGraphBuilder bridge script.

Prepares inputs for KnowledgeGraphBuilder from the KG-explorer DAQ output,
then launches the KGB quickstart pipeline.

What this does
--------------
1. Regenerates the fusion OWL ontology (ensures it reflects the current KG).
2. Copies downloaded PDFs from data/fulltexts/pdfs/ into kgbuilder/data/fusion-docs/
   (symlinks on Linux; copies on Windows).
3. Uploads the ontology to Fuseki (creates the 'fusion-ontology' dataset).
4. Runs the KGB quickstart pipeline with the fusion profile.

Prerequisites
-------------
- docker compose up -d  (starts Neo4j, Qdrant, Fuseki)
- KnowledgeGraphBuilder cloned at ../KnowledgeGraphBuilder
- ollama serve is running with gemma4:e2b and nomic-embed-text pulled:
    ollama pull gemma4:e2b
    ollama pull nomic-embed-text

Usage
-----
    cd D:\\Documents\\KG-explorer
    .venv\\Scripts\\python.exe scripts/run_kgbuilder.py
    .venv\\Scripts\\python.exe scripts/run_kgbuilder.py --dry-run
    .venv\\Scripts\\python.exe scripts/run_kgbuilder.py --max-iterations 1 --skip-ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KGB_ROOT = PROJECT_ROOT.parent / "KnowledgeGraphBuilder"
ONTOLOGY_TTL = PROJECT_ROOT / "output" / "fusion_ontology.ttl"
ONTOLOGY_OWL = PROJECT_ROOT / "output" / "fusion_ontology.owl"
PDF_SOURCE = PROJECT_ROOT / "data" / "fulltexts" / "pdfs"
KGB_DOCS = PROJECT_ROOT / "kgbuilder" / "data" / "fusion-docs"
KGB_ONTOLOGY = PROJECT_ROOT / "kgbuilder" / "data" / "ontology" / "fusion.ttl"
CQS_FILE = PROJECT_ROOT / "kgbuilder" / "data" / "fusion_competency_questions.txt"
PROFILE = PROJECT_ROOT / "kgbuilder" / "data" / "profiles" / "fusion-energy.json"

# Fuseki endpoint
FUSEKI_URL = os.getenv("FUSEKI_URL", "http://localhost:3030")
FUSEKI_DATASET = "fusion-ontology"
FUSEKI_ADMIN_PW = os.getenv("FUSEKI_ADMIN_PASSWORD", "fusion2026")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _print(msg: str, prefix: str = "►") -> None:
    print(f"\n{prefix} {msg}")


def step_regenerate_ontology() -> bool:
    """Regenerate the OWL ontology from the current Neo4j graph."""
    _print("Regenerating fusion ontology from Neo4j …")
    result = subprocess.run(
        [sys.executable, "-m", "analysis.ontology_generator"],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        print("  [WARN] Ontology generation failed — using existing file if present.")
        if not ONTOLOGY_TTL.exists() and not ONTOLOGY_OWL.exists():
            print("  [ERROR] No ontology file found. Run analysis/ontology_generator.py first.")
            return False
    return True


def step_copy_pdfs() -> int:
    """Copy PDFs from DAQ output into the KGB docs directory."""
    _print("Copying PDFs to kgbuilder/data/fusion-docs/ …")
    KGB_DOCS.mkdir(parents=True, exist_ok=True)

    if not PDF_SOURCE.exists():
        print(f"  [WARN] PDF source directory not found: {PDF_SOURCE}")
        print("  Run the DAQ pipeline first: .venv\\Scripts\\python.exe -m daq --limit 500")
        return 0

    copied = 0
    for pdf in PDF_SOURCE.glob("*.pdf"):
        dest = KGB_DOCS / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)
            copied += 1

    total = len(list(PDF_SOURCE.glob("*.pdf")))
    print(f"  Copied {copied} new PDFs ({total} total in source)")
    return total


def step_prepare_ontology() -> Path | None:
    """Copy the ontology TTL to the KGB ontology directory."""
    _print("Preparing ontology for KGB …")
    KGB_ONTOLOGY.parent.mkdir(parents=True, exist_ok=True)

    src = ONTOLOGY_TTL if ONTOLOGY_TTL.exists() else ONTOLOGY_OWL
    if not src.exists():
        print(f"  [ERROR] Ontology not found at {ONTOLOGY_TTL} or {ONTOLOGY_OWL}")
        return None

    shutil.copy2(src, KGB_ONTOLOGY)
    print(f"  Copied {src.name} → {KGB_ONTOLOGY}")
    return KGB_ONTOLOGY


def step_upload_to_fuseki(ontology_path: Path) -> bool:
    """Create Fuseki dataset and upload the ontology."""
    import requests
    _print(f"Uploading ontology to Fuseki ({FUSEKI_URL}) …")

    try:
        # Create dataset if it doesn't exist
        admin_url = f"{FUSEKI_URL}/$/datasets"
        resp = requests.post(
            admin_url,
            data={"dbName": FUSEKI_DATASET, "dbType": "tdb2"},
            auth=("admin", FUSEKI_ADMIN_PW),
            timeout=15,
        )
        if resp.status_code in (200, 409):  # 409 = already exists
            print(f"  Dataset '{FUSEKI_DATASET}' ready.")
        else:
            print(f"  [WARN] Dataset creation returned {resp.status_code}: {resp.text[:200]}")

        # Upload ontology
        with open(ontology_path, "rb") as f:
            content_type = "text/turtle" if ontology_path.suffix == ".ttl" else "application/rdf+xml"
            upload_url = f"{FUSEKI_URL}/{FUSEKI_DATASET}/data"
            resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": content_type},
                auth=("admin", FUSEKI_ADMIN_PW),
                timeout=30,
            )
        if resp.status_code in (200, 201, 204):
            print(f"  Ontology uploaded successfully.")
            return True
        else:
            print(f"  [WARN] Upload returned {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as exc:
        print(f"  [WARN] Fuseki upload failed: {exc}")
        print("  Continuing — KGB will upload the ontology itself via quickstart.py.")
        return False


def step_run_kgbuilder(
    docs_dir: Path,
    ontology_path: Path,
    max_iterations: int,
    dry_run: bool,
    skip_ingest: bool,
    skip_validation: bool,
) -> int:
    """Run KGB quickstart.py with the fusion inputs."""
    _print("Launching KnowledgeGraphBuilder …")

    # Prefer KGB from its own repo; fall back to pip-installed kgbuilder
    kgb_script = KGB_ROOT / "scripts" / "quickstart.py"
    if not kgb_script.exists():
        # Try pip-installed entry point
        kgb_script_str = "kgbuilder"
        use_module = False
        _print(f"KGB repo not found at {KGB_ROOT}. Trying installed 'kgbuilder' command.")
    else:
        kgb_script_str = str(kgb_script)
        use_module = True

    # Use KGB's own venv if it exists (Python 3.11+); fall back to current interpreter
    kgb_python = KGB_ROOT / ".venv" / "Scripts" / "python.exe"
    python_exe = str(kgb_python) if kgb_python.exists() else sys.executable
    cmd = [python_exe, kgb_script_str] if use_module else ["kgbuilder", "quickstart"]
    cmd += [
        "--ontology", str(ontology_path),
        "--documents", str(docs_dir),
        "--cqs", str(CQS_FILE),
        "--max-iterations", str(max_iterations),
        "--confidence-threshold", "0.5",
        "--top-k", "10",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if skip_ingest:
        cmd.append("--skip-ingest")
    if skip_validation:
        cmd.append("--skip-validation")

    env = os.environ.copy()
    env.update({
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "fusion2026",
        "QDRANT_URL": "http://localhost:6333",
        "FUSEKI_URL": FUSEKI_URL,
        "FUSEKI_USER": "admin",
        "FUSEKI_PASSWORD": "fusion2026",
        "OLLAMA_URL": "http://localhost:11434",
    })
    if use_module:
        env["PYTHONPATH"] = str(KGB_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Docs   : {docs_dir}  ({len(list(docs_dir.glob('*.pdf')))} PDFs)")
    print(f"  Onto   : {ontology_path}")
    print(f"  CQs    : {CQS_FILE}  ({sum(1 for l in CQS_FILE.read_text().splitlines() if l.strip() and not l.startswith('#'))} questions)")

    if dry_run:
        print("\n  [dry-run] KGB would be launched with the above command.")
        return 0

    result = subprocess.run(cmd, env=env, cwd=KGB_ROOT if use_module else PROJECT_ROOT)
    return result.returncode


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="run_kgbuilder",
        description="Prepare fusion inputs and run KnowledgeGraphBuilder.",
    )
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="KGB discovery loop iterations (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without launching KGB")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip document ingestion (docs already in Qdrant)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip SHACL validation step")
    parser.add_argument("--skip-ontology-regen", action="store_true",
                        help="Use existing ontology file without regenerating")
    parser.add_argument("--no-fuseki-upload", action="store_true",
                        help="Skip Fuseki upload (KGB will do it itself)")
    args = parser.parse_args(argv)

    print("=" * 65)
    print("  Fusion KG → KnowledgeGraphBuilder bridge")
    print("=" * 65)

    # Step 1: Ontology
    if not args.skip_ontology_regen:
        step_regenerate_ontology()

    onto_path = step_prepare_ontology()
    if onto_path is None:
        sys.exit(1)

    # Step 2: PDFs
    n_docs = step_copy_pdfs()
    if n_docs == 0:
        print("\n  [WARN] No PDFs found. KGB will run on an empty docs dir.")
        print("  Run: .venv\\Scripts\\python.exe -m daq --limit 200 --email your@email.com")

    # Step 3: Fuseki
    if not args.no_fuseki_upload:
        step_upload_to_fuseki(onto_path)

    # Step 4: KGB
    rc = step_run_kgbuilder(
        docs_dir=KGB_DOCS,
        ontology_path=onto_path,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        skip_ingest=args.skip_ingest,
        skip_validation=args.skip_validation,
    )

    print("\n" + "=" * 65)
    if rc == 0:
        print("  KGB pipeline complete.")
        print(f"  Results in: output/kgbuilder_output/")
        print(f"  Neo4j: MATCH (n {{graph_type:'fusion-energy'}}) RETURN n LIMIT 100")
        print(f"  Qdrant dashboard: http://localhost:6333/dashboard")
        print(f"  Fuseki: http://localhost:3030")
    else:
        print(f"  KGB exited with code {rc}.")
        print("  Check output above for errors.")
    print("=" * 65)


if __name__ == "__main__":
    main()
