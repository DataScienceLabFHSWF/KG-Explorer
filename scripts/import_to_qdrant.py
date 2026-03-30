"""
Import NER JSON data into a Qdrant collection
==============================================
Vector  : embedding of the paper's abstract (sentence-transformers)
Payload : title, abstract, fields_of_study, scholarly_citations_count

Usage:
    python scripts/import_to_qdrant.py
    python scripts/import_to_qdrant.py --data-dir data --collection fusion_papers
    python scripts/import_to_qdrant.py --qdrant-url http://localhost:6333 --recreate

Environment variables:
    QDRANT_URL      default: http://localhost:6333
    QDRANT_API_KEY  optional (Qdrant Cloud)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import fusion papers into Qdrant")
    p.add_argument("--data-dir",       default="data",
                   help="Folder with *_NER.json files (default: data)")
    p.add_argument("--collection",     default="fusion_papers",
                   help="Qdrant collection name (default: fusion_papers)")
    p.add_argument("--qdrant-url",     default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    p.add_argument("--model",          default="all-MiniLM-L6-v2",
                   help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    p.add_argument("--batch-size",     type=int, default=64)
    p.add_argument("--recreate",       action="store_true",
                   help="Drop and recreate the collection if it exists")
    return p.parse_args()


def load_papers(data_dir: Path) -> list[dict]:
    files = sorted(data_dir.glob("*_NER.json"))
    if not files:
        sys.exit(f"[ERROR] No *_NER.json files found in {data_dir}")

    papers: list[dict] = []
    seen: set[str] = set()

    for path in files:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)

        count = 0
        for entry in doc.get("data", []):
            abstract = (entry.get("abstract") or "").strip()
            if not abstract:
                continue
            title = (entry.get("title") or "").strip()
            key = title.lower() or abstract[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            papers.append({
                "title":                     title,
                "abstract":                  abstract,
                "fields_of_study":           entry.get("fields_of_study") or [],
                "scholarly_citations_count": entry.get("scholarly_citations_count") or 0,
            })
            count += 1
        print(f"  {path.name}: {count} papers loaded")

    return papers


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir  = (repo_root / args.data_dir).resolve()

    print(f"\n=== Fusion KG → Qdrant importer ===")
    print(f"  data dir   : {data_dir}")
    print(f"  collection : {args.collection}")
    print(f"  qdrant url : {args.qdrant_url}")
    print(f"  embed model: {args.model}\n")

    # 1. Load
    print("Loading papers…")
    papers = load_papers(data_dir)
    print(f"  → {len(papers)} unique papers\n")

    # 2. Embed abstracts
    print(f"Loading embedding model '{args.model}'…")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("[ERROR] Run: pip install sentence-transformers")

    encoder = SentenceTransformer(args.model)
    abstracts = [p["abstract"] for p in papers]

    print(f"Embedding {len(abstracts)} abstracts…")
    embeddings = encoder.encode(
        abstracts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vector_dim = embeddings.shape[1]
    print(f"  → shape {embeddings.shape}\n")

    # 3. Connect to Qdrant
    print(f"Connecting to Qdrant at {args.qdrant_url}…")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams
    except ImportError:
        sys.exit("[ERROR] Run: pip install qdrant-client")

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    # 4. Create / recreate collection
    existing = {c.name for c in client.get_collections().collections}
    if args.collection in existing:
        if args.recreate:
            print(f"  dropping existing collection '{args.collection}'")
            client.delete_collection(args.collection)
        else:
            print(f"  collection '{args.collection}' already exists (use --recreate to reset)")
    if args.collection not in existing or args.recreate:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"  created '{args.collection}' (dim={vector_dim}, cosine)\n")

    # 5. Upload in batches
    print(f"Uploading {len(papers)} points…")
    bs = args.batch_size
    for start in tqdm(range(0, len(papers), bs), unit="batch"):
        batch_papers = papers[start : start + bs]
        batch_vecs   = embeddings[start : start + bs]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "title":                     p["title"],
                    "abstract":                  p["abstract"],
                    "fields_of_study":           p["fields_of_study"],
                    "scholarly_citations_count": p["scholarly_citations_count"],
                },
            )
            for p, vec in zip(batch_papers, batch_vecs)
        ]
        client.upsert(collection_name=args.collection, points=points)

    print(f"\nDone. '{args.collection}' now holds {len(papers)} vectors.")


if __name__ == "__main__":
    main()
