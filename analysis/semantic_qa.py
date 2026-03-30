"""
Semantic QA Agent for the Fusion Papers Qdrant collection
==========================================================
Pipeline per question:
  1. Embed the query with sentence-transformers (same model used during import)
  2. Search Qdrant for the most similar paper abstracts
  3. Synthesise a natural-language answer from the retrieved papers via Ollama

The ask() return dict mirrors FusionCypherAgent so chat_app.py can treat
both agents identically.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Answer synthesis prompt ────────────────────────────────────────────────

SYNTHESIS_TEMPLATE = """\
You are a scientific assistant specialising in nuclear fusion research.
Answer the question using ONLY the paper abstracts provided below.
- Cite specific titles or findings from the papers.
- If the retrieved papers are not relevant, say so clearly.
- Structure longer answers with short bullet points.
- Answer in the same language as the question.

Retrieved papers:
{context}

Question: {question}

Answer:"""


class SemanticQAAgent:
    """Embedding-based retrieval + LLM synthesis over the Qdrant fusion_papers collection.

    Parameters
    ----------
    ollama_url   : Ollama server base URL
    model        : Ollama model name used for answer synthesis
    qdrant_url   : Qdrant server URL
    qdrant_api_key: optional API key (Qdrant Cloud)
    collection   : Qdrant collection to search
    embed_model  : sentence-transformers model (must match the one used during import)
    top_k        : number of papers to retrieve per query
    """

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection: str = "fusion_papers",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self._ollama_url  = ollama_url  or os.getenv("OLLAMA_URL",  "http://localhost:11434")
        self._model       = model       or os.getenv("OLLAMA_MODEL", "nemotron:70b")
        self._qdrant_url  = qdrant_url  or os.getenv("QDRANT_URL",  "http://localhost:6333")
        self._qdrant_key  = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self._collection  = collection
        self._top_k       = top_k

        logger.info(
            "SemanticQAAgent init | qdrant=%s  collection=%s  ollama=%s  model=%s",
            self._qdrant_url, collection, self._ollama_url, self._model,
        )

        # Sentence-transformers encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Run: pip install sentence-transformers") from exc

        self._encoder = SentenceTransformer(embed_model)

        # Qdrant client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("Run: pip install qdrant-client") from exc

        self._qdrant = QdrantClient(url=self._qdrant_url, api_key=self._qdrant_key)

        # Ollama LLM (via LangChain)
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("Run: pip install langchain-ollama") from exc

        self._llm = ChatOllama(base_url=self._ollama_url, model=self._model, temperature=0)

    # ── Public ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant papers from Qdrant and synthesise an answer.

        Returns the same dict shape as FusionCypherAgent.ask():
            answer          — LLM-synthesised answer
            cypher          — always "" (not applicable here)
            context         — list of retrieved paper dicts with score
            linked_entities — always [] (not applicable here)
            fallback_used   — always False
            error           — error string or None
        """
        # Step 1: embed query
        try:
            query_vec = self._encoder.encode(
                question,
                normalize_embeddings=True,
            ).tolist()
        except Exception as exc:
            return _error_result(f"Embedding failed: {exc}")

        # Step 2: search Qdrant
        try:
            from qdrant_client.models import NamedVector
            result = self._qdrant.query_points(
                collection_name=self._collection,
                query=query_vec,
                limit=self._top_k,
                with_payload=True,
            )
            hits = result.points
        except Exception as exc:
            return _error_result(f"Qdrant search failed: {exc}")

        if not hits:
            return _error_result(
                f"No results found in collection '{self._collection}'. "
                "Make sure the import script has been run."
            )

        # Build context rows (returned as-is so the UI can render them)
        context = []
        for hit in hits:
            p = hit.payload or {}
            context.append({
                "score":                     round(float(hit.score), 4),
                "title":                     p.get("title", ""),
                "abstract":                  p.get("abstract", ""),
                "fields_of_study":           ", ".join(p.get("fields_of_study") or []),
                "scholarly_citations_count": p.get("scholarly_citations_count", 0),
            })

        # Step 3: synthesise answer
        context_text = "\n\n".join(
            f"[{i+1}] \"{r['title']}\" (score {r['score']})\n{r['abstract'][:600]}"
            for i, r in enumerate(context)
        )
        prompt = SYNTHESIS_TEMPLATE.format(context=context_text, question=question)

        try:
            resp   = self._llm.invoke(prompt)
            answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as exc:
            answer = f"(LLM synthesis failed: {exc})"
            logger.warning("Synthesis error: %s", exc)

        return {
            "answer":          answer,
            "cypher":          "",
            "context":         context,
            "linked_entities": [],
            "fallback_used":   False,
            "error":           None,
        }


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "answer":          "",
        "cypher":          "",
        "context":         [],
        "linked_entities": [],
        "fallback_used":   False,
        "error":           msg,
    }
