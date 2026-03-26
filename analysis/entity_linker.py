"""
Semantic Entity Linker for the Fusion Knowledge Graph
======================================================
Uses sentence-transformers to embed all Entity.name_norm strings from Neo4j
and find the closest matches to a free-text user query via cosine similarity.

Why this matters
----------------
Small LLMs (like nemotron-3-nano:4b) tend to hallucinate entity names in Cypher
(e.g., writing WHERE e.name_norm = 'Tokamak' when the stored string is 'tokamak',
or WHERE ... = 'plasma instability' when no such node exists).

By finding the ACTUAL entity strings before Cypher generation happens, we inject
verified names into the prompt — the LLM only needs to write the query structure.

Performance
-----------
- First run: fetches top 12 k entities from Neo4j (~2 s), embeds them (~25 s CPU),
  saves to output/entity_linker_cache.pkl.
- Subsequent runs: loads cache in < 1 s.
- Query time: < 50 ms (pure numpy matmul).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from analysis.neo4j_utils import get_database, OUTPUT_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = OUTPUT_DIR / "entity_linker_cache.pkl"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_TOP_N = 12_000   # top entities by total mention count


class EntityLinker:
    """Semantic nearest-entity lookup over the Fusion KG entity vocabulary.

    Parameters
    ----------
    driver
        Neo4j driver (from ``analysis.neo4j_utils.get_driver()``).
    embed_model
        Sentence-transformers model name. ``all-MiniLM-L6-v2`` is fast (80 MB)
        and good at short phrases.
    top_n
        How many entities to index (ranked by total mention count).
        12 k covers >95 % of meaningful fusion concepts.
    cache_path
        Path where embeddings are persisted as a pickle file.
    """

    def __init__(
        self,
        driver,
        embed_model: str = _EMBED_MODEL,
        top_n: int = _TOP_N,
        cache_path: Path = _CACHE_PATH,
    ) -> None:
        # Import here so the slow transformers scan happens only when truly needed
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(embed_model, device="cpu")
        self._names: list[str] = []
        self._embeds: np.ndarray | None = None
        self._top_n = top_n

        self._build(driver, cache_path)

    # ── Public ────────────────────────────────────────────────────────────

    def link(
        self,
        query: str,
        top_k: int = 6,
        min_score: float = 0.30,
    ) -> list[tuple[str, float]]:
        """Return the top_k most semantically similar entity names.

        Parameters
        ----------
        query
            The user's natural language question (or any text).
        top_k
            Maximum number of entities to return.
        min_score
            Minimum cosine similarity threshold (0–1).

        Returns
        -------
        list of (name_norm, cosine_score) sorted descending by score.
        """
        if self._embeds is None or not self._names:
            return []

        q_vec = self._model.encode([query], normalize_embeddings=True)
        scores: np.ndarray = (self._embeds @ q_vec.T).squeeze()

        top_idx = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for i in top_idx:
            s = float(scores[i])
            if s < min_score:
                break
            results.append((self._names[int(i)], round(s, 3)))
            if len(results) >= top_k:
                break
        return results

    @property
    def entity_count(self) -> int:
        return len(self._names)

    # ── Internal ──────────────────────────────────────────────────────────

    def _build(self, driver, cache_path: Path) -> None:
        """Load cache or build embeddings fresh from Neo4j."""
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cache = pickle.load(f)
                self._names = cache["names"]
                self._embeds = cache["embeds"]
                logger.info("EntityLinker: loaded %d entities from cache", len(self._names))
                return
            except Exception as exc:
                logger.warning("EntityLinker: cache load failed (%s) — rebuilding", exc)

        logger.info("EntityLinker: fetching top-%d entity names from Neo4j…", self._top_n)
        self._names = self._fetch_names(driver)

        logger.info("EntityLinker: embedding %d entity names (first-run only)…", len(self._names))
        self._embeds = self._model.encode(
            self._names,
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        try:
            with cache_path.open("wb") as f:
                pickle.dump({"names": self._names, "embeds": self._embeds}, f)
            logger.info("EntityLinker: cache saved → %s", cache_path)
        except Exception as exc:
            logger.warning("EntityLinker: could not save cache: %s", exc)

    def _fetch_names(self, driver) -> list[str]:
        db = get_database()
        cypher = """
            MATCH (e:Entity)
            OPTIONAL MATCH (:Paper)-[m:MENTIONS]->(e)
            WITH e.name_norm AS name, sum(m.count) AS mentions
            WHERE name IS NOT NULL AND size(name) > 1
            RETURN name
            ORDER BY mentions DESC
            LIMIT $top_n
        """
        with driver.session(database=db) as sess:
            return [r["name"] for r in sess.run(cypher, top_n=self._top_n)]
