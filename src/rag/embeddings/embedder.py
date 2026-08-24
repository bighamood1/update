"""Multilingual embedding service built on sentence-transformers.

Model selection rationale (see README for full details):
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
- 384 dimensions, ~118 MB — fast on CPU.
- Trained on 50+ languages including Arabic and English with a shared
  embedding space (verified cross-lingual similarity in local tests).
- Practical on this machine: no CUDA, 16 GB RAM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from ..config import get_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loaded sentence-transformer wrapper with batch support."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        cfg = get_config()
        self.model_name = model_name or cfg["embedding_model"]
        self.device = device or cfg["embedding_device"]
        self.batch_size = cfg["embedding_batch_size"]
        self.normalize = cfg["embedding_normalize"]
        self.query_prefix = cfg["embedding_query_prefix"]
        self.passage_prefix = cfg["embedding_passage_prefix"]
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from ..utils import set_cpu_threads

            set_cpu_threads()
            logger.info(
                "[MODEL] Loading embedding model '%s' on device '%s'",
                self.model_name, self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def dimension(self) -> int:
        """Return the embedding dimension without computing anything."""
        return self.model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user question, applying the query prefix if configured.

        Repeated identical queries are served from the shared embedding cache
        (bounded LRU) so the CPU encoder is not re-run.
        """
        cache_enabled = get_config().get("cache_enabled", True)
        if cache_enabled:
            from ..utils.cache import get_cache_registry

            registry = get_cache_registry()
            cached = registry.embeddings.get(text)
            if cached is not None:
                return cached
            vector = self.embed(f"{self.query_prefix}{text}")
            registry.embeddings.put(text, vector)
            return vector
        return self.embed(f"{self.query_prefix}{text}")

    def embed(self, texts: str | Iterable[str]) -> np.ndarray:
        """Embed a single text or an iterable of texts.

        Returns a 1-D array for a single string, otherwise a 2-D array
        (n_texts, dim).
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dimension()), dtype=np.float32)

        vectors = self.model.encode(
            items,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = np.asarray(vectors, dtype=np.float32)
        return result[0] if single else result

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Explicit batch embedding for passages; always returns a 2-D array.

        Applies the passage prefix (e.g. ``passage:``) if configured, matching
        how the documents were embedded at index time.
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        if self.passage_prefix:
            items = [f"{self.passage_prefix}{t}" for t in items]
        vectors = self.model.encode(
            items,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
