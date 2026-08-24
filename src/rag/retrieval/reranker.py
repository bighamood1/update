"""Optional cross-encoder reranker.

A cross-encoder scores the (question, chunk) pair jointly, which is more
accurate than bi-encoder cosine similarity. Runs locally via
sentence-transformers. Purely optional — the base pipeline works without it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..config import get_config
from ..schemas.documents import RetrievedChunk
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder


class Reranker:
    """Rerank retrieved chunks using a local cross-encoder model."""

    def __init__(self, model_name: str | None = None, top_k: int | None = None) -> None:
        cfg = get_config()
        self.model_name = model_name or cfg["reranker_model"]
        self.top_k = top_k if top_k is not None else cfg["rerank_top_k"]
        self.device = cfg["reranker_device"]
        self.batch_size = max(1, int(cfg["reranker_batch_size"]))
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from ..utils import set_cpu_threads

            set_cpu_threads()
            logger.info(
                "[MODEL] Loading reranker '%s' on device '%s'",
                self.model_name, self.device,
            )
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [(question, c.text) for c in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        # Cross-encoder scores are logits; map to [0,1] via sigmoid.
        probs = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
        ranked = sorted(
            zip(chunks, probs),
            key=lambda item: item[1],
            reverse=True,
        )
        for chunk, prob in ranked:
            chunk.score = round(prob, 4)
            chunk.rerank_score = round(prob, 4)
        result = [c for c, _ in ranked]
        logger.info("Reranked %d chunks", len(result))
        return result
