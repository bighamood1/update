"""Pure-Python BM25 (Okapi) lexical index for hybrid retrieval.

No third-party dependency is required: the index is built once from the
chunk texts held in ChromaDB and queried with standard Okapi-BM25 scoring.
Tokenization is deliberately permissive (whitespace + punctuation split on a
Unicode-aware boundary) so it works for both English and Arabic without any
NLP tooling, with a light Lucene-style Arabic normalizer (hamza unification,
diacritic removal, ``ال``/conjunction-prefix stripping) so ``الشروط`` matches
``شروط``.

Only used for retrieval candidates: BM25 scores are fused with dense
embeddings via reciprocal-rank fusion (RRF) in the retriever.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..config import get_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)

_DIACRITICS = {
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650",
    "\u0651", "\u0652", "\u0640", "\u0670", "\u0671",
}
_HAMZA_ALEF = {"\u0623", "\u0625", "\u0622", "\u0621", "\u0624", "\u0626"}
_CONJUNCTIONS = {"\u0648", "\u0628", "\u0643", "\u0641", "\u0644"}


def _normalize_ar(word: str) -> str:
    """Light Lucene-style Arabic normalization for a single token."""
    if "\u0600" <= word[0] <= "\u06FF":
        word = "".join(ch for ch in word if ch not in _DIACRITICS)
        word = word.replace("\u0649", "\u064A").replace("\u0629", "\u0647")
        # Strip definite article / leading conjunctions (bounded passes so
        # short words like ``في`` are never mangled).  وال -> ال -> single
        # conjunction letter -> ال (covers ``و`` ``ب`` ``ك`` ``ف`` ``ل`` ``لل``).
        for _ in range(3):
            if len(word) > 5 and word.startswith("\u0648\u0627\u0644"):  # وال
                word = word[3:]
            elif len(word) > 4 and word.startswith("\u0627\u0644"):  # ال
                word = word[2:]
            elif len(word) > 4 and word.startswith("\u0644\u0644"):  # لل
                word = word[2:]
            elif len(word) > 3 and word[0] in _CONJUNCTIONS:  # و/ب/ك/ف/ل
                word = word[1:]
            else:
                break
        # Unify remaining hamza forms -> alef.
        if word and word[0] in _HAMZA_ALEF:
            word = "\u0627" + word[1:]
    return word


def tokenize(text: str) -> list[str]:
    """Split text into normalized lower-case Unicode word tokens (EN + Arabic)."""
    if not text:
        return []
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        out.append(_normalize_ar(tok))
    return out


class BM25Index:
    """Okapi BM25 over a fixed corpus of chunk texts."""

    def __init__(self, texts: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        cfg = get_config()
        self.k1 = k1
        self.b = b
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self._ids: list[str] = []
        self._dl: list[int] = []
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        self._avgdl = 0.0
        self._build(texts)
        logger.info("BM25 index ready: %d documents", len(self._ids))

    def _build(self, texts: dict[str, str]) -> None:
        self._ids = list(texts.keys())
        total = 0
        for cid in self._ids:
            toks = tokenize(texts[cid])
            tf = Counter(toks)
            self._tf.append(tf)
            self._dl.append(len(toks))
            total += len(toks)
            for term in set(toks):
                self._df[term] += 1
        n = len(self._ids)
        self._avgdl = total / n if n else 0.0
        # Precompute idf for every term that appears (idf of 0.0 for terms in
        # every document).
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }

    # -- scoring ---------------------------------------------------------

    def _score(self, terms: list[str], doc_index: int) -> float:
        dl = self._dl[doc_index]
        tf = self._tf[doc_index]
        denom = self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else 1.0
        score = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (freq * (self.k1 + 1)) / (freq + denom)
        return score

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, bm25_score), ...]`` sorted by score desc.

        ``allowed`` optionally restricts scoring to a set of chunk ids (used to
        keep lexical retrieval inside the same routed metadata scope as dense
        retrieval). The broad-safety fallback in the retriever still covers the
        unfiltered case.
        """
        if not self._ids:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scored = []
        for i, cid in enumerate(self._ids):
            if allowed is not None and cid not in allowed:
                continue
            s = self._score(terms, i)
            if s > 0:
                scored.append((cid, s))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    # -- fusion helper -----------------------------------------------------

    def rrf_score(self, dense_rank: int, bm25_rank: int | None) -> float:
        """Reciprocal-rank fusion score for one chunk across both systems."""
        k = self.rrf_k
        score = 1.0 / (k + dense_rank + 1)
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank + 1)
        return round(score, 6)