"""Utility helpers: logging setup and content hashing."""

from .logging_utils import setup_logging, get_logger

__all__ = ["setup_logging", "get_logger"]


def set_cpu_threads() -> None:
    """Pin torch's CPU thread pool to ``CPU_THREADS`` (0 = framework default).

    Sentence-transformers loads torch on import; overriding the thread count
    before a model loads avoids oversubscribing the shared i7-1165G7 (4 cores
    / 8 threads) during embedding/reranking. Best-effort: never raises.
    """
    from ..config import get_config

    count = int(get_config().get("cpu_threads", 0) or 0)
    if count <= 0:
        return
    try:
        import torch

        torch.set_num_threads(count)
        get_logger(__name__).info(
            "[MODEL] Pinned torch to %d CPU thread(s)", count
        )
    except Exception:  # pragma: no cover - best-effort tuning
        pass
