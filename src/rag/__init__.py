"""NMU RAG: a local, grounded retrieval-augmented generation system.

PHASE 2 — text-only RAG over the official New Mansoura University website
dataset (``data/documents.jsonl``). Fully local: embeddings and generation
run on this machine (Ollama for the LLM).
"""

from .config import Settings, settings

__version__ = "2.0.0"

__all__ = ["Settings", "settings", "__version__"]
