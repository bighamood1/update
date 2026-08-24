"""Generation: prompts and Ollama client."""

from .prompts import SYSTEM_PROMPT, build_rag_prompt
from .ollama_client import OllamaClient

__all__ = ["SYSTEM_PROMPT", "build_rag_prompt", "OllamaClient"]
