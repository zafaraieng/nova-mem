"""Inference clients and the generation cache."""

from .cache import GenerationCache
from .client import (
    GenerationResult,
    InferenceClient,
    MockLLMClient,
    OllamaClient,
    OpenAICompatClient,
    client_from_config,
)
from .zai_client import ZAIChatClient


def _lazy_embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Construct the real dense embedder (downloads weights on first use)."""
    from ..retrieval import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name)


__all__ = [
    "GenerationResult",
    "InferenceClient",
    "MockLLMClient",
    "OllamaClient",
    "OpenAICompatClient",
    "ZAIChatClient",
    "client_from_config",
    "GenerationCache",
    "_lazy_embedder",
]
