"""Embeddings management for the motorcycle recommendation system."""

import os
import hashlib
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from ..core.config import (
    MODEL_PROVIDER, USE_DUMMY, OLLAMA_EMBEDDINGS_MODEL,
    OPENAI_EMBEDDINGS_MODEL, get_openai_api_key
)

# Optional dependencies
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None


class DummyEmbeddings:
    """A tiny deterministic embedding generator for CI/tests.

    Produces short fixed-size vectors derived from an MD5 hash of the
    input text. Fast, deterministic, and doesn't require network access.
    """
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        out = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).hexdigest()
            # Turn pairs of hex chars into bytes and normalize to [0,1]
            vals = [int(h[i:i+2], 16) / 255.0 
                   for i in range(0, min(len(h), self.dim * 2), 2)]
            # Pad with zeros if needed
            if len(vals) < self.dim:
                vals.extend([0.0] * (self.dim - len(vals)))
            out.append([float(x) for x in vals[:self.dim]])
        return out

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a single query text."""
        return self.embed_documents([text])[0]


def init_embeddings() -> Embeddings:
    """Initialize embeddings according to MODEL_PROVIDER with fallbacks.

    Returns:
        Embeddings: A configured embeddings model

    Raises:
        RuntimeError: If no embeddings provider is available
    """
    # If CI or forced dummy, use dummy
    if USE_DUMMY:
        return DummyEmbeddings()

    # If MODEL_PROVIDER=openai, try OpenAIEmbeddings first and require success
    if MODEL_PROVIDER == "openai":
        if OpenAIEmbeddings is not None:
            try:
                key = get_openai_api_key()
                return OpenAIEmbeddings(
                    model=OPENAI_EMBEDDINGS_MODEL,
                    openai_api_key=key
                )
            except Exception as e:
                # With openai provider, we want to fail if OpenAI embeddings aren't available
                raise RuntimeError(f"Failed to initialize OpenAI embeddings (required when MODEL_PROVIDER=openai): {e}")
        else:
            raise RuntimeError("OpenAI embeddings not available. Install langchain-openai or set MODEL_PROVIDER=ollama.")

    # For ollama provider or unspecified, try Ollama first
    if OllamaEmbeddings is not None:
        try:
            return OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
        except Exception:
            # Ollama failed, try OpenAI as fallback if available
            if OpenAIEmbeddings is not None:
                try:
                    key = get_openai_api_key()
                    return OpenAIEmbeddings(openai_api_key=key)
                except Exception:
                    pass

    # Last resort: if we're in CI, use dummy embeddings
    if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
        return DummyEmbeddings()

    raise RuntimeError(
        "No embeddings provider available. Install and configure either:\n"
        "1. Ollama (recommended, local): pip install langchain-ollama && "
        f"ollama pull {OLLAMA_EMBEDDINGS_MODEL}\n"
        "2. OpenAI (remote): pip install langchain-openai && "
        "export OPENAI_API_KEY=sk-..."
    )