"""Embeddings management for the motorcycle recommendation system.

This module provides a factory pattern for creating embeddings with clear
logging and test overrides. The initialization follows a deterministic path:

1. If USE_DUMMY or CI environment detected -> DummyEmbeddings
2. If MODEL_PROVIDER=openai -> OpenAI (fail if not available)
3. If MODEL_PROVIDER=ollama -> Ollama (with OpenAI fallback)
4. Last resort for CI -> DummyEmbeddings

Test override: Use set_embeddings_override() to inject custom embeddings.
"""

import os
import hashlib
import logging
from typing import List, Optional, Callable

from langchain_core.embeddings import Embeddings
from ..core.config import (
    MODEL_PROVIDER, USE_DUMMY, OLLAMA_EMBEDDINGS_MODEL,
    OPENAI_EMBEDDINGS_MODEL, get_openai_api_key, DEBUG
)

# Set up module logger
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Optional dependencies
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None
    if DEBUG:
        logger.debug("langchain_ollama not available")

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
    if DEBUG:
        logger.debug("langchain_openai not available")

# Global override for testing
_embeddings_override: Optional[Callable[[], Embeddings]] = None


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


def set_embeddings_override(factory: Optional[Callable[[], Embeddings]]) -> None:
    """Set a factory function to override embeddings initialization for testing.
    
    Args:
        factory: Callable that returns an Embeddings instance, or None to clear override
        
    Example:
        >>> set_embeddings_override(lambda: DummyEmbeddings(dim=16))
        >>> embeddings = init_embeddings()  # Will use override
        >>> set_embeddings_override(None)  # Clear override
    """
    global _embeddings_override
    _embeddings_override = factory
    if factory:
        logger.info("Embeddings override set for testing")
    else:
        logger.debug("Embeddings override cleared")


def get_embeddings_override() -> Optional[Callable[[], Embeddings]]:
    """Get the current embeddings override factory (for testing inspection)."""
    return _embeddings_override


def init_embeddings() -> Embeddings:
    """Initialize embeddings according to MODEL_PROVIDER with fallbacks.
    
    This function follows a deterministic initialization path with clear logging:
    1. Check for test override (set via set_embeddings_override)
    2. If USE_DUMMY=1 or CI environment -> DummyEmbeddings
    3. If MODEL_PROVIDER=openai -> OpenAI (strict, fail if unavailable)
    4. If MODEL_PROVIDER=ollama -> Ollama with OpenAI fallback
    5. Last resort for CI environments -> DummyEmbeddings

    Returns:
        Embeddings: A configured embeddings model

    Raises:
        RuntimeError: If no embeddings provider is available
    """
    # Check for test override first
    if _embeddings_override is not None:
        logger.info("Using embeddings override from test configuration")
        return _embeddings_override()
    
    # If CI or forced dummy, use dummy
    if USE_DUMMY:
        logger.info(f"Using DummyEmbeddings (USE_DUMMY={USE_DUMMY}, CI environment detected)")
        return DummyEmbeddings()

    # If MODEL_PROVIDER=openai, try OpenAIEmbeddings first and require success
    if MODEL_PROVIDER == "openai":
        logger.info(f"Initializing OpenAI embeddings (provider={MODEL_PROVIDER})")
        if OpenAIEmbeddings is not None:
            try:
                key = get_openai_api_key()
                embeddings = OpenAIEmbeddings(
                    model=OPENAI_EMBEDDINGS_MODEL,
                    openai_api_key=key
                )
                logger.info(f"Successfully initialized OpenAI embeddings (model={OPENAI_EMBEDDINGS_MODEL})")
                return embeddings
            except Exception as e:
                # With openai provider, we want to fail if OpenAI embeddings aren't available
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise RuntimeError(f"Failed to initialize OpenAI embeddings (required when MODEL_PROVIDER=openai): {e}")
        else:
            logger.error("OpenAI embeddings library not installed")
            raise RuntimeError("OpenAI embeddings not available. Install langchain-openai or set MODEL_PROVIDER=ollama.")

    # For ollama provider or unspecified, try Ollama first
    logger.info(f"Initializing Ollama embeddings (provider={MODEL_PROVIDER})")
    if OllamaEmbeddings is not None:
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
            logger.info(f"Successfully initialized Ollama embeddings (model={OLLAMA_EMBEDDINGS_MODEL})")
            return embeddings
        except Exception as e:
            logger.warning(f"Ollama embeddings failed: {e}, attempting OpenAI fallback")
            # Ollama failed, try OpenAI as fallback if available
            if OpenAIEmbeddings is not None:
                try:
                    key = get_openai_api_key()
                    embeddings = OpenAIEmbeddings(openai_api_key=key)
                    logger.info("Fallback to OpenAI embeddings successful")
                    return embeddings
                except Exception as fallback_e:
                    logger.warning(f"OpenAI fallback also failed: {fallback_e}")
    else:
        logger.warning("Ollama embeddings library not installed")

    # Last resort: if we're in CI, use dummy embeddings
    if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
        logger.info("Using DummyEmbeddings as last resort for CI environment")
        return DummyEmbeddings()

    error_msg = (
        "No embeddings provider available. Install and configure either:\n"
        "1. Ollama (recommended, local): pip install langchain-ollama && "
        f"ollama pull {OLLAMA_EMBEDDINGS_MODEL}\n"
        "2. OpenAI (remote): pip install langchain-openai && "
        "export OPENAI_API_KEY=sk-..."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)