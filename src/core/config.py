"""Configuration settings for the motorcycle recommendation system."""

import os
from typing import Literal

# Provider configuration
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()
PROVIDERS = Literal["ollama", "openai"]

# Environment detection
CI_ENV = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"
USE_DUMMY = os.getenv("USE_DUMMY_EMBEDDINGS") == "1" or CI_ENV
DEBUG = os.getenv("AIAGENT_DEBUG", "0") in ("1", "true", "True")

# Data paths
DB_LOCATION = "./chroma_langchain_db"
DATA_FILE = "motorcycle_reviews.csv"

# Model settings
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_EMBEDDINGS_MODEL = "mxbai-embed-large"
OPENAI_MODEL = "gpt-3.5-turbo"  # Default model for OpenAI
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"  # Latest efficient embeddings model

# Default search settings
DEFAULT_SEARCH_KWARGS = {"k": 5}

# Validation settings
MAX_QUERY_WORDS = 12
MAX_RETRIES = 1

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment, with informative error if missing."""
    key = os.getenv("OPENAI_API_KEY")
    if not key and MODEL_PROVIDER == "openai":
        raise ValueError(
            "OPENAI_API_KEY environment variable is required when MODEL_PROVIDER=openai"
        )
    return key