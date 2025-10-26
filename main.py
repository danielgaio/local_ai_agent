#!/usr/bin/env python3
"""Compatibility shim: provide a top-level `main` module for tests and README.

This file intentionally re-exports a small set of functions from the
refactored package layout so older imports like `import main` still work.
"""

from src.cli.main import main_cli
from src.conversation.history import (
    generate_retriever_query,
    generate_retriever_query_str,
    keyword_extract_query,
)
from src.llm.providers import get_llm, invoke_model_with_prompt
from src.vector.store import load_vector_store

__all__ = [
    "main_cli",
    "generate_retriever_query",
    "generate_retriever_query_str",
    "keyword_extract_query",
    "get_llm",
    "invoke_model_with_prompt",
    "load_vector_store",
]


if __name__ == "__main__":
    main_cli()
