#!/usr/bin/env python3
"""Entry point for the motorcycle recommendation system using Typer CLI.

This provides both interactive and non-interactive modes for better testability
and scripting capabilities.

Examples:
    # Interactive mode
    python run_typer.py
    
    # Single query
    python run_typer.py --query "adventure bike under 10000"
    
    # JSON output
    python run_typer.py --query "touring bike" --json
    
    # Batch processing
    python run_typer.py --batch queries.txt --output results.json
"""
import logging
from src.core.config import DEBUG
from src.cli.typer_main import app


def _configure_logging() -> None:
    """Configure logging based on DEBUG flag."""
    level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()
    app()
