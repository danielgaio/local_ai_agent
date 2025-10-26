#!/usr/bin/env python3
"""Entry point for the motorcycle recommendation system.

Configures basic logging and delegates to the package CLI.
"""
import logging
from src.core.config import DEBUG

from src.cli.main import main_cli


def _configure_logging() -> None:
    level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()
    main_cli()