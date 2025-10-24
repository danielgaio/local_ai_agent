"""String utility functions."""

from typing import Dict


def simple_spell_correct(text: str) -> str:
    """Very small, deterministic spell-corrections for common domain-specific typos.

    Args:
        text: The text to correct

    Returns:
        str: The text with common typos corrected
    """
    if not text:
        return text

    corrections: Dict[str, str] = {
        "suspention": "suspension",
        "longtravel": "long-travel",
        "travle": "travel",
        "dampning": "damping"
    }

    out = text
    for k, v in corrections.items():
        out = out.replace(k, v)
        out = out.replace(k.capitalize(), v)
    return out