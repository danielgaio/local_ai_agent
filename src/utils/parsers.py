"""Parsers for motorcycle-specific data fields."""

import re
from typing import Optional, Set


def parse_price(s: str) -> Optional[float]:
    """Parse price values from text in various formats.

    Supported formats:
    - $12,000
    - 12000
    - 12k
    - Plain 3-6 digit numbers

    Args:
        s: Text containing a price value

    Returns:
        float: The parsed price value, or None if no valid price found
    """
    if not s:
        return None

    # look for $12,000 or 12000 or 12k
    m = re.search(r"\$\s*([0-9,]+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except (ValueError, TypeError):
            return None

    m = re.search(r"([0-9,]+(?:\.\d+)?)[\s]*k\b", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * 1000
        except (ValueError, TypeError):
            return None

    # plain number
    m = re.search(r"\b([0-9]{3,6})(?:\.[0-9]+)?\b", s)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            return None

    return None


def parse_engine_cc(s: str) -> Optional[int]:
    """Parse engine displacement values from text.

    Supported formats:
    - 650cc
    - 650 cc
    - Plain 2-4 digit numbers in context

    Args:
        s: Text containing an engine displacement value

    Returns:
        int: The parsed displacement in cc, or None if no valid value found
    """
    if not s:
        return None

    # Try "650 cc" format
    m = re.search(r"(\d{2,4})\s?cc\b", s, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            return None

    # Try "650cc" format (no space)
    m = re.search(r"\b(\d{2,4})cc\b", s, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            return None

    return None


def extract_suspension_notes(s: str) -> Optional[str]:
    """Extract suspension-related keywords from text.

    Args:
        s: Text to search for suspension keywords

    Returns:
        str: Comma-joined list of found suspension keywords, or None if none found
    """
    if not s:
        return None

    keywords: Set[str] = {
        "suspension", "travel", "long-travel", "long travel",
        "damping", "firm", "plush", "soft", "wp", "showa",
        "fork travel"
    }

    found = set()
    text = s.lower()
    for k in keywords:
        if k in text:
            found.add(k)

    return ", ".join(sorted(found)) if found else None


def extract_ride_type(s: str) -> Optional[str]:
    """Extract the primary riding type from text.

    Args:
        s: Text to search for ride type keywords

    Returns:
        str: The first matching ride type found, or None if none found
    """
    if not s:
        return None

    types = [
        "adventure", "touring", "cruiser", "sport",
        "offroad", "dual-sport", "enduro", "supermoto"
    ]

    text = s.lower()
    for t in types:
        if t in text:
            return t

    return None