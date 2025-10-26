"""Response validation and filtering logic."""

import re
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..core.models import (
    ClarifyingQuestion, MotorcyclePick, MotorcycleReview,
    Recommendation, ValidationError
)


def validate_and_filter(
    parsed: Union[ClarifyingQuestion, Recommendation, Dict],
    conversation_history: List[str]
) -> Tuple[bool, Union[ClarifyingQuestion, Recommendation, ValidationError]]:
    """Validate and filter parsed LLM responses.

    Validates:
    1. Budget constraints from conversation
    2. Attribute presence in recommendations

    Args:
        parsed: The parsed LLM response
        conversation_history: List of user messages

    Returns:
        tuple: (is_valid, result) where result is either the validated/filtered
               response or validation error information
    """
    try:
        if not isinstance(parsed, (dict, ClarifyingQuestion, Recommendation)):
            return False, ValidationError(
                reason="parsed response is not an object",
                action="reject"
            )

        if isinstance(parsed, dict) and parsed.get("type") != "recommendation":
            # Nothing to validate for clarifying questions
            return True, parsed

        # Extract all picks for validation
        all_picks: List[MotorcyclePick] = []
        if isinstance(parsed, Recommendation):
            if parsed.primary:
                all_picks.append(parsed.primary)
            all_picks.extend(parsed.alternatives)
        elif isinstance(parsed, dict):
            # Legacy format support
            if "picks" in parsed:
                all_picks = parsed.get("picks", [])
            else:
                primary = parsed.get("primary")
                if primary:
                    all_picks.append(primary)
                all_picks.extend(parsed.get("alternatives", []))

        # Parse budget from conversation
        budget = _extract_budget(conversation_history)

        # Filter by budget if specified
        if budget is not None and all_picks:
            valid_picks = []
            for pick in all_picks:
                if _is_within_budget(pick, budget):
                    valid_picks.append(pick)

            # Update response with filtered picks
            if isinstance(parsed, Recommendation):
                if parsed.primary and not _is_within_budget(parsed.primary, budget):
                    parsed.primary = None
                parsed.alternatives = [
                    alt for alt in parsed.alternatives 
                    if _is_within_budget(alt, budget)
                ]
                if not parsed.primary and not parsed.alternatives:
                    parsed.note = (
                        f"No items at or below the parsed budget ${int(budget)} "
                        "found in dataset."
                    )
            elif isinstance(parsed, dict):
                if "picks" in parsed:
                    parsed["picks"] = valid_picks
                    if not valid_picks:
                        parsed["note"] = (
                            f"No items at or below the parsed budget ${int(budget)} "
                            "found in dataset."
                        )
                else:
                    if parsed.get("primary"):
                        if not _is_within_budget(parsed["primary"], budget):
                            parsed["primary"] = None
                    parsed["alternatives"] = [
                        alt for alt in parsed.get("alternatives", [])
                        if _is_within_budget(alt, budget)
                    ]
                    if not parsed.get("primary") and not parsed.get("alternatives"):
                        parsed["note"] = (
                            f"No items at or below the parsed budget ${int(budget)} "
                            "found in dataset."
                        )

        # Check attribute presence
        prioritized = _extract_prioritized_attribute(conversation_history)
        if prioritized and all_picks:
            any_mention = any(_mentions_attr(p, prioritized) for p in all_picks)
            if not any_mention:
                return False, ValidationError(
                    reason=(
                        f"None of the picks mention the prioritized attribute "
                        f"'{prioritized}' in reason or evidence."
                    ),
                    action="retry",
                    attribute=prioritized
                )

        return True, parsed

    except Exception as e:
        logging.getLogger(__name__).exception("Unexpected validation error")
        return False, ValidationError(
            reason=f"validation error: {e}",
            action="reject"
        )


def _extract_budget(conversation_history: List[str]) -> Optional[float]:
    """Extract budget value from conversation history."""
    joined = " ".join(conversation_history or [])
    text = joined.strip()
    if not text:
        return None

    low = text.lower()

    def _to_float(num_str: str, has_k: bool = False) -> Optional[float]:
        try:
            n = float(num_str.replace(",", ""))
            return n * 1000.0 if has_k else n
        except (ValueError, TypeError):
            return None

    # 1) Explicit dollar amounts like $12,000 or $ 12,000
    m = re.search(r"\$\s*([0-9,]+(?:\.\d+)?)", low)
    if m:
        return _to_float(m.group(1), False)

    # 2) Budget: 12000 USD or 12000 dollars
    m = re.search(r"budget[:\s]*\$?\s*([0-9,]+(?:\.\d+)?)(k?)\b(?:\s*(?:usd|dollars))?", low)
    if m:
        return _to_float(m.group(1), bool(m.group(2)))

    # 3) Comparator patterns like 'under 12k', 'up to 9000', '<= 15k', 'less than 10k', 'at most 9k'
    m = re.search(
        r"(?:under|less than|below|up to|upto|at most|max(?:imum)?|<=|<)\s*\$?\s*([0-9,]+(?:\.\d+)?)(k?)\b",
        low,
    )
    if m:
        return _to_float(m.group(1), bool(m.group(2)))

    # 4) Approximate words like 'around 10k' or 'about 8k'
    m = re.search(r"(?:around|about|approx(?:\.|imately)?)\s*\$?\s*([0-9,]+(?:\.\d+)?)(k?)\b", low)
    if m:
        return _to_float(m.group(1), bool(m.group(2)))

    # 5) Ranges like '12k-15k' or '12k to 15k' -> prefer the upper bound as the budget ceiling
    m = re.search(r"([0-9,]+(?:\.\d+)?)(k?)\s*(?:-|to|–|and)\s*([0-9,]+(?:\.\d+)?)(k?)\b", low)
    if m:
        # use the second group's number and its k-flag if present
        upper_num = m.group(3)
        upper_k = bool(m.group(4))
        return _to_float(upper_num, upper_k)

    # 6) Numbers with unit USD or 'dollars'
    m = re.search(r"([0-9,]+(?:\.\d+)?)\s*(?:usd|dollars)\b", low)
    if m:
        return _to_float(m.group(1), False)

    # 7) Trailing 'k' numbers like '12k' or '12 k'
    m = re.search(r"([0-9]+(?:\.\d+)?)\s*k\b", low)
    if m:
        return _to_float(m.group(1), True)

    # Fallback: look for standalone numbers but only when explicitly prefixed with 'budget' was not found
    m = re.search(r"budget[:\s]*([0-9,]+(?:\.\d+)?)\b", low)
    if m:
        return _to_float(m.group(1), False)

    return None


def _is_within_budget(pick: Union[MotorcyclePick, Dict], budget: float) -> bool:
    """Check if a motorcycle pick is within budget."""
    if isinstance(pick, Dict):
        price = pick.get("price_est")
    else:
        price = pick.price_est

    try:
        # Handle string prices
        if isinstance(price, str):
            price_clean = re.sub(r"[^0-9.]", "", price)
            price_val = float(price_clean) if price_clean else None
        else:
            price_val = float(price) if price is not None else None
    except (ValueError, TypeError):
        price_val = None
    
    if price_val is None:
        return True  # Keep items with unknown price
    return price_val <= float(budget)


def _extract_prioritized_attribute(conversation_history: List[str]) -> Optional[str]:
    """Extract prioritized attribute from most recent message."""
    if not conversation_history:
        return None

    last = conversation_history[-1].lower()
    keywords = [
        "suspension", "long-travel", "long travel", "travel",
        "soft", "firm", "damping", "offroad", "touring",
        "traveling", "comfort"
    ]
    
    for k in keywords:
        if k in last:
            return k
    
    return None


def _mentions_attr(pick: Union[MotorcyclePick, Dict], attr: str) -> bool:
    """Check if a pick mentions a specific attribute."""
    for field in ("reason", "evidence"):
        if isinstance(pick, Dict):
            v = pick.get(field, "") or ""
        else:
            v = getattr(pick, field, "") or ""
            
        if isinstance(v, (int, float)):
            v = str(v)
        if v and attr in v.lower():
            return True
    return False