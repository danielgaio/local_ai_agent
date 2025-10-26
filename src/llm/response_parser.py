"""Centralized parsing for LLM responses.

Provides helpers to convert raw LLM text into pydantic models used across the
codebase (ClarifyingQuestion, Recommendation) while preserving legacy dict
behavior when appropriate.
"""
import json
import logging
from typing import Any, Dict, Union

from ..core.models import ClarifyingQuestion, Recommendation

logger = logging.getLogger(__name__)


def parse_llm_response(raw_text: str) -> Union[ClarifyingQuestion, Recommendation, Dict[str, Any]]:
    """Parse raw LLM output into a pydantic model or dict.

    - If the text is not valid JSON, this function will raise json.JSONDecodeError
      so callers can fall back to returning the raw string as before.
    - If the JSON matches the expected pydantic shapes, a model instance will be
      returned. If validation fails or the JSON is a legacy format, the raw dict
      will be returned for backward compatibility.

    Args:
        raw_text: The raw text returned by the LLM

    Returns:
        ClarifyingQuestion | Recommendation | dict
    """
    txt = (raw_text or "").strip()
    data = json.loads(txt)

    if not isinstance(data, dict):
        # keep behavior: callers expect an object
        return data

    # Attempt to coerce into the more-structured pydantic models
    try:
        if data.get("type") == "clarify":
            return ClarifyingQuestion(**data)
    except Exception:
        logger.exception("Failed to parse ClarifyingQuestion from LLM output")

    try:
        if data.get("type") == "recommendation":
            return Recommendation(**data)
    except Exception:
        logger.exception("Failed to parse Recommendation from LLM output")

    # Fallback: return the raw dict so legacy code can handle it
    return data
