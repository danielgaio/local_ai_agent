"""Prompt construction and system instructions for the LLM."""

from typing import List
from ..core.models import MotorcycleReview
from .schema import get_system_instructions_with_schema

def build_llm_prompt(conversation_history: List[str], top_reviews: List[MotorcycleReview]) -> str:
    """Build a complete prompt for the LLM including system instructions and context.

    Args:
        conversation_history: List of user messages in chronological order
        top_reviews: List of relevant motorcycle reviews to consider

    Returns:
        str: The complete formatted prompt with canonical schema
    """
    # Get system instructions with canonical schema
    system_instructions = get_system_instructions_with_schema()
    
    # Format conversation history
    convo_text = "\n".join([f"User: {m}" for m in conversation_history])

    # Format reviews with metadata
    reviews_parts = []
    for r in top_reviews:
        parts = [f"- {r.brand} {r.model} ({r.year}): {r.full_text}"]
        parts.append(f"Price est: ${r.price_usd_estimate or r.price_est}")
        if r.suspension_notes:
            parts.append(f"Suspension notes: {r.suspension_notes}")
        if r.engine_cc:
            parts.append(f"Engine (cc): {r.engine_cc}")
        if r.ride_type:
            parts.append(f"Ride type: {r.ride_type}")
        reviews_parts.append(" | ".join(parts))
    reviews_text = "\n".join(reviews_parts)

    # Add user focus hint from most recent message
    user_focus = conversation_history[-1] if conversation_history else ""

    prompt = (
        f"SYSTEM:\n{system_instructions}\n\n"
        f"CONVERSATION:\n{convo_text}\n\n"
        f"REVIEWS:\n{reviews_text}\n\n"
        f"USER FOCUS: {user_focus} -- prioritize this attribute when selecting the primary pick and alternatives.\n\n"
        "TASK: Based on the conversation above, either ask one short clarifying question (if you need more info) "
        "or recommend motorcycles from the REVIEWS with one primary pick and up to 2 alternatives. "
        "Be explicit about why each pick matches.\n\n"
        "If you cannot find direct evidence for the prioritized attribute inside the provided REVIEWS or metadata for a pick, "
        "set that pick's evidence to the literal string 'none in dataset'.\n"
        "Prefer suspension_notes and engine_cc fields from REVIEWS as primary evidence when available; "
        "use comment text only as secondary support.\n"
    )

    return prompt