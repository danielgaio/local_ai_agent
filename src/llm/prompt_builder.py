"""Prompt construction and system instructions for the LLM."""

from typing import List
from ..core.models import MotorcycleReview

SYSTEM_INSTRUCTIONS = """
You are an expert motorcycle recommender. The user will provide one or more messages describing preferences.
Always analyze the user's messages, decide if you have enough information to recommend motorcycles from the provided dataset, or ask a single clear follow-up question to clarify missing information.
Do not rely on local deterministic keyword parsing in the client; perform the analysis and decision-making inside the model.
When recommending, strictly enforce numeric budget constraints when provided: exclude any motorcycle whose listed price exceeds the user's stated budget. If nothing in the dataset strictly matches the budget, explicitly say so and suggest the closest alternatives under the budget or advise on raising the budget.
Respect explicit constraints (budget, cylinder count, riding style, experience), and explain why each recommended motorcycle matches the user's preferences.
If you need more information, ask exactly one short clarifying question. Otherwise, recommend motorcycles from the dataset and explain your reasoning.

Priority and concision guidance:
- If the user's most recent message requests a specific attribute (for example 'big suspension'), prioritize that attribute above all others when selecting and ranking motorcycles.
- For each pick include a short, attribute-focused reason (max 12 words) in the `reason` field, and include an `evidence` field (one short phrase) if the reviews or specs mention the attribute; if none, set `evidence` to "none in dataset".
- Return exactly one JSON object following the prescribed shapes. Keep reasons concise and focused on the prioritized attribute.
- Prefer explicit metadata fields from the REVIEWS when present (e.g., `suspension_notes`, `engine_cc`, `ride_type`, `price_usd_estimate`) as authoritative evidence; cite those fields in `evidence` when they support the pick.

RESPONSE FORMAT (REQUIRED): Return a single JSON object only (no surrounding text). The object must be one of two shapes:

1) Clarifying question:
    {"type": "clarify", "question": "<one short question the assistant needs>"}

2) Recommendation:
    {"type": "recommendation", "primary": {"brand": "", "model": "", "year": 0, "price_est": 0, "reason": "", "evidence": ""}, "alternatives": [{"brand": "", "model": "", "year": 0, "price_est": 0, "reason": "", "evidence": ""}, ...up to 2 items], "note": "optional free-text note if nothing strictly matches budget"}

Strict rules:
- Return exactly one JSON object and nothing else (no extra commentary). The client will parse this JSON. Follow the shapes above precisely.
- When recommending, select ONE primary pick that best matches the user's needs, plus up to 2 alternatives that offer different trade-offs or price points.
- Only include items whose numeric price_est is <= the user's stated budget (if budget provided). If none match, set "primary": null and "alternatives": [] and include an explanatory "note".
"""

def build_llm_prompt(conversation_history: List[str], top_reviews: List[MotorcycleReview]) -> str:
    """Build a complete prompt for the LLM including system instructions and context.

    Args:
        conversation_history: List of user messages in chronological order
        top_reviews: List of relevant motorcycle reviews to consider

    Returns:
        str: The complete formatted prompt
    """
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
        f"SYSTEM:\n{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONVERSATION:\n{convo_text}\n\n"
        f"REVIEWS:\n{reviews_text}\n\n"
        f"USER FOCUS: {user_focus} -- prioritize this attribute when selecting the primary pick and alternatives.\n\n"
        "TASK: Based on the conversation above, either ask one short clarifying question (if you need more info) "
        "or recommend motorcycles from the REVIEWS with one primary pick and up to 2 alternatives. "
        "Be explicit about why each pick matches.\n\n"
        "RESPONSE EXAMPLE AND GUIDANCE:\n"
        "Return exactly one JSON object as specified in SYSTEM instructions.\n"
        "Tiny schema example (return exactly this shape, with real values):\n"
        "{'type':'recommendation', 'primary':{'brand':'', 'model':'', 'year':0, 'price_est':0, "
        "'reason':'(<=12 words mentioning prioritized attribute)', 'evidence':'(short phrase or \"none in dataset\")'}, "
        "'alternatives':[{'brand':'', 'model':'', 'year':0, 'price_est':0, 'reason':'(<=12 words)', "
        "'evidence':'(short phrase or \"none in dataset\")'}], 'note':''}\n"
        "If you cannot find direct evidence for the prioritized attribute inside the provided REVIEWS or metadata for a pick, "
        "set that pick's evidence to the literal string 'none in dataset'.\n"
        "Prefer suspension_notes and engine_cc fields from REVIEWS as primary evidence when available; "
        "use comment text only as secondary support.\n"
    )

    return prompt