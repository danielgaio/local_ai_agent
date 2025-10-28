"""JSON schema generation and validation for LLM responses.

This module provides the canonical source of truth for response schemas,
generating them directly from Pydantic models to ensure consistency.
"""

from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel

from ..core.models import (
    MotorcyclePick, ClarifyingQuestion, Recommendation, LLMResponse
)


def get_schema_for_model(model: type[BaseModel]) -> Dict[str, Any]:
    """Get JSON schema for a Pydantic model.
    
    Args:
        model: Pydantic model class
        
    Returns:
        Dict containing the JSON schema
    """
    return model.model_json_schema()


def get_llm_response_schema() -> Dict[str, Any]:
    """Get the complete schema for LLM responses.
    
    Returns:
        Dict containing schemas for both response types
    """
    return {
        "clarify": get_schema_for_model(ClarifyingQuestion),
        "recommendation": get_schema_for_model(Recommendation)
    }


def get_schema_example_clarify() -> str:
    """Get an example of a clarifying question response.
    
    Returns:
        JSON string with example clarifying question
    """
    example = ClarifyingQuestion(
        type="clarify",
        question="What is your budget for the motorcycle?"
    )
    return example.model_dump_json()


def get_schema_example_recommendation() -> str:
    """Get an example of a recommendation response.
    
    Returns:
        JSON string with example recommendation
    """
    primary = MotorcyclePick(
        brand="KTM",
        model="890 Adventure R",
        year=2023,
        price_est=14000,
        reason="Excellent long-travel suspension for off-road",
        evidence="suspension_notes: 220mm travel front and rear"
    )
    
    alternative = MotorcyclePick(
        brand="BMW",
        model="F850GS",
        year=2022,
        price_est=13500,
        reason="Adventure-ready suspension and comfort",
        evidence="ride_type: adventure"
    )
    
    example = Recommendation(
        type="recommendation",
        primary=primary,
        alternatives=[alternative],
        note=None
    )
    return example.model_dump_json()


def format_schema_for_prompt() -> str:
    """Format schema information for inclusion in LLM prompts.
    
    Returns:
        String containing formatted schema documentation
    """
    clarify_example = get_schema_example_clarify()
    rec_example = get_schema_example_recommendation()
    
    schema_text = f"""
RESPONSE FORMAT (REQUIRED): Return a single JSON object only (no surrounding text). 
The object must be one of two shapes:

1) Clarifying question:
{clarify_example}

2) Recommendation:
{rec_example}

Field requirements:
- type: Must be either "clarify" or "recommendation"
- For recommendations:
  - primary: Single best match (can be null if no matches)
  - alternatives: Array of 0-2 alternative picks
  - Each pick must have: brand, model, year, price_est, reason (<=12 words), evidence
  - evidence: Quote from metadata/reviews or "none in dataset"
  - note: Optional explanation if no strict matches (e.g., budget issues)
"""
    return schema_text.strip()


def validate_response_format(response_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate that a response matches one of the expected schemas.
    
    Args:
        response_dict: Parsed JSON response from LLM
        
    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        response_type = response_dict.get("type")
        
        if response_type == "clarify":
            ClarifyingQuestion(**response_dict)
            return (True, None)
        elif response_type == "recommendation":
            Recommendation(**response_dict)
            return (True, None)
        else:
            return (False, f"Invalid response type: {response_type}")
    except Exception as e:
        return (False, str(e))


def get_compact_schema_hint() -> str:
    """Get a very compact schema hint for space-constrained prompts.
    
    Returns:
        Compact string describing the schema
    """
    return (
        "Return JSON: {'type':'clarify', 'question':'...'} OR "
        "{'type':'recommendation', 'primary':{...}, 'alternatives':[...], 'note':'...'}"
    )


def get_system_instructions_with_schema() -> str:
    """Get system instructions combined with the canonical JSON schema.
    
    Returns:
        Complete system instructions including response format requirements
    """
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
"""
    
    schema_docs = format_schema_for_prompt()
    
    full_instructions = f"""{SYSTEM_INSTRUCTIONS.strip()}

{schema_docs}

Strict rules:
- Return exactly one JSON object and nothing else (no extra commentary). The client will parse this JSON. Follow the shapes above precisely.
- When recommending, select ONE primary pick that best matches the user's needs, plus up to 2 alternatives that offer different trade-offs or price points.
- Only include items whose numeric price_est is <= the user's stated budget (if budget provided). If none match, set "primary": null and "alternatives": [] and include an explanatory "note".
"""
    return full_instructions
