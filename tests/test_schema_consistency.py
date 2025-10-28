"""
Unit tests for JSON schema consistency and consolidation.

Verifies that:
1. Generated JSON schemas match Pydantic model definitions
2. Schema examples are valid against the schemas
3. Prompt builder uses canonical schema
4. Response validation works with generated schemas
"""

import json
from typing import Any, Dict

import pytest

from src.core.models import ClarifyingQuestion, MotorcyclePick, Recommendation
from src.llm.schema import (
    format_schema_for_prompt,
    get_compact_schema_hint,
    get_llm_response_schema,
    get_schema_example_clarify,
    get_schema_example_recommendation,
    get_schema_for_model,
    get_system_instructions_with_schema,
    validate_response_format,
)
from src.llm.prompt_builder import build_llm_prompt


def test_schema_generation_from_pydantic_models():
    """Verify schema is correctly generated from Pydantic models."""
    # Test individual model schemas
    clarify_schema = get_schema_for_model(ClarifyingQuestion)
    assert "properties" in clarify_schema
    assert "type" in clarify_schema["properties"]
    assert "question" in clarify_schema["properties"]
    
    pick_schema = get_schema_for_model(MotorcyclePick)
    assert "properties" in pick_schema
    assert "brand" in pick_schema["properties"]
    assert "model" in pick_schema["properties"]
    assert "year" in pick_schema["properties"]
    assert "price_est" in pick_schema["properties"]
    assert "reason" in pick_schema["properties"]
    assert "evidence" in pick_schema["properties"]
    
    recommendation_schema = get_schema_for_model(Recommendation)
    assert "properties" in recommendation_schema
    assert "type" in recommendation_schema["properties"]
    assert "primary" in recommendation_schema["properties"]
    assert "alternatives" in recommendation_schema["properties"]
    assert "note" in recommendation_schema["properties"]


def test_llm_response_schema_structure():
    """Verify the complete LLM response schema includes both response types."""
    schema = get_llm_response_schema()
    
    assert "clarify" in schema, "Schema should include clarify response type"
    assert "recommendation" in schema, "Schema should include recommendation response type"
    
    # Verify clarify schema has required fields
    clarify_schema = schema["clarify"]
    assert "properties" in clarify_schema
    assert "type" in clarify_schema["properties"]
    assert "question" in clarify_schema["properties"]
    
    # Verify recommendation schema has required fields
    rec_schema = schema["recommendation"]
    assert "properties" in rec_schema
    assert "type" in rec_schema["properties"]
    assert "primary" in rec_schema["properties"]
    assert "alternatives" in rec_schema["properties"]


def test_schema_examples_are_valid_json():
    """Verify schema examples can be parsed as JSON."""
    clarify_example = get_schema_example_clarify()
    recommendation_example = get_schema_example_recommendation()
    
    # Should parse without error
    clarify_data = json.loads(clarify_example)
    recommendation_data = json.loads(recommendation_example)
    
    # Verify basic structure
    assert clarify_data["type"] == "clarify"
    assert "question" in clarify_data
    
    assert recommendation_data["type"] == "recommendation"
    assert "primary" in recommendation_data
    assert "alternatives" in recommendation_data


def test_schema_examples_match_pydantic_models():
    """Verify schema examples can be validated by Pydantic models."""
    clarify_example = get_schema_example_clarify()
    recommendation_example = get_schema_example_recommendation()
    
    clarify_data = json.loads(clarify_example)
    recommendation_data = json.loads(recommendation_example)
    
    # Should validate without raising exceptions
    ClarifyingQuestion(**clarify_data)
    
    # For Recommendation, we need to construct MotorcyclePick objects
    primary_pick = MotorcyclePick(**recommendation_data["primary"])
    alt_picks = [MotorcyclePick(**alt) for alt in recommendation_data["alternatives"]]
    Recommendation(
        type=recommendation_data["type"],
        primary=primary_pick,
        alternatives=alt_picks,
        note=recommendation_data.get("note", "")
    )


def test_compact_schema_hint_is_substring():
    """Verify compact schema hint is a shortened version of full schema."""
    full_schema = format_schema_for_prompt()
    compact_hint = get_compact_schema_hint()
    
    assert len(compact_hint) < len(full_schema), "Compact hint should be shorter than full schema"
    assert "recommendation" in compact_hint.lower(), "Compact hint should mention recommendation"
    assert "clarify" in compact_hint.lower(), "Compact hint should mention clarify"


def test_format_schema_for_prompt_includes_examples():
    """Verify formatted schema includes both schema and examples."""
    formatted = format_schema_for_prompt()
    
    assert "RESPONSE FORMAT" in formatted, "Should have schema section header"
    assert "clarify" in formatted, "Should show clarify response type"
    assert "recommendation" in formatted, "Should show recommendation response type"
    assert "question" in formatted, "Should show clarify question field"
    assert "primary" in formatted, "Should show primary recommendation field"


def test_system_instructions_include_schema():
    """Verify system instructions dynamically include the canonical schema."""
    instructions = get_system_instructions_with_schema()
    
    assert "RESPONSE FORMAT" in instructions, "Instructions should include schema documentation"
    assert "clarify" in instructions, "Instructions should mention clarify type"
    assert "recommendation" in instructions, "Instructions should mention recommendation type"
    assert "You are an expert motorcycle recommender" in instructions, "Should include base instructions"


def test_build_llm_prompt_uses_canonical_schema():
    """Verify prompt builder uses the canonical schema from schema.py."""
    from src.core.models import MotorcycleReview
    
    # Create minimal test data
    conversation_history = ["I need a bike for commuting"]
    top_reviews = [
        MotorcycleReview(
            brand="Honda",
            model="CB500X",
            year=2022,
            comment="Great commuter bike",
            price_usd_estimate=7000,
            engine_cc=500,
            suspension_notes="comfortable",
            ride_type="standard",
            text="Great commuter bike with comfortable ergonomics"
        )
    ]
    
    prompt = build_llm_prompt(conversation_history, top_reviews)
    
    # Verify prompt includes canonical schema elements
    assert "RESPONSE FORMAT" in prompt, "Prompt should include schema from schema.py"
    assert "clarify" in prompt, "Prompt should mention clarify type"
    assert "recommendation" in prompt, "Prompt should mention recommendation type"
    assert "User: I need a bike for commuting" in prompt, "Prompt should include conversation"
    assert "Honda CB500X" in prompt, "Prompt should include reviews"


def test_validate_response_format_accepts_valid_responses():
    """Verify validation accepts valid clarify and recommendation responses."""
    # Valid clarifying question
    clarify_response = {
        "type": "clarify",
        "question": "What is your budget?"
    }
    is_valid, error = validate_response_format(clarify_response)
    assert is_valid, f"Valid clarify response rejected: {error}"
    assert error is None
    
    # Valid recommendation
    recommendation_response = {
        "type": "recommendation",
        "primary": {
            "brand": "Honda",
            "model": "CB500X",
            "year": 2022,
            "price_est": 7000,
            "reason": "Great commuter bike",
            "evidence": "Comfortable upright position"
        },
        "alternatives": [
            {
                "brand": "Yamaha",
                "model": "MT-07",
                "year": 2021,
                "price_est": 8500,
                "reason": "Fun and nimble",
                "evidence": "Torquey parallel twin engine"
            }
        ],
        "note": "Both bikes are excellent for daily use"
    }
    is_valid, error = validate_response_format(recommendation_response)
    assert is_valid, f"Valid recommendation response rejected: {error}"
    assert error is None


def test_validate_response_format_rejects_invalid_responses():
    """Verify validation rejects malformed responses."""
    # Missing required field
    invalid_clarify = {
        "type": "clarify"
        # missing 'question' field
    }
    is_valid, error = validate_response_format(invalid_clarify)
    assert not is_valid, "Should reject clarify response missing 'question'"
    assert error is not None
    assert "question" in error.lower() or "missing" in error.lower()
    
    # Invalid type
    invalid_type = {
        "type": "unknown_type",
        "data": "something"
    }
    is_valid, error = validate_response_format(invalid_type)
    assert not is_valid, "Should reject response with invalid type"
    assert error is not None
    
    # Missing required nested field
    invalid_recommendation = {
        "type": "recommendation",
        "primary": {
            "brand": "Honda",
            # missing other required fields
        },
        "alternatives": [],
        "note": ""
    }
    is_valid, error = validate_response_format(invalid_recommendation)
    assert not is_valid, "Should reject recommendation with incomplete primary pick"
    assert error is not None


def test_schema_consistency_across_modules():
    """Verify schema definitions are consistent across different access patterns."""
    # Get schema through different paths
    direct_clarify_schema = get_schema_for_model(ClarifyingQuestion)
    full_response_schema = get_llm_response_schema()
    
    # Extract clarify schema from the response schema dict
    clarify_from_response = full_response_schema.get("clarify")
    
    assert clarify_from_response is not None, "Should find clarify schema in response schema"
    
    # Verify they define the same fields
    direct_props = set(direct_clarify_schema.get("properties", {}).keys())
    response_props = set(clarify_from_response.get("properties", {}).keys())
    
    assert direct_props == response_props, "Schema should be consistent across access patterns"


def test_pydantic_as_single_source_of_truth():
    """Verify that all schema generation traces back to Pydantic models."""
    # This test ensures the principle: if we change a Pydantic model,
    # all schema representations automatically update
    
    # Get the schema for a model
    pick_schema = get_schema_for_model(MotorcyclePick)
    
    # Verify it includes all Pydantic fields
    pydantic_fields = MotorcyclePick.model_fields.keys()
    schema_properties = pick_schema.get("properties", {}).keys()
    
    for field in pydantic_fields:
        assert field in schema_properties, f"Pydantic field '{field}' should appear in generated schema"
    
    # Verify example can be validated by Pydantic
    example = json.loads(get_schema_example_recommendation())
    primary_data = example["primary"]
    
    # Should validate successfully (will raise if schema mismatch)
    MotorcyclePick(**primary_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
