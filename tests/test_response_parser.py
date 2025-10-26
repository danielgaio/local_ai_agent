import json

from src.llm.response_parser import parse_llm_response
from src.core.models import ClarifyingQuestion, Recommendation


def test_parse_clarify_json():
    raw = json.dumps({"type": "clarify", "question": "What is your budget?"})
    parsed = parse_llm_response(raw)
    assert isinstance(parsed, ClarifyingQuestion)
    assert parsed.question == "What is your budget?"


def test_parse_recommendation_json():
    raw = json.dumps({
        "type": "recommendation",
        "primary": {"brand": "Yamaha", "model": "XT", "year": 2020, "price_est": 9000, "reason": "good suspension", "evidence": "suspension_notes"},
        "alternatives": []
    })
    parsed = parse_llm_response(raw)
    assert isinstance(parsed, Recommendation)
    assert parsed.primary.brand == "Yamaha"


def test_parse_invalid_json_raises():
    bad = "not a json"
    try:
        parse_llm_response(bad)
        assert False, "Expected json.JSONDecodeError"
    except json.JSONDecodeError:
        pass
