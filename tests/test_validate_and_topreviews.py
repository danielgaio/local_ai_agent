import pytest

from main import validate_and_filter


def test_budget_filtering_keeps_only_under_budget():
    parsed = {
        "type": "recommendation",
        "picks": [
            {"brand": "A", "model": "One", "price_est": 7000, "reason": "long-travel suspension", "evidence": "fork travel 200mm"},
            {"brand": "B", "model": "Two", "price_est": 9000, "reason": "long-travel suspension", "evidence": "fork travel 210mm"},
        ],
    }
    conversation = ["I want long-travel suspension", "Budget $8000"]

    valid, result = validate_and_filter(parsed, conversation)
    assert valid is True
    assert isinstance(result, dict)
    # Expect only the 7000 pick remains
    assert len(result.get("picks", [])) == 1
    assert result["picks"][0]["brand"] == "A"


def test_attribute_presence_triggers_retry_when_missing():
    parsed = {
        "type": "recommendation",
        "picks": [
            {"brand": "A", "model": "X", "price_est": 5000, "reason": "comfortable seat", "evidence": "none"},
        ],
    }
    conversation = ["I need long-travel suspension for offroad touring"]

    valid, info = validate_and_filter(parsed, conversation)
    assert valid is False
    assert isinstance(info, dict)
    assert info.get("action") == "retry"
    assert "attribute" in info and info["attribute"] in ("long-travel", "long travel", "travel", "suspension")


def test_top_reviews_includes_metadata_fields():
    # stub a simple Document-like object
    class FakeDoc:
        def __init__(self, metadata, page_content=""):
            self.metadata = metadata
            self.page_content = page_content

    meta = {
        "brand": "KTM",
        "model": "790 Adventure",
        "year": 2019,
        "price_usd_estimate": 10000,
        "engine_cc": 790,
        "suspension_notes": "long-travel, plush",
        "ride_type": "adventure",
    }
    doc = FakeDoc(meta, page_content="Test review text")

    # replicate the small transformation used in main.py
    docs = [doc]
    top_reviews = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        top_reviews.append({
            "brand": meta.get("brand"),
            "model": meta.get("model"),
            "year": meta.get("year"),
            "comment": meta.get("comment") or getattr(d, "page_content", ""),
            "price_usd_estimate": meta.get("price_usd_estimate") or meta.get("price") or None,
            "engine_cc": meta.get("engine_cc"),
            "suspension_notes": meta.get("suspension_notes"),
            "ride_type": meta.get("ride_type"),
            "text": getattr(d, "page_content", ""),
        })

    assert len(top_reviews) == 1
    tr = top_reviews[0]
    assert tr["engine_cc"] == 790
    assert "long-travel" in tr["suspension_notes"]
    assert tr["ride_type"] == "adventure"


def test_keyword_extract_query_basic():
    from main import keyword_extract_query
    q = keyword_extract_query("I want long-travel suspension for adventure touring, budget $12k")
    assert q is not None
    # should include an attribute token and a ride type token
    assert any(tok in q for tok in ["long-travel", "suspension", "travel"]) or "cc" in q
    assert any(tok in q for tok in ["adventure", "touring"]) 
