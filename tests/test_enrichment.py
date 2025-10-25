import json
from tests.test_config import setup_test_modules, MockOllama
from src.conversation import analyze_with_llm
from src.core.models import MotorcycleReview


def test_enrichment_populates_evidence():
    """Test that evidence is populated from top_reviews metadata"""
    # Create mock LLM
    mock_llm = MockOllama()
    
    # Set up test environment with our mock LLM
    setup_test_modules(mock_llm)
    setup_test_modules(mock_llm)
    
    # Model returns a recommendation with empty evidence
    model_output = {
        "type": "recommendation",
        "picks": [
            {
                "brand": "KTM",
                "model": "790 Adventure",
                "year": 2019,
                "price_est": 10000,
                "reason": "long-travel suspension for offroad comfort",
                "evidence": ""
            }
        ],
        "note": ""
    }
    
    # Set mock response
    mock_llm.set_mock_response(json.dumps(model_output))

    # Provide top_reviews with matching metadata (suspension_notes should be used as evidence)
    top_reviews = [
        MotorcycleReview(
            brand="KTM",
            model="790 Adventure",
            year=2019,
            comment="Great suspension and long travel.",
            price_usd_estimate=10000,
            engine_cc=790,
            suspension_notes="long-travel, plush",
            ride_type="adventure",
            text="Great suspension and long travel for offroad"
        )
    ]
    
    convo = ["I want long-travel suspension for offroad touring"]

    out = analyze_with_llm(convo, top_reviews)
    # The returned display should include the suspension_notes as Evidence
    assert "Evidence: long-travel, plush" in out