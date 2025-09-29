import json

from main import analyze_with_llm


def test_enrichment_populates_evidence(monkeypatch):
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

    def fake_invoke(prompt_text):
        return json.dumps(model_output)

    monkeypatch.setattr('main.invoke_model_with_prompt', fake_invoke)

    # Provide top_reviews with matching metadata (suspension_notes should be used as evidence)
    top_reviews = [
        {
            "brand": "KTM",
            "model": "790 Adventure",
            "year": 2019,
            "comment": "Great suspension and long travel.",
            "price_usd_estimate": 10000,
            "engine_cc": 790,
            "suspension_notes": "long-travel, plush",
            "ride_type": "adventure",
            "text": "Great suspension and long travel for offroad"
        }
    ]

    convo = ["I want long-travel suspension for offroad touring"]

    out = analyze_with_llm(convo, top_reviews)
    # The returned display should include the suspension_notes as Evidence
    assert "Evidence: long-travel, plush" in out
