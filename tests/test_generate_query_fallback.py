import pytest

from main import generate_retriever_query, keyword_extract_query


def test_generate_query_falls_back_on_long_output(monkeypatch):
    convo = ["I want long-travel suspension for adventure touring"]

    def fake_invoke(prompt_text):
        # return a very long response >12 words
        return "this is a very long response that clearly exceeds twelve words and is not a short query"

    monkeypatch.setattr('main.invoke_model_with_prompt', fake_invoke)

    q, used = generate_retriever_query(convo)
    assert used is True
    assert isinstance(q, str)
    # ensure fallback query matches the deterministic extractor for the same input
    assert q == keyword_extract_query(convo[-1])


def test_generate_query_falls_back_on_empty_output(monkeypatch):
    convo = ["I want long-travel suspension for adventure touring"]

    def fake_invoke(prompt_text):
        return ""

    monkeypatch.setattr('main.invoke_model_with_prompt', fake_invoke)

    q, used = generate_retriever_query(convo)
    assert used is True
    assert isinstance(q, str)
    assert q == keyword_extract_query(convo[-1])
