import pytest

from main import generate_retriever_query


def test_generate_retriever_query_monkeypatch(monkeypatch):
    convo = [
        "I need a bike with big suspension for long off-road trips",
        "Budget under 10000"
    ]

    # Monkeypatch the LLM invoker to return a controlled string
    def fake_invoke(prompt_text):
        return 'long-travel suspension offroad touring'

    monkeypatch.setattr('main.invoke_model_with_prompt', fake_invoke)

    q = generate_retriever_query(convo)
    assert q is not None
    assert isinstance(q, str)
    # single-line and non-empty
    assert '\n' not in q
    assert q.strip() != ''
