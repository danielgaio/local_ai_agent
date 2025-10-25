import pytest
from tests.test_config import setup_test_modules, MockOllama
from src.conversation.history import generate_retriever_query_str as generate_retriever_query


def test_generate_retriever_query():
    # Set up test environment
    setup_test_modules()
    mock_llm = MockOllama()
    
    # Test input
    convo = [
        "I need a bike with big suspension for long off-road trips",
        "Budget under 10000"
    ]
    
    # Set mock response
    mock_llm.set_mock_response('long-travel suspension offroad touring')

    q = generate_retriever_query(convo)
    assert q is not None
    assert isinstance(q, str)
    # single-line and non-empty
    assert '\n' not in q
    assert q.strip() != ''
