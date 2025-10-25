import pytest
from tests.test_config import setup_test_modules, MockOllama
from src.conversation.history import generate_retriever_query, keyword_extract_query


def test_generate_query_falls_back_on_long_output():
    # Set up test environment
    setup_test_modules()
    mock_llm = MockOllama()
    
    # Test input
    convo = ["I want long-travel suspension for adventure touring"]
    
    # Set mock response
    mock_llm.set_mock_response("this is a very long response that clearly exceeds twelve words and is not a short query")

    q, used = generate_retriever_query(convo)
    assert used is True
    assert isinstance(q, str)
    # ensure fallback query matches the deterministic extractor for the same input
    assert q == keyword_extract_query(convo[-1])


def test_generate_query_falls_back_on_empty_output():
    # Set up test environment
    setup_test_modules()
    mock_llm = MockOllama()
    
    # Test input
    convo = ["I want long-travel suspension for adventure touring"]
    
    # Set mock response
    mock_llm.set_mock_response("")

    q, used = generate_retriever_query(convo)
    assert used is True
    assert isinstance(q, str)
    assert q == keyword_extract_query(convo[-1])
