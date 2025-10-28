"""Test embeddings initialization with overrides and deterministic behavior."""

import sys
import os
from unittest.mock import patch, Mock
import pytest

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector.embeddings import (
    init_embeddings, DummyEmbeddings,
    set_embeddings_override, get_embeddings_override
)


def test_dummy_embeddings_deterministic():
    """Test that DummyEmbeddings produces consistent results."""
    emb = DummyEmbeddings(dim=16)
    
    # Same text should always produce same embedding
    text = "test motorcycle"
    vec1 = emb.embed_query(text)
    vec2 = emb.embed_query(text)
    
    assert vec1 == vec2, "DummyEmbeddings should be deterministic"
    assert len(vec1) == 16, f"Expected dim=16, got {len(vec1)}"
    
    # Different texts should produce different embeddings
    vec3 = emb.embed_query("different text")
    assert vec1 != vec3, "Different texts should have different embeddings"


def test_dummy_embeddings_batch():
    """Test DummyEmbeddings batch processing."""
    emb = DummyEmbeddings(dim=8)
    
    texts = ["bike one", "bike two", "bike three"]
    vectors = emb.embed_documents(texts)
    
    assert len(vectors) == 3, "Should return 3 vectors"
    assert all(len(v) == 8 for v in vectors), "All vectors should have correct dimension"
    
    # Each text should have unique embedding
    assert vectors[0] != vectors[1]
    assert vectors[1] != vectors[2]


def test_embeddings_override():
    """Test that embeddings override works correctly."""
    # Create a custom embeddings mock
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    
    # Set override
    set_embeddings_override(lambda: mock_embeddings)
    
    # Verify override is set
    assert get_embeddings_override() is not None
    
    # Initialize should use override
    result = init_embeddings()
    assert result == mock_embeddings
    
    # Clear override
    set_embeddings_override(None)
    assert get_embeddings_override() is None


def test_init_embeddings_uses_dummy_in_ci():
    """Test that init_embeddings uses DummyEmbeddings when USE_DUMMY=1."""
    with patch('src.vector.embeddings.USE_DUMMY', True):
        embeddings = init_embeddings()
        assert isinstance(embeddings, DummyEmbeddings)


def test_init_embeddings_with_override():
    """Test that override takes precedence over environment variables."""
    custom_emb = DummyEmbeddings(dim=64)
    
    try:
        set_embeddings_override(lambda: custom_emb)
        
        # Even with USE_DUMMY, override should be used
        with patch('src.vector.embeddings.USE_DUMMY', True):
            result = init_embeddings()
            assert result == custom_emb
            assert result.dim == 64
    finally:
        set_embeddings_override(None)


def test_init_embeddings_openai_provider():
    """Test that OpenAI provider fails appropriately when unavailable."""
    with patch('src.vector.embeddings.MODEL_PROVIDER', 'openai'), \
         patch('src.vector.embeddings.USE_DUMMY', False), \
         patch('src.vector.embeddings.OpenAIEmbeddings', None):
        
        with pytest.raises(RuntimeError, match="OpenAI embeddings not available"):
            init_embeddings()


def test_init_embeddings_logging():
    """Test that initialization logs appropriately."""
    import logging
    from src.vector.embeddings import logger
    
    # Capture log output
    with patch.object(logger, 'info') as mock_info:
        with patch('src.vector.embeddings.USE_DUMMY', True):
            init_embeddings()
            
            # Verify logging occurred
            assert mock_info.called
            calls = [str(call) for call in mock_info.call_args_list]
            assert any('DummyEmbeddings' in str(call) for call in calls)


def test_embeddings_factory_pattern():
    """Test the factory pattern allows flexible test configuration."""
    # Define multiple custom embeddings
    emb_configs = [
        DummyEmbeddings(dim=8),
        DummyEmbeddings(dim=16),
        DummyEmbeddings(dim=32)
    ]
    
    for config in emb_configs:
        try:
            set_embeddings_override(lambda c=config: c)
            result = init_embeddings()
            assert result.dim == config.dim
        finally:
            set_embeddings_override(None)


if __name__ == '__main__':
    print("Testing embeddings initialization...")
    
    test_dummy_embeddings_deterministic()
    print("✓ DummyEmbeddings deterministic test passed")
    
    test_dummy_embeddings_batch()
    print("✓ DummyEmbeddings batch test passed")
    
    test_embeddings_override()
    print("✓ Embeddings override test passed")
    
    test_init_embeddings_uses_dummy_in_ci()
    print("✓ CI dummy embeddings test passed")
    
    test_init_embeddings_with_override()
    print("✓ Override precedence test passed")
    
    test_embeddings_factory_pattern()
    print("✓ Factory pattern test passed")
    
    print("\n✅ All embeddings initialization tests passed!")
