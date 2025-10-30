"""Integration test for OpenAI provider.

This test verifies that the system works end-to-end with OpenAI as the LLM
and embeddings provider. It's skipped unless OPENAI_API_KEY is available.

To run this test:
    export OPENAI_API_KEY=sk-...
    export MODEL_PROVIDER=openai
    pytest tests/test_openai_integration.py -v
"""

import os
import pytest
from unittest.mock import patch

from src.core.config import MODEL_PROVIDER, get_openai_api_key, DEFAULT_SEARCH_KWARGS
from src.llm.providers import get_llm, invoke_model_with_prompt
from src.vector.embeddings import init_embeddings
from src.vector.store import init_vector_store, load_vector_store
from src.vector.retriever import EnhancedVectorStoreRetriever


# Skip entire module if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping OpenAI integration tests"
)


class TestOpenAIIntegration:
    """Test suite for OpenAI provider integration."""

    def test_openai_api_key_required(self):
        """Test that get_openai_api_key returns a valid key."""
        key = get_openai_api_key()
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0
        assert key.startswith("sk-")  # OpenAI keys start with sk-

    def test_openai_api_key_error_when_missing(self):
        """Test that missing API key raises helpful error when provider is openai."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False), \
             patch("src.core.config.MODEL_PROVIDER", "openai"):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                get_openai_api_key()

    def test_openai_embeddings_initialization(self):
        """Test that OpenAI embeddings can be initialized."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            embeddings = init_embeddings()
            assert embeddings is not None
            
            # Test embedding a simple query
            test_text = "fast sport bike"
            embedding = embeddings.embed_query(test_text)
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0  # Should have dimensions
            assert all(isinstance(x, (int, float)) for x in embedding)

    def test_openai_llm_initialization(self):
        """Test that OpenAI LLM can be initialized."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            llm = get_llm()
            assert llm is not None
            
            # Verify it's a ChatOpenAI instance
            from langchain_openai import ChatOpenAI
            assert isinstance(llm, ChatOpenAI)

    def test_openai_llm_invocation(self):
        """Test that OpenAI LLM can be invoked with a simple prompt."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            llm = get_llm()
            
            # Simple test prompt that should get a brief response
            prompt = "Say 'Hello' and nothing else."
            response = invoke_model_with_prompt(llm, prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            # OpenAI should respond with something containing "hello"
            assert "hello" in response.lower() or "hi" in response.lower()

    def test_openai_vector_store_creation(self):
        """Test that vector store can be created with OpenAI embeddings."""
        import tempfile
        import shutil
        
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            # Create a temporary directory for the test
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Initialize embeddings
                embeddings = init_embeddings()
                
                # This should create a new vector store with OpenAI embeddings
                vector_store = init_vector_store(
                    collection_name="test_motorcycles",
                    embeddings=embeddings,
                    persist_dir=temp_dir,
                    provider="openai"
                )
                
                assert vector_store is not None
                assert vector_store._collection is not None
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    def test_openai_retriever_search(self):
        """Test that retriever can perform searches with OpenAI."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            # Load the vector store (will use existing DB)
            vector_store = load_vector_store()
            
            # Create retriever with OpenAI settings
            retriever = EnhancedVectorStoreRetriever(
                vectorstore=vector_store,
                search_kwargs=DEFAULT_SEARCH_KWARGS,
                provider="openai",
                batch_size=10
            )
            
            assert retriever is not None
            assert retriever.provider == "openai"
            
            # Perform a search
            query = "fast sport bikes under $15000"
            results = retriever.get_relevant_documents(query)
            
            # Should return some results (or empty list if no matches)
            assert isinstance(results, list)
            # Each result should be a document with page_content and metadata
            for doc in results:
                assert hasattr(doc, "page_content")
                assert hasattr(doc, "metadata")

    def test_openai_end_to_end_simple(self):
        """Test simple end-to-end flow with OpenAI provider."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            # Step 1: Load vector store and create retriever
            vector_store = load_vector_store()
            retriever = EnhancedVectorStoreRetriever(
                vectorstore=vector_store,
                search_kwargs=DEFAULT_SEARCH_KWARGS,
                provider="openai",
                batch_size=10
            )
            assert retriever is not None
            assert retriever.provider == "openai"
            
            # Step 2: Search for bikes
            user_query = "comfortable touring bike under $12000"
            docs = retriever.get_relevant_documents(user_query)
            
            # Should return list of documents (may be empty if no matches)
            assert isinstance(docs, list)
            
            # Step 3: Test LLM invocation with simple prompt
            llm = get_llm()
            
            # Create a simple test prompt
            if docs:
                # If we have docs, test with real data
                doc_texts = "\n".join([f"- {d.page_content[:100]}..." for d in docs[:2]])
                test_prompt = f"Given these motorcycles:\n{doc_texts}\n\nWhich one is best for touring? Answer briefly."
            else:
                # If no docs, just test basic LLM response
                test_prompt = "List one popular motorcycle brand. Answer with just the brand name."
            
            response = invoke_model_with_prompt(llm, test_prompt)
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Basic validation that we got a real response
            assert len(response.split()) > 0  # Has at least one word


class TestOpenAIConfiguration:
    """Test configuration handling for OpenAI provider."""

    def test_openai_model_config_values(self):
        """Test that OpenAI model configuration values are reasonable."""
        from src.core.config import OPENAI_MODEL, OPENAI_EMBEDDINGS_MODEL
        
        # Check LLM model
        assert OPENAI_MODEL is not None
        assert isinstance(OPENAI_MODEL, str)
        assert len(OPENAI_MODEL) > 0
        # Common OpenAI models
        assert any(x in OPENAI_MODEL for x in ["gpt-3.5", "gpt-4", "gpt"])
        
        # Check embeddings model
        assert OPENAI_EMBEDDINGS_MODEL is not None
        assert isinstance(OPENAI_EMBEDDINGS_MODEL, str)
        assert len(OPENAI_EMBEDDINGS_MODEL) > 0
        # Common OpenAI embedding models
        assert "embedding" in OPENAI_EMBEDDINGS_MODEL.lower()

    @pytest.mark.skipif(
        os.getenv("OPENAI_API_KEY") is None,
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_provider_environment_variable(self):
        """Test that MODEL_PROVIDER can be set to openai."""
        with patch("src.core.config.MODEL_PROVIDER", "openai"):
            from src.core.config import MODEL_PROVIDER
            
            # After patching, should be able to access openai provider
            llm = get_llm()
            assert llm is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
