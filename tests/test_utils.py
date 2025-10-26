"""Test utilities for mocking dependencies and common setup"""

import sys
import types
from typing import Any, Dict, List, Optional


def setup_test_dependencies() -> None:
    """Set up mock dependencies for tests"""
    
    # Mock langchain_ollama.llms module
    ll_ms = types.SimpleNamespace()
    class FakeOllama:
        def __init__(self, model: Optional[str] = None):
            self.model = model
            
        async def ainvoke(self, prompt: str) -> str:
            """Mock async invoke method"""
            return "Mocked response"
            
        def invoke(self, prompt: str) -> str:
            """Mock sync invoke method"""
            return "Mocked response"
            
        def generate(self, msgs: list) -> Any:
            """Mock generate method for backward compatibility"""
            class G:
                def __init__(self):
                    self.generations = [[types.SimpleNamespace(text="mock response")]]
            return G()
    
    ll_ms.OllamaLLM = FakeOllama
    sys.modules['langchain_ollama'] = types.SimpleNamespace()
    sys.modules['langchain_ollama.llms'] = ll_ms

    # Mock ChromaDB
    class FakeChroma:
        def from_documents(self, *args, **kwargs):
            return self
            
        def as_retriever(self, *args, **kwargs):
            return FakeRetriever()
    
    class FakeRetriever:
        def invoke(self, prompt: str) -> List[Dict[str, Any]]:
            """Mock invoke method returning empty results"""
            return []
            
        def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
            """Mock get_relevant_documents method"""
            return []
    
    sys.modules['chromadb'] = types.SimpleNamespace(
        Client=lambda **kwargs: None,
        PersistentClient=lambda **kwargs: None
    )
    sys.modules['langchain_community'] = types.SimpleNamespace()
    sys.modules['langchain_community.vectorstores'] = types.SimpleNamespace(
        Chroma=FakeChroma
    )
    
    # Mock vector store retriever
    class FakeVectorStore:
        def __init__(self):
            self.retriever = FakeRetriever()
            
        def as_retriever(self):
            return self.retriever
    
    sys.modules['src.vector.store'] = types.SimpleNamespace(
        init_vector_store=lambda *args, **kwargs: FakeVectorStore(),
        # Add attributes that other tests may need to patch
        DATA_FILE='mock_data.csv',
        DB_LOCATION='./mock_db',
        MODEL_PROVIDER='ollama',
        os=__import__('os'),
        init_embeddings=lambda: None,
        load_vector_store=lambda *args, **kwargs: FakeVectorStore()
    )