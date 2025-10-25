"""Test configuration and mock setup."""

import os
import sys
import types
from typing import Any, Dict, List, Optional

# Module-level singleton for the mock so tests that instantiate a new
# MockOllama() without passing it to setup_test_modules() still affect
# the same underlying instance used by the providers override.
_MOCK_OLLAMA_SINGLETON: Optional["MockOllama"] = None

# Mock Ollama LLM
class MockOllama:
    """Mock Ollama LLM implementation."""
    def __new__(cls, *args, **kwargs):
        global _MOCK_OLLAMA_SINGLETON
        if _MOCK_OLLAMA_SINGLETON is not None:
            return _MOCK_OLLAMA_SINGLETON
        inst = super().__new__(cls)
        _MOCK_OLLAMA_SINGLETON = inst
        return inst

    def __init__(self, model: Optional[str] = None):
        self.model = model or "mock"
        self._mock_response = "mock response"
        self._is_mock = True  # Flag to identify as mock LLM
        
    def _get_text_from_msgs(self, msgs: List[Any]) -> str:
        """Extract text from messages list."""
        if not msgs:
            return ""
        if isinstance(msgs, str):
            return msgs
        # Handle both string and message dict formats
        msg = msgs[-1] if isinstance(msgs, list) else msgs
        if isinstance(msg, str):
            return msg
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    
    def generate(self, msgs: List[Any]) -> Any:
        """Mock generate method."""
        class Generation:
            def __init__(self, text: str):
                self.text = text
                
        class GenerationResult:
            def __init__(self, text: str):
                self.generations = [[Generation(text)]]
                
        # Support both string and message dict formats
        input_text = self._get_text_from_msgs(msgs)
        result = self._handle_input(input_text)
        return GenerationResult(result)
        
    async def ainvoke(self, prompt: Any) -> str:
        """Mock async invoke."""
        text = self._get_text_from_msgs(prompt)
        return self._handle_input(text)
        
    def invoke(self, prompt: Any) -> str:
        """Mock sync invoke with flexible input."""
        text = self._get_text_from_msgs(prompt)
        return self._handle_input(text)
    
    def _handle_input(self, prompt: str) -> str:
        """Process input and return appropriate mock response."""
        # If tests have overridden the mock response, honor it first.
        if self._mock_response != "mock response":
            return self._mock_response

        # For prompts looking for a query, return a valid canned query
        if "return a concise search query" in str(prompt).lower():
            return "long-travel suspension offroad touring"

        # Default mock response
        return self._mock_response
        
    def set_mock_response(self, response: str) -> None:
        """Set the mock response for testing."""
        self._mock_response = response
        
    def chat(self, *args, **kwargs) -> str:
        """Support chat interface."""
        msgs = kwargs.get("messages", args[0] if args else "")
        text = self._get_text_from_msgs(msgs)
        return self._handle_input(text)
        
    def complete(self, *args, **kwargs) -> str:
        """Support complete interface."""
        return self._handle_input(args[0] if args else kwargs.get("prompt", ""))
        
    def __call__(self, *args, **kwargs) -> str:
        """Support callable interface."""
        return self.invoke(args[0] if args else kwargs.get("prompt", ""))

# Mock module configurations
def setup_test_modules(mock_llm: Optional[MockOllama] = None) -> None:
    """Set up mock module imports for tests.
    
    Args:
        mock_llm: Optional instance of MockOllama to use. If not provided,
                 a new instance will be created.
    """
    # Configure environment for testing
    os.environ["MODEL_PROVIDER"] = "ollama"  # Force Ollama provider
    
    # Use provided mock or create new one
    mock_instance = mock_llm if mock_llm is not None else MockOllama()
    
    # Reset any existing mocks
    keys_to_clear = [
        'langchain', 'langchain.chains', 'langchain.prompts',
        'langchain_community', 'langchain_community.vectorstores',
        'langchain_ollama', 'langchain_ollama.llms',
        'chromadb'
    ]
    for key in keys_to_clear:
        if key in sys.modules:
            del sys.modules[key]

    # Set up clean mock modules
    mock_ollama = types.ModuleType('langchain_ollama.llms')
    mock_ollama.OllamaLLM = lambda model=None: mock_instance  # Return same instance
    
    sys.modules.update({
        'langchain': types.SimpleNamespace(),
        'langchain.chains': types.SimpleNamespace(),
        'langchain.prompts': types.SimpleNamespace(),
        'langchain_community': types.SimpleNamespace(),
        'langchain_community.vectorstores': types.SimpleNamespace(
            Chroma=lambda *args, **kwargs: None
        ),
        'chromadb': types.SimpleNamespace(
            Client=lambda **kwargs: None,
            PersistentClient=lambda **kwargs: None
        ),
        'langchain_ollama': types.SimpleNamespace(),
        'langchain_ollama.llms': mock_ollama
    })
    # Also update the in-memory providers module so get_llm() picks up the mock
    try:
        # Import the providers module and override its OllamaLLM symbol
        import importlib
        providers = importlib.import_module('src.llm.providers')
        providers.OllamaLLM = mock_ollama.OllamaLLM
    except Exception:
        # If providers is not yet imported or cannot be set, tests will still
        # pick up the mock via sys.modules for future imports.
        pass

# Mock Retriever 
class MockRetriever:
    """Mock vector store retriever."""
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self._documents = documents or []
        
    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Mock document retrieval."""
        return self._documents
        
    def invoke(self, prompt: str) -> List[Dict[str, Any]]:
        """Mock invoke method."""
        return self._documents
        
    def set_mock_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Set mock documents for testing."""
        self._documents = documents