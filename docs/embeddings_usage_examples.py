"""
Example usage of the improved embeddings initialization system.

The embeddings module now provides:
1. Clear logging of initialization decisions
2. Test override mechanism via factory pattern
3. Deterministic behavior for testing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# For this example, force dummy embeddings so it works without Ollama
os.environ['USE_DUMMY_EMBEDDINGS'] = '1'

# Example 1: Normal usage (unchanged from before)
from src.vector.embeddings import init_embeddings

# Initialize embeddings - will use configured provider
embeddings = init_embeddings()
vector = embeddings.embed_query("test motorcycle")
print(f"✓ Example 1: Vector dimension: {len(vector)}")


# Example 2: Override for testing
from src.vector.embeddings import set_embeddings_override, DummyEmbeddings

# Set up a deterministic embeddings instance for testing
test_embeddings = DummyEmbeddings(dim=16)
set_embeddings_override(lambda: test_embeddings)

# Now init_embeddings() will use your override
embeddings = init_embeddings()
assert embeddings == test_embeddings

# Clear override when done
set_embeddings_override(None)


# Example 3: Custom test embeddings
from unittest.mock import Mock

# Create a fully mocked embeddings instance
mock_embeddings = Mock()
mock_embeddings.embed_query.return_value = [0.1] * 32
mock_embeddings.embed_documents.return_value = [[0.1] * 32, [0.2] * 32]

# Use it in tests
set_embeddings_override(lambda: mock_embeddings)
embeddings = init_embeddings()
result = embeddings.embed_query("test")
assert result == [0.1] * 32

# Clean up
set_embeddings_override(None)


# Example 4: Checking override status
from src.vector.embeddings import get_embeddings_override

override = get_embeddings_override()
if override is None:
    print("No override set - using default initialization")
else:
    print("Override is active")


# Example 5: Deterministic embeddings for reproducible tests
emb1 = DummyEmbeddings(dim=8)
emb2 = DummyEmbeddings(dim=8)

# Same text always produces same embedding
text = "adventure motorcycle"
vec1 = emb1.embed_query(text)
vec2 = emb2.embed_query(text)
assert vec1 == vec2  # Deterministic!

print("✓ All examples completed successfully")
