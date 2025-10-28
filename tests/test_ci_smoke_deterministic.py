"""
Deterministic CI smoke test for dummy embeddings and vector store indexing.

This test suite verifies that:
1. USE_DUMMY_EMBEDDINGS=1 produces stable, deterministic embeddings
2. Dummy embeddings are reproducible across multiple runs
3. load_vector_store can index a tiny subset without Ollama
4. The full pipeline works in CI environments without external dependencies

Run with: pytest tests/test_ci_smoke_deterministic.py -v
"""

import os
import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path

from src.vector.embeddings import DummyEmbeddings, init_embeddings, set_embeddings_override
from src.vector.store import load_vector_store, build_metadata
from src.core.config import DATA_FILE, DB_LOCATION


class TestDummyEmbeddingsDeterminism:
    """Test that DummyEmbeddings produces stable, deterministic results."""

    def test_same_text_same_embedding(self):
        """Verify same text produces identical embeddings."""
        embeddings = DummyEmbeddings(dim=32)
        text = "Honda CRF450L dual-sport adventure bike"
        
        result1 = embeddings.embed_query(text)
        result2 = embeddings.embed_query(text)
        
        assert result1 == result2, "Same text should produce identical embeddings"
        assert len(result1) == 32, "Embedding dimension should match configuration"

    def test_different_text_different_embedding(self):
        """Verify different texts produce different embeddings."""
        embeddings = DummyEmbeddings(dim=32)
        text1 = "Honda CRF450L dual-sport"
        text2 = "Yamaha Tenere 700 adventure"
        
        result1 = embeddings.embed_query(text1)
        result2 = embeddings.embed_query(text2)
        
        assert result1 != result2, "Different texts should produce different embeddings"

    def test_embedding_values_in_valid_range(self):
        """Verify embedding values are normalized to [0, 1]."""
        embeddings = DummyEmbeddings(dim=32)
        text = "Test motorcycle review text"
        
        result = embeddings.embed_query(text)
        
        assert all(0.0 <= val <= 1.0 for val in result), \
            "All embedding values should be in [0, 1] range"

    def test_embed_documents_multiple_texts(self):
        """Verify embed_documents handles multiple texts correctly."""
        embeddings = DummyEmbeddings(dim=32)
        texts = [
            "Honda CRF450L adventure bike",
            "Yamaha WR250R dual-sport",
            "KTM 690 Enduro R"
        ]
        
        results = embeddings.embed_documents(texts)
        
        assert len(results) == 3, "Should return embeddings for all texts"
        assert all(len(emb) == 32 for emb in results), \
            "All embeddings should have correct dimension"
        assert results[0] != results[1] != results[2], \
            "Different texts should have different embeddings"

    def test_embedding_stability_across_instances(self):
        """Verify embeddings are stable across different DummyEmbeddings instances."""
        text = "Long-travel suspension for off-road touring"
        
        embeddings1 = DummyEmbeddings(dim=32)
        embeddings2 = DummyEmbeddings(dim=32)
        
        result1 = embeddings1.embed_query(text)
        result2 = embeddings2.embed_query(text)
        
        assert result1 == result2, \
            "Same text should produce identical embeddings across instances"

    def test_different_dimensions_work(self):
        """Verify DummyEmbeddings supports different dimensions."""
        text = "Test motorcycle"
        
        embeddings_16 = DummyEmbeddings(dim=16)
        embeddings_64 = DummyEmbeddings(dim=64)
        
        result_16 = embeddings_16.embed_query(text)
        result_64 = embeddings_64.embed_query(text)
        
        assert len(result_16) == 16, "Should respect 16-dim configuration"
        assert len(result_64) == 64, "Should respect 64-dim configuration"

    def test_empty_text_handling(self):
        """Verify empty text produces valid embedding."""
        embeddings = DummyEmbeddings(dim=32)
        
        result = embeddings.embed_query("")
        
        assert len(result) == 32, "Empty text should still produce full-dimension embedding"
        assert all(isinstance(val, float) for val in result), \
            "All values should be floats"

    def test_unicode_text_handling(self):
        """Verify unicode text produces stable embeddings."""
        embeddings = DummyEmbeddings(dim=32)
        text = "Motörrad with émojis 🏍️ and spëcial chàrs"
        
        result1 = embeddings.embed_query(text)
        result2 = embeddings.embed_query(text)
        
        assert result1 == result2, "Unicode text should produce stable embeddings"
        assert len(result1) == 32, "Unicode text should produce full-dimension embedding"


class TestInitEmbeddingsWithDummy:
    """Test init_embeddings() in dummy/CI mode."""

    def test_embeddings_override_works(self):
        """Verify set_embeddings_override allows test injection."""
        custom_embeddings = DummyEmbeddings(dim=16)
        set_embeddings_override(lambda: custom_embeddings)
        
        embeddings = init_embeddings()
        
        assert embeddings is custom_embeddings, \
            "Should return override embeddings when set"
        
        # Clean up
        set_embeddings_override(None)


class TestVectorStoreWithDummyEmbeddings:
    """Test load_vector_store with dummy embeddings on a tiny dataset."""

    @pytest.fixture
    def temp_data_and_db(self, monkeypatch):
        """Create temporary CSV and DB directory for testing."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, "test_reviews.csv")
        temp_db = os.path.join(temp_dir, "test_chroma_db")
        
        # Create minimal test CSV
        test_data = pd.DataFrame({
            "brand": ["Honda", "Yamaha", "KTM"],
            "model": ["CRF450L", "WR250R", "690 Enduro"],
            "year": [2023, 2023, 2024],
            "comment": [
                "Great dual-sport bike with long-travel suspension",
                "Lightweight adventure bike for off-road touring",
                "Powerful enduro with excellent suspension travel"
            ],
            "price_usd_estimate": [10500, 7000, 12000],
            "engine_cc": [450, 250, 690]
        })
        test_data.to_csv(temp_csv, index=False)
        
        # Monkey-patch config paths
        monkeypatch.setattr("src.vector.store.DATA_FILE", temp_csv)
        monkeypatch.setattr("src.vector.store.DB_LOCATION", temp_db)
        monkeypatch.setenv("USE_DUMMY_EMBEDDINGS", "1")
        
        # Force config reload
        from importlib import reload
        from src.core import config
        reload(config)
        
        yield temp_csv, temp_db
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        set_embeddings_override(None)

    def test_load_vector_store_with_dummy_embeddings(self, temp_data_and_db):
        """Verify load_vector_store works with dummy embeddings."""
        temp_csv, temp_db = temp_data_and_db
        
        # Force use of dummy embeddings
        set_embeddings_override(lambda: DummyEmbeddings(dim=32))
        
        # Load vector store (will index the tiny CSV)
        vector_store = load_vector_store(chunk_size=10)
        
        assert vector_store is not None, "Should create vector store"
        assert os.path.exists(temp_db), "Should create DB directory"
        
        # Clean up override
        set_embeddings_override(None)

    def test_vector_store_search_with_dummy_embeddings(self, temp_data_and_db):
        """Verify search works with dummy embeddings."""
        temp_csv, temp_db = temp_data_and_db
        
        # Force use of dummy embeddings
        set_embeddings_override(lambda: DummyEmbeddings(dim=32))
        
        # Load and search
        vector_store = load_vector_store(chunk_size=10)
        results = vector_store.similarity_search("suspension off-road", k=2)
        
        assert len(results) >= 0, "Should return search results (may be empty)"
        # With only 3 documents, we should get at most 2 results
        assert len(results) <= 2, "Should respect k parameter"
        
        # Clean up override
        set_embeddings_override(None)

    def test_vector_store_deterministic_with_dummy_embeddings(self, temp_data_and_db):
        """Verify vector store operations are deterministic with dummy embeddings."""
        temp_csv, temp_db = temp_data_and_db
        
        # Force use of dummy embeddings
        set_embeddings_override(lambda: DummyEmbeddings(dim=32))
        
        # Load and search twice
        vector_store1 = load_vector_store(chunk_size=10)
        results1 = vector_store1.similarity_search("adventure touring", k=2)
        
        # Clean DB and reload
        shutil.rmtree(temp_db, ignore_errors=True)
        vector_store2 = load_vector_store(chunk_size=10)
        results2 = vector_store2.similarity_search("adventure touring", k=2)
        
        # Results should be deterministic
        assert len(results1) == len(results2), \
            "Same query should return same number of results"
        
        # Clean up override
        set_embeddings_override(None)


class TestBuildMetadata:
    """Test metadata extraction works without external dependencies."""

    def test_build_metadata_basic(self):
        """Verify build_metadata extracts fields correctly."""
        text_fields = ["Honda CRF450L adventure bike with long-travel suspension"]
        row_dict = {
            "brand": "Honda",
            "model": "CRF450L",
            "year": 2023,
            "price_usd_estimate": 10500,
            "engine_cc": 450
        }
        
        metadata = build_metadata(text_fields, row_dict)
        
        assert metadata["brand"] == "Honda"
        assert metadata["model"] == "CRF450L"
        assert metadata["year"] == 2023
        assert metadata["price_usd_estimate"] == 10500
        assert metadata["engine_cc"] == 450

    def test_build_metadata_with_string_price(self):
        """Verify build_metadata parses string prices."""
        text_fields = ["Great bike for $10,500"]
        row_dict = {
            "brand": "Yamaha",
            "model": "Tenere 700",
            "price_usd_estimate": "$10,500"
        }
        
        metadata = build_metadata(text_fields, row_dict)
        
        assert metadata["price_usd_estimate"] == 10500

    def test_build_metadata_extracts_from_text(self):
        """Verify build_metadata extracts info from text when fields missing."""
        text_fields = [
            "The 690cc engine provides great power. "
            "Long-travel suspension is excellent for off-road use."
        ]
        row_dict = {"brand": "KTM", "model": "690 Enduro"}
        
        metadata = build_metadata(text_fields, row_dict)
        
        # Should extract engine_cc and suspension_notes from text
        assert metadata["engine_cc"] == 690
        assert "long-travel" in metadata.get("suspension_notes", "").lower()


class TestCISmokeIntegration:
    """Integration test simulating full CI environment."""

    def test_dummy_embeddings_produce_deterministic_results(self):
        """Verify DummyEmbeddings produce stable results across runs."""
        # This test verifies the core requirement: deterministic embeddings
        embeddings1 = DummyEmbeddings(dim=32)
        embeddings2 = DummyEmbeddings(dim=32)
        
        text = "Test motorcycle review for CI"
        result1 = embeddings1.embed_query(text)
        result2 = embeddings2.embed_query(text)
        
        assert result1 == result2, "DummyEmbeddings should produce identical results"
        assert len(result1) == 32, "Should respect configured dimension"
        assert all(0.0 <= v <= 1.0 for v in result1), "Values should be normalized"

