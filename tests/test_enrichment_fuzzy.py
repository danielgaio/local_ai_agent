"""
Tests for fuzzy matching in enrichment logic.

Verifies that the enrichment matching can handle:
- Punctuation variations (e.g., "790 Adventure" vs "790-Adventure")
- Case variations
- Partial matches
- Model year mismatches
- Minor spelling variations
- Token reordering
"""

import pytest
from src.conversation.enrichment import (
    _aggressive_normalize,
    _fuzzy_match_score,
    _find_best_matching_review,
    enrich_picks_with_metadata
)
from src.core.models import MotorcycleReview, MotorcyclePick, Recommendation


class TestAggressiveNormalization:
    """Test the aggressive normalization function."""
    
    def test_removes_punctuation(self):
        """Verify punctuation is removed."""
        assert _aggressive_normalize("KTM 790-Adventure") == "ktm 790 adventure"
        assert _aggressive_normalize("F850GS") == "f850gs"
        assert _aggressive_normalize("CB500X") == "cb500x"
    
    def test_normalizes_whitespace(self):
        """Verify whitespace is normalized."""
        assert _aggressive_normalize("Honda  CB500X") == "honda cb500x"
        assert _aggressive_normalize("  Yamaha MT-07  ") == "yamaha mt 07"
    
    def test_removes_filler_words(self):
        """Verify common filler words are removed."""
        assert _aggressive_normalize("The BMW GS") == "bmw gs"
        assert _aggressive_normalize("A Yamaha MT") == "yamaha mt"
        assert _aggressive_normalize("An adventure bike") == "adventure bike"
    
    def test_handles_none_and_empty(self):
        """Verify None and empty strings are handled."""
        assert _aggressive_normalize(None) == ""
        assert _aggressive_normalize("") == ""
        assert _aggressive_normalize("   ") == ""
    
    def test_case_insensitive(self):
        """Verify case normalization."""
        assert _aggressive_normalize("KTM") == "ktm"
        assert _aggressive_normalize("BMW") == "bmw"


class TestFuzzyMatchScore:
    """Test the fuzzy matching score function."""
    
    def test_exact_match(self):
        """Verify exact matches score 1.0."""
        assert _fuzzy_match_score("ktm 790 adventure", "ktm 790 adventure") == 1.0
        assert _fuzzy_match_score("bmw", "bmw") == 1.0
    
    def test_substring_match(self):
        """Verify substring matches score high."""
        score = _fuzzy_match_score("790 adventure", "ktm 790 adventure")
        assert score >= 0.9
        
        score = _fuzzy_match_score("ktm 790", "790")
        assert score >= 0.9
    
    def test_token_overlap(self):
        """Verify token overlap scoring."""
        # 2 out of 3 tokens match - sequence similarity will be high due to small diff
        score = _fuzzy_match_score("ktm 790 adventure", "ktm 890 adventure")
        assert score > 0.7  # At least 70% match
        
        # Less overlap should score lower
        score = _fuzzy_match_score("ktm adventure", "yamaha touring")
        assert score < 0.5
    
    def test_sequence_similarity(self):
        """Verify edit distance based scoring."""
        # Minor spelling variation
        score = _fuzzy_match_score("adventure", "adventur")
        assert score > 0.8
        
        # More significant difference
        score = _fuzzy_match_score("adventure", "touring")
        assert score < 0.5
    
    def test_empty_strings(self):
        """Verify empty strings return 0.0."""
        assert _fuzzy_match_score("", "test") == 0.0
        assert _fuzzy_match_score("test", "") == 0.0
        assert _fuzzy_match_score("", "") == 0.0


class TestFindBestMatchingReview:
    """Test the best matching review finder."""
    
    def test_exact_brand_model_match(self):
        """Verify exact brand+model match is found."""
        reviews = [
            MotorcycleReview(
                brand="KTM",
                model="790 Adventure",
                year=2019,
                comment="Great bike",
                price_usd_estimate=10000,
                suspension_notes="long-travel",
                text="Adventure ready"
            ),
            MotorcycleReview(
                brand="BMW",
                model="F850GS",
                year=2020,
                comment="Good touring",
                price_usd_estimate=12000,
                text="Touring bike"
            )
        ]
        
        match = _find_best_matching_review("ktm", "790 adventure", None, reviews)
        assert match is not None
        assert match.brand == "KTM"
        assert match.model == "790 Adventure"
    
    def test_punctuation_variation_match(self):
        """Verify matching works with punctuation differences."""
        reviews = [
            MotorcycleReview(
                brand="KTM",
                model="790-Adventure",
                year=2019,
                comment="Great bike",
                price_usd_estimate=10000,
                suspension_notes="long-travel",
                text="Adventure ready"
            )
        ]
        
        # Query without hyphen should still match
        match = _find_best_matching_review("ktm", "790 adventure", None, reviews)
        assert match is not None
        assert match.brand == "KTM"
    
    def test_model_year_mismatch_still_matches(self):
        """Verify model year difference doesn't prevent match."""
        reviews = [
            MotorcycleReview(
                brand="Honda",
                model="CB500X",
                year=2022,
                comment="Reliable",
                price_usd_estimate=7000,
                suspension_notes="comfortable",
                text="Great commuter"
            )
        ]
        
        # Query with different year should still match
        match = _find_best_matching_review("honda", "cb500x", "2023", reviews)
        assert match is not None
        assert match.brand == "Honda"
        assert match.model == "CB500X"
    
    def test_model_only_match(self):
        """Verify model-only matching works."""
        reviews = [
            MotorcycleReview(
                brand="Yamaha",
                model="MT-07",
                year=2021,
                comment="Fun bike",
                price_usd_estimate=8500,
                engine_cc=689,
                text="Torquey parallel twin"
            )
        ]
        
        # Brand empty, should match on model
        match = _find_best_matching_review("", "mt 07", None, reviews)
        assert match is not None
        assert match.model == "MT-07"
    
    def test_partial_model_match(self):
        """Verify partial model name matches."""
        reviews = [
            MotorcycleReview(
                brand="BMW",
                model="R1250GS Adventure",
                year=2021,
                comment="Flagship",
                price_usd_estimate=18000,
                suspension_notes="electronically adjustable",
                text="Top of the line"
            )
        ]
        
        # Partial model name
        match = _find_best_matching_review("bmw", "r1250gs", None, reviews)
        assert match is not None
        assert "R1250GS" in match.model
    
    def test_no_match_below_threshold(self):
        """Verify poor matches return None."""
        reviews = [
            MotorcycleReview(
                brand="Honda",
                model="CB500X",
                year=2022,
                comment="Reliable",
                price_usd_estimate=7000,
                text="Great commuter"
            )
        ]
        
        # Completely different bike
        match = _find_best_matching_review("ducati", "panigale", None, reviews)
        assert match is None
    
    def test_best_match_selection(self):
        """Verify best match is selected from multiple candidates."""
        reviews = [
            MotorcycleReview(
                brand="KTM",
                model="690 Enduro",
                year=2018,
                comment="Similar name",
                price_usd_estimate=9000,
                text="Enduro bike"
            ),
            MotorcycleReview(
                brand="KTM",
                model="790 Adventure",
                year=2019,
                comment="Exact match",
                price_usd_estimate=10000,
                suspension_notes="long-travel",
                text="Adventure ready"
            ),
            MotorcycleReview(
                brand="KTM",
                model="890 Adventure",
                year=2021,
                comment="Similar model",
                price_usd_estimate=11000,
                text="Updated version"
            )
        ]
        
        # Should match 790 Adventure as the best match
        match = _find_best_matching_review("ktm", "790 adventure", None, reviews)
        assert match is not None
        assert match.model == "790 Adventure"


class TestEnrichmentWithFuzzyMatching:
    """Test end-to-end enrichment with fuzzy matching."""
    
    def test_enrichment_with_punctuation_variation(self):
        """Verify enrichment works with punctuation differences."""
        pick = MotorcyclePick(
            brand="KTM",
            model="790 Adventure",  # No hyphen
            year=2019,
            price_est=10000,
            reason="Great suspension",
            evidence=""
        )
        
        reviews = [
            MotorcycleReview(
                brand="KTM",
                model="790-Adventure",  # With hyphen
                year=2019,
                comment="Excellent",
                price_usd_estimate=10000,
                suspension_notes="long-travel, plush",
                text="Adventure ready"
            )
        ]
        
        recommendation = Recommendation(
            type="recommendation",
            primary=pick,
            alternatives=[],
            note=None
        )
        
        enriched = enrich_picks_with_metadata(recommendation, reviews)
        assert enriched.primary.evidence == "long-travel, plush"
        assert enriched.primary.evidence_source == "suspension_notes"
    
    def test_enrichment_with_case_variation(self):
        """Verify enrichment works with case differences."""
        pick = MotorcyclePick(
            brand="honda",  # Lowercase
            model="cb500x",
            year=2022,
            price_est=7000,
            reason="Reliable commuter",
            evidence=""
        )
        
        reviews = [
            MotorcycleReview(
                brand="Honda",  # Capitalized
                model="CB500X",
                year=2022,
                comment="Great bike",
                price_usd_estimate=7000,
                engine_cc=500,
                text="Excellent commuter"
            )
        ]
        
        recommendation = Recommendation(
            type="recommendation",
            primary=pick,
            alternatives=[],
            note=None
        )
        
        enriched = enrich_picks_with_metadata(recommendation, reviews)
        assert enriched.primary.evidence == "500 cc"
        assert enriched.primary.evidence_source == "engine_cc"
    
    def test_enrichment_with_year_mismatch(self):
        """Verify enrichment works when year doesn't match."""
        pick = MotorcyclePick(
            brand="Yamaha",
            model="MT-07",
            year=2023,  # Different year
            price_est=8500,
            reason="Fun bike",
            evidence=""
        )
        
        reviews = [
            MotorcycleReview(
                brand="Yamaha",
                model="MT-07",
                year=2021,  # Different year in review
                comment="Torquey",
                price_usd_estimate=8500,
                ride_type="naked",
                text="Parallel twin"
            )
        ]
        
        recommendation = Recommendation(
            type="recommendation",
            primary=pick,
            alternatives=[],
            note=None
        )
        
        enriched = enrich_picks_with_metadata(recommendation, reviews)
        assert enriched.primary.evidence == "naked"
        assert enriched.primary.evidence_source == "ride_type"
    
    def test_enrichment_with_partial_name_match(self):
        """Verify enrichment works with partial model names."""
        pick = MotorcyclePick(
            brand="BMW",
            model="R1250GS",  # Without "Adventure"
            year=2021,
            price_est=18000,
            reason="Flagship tourer",
            evidence=""
        )
        
        reviews = [
            MotorcycleReview(
                brand="BMW",
                model="R1250GS Adventure",  # Full name
                year=2021,
                comment="Amazing",
                price_usd_estimate=18000,
                suspension_notes="electronically adjustable ESA",
                text="Top tier"
            )
        ]
        
        recommendation = Recommendation(
            type="recommendation",
            primary=pick,
            alternatives=[],
            note=None
        )
        
        enriched = enrich_picks_with_metadata(recommendation, reviews)
        assert enriched.primary.evidence == "electronically adjustable ESA"
        assert enriched.primary.evidence_source == "suspension_notes"
    
    def test_no_match_sets_none_in_dataset(self):
        """Verify 'none in dataset' is set when no match found."""
        pick = MotorcyclePick(
            brand="Ducati",
            model="Panigale V4",
            year=2023,
            price_est=25000,
            reason="Track-focused",
            evidence=""
        )
        
        reviews = [
            MotorcycleReview(
                brand="KTM",
                model="790 Adventure",
                year=2019,
                comment="Different bike",
                price_usd_estimate=10000,
                suspension_notes="long-travel",
                text="Adventure bike"
            )
        ]
        
        recommendation = Recommendation(
            type="recommendation",
            primary=pick,
            alternatives=[],
            note=None
        )
        
        enriched = enrich_picks_with_metadata(recommendation, reviews)
        assert enriched.primary.evidence == "none in dataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
