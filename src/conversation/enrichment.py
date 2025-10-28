"""Response enrichment with metadata from reviews."""

import re
import string
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

from ..core.models import (
    MotorcyclePick, MotorcycleReview, Recommendation
)


def _aggressive_normalize(s: Optional[str]) -> str:
    """Aggressively normalize a string for matching.
    
    - Converts to lowercase
    - Strips leading/trailing whitespace
    - Replaces hyphens with spaces (before removing punctuation)
    - Removes all punctuation
    - Normalizes internal whitespace to single spaces
    - Removes common filler words
    
    Args:
        s: String to normalize
        
    Returns:
        Normalized string
    """
    if not s:
        return ""
    
    # Convert to lowercase and strip
    normalized = s.lower().strip()
    
    # Replace hyphens with spaces before removing punctuation
    # This ensures "790-Adventure" becomes "790 adventure" not "790adventure"
    normalized = normalized.replace('-', ' ')
    
    # Remove punctuation
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace (multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common filler words that don't help matching
    filler_words = {'the', 'a', 'an'}
    tokens = normalized.split()
    tokens = [t for t in tokens if t not in filler_words]
    
    return ' '.join(tokens)


def _fuzzy_match_score(s1: str, s2: str) -> float:
    """Calculate fuzzy match score between two strings.
    
    Uses sequence matching to handle minor variations in spelling.
    
    Args:
        s1: First string (already normalized)
        s2: Second string (already normalized)
        
    Returns:
        Float between 0.0 and 1.0, where 1.0 is perfect match
    """
    if not s1 or not s2:
        return 0.0
    
    # Exact match
    if s1 == s2:
        return 1.0
    
    # Substring match (one contains the other)
    if s1 in s2 or s2 in s1:
        return 0.9
    
    # Token overlap - split into tokens and check overlap
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        if union > 0:
            token_score = overlap / union
            if token_score > 0.5:  # At least 50% overlap
                return 0.7 + (token_score - 0.5) * 0.4  # Scale to 0.7-0.9
    
    # Sequence similarity (edit distance based)
    seq_score = SequenceMatcher(None, s1, s2).ratio()
    
    return seq_score


def _find_best_matching_review(
    brand: str,
    model: str,
    year: Optional[str],
    reviews: List[MotorcycleReview]
) -> Optional[MotorcycleReview]:
    """Find the best matching review using fuzzy matching.
    
    Args:
        brand: Normalized brand name
        model: Normalized model name
        year: Normalized year (optional)
        reviews: List of reviews to search
        
    Returns:
        Best matching review or None
    """
    best_match = None
    best_score = 0.0
    
    for r in reviews:
        rb = _aggressive_normalize(r.brand)
        rm = _aggressive_normalize(r.model)
        ry = _aggressive_normalize(str(r.year)) if r.year is not None else ""
        
        # Calculate component scores
        brand_score = _fuzzy_match_score(brand, rb) if brand else 0.0
        model_score = _fuzzy_match_score(model, rm) if model else 0.0
        year_score = _fuzzy_match_score(year, ry) if year and ry else 0.0
        
        # Weighted total score
        # Model is most important (0.5), brand is important (0.35), year is bonus (0.15)
        if model:
            if brand:
                total_score = model_score * 0.5 + brand_score * 0.35
                if year and ry:
                    total_score += year_score * 0.15
                else:
                    # Redistribute year weight to model and brand
                    total_score = model_score * 0.575 + brand_score * 0.425
            else:
                # Model only match
                total_score = model_score * 0.85
                if year and ry:
                    total_score += year_score * 0.15
        elif brand:
            # Brand only match (less reliable)
            total_score = brand_score * 0.7
            if year and ry:
                total_score += year_score * 0.3
        else:
            continue  # Need at least brand or model
        
        # Require minimum score threshold to prevent poor matches
        if total_score > best_score and total_score >= 0.6:
            best_score = total_score
            best_match = r
    
    return best_match


def enrich_picks_with_metadata(
    parsed: Union[Recommendation, Dict],
    top_reviews: List[MotorcycleReview]
) -> Union[Recommendation, Dict]:
    """Enrich recommendation picks with metadata from reviews.

    Adds evidence from review metadata when a pick lacks it.

    Args:
        parsed: The recommendation response to enrich
        top_reviews: List of relevant reviews to pull metadata from

    Returns:
        The enriched recommendation response
    """
    try:
        if not isinstance(parsed, (dict, Recommendation)):
            return parsed

        if isinstance(parsed, Dict):
            if parsed.get("type") != "recommendation":
                return parsed
        else:
            if not isinstance(parsed, Recommendation):
                return parsed

        def evidence_from_review(r: MotorcycleReview) -> Optional[Tuple[str, str]]:
            """Extract evidence and its source from a review."""
            # Priority order for evidence sources
            if r.suspension_notes:
                return r.suspension_notes, "suspension_notes"
            if r.engine_cc:
                return f"{r.engine_cc} cc", "engine_cc"
            if r.ride_type:
                return r.ride_type, "ride_type"
            if r.price_usd_estimate:
                return f"Price est ${r.price_usd_estimate}", "price_usd_estimate"
            if r.comment:
                return (r.comment or "")[:200], "comment"
            if r.text:
                return (r.text or "")[:200], "text"
            return None

        def enrich_pick(p: Union[MotorcyclePick, Dict], reviews: List[MotorcycleReview]) -> None:
            """Enrich a single pick with metadata using fuzzy matching."""
            # Skip if pick already has valid evidence
            if isinstance(p, Dict):
                ev = p.get("evidence", "") or ""
            else:
                ev = p.evidence or ""

            if isinstance(ev, str) and ev.strip().lower() not in (
                "", "none", "none in dataset", "n/a", "na"
            ):
                return

            # Extract and normalize pick identifiers
            if isinstance(p, Dict):
                brand = _aggressive_normalize(p.get("brand"))
                model = _aggressive_normalize(p.get("model"))
                year = _aggressive_normalize(str(p.get("year"))) if p.get("year") is not None else None
            else:
                brand = _aggressive_normalize(p.brand)
                model = _aggressive_normalize(p.model)
                year = _aggressive_normalize(str(p.year)) if p.year is not None else None

            # Find best matching review using fuzzy matching
            found = _find_best_matching_review(brand, model, year, reviews)

            # Extract evidence if review found
            if found:
                ev_result = evidence_from_review(found)
                if ev_result:
                    evidence_text, source_field = ev_result
                    if isinstance(p, Dict):
                        p["evidence"] = evidence_text
                        p["evidence_source"] = source_field
                    else:
                        p.evidence = evidence_text
                        p.evidence_source = source_field
                    return

            # Set explicit 'none in dataset' if no evidence found
            if isinstance(p, Dict):
                p["evidence"] = "none in dataset"
            else:
                p.evidence = "none in dataset"

        # Handle both old and new response formats
        if isinstance(parsed, Dict):
            if "picks" in parsed:
                # Old format
                picks = parsed.get("picks", []) or []
                for p in picks:
                    enrich_pick(p, top_reviews)
                parsed["picks"] = picks
            else:
                # New format
                primary = parsed.get("primary")
                if primary:
                    enrich_pick(primary, top_reviews)

                alternatives = parsed.get("alternatives", []) or []
                for alt in alternatives:
                    enrich_pick(alt, top_reviews)
        else:
            # Recommendation model
            if parsed.primary:
                enrich_pick(parsed.primary, top_reviews)
            
            for alt in parsed.alternatives:
                enrich_pick(alt, top_reviews)

        return parsed

    except Exception as e:
        # If enrichment fails, log the error and return original
        import logging
        logging.getLogger(__name__).exception("enrich_picks_with_metadata failed: %s", e)
        return parsed