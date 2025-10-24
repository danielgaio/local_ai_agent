"""Response enrichment with metadata from reviews."""

from typing import Dict, List, Optional, Tuple, Union

from ..core.models import (
    MotorcyclePick, MotorcycleReview, Recommendation
)


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

        def _normalize(s: Optional[str]) -> str:
            return (s or "").strip().lower()

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
            """Enrich a single pick with metadata."""
            # Skip if pick already has valid evidence
            if isinstance(p, Dict):
                ev = p.get("evidence", "") or ""
            else:
                ev = p.evidence or ""

            if isinstance(ev, str) and ev.strip().lower() not in (
                "", "none", "none in dataset", "n/a", "na"
            ):
                return

            # Extract pick identifiers
            if isinstance(p, Dict):
                brand = _normalize(p.get("brand"))
                model = _normalize(p.get("model"))
                year = _normalize(str(p.get("year"))) if p.get("year") is not None else ""
            else:
                brand = _normalize(p.brand)
                model = _normalize(p.model)
                year = _normalize(str(p.year)) if p.year is not None else ""

            # Find matching review
            found = None
            for r in reviews:
                rb = _normalize(r.brand)
                rm = _normalize(r.model)
                ry = _normalize(str(r.year)) if r.year is not None else ""

                # Try brand+model match
                if brand and model:
                    if brand in rb and model in rm or rb in brand and rm in model:
                        found = r
                        break
                # Try model-only match
                elif model and (model in rm or rm in model):
                    found = r
                    break
                # Try brand-only match
                elif brand and (brand in rb or rb in brand):
                    found = r
                    break

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

    except Exception:
        # If enrichment fails, return original
        return parsed