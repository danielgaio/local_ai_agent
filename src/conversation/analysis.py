"""LLM analysis and response handling."""

import json
from typing import List

from ..core.models import MotorcycleReview
from ..llm.providers import get_llm, invoke_model_with_prompt
from ..llm.prompt_builder import build_llm_prompt
from .validation import validate_and_filter
from .enrichment import enrich_picks_with_metadata


def analyze_with_llm(
    conversation_history: List[str],
    top_reviews: List[MotorcycleReview]
) -> str:
    """Analyze conversation and provide recommendations or questions.

    Args:
        conversation_history: List of user messages
        top_reviews: List of relevant reviews to consider

    Returns:
        str: Formatted response string
    """
    # Build prompt and get response
    prompt = build_llm_prompt(conversation_history, top_reviews)
    response = invoke_model_with_prompt(get_llm(), prompt)

    # Clean response of debug markers
    def _sanitize_raw(text: str) -> str:
        lines = text.splitlines()
        cleaned = [ln for ln in lines
                  if not any(ln.strip().startswith(f"[{t}]")
                          for t in ("DEBUG", "WARN", "ERROR"))]
        return "\n".join(cleaned).strip()

    response = _sanitize_raw(response)

    try:
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        return response

    # Validate and allow one retry
    valid, info = validate_and_filter(parsed, conversation_history)
    if not valid and getattr(info, "action", None) == "retry":
        # Retry with enhanced prompt
        prioritized = getattr(info, "attribute", None)
        retry_msg = (
            "Previous response did not mention the prioritized attribute in any pick. "
            "Please return the SAME JSON schema again, ensuring each pick.reason "
            f"(<=12 words) mentions '{prioritized or 'the prioritized attribute'}' "
            "or set evidence to 'none in dataset'. Also strictly enforce numeric "
            "budget constraints if a budget was provided."
        )
        retry_prompt = prompt + "\n\nRETRY_INSTRUCTION: " + retry_msg
        retry_resp = invoke_model_with_prompt(get_llm(), retry_prompt)
        retry_resp = retry_resp and retry_resp.strip()

        try:
            parsed_retry = json.loads(retry_resp)
            valid2, info2 = validate_and_filter(parsed_retry, conversation_history)
            if valid2:
                parsed = parsed_retry
            else:
                return (
                    f"Model retry failed validation: {getattr(info2, 'reason', None)}. "
                    f"Returning model output for debugging: {retry_resp}"
                )
        except json.JSONDecodeError:
            return f"Model retry did not return valid JSON. Raw retry response: {retry_resp}"

    # Enrich picks with metadata
    try:
        parsed = enrich_picks_with_metadata(parsed, top_reviews)
    except Exception:
        pass

    # Format response for display
    try:
        if parsed.get("type") == "clarify":
            return parsed.get("question", "(no question provided)")
        elif parsed.get("type") == "recommendation":
            lines = []

            if "primary" in parsed:
                # New format
                primary = parsed.get("primary")
                alternatives = parsed.get("alternatives", [])
                
                if primary:
                    lines.append("Top recommendation:")
                    ev = primary.get('evidence')
                    ev_source = primary.get('evidence_source')
                    if ev:
                        ev_text = f" Evidence: {ev}"
                    else:
                        ev_text = ""
                    lines.append(
                        f"• {primary.get('brand')} {primary.get('model')} "
                        f"({primary.get('year')}), Price est: ${primary.get('price_est')}. "
                        f"Reason: {primary.get('reason')}.{ev_text}"
                    )
                    
                    if alternatives:
                        lines.append("\nAlternatives:")
                        for alt in alternatives:
                            lines.append(
                                f"• {alt.get('brand')} {alt.get('model')} "
                                f"({alt.get('year')}) - ${alt.get('price_est')}. "
                                f"{alt.get('reason')}"
                            )
                else:
                    note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                    lines.append(f"No picks matched strictly. Note: {note}")
                    
            else:
                # Old format
                picks = parsed.get("picks", [])
                lines.append("Top recommendations:")
                if not picks:
                    note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                    lines.append(f"No picks matched strictly. Note: {note}")
                else:
                    for p in picks:
                        ev = p.get('evidence')
                        ev_source = p.get('evidence_source')
                        if ev:
                            ev_text = f" Evidence: {ev}"
                        else:
                            ev_text = ""
                        lines.append(
                            f"- {p.get('brand')} {p.get('model')} "
                            f"({p.get('year')}), Price est: ${p.get('price_est')}. "
                            f"Reason: {p.get('reason')}.{ev_text}"
                        )
                        
            if parsed.get("note") and (parsed.get("primary") or parsed.get("picks")):
                lines.append(f"\nNote: {parsed.get('note')}")
            return "\n".join(lines)
        else:
            return response
    except Exception:
        return response