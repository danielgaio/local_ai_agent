"""Main CLI entry point for the motorcycle recommendation system."""

import json
import sys
from typing import List, Optional

from ..core.models import MotorcycleReview
from ..llm.providers import get_llm, invoke_model_with_prompt
from ..llm.prompt_builder import build_llm_prompt
from ..conversation.history import (
    is_vague_input, generate_retriever_query, keyword_extract_query
)
from ..conversation.validation import validate_and_filter
from ..conversation.enrichment import enrich_picks_with_metadata
from ..vector.store import load_vector_store
from ..vector.retriever import StandardVectorStoreRetriever
from ..core.config import DEFAULT_SEARCH_KWARGS, DEBUG


def get_docs_from_retriever(retriever: StandardVectorStoreRetriever, query: str) -> List[MotorcycleReview]:
    """Get relevant reviews from retriever and convert to domain models.

    Args:
        retriever: The configured retriever
        query: Query string to search with

    Returns:
        list: List of MotorcycleReview objects
    """
    docs = retriever.get_relevant_documents(query)
    
    # Convert to MotorcycleReview models
    reviews = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        reviews.append(MotorcycleReview(
            brand=meta.get("brand"),
            model=meta.get("model"),
            year=meta.get("year"),
            comment=meta.get("comment") or getattr(d, "page_content", ""),
            text=getattr(d, "page_content", ""),
            price_usd_estimate=(
                int(meta.get("price_usd_estimate"))
                if meta.get("price_usd_estimate") is not None
                else (
                    int(meta.get("price"))
                    if meta.get("price") is not None
                    else None
                )
            ),
            price_est=(
                int(meta.get("price_usd_estimate"))
                if meta.get("price_usd_estimate") is not None
                else (
                    int(meta.get("price"))
                    if meta.get("price") is not None
                    else None
                )
            ),
            engine_cc=meta.get("engine_cc"),
            suspension_notes=meta.get("suspension_notes"),
            ride_type=meta.get("ride_type")
        ))

    return reviews


def generate_clarifying_question(conversation_history: List[str]) -> Optional[str]:
    """Generate a single clarifying question based on conversation context.

    Args:
        conversation_history: List of user messages

    Returns:
        str: A clarifying question, or None if no question needed
    """
    recent = conversation_history[-4:] if conversation_history else []
    convo_block = "\n".join([f"- {m}" for m in recent])
    prompt = (
        "You are a concise assistant that asks a single short clarifying question "
        "when the user's message is vague.\n"
        "Given the recent conversation, return exactly one short question (one line) "
        "that will help you clarify the user's needs for motorcycle recommendations. "
        "Do not add any extra text.\n\n"
        f"Conversation:\n{convo_block}\n"
    )

    try:
        out = invoke_model_with_prompt(get_llm(), prompt)
        if not out:
            return None

        # Take first non-empty line
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # Ignore greetings
            low = ln.lower()
            if low in ("hi", "hello", "hey") or low.startswith("hi ") or low.startswith("hello "):
                return None
            # Ensure it looks like a question
            if not ln.endswith("?"):
                ln = ln.rstrip('.') + "?"
            return ln

    except Exception:
        return None


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


def main_cli() -> None:
    """Main CLI entry point."""
    # Initialize vector store and retriever
    vector_store = load_vector_store()
    retriever = StandardVectorStoreRetriever(
        vectorstore=vector_store,
        search_kwargs=DEFAULT_SEARCH_KWARGS
    )

    # Initialize conversation history
    conversation_history: List[str] = []

    while True:
        print("\n\n--------------------------------")
        user_preferences = input("What are your motorcycle preferences? (Type 'q' to quit): ")
        print("\nThinking...\n")

        if user_preferences.lower() == 'q':
            break

        conversation_history.append(user_preferences)

        # Check for vague input
        if is_vague_input(user_preferences):
            cq = generate_clarifying_question(conversation_history)
            if cq:
                print("\nClarifying question:\n", cq)
                conversation_history.append(cq)
                continue

        # Generate retrieval query
        try:
            q_res = generate_retriever_query(conversation_history)
            if isinstance(q_res, tuple):
                query, used_fallback = q_res
            else:
                # backward compatibility
                query = q_res
                used_fallback = False

            if not query:
                query = " ".join(conversation_history[-3:] if conversation_history else [user_preferences])

            if used_fallback and DEBUG:
                print("[INFO] Using deterministic fallback query for retriever.")

            # Get relevant reviews
            reviews = get_docs_from_retriever(retriever, query)

        except Exception as e:
            print(f"[ERROR] Failed to query retriever: {e}")
            sys.exit(1)

        # Get recommendation or clarifying question
        llm_response = None
        retry_count = 0

        while retry_count <= 1:  # Max one retry
            try:
                llm_response = analyze_with_llm(conversation_history, reviews)

                # Check for errors that warrant retry
                if isinstance(llm_response, str):
                    error_indicators = [
                        "Model retry failed validation:",
                        "Model retry did not return valid JSON",
                        "Error invoking model"
                    ]
                    
                    needs_retry = any(ind in llm_response for ind in error_indicators)
                    
                    if needs_retry and retry_count < 1:
                        print(f"[RETRY {retry_count + 1}/1] Retrying due to error...")
                        retry_count += 1
                        continue
                    elif needs_retry:
                        print("[ERROR] LLM failed to provide a valid response.")
                        print("This may be due to:")
                        print("- The model not following the JSON schema requirements")
                        print("- Budget constraints that can't be met with available data")
                        print("- Missing attribute evidence in the dataset")
                        print("Try rephrasing your request or adjusting your requirements.\n")
                        if DEBUG:
                            print("Debug info:", llm_response[:200])
                        break
                    elif llm_response.startswith("Error invoking model"):
                        print("[ERROR] LLM invocation failed:\n")
                        print(llm_response)
                        break
                
                break
                
            except Exception as e:
                if retry_count < 1:
                    print(f"[RETRY {retry_count + 1}/1] Unexpected error, retrying...")
                    retry_count += 1
                    continue
                else:
                    print(f"[ERROR] Failed to get valid response after retries.")
                    print("This could be due to:")
                    print("- Temporary connectivity issues") 
                    print("- Model loading problems")
                    print("- Invalid conversation context")
                    print(f"Last error: {e}")
                    llm_response = None
                    break
        
        # Display final response or error
        if llm_response is None:
            print("[ERROR] No response received after all attempts")
            continue
        elif isinstance(llm_response, str) and llm_response.startswith("[ERROR]"):
            print(llm_response)
            continue
        
        print(llm_response)