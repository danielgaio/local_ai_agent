"""Typer-based CLI for the motorcycle recommendation system.

This module provides both interactive and non-interactive modes:
- Interactive mode: Traditional conversational interface (default)
- Non-interactive mode: Single query with JSON output for scripting

Usage:
    # Interactive mode (default)
    python run.py

    # Non-interactive mode
    python run.py --query "adventure bike under 10000" --json

    # Batch mode from file
    python run.py --batch queries.txt --json-output results.json
"""

import json
import sys
import logging
from typing import List, Optional
from pathlib import Path

import typer
from typing_extensions import Annotated

from ..core.models import MotorcycleReview
from ..llm.providers import get_llm, invoke_model_with_prompt
from ..llm.response_parser import parse_llm_response
from ..llm.prompt_builder import build_llm_prompt
from ..conversation.history import (
    is_vague_input, generate_retriever_query, keyword_extract_query
)
from ..conversation.validation import validate_and_filter
from ..conversation.enrichment import enrich_picks_with_metadata
from ..vector.store import load_vector_store
from ..vector.retriever import EnhancedVectorStoreRetriever
from ..core.config import DEFAULT_SEARCH_KWARGS, DEBUG, MODEL_PROVIDER


# Create typer app
app = typer.Typer(
    name="motorcycle-recommender",
    help="AI-powered motorcycle recommendation system",
    add_completion=False
)

logger = logging.getLogger(__name__)


def get_docs_from_retriever(retriever: EnhancedVectorStoreRetriever, query: str) -> List[MotorcycleReview]:
    """Get relevant reviews from retriever and convert to domain models."""
    docs = retriever.get_relevant_documents(query)
    
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


def analyze_with_llm(
    conversation_history: List[str],
    top_reviews: List[MotorcycleReview]
) -> dict:
    """Analyze conversation and provide recommendations or questions.
    
    Returns:
        dict: Parsed LLM response
    """
    prompt = build_llm_prompt(conversation_history, top_reviews)
    response = invoke_model_with_prompt(get_llm(), prompt)

    # Clean response
    def _sanitize_raw(text: str) -> str:
        lines = text.splitlines()
        cleaned = [ln for ln in lines
                  if not any(ln.strip().startswith(f"[{t}]")
                          for t in ("DEBUG", "WARN", "ERROR"))]
        return "\n".join(cleaned).strip()

    response = _sanitize_raw(response)

    try:
        parsed = parse_llm_response(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"type": "error", "message": f"Invalid JSON response: {response[:200]}"}

    # Validate and retry if needed
    valid, info = validate_and_filter(parsed, conversation_history)
    if not valid and getattr(info, "action", None) == "retry":
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
            parsed_retry = parse_llm_response(retry_resp)
            valid2, info2 = validate_and_filter(parsed_retry, conversation_history)
            if valid2:
                parsed = parsed_retry
        except json.JSONDecodeError:
            logger.warning("Retry response was not valid JSON")

    # Enrich picks
    try:
        parsed = enrich_picks_with_metadata(parsed, top_reviews)
    except Exception:
        logger.exception("enrich_picks_with_metadata failed")

    return parsed


def format_response_text(parsed: dict) -> str:
    """Format parsed LLM response as human-readable text."""
    try:
        if parsed.get("type") == "clarify":
            return parsed.get("question", "(no question provided)")
        
        elif parsed.get("type") == "recommendation":
            lines = []

            if "primary" in parsed:
                primary = parsed.get("primary")
                alternatives = parsed.get("alternatives", [])
                
                if primary:
                    lines.append("Top recommendation:")
                    ev = primary.get('evidence')
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
                picks = parsed.get("picks", [])
                lines.append("Top recommendations:")
                if not picks:
                    note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                    lines.append(f"No picks matched strictly. Note: {note}")
                else:
                    for p in picks:
                        ev = p.get('evidence')
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
        
        elif parsed.get("type") == "error":
            return f"Error: {parsed.get('message', 'Unknown error')}"
        
        else:
            return json.dumps(parsed, indent=2)
            
    except Exception:
        logger.exception("formatting LLM response failed")
        return json.dumps(parsed, indent=2)


def process_query(query: str, retriever: EnhancedVectorStoreRetriever) -> dict:
    """Process a single query and return structured result."""
    conversation_history = [query]
    
    try:
        q_res = generate_retriever_query(conversation_history)
        if isinstance(q_res, tuple):
            retrieval_query, used_fallback = q_res
        else:
            retrieval_query = q_res
            used_fallback = False

        if not retrieval_query:
            retrieval_query = query

        reviews = get_docs_from_retriever(retriever, retrieval_query)
        result = analyze_with_llm(conversation_history, reviews)
        
        return {
            "query": query,
            "success": True,
            "response": result
        }
        
    except Exception as e:
        logger.exception(f"Failed to process query: {query}")
        return {
            "query": query,
            "success": False,
            "error": str(e)
        }


@app.command()
def main(
    query: Annotated[Optional[str], typer.Option(
        "--query", "-q",
        help="Single query to process (non-interactive mode)"
    )] = None,
    json_output: Annotated[bool, typer.Option(
        "--json", "-j",
        help="Output results as JSON"
    )] = False,
    batch_file: Annotated[Optional[Path], typer.Option(
        "--batch", "-b",
        help="Process queries from file (one per line)"
    )] = None,
    output_file: Annotated[Optional[Path], typer.Option(
        "--output", "-o",
        help="Write JSON results to file"
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose debug output"
    )] = False
):
    """
    Motorcycle recommendation system with interactive and non-interactive modes.
    
    Examples:
        # Interactive mode (default)
        python run.py
        
        # Single query with JSON output
        python run.py --query "adventure bike under 10000" --json
        
        # Batch processing
        python run.py --batch queries.txt --output results.json
        
        # Verbose mode
        python run.py --query "touring bike" --verbose --json
    """
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Initialize vector store and retriever
    logger.info("Loading vector store...")
    vector_store = load_vector_store()
    retriever = EnhancedVectorStoreRetriever(
        vectorstore=vector_store,
        search_kwargs=DEFAULT_SEARCH_KWARGS,
        provider=MODEL_PROVIDER,
        batch_size=10
    )
    logger.info("Vector store loaded successfully")

    # Non-interactive: Single query mode
    if query:
        result = process_query(query, retriever)
        
        if json_output or output_file:
            output_data = result
            if output_file:
                output_file.write_text(json.dumps(output_data, indent=2))
                typer.echo(f"Results written to {output_file}")
            else:
                typer.echo(json.dumps(output_data, indent=2))
        else:
            if result["success"]:
                typer.echo(format_response_text(result["response"]))
            else:
                typer.echo(f"Error: {result['error']}", err=True)
                raise typer.Exit(1)
        return

    # Non-interactive: Batch mode
    if batch_file:
        if not batch_file.exists():
            typer.echo(f"Error: Batch file not found: {batch_file}", err=True)
            raise typer.Exit(1)
        
        queries = [line.strip() for line in batch_file.read_text().splitlines() if line.strip()]
        results = []
        
        with typer.progressbar(queries, label="Processing queries") as progress:
            for q in progress:
                result = process_query(q, retriever)
                results.append(result)
        
        output_data = {"queries": len(queries), "results": results}
        
        if output_file:
            output_file.write_text(json.dumps(output_data, indent=2))
            typer.echo(f"\nProcessed {len(queries)} queries")
            typer.echo(f"Results written to {output_file}")
        else:
            typer.echo(json.dumps(output_data, indent=2))
        return

    # Interactive mode (default)
    typer.echo("=" * 60)
    typer.echo("  Motorcycle Recommendation System")
    typer.echo("=" * 60)
    typer.echo("\nType 'quit' or 'q' to exit\n")
    
    conversation_history: List[str] = []

    while True:
        try:
            user_input = typer.prompt("\nWhat are your motorcycle preferences?").strip()
            
            if user_input.lower() in ('q', 'quit', 'exit'):
                typer.echo("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            typer.echo("\nThinking...")
            conversation_history.append(user_input)
            
            # Check for vague input
            if is_vague_input(user_input):
                cq_prompt = (
                    "You are a concise assistant that asks a single short clarifying question "
                    "when the user's message is vague.\n"
                    "Given the recent conversation, return exactly one short question (one line) "
                    "that will help you clarify the user's needs for motorcycle recommendations. "
                    "Do not add any extra text.\n\n"
                    f"Conversation:\n{chr(10).join(['- ' + m for m in conversation_history[-4:]])}\n"
                )
                
                try:
                    cq_response = invoke_model_with_prompt(get_llm(), cq_prompt)
                    if cq_response:
                        for ln in cq_response.splitlines():
                            ln = ln.strip()
                            if ln and not ln.lower().startswith(("hi", "hello", "hey")):
                                if not ln.endswith("?"):
                                    ln = ln.rstrip('.') + "?"
                                typer.echo(f"\nClarifying question: {ln}")
                                conversation_history.append(ln)
                                break
                        continue
                except Exception:
                    logger.debug("Clarifying question generation failed", exc_info=True)
            
            # Process query
            try:
                q_res = generate_retriever_query(conversation_history)
                if isinstance(q_res, tuple):
                    retrieval_query, used_fallback = q_res
                else:
                    retrieval_query = q_res
                
                if not retrieval_query:
                    retrieval_query = " ".join(conversation_history[-3:])
                
                reviews = get_docs_from_retriever(retriever, retrieval_query)
                result = analyze_with_llm(conversation_history, reviews)
                
                typer.echo("\n" + format_response_text(result))
                
            except Exception as e:
                logger.exception("Failed to process query")
                typer.echo(f"\nError: {e}", err=True)
                
        except (KeyboardInterrupt, EOFError):
            typer.echo("\n\nGoodbye!")
            break


if __name__ == "__main__":
    app()
