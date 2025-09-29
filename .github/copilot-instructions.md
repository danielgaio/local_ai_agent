# Copilot Instructions for Local AI Agent

## Project Overview

This is a local AI-powered motorcycle recommendation system using Ollama (llama3.2:3b) with RAG integration via ChromaDB. The system provides structured motorcycle recommendations based on user conversations, prioritizing user-specified attributes and enforcing budget constraints.

## Core Architecture

### Two-Module Design

- **`main.py`**: Contains the conversation engine with sophisticated prompt engineering and validation
- **`vector.py`**: Handles vector store initialization, document embedding, and retriever setup
- Data source: `motorcycle_reviews.csv` with structured motorcycle review data

### Key Components

1. **LLM Integration**: Uses OllamaLLM with fallback-compatible invocation patterns (`invoke()` -> `generate()`)
2. **RAG Pipeline**: ChromaDB vector store with Ollama embeddings (`mxbai-embed-large`)
3. **Conversation Management**: Maintains conversation history for context-aware responses
4. **Response Validation**: Client-side validation for budget constraints and attribute presence

## Critical Patterns

### Structured JSON Response Format

The system enforces strict JSON response schemas:

```python
# Clarifying question
{"type": "clarify", "question": "<question>"}

# Recommendation
{"type": "recommendation", "picks": [...], "note": "optional"}
```

### Attribute Prioritization System

- User's most recent message determines prioritized attribute (e.g., "suspension", "long-travel")
- Each recommendation must mention the prioritized attribute in `reason` (≤12 words) or set `evidence` to "none in dataset"
- Validation triggers retry if no picks mention the prioritized attribute

### Budget Enforcement

- Regex-based budget extraction from conversation history (`$12,000`, `12k`, etc.)
- Hard filtering: removes picks exceeding budget, returns empty `picks` array with explanatory `note` if none match

### Metadata-Rich Vector Store

Documents include structured metadata extracted at index-time:

```python
metadata = {
    "brand", "model", "year", "price_usd_estimate",
    "engine_cc", "suspension_notes", "ride_type"
}
```

## Development Workflows

### Testing Strategy

- **Smoke tests**: `tests/run_smoke.py` uses stubbed modules to test core query generation
- **Unit tests**: Focus on validation logic (`test_validate_and_topreviews.py`)
- **Integration tests**: `end_to_end_chroma.py` tests real vector store retrieval
- Run tests with: `python -m pytest tests/` or individual files

### Local Development

1. Ensure Ollama is running with required models:
   - `ollama pull llama3.2:3b` (main LLM)
   - `ollama pull mxbai-embed-large` (embeddings)
2. First run initializes ChromaDB at `./chroma_langchain_db`
3. Main CLI: `python main.py`

### Key Files to Understand

- `todo.md`: Contains implementation roadmap and completed improvements
- `vector.py`: Study metadata extraction functions (`parse_price`, `extract_suspension_notes`)
- `main.py`: Focus on `validate_and_filter()` and `build_llm_prompt()` functions

## Project-Specific Conventions

### Query Generation Patterns

- Model-generated queries with deterministic fallback using `keyword_extract_query()`
- Prioritizes attribute keywords over generic terms
- Fallback triggers when model query is empty or >12 words

### Error Handling Philosophy

- Graceful degradation: fallback to deterministic methods when AI fails
- Surface LLM errors to users rather than silent failures
- Compatibility layers for different LangChain API versions

### Prompt Engineering Approach

- System instructions emphasize attribute prioritization and evidence requirements
- Include schema examples directly in prompts
- Use retry mechanism with enhanced prompts when validation fails

## Dependencies & Environment

- **Core**: langchain, langchain-ollama, langchain-chroma, pandas
- **Testing**: pytest with module stubbing for external dependencies
- **Data**: CSV with structured motorcycle reviews including price estimates and technical specs

When modifying this system, prioritize maintaining the structured response format and attribute prioritization logic, as these are core to the user experience.
