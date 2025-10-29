# CLI Usage Guide

The motorcycle recommendation system provides both **interactive** and **non-interactive** modes for flexibility in different use cases.

## Entry Points

- **`run.py`**: Original interactive-only CLI (deprecated but maintained for compatibility)
- **`run_typer.py`**: New Typer-based CLI with both interactive and non-interactive modes ✨ **Recommended**

## Interactive Mode (Default)

The traditional conversational interface - great for exploring recommendations:

```bash
python run_typer.py
```

Features:
- Conversational back-and-forth
- Clarifying questions when input is vague
- Context-aware recommendations based on conversation history
- Type `q`, `quit`, or `exit` to exit

Example session:
```
What are your motorcycle preferences? adventure bike under 10000
Thinking...

Top recommendation:
• Honda CRF450L (2023), Price est: $10500...
```

## Non-Interactive Mode

Perfect for scripting, automation, and testing:

### Single Query

```bash
# Text output (human-readable)
python run_typer.py --query "adventure bike under 10000"

# JSON output (machine-readable)
python run_typer.py --query "adventure bike under 10000" --json
```

JSON output structure:
```json
{
  "query": "adventure bike under 10000",
  "success": true,
  "response": {
    "type": "recommendation",
    "primary": {
      "brand": "Honda",
      "model": "CRF450L",
      "year": 2023,
      "price_est": 10500,
      "reason": "Great dual-sport...",
      "evidence": "Long-travel suspension...",
      "evidence_source": "motorcycle_reviews.csv - row 42"
    },
    "alternatives": [...]
  }
}
```

### Batch Processing

Process multiple queries from a file:

```bash
# queries.txt:
# adventure bike under 10000
# touring motorcycle for long trips
# beginner-friendly sport bike

python run_typer.py --batch queries.txt --output results.json
```

Output structure:
```json
{
  "queries": 3,
  "results": [
    {
      "query": "adventure bike under 10000",
      "success": true,
      "response": {...}
    },
    ...
  ]
}
```

### Output to File

```bash
# Save single query result
python run_typer.py --query "touring bike" --json --output result.json

# Save batch results
python run_typer.py --batch queries.txt --output results.json
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--query TEXT` | `-q` | Single query to process (non-interactive) |
| `--json` | `-j` | Output results as JSON |
| `--batch PATH` | `-b` | Process queries from file (one per line) |
| `--output PATH` | `-o` | Write JSON results to file |
| `--verbose` | `-v` | Enable verbose debug output |
| `--help` | | Show help message |

## Examples

### Testing and CI

```bash
# Quick test with JSON output for validation
python run_typer.py -q "adventure bike" --json | python -m json.tool

# Test multiple scenarios
python run_typer.py -b test_queries.txt -o test_results.json
pytest tests/test_cli_typer.py  # Unit tests for CLI
```

### Scripting

```bash
# Shell script example
#!/bin/bash
RESULT=$(python run_typer.py -q "adventure bike under 10000" --json)
echo "$RESULT" | jq '.response.primary.brand'
```

```python
# Python script example
import subprocess
import json

result = subprocess.run(
    ["python", "run_typer.py", "-q", "touring bike", "--json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
if data["success"]:
    print(f"Recommended: {data['response']['primary']['brand']}")
```

### Debugging

```bash
# Verbose output for troubleshooting
python run_typer.py --query "test" --verbose --json
```

## Migration from Old CLI

The old `run.py` still works but only supports interactive mode. To migrate:

**Before:**
```bash
python run.py  # Interactive only
```

**After:**
```bash
python run_typer.py  # Interactive (default)
python run_typer.py --query "..."  # Non-interactive
```

## Testing

Run CLI tests:
```bash
# Test CLI functionality
pytest tests/test_cli_typer.py -v

# Test with coverage
pytest tests/test_cli_typer.py --cov=src.cli.typer_main
```

## Environment Variables

- `MODEL_PROVIDER`: `ollama` (default) or `openai`
- `USE_DUMMY_EMBEDDINGS`: `1` to use dummy embeddings (CI mode)
- `AIAGENT_DEBUG`: `1` for debug logging
- `OPENAI_API_KEY`: Required when using OpenAI provider

## Error Handling

Non-interactive mode provides structured error responses:

```json
{
  "query": "invalid query",
  "success": false,
  "error": "Failed to process query: <error details>"
}
```

Exit codes:
- `0`: Success
- `1`: Error (query failed, file not found, etc.)

## Performance Tips

- **Batch mode**: Use `--batch` for multiple queries - more efficient than multiple single queries
- **JSON output**: Use `--json` for programmatic processing - faster to parse than text
- **Verbose mode**: Only use `--verbose` when debugging - impacts performance

## Common Use Cases

### CI/CD Pipeline
```bash
# Validate recommendations in CI
python run_typer.py -b test_scenarios.txt -o ci_results.json
python scripts/validate_results.py ci_results.json
```

### API Backend
```python
# Use as backend for API service
from src.cli.typer_main import process_query
result = process_query("adventure bike", retriever)
```

### Data Analysis
```bash
# Generate recommendations for analysis
python run_typer.py -b customer_queries.txt -o recommendations.json
python scripts/analyze_recommendations.py recommendations.json
```

## Troubleshooting

**Issue**: `typer` module not found
```bash
pip install -r requirements-dev.txt  # Install development dependencies
```

**Issue**: Vector store initialization fails
```bash
# Rebuild vector store
rm -rf ./chroma_langchain_db
python run_typer.py --query "test"
```

**Issue**: LLM not responding
```bash
# Check Ollama is running
ollama list

# Or use dummy embeddings for testing
USE_DUMMY_EMBEDDINGS=1 python run_typer.py --query "test"
```
