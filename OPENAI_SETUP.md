# OpenAI Provider Setup Guide

This guide explains how to configure and use the OpenAI provider instead of local Ollama.

## Why Use OpenAI?

**Advantages:**
- No local setup required - works immediately with just an API key
- Faster cold start (no model loading)
- Consistent performance across machines
- Latest GPT models with high-quality responses

**Disadvantages:**
- Requires internet connection
- Costs per API call (embeddings + LLM tokens)
- Data sent to OpenAI servers

## Quick Start

### 1. Get an OpenAI API Key

Sign up at [https://platform.openai.com/](https://platform.openai.com/) and create an API key.

### 2. Configure Environment

```bash
# Option A: Export in shell
export MODEL_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Option B: Create .env file
cp .env.example .env
# Edit .env and set:
# MODEL_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

### 3. Run the Application

```bash
# Interactive mode
python run_typer.py

# Non-interactive mode
python run_typer.py --query "adventure bike under 10000"

# With JSON output
python run_typer.py --query "touring bike" --json
```

## Configuration Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| LLM | `gpt-3.5-turbo` | Conversation and recommendations |
| Embeddings | `text-embedding-3-small` | Vector search and retrieval |

These can be customized in `src/core/config.py`:
```python
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4", etc.
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"
```

### Cost Estimation

Approximate costs per query (as of 2024):
- **Embeddings**: $0.00002 per query (text-embedding-3-small)
- **LLM**: $0.0015 - $0.002 per query (gpt-3.5-turbo)
- **Total**: ~$0.002 per query

For 1000 queries: ~$2.00

Note: Costs vary based on:
- Query complexity
- Number of retrieved documents
- Conversation history length

## Switching Between Providers

### Ollama (Local, Free)

```bash
export MODEL_PROVIDER=ollama
# Make sure Ollama is running
ollama list
```

### OpenAI (Remote, Paid)

```bash
export MODEL_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
```

### Inline Override

```bash
# Run single query with OpenAI
MODEL_PROVIDER=openai OPENAI_API_KEY="sk-..." python run_typer.py -q "test"

# Run with Ollama (even if .env has openai)
MODEL_PROVIDER=ollama python run_typer.py -q "test"
```

## Testing OpenAI Integration

### Run Integration Tests

```bash
# Set environment
export MODEL_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Run OpenAI-specific tests
pytest tests/test_openai_integration.py -v

# All tests pass if OpenAI is configured correctly
```

### Manual Testing

```bash
# Test basic functionality
python run_typer.py --query "list one motorcycle brand" --json

# Test vector search
python run_typer.py --query "adventure bike under 10000" --verbose

# Test batch processing
echo -e "touring bike\nsport bike\nadventure bike" > test_queries.txt
python run_typer.py --batch test_queries.txt --output results.json
```

## Troubleshooting

### Error: "OPENAI_API_KEY environment variable is required"

**Solution**: Set your API key
```bash
export OPENAI_API_KEY="sk-..."
```

### Error: "AuthenticationError: Invalid API key"

**Solutions**:
1. Check API key is correct: `echo $OPENAI_API_KEY`
2. Verify key is active in OpenAI dashboard
3. Check for extra spaces/quotes: `export OPENAI_API_KEY="sk-..."`

### Error: "RateLimitError: Rate limit exceeded"

**Solutions**:
1. Wait a few minutes and retry
2. Upgrade your OpenAI plan
3. Reduce batch size if using `--batch`

### Slow Response Times

**Possible causes**:
1. Internet connection latency
2. OpenAI API load
3. Large conversation history

**Solutions**:
- Use local Ollama for faster responses (no network)
- Reduce number of retrieved documents
- Clear conversation history

### Vector Store Issues

If switching between providers, you may need to rebuild the vector store:

```bash
# Remove existing vector store
rm -rf ./chroma_langchain_db

# Rebuild with OpenAI embeddings
export MODEL_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
python run_typer.py --query "test"
```

**Note**: Vector stores built with different embeddings models (Ollama vs OpenAI) are **not compatible**. You need separate vector stores for each provider.

## Best Practices

### 1. Cost Management

```bash
# Use batch mode for efficiency
python run_typer.py --batch queries.txt --output results.json

# Limit retrieved documents (fewer embeddings)
# Edit src/core/config.py:
# DEFAULT_SEARCH_KWARGS = {"k": 3}  # Default is 5
```

### 2. Performance Optimization

```bash
# Use faster models for testing
# Edit src/core/config.py:
# OPENAI_MODEL = "gpt-3.5-turbo"  # Faster and cheaper than GPT-4

# Cache results for repeated queries
python run_typer.py -q "adventure bike" --json > cached_result.json
```

### 3. Security

```bash
# Never commit API keys
echo ".env" >> .gitignore

# Use environment variables, not hardcoded values
export OPENAI_API_KEY="sk-..."

# Rotate keys regularly in OpenAI dashboard
```

### 4. Development Workflow

```bash
# Development: Use local Ollama (free, private)
MODEL_PROVIDER=ollama python run_typer.py

# Production: Use OpenAI (reliable, consistent)
MODEL_PROVIDER=openai OPENAI_API_KEY="sk-..." python run_typer.py

# CI/Testing: Use dummy embeddings (fast, no API)
USE_DUMMY_EMBEDDINGS=1 pytest
```

## API Key Management

### Option 1: Environment Variables (Recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-..."
export MODEL_PROVIDER=openai
```

### Option 2: .env File

```bash
# Create .env file
cat > .env << EOF
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
EOF

# Load in shell
set -a; source .env; set +a
```

### Option 3: Inline (For Scripts)

```bash
MODEL_PROVIDER=openai OPENAI_API_KEY="sk-..." python run_typer.py -q "test"
```

## Monitoring Usage

Track your OpenAI usage in the dashboard:
1. Visit [https://platform.openai.com/usage](https://platform.openai.com/usage)
2. Monitor daily costs
3. Set up billing alerts
4. Review usage patterns

## Support

For issues specific to:
- **OpenAI API**: Check [OpenAI Status](https://status.openai.com/) or [OpenAI Help](https://help.openai.com/)
- **This Project**: Open an issue on GitHub with logs and error messages

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [OpenAI Pricing](https://openai.com/pricing)
- [LangChain OpenAI Integration](https://python.langchain.com/docs/integrations/providers/openai)
- [CLI Usage Guide](CLI_USAGE.md)
