# local_ai_agent (developer README)

[![CI](https://github.com/danielgaio/local_ai_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/danielgaio/local_ai_agent/actions/workflows/ci.yml)  
[![Lint](https://github.com/danielgaio/local_ai_agent/actions/workflows/lint.yml/badge.svg)](https://github.com/danielgaio/local_ai_agent/actions/workflows/lint.yml)  
[![Coverage](https://github.com/danielgaio/local_ai_agent/actions/workflows/coverage.yml/badge.svg)](https://github.com/danielgaio/local_ai_agent/actions/workflows/coverage.yml)

This repository provides a small local RAG-powered motorcycle recommender using Ollama (LLM) and ChromaDB (vector store).

Prerequisites
-------------
- Python 3.10+
- Ollama (if you want to run LLM locally) and the necessary models pulled (see below).

Quick setup
-----------
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Pull Ollama models used by the project:

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

Rebuild the Chroma DB
---------------------
- The Chroma DB is persisted at `./chroma_langchain_db`.
- To rebuild the index from `motorcycle_reviews.csv`, delete the `./chroma_langchain_db` directory and run the app once:

```bash
rm -rf ./chroma_langchain_db
python main.py
```

This will call `vector.py` which reads `motorcycle_reviews.csv`, extracts lightweight metadata (price, suspension notes, engine cc, ride_type) and populates the Chroma collection `motorcycle_reviews`.

Running the recommender
-----------------------
Run:

```bash
python main.py
```

The CLI will prompt for preferences. The system uses a combination of model-driven and deterministic retrieval query generation and will ask clarifying questions when input is vague.

Tests
-----
Run the test suite with:

```bash
python -m pytest -q
```

Notes
-----
- If you run into issues with `chromadb` or `langchain-chroma`, check package compatibility in `requirements.txt` and update accordingly.
- The project contains a smoke test `tests/end_to_end_chroma.py` that demonstrates retrieval against the local Chroma DB.

Contact
-------
If you change indexing code or dependencies, update this README accordingly.

CI and test environment notes
----------------------------
On GitHub Actions and many CI environments the project can't contact a local Ollama
server. To make tests deterministic and fast in CI we use a lightweight, local
fallback for embeddings when running under CI or when the environment variable
`USE_DUMMY_EMBEDDINGS=1` is set.

- CI runners: the code automatically detects `GITHUB_ACTIONS=true` or `CI=true`
	and will use the dummy embeddings implementation.
- Local dev: if you have Ollama installed and running, the project will use
	`OllamaEmbeddings` by default. To force the dummy embeddings locally set
	`USE_DUMMY_EMBEDDINGS=1` in your environment.

This keeps unit tests fast and avoids external service dependency during CI runs.

Model provider configuration
----------------------------
This project supports two model providers for LLM and embeddings. Choose which provider to use via the `MODEL_PROVIDER` environment variable.

- MODEL_PROVIDER=ollama (default)
  - Uses a local Ollama daemon for both LLM and (when available) embeddings.
  - Make sure Ollama is installed and running, and pull the models used by this project:

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

- MODEL_PROVIDER=openai
  - Uses OpenAI through LangChain's `OpenAI` and `OpenAIEmbeddings` classes.
  - Requires an `OPENAI_API_KEY` environment variable. Set it in your shell or in a `.env` file:

```bash
export OPENAI_API_KEY="sk-..."
# or add to .env and source it
```

Fallback and testing helpers
----------------------------
- CI/deterministic tests: set `USE_DUMMY_EMBEDDINGS=1` to force a local deterministic embedding implementation. In GitHub Actions this is automatic unless you override it.
- To force OpenAI usage locally even if Ollama is available: set `MODEL_PROVIDER=openai` and ensure `OPENAI_API_KEY` is set.

See `.env.example` for a template of recommended environment variables.
