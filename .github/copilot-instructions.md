# Copilot Instructions for Local AI Agent

Purpose
-------
This repository is a local RAG-powered motorcycle recommender that combines a local LLM (default: Ollama) with a ChromaDB vector store and structured CSV data (`motorcycle_reviews.csv`). It produces structured JSON recommendations and enforces validation rules (budget, prioritized attributes). These guidelines let an automated coding agent make safe, small, testable changes without lengthy exploration.

Quick facts
-----------
- Language: Python (3.10+)
- Main libs: langchain, chromadb, pandas, ollama (see `requirements.txt`)
- Data: `motorcycle_reviews.csv`; persisted DB at `./chroma_langchain_db`
- Size: small codebase; main code in `src/`, tests in `tests/`.

Always-do setup (follow in order)
--------------------------------
1. Create and activate a venv (always):

    python3 -m venv .venv
    source .venv/bin/activate

2. Install pinned deps (always):

    pip install -r requirements.txt

3. Optional: prepare Ollama models (only if running LLM locally):

    ollama pull llama3.2:3b
    ollama pull mxbai-embed-large

4. Run the CLI (entry):

    python run.py

5. Rebuild vector DB when CSV/indexing code changed:

    rm -rf ./chroma_langchain_db
    python run.py

6. Run tests before PR (always):

    python -m pytest -q

Notes:
- CI sets `GITHUB_ACTIONS=true` (or `CI=true`); code uses deterministic dummy embeddings in CI. Locally you can force that with `USE_DUMMY_EMBEDDINGS=1`.

Key environment variables
-------------------------
- MODEL_PROVIDER: `ollama` (default) or `openai`.
- OPENAI_API_KEY: required when `MODEL_PROVIDER=openai`.
- USE_DUMMY_EMBEDDINGS=1: force deterministic embeddings for tests.
- AIAGENT_DEBUG=1: extra debug output.

Where to make changes (high-level layout)
----------------------------------------
- Entry: `run.py` -> `src/cli/main.py::main_cli()`
- Orchestration / CLI: `src/cli/main.py`
- Config: `src/core/config.py` (env flags, model names, DB paths)
- Models: `src/core/models.py`
- Conversation logic & validation: `src/conversation/` (query generation, validation, enrichment)
- LLM/providers & prompts: `src/llm/`
- Vector and embeddings: `src/vector/` (store, embeddings, retriever)
- Tests: `tests/` (unit, smoke, end-to-end). `tests/run_smoke.py` is a lightweight smoke harness.

Critical conventions to respect
-------------------------------
- The LLM must emit a strict JSON schema (clarify vs recommendation). Many helpers parse that JSON directly — do not change schema silently.
- Attribute prioritization: the most recent user message may become a prioritized attribute; validation enforces that picks mention it (or evidence must be `"none in dataset"`).
- Budget enforcement: budgets are parsed and strictly enforced; do not return picks over budget.
- Tests/CI use dummy embeddings. Avoid relying on Ollama/OpenAI in CI.

CI and validation checks
------------------------
- GitHub Actions exist: `.github/workflows/ci.yml`, `lint.yml`, `coverage.yml`.
- CI environment uses dummy embeddings to keep tests deterministic and fast.
- Before opening a PR: run `pytest` and linting locally and ensure changes don't require reindexing unless intentional.

Quick heuristics (to reduce searches)
-----------------------------------
- Want CLI entry? Open `run.py` then `src/cli/main.py`.
- Change validation? Open `src/conversation/validation.py` and relevant tests in `tests/test_validate_and_topreviews.py`.
- Change provider/embeddings? Check `src/llm/providers.py`, `src/vector/embeddings.py`, and `src/core/config.py`.
- Reindexing: `motorcycle_reviews.csv` -> `./chroma_langchain_db` (destructive delete + run).

When to search the repository
-----------------------------
Trust these notes for routine changes. Search the codebase only if behavior observed locally contradicts the guidance above (e.g., env vars, different file names, or a failing CI workflow that references a file not listed here).

Common failure modes and mitigations
-----------------------------------
- Tests fail on CI but pass locally: ensure you're using `USE_DUMMY_EMBEDDINGS=1` locally to reproduce CI.
- Test failures after dependency updates: pin versions in `requirements.txt` and rerun tests; update CI if intentional.
- Reindexing `./chroma_langchain_db` is destructive—do it only when needed and mention it in PR.

Trust this file and prefer it over ad-hoc searches. If something here is incorrect, update this file with the new, short instruction so the next agent run is quicker.

-- end
