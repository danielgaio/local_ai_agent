# CODE-LEVEL ACTIONS, BUGS, AND OPTIMIZATIONS

This file summarizes code-level findings after inspecting the main modules and recommends actionable tasks with file pointers and priority levels. Tasks are written to be small, reviewable, and low-risk where possible.

## Executive summary

- The repository is well-structured and contains reasonable fallbacks for CI (dummy embeddings). Key risks relate to inconsistent entrypoints, broad exception handling that masks errors, mixed dict/pydantic model usage, and logging vs print usage. There are also opportunities for improving robustness (parsing/validation), test coverage, and small performance wins (chunked indexing).

## Low-priority / long-term suggestions

- Add typed stubs and run `mypy` as optional CI to increase confidence in refactors.
- Add caching for frequent operations (e.g., parsed budget per conversation) if workload grows.

## Quick grep notes I used (helpful locations)

- Prints found in: `src/cli/main.py` (many user-facing prints and error prints)
- Broad excepts found in: `src/llm/providers.py`, `src/conversation/enrichment.py`, `src/conversation/validation.py`, `src/conversation/history.py`
- Env/config entrypoints: `src/core/config.py` (MODEL_PROVIDER, USE_DUMMY, flags)
