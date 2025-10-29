# CODE-LEVEL ACTIONS, BUGS, AND OPTIMIZATIONS

This file summarizes code-level findings after inspecting the main modules and recommends actionable tasks with file pointers and priority levels. Tasks are written to be small, reviewable, and low-risk where possible.

## Executive summary

- The repository is well-structured and contains reasonable fallbacks for CI (dummy embeddings). Key risks relate to inconsistent entrypoints, broad exception handling that masks errors, mixed dict/pydantic model usage, and logging vs print usage. There are also opportunities for improving robustness (parsing/validation), test coverage, and small performance wins (chunked indexing).

## Low-priority / long-term suggestions

- Consider converting the CLI to use `argparse`/`typer` so it can be non-interactive for tests and easier to script.
- Add typed stubs and run `mypy` as optional CI to increase confidence in refactors.
- Add caching for frequent operations (e.g., parsed budget per conversation) if workload grows.

## Quick grep notes I used (helpful locations)

- Prints found in: `src/cli/main.py` (many user-facing prints and error prints)
- Broad excepts found in: `src/llm/providers.py`, `src/conversation/enrichment.py`, `src/conversation/validation.py`, `src/conversation/history.py`
- Env/config entrypoints: `src/core/config.py` (MODEL_PROVIDER, USE_DUMMY, flags)

## Next recommended step (I can implement)

Start with the compatibility shim (`main.py`) and then a small logging/exception hygiene PR for `src/cli/main.py`. These are small, low-risk, fix common test issues, and will make subsequent changes easier to test.

If you want, I will implement the shim now and run the test suite. Tell me to proceed and I'll: add the file, run pytest, and report results.

End of file.
