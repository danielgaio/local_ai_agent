# Dependency Management

This project uses pinned dependencies to ensure reproducible builds and consistent behavior across environments.

## Files

- **`requirements.txt`**: Pinned production dependencies with exact versions. These versions are tested and known to work together.
- **`requirements.in`**: Original unpinned requirements (for reference). Shows the minimum version constraints.
- **`requirements-dev.txt`**: Development tools (linting, formatting, testing utilities).

## Installation

### Production/Runtime
```bash
pip install -r requirements.txt
```

### Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Updating Dependencies

When updating dependencies:

1. **Test first**: Update `requirements.in` with new minimum versions
2. **Install and test**: 
   ```bash
   pip install -r requirements.in
   pytest
   ```
3. **Pin working versions**: If tests pass, update `requirements.txt` with exact versions:
   ```bash
   pip freeze > requirements-frozen.txt
   # Manually copy relevant packages to requirements.txt
   ```
4. **Verify**: Fresh install and test:
   ```bash
   python -m venv fresh_env
   fresh_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   pytest
   ```

## Why Pinned Dependencies?

- **Reproducibility**: Same versions across all environments (dev, CI, production)
- **Stability**: Prevents unexpected breaking changes from upstream packages
- **CI Reliability**: Tests run against known-good versions
- **Debugging**: Easier to isolate issues when everyone uses identical versions

## Version Strategy

- LangChain ecosystem: Pinned to 1.0.x compatible versions
- Testing: Pinned to pytest 8.x
- Data processing: Pinned to pandas 2.3.x with numpy 2.3.x
- LLM providers: Pinned to current stable releases (ollama 0.6.0, openai 2.6.1)

## CI Considerations

- CI uses dummy embeddings (`USE_DUMMY_EMBEDDINGS=1`) to avoid external LLM dependencies
- ChromaDB and core LangChain components are required even in CI mode
- All pinned versions are tested in CI on each commit
