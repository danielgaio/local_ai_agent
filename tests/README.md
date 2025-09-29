# Test Suite Documentation

This directory contains automated tests for the motorcycle recommendation system.

## Test Files

### Unit Tests (Fast)

#### `test_index_metadata_presence.py`

Tests that the metadata extraction functions correctly parse:

- **Suspension notes**: Extracts keywords like "suspension", "long-travel", "WP", "damping", etc.
- **Price parsing**: Handles formats like "$12,000", "12k", "7000"
- **Coverage**: Tests the core parsing logic used in `vector.py` for metadata enrichment

#### `test_llm_response_shape.py`

Tests that `analyze_with_llm()` correctly handles different response types:

- **Valid JSON**: Recommendation and clarifying question formats
- **Invalid JSON**: Malformed JSON, plain text responses
- **Error handling**: Ensures raw responses are returned when JSON parsing fails
- **Coverage**: Tests the response parsing and error handling pipeline

#### `test_budget_enforcement.py`

Tests the budget validation pipeline:

- **Budget extraction**: Parses budget from conversation ("$10k", "Budget $5,000", etc.)
- **Pick filtering**: Removes picks that exceed budget constraints
- **Edge cases**: No budget, all picks over budget, mixed scenarios
- **Coverage**: Tests `validate_and_filter()` function thoroughly

### Integration Tests

#### `smoke_test.py`

End-to-end tests that run the full system with piped input:

- **Prerequisites**: Checks for Ollama and required models
- **Real scenarios**: Tests with actual user inputs
- **Output validation**: Verifies expected patterns in responses
- **Requirements**: Needs Ollama running with `llama3.2:3b` and `mxbai-embed-large`

## Running Tests

### Quick Unit Tests

```bash
# Run just the fast unit tests
python tests/run_all_tests.py
```

### Full Test Suite (Including Smoke Tests)

```bash
# Run unit tests + smoke tests (requires Ollama)
python tests/run_all_tests.py --smoke
```

### Individual Tests

```bash
# Run specific tests
python tests/test_index_metadata_presence.py
python tests/test_llm_response_shape.py
python tests/test_budget_enforcement.py
python tests/smoke_test.py
```

## Test Strategy

### Fast Feedback Loop

- **Unit tests** run in seconds without external dependencies
- Use mocking/stubbing for LLM and vector store dependencies
- Focus on core business logic validation

### Comprehensive Coverage

- **Metadata extraction**: Ensures data quality in vector store
- **Response handling**: Validates JSON parsing and error recovery
- **Budget constraints**: Critical user requirement enforcement
- **End-to-end**: Real system behavior validation

### Continuous Integration Ready

- Unit tests can run in any environment (no external dependencies)
- Smoke tests can be optional in CI/CD (require Ollama setup)
- Clear pass/fail reporting with detailed error messages

## Adding New Tests

### For New Features

1. Add unit tests for core logic in isolated functions
2. Mock external dependencies (LLM, vector store, file I/O)
3. Test both happy path and error conditions
4. Add integration tests for user-facing behavior

### Test Naming Convention

- `test_<feature_name>.py` for unit test files
- `<test_function>()` functions within each file
- Descriptive test names that explain what is being tested

### Mock Patterns

See existing tests for patterns on mocking:

- LangChain LLM responses
- Vector store retrievers
- Module imports for dependency isolation

## Expected Output

### Unit Tests Success

```
🎉 All tests passed!
Unit Tests: 3/3 passed
  ✅ test_index_metadata_presence.py
  ✅ test_llm_response_shape.py
  ✅ test_budget_enforcement.py
```

### Smoke Test Success (with Ollama)

```
🎉 All smoke tests passed!
The system is working correctly with piped input.
```

This test suite provides confidence that the system's core functionality works correctly and handles edge cases gracefully.
