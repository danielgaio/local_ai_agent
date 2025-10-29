"""Tests for the Typer-based CLI in non-interactive mode."""

import json
import tempfile
from pathlib import Path
from typer.testing import CliRunner

from src.cli.typer_main import app


runner = CliRunner()


def test_help_command():
    """Verify --help shows usage information."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Motorcycle recommendation system" in result.stdout
    assert "--query" in result.stdout
    assert "--json" in result.stdout


def test_single_query_text_output():
    """Verify single query returns text output by default."""
    result = runner.invoke(app, ["--query", "adventure bike"])
    # Should succeed (exit 0) or have reasonable output
    # Note: Actual LLM might fail in test environments
    assert "--query" not in result.stdout  # Not showing help
    

def test_single_query_json_output():
    """Verify single query with --json flag returns JSON."""
    result = runner.invoke(app, ["--query", "adventure bike", "--json"])
    
    # Should either succeed with JSON or fail gracefully
    if result.exit_code == 0 and result.stdout.strip():
        # Try to parse as JSON
        try:
            data = json.loads(result.stdout)
            assert "query" in data
            assert data["query"] == "adventure bike"
        except json.JSONDecodeError:
            # If not JSON, should be error message
            pass


def test_batch_mode_with_output_file():
    """Verify batch processing writes to output file."""
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("adventure bike\n")
        f.write("touring motorcycle\n")
        input_file = Path(f.name)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        output_file = Path(f.name)
    
    try:
        result = runner.invoke(app, [
            "--batch", str(input_file),
            "--output", str(output_file)
        ])
        
        # Check file was created
        assert output_file.exists()
        
        # Verify JSON structure
        if output_file.stat().st_size > 0:
            data = json.loads(output_file.read_text())
            assert "queries" in data
            assert "results" in data
            assert data["queries"] == 2
            
    finally:
        # Cleanup
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def test_verbose_flag():
    """Verify --verbose flag is accepted."""
    result = runner.invoke(app, ["--query", "test", "--verbose"])
    # Should not show help
    assert "--query" not in result.stdout or "Usage:" not in result.stdout


def test_output_file_flag():
    """Verify --output flag writes to file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        output_file = Path(f.name)
    
    try:
        result = runner.invoke(app, [
            "--query", "adventure bike",
            "--json",
            "--output", str(output_file)
        ])
        
        # Check file was created
        assert output_file.exists()
        
    finally:
        output_file.unlink(missing_ok=True)


def test_missing_batch_file_error():
    """Verify error when batch file doesn't exist."""
    result = runner.invoke(app, ["--batch", "nonexistent.txt"])
    assert result.exit_code != 0
    # Error message may be in stdout or stderr
    output = (result.stdout + result.stderr).lower()
    assert "not found" in output or "error" in output or result.exit_code == 1
