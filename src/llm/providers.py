"""LLM provider management and model invocation."""

from typing import Any, Dict, List, Optional, Union
import os

from ..core.config import MODEL_PROVIDER, OLLAMA_MODEL, get_openai_api_key

# Optional dependencies - lazy import based on provider
OllamaLLM = None
OpenAI = None

try:
    from langchain_ollama.llms import OllamaLLM  # type: ignore
except ImportError:
    pass

try:
    from langchain.llms import OpenAI  # type: ignore
except ImportError:
    pass


def get_llm() -> Any:
    """Return an LLM object according to MODEL_PROVIDER. Falls back safely.

    Returns:
        Any: A configured LLM instance ready for use.

    Raises:
        RuntimeError: If no LLM provider is available.
    """
    if MODEL_PROVIDER == "openai" and OpenAI is not None:
        # OpenAI via LangChain will read OPENAI_API_KEY from env
        key = get_openai_api_key()
        return OpenAI(temperature=0, openai_api_key=key)

    if MODEL_PROVIDER == "ollama" and OllamaLLM is not None:
        try:
            return OllamaLLM(model=OLLAMA_MODEL)
        except Exception:
            # fall through to other available LLMs
            pass

    # Fallback: prefer Ollama if available, else OpenAI if available
    if OllamaLLM is not None:
        return OllamaLLM(model=OLLAMA_MODEL)
    if OpenAI is not None:
        key = get_openai_api_key()
        return OpenAI(temperature=0, openai_api_key=key)

    raise RuntimeError(
        "No LLM provider available. Install and configure Ollama or OpenAI."
    )


def invoke_model_with_prompt(model: Any, prompt_text: str) -> str:
    """Try calling the LLM in a consistent way across different providers.
    
    Args:
        model: The LLM instance to use
        prompt_text: The prompt text to send to the model

    Returns:
        str: The model's response text

    Note:
        Supports various LLM interfaces by attempting multiple invocation patterns.
    """
    try:
        messages = [{"role": "user", "content": prompt_text}]

        # Try a set of common chat-style method names
        chat_methods = ["chat", "generate_chat", "complete_chat", "chat_complete", 
                       "invoke", "call"]
        for meth in chat_methods:
            if hasattr(model, meth):
                func = getattr(model, meth)
                try:
                    # try calling with a messages arg or positional
                    try:
                        out = func(messages=messages)
                    except TypeError:
                        out = func(messages)
                except Exception:
                    # try a single-arg call
                    out = func(messages)

                # common extraction patterns
                if hasattr(out, "generations"):
                    try:
                        return out.generations[0][0].text
                    except Exception:
                        return str(out)
                if isinstance(out, str):
                    return out
                if isinstance(out, dict):
                    # try common keys
                    for k in ("text", "content", "message"):
                        if k in out:
                            v = out[k]
                            if isinstance(v, dict) and "content" in v:
                                return v["content"]
                            return v
                if hasattr(out, "text"):
                    return out.text
                if hasattr(out, "content"):
                    return out.content
                return str(out)

        # Fallback: keep using generate for older/langchain-llm implementations
        out = model.generate([prompt_text])
        if hasattr(out, "generations"):
            try:
                return out.generations[0][0].text
            except Exception:
                return str(out)
        return str(out)

    except Exception as e:
        return f"Error invoking model: {e}\n\nFormatted prompt:\n{prompt_text}"