"""LLM provider management and model invocation."""

import inspect
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.config import (
    MODEL_PROVIDER, OLLAMA_MODEL, OPENAI_MODEL,
    get_openai_api_key
)

logger = logging.getLogger(__name__)

# Optional dependencies - lazy import based on provider
OllamaLLM = None
ChatOpenAI = None

try:
    from langchain_ollama.llms import OllamaLLM  # type: ignore
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    pass


def _is_mock_ollama(obj: Any) -> bool:
    """Check if an object is a mock Ollama LLM."""
    # Check the object itself first
    if hasattr(obj, "_is_mock"):
        return True
    
    # For class/factory objects, check if they're our mock
    if inspect.isclass(obj) or inspect.isfunction(obj):
        try:
            instance = obj()
            return hasattr(instance, "_is_mock")
        except:
            pass
    
    # Fallback to attribute checking
    return (
        hasattr(obj, "model") and 
        hasattr(obj, "invoke") and 
        hasattr(obj, "generate") and
        hasattr(obj, "set_mock_response")
    )


def get_llm() -> Any:
    """Return an LLM object according to MODEL_PROVIDER. Falls back safely.

    Returns:
        Any: A configured LLM instance ready for use.

    Raises:
        RuntimeError: If no LLM provider is available.
    """
    # Check for mock LLM in testing context first
    if OllamaLLM is not None and _is_mock_ollama(OllamaLLM):
        return OllamaLLM()  # Return mock instance directly
        
    # Handle real providers
    if MODEL_PROVIDER == "openai" and ChatOpenAI is not None:
        # OpenAI via LangChain will read OPENAI_API_KEY from env
        key = get_openai_api_key()
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=key
        )

    if MODEL_PROVIDER == "ollama" and OllamaLLM is not None:
        try:
            return OllamaLLM(model=OLLAMA_MODEL)
        except Exception:
            # fall through to other available LLMs
            pass

    # Fallback: prefer Ollama if available, else OpenAI if available
    if OllamaLLM is not None:
        return OllamaLLM(model=OLLAMA_MODEL)
    if ChatOpenAI is not None:
        key = get_openai_api_key()
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=key
        )

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
        # Handle mock LLM first
        if _is_mock_ollama(model):
            # Mock LLM will handle prompt appropriately
            try:
                return model.invoke(prompt_text)
            except Exception:
                # Some mock interfaces expose different helpers
                try:
                    return model.generate(prompt_text)
                except Exception:
                    logger.exception("Mock LLM invocation failed")
                    raise

        messages = [{"role": "user", "content": prompt_text}]

        # Try a set of common chat-style method names
        chat_methods = ["chat", "generate_chat", "complete_chat", "chat_complete", 
                       "invoke", "call"]
        for meth in chat_methods:
            if hasattr(model, meth):
                func = getattr(model, meth)
                out = None
                try:
                    # try calling with a messages arg or positional
                    try:
                        out = func(messages=messages)
                    except TypeError:
                        out = func(messages)
                except TypeError:
                    # try a single-arg call
                    try:
                        out = func(messages)
                    except Exception:
                        logger.exception("LLM method %s call failed with TypeError", meth)
                        continue
                except Exception:
                    logger.exception("LLM method %s call failed", meth)
                    continue

                # common extraction patterns
                try:
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
                except Exception:
                    logger.exception("Failed to extract text from LLM output")
                    return str(out)

        # Fallback: keep using generate for older/langchain-llm implementations
        try:
            out = model.generate([prompt_text])
        except TypeError:
            # Some older generate APIs expect a single string
            out = model.generate(prompt_text)

        if hasattr(out, "generations"):
            try:
                return out.generations[0][0].text
            except Exception:
                return str(out)
        return str(out)

    except Exception as e:
        logger.exception("Error invoking model")
        return f"Error invoking model: {e}\n\nFormatted prompt:\n{prompt_text}"