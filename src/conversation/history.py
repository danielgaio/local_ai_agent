"""Conversation history management."""

import re
from typing import List, Optional, Tuple
from ..core.config import MAX_QUERY_WORDS


def is_vague_input(text: str) -> bool:
    """Check if user input is too vague (greeting/pleasantry or lacks substance).

    Args:
        text: The user's input text

    Returns:
        bool: True if input is vague, False otherwise
    """
    if not text or not text.strip():
        return True

    low = text.lower().strip()

    # Check for substantive attribute tokens
    attr_tokens = [
        "suspension", "travel", "long-travel", "long travel", 
        "budget", "touring", "adventure", "engine", "cc", "price"
    ]
    for a in attr_tokens:
        if a in low:
            return False

    # Check common greeting patterns
    greetings = [
        "hi", "hello", "hey", "how are you", "how's it going",
        "what's up", "how are ya", "how r u", "good morning",
        "good afternoon", "good evening",
    ]
    for g in greetings:
        if (low == g or low.startswith(g + " ") or 
            low.endswith(" " + g) or (g in low and len(low.split()) <= 4)):
            return True

    # Remove punctuation and analyze remaining tokens
    tokens = re.findall(r"[A-Za-z0-9\-']+", low)
    if not tokens:
        return True

    # Filter out stopwords
    stop = {
        'i', 'want', 'need', 'for', 'the', 'and', 'a', 'an', 'to', 'with',
        'that', 'is', 'on', 'in', 'of', 'my', 'me', 'it', 'are', 'please',
        'would', 'like', 'looking', 'who', 'how', 'what', 'your', 'you', 'we'
    }

    informative = [t for t in tokens if t not in stop and len(t) > 2]
    return len(informative) < 2


def generate_retriever_query(conversation_history: List[str]) -> Tuple[Optional[str], bool]:
    """Generate a short, focused query for retrieval.
    
    Args:
        conversation_history: List of user messages

    Returns:
        tuple: (query_string, used_fallback) where used_fallback indicates if
               keyword extraction was used instead of LLM generation
    """
    # Use the most recent up to 6 messages for context
    recent = conversation_history[-6:]
    convo_block = "\n".join([f"- {m}" for m in recent])

    query = keyword_extract_query(conversation_history[-1] if conversation_history else "")
    return query, True


def keyword_extract_query(user_message: str) -> Optional[str]:
    """Extract important keywords for a deterministic fallback query.

    Args:
        user_message: The user's message to analyze

    Returns:
        str: A space-joined query of important keywords, or None if no keywords found
    """
    if not user_message:
        return None

    msg = user_message.lower()
    stop = {
        'i', 'want', 'need', 'for', 'the', 'and', 'a', 'an', 'to', 'with',
        'that', 'is', 'on', 'in', 'of', 'my', 'me', 'it', 'are', 'please',
        'would', 'like', 'want', 'looking', 'who'
    }

    # Preserve important attribute keywords
    attributes = [
        "long-travel", "long travel", "suspension", "travel",
        "damping", "soft", "firm", "comfortable", "comfort",
        "fork", "shock"
    ]
    ride_types = [
        "adventure", "touring", "cruiser", "sport",
        "offroad", "dual-sport", "enduro", "supermoto"
    ]

    tokens = re.findall(r"[0-9]+cc|[a-zA-Z0-9\-]+", msg)
    seen = []

    # Prioritize attributes & ride types
    for k in attributes + ride_types:
        if k in msg and k not in seen:
            seen.append(k)

    # Add other informative tokens
    for t in tokens:
        t = t.strip()
        if not t or t in stop:
            continue
        if t in seen:
            continue
        # Ignore short tokens or pure numbers (unless cc)
        if re.fullmatch(r"\d+", t) and not t.endswith('cc'):
            continue
        if len(t) <= 2:
            continue
        seen.append(t)

    # Limit to MAX_QUERY_WORDS
    if not seen:
        return None
    query_tokens = seen[:MAX_QUERY_WORDS]
    return " ".join(query_tokens)