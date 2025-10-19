from langchain_ollama.llms import OllamaLLM
import json
import sys
# Use the prebuilt retriever exported by vector.py. This module must provide a `retriever` variable.
from vector import retriever


# Initialize model
model = OllamaLLM(model="llama3.2:3b")

SYSTEM_INSTRUCTIONS = """
You are an expert motorcycle recommender. The user will provide one or more messages describing preferences.
Always analyze the user's messages, decide if you have enough information to recommend motorcycles from the provided dataset, or ask a single clear follow-up question to clarify missing information.
Do not rely on local deterministic keyword parsing in the client; perform the analysis and decision-making inside the model.
When recommending, strictly enforce numeric budget constraints when provided: exclude any motorcycle whose listed price exceeds the user's stated budget. If nothing in the dataset strictly matches the budget, explicitly say so and suggest the closest alternatives under the budget or advise on raising the budget.
Respect explicit constraints (budget, cylinder count, riding style, experience), and explain why each recommended motorcycle matches the user's preferences.
If you need more information, ask exactly one short clarifying question. Otherwise, recommend motorcycles from the dataset and explain your reasoning.

Priority and concision guidance:
- If the user's most recent message requests a specific attribute (for example 'big suspension'), prioritize that attribute above all others when selecting and ranking motorcycles.
- For each pick include a short, attribute-focused reason (max 12 words) in the `reason` field, and include an `evidence` field (one short phrase) if the reviews or specs mention the attribute; if none, set `evidence` to "none in dataset".
- Return exactly one JSON object following the prescribed shapes. Keep reasons concise and focused on the prioritized attribute.
- Prefer explicit metadata fields from the REVIEWS when present (e.g., `suspension_notes`, `engine_cc`, `ride_type`, `price_usd_estimate`) as authoritative evidence; cite those fields in `evidence` when they support the pick.

RESPONSE FORMAT (REQUIRED): Return a single JSON object only (no surrounding text). The object must be one of two shapes:

1) Clarifying question:
    {"type": "clarify", "question": "<one short question the assistant needs>"}

2) Recommendation:
    {"type": "recommendation", "primary": {"brand": "", "model": "", "year": 0, "price_est": 0, "reason": "", "evidence": ""}, "alternatives": [{"brand": "", "model": "", "year": 0, "price_est": 0, "reason": "", "evidence": ""}, ...up to 2 items], "note": "optional free-text note if nothing strictly matches budget"}

Strict rules:
- Return exactly one JSON object and nothing else (no extra commentary). The client will parse this JSON. Follow the shapes above precisely.
- When recommending, select ONE primary pick that best matches the user's needs, plus up to 2 alternatives that offer different trade-offs or price points.
- Only include items whose numeric price_est is <= the user's stated budget (if budget provided). If none match, set "primary": null and "alternatives": [] and include an explanatory "note".

Template requirement (strict):
- Each pick.reason MUST be 12 words or fewer and should explicitly mention the prioritized attribute (for example: "long-travel suspension"). If the provided REVIEWS or metadata do not contain direct evidence for the prioritized attribute, put the literal string "none in dataset" in the `evidence` field for that pick.
- Primary pick should be the best overall match; alternatives should offer variety (different price points, brands, or trade-offs).
- Example: {"type":"recommendation","primary":{"brand":"KTM","model":"790 Adventure","year":2019,"price_est":10000,"reason":"excellent long-travel suspension","evidence":"fork travel 210mm"},"alternatives":[{"brand":"Honda","model":"CB500X","year":2023,"price_est":7000,"reason":"good suspension at lower price","evidence":"basic travel"}]}

Return the JSON object as the entire model response.
"""


def invoke_model_with_prompt(prompt_text: str):
    """Try calling the OllamaLLM in a couple of safe ways and return a string result."""
    # Prefer chat-style invocation when available, then fall back to older generate-based APIs.
    try:
        messages = [{"role": "user", "content": prompt_text}]

        # Try a set of common chat-style method names
        chat_methods = ["chat", "generate_chat", "complete_chat", "chat_complete", "invoke", "call"]
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


def build_llm_prompt(conversation_history: list, top_reviews: list):
    """Assemble a compact prompt including system instructions, recent conversation, and a small set of reviews.
    conversation_history is a list of strings (user messages in order). top_reviews is a list of dicts.
    """
    convo_text = "\n".join([f"User: {m}" for m in conversation_history])
    # top_reviews are dicts with keys brand/model/year/comment/price_usd_estimate when available
    # For each review, include explicit evidence fields when available so the LLM can reference them.
    reviews_parts = []
    for r in top_reviews:
        brand = r.get('brand', '') or ''
        model = r.get('model', '') or ''
        year = r.get('year', '') or ''
        comment = r.get('comment', r.get('text', '')) or ''
        price = r.get('price_usd_estimate', r.get('price_est', 'unknown'))
        suspension = r.get('suspension_notes')
        engine = r.get('engine_cc')
        ride = r.get('ride_type')
        parts = [f"- {brand} {model} ({year}): {comment}"]
        parts.append(f"Price est: ${price}")
        if suspension:
            parts.append(f"Suspension notes: {suspension}")
        if engine:
            parts.append(f"Engine (cc): {engine}")
        if ride:
            parts.append(f"Ride type: {ride}")
        reviews_parts.append(" | ".join(parts))
    reviews_text = "\n".join(reviews_parts)
    # Insert a short USER FOCUS hint using the most recent user message to prioritize attributes
    user_focus = conversation_history[-1] if conversation_history else ""
    # lightly spell-correct obvious misspellings for the focus hint
    user_focus = simple_spell_correct(user_focus)
    prompt = (
        f"SYSTEM:\n{SYSTEM_INSTRUCTIONS}\n\nCONVERSATION:\n{convo_text}\n\nREVIEWS:\n{reviews_text}\n\n"
        f"USER FOCUS: {user_focus} -- prioritize this attribute when selecting the primary pick and alternatives.\n\n"
        "TASK: Based on the conversation above, either ask one short clarifying question (if you need more info) or recommend motorcycles from the REVIEWS with one primary pick and up to 2 alternatives. Be explicit about why each pick matches.\n\n"
        "RESPONSE EXAMPLE AND GUIDANCE:\n"
        "Return exactly one JSON object as specified in SYSTEM instructions.\n"
        "Tiny schema example (return exactly this shape, with real values):\n"
        "{'type':'recommendation', 'primary':{'brand':'', 'model':'', 'year':0, 'price_est':0, 'reason':'(<=12 words mentioning prioritized attribute)', 'evidence':'(short phrase or \"none in dataset\")'}, 'alternatives':[{'brand':'', 'model':'', 'year':0, 'price_est':0, 'reason':'(<=12 words)', 'evidence':'(short phrase or \"none in dataset\")'}], 'note':''}\n"
        "If you cannot find direct evidence for the prioritized attribute inside the provided REVIEWS or metadata for a pick, set that pick's evidence to the literal string 'none in dataset'.\n"
    "Prefer suspension_notes and engine_cc fields from REVIEWS as primary evidence when available; use comment text only as secondary support.\n"
    )
    return prompt


def simple_spell_correct(text: str) -> str:
    """Very small, deterministic spell-corrections for common typos relevant to this domain."""
    if not text:
        return text
    corrections = {"suspention": "suspension", "longtravel": "long-travel"}
    out = text
    for k, v in corrections.items():
        out = out.replace(k, v)
        out = out.replace(k.capitalize(), v)
    return out


def log_debug(msg: str):
    """Simple debug logger controlled by AIAGENT_DEBUG env var."""
    import os
    if os.environ.get("AIAGENT_DEBUG", "0") in ("1", "true", "True"):
        print("[DEBUG] ", msg)


def validate_and_filter(parsed: dict, conversation_history: list):
    """Validate and filter parsed JSON from the LLM.
    Returns (True, parsed) when valid (possibly with modified picks).
    Returns (False, info) when invalid; info contains 'reason' and 'action' keys.

    Behaviour implemented:
    - Numeric budget check: parse a budget from conversation_history (simple regex). If found,
      remove picks whose price_est > budget. For new format, filter primary and alternatives.
    - Attribute presence check: determines prioritized attribute token from last user message and checks
      that at least one pick mentions that attribute in 'reason' or 'evidence'. If no picks mention it,
      suggest a retry (action='retry') so the model can try again.
    """
    try:
        if not isinstance(parsed, dict):
            return False, {"reason": "parsed response is not an object", "action": "reject"}

        if parsed.get("type") != "recommendation":
            # nothing to validate for clarifying questions
            return True, parsed

        # Handle both old format (picks array) and new format (primary + alternatives)
        if "picks" in parsed:
            # Old format compatibility
            picks = parsed.get("picks", []) or []
            all_picks = picks
        else:
            # New format: primary + alternatives
            primary = parsed.get("primary")
            alternatives = parsed.get("alternatives", []) or []
            all_picks = []
            if primary:
                all_picks.append(primary)
            all_picks.extend(alternatives)

        # Simple budget extraction: find the last numeric token in the conversation that looks like a money value
        joined = " ".join(conversation_history or [])
        budget = None
        import re
        # look for patterns like $12,000 or 12000 or 12k
        m = re.search(r"\$\s*([0-9,]+(?:\.\d+)?)", joined)
        if not m:
            m = re.search(r"([0-9,]+(?:\.\d+)?)[\s]*k\b", joined, re.IGNORECASE)
            if m:
                try:
                    budget = float(m.group(1).replace(",", "")) * 1000
                except Exception:
                    budget = None
        else:
            try:
                budget = float(m.group(1).replace(",", ""))
            except Exception:
                budget = None

        # Helper function to check if a pick is within budget
        def is_within_budget(pick, budget_limit):
            if budget_limit is None:
                return True
            price = pick.get("price_est")
            try:
                # Some responses may use strings; sanitize
                if isinstance(price, str):
                    price_clean = re.sub(r"[^0-9.]", "", price)
                    price_val = float(price_clean) if price_clean else None
                else:
                    price_val = float(price) if price is not None else None
            except Exception:
                price_val = None
            
            if price_val is None:
                return True  # keep items with unknown price for now
            return price_val <= float(budget_limit)

        # If budget present, filter out picks that exceed it
        if budget is not None and all_picks:
            if "picks" in parsed:
                # Old format
                valid_picks = [p for p in all_picks if is_within_budget(p, budget)]
                if not valid_picks:
                    parsed["picks"] = []
                    parsed["note"] = parsed.get("note") or f"No items at or below the parsed budget ${int(budget)} found in dataset."
                else:
                    parsed["picks"] = valid_picks
            else:
                # New format: filter primary and alternatives separately
                primary = parsed.get("primary")
                alternatives = parsed.get("alternatives", [])
                
                # Filter primary
                if primary and not is_within_budget(primary, budget):
                    parsed["primary"] = None
                
                # Filter alternatives
                valid_alternatives = [a for a in alternatives if is_within_budget(a, budget)]
                parsed["alternatives"] = valid_alternatives
                
                # If no primary and no alternatives remain, add note
                if not parsed.get("primary") and not parsed.get("alternatives"):
                    parsed["note"] = parsed.get("note") or f"No items at or below the parsed budget ${int(budget)} found in dataset."

        # Attribute presence check: derive prioritized attribute from most recent user message
        prioritized = None
        if conversation_history:
            last = conversation_history[-1].lower()
            # keyword list (can be extended)
            keywords = ["suspension", "long-travel", "long travel", "travel", "soft", "firm", "damping", "offroad", "touring", "traveling", "comfort"]
            for k in keywords:
                if k in last:
                    prioritized = k
                    break

        if prioritized:
            # ensure at least one pick mentions the prioritized token in reason or evidence
            def mentions_attr(pick):
                for f in ("reason", "evidence"):
                    v = pick.get(f, "") or ""
                    if isinstance(v, (int, float)):
                        v = str(v)
                    if v and prioritized in v.lower():
                        return True
                return False

            # Check all picks for attribute mentions
            remaining_picks = []
            if "picks" in parsed:
                remaining_picks = parsed.get("picks", [])
            else:
                if parsed.get("primary"):
                    remaining_picks.append(parsed.get("primary"))
                remaining_picks.extend(parsed.get("alternatives", []))
            
            if remaining_picks:
                any_mention = any(mentions_attr(p) for p in remaining_picks)
                if not any_mention:
                    # tell the caller to retry once — the model should include the attribute in reasons or evidence
                    return False, {"reason": f"None of the picks mention the prioritized attribute '{prioritized}' in reason or evidence.", "action": "retry", "attribute": prioritized}

        return True, parsed
    except Exception as e:
        return False, {"reason": f"validation error: {e}", "action": "reject"}


def generate_retriever_query(conversation_history: list):
    """Ask the LLM to produce a short, focused retrieval query (single-line) from the recent conversation.
    Returns the query string or None on failure.
    """
    # Use the most recent up to 6 messages to give context but keep prompt short
    recent = conversation_history[-6:]
    convo_block = "\n".join([f"- {m}" for m in recent])
    prompt = (
        "You are a query generator for a document retriever.\n"
        "Given the recent user conversation below, produce a single short search query (no more than 12 words) that will retrieve the most relevant user reviews for satisfying the user's needs.\n"
        "Return only the query string on a single line and nothing else.\n\n"
        f"Conversation:\n{convo_block}\n"
    )
    try:
        out = invoke_model_with_prompt(prompt)
        if not out:
            # fallback: derive a deterministic query from the most recent user message
            fb = keyword_extract_query(conversation_history[-1] if conversation_history else "")
            return fb, True
        # sanitize: take first non-empty line
        for ln in out.splitlines():
            ln = ln.strip().strip('"')
            if ln:
                # avoid model returning JSON - if JSON, try to extract 'query'
                try:
                    j = json.loads(ln)
                    if isinstance(j, dict) and j.get("query"):
                        candidate = str(j.get("query")).strip()
                        # if candidate is too long, fallback
                        if len(candidate.split()) > 12:
                            fb = keyword_extract_query(conversation_history[-1] if conversation_history else "")
                            return fb, True
                        return candidate, False
                except Exception:
                    pass
                # if the model returned a very long line, use deterministic fallback
                if len(ln.split()) > 12:
                    fb = keyword_extract_query(conversation_history[-1] if conversation_history else "")
                    return fb, True
                return ln, False
    except Exception:
        fb = keyword_extract_query(conversation_history[-1] if conversation_history else "")
        return fb, True
    return None, False


def generate_retriever_query_str(conversation_history: list):
    """Compatibility wrapper that returns only the query string (old behavior)."""
    q, _ = generate_retriever_query(conversation_history)
    return q


def keyword_extract_query(user_message: str):
    """Deterministic fallback: extract important keywords from the user's last message and
    assemble a short query (<=12 words). This is lightweight (no external NLP libs).
    """
    if not user_message:
        return None
    import re
    msg = user_message.lower()
    # common stopwords to remove
    stop = set([
        'i', 'want', 'need', 'for', 'the', 'and', 'a', 'an', 'to', 'with', 'that', 'is', 'on', 'in', 'of', 'my', 'me',
        'it', 'are', 'please', 'would', 'like', 'want', 'looking', 'who'
    ])

    # preserve attribute and ride-type keywords with priority
    attributes = ["long-travel", "long travel", "suspension", "travel", "damping", "soft", "firm", "comfortable", "comfort", "fork", "shock"]
    ride_types = ["adventure", "touring", "cruiser", "sport", "offroad", "dual-sport", "enduro", "supermoto"]

    tokens = re.findall(r"[0-9]+cc|[a-zA-Z0-9\-]+", msg)
    seen = []
    # prioritize attributes & ride types
    for k in attributes + ride_types:
        if k in msg and k not in seen:
            seen.append(k)

    # then add other informative tokens (excluding stopwords)
    for t in tokens:
        t = t.strip()
        if not t or t in stop:
            continue
        if t in seen:
            continue
        # ignore very short tokens or pure numbers (unless cc)
        if re.fullmatch(r"\d+", t) and not t.endswith('cc'):
            continue
        if len(t) <= 2:
            continue
        seen.append(t)

    # final assembly: join unique tokens, limit to 12
    if not seen:
        return None
    query_tokens = seen[:12]
    return " ".join(query_tokens)


def generate_clarifying_question(conversation_history: list):
    """Ask the LLM to produce a single short clarifying question based on the recent conversation.
    Returns the question string, or None if no useful question could be generated.
    """
    recent = conversation_history[-4:] if conversation_history else []
    convo_block = "\n".join([f"- {m}" for m in recent])
    prompt = (
        "You are a concise assistant that asks a single short clarifying question when the user's message is vague.\n"
        "Given the recent conversation, return exactly one short question (one line) that will help you clarify the user's needs for motorcycle recommendations. "
        "Do not add any extra text.\n\n"
        f"Conversation:\n{convo_block}\n"
    )
    try:
        out = invoke_model_with_prompt(prompt)
        if not out:
            return None
        # take first non-empty line
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # naive guard: ignore greetings
            low = ln.lower()
            if low in ("hi", "hello", "hey") or low.startswith("hi ") or low.startswith("hello "):
                return None
            # ensure it looks like a question
            if not ln.endswith("?"):
                ln = ln.rstrip('.') + "?"
            return ln
    except Exception:
        return None


def analyze_with_llm(conversation_history: list, top_reviews: list):
    prompt = build_llm_prompt(conversation_history, top_reviews)
    response = invoke_model_with_prompt(prompt)
    # Attempt to parse JSON per the required format
    # Some LLMs or middleware may prepend debug-like lines (e.g. '[DEBUG]' or '[WARN]') — strip such leading lines
    def _sanitize_raw(text: str) -> str:
        lines = text.splitlines()
        cleaned = [ln for ln in lines if not ln.strip().startswith("[DEBUG]") and not ln.strip().startswith("[WARN]") and not ln.strip().startswith("[ERROR]")]
        return "\n".join(cleaned).strip()

    response = _sanitize_raw(response)
    try:
        parsed = json.loads(response.strip())
    except Exception:
        # LLM didn't return JSON — hand the raw (sanitized) response back to the caller
        return response

    # Perform client-side validation and allow one retry when attribute presence is missing
    valid, info = validate_and_filter(parsed, conversation_history)
    if not valid and isinstance(info, dict) and info.get("action") == "retry":
        # build a short re-prompt asking to follow the schema and emphasize the attribute/budget
        prioritized = info.get("attribute")
        retry_msg = (
            "Previous response did not mention the prioritized attribute in any pick. "
            "Please return the SAME JSON schema again, ensuring each pick.reason (<=12 words) mentions '" + (prioritized or "the prioritized attribute") + "' or set evidence to 'none in dataset'. "
            "Also strictly enforce numeric budget constraints if a budget was provided."
        )
        # ask model to regenerate following the same SYSTEM instructions
        retry_prompt = prompt + "\n\nRETRY_INSTRUCTION: " + retry_msg
        retry_resp = invoke_model_with_prompt(retry_prompt)
        retry_resp = retry_resp and retry_resp.strip()
        try:
            parsed_retry = json.loads(retry_resp)
            valid2, info2 = validate_and_filter(parsed_retry, conversation_history)
            if valid2:
                parsed = parsed_retry
            else:
                # fall back to reporting the validation issue
                return f"Model retry failed validation: {info2.get('reason')}. Returning model output for debugging: {retry_resp}"
        except Exception:
            return f"Model retry did not return valid JSON. Raw retry response: {retry_resp}"

    # Enrich picks using top_reviews metadata: if a pick lacks evidence, try to find it from the retrieved docs
    def _enrich_picks_with_metadata(parsed_obj, top_reviews_list):
        try:
            if not isinstance(parsed_obj, dict):
                return parsed_obj
            if parsed_obj.get("type") != "recommendation":
                return parsed_obj

            def _normalize(s):
                return (s or "").strip().lower()

            def evidence_from_review(r):
                # priority: suspension_notes, engine_cc, ride_type, price, comment/text
                # returns tuple (evidence, source_field) or None
                if r.get("suspension_notes"):
                    return r.get("suspension_notes"), "suspension_notes"
                if r.get("engine_cc"):
                    return f"{r.get('engine_cc')} cc", "engine_cc"
                if r.get("ride_type"):
                    return r.get("ride_type"), "ride_type"
                if r.get("price_usd_estimate"):
                    return f"Price est ${r.get('price_usd_estimate')}", "price_usd_estimate"
                if r.get("comment"):
                    return (r.get("comment") or "")[:200], "comment"
                if r.get("text"):
                    return (r.get("text") or "")[:200], "text"
                return None

            # Helper function to enrich a single pick
            def enrich_pick(p, top_reviews_list):
                ev = p.get("evidence") or ""
                if isinstance(ev, str) and ev.strip().lower() not in ("", "none", "none in dataset", "n/a", "na"):
                    return

                brand = _normalize(p.get("brand"))
                model = _normalize(p.get("model"))
                year = _normalize(str(p.get("year"))) if p.get("year") is not None else ""

                found = None
                for r in top_reviews_list:
                    # compare brand+model and optionally year
                    rb = _normalize(r.get("brand"))
                    rm = _normalize(r.get("model"))
                    ry = _normalize(str(r.get("year"))) if r.get("year") is not None else ""
                    if brand and model:
                        if brand in rb and model in rm or rb in brand and rm in model:
                            found = r
                            break
                    # fallback: match on model only
                    if model and (model in rm or rm in model):
                        found = r
                        break
                    # fallback: match on brand
                    if brand and (brand in rb or rb in brand):
                        found = r
                        break

                if found:
                    ev_result = evidence_from_review(found)
                    if ev_result:
                        evidence_text, source_field = ev_result
                        p["evidence"] = evidence_text
                        p["evidence_source"] = source_field
                        return

                # if we reach here, ensure evidence is explicit 'none in dataset' per prompt guidance
                p["evidence"] = "none in dataset"

            # Handle both old format (picks array) and new format (primary + alternatives)
            if "picks" in parsed_obj:
                # Old format
                picks = parsed_obj.get("picks", []) or []
                for p in picks:
                    enrich_pick(p, top_reviews_list)
                parsed_obj["picks"] = picks
            else:
                # New format: enrich primary and alternatives
                primary = parsed_obj.get("primary")
                if primary:
                    enrich_pick(primary, top_reviews_list)
                
                alternatives = parsed_obj.get("alternatives", []) or []
                for alt in alternatives:
                    enrich_pick(alt, top_reviews_list)
            return parsed_obj
        except Exception:
            return parsed_obj

    try:
        parsed = _enrich_picks_with_metadata(parsed, top_reviews)
    except Exception:
        # if enrichment fails, continue with original parsed
        pass

    # If parsed successfully, convert to a readable display
    try:
        if parsed.get("type") == "clarify":
            return parsed.get("question", "(no question provided)")
        elif parsed.get("type") == "recommendation":
            # Handle both old format (picks array) and new format (primary + alternatives)
            if "primary" in parsed or "alternatives" in parsed:
                # New format: primary + alternatives
                primary = parsed.get("primary")
                alternatives = parsed.get("alternatives", [])
                lines = []
                
                if primary:
                    # Display primary pick prominently
                    lines.append("Top recommendation:")
                    ev = primary.get('evidence')
                    ev_source = primary.get('evidence_source')
                    if ev:
                        ev_text = f" Evidence: {ev}"
                    else:
                        ev_text = ""
                    lines.append(f"• {primary.get('brand')} {primary.get('model')} ({primary.get('year')}), Price est: ${primary.get('price_est')}. Reason: {primary.get('reason')}.{ev_text}")
                    
                    # Display alternatives concisely if present
                    if alternatives:
                        lines.append("\nAlternatives:")
                        for alt in alternatives:
                            lines.append(f"• {alt.get('brand')} {alt.get('model')} ({alt.get('year')}) - ${alt.get('price_est')}. {alt.get('reason')}")
                else:
                    # No primary pick
                    note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                    lines.append(f"No picks matched strictly. Note: {note}")
                    
            else:
                # Old format: picks array (for backward compatibility)
                picks = parsed.get("picks", [])
                lines = ["Top recommendations:"]
                if not picks:
                    note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                    lines.append(f"No picks matched strictly. Note: {note}")
                else:
                    for p in picks:
                        ev = p.get('evidence')
                        ev_source = p.get('evidence_source')
                        if ev:
                            ev_text = f" Evidence: {ev}"
                        else:
                            ev_text = ""
                        lines.append(f"- {p.get('brand')} {p.get('model')} ({p.get('year')}), Price est: ${p.get('price_est')}. Reason: {p.get('reason')}.{ev_text}")
                        
            if parsed.get("note") and (parsed.get("primary") or parsed.get("picks")):
                lines.append(f"Note: {parsed.get('note')}")
            return "\n".join(lines)
        else:
            return response
    except Exception:
        return response


def main_cli():
    while True:
        print("\n\n--------------------------------")
        user_preferences = input("What are your motorcycle preferences? (Type 'q' to quit): ")
        print("\nThinking...\n")
        if user_preferences.lower() == 'q':
            break

        # Maintain a short conversation history (user messages)
        try:
            conversation_history
        except NameError:
            conversation_history = []

        conversation_history.append(user_preferences)

        # Quick vagueness check: if the user message is very short and doesn't contain attribute tokens,
        # ask the LLM for a clarifying question instead of querying the retriever.
        tokens = [t for t in user_preferences.split() if t.strip()]
        if len(tokens) < 3:
            # look for attribute tokens
            low = user_preferences.lower()
            attr_tokens = ["suspension", "travel", "long-travel", "long travel", "budget", "touring", "adventure"]
            if not any(a in low for a in attr_tokens):
                cq = generate_clarifying_question(conversation_history)
                if cq:
                    print("\nClarifying question:\n", cq)
                    # append the clarifying question to the conversation history (so subsequent messages are contextual)
                    conversation_history.append(cq)
                    continue

        # Fetch top relevant reviews from the prebuilt retriever in vector.py
        # Ensure the retriever exported by vector.py is available
        if retriever is None:
            print("[ERROR] Retriever not available. Ensure `vector.py` builds and exports `retriever`. Exiting.")
            sys.exit(1)

        # Query the retriever using a model-generated query (fallback to conversation join)
        try:
            q_res = generate_retriever_query(conversation_history)
            if isinstance(q_res, tuple):
                query, used_fallback = q_res
            else:
                # backward compatibility: single-value return
                query = q_res
                used_fallback = False
            if not query:
                query = (" ".join(conversation_history[-3:]) if conversation_history else user_preferences)

            # Inform the user when the deterministic fallback was used for transparency
            if used_fallback:
                print("[INFO] Using deterministic fallback query for retriever (model query was empty/too long).")
            def get_docs_from_retriever(ret, q):
                """Compatibility helper: prefer new `invoke` method, fall back to `get_relevant_documents`.
                Normalize return shapes to a list of Document-like objects with `metadata` and `page_content`.
                """
                # prefer invoke (newer LangChain)
                try:
                    if hasattr(ret, "invoke"):
                        out = ret.invoke(q)
                        # out might be a dict {'results': [...]}, or have .docs
                        if isinstance(out, dict) and "results" in out:
                            # results can be list of dicts with 'documents'
                            docs = []
                            for r in out.get("results", []):
                                docs.extend(r.get("documents", []) or [])
                            return docs
                        # if it's already a list-like
                        if isinstance(out, list):
                            return out
                        # if it has attribute 'docs' or 'documents'
                        if hasattr(out, "docs"):
                            return list(out.docs)
                        if hasattr(out, "documents"):
                            return list(out.documents)
                        # unknown shape — return as single-item list
                        return [out]
                except Exception:
                    pass

                # fallback for older versions
                if hasattr(ret, "get_relevant_documents"):
                    return ret.get_relevant_documents(q)

                raise RuntimeError("Retriever has neither 'invoke' nor 'get_relevant_documents' methods")

            docs = get_docs_from_retriever(retriever, query)
            # Convert docs to a simple dict list expected by the prompt builder
            top_reviews = []
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                top_reviews.append({
                    "brand": meta.get("brand"),
                    "model": meta.get("model"),
                    "year": meta.get("year"),
                    "comment": meta.get("comment") or getattr(d, "page_content", ""),
                    "price_usd_estimate": (int(meta.get("price_usd_estimate")) if meta.get("price_usd_estimate") is not None else (int(meta.get("price")) if meta.get("price") is not None else None)),
                    "price_est": (int(meta.get("price_usd_estimate")) if meta.get("price_usd_estimate") is not None else (int(meta.get("price")) if meta.get("price") is not None else None)),
                    "engine_cc": meta.get("engine_cc"),
                    "suspension_notes": meta.get("suspension_notes"),
                    "ride_type": meta.get("ride_type"),
                    "text": getattr(d, "page_content", ""),
                })
        except Exception as e:
            print(f"[ERROR] Failed to query retriever: {e}")
            sys.exit(1)

        # Let the LLM analyze the conversation history and decide whether to ask a follow-up or recommend
        # Implement retry-on-invalid-response loop with up to 1 retry
        max_retries = 1
        retry_count = 0
        llm_response = None
        
        while retry_count <= max_retries:
            try:
                llm_response = analyze_with_llm(conversation_history, top_reviews)
                
                # Check if response indicates a validation failure or JSON parsing error
                if isinstance(llm_response, str):
                    # Check for specific error patterns that indicate retry-worthy failures
                    error_indicators = [
                        "Model retry failed validation:",
                        "Model retry did not return valid JSON",
                        "Error invoking model"
                    ]
                    
                    is_retry_worthy_error = any(indicator in llm_response for indicator in error_indicators)
                    
                    if is_retry_worthy_error and retry_count < max_retries:
                        print(f"[RETRY {retry_count + 1}/{max_retries}] LLM response didn't follow the required format, retrying...")
                        retry_count += 1
                        continue
                    elif is_retry_worthy_error and retry_count >= max_retries:
                        # Exhausted retries, provide helpful error message
                        print("[ERROR] LLM failed to provide a valid response after multiple attempts.")
                        print("This may be due to:")
                        print("- The model not following the JSON schema requirements")
                        print("- Budget constraints that can't be met with available data")
                        print("- Missing attribute evidence in the dataset")
                        print("Try rephrasing your request or adjusting your requirements.\n")
                        print("Debug info:", llm_response[:200] + ("..." if len(llm_response) > 200 else ""))
                        break
                    elif llm_response.startswith("Error invoking model.generate"):
                        # Surface LLM invocation errors directly to the user (no retry for connection issues)
                        print("[ERROR] LLM invocation failed:\n")
                        print(llm_response)
                        break
                
                # If we get here, either the response is valid or we've exhausted retries
                break
                
            except Exception as e:
                if retry_count < max_retries:
                    print(f"[RETRY {retry_count + 1}/{max_retries}] Unexpected error during LLM analysis, retrying...")
                    retry_count += 1
                    continue
                else:
                    print(f"[ERROR] Failed to get valid response from LLM after {max_retries + 1} attempts.")
                    print("This could be due to:")
                    print("- Temporary connectivity issues with the Ollama service") 
                    print("- Model loading problems")
                    print("- Invalid conversation context")
                    print(f"Last error: {e}")
                    llm_response = None
                    break
        
        # Handle final response or error
        if llm_response is None:
            print("[ERROR] No response received from LLM after all attempts")
            continue
        elif isinstance(llm_response, str) and llm_response.startswith("[ERROR]"):
            print(llm_response)
            continue
        
        print(llm_response)


if __name__ == "__main__":
    main_cli()