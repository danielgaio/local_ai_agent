from langchain_ollama.llms import OllamaLLM
import json
import sys
# Use the prebuilt retriever exported by vector.py. This module must provide a `retriever` variable.
from vector import retriever


# Initialize model
model = OllamaLLM(model="llama3.2:3b")

SYSTEM_INSTRUCTIONS = """
You are an expert motorcycle recommender. The user will provide one or more messages describing preferences.
Always analyze the user's messages, decide if you have enough information to recommend up to 3 motorcycles from the provided dataset, or ask a single clear follow-up question to clarify missing information.
Do not rely on local deterministic keyword parsing in the client; perform the analysis and decision-making inside the model.
When recommending, strictly enforce numeric budget constraints when provided: exclude any motorcycle whose listed price exceeds the user's stated budget. If nothing in the dataset strictly matches the budget, explicitly say so and suggest the closest alternatives under the budget or advise on raising the budget.
Respect explicit constraints (budget, cylinder count, riding style, experience), and explain why each recommended motorcycle matches the user's preferences.
If you need more information, ask exactly one short clarifying question. Otherwise, recommend up to 3 motorcycles from the dataset and explain your reasoning.

Priority and concision guidance:
- If the user's most recent message requests a specific attribute (for example 'big suspension'), prioritize that attribute above all others when selecting and ranking motorcycles.
- For each pick include a short, suspension/attribute-focused reason (max 12 words) in the `reason` field, and include an `evidence` field (one short phrase) if the reviews or specs mention the attribute; if none, set `evidence` to "none in dataset".
- Return exactly one JSON object following the prescribed shapes. Keep reasons concise and focused on the prioritized attribute.

RESPONSE FORMAT (REQUIRED): Return a single JSON object only (no surrounding text). The object must be one of two shapes:

1) Clarifying question:
    {"type": "clarify", "question": "<one short question the assistant needs>"}

2) Recommendation:
    {"type": "recommendation", "picks": [ {"brand": "", "model": "", "year": 0, "price_est": 0, "reason": "", "evidence": ""}, ... up to 3 items ], "note": "optional free-text note if nothing strictly matches budget"}

Strict rules:
- Return exactly one JSON object and nothing else (no extra commentary). The client will parse this JSON. Follow the shapes above precisely.
- When recommending, only include items whose numeric price_est is <= the user's stated budget (if budget provided). If none match, set "picks": [] and include an explanatory "note".

Template requirement (strict):
- Each pick.reason MUST be 12 words or fewer and should explicitly mention the prioritized attribute (for example: "long-travel suspension"). If the provided REVIEWS or metadata do not contain direct evidence for the prioritized attribute, put the literal string "none in dataset" in the `evidence` field for that pick.
- Example pick (for clarity): {"brand":"KTM","model":"790 Adventure","year":2019,"price_est":10000,"reason":"long-travel suspension for offroad comfort","evidence":"fork travel 210mm"}

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
    reviews_text = "\n".join([
        f"- {r.get('brand', '')} {r.get('model', '')} ({r.get('year', '')}): {r.get('comment', r.get('text', ''))} Price est ${r.get('price_usd_estimate', 'unknown')}"
        for r in top_reviews
    ])
    # Insert a short USER FOCUS hint using the most recent user message to prioritize attributes
    user_focus = conversation_history[-1] if conversation_history else ""
    prompt = (
        f"SYSTEM:\n{SYSTEM_INSTRUCTIONS}\n\nCONVERSATION:\n{convo_text}\n\nREVIEWS:\n{reviews_text}\n\n"
        f"USER FOCUS: {user_focus} -- prioritize this attribute when ranking and in each reason.\n\n"
        "TASK: Based on the conversation above, either ask one short clarifying question (if you need more info) or recommend up to 3 motorcycles from the REVIEWS that best match the user's needs. Be explicit about why each pick matches.\n\n"
        "RESPONSE EXAMPLE AND GUIDANCE:\n"
        "Return exactly one JSON object as specified in SYSTEM instructions.\n"
        "Tiny schema example (return exactly this shape, with real values):\n"
        "{'type':'recommendation', 'picks':[{'brand':'', 'model':'', 'year':0, 'price_est':0, 'reason':'(<=12 words mentioning prioritized attribute)', 'evidence':'(short phrase or \"none in dataset\")'}], 'note':''}\n"
        "If you cannot find direct evidence for the prioritized attribute inside the provided REVIEWS or metadata for a pick, set that pick's evidence to the literal string 'none in dataset'.\n"
    )
    return prompt


def validate_and_filter(parsed: dict, conversation_history: list):
    """Validate and filter parsed JSON from the LLM.
    Returns (True, parsed) when valid (possibly with modified picks).
    Returns (False, info) when invalid; info contains 'reason' and 'action' keys.

    Behaviour implemented:
    - Numeric budget check: parse a budget from conversation_history (simple regex). If found,
      remove picks whose price_est > budget. If none remain, set picks=[] and return valid with a note.
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

        picks = parsed.get("picks", []) or []

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

        # If budget present, filter out picks that exceed it
        if budget is not None and picks:
            valid_picks = []
            for p in picks:
                price = p.get("price_est")
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
                    # keep items with unknown price for now
                    valid_picks.append(p)
                else:
                    if price_val <= float(budget):
                        valid_picks.append(p)
            if not valid_picks:
                # No picks under budget — enforce requirement: set picks empty and include explanatory note
                parsed["picks"] = []
                parsed["note"] = parsed.get("note") or f"No items at or below the parsed budget ${int(budget)} found in dataset."
                return True, parsed
            else:
                parsed["picks"] = valid_picks

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

            if parsed.get("picks"):
                any_mention = any(mentions_attr(p) for p in parsed.get("picks", []))
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
            return None
        # sanitize: take first non-empty line
        for ln in out.splitlines():
            ln = ln.strip().strip('"')
            if ln:
                # avoid model returning JSON - if JSON, try to extract 'query'
                try:
                    j = json.loads(ln)
                    if isinstance(j, dict) and j.get("query"):
                        return str(j.get("query")).strip()
                except Exception:
                    pass
                return ln
    except Exception:
        return None
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

    # If parsed successfully, convert to a readable display
    try:
        if parsed.get("type") == "clarify":
            return parsed.get("question", "(no question provided)")
        elif parsed.get("type") == "recommendation":
            picks = parsed.get("picks", [])
            lines = ["Top recommendations:"]
            if not picks:
                note = parsed.get("note", "No recommendations match the strict budget or constraints.")
                lines.append(f"No picks matched strictly. Note: {note}")
            else:
                for p in picks:
                    ev = p.get('evidence')
                    ev_text = f" Evidence: {ev}" if ev else ""
                    lines.append(f"- {p.get('brand')} {p.get('model')} ({p.get('year')}), Price est: ${p.get('price_est')}. Reason: {p.get('reason')}.{ev_text}")
            if parsed.get("note") and picks:
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

        # Fetch top relevant reviews from the prebuilt retriever in vector.py
        # Ensure the retriever exported by vector.py is available
        if retriever is None:
            print("[ERROR] Retriever not available. Ensure `vector.py` builds and exports `retriever`. Exiting.")
            sys.exit(1)

        # Query the retriever using a model-generated query (fallback to conversation join)
        try:
            query = generate_retriever_query(conversation_history) or (" ".join(conversation_history[-3:]) if conversation_history else user_preferences)
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
                    "price_usd_estimate": meta.get("price_usd_estimate") or meta.get("price") or None,
                    "text": getattr(d, "page_content", ""),
                })
        except Exception as e:
            print(f"[ERROR] Failed to query retriever: {e}")
            sys.exit(1)

        # Let the LLM analyze the conversation history and decide whether to ask a follow-up or recommend
        llm_response = analyze_with_llm(conversation_history, top_reviews)
        # Surface any LLM invocation errors directly to the user (no local fallback)
        if isinstance(llm_response, str) and llm_response.startswith("Error invoking model.generate"):
            print("[ERROR] LLM invocation failed:\n")
            print(llm_response)
            # continue loop so the user can try again or quit
            continue
        print(llm_response)


if __name__ == "__main__":
    main_cli()