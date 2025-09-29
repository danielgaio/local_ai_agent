Concrete, prioritized improvements (highest impact first)

1. ~~Strengthen prompt + enforce compact reasons (apply immediately)
Why: prompt is the single biggest lever for LLM behavior.
Changes:
Add a short template to SYSTEM_INSTRUCTIONS (already partially added) requiring each pick.reason <= 12 words and must mention the prioritized attribute or “none in dataset”.
In build_llm_prompt(), include a tiny JSON schema example for responses and a clear "If you cannot find direct evidence, say 'none in dataset' in the evidence field".
Risk: none; purely prompt-level.~~

2. ~~Add minimal client-side validation for critical constraints (budget & attribute presence) — must for correctness
Why: LLM may ignore numeric constraints or format incorrectly.
Behaviour:
After parsing JSON from model, perform:
Numeric budget check: if user provided a budget in conversation_history, parse numeric budget (simple regex), filter picks or mark picks that exceed budget and set picks = [] + note when no valid picks.
Attribute presence check: verify pick.reason or pick.evidence contains prioritized attribute token (e.g., "suspension", "long-travel", "travel", "soft", "firm", "damping"); if none of picks mention attribute, either ask model to retry once or flag note.
Implementation: small function validate_and_filter(parsed_json, conversation_history) returning either valid parsed object or a signal to re-prompt/ask for clarification.~~

3. ~~Enrich vector index metadata (index-time evidence)
Why: retrieval will be more precise when metadata contains fields for common attributes (suspension notes, engine_cc, riding_style, price_est).
Changes in vector.py:
During indexing, parse the review text (or row dict) to extract:
price_est numeric (if available) => metadata["price_usd_estimate"]
suspension-related short notes (scan for keywords: suspension, travel, long-travel, damping, firm, plush) => metadata["suspension_notes"]
cylinder count or engine_cc => metadata["engine_cc"]
ride_type => metadata["ride_type"]
Store structured metadata per document (so main.py can show evidence from metadata rather than rely on LLM to find it).
Small NLP snippet: a short regex/keyword extractor in vector.py prior to Document creation.~~

    ~~Update the prompt guidance to tell the model to prefer metadata evidence fields (e.g., "Prefer suspension_notes and engine_cc from REVIEWS when available") — small change I can apply now.~~
    ~~Add a unit test for validate_and_filter and a test that ensures top_reviews contains the new fields for stubbed documents.~~
    ~~Run an end-to-end interactive test with the real Chroma DB (if available) to verify retrieval/evidence flows.~~

4. ~~Make the retriever-query generation deterministic/fallback-safe
Why: current approach asks LLM to produce a query; this is flexible but fragile.
Changes:
Keep LLM generator, but add deterministic fallback generator: extract important keywords from last user message (nouns and phrases) and build a short query (e.g., "suspension travel long-travel adventure touring").
If the LLM-generated query is empty or >12 words, use deterministic fallback.
Implementation: small function keyword_extract_query(user_message).~~

    ~~Integrating the deterministic fallback into main_cli() display or logging when it's used (for transparency).~~
        ~~Force clarifying-question behavior for vague input (highly recommended)
        Add a short check in main_cli() before retrieval: if the user message is very short (<3 tokens) and lacks attribute tokens, call the LLM with a clarifying-question prompt (or directly ask the user) instead of fetching documents. This will prevent cases like input "hi" from returning recommendations.~~
        ~~Add a unit test that monkeypatches invoke_model_with_prompt to return empty/long output and asserts main_cli() prints the fallback info line (or test generate_retriever_query() returns used_fallback True).~~
    ~~Adding unit tests that verify generate_retriever_query() falls back appropriately when the model returns empty/long outputs (by monkeypatching invoke_model_with_prompt).~~

5. ~~Add evidence field to picks automatically when metadata exists
Why: if index metadata contains suspension_notes, automatically propagate it into the evidence field for picks that were selected from retrieved docs.
Changes:
In main.py, when building top_reviews, include suspension_notes from doc metadata; then the prompt provides reviews with clear evidence and the LLM can reference them.~~

    ~~Add a unit test for enrichment behavior (stub parsed JSON and top_reviews to assert evidence gets filled).~~
    
    ~~Change the UI to display which metadata field provided the evidence (e.g., "Evidence (suspension_notes): ...").~~

6. ~~Add a retry-on-invalid-response loop
Why: the model may output invalid JSON or not follow the schema.
Behavior:
If parsed JSON fails budget/attribute checks, reprompt the LLM with a short message: "Previous response did not meet X; please return again following the JSON schema" — allow 1 retry then surface a clear error.
Implementation: small loop in main.py around analyze_with_llm call.~~

7. ~~Tests and smoke checks (automated)
Add fast unit tests:
test_index_metadata_presence: that vector.py sets suspension_notes and price metadata.
test_llm_response_shape: stub invoke_model_with_prompt to return valid/invalid JSON and test analyze_with_llm behavior.
test_budget_enforcement: run parse->validate pipeline on sample parsed JSON with budget.
Add one smoke script that runs python main.py with a piped input sample and asserts output contains "Top recommendations" or a clarifying question.~~

8. ~~UX: return 1 primary pick + 1-2 alternatives (concise)
Implementation: change prompt to instruct model to mark one primary; client displays primary first and succinct alternatives.~~

9. Add requirements and README (developer ergonomics)
Ensure requirements.txt includes exact libs used: langchain, langchain-chroma or chromadb, langchain-ollama/ollama, pandas, chromadb adapter versions. Provide a README with steps to rebuild chroma_langchain_db.

10. Small improvements (low effort)
Normalize price metadata keys to numeric in index and in top_reviews (ensure price_est is numeric int).
Spell-check user input lightly (fix "suspention" -> "suspension" before passing as focus).
Log-level toggles for debugging (hide debug by default).

11. The following behaviour is incorrect, the model should have asked for preferences. Based on the log, add a fix to ensure the model prompts for user preferences when the initial input is vague or incomplete. Log: (env) danielgaio@Daniels-MacBook-Air local_ai_agent % python main.py


--------------------------------
What are your motorcycle preferences? (Type 'q' to quit): hi

Thinking...

Top recommendations:
- Yonder Explorer 700 (2022), Price est: $8999. Reason: long-distance capabilities for adventure riding. Evidence: none in dataset