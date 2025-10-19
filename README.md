# local_ai_agent (developer README)

This repository provides a small local RAG-powered motorcycle recommender using Ollama (LLM) and ChromaDB (vector store).

Prerequisites
-------------
- Python 3.10+
- Ollama (if you want to run LLM locally) and the necessary models pulled (see below).

Quick setup
-----------
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Pull Ollama models used by the project:

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

Rebuild the Chroma DB
---------------------
- The Chroma DB is persisted at `./chroma_langchain_db`.
- To rebuild the index from `motorcycle_reviews.csv`, delete the `./chroma_langchain_db` directory and run the app once:

```bash
rm -rf ./chroma_langchain_db
python main.py
```

This will call `vector.py` which reads `motorcycle_reviews.csv`, extracts lightweight metadata (price, suspension notes, engine cc, ride_type) and populates the Chroma collection `motorcycle_reviews`.

Running the recommender
-----------------------
Run:

```bash
python main.py
```

The CLI will prompt for preferences. The system uses a combination of model-driven and deterministic retrieval query generation and will ask clarifying questions when input is vague.

Tests
-----
Run the test suite with:

```bash
python -m pytest -q
```

Notes
-----
- If you run into issues with `chromadb` or `langchain-chroma`, check package compatibility in `requirements.txt` and update accordingly.
- The project contains a smoke test `tests/end_to_end_chroma.py` that demonstrates retrieval against the local Chroma DB.

Contact
-------
If you change indexing code or dependencies, update this README accordingly.
