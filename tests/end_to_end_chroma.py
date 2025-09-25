"""
End-to-end check: load the real retriever from `vector.py`, run a suspension-focused query,
and print the returned documents' metadata to verify index-time evidence fields are present.

Run with: python3 tests/end_to_end_chroma.py
"""
import sys
import os
# ensure project root is on sys.path so local modules like `vector` import correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vector import retriever


def run_retrieval(query: str):
    print(f"Running retrieval for query: {query}\n")
    # try both modern invoke and fallback
    docs = None
    try:
        if hasattr(retriever, "invoke"):
            out = retriever.invoke(query)
            # If it's dict-like with results -> documents
            if isinstance(out, dict) and "results" in out:
                docs = []
                for r in out.get("results", []):
                    docs.extend(r.get("documents", []) or [])
            elif hasattr(out, "docs"):
                docs = list(out.docs)
            elif isinstance(out, list):
                docs = out
            else:
                docs = [out]
        else:
            docs = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"Retriever invocation failed: {e}")
        sys.exit(1)

    print(f"Retrieved {len(docs)} documents\n")
    for i, d in enumerate(docs[:10], 1):
        meta = getattr(d, "metadata", {}) or {}
        print(f"Doc {i} metadata:")
        for k in ("brand", "model", "year", "price_usd_estimate", "engine_cc", "suspension_notes", "ride_type", "source"):
            if k in meta:
                print(f"  {k}: {meta.get(k)}")
        # print a short snippet of content
        content = getattr(d, "page_content", "")
        print(f"  snippet: {content[:200]}\n")


if __name__ == "__main__":
    q = "long-travel suspension adventure touring"
    run_retrieval(q)
