from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import re

df = pd.read_csv("motorcycle_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Build a simple metadata object by extracting common fields and quick features from the row text
        row_dict = row.to_dict()
        # helper text fields
        text_fields = []
        for k in ("comment", "text", "review", "notes"):
            if k in row_dict and pd.notna(row_dict.get(k)):
                text_fields.append(str(row_dict.get(k)))
        # fallback to the stringified row if no specific text field
        fallback_text = str(row_dict)
        full_text = " ".join(text_fields) if text_fields else fallback_text

        def parse_price(s: str):
            if not s:
                return None
            # look for $12,000 or 12000 or 12k
            m = re.search(r"\$\s*([0-9,]+(?:\.\d+)?)", s)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    return None
            m = re.search(r"([0-9,]+(?:\.\d+)?)[\s]*k\b", s, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", "")) * 1000
                except Exception:
                    return None
            # plain number
            m = re.search(r"\b([0-9]{3,6})(?:\.[0-9]+)?\b", s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

        def parse_engine_cc(s: str):
            if not s:
                return None
            m = re.search(r"(\d{2,4})\s?cc\b", s, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            # sometimes written as '650cc' without space
            m = re.search(r"\b(\d{2,4})cc\b", s, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        def extract_suspension_notes(s: str):
            if not s:
                return None
            keywords = ["suspension", "travel", "long-travel", "long travel", "damping", "firm", "plush", "soft", "wp", "showa", "fork travel"]
            found = []
            for k in keywords:
                if k in s.lower():
                    found.append(k)
            # return unique, comma-joined short notes
            return ", ".join(sorted(set(found))) if found else None

        def extract_ride_type(s: str):
            if not s:
                return None
            types = ["adventure", "touring", "cruiser", "sport", "offroad", "dual-sport", "enduro", "supermoto"]
            for t in types:
                if t in s.lower():
                    return t
            return None

        # Try to pull a price from dedicated row keys first
        price = None
        for pk in ("price_usd_estimate", "price_est", "price", "msrp", "price_usd"):
            if pk in row_dict and pd.notna(row_dict.get(pk)):
                price = parse_price(str(row_dict.get(pk)))
                if price is not None:
                    break

        # If still no price, try parsing the text
        if price is None:
            price = parse_price(full_text)

        engine_cc = None
        for ek in ("engine_cc", "cc", "displacement"):
            if ek in row_dict and pd.notna(row_dict.get(ek)):
                try:
                    engine_cc = int(float(str(row_dict.get(ek))))
                    break
                except Exception:
                    engine_cc = None

        if engine_cc is None:
            engine_cc = parse_engine_cc(full_text)

        suspension_notes = extract_suspension_notes(full_text)
        ride_type = extract_ride_type(full_text)

        metadata = {
            "source": f"motorcycle_reviews.csv - row {row.name}",
            "brand": row_dict.get("brand") if "brand" in row_dict else None,
            "model": row_dict.get("model") if "model" in row_dict else None,
            "year": row_dict.get("year") if "year" in row_dict else None,
            "price_usd_estimate": int(price) if price is not None else None,
            "engine_cc": engine_cc,
            "suspension_notes": suspension_notes,
            "ride_type": ride_type,
        }

        document = Document(
            page_content=full_text,
            metadata=metadata,
            id=str(i)
        )

        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="motorcycle_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})