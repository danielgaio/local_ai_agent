"""Vector store implementation for the motorcycle recommendation system."""

import os
import pandas as pd
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..core.config import DB_LOCATION, DATA_FILE
from ..core.models import MotorcycleReview
from ..utils.parsers import (
    parse_price, parse_engine_cc,
    extract_suspension_notes, extract_ride_type
)
from .embeddings import init_embeddings


def build_metadata(text_fields: List[str], row_dict: Dict) -> Dict:
    """Build metadata dict from review text and fields."""
    # Join text fields or use fallback
    full_text = " ".join(text_fields) if text_fields else str(row_dict)

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

    # Extract engine displacement
    engine_cc = None
    for ek in ("engine_cc", "cc", "displacement"):
        if ek in row_dict and pd.notna(row_dict.get(ek)):
            try:
                engine_cc = int(float(str(row_dict.get(ek))))
                break
            except (ValueError, TypeError):
                engine_cc = None

    if engine_cc is None:
        engine_cc = parse_engine_cc(full_text)

    # Extract other metadata
    suspension_notes = extract_suspension_notes(full_text)
    ride_type = extract_ride_type(full_text)

    return {
        "source": f"{DATA_FILE} - row {row_dict.get('name', 'unknown')}",
        "brand": row_dict.get("brand") if "brand" in row_dict else None,
        "model": row_dict.get("model") if "model" in row_dict else None,
        "year": row_dict.get("year") if "year" in row_dict else None,
        "price_usd_estimate": int(price) if price is not None else None,
        "engine_cc": engine_cc,
        "suspension_notes": suspension_notes,
        "ride_type": ride_type,
    }


def load_vector_store() -> Chroma:
    """Initialize or load the vector store with motorcycle reviews.

    Returns:
        Chroma: The initialized vector store
    """
    # Initialize embeddings
    embeddings = init_embeddings()

    # Create/load vector store
    add_documents = not os.path.exists(DB_LOCATION)
    vector_store = Chroma(
        collection_name="motorcycle_reviews",
        embedding_function=embeddings,
        persist_directory=DB_LOCATION
    )

    # Add documents if needed
    if add_documents:
        df = pd.read_csv(DATA_FILE)
        documents = []
        ids = []

        for i, row in df.iterrows():
            # Extract text fields
            row_dict = row.to_dict()
            text_fields = []
            for k in ("comment", "text", "review", "notes"):
                if k in row_dict and pd.notna(row_dict.get(k)):
                    text_fields.append(str(row_dict.get(k)))

            # Build metadata
            metadata = build_metadata(text_fields, row_dict)

            # Create document
            document = Document(
                page_content=" ".join(text_fields) if text_fields else str(row_dict),
                metadata=metadata,
                id=str(i)
            )

            documents.append(document)
            ids.append(str(i))

        # Add all documents
        vector_store.add_documents(documents, ids=ids)

    return vector_store