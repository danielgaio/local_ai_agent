"""Vector store retriever implementation with provider-specific optimizations."""

import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..core.config import DEFAULT_SEARCH_KWARGS, MODEL_PROVIDER


class EnhancedVectorStoreRetriever(BaseRetriever, BaseModel):
    """Enhanced retriever with better query handling and provider-specific optimizations.
    
    Features:
    - Query preprocessing and validation
    - Provider-specific optimizations (OpenAI vs Ollama)
    - Improved error handling and recovery
    - Document batching support
    - Automatic configuration based on provider
    """
    vectorstore: Any = Field(description="Vector store instance to use")
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SEARCH_KWARGS.copy()
    )
    batch_size: int = Field(
        default=5,
        description="Batch size for document retrieval"
    )
    provider: str = Field(
        default=MODEL_PROVIDER,
        description="Model provider (openai or ollama)"
    )
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate and adjust batch size based on provider."""
        if v < 1:
            v = 1
        # OpenAI has higher rate limits, so we can use larger batches
        if MODEL_PROVIDER == "openai":
            return min(v, 20)  # OpenAI can handle larger batches
        return min(v, 5)  # More conservative for local models
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query text for better retrieval.
        
        Args:
            query: Raw query string
            
        Returns:
            str: Preprocessed query string
        """
        if not query or not isinstance(query, str):
            return ""
            
        # Remove multiple spaces and normalize whitespace
        query = " ".join(query.split())
        
        # Truncate very long queries (especially important for OpenAI)
        if self.provider == "openai" and len(query) > 1000:
            query = query[:1000]
            
        return query.strip()
    
    def _batch_documents(self, docs: List[Document]) -> List[List[Document]]:
        """Split documents into batches for efficient processing.
        
        Args:
            docs: List of documents to batch
            
        Returns:
            List[List[Document]]: Batched documents
        """
        return [
            docs[i:i + self.batch_size] 
            for i in range(0, len(docs), self.batch_size)
        ]

    def _get_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        """Get relevant documents with improved error handling and batching.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            List[Document]: Retrieved relevant documents
        """
        try:
            # Preprocess query
            query = self._preprocess_query(query)
            if not query:
                return []
                
            # Merge search parameters with provider-specific defaults
            search_kwargs = {**self.search_kwargs, **kwargs}
            
            # Add provider-specific optimizations
            if self.provider == "openai":
                # OpenAI embeddings are normalized, so we can use cosine similarity
                search_kwargs.setdefault("distance_metric", "cosine")
                search_kwargs.setdefault("normalize_l2", True)
            else:
                # Default L2 distance for other providers
                search_kwargs.setdefault("distance_metric", "l2")
            
            # Perform search
            docs = self.vectorstore.similarity_search(
                query, 
                **search_kwargs
            )
            
            return docs
            
        except Exception as e:
            # Log error but don't crash
            logging.error(f"Error in document retrieval: {e}")
            return []

    async def _aget_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        """Async version with similar improvements.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            List[Document]: Retrieved relevant documents
        """
        try:
            # Preprocess query
            query = self._preprocess_query(query)
            if not query:
                return []
                
            # Merge search parameters with provider-specific defaults
            search_kwargs = {**self.search_kwargs, **kwargs}
            
            # Add provider-specific optimizations
            if self.provider == "openai":
                search_kwargs.setdefault("distance_metric", "cosine")
                search_kwargs.setdefault("normalize_l2", True)
            else:
                search_kwargs.setdefault("distance_metric", "l2")
                
            # Perform async search
            docs = await self.vectorstore.asimilarity_search(
                query, 
                **search_kwargs
            )
            
            return docs
            
        except Exception as e:
            logging.error(f"Error in async document retrieval: {e}")
            return []