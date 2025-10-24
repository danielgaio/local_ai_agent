"""Vector store retriever implementation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..core.config import DEFAULT_SEARCH_KWARGS


class StandardVectorStoreRetriever(BaseRetriever, BaseModel):
    """A standard retriever implementation for vector stores.
    
    Implements BaseRetriever for consistent retrieval interface and
    BaseModel for configuration validation.
    """
    vectorstore: Any = Field(description="Vector store instance to use")
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SEARCH_KWARGS.copy()
    )

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Get relevant documents using vector store similarity search."""
        search_kwargs = {**self.search_kwargs, **kwargs}
        docs = self.vectorstore.similarity_search(query, **search_kwargs)
        return docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Async version of document retrieval."""
        search_kwargs = {**self.search_kwargs, **kwargs}
        docs = await self.vectorstore.asimilarity_search(query, **search_kwargs)
        return docs