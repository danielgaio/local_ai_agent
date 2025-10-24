"""Data models and schemas for the motorcycle recommendation system."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MotorcycleReview(BaseModel):
    """A motorcycle review with metadata."""
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    comment: Optional[str] = None
    text: Optional[str] = None
    price_usd_estimate: Optional[int] = None
    price_est: Optional[int] = None  # Alias for price_usd_estimate
    engine_cc: Optional[int] = None
    suspension_notes: Optional[str] = None
    ride_type: Optional[str] = None
    source: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get full text content combining comment and text fields."""
        text_fields = []
        if self.comment:
            text_fields.append(self.comment)
        if self.text:
            text_fields.append(self.text)
        return " ".join(text_fields) if text_fields else ""


class MotorcyclePick(BaseModel):
    """A motorcycle recommendation pick."""
    brand: str
    model: str
    year: int
    price_est: float
    reason: str = Field(..., description="Short reason (<=12 words) for recommendation")
    evidence: str = Field(..., description="Evidence from reviews or 'none in dataset'")
    evidence_source: Optional[str] = None


class ClarifyingQuestion(BaseModel):
    """A clarifying question from the LLM."""
    type: str = "clarify"
    question: str


class Recommendation(BaseModel):
    """A motorcycle recommendation response."""
    type: str = "recommendation"
    primary: Optional[MotorcyclePick] = None
    alternatives: List[MotorcyclePick] = Field(default_factory=list)
    note: Optional[str] = None


LLMResponse = Union[ClarifyingQuestion, Recommendation]


class ValidationError(BaseModel):
    """Validation error information."""
    reason: str
    action: str = Field(..., description="One of: 'reject', 'retry'")
    attribute: Optional[str] = None  # For attribute presence validation


class ConversationContext(BaseModel):
    """Full conversation context for the LLM."""
    history: List[str] = Field(default_factory=list)
    top_reviews: List[MotorcycleReview] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)