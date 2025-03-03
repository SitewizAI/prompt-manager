"""Pydantic models for validating prompt data structures."""

from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, RootModel

class QuestionObject(BaseModel):
    """Model for a single evaluation question object."""
    question: str = Field(..., description="The question to evaluate")
    output: List[str] = Field(..., description="Fields from the output to evaluate")
    reference: List[str] = Field(default_factory=list, description="Reference fields for comparison")
    confidence_threshold: float = Field(..., description="Threshold for this question to pass", ge=0.0, le=1.0)
    feedback: str = Field(..., description="Feedback to provide if the question evaluation fails")
    
    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

# Use RootModel instead of __root__ field for Pydantic v2 compatibility
class QuestionsArray(RootModel):
    """Model for an array of evaluation questions."""
    root: List[QuestionObject] = Field(..., description="Array of evaluation questions")
    
    def __iter__(self):
        return iter(self.root)
    
    def __getitem__(self, item):
        return self.root[item]
    
    def __len__(self):
        return len(self.root)
