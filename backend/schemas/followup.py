"""
ARIES — backend/schemas/followup.py
Pydantic models for request/response validation.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


Category = Literal["AI/ML", "Technical", "HR", "Behavioral", "System Design", "DSA", "General"]


class FollowUpRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        json_schema_extra={"example": "Tell me about your project"},
    )
    answer: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        json_schema_extra={"example": "I built an AI interview assessment platform using Whisper and NLP."},
    )
    category: Category = Field(
        ...,
        json_schema_extra={"example": "AI/ML"},
    )

    @field_validator("question", "answer")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank.")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Tell me about your project",
                "answer": "I built an AI interview assessment platform using Whisper and NLP.",
                "category": "AI/ML",
            }
        }
    )


class FollowUpResponse(BaseModel):
    followup_question: str
    category: str
    based_on_keywords: Optional[list[str]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "followup_question": "What challenges did you face while integrating Whisper?",
                "category": "AI/ML",
                "based_on_keywords": ["Whisper", "NLP", "platform"],
            }
        }
    )


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
