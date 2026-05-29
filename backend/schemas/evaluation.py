from typing import Any

from pydantic import BaseModel, Field


class TextEvaluationRequest(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=1)
    category: str = Field(default="General", min_length=1)


class TextEvaluationResponse(BaseModel):
    question: str
    category: str
    answer: str
    emotion: str
    nlp: dict[str, Any]
    fluency: dict[str, Any]
    report: dict[str, Any]


class AudioEvaluationResponse(BaseModel):
    question: str
    category: str
    transcript: str
    emotion: str
    relevance_score: float
    fluency_score: float
    final_score: float
    grade: str
    nlp: dict[str, Any]
    fluency: dict[str, Any]
    report: dict[str, Any]
