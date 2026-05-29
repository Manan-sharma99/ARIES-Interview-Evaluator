"""
ARIES — backend/routes/followup.py
FastAPI router: POST /api/interview/followup
"""

import logging

from fastapi import APIRouter, HTTPException, status

from backend.schemas.followup import ErrorResponse, FollowUpRequest, FollowUpResponse
from backend.services.followup_service import generate_followup

logger = logging.getLogger("aries.routes.followup")

router = APIRouter(
    prefix="/api/interview",
    tags=["Interview Intelligence"],
)


@router.post(
    "/followup",
    response_model=FollowUpResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate an intelligent follow-up interview question",
    description=(
        "Analyses the candidate's previous answer and generates a single, "
        "context-aware follow-up question tailored to the interview category."
    ),
    responses={
        200: {"model": FollowUpResponse},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_followup_question(req: FollowUpRequest) -> FollowUpResponse:
    """
    **POST /api/interview/followup**

    Generate one intelligent follow-up question from a previous Q&A pair.

    - **question**: The original interview question
    - **answer**: The candidate's response (verbatim or Whisper-transcribed)
    - **category**: `AI/ML` · `Technical` · `HR` · `Behavioral` · `System Design` · `DSA` · `General`
    """
    try:
        return await generate_followup(req)

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    except RuntimeError as exc:
        msg = str(exc)
        logger.error("RuntimeError in /followup: %s", msg)
        if any(token in msg.lower() for token in ("status", "api_key", "llm_provider")):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is temporarily unavailable. Please retry.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate follow-up question.",
        )

    except Exception as exc:
        logger.exception("Unexpected error in /followup: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )
