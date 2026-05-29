from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.config.settings import settings
from backend.schemas.evaluation import AudioEvaluationResponse, TextEvaluationRequest, TextEvaluationResponse
from backend.services.evaluation_service import evaluate_audio_answer, evaluate_text_answer


router = APIRouter(prefix=f"{settings.api_prefix}/evaluations", tags=["evaluations"])


@router.post("/text", response_model=TextEvaluationResponse)
def evaluate_text_response(payload: TextEvaluationRequest) -> TextEvaluationResponse:
    """Evaluate one typed interview answer."""
    try:
        return evaluate_text_answer(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/audio", response_model=AudioEvaluationResponse)
async def evaluate_audio_response(
    question: str = Form(...),
    category: str = Form("General"),
    audio: UploadFile = File(...),
) -> AudioEvaluationResponse:
    """Evaluate one uploaded audio interview answer."""
    try:
        return await evaluate_audio_answer(
            question=question,
            category=category,
            audio_file=audio,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
