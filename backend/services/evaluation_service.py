import os
import tempfile
from pathlib import Path

from fastapi import UploadFile

from backend.schemas.evaluation import (
    AudioEvaluationResponse,
    TextEvaluationRequest,
    TextEvaluationResponse,
)


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}

# Use the new ARIES evaluator in the API path
from modules.aries_evaluator import evaluate_answer as aries_evaluate


def evaluate_text_answer(payload: TextEvaluationRequest) -> TextEvaluationResponse:
    """
    Run the same typed-answer evaluation used by the Streamlit practice screen.

    Text-only evaluation has no audio signal, so emotion defaults to neutral.
    Audio transcription and emotion detection will be migrated as a separate
    endpoint to keep this first slice small and reliable.
    """
    question = payload.question.strip()
    answer = payload.answer.strip()
    category = payload.category.strip() or "General"

    if not answer:
        raise ValueError("Answer is required for evaluation.")

    from modules.fluency_analyzer import analyze_fluency
    from modules.scoring_engine import generate_report

    # Run ARIES evaluator (replaces legacy modules.nlp_evaluator.evaluate_answer)
    aries_result = aries_evaluate(question, answer)

    # Preserve original NLP-style fields for backward compatibility
    nlp = {
        "relevance_score": aries_result.get("relevance_score"),
        "sentiment": aries_result.get("sentiment"),
        "sentiment_score": aries_result.get("sentiment_score"),
        "confidence_score": aries_result.get("confidence_score"),
        "word_count": aries_result.get("word_count"),
    }

    fluency = analyze_fluency(answer)
    emotion = "neutral"
    report = generate_report(
        question=question,
        answer=answer,
        emotion=emotion,
        relevance=nlp["relevance_score"],
        fluency=fluency["fluency_score"],
        sentiment=nlp["sentiment_score"],
        aries_scores=aries_result,
    )

    return TextEvaluationResponse(
        question=question,
        category=category,
        answer=answer,
        emotion=emotion,
        nlp=nlp,
        fluency=fluency,
        report=report,
    )


async def evaluate_audio_answer(
    question: str,
    audio_file: UploadFile,
    category: str = "General",
) -> AudioEvaluationResponse:
    """
    Evaluate one uploaded audio answer.

    The uploaded file is written to a temporary path because Whisper, Librosa,
    and the emotion model all work with file paths. The file is always removed
    after processing, even if transcription or scoring fails.
    """
    question = (question or "").strip()
    category = (category or "").strip() or "General"

    if len(question) < 3:
        raise ValueError("Question must be at least 3 characters long.")
    if not audio_file or not audio_file.filename:
        raise ValueError("Audio file is required.")

    temp_audio_path = await _save_upload_to_temp_file(audio_file)
    try:
        from interview_evaluator.modules.emotion_model_legacy import predict_emotion
        from modules.fluency_analyzer import analyze_fluency
        from modules.scoring_engine import generate_report
        from modules.whisper_stt import transcribe_audio
        transcript = transcribe_audio(temp_audio_path).strip()
        if not transcript:
            raise ValueError("No speech was detected in the uploaded audio.")

        # Use ARIES evaluator for audio transcript
        aries_result = aries_evaluate(question, transcript)

        # Preserve original NLP-style fields for backward compatibility
        nlp = {
            "relevance_score": aries_result.get("relevance_score"),
            "sentiment": aries_result.get("sentiment"),
            "sentiment_score": aries_result.get("sentiment_score"),
            "confidence_score": aries_result.get("confidence_score"),
            "word_count": aries_result.get("word_count"),
        }

        fluency = analyze_fluency(transcript)
        emotion = predict_emotion(temp_audio_path) or "neutral"
        report = generate_report(
            question=question,
            answer=transcript,
            emotion=emotion,
            relevance=nlp["relevance_score"],
            fluency=fluency["fluency_score"],
            sentiment=nlp["sentiment_score"],
            aries_scores=aries_result,
            fluency_details={
                "word_count": nlp.get("word_count"),
            },
        )

        scores = report["scores"]
        return AudioEvaluationResponse(
            question=question,
            category=category,
            transcript=transcript,
            emotion=emotion,
            relevance_score=scores["relevance"],
            fluency_score=scores["fluency"],
            final_score=scores["final"],
            grade=report["grade"],
            nlp=nlp,
            fluency=fluency,
            report=report,
        )
    finally:
        _remove_file_if_exists(temp_audio_path)


async def _save_upload_to_temp_file(audio_file: UploadFile) -> str:
    suffix = Path(audio_file.filename or "").suffix.lower() or ".wav"
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. Use wav, mp3, m4a, webm, ogg, flac, or mp4."
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    try:
        with temp_file:
            while chunk := await audio_file.read(1024 * 1024):
                temp_file.write(chunk)

        if os.path.getsize(temp_path) == 0:
            raise ValueError("Uploaded audio file is empty.")

        return temp_path
    except Exception:
        _remove_file_if_exists(temp_path)
        raise


def _remove_file_if_exists(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
