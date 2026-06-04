"""
evaluation_pipeline.py
Single entry point for all answer evaluation in ARIES.

Handles:
  - NLP relevance (nlp_evaluator)
  - AI relevance (free_ai_evaluator, lazy-loaded singleton)
  - Blended relevance: NLP*0.6 + AI*0.4 (Task 3)
  - Fluency analysis
  - Emotion detection from audio
  - Confidence score from emotion map (Task 4)
  - Edge cases: empty answer → None; mic fail → neutral; AI fail → NLP only
"""

import os
from datetime import datetime

# ── Emotion → confidence numeric mapping (Task 4) ───────────────────────────
CONFIDENCE_MAP = {
    "confident": 90,
    "neutral":   70,
    "nervous":   50,
    "stressed":  30,
}

# ── Lazy singleton for FreeAIEvaluator (expensive to load) ──────────────────
_ai_evaluator_instance = None

def _get_ai_evaluator():
    """Load FreeAIEvaluator once and cache it for the process lifetime."""
    global _ai_evaluator_instance
    if _ai_evaluator_instance is None:
        from modules.free_ai_evaluator import FreeAIEvaluator
        _ai_evaluator_instance = FreeAIEvaluator()
    return _ai_evaluator_instance


# ── Main pipeline function ───────────────────────────────────────────────────

def run_evaluation(
    question:   str,
    transcript: str,
    audio_path: str  = None,
    duration:   int  = None,
    category:   str  = "General",
) -> dict | None:
    """
    Evaluate one answer through the full pipeline.

    Returns:
        result dict  — ready to store in session / display in UI
        None         — if transcript is empty (edge case, Task 7)
    """
    # ── Edge case: empty answer ──────────────────────────────────────────────
    if not (transcript or "").strip() and audio_path and os.path.exists(audio_path):
        try:
            from modules.whisper_stt import transcribe_audio
            transcript = transcribe_audio(audio_path)
        except Exception:
            transcript = ""

    transcript = (transcript or "").strip()
    if not transcript:
        return None   # caller should show a warning, not crash

    # ── 1. NLP evaluation ────────────────────────────────────────────────────
    from modules.nlp_evaluator import evaluate_answer as nlp_evaluate
    nlp = nlp_evaluate(question, transcript)
    nlp_relevance = float(nlp.get("relevance_score", 0))

    # ── 2. AI evaluation — fallback to NLP-only on any failure ───────────────
    ai_relevance = nlp_relevance   # safe default
    ai_scores    = None
    ai_failed    = False
    try:
        ai_ev     = _get_ai_evaluator()
        ai_result = ai_ev.evaluate_answer(question, transcript, category=category)
        ai_relevance = float(ai_result.get("relevance_score", nlp_relevance))
        detail    = ai_result.get("detailed_scores", {})
        ai_scores = {
            "clarity":   detail.get("sentiment_score",    0),
            "structure": detail.get("structure_quality",  0),
            "depth":     detail.get("answer_completeness",0),
        }
    except Exception:
        # AI unavailable / import error / model error → use NLP score only
        ai_failed    = True
        ai_relevance = nlp_relevance

    # ── 3. Blend relevance scores (Task 3) ───────────────────────────────────
    # If AI failed, weight falls entirely on NLP (0.6+0.4 → 1.0 × NLP)
    if ai_failed:
        final_relevance = nlp_relevance
    else:
        final_relevance = round(nlp_relevance * 0.6 + ai_relevance * 0.4, 2)

    # ── 4. Fluency analysis ──────────────────────────────────────────────────
    from modules.fluency_analyzer import analyze_fluency
    fluency       = analyze_fluency(transcript, duration)
    fluency_score = float(fluency.get("fluency_score", 0))

    # ── 5. Emotion detection — fallback to "neutral" on mic failure ──────────
    emotion = "neutral"
    if audio_path and os.path.exists(audio_path):
        try:
            from interview_evaluator.modules.emotion_model_legacy import predict_emotion
            emotion = predict_emotion(audio_path) or "neutral"
        except Exception:
            emotion = "neutral"   # mic / model failure

    # ── 6. Confidence score from emotion map (Task 4) ────────────────────────
    confidence = CONFIDENCE_MAP.get(emotion, 70)

    # ── 7. Generate scored report ────────────────────────────────────────────
    from modules.scoring_engine import generate_report
    report = generate_report(
        question,
        transcript,
        emotion,
        final_relevance,
        fluency_score,
        float(nlp.get("sentiment_score", 50)),
        ai_scores=ai_scores,
    )

    # ── 8. Build result record ───────────────────────────────────────────────
    return {
        "date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question":         question,
        "category":         category,
        "answer":           transcript,
        "emotion":          emotion,
        "scores":           report["scores"],      # relevance, fluency, confidence, final
        "grade":            report["grade"],
        "grade_label":      report["grade_label"],
        "feedback":         report["feedback"],    # strengths + improvements
        "fluency_feedback": fluency.get("feedback", []),
        "sentiment":        nlp.get("sentiment", "Neutral"),
        "ai_clarity":       report.get("ai_clarity", 0),
        "ai_structure":     report.get("ai_structure", 0),
        "ai_depth":         report.get("ai_depth", 0),
        "ai_failed":        ai_failed,             # flag for UI warning
        "mode":             "interview",
    }
