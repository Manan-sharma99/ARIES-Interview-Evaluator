"""
Enhanced Scoring Engine for ARIES Interview System.

Keeps the original class-based scorer for backward compatibility and exposes
module-level helpers used by the production evaluation pipeline.
"""

from typing import Dict, List, Optional

import numpy as np


class ScoringEngine:
    """
    Legacy class-based scorer kept intact for older call sites.
    """

    def __init__(self):
        self.weights = {
            "relevance": 0.40,
            "emotion": 0.25,
            "fluency": 0.20,
            "communication": 0.15,
        }
        self.emotion_scores = {
            "confident": 90,
            "neutral": 65,
            "nervous": 40,
            "stressed": 30,
        }
        self.grade_boundaries = {
            "A+": 95,
            "A": 85,
            "B+": 75,
            "B": 70,
            "C+": 60,
            "C": 55,
            "D": 40,
            "F": 0,
        }

    def calculate_final_score(
        self,
        ai_relevance_score: float,
        emotion: str,
        filler_count: int,
        answer_length_words: int,
        sentiment_score: Optional[float] = None,
    ) -> Dict:
        relevance_score = max(0, min(100, ai_relevance_score))
        emotion_score = self.emotion_scores.get(emotion.lower(), 50)
        fluency_score = self._calculate_fluency_score(filler_count, answer_length_words)
        communication_score = (
            max(0, min(100, sentiment_score)) if sentiment_score is not None else emotion_score
        )

        final_score = round(
            relevance_score * self.weights["relevance"]
            + emotion_score * self.weights["emotion"]
            + fluency_score * self.weights["fluency"]
            + communication_score * self.weights["communication"],
            1,
        )
        grade = self._get_grade(final_score)
        components = {
            "relevance": round(relevance_score, 1),
            "emotion": round(emotion_score, 1),
            "fluency": round(fluency_score, 1),
            "communication": round(communication_score, 1),
        }
        return {
            "final_score": final_score,
            "grade": grade,
            "components": components,
            "weights": self.weights,
            "emotion_detected": emotion,
            "filler_words": filler_count,
            "assessment": self._get_assessment(final_score, grade),
        }

    def _calculate_fluency_score(self, filler_count: int, answer_length_words: int) -> float:
        if answer_length_words == 0:
            return 0

        filler_ratio = filler_count / answer_length_words
        if filler_ratio <= 0.02:
            score = 100
        elif filler_ratio <= 0.05:
            score = 90 - (filler_ratio - 0.02) * 1000
        elif filler_ratio <= 0.10:
            score = 70 - (filler_ratio - 0.05) * 800
        elif filler_ratio <= 0.15:
            score = 50 - (filler_ratio - 0.10) * 600
        else:
            score = max(20, 50 - (filler_ratio - 0.15) * 400)

        return round(max(0, min(100, score)), 1)

    def _get_grade(self, score: float) -> str:
        for grade, threshold in self.grade_boundaries.items():
            if score >= threshold:
                return grade
        return "F"

    def _get_assessment(self, score: float, grade: str) -> str:
        del grade
        if score >= 85:
            return "Outstanding performance! You demonstrated excellent answer quality, strong confidence, and clear communication."
        if score >= 70:
            return "Good performance. Your answer was relevant and well-delivered with minor areas for improvement."
        if score >= 55:
            return "Average performance. Focus on improving answer depth and reducing filler words."
        if score >= 40:
            return "Below average. Work on answer relevance, confidence, and fluency before your next interview."
        return "Needs significant improvement. Practice structuring answers and speaking more confidently."

    def calculate_session_metrics(self, session_scores: List[Dict]) -> Dict:
        if not session_scores:
            return {}

        final_scores = [s["final_score"] for s in session_scores]
        emotion_scores = [s["components"]["emotion"] for s in session_scores]
        relevance_scores = [s["components"]["relevance"] for s in session_scores]
        avg_final_score = round(np.mean(final_scores), 1)
        avg_emotion_score = round(np.mean(emotion_scores), 1)
        avg_relevance_score = round(np.mean(relevance_scores), 1)
        emotion_std = np.std(emotion_scores)
        esi = round(max(0, 1 - (emotion_std / 100)), 3)
        emotions = [s["emotion_detected"] for s in session_scores]
        emotion_counts = {
            "confident": emotions.count("confident"),
            "neutral": emotions.count("neutral"),
            "nervous": emotions.count("nervous"),
            "stressed": emotions.count("stressed"),
        }
        overall_grade = self._get_grade(avg_final_score)
        return {
            "total_questions": len(session_scores),
            "average_score": avg_final_score,
            "overall_grade": overall_grade,
            "average_relevance": avg_relevance_score,
            "average_emotion": avg_emotion_score,
            "emotion_stability_index": esi,
            "emotion_distribution": emotion_counts,
            "highest_score": max(final_scores),
            "lowest_score": min(final_scores),
            "score_range": round(max(final_scores) - min(final_scores), 1),
            "assessment": self._get_assessment(avg_final_score, overall_grade),
        }

    def compare_sessions(self, session1: Dict, session2: Dict) -> Dict:
        score_change = round(session2["average_score"] - session1["average_score"], 1)
        relevance_change = round(session2["average_relevance"] - session1["average_relevance"], 1)
        esi_change = round(
            session2["emotion_stability_index"] - session1["emotion_stability_index"], 3
        )
        return {
            "score_change": score_change,
            "relevance_change": relevance_change,
            "esi_change": esi_change,
            "improved": score_change > 0,
            "improvement_percentage": round((score_change / session1["average_score"]) * 100, 1)
            if session1["average_score"] > 0
            else 0,
        }


CONFIDENCE_MAP = {
    "confident": 90,
    "neutral": 70,
    "nervous": 50,
    "stressed": 30,
}

_GRADE_THRESHOLDS = [
    (90, "A+", "Outstanding"),
    (80, "A", "Excellent"),
    (70, "B", "Good"),
    (60, "C", "Satisfactory"),
    (50, "D", "Needs Work"),
    (0, "F", "Poor"),
]

QUESTION_TYPE_WEIGHTS = {
    "project": {
        "technical_competency": 0.50,
        "interview_readiness": 0.20,
        "communication": 0.15,
        "confidence": 0.15,
    },
    "technical": {
        "technical_competency": 0.45,
        "interview_readiness": 0.25,
        "communication": 0.15,
        "confidence": 0.15,
    },
    "behavioral": {
        "technical_competency": 0.10,
        "interview_readiness": 0.50,
        "communication": 0.25,
        "confidence": 0.15,
    },
    "leadership": {
        "technical_competency": 0.15,
        "interview_readiness": 0.40,
        "communication": 0.20,
        "confidence": 0.25,
    },
    "hr": {
        "technical_competency": 0.10,
        "interview_readiness": 0.35,
        "communication": 0.35,
        "confidence": 0.20,
    },
}

COMMUNICATION_COMPONENT_WEIGHTS = {
    "fluency_score": 0.40,
    "speech_rate_score": 0.20,
    "filler_score": 0.20,
    "confidence_language_score": 0.20,
}

TECHNICAL_COMPETENCY_COMPONENT_WEIGHTS = {
    "relevance_score": 0.35,
    "technical_depth_score": 0.35,
    "evidence_score": 0.15,
    "impact_score": 0.15,
}

TECHNICAL_COMPETENCY_COMPONENT_WEIGHTS_BY_TYPE = {
    "project": {
        "relevance_score": 0.20,
        "technical_depth_score": 0.45,
        "evidence_score": 0.15,
        "impact_score": 0.20,
    }
}

CONFIDENCE_COMPONENT_WEIGHTS = {
    "ownership_score": 0.40,
    "confidence_score": 0.40,
    "emotion_score": 0.20,
}

INTERVIEW_READINESS_COMPONENT_WEIGHTS = {
    "star_score": 0.30,
    "ownership_score": 0.20,
    "impact_score": 0.20,
    "evidence_score": 0.15,
    "relevance_score": 0.15,
}

INTERVIEW_READINESS_COMPONENT_WEIGHTS_BY_TYPE = {
    "project": {
        "star_score": 0.10,
        "ownership_score": 0.25,
        "impact_score": 0.30,
        "evidence_score": 0.15,
        "relevance_score": 0.20,
    }
}


def calculate_esi(emotions: list) -> float:
    if not emotions:
        return 0.0
    scores = [CONFIDENCE_MAP.get(e, 50) for e in emotions]
    std = float(np.std(scores))
    return round(max(0.0, 1.0 - std / 100.0), 3)


def _round_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


def _score_from_words_per_minute(words_per_minute: float) -> float:
    if words_per_minute <= 0:
        return 70.0
    distance = abs(words_per_minute - 130.0)
    return _round_score(100.0 - min(50.0, distance * 0.75))


def _score_from_filler_count(filler_count: int, word_count: int) -> float:
    if word_count <= 0:
        return 100.0
    filler_ratio = filler_count / max(word_count, 1)
    return _round_score(100.0 - min(70.0, filler_ratio * 400.0))


def _weighted_score(components: dict, weights: dict) -> float:
    total = sum(float(components.get(key, 0)) * weight for key, weight in weights.items())
    return _round_score(total)


def _legacy_feedback(relevance: float, fluency: float, confidence: float, emotion: str) -> dict:
    strengths, improvements = [], []
    if relevance >= 70:
        strengths.append("Answer is relevant to the question asked.")
    else:
        improvements.append("Work on directly addressing the question.")
    if fluency >= 70:
        strengths.append("Good fluency - minimal filler words and clear pace.")
    else:
        improvements.append("Reduce filler words and work on speaking pace.")
    if confidence >= 70:
        strengths.append(f"Vocal confidence detected ({emotion}).")
    else:
        improvements.append(f"Project more confidence (detected: {emotion}).")
    return {"strengths": strengths, "improvements": improvements}


def generate_report(
    question: str,
    answer: str,
    emotion: str,
    relevance: float,
    fluency: float,
    sentiment: float,
    ai_scores: dict = None,
    aries_scores: dict = None,
    fluency_details: dict = None,
) -> dict:
    del question

    fluency_details = fluency_details or {}
    confidence = _round_score(CONFIDENCE_MAP.get(emotion, 70))
    emotion_score = confidence
    report = {
        "emotion": emotion,
        "aries_used": False,
    }

    if aries_scores:
        question_type = (aries_scores.get("question_type") or "hr").lower()
        weights = QUESTION_TYPE_WEIGHTS.get(question_type, QUESTION_TYPE_WEIGHTS["hr"])
        technical_competency_weights = TECHNICAL_COMPETENCY_COMPONENT_WEIGHTS_BY_TYPE.get(
            question_type, TECHNICAL_COMPETENCY_COMPONENT_WEIGHTS
        )
        interview_readiness_weights = INTERVIEW_READINESS_COMPONENT_WEIGHTS_BY_TYPE.get(
            question_type, INTERVIEW_READINESS_COMPONENT_WEIGHTS
        )
        word_count = int(fluency_details.get("word_count") or len((answer or "").split()))
        words_per_minute = float(fluency_details.get("words_per_minute", 0))
        filler_count = int(fluency_details.get("filler_count", 0))
        speech_rate_score = _score_from_words_per_minute(words_per_minute)
        filler_score = _score_from_filler_count(filler_count, word_count)
        confidence_language_score = _round_score(aries_scores.get("confidence_score", confidence))

        component_inputs = {
            "fluency_score": _round_score(fluency),
            "speech_rate_score": speech_rate_score,
            "filler_score": filler_score,
            "confidence_language_score": confidence_language_score,
            "relevance_score": _round_score(relevance),
            "technical_depth_score": _round_score(aries_scores.get("technical_depth_score", 0)),
            "evidence_score": _round_score(aries_scores.get("evidence_score", 0)),
            "impact_score": _round_score(aries_scores.get("impact_score", 0)),
            "ownership_score": _round_score(aries_scores.get("ownership_score", 0)),
            "confidence_score": confidence,
            "emotion_score": emotion_score,
            "star_score": _round_score(aries_scores.get("star_score", 0)),
        }

        communication_score = _weighted_score(component_inputs, COMMUNICATION_COMPONENT_WEIGHTS)
        technical_competency_score = _weighted_score(
            component_inputs, technical_competency_weights
        )
        confidence_category_score = _weighted_score(
            component_inputs, CONFIDENCE_COMPONENT_WEIGHTS
        )
        interview_readiness_score = _weighted_score(
            component_inputs, interview_readiness_weights
        )
        overall_interview_score = _round_score(
            communication_score * weights["communication"]
            + technical_competency_score * weights["technical_competency"]
            + confidence_category_score * weights["confidence"]
            + interview_readiness_score * weights["interview_readiness"]
        )

        report.update(
            {
                "aries_used": True,
                "question_type": question_type,
                "speech_rate_score": speech_rate_score,
                "filler_score": filler_score,
                "confidence_language_score": confidence_language_score,
                "emotion_score": emotion_score,
                "star_score": component_inputs["star_score"],
                "ownership_score": component_inputs["ownership_score"],
                "impact_score": component_inputs["impact_score"],
                "technical_depth_score": component_inputs["technical_depth_score"],
                "evidence_score": component_inputs["evidence_score"],
                "recruiter_feedback": list(aries_scores.get("recruiter_feedback", [])),
                "candidate_feedback": list(aries_scores.get("candidate_feedback", [])),
                "score_weights": weights,
                "score_breakdown": {
                    "communication_components": COMMUNICATION_COMPONENT_WEIGHTS,
                    "technical_competency_components": technical_competency_weights,
                    "confidence_components": CONFIDENCE_COMPONENT_WEIGHTS,
                    "interview_readiness_components": interview_readiness_weights,
                },
                "scores": {
                    "relevance": component_inputs["relevance_score"],
                    "fluency": component_inputs["fluency_score"],
                    "confidence": confidence,
                    "sentiment": _round_score(sentiment),
                    "communication": communication_score,
                    "technical_competency": technical_competency_score,
                    "confidence_category": confidence_category_score,
                    "interview_readiness": interview_readiness_score,
                    "final": overall_interview_score,
                },
                "feedback": {
                    "strengths": list(aries_scores.get("recruiter_feedback", []))[:3],
                    "improvements": list(aries_scores.get("candidate_feedback", []))[:3],
                },
            }
        )
    else:
        final = _round_score(relevance * 0.40 + fluency * 0.30 + confidence * 0.30)
        report.update(
            {
                "question_type": "fallback",
                "speech_rate_score": 0.0,
                "filler_score": 0.0,
                "confidence_language_score": 0.0,
                "emotion_score": emotion_score,
                "star_score": 0.0,
                "ownership_score": 0.0,
                "impact_score": 0.0,
                "technical_depth_score": 0.0,
                "evidence_score": 0.0,
                "recruiter_feedback": [],
                "candidate_feedback": [],
                "score_weights": {},
                "score_breakdown": {},
                "scores": {
                    "relevance": _round_score(relevance),
                    "fluency": _round_score(fluency),
                    "confidence": confidence,
                    "sentiment": _round_score(sentiment),
                    "final": final,
                },
                "feedback": _legacy_feedback(relevance, fluency, confidence, emotion),
            }
        )

    grade, grade_label = "F", "Poor"
    for threshold, grade_code, label in _GRADE_THRESHOLDS:
        if report["scores"]["final"] >= threshold:
            grade, grade_label = grade_code, label
            break

    report["grade"] = grade
    report["grade_label"] = grade_label

    if ai_scores:
        report["ai_clarity"] = round(float(ai_scores.get("clarity", 0)), 1)
        report["ai_structure"] = round(float(ai_scores.get("structure", 0)), 1)
        report["ai_depth"] = round(float(ai_scores.get("depth", 0)), 1)

    return report
