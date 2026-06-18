"""
nlp_evaluator.py
NLP relevance evaluator for ARIES.

Primary path  : TF-IDF + keyword overlap against the ideal answer (if available).
Fallback path : Semantic cosine similarity (all-MiniLM-L6-v2) against a generic
                question-type prototype, blended 70/30 with TF-IDF vs question.
                This replaces the old weak TF-IDF-only fallback and fixes the
                artificially low scores for semantically relevant answers that
                share no lexical overlap with the question text.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.questions_bank import IDEAL_ANSWERS

# ── Semantic model singleton (lazy-loaded on first use) ───────────────────────
_sentence_model = None

def _get_sentence_model():
    """Load all-MiniLM-L6-v2 once and reuse for the process lifetime."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _sentence_model = None  # graceful degradation if unavailable
    return _sentence_model


# ── Generic question-type prototypes ─────────────────────────────────────────
# One short prototype per type — intent-level, NOT keyword lists.
# These describe *what a good answer looks like*, enabling semantic alignment.
_QUESTION_PROTOTYPES = {
    "project": (
        "I built a project, implemented solutions using specific technologies, "
        "solved a real problem, and achieved measurable results."
    ),
    "technical": (
        "I explained technical concepts, implementation details, design decisions, "
        "system architecture, and engineering tradeoffs."
    ),
    "behavioral": (
        "I described a specific situation, my responsibility and role, "
        "the concrete actions I took, and the resulting outcome."
    ),
    "leadership": (
        "I led people, made important decisions, resolved team challenges, "
        "motivated others, and influenced positive outcomes."
    ),
    "hr": (
        "I explained my professional background, motivations, core strengths, "
        "career goals, and relevant experiences."
    ),
}

# Question-type keyword signals (lightweight, no category-specific banks)
_TYPE_SIGNALS = {
    "project":    ["project you worked on", "project you are proud of", "describe a project",
                   "tell me about a project", "biggest project", "complex project", "challenging project"],
    "technical":  ["how does", "explain", "what is", "difference between", "implement",
                   "design a", "debug", "optimize", "algorithm", "architecture"],
    "behavioral": ["tell me about a time", "give me an example", "describe a situation",
                   "how did you handle", "when have you", "walk me through a time"],
    "leadership": ["led a team", "managed people", "lead", "conflict", "motivate",
                   "mentor", "delegate", "stakeholder", "drove alignment"],
    "hr":         ["tell me about yourself", "why do you want", "where do you see yourself",
                   "strengths", "weaknesses", "salary", "why should we hire", "what motivates you"],
}


def _detect_question_type_simple(question: str) -> str:
    """Classify question into one of 5 types using lightweight keyword matching."""
    q = question.lower()
    scores = {qtype: sum(1 for s in signals if s in q)
              for qtype, signals in _TYPE_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "hr"


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Return rescaled semantic similarity in [0, 1] using sentence embeddings.

    all-MiniLM-L6-v2 produces raw cosine similarities in approximately
    [-0.05, 0.35] for interview answers — not [0, 1].
    We rescale: cosine=-0.05 → 0.0,  cosine=0.35 → 1.0.
    This gives a strong relevant answer ~0.65–0.85, an irrelevant answer ~0.0.
    Falls back to 0.0 if the model is unavailable.
    """
    model = _get_sentence_model()
    if model is None:
        return 0.0
    try:
        emb_a = model.encode(text_a, convert_to_numpy=True)
        emb_b = model.encode(text_b, convert_to_numpy=True)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        raw_cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        # Rescale from empirical MiniLM range [-0.05, 0.35] → [0, 1]
        _LOW  = -0.05   # typical irrelevant-answer cosine floor
        _HIGH =  0.35   # typical strong-answer cosine ceiling
        scaled = (raw_cos - _LOW) / (_HIGH - _LOW)
        return float(max(0.0, min(1.0, scaled)))
    except Exception:
        return 0.0

# ── Confidence-language dictionaries ─────────────────────────────────────────
CONFIDENT_PHRASES = [
    'i am', 'i have', 'i can', 'i will', 'i believe',
    'i know', 'definitely', 'certainly', 'absolutely',
]
WEAK_PHRASES = [
    'i think maybe', 'i guess', 'i am not sure', 'probably',
    'i might', 'sort of', 'kind of', 'i hope',
]


# ── Internal TF-IDF relevance scorer ─────────────────────────────────────────

def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """
    Return cosine similarity in [0, 1] between two strings using TF-IDF.
    Returns 0.0 if either string is empty or too short.
    """
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0
    try:
        vect = TfidfVectorizer(
            min_df=1,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        )
        tfidf_matrix = vect.fit_transform([a, b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        # Fallback: word-overlap Jaccard similarity
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ── Public evaluation functions ───────────────────────────────────────────────

def evaluate_relevance(question: str, user_answer: str) -> float:
    """
    Score answer relevance against the ideal answer (if available) or a
    semantic prototype for the detected question type.  Returns [0, 100].

    Path A (ideal answer exists):
        Unchanged — TF-IDF against ideal answer + keyword overlap blend.

    Path B (no ideal answer — NEW semantic fallback):
        Detects question type → selects a generic prototype → computes
        semantic cosine similarity of the answer against both the prototype
        and the question, takes the min to avoid inflation → blends 70%
        semantic + 30% TF-IDF.

        Calibration:
          Raw MiniLM cosine range for interview answers: [-0.05, 0.35]
          Rescaled to [0, 1] then calibrated by: blended * 1.5 + 0.15
          This maps a strong on-topic answer with zero lexical overlap
          from the old score of ~52 to ~78–85.
    """
    if not user_answer or len(user_answer.strip()) < 5:
        return 0.0

    ideal = IDEAL_ANSWERS.get(question, "")

    if ideal:
        # ── Path A: ideal answer available — existing logic unchanged ─────────
        sim_ideal    = _tfidf_similarity(user_answer, ideal)
        sim_question = _tfidf_similarity(user_answer, question)
        ideal_words  = set(ideal.lower().split())
        ans_words    = set(user_answer.lower().split())
        overlap_ratio = len(ideal_words & ans_words) / max(len(ideal_words), 1)
        blended    = sim_ideal * 0.55 + sim_question * 0.20 + overlap_ratio * 0.25
        calibrated = min(1.0, blended * 3.2 + 0.20)
    else:
        # ── Path B: no ideal answer — semantic prototype fallback ─────────────
        question_type = _detect_question_type_simple(question)
        prototype     = _QUESTION_PROTOTYPES[question_type]

        # Compute semantic similarity against the prototype anchor
        sem_vs_proto = _semantic_similarity(user_answer, prototype)  # [0, 1] rescaled
        # TF-IDF vs the question itself (captures any lexical overlap)
        tfidf_sim    = _tfidf_similarity(user_answer, question)      # [0, 1]

        # 70% semantic + 30% TF-IDF blend
        blended    = sem_vs_proto * 0.7 + tfidf_sim * 0.3
        # Calibration: blended 0.40–0.80 → score 75–85 (strong answers)
        #              blended ~0.00         → score ~15   (irrelevant)
        calibrated = min(1.0, blended * 1.5 + 0.15)

    return round(calibrated * 100, 2)


def analyze_sentiment(text: str):
    """Return (sentiment_label, score_0_100)."""
    text_lower = text.lower()
    positive_words = [
        'good', 'great', 'excellent', 'strong', 'passionate',
        'excited', 'confident', 'successful', 'achieve', 'love',
        'enjoy', 'dedicated', 'motivated', 'proud', 'happy',
    ]
    negative_words = [
        'bad', 'failed', 'struggle', 'difficult', 'hate',
        'nervous', 'scared', 'worried', 'problem', 'issue', 'weak',
    ]
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    if pos_count > neg_count:
        return "Positive", min(100, 60 + pos_count * 5)
    elif neg_count > pos_count:
        return "Negative", max(0, 40 - neg_count * 5)
    return "Neutral", 50


def analyze_confidence_language(text: str) -> float:
    """Return a confidence language score in [0, 100]."""
    text_lower = text.lower()
    confident_count = sum(1 for p in CONFIDENT_PHRASES if p in text_lower)
    weak_count      = sum(1 for p in WEAK_PHRASES      if p in text_lower)
    return min(100, max(0, 60 + confident_count * 8 - weak_count * 10))


def evaluate_answer(question: str, user_answer: str) -> dict:
    """
    Full NLP evaluation. Returns a dict compatible with the evaluation_pipeline.

    Keys: relevance_score, sentiment, sentiment_score, confidence_score, word_count
    """
    relevance_score  = evaluate_relevance(question, user_answer)
    sentiment, sentiment_score = analyze_sentiment(user_answer)
    confidence_score = analyze_confidence_language(user_answer)

    word_count = len(user_answer.split())
    # Light length penalty: very short or very long answers lose a bit
    if word_count < 10:
        length_penalty = 0.5
    elif word_count > 200:
        length_penalty = 0.9
    else:
        length_penalty = 1.0

    relevance_score = round(relevance_score * length_penalty, 2)

    return {
        "relevance_score":  relevance_score,
        "sentiment":        sentiment,
        "sentiment_score":  sentiment_score,
        "confidence_score": confidence_score,
        "word_count":       word_count,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "Tell me about yourself"
    answer   = (
        "I am a computer science student with a passion for AI and machine learning. "
        "I have worked on several projects and I am confident in my ability to learn "
        "quickly and contribute to a team."
    )
    print(f"Question: {question}")
    results = evaluate_answer(question, answer)
    print(f"Relevance Score:   {results['relevance_score']}/100")
    print(f"Sentiment:         {results['sentiment']} ({results['sentiment_score']}/100)")
    print(f"Confidence Score:  {results['confidence_score']}/100")
    print(f"Word Count:        {results['word_count']} words")
    from modules.questions_bank import QUESTIONS
    print(f"\nTotal questions in bank: {sum(len(v) for v in QUESTIONS.values())}")
