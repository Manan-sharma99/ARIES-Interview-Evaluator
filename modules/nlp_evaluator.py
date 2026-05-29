"""
nlp_evaluator.py
Lightweight NLP relevance evaluator — no PyTorch / sentence-transformers required.

Uses TF-IDF vectorisation + cosine similarity (scikit-learn) which is already
installed in the project environment and produces comparable relevance scores.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.questions_bank import IDEAL_ANSWERS

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
    Score answer relevance against the ideal answer (if available) or the
    question itself as a proxy.  Returns a score in [0, 100].
    """
    if not user_answer or len(user_answer.strip()) < 5:
        return 0.0

    ideal = IDEAL_ANSWERS.get(question, question)

    # Primary similarity: answer vs ideal answer
    sim_ideal = _tfidf_similarity(user_answer, ideal)

    # Secondary similarity: answer vs the question (captures on-topic content)
    sim_question = _tfidf_similarity(user_answer, question)

    # Keyword overlap bonus: reward answers that cover key terms from the ideal
    ideal_words = set(ideal.lower().split())
    ans_words   = set(user_answer.lower().split())
    overlap_ratio = len(ideal_words & ans_words) / max(len(ideal_words), 1)

    # Weighted blend — ideal answer is the primary signal
    blended = sim_ideal * 0.55 + sim_question * 0.20 + overlap_ratio * 0.25

    # Scale to [0, 100] with a calibration curve.
    # TF-IDF cosine values are typically 0.02–0.25 for interview answers;
    # we map this range to approximately 30–85 to match human expectation.
    calibrated = min(1.0, blended * 3.2 + 0.20)

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
