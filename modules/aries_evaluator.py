"""
aries_evaluator.py
ARIES — Advanced Recruiter Interview Evaluation System
=======================================================
Transforms raw answer scoring into a recruiter-style interview quality analysis.

Preserves all original outputs (relevance, sentiment, confidence, word_count)
and adds a full interview quality layer on top.

NO external APIs. Fully deterministic. Explainable logic throughout.
"""

import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Try to import ideal answers; fall back to empty dict ─────────────────────
try:
    from modules.questions_bank import IDEAL_ANSWERS
except ImportError:
    IDEAL_ANSWERS = {}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LEXICONS & SIGNAL BANKS
# ═════════════════════════════════════════════════════════════════════════════

CONFIDENT_PHRASES = [
    'i am', 'i have', 'i can', 'i will', 'i believe',
    'i know', 'definitely', 'certainly', 'absolutely',
]
WEAK_PHRASES = [
    'i think maybe', 'i guess', 'i am not sure', 'probably',
    'i might', 'sort of', 'kind of', 'i hope',
]

# ── STAR framework signals ────────────────────────────────────────────────────
STAR_SIGNALS = {
    "situation": [
        "when i was", "in my previous", "at my last", "during my",
        "while working", "in that project", "the context was", "the situation was",
        "i was working at", "we were facing", "the team was", "at the time",
        "in my role at", "the background was", "we had a situation", "back then",
    ],
    "task": [
        "my responsibility was", "i was tasked with", "i needed to",
        "my job was to", "i was assigned", "the goal was", "my objective was",
        "i had to", "the task required", "it was my role to", "i was expected to",
        "my challenge was",
    ],
    "action": [
        "i decided to", "i implemented", "i developed", "i created",
        "i led", "i collaborated", "i built", "i designed", "i proposed",
        "i initiated", "i resolved", "i worked with", "i took the initiative",
        "i approached", "i focused on", "i used", "i applied", "i set up",
        "i introduced", "i coordinated",
    ],
    "result": [
        "as a result", "this resulted in", "we achieved", "i improved",
        "the outcome was", "this led to", "we reduced", "we increased",
        "successfully", "this helped", "the impact was", "we saved",
        "by doing this", "in the end", "ultimately", "the team delivered",
        "i was able to", "we exceeded", "this allowed",
    ],
}

# ── Ownership signals ─────────────────────────────────────────────────────────
OWNERSHIP_FIRST_PERSON = [
    r"\bi led\b", r"\bi built\b", r"\bi designed\b", r"\bi owned\b",
    r"\bi drove\b", r"\bi initiated\b", r"\bi delivered\b", r"\bi managed\b",
    r"\bi created\b", r"\bi was responsible\b", r"\bi took ownership\b",
    r"\bi spearheaded\b", r"\bi championed\b", r"\bi executed\b",
]
DIFFUSE_OWNERSHIP = [
    r"\bwe did\b", r"\bthe team\b", r"\beveryone\b", r"\bsomebody\b",
    r"\bthey handled\b", r"\bit was handled\b", r"\bwas done\b",
]

# ── Impact signals ────────────────────────────────────────────────────────────
QUANTIFIED_IMPACT = [
    r"\d+\s*%", r"\$\d+", r"\d+x\b", r"\d+\s*(hours?|days?|weeks?|months?)\b",
    r"(increased|reduced|improved|saved|grew|cut|boosted)\s+by\s+\d+",
    r"\d+\s*(users?|customers?|clients?|requests?|tickets?)",
    r"(million|billion|thousand)\b",
]
QUALITATIVE_IMPACT = [
    "improved", "reduced", "increased", "faster", "better",
    "significant", "notable", "major", "key improvement",
    "positive feedback", "recognized", "promoted", "awarded",
]

# ── Technical depth signals (language-agnostic) ───────────────────────────────
TECH_TERMS = [
    # Architecture & patterns
    "microservices", "api", "apis", "rest", "graphql", "oauth", "jwt",
    "cache", "caching", "parallel processing", "concurrency", "async",
    "thread", "latency", "throughput", "queue", "pipeline", "service",
    "services", "endpoint", "architecture",
    # Data / storage
    "database", "sql", "nosql", "redis", "postgresql", "mongodb", "kafka",
    # Methodology
    "algorithm", "complexity", "big o", "optimization", "refactor",
    "design pattern", "solid", "dry", "tdd", "unit test", "integration test",
    "feature engineering", "feature extraction", "model training",
    # DevOps / infra
    "docker", "kubernetes", "ci/cd", "deployment", "cloud",
    "aws", "azure", "gcp", "terraform", "monitoring", "logging",
    # ML / data
    "model", "training", "inference", "dataset", "neural", "xgboost",
    "tensorflow", "pytorch", "transformers", "bert", "sentence-bert",
    "whisper", "streamlit", "classification", "precision", "recall",
    "speaker-aware", "emotion recognition",
]

SPECIFIC_TECH_TERMS = [
    "fastapi", "docker", "redis", "postgresql", "mongodb", "xgboost",
    "tensorflow", "pytorch", "transformers", "bert", "sentence-bert",
    "whisper", "streamlit", "kubernetes", "aws", "azure", "gcp",
    "microservices", "caching", "parallel processing", "feature engineering",
    "feature extraction", "model training", "speaker-aware",
    "emotion recognition",
]

TECH_IMPLEMENTATION_SIGNALS = [
    "implemented", "integrated", "built", "designed", "deployed", "optimized",
    "refactored", "containerized", "trained", "fine-tuned", "cached",
    "parallelized", "orchestrated", "exposed", "scaled", "migrated",
    "served", "instrumented", "set up", "configured",
]

ARCHITECTURE_SIGNALS = [
    "architecture", "service", "services", "microservice", "microservices",
    "api", "apis", "endpoint", "pipeline", "workflow", "deployment",
    "caching layer", "cache layer", "queue", "orchestration",
]

TECH_OUTCOME_SIGNALS = [
    "latency", "throughput", "feature extraction", "response time",
    "training time", "inference time", "cache hit", "parallel processing",
    "scalability", "uptime", "speaker-aware", "emotion recognition",
]

# ── Evidence signals ──────────────────────────────────────────────────────────
EVIDENCE_PHRASES = [
    "for example", "for instance", "specifically", "to illustrate",
    "in one case", "as an example", "such as", "like when", "once when",
    "in that particular", "a concrete example", "in practice",
]

# ── Question-type keyword banks ───────────────────────────────────────────────
QUESTION_TYPE_SIGNALS = {
    "behavioral": [
        "tell me about a time", "give me an example", "describe a situation",
        "how did you handle", "when have you", "can you share an experience",
        "walk me through a time", "have you ever", "what did you do when",
    ],
    "technical": [
        "how does", "explain", "what is", "difference between", "implement",
        "design a", "write a", "debug", "optimize", "what happens when",
        "how would you build", "describe the architecture", "code", "algorithm",
    ],
    "leadership": [
        "led a team", "managed people", "lead", "conflict", "motivate",
        "mentor", "delegate", "influence without authority", "stakeholder",
        "drove alignment", "cross-functional",
    ],
    "project": [
        "project you worked on", "project you are proud of", "describe a project",
        "tell me about a project", "biggest project", "complex project",
        "challenging project", "shipped", "delivered",
    ],
    "hr": [
        "tell me about yourself", "why do you want", "where do you see yourself",
        "strengths and weaknesses", "salary", "why should we hire", "notice period",
        "what motivates you", "why are you leaving", "what are your goals",
        "culture fit", "work style", "greatest achievement", "outside of work",
    ],
}

# ── Scoring weights per question type ────────────────────────────────────────
# Each tuple: (star, ownership, impact, technical_depth, evidence, relevance, confidence)
QUESTION_WEIGHTS = {
    "behavioral":  (0.25, 0.20, 0.20, 0.05, 0.15, 0.10, 0.05),
    "technical":   (0.10, 0.10, 0.15, 0.35, 0.15, 0.10, 0.05),
    "leadership":  (0.20, 0.25, 0.20, 0.05, 0.15, 0.10, 0.05),
    "project":     (0.10, 0.20, 0.20, 0.25, 0.15, 0.05, 0.05),
    "hr":          (0.05, 0.15, 0.10, 0.05, 0.10, 0.30, 0.25),
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ORIGINAL CORE METRICS (preserved unchanged in public API)
# ═════════════════════════════════════════════════════════════════════════════

def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity via TF-IDF bigrams. Returns [0, 1]."""
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0
    try:
        vect = TfidfVectorizer(min_df=1, stop_words="english", ngram_range=(1, 2))
        mat = vect.fit_transform([a, b])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def evaluate_relevance(question: str, user_answer: str) -> float:
    """
    Returns relevance score in [0, 100].

    Path A (ideal answer exists): TF-IDF + keyword overlap — unchanged.
    Path B (no ideal answer):     Semantic prototype fallback (70% semantic
                                  cosine similarity + 30% TF-IDF vs question).
                                  Shared helpers from nlp_evaluator ensure the
                                  sentence model singleton is loaded only once.
    """
    if not user_answer or len(user_answer.strip()) < 5:
        return 0.0
    ideal = IDEAL_ANSWERS.get(question, "")
    if ideal:
        # ── Path A: ideal answer available — unchanged ────────────────────────
        sim_ideal    = _tfidf_similarity(user_answer, ideal)
        sim_question = _tfidf_similarity(user_answer, question)
        ideal_words  = set(ideal.lower().split())
        ans_words    = set(user_answer.lower().split())
        overlap      = len(ideal_words & ans_words) / max(len(ideal_words), 1)
        blended = sim_ideal * 0.55 + sim_question * 0.20 + overlap * 0.25
        calibrated = min(1.0, blended * 3.2 + 0.20)
    else:
        # ── Path B: no ideal answer — semantic prototype fallback ─────────────
        # Reuse helpers from nlp_evaluator so the sentence model is a shared
        # singleton (loaded once per process, not twice).
        from modules.nlp_evaluator import (
            _detect_question_type_simple,
            _QUESTION_PROTOTYPES,
            _semantic_similarity,
        )
        question_type = _detect_question_type_simple(question)
        prototype     = _QUESTION_PROTOTYPES[question_type]
        sem_sim       = _semantic_similarity(user_answer, prototype)   # [0, 1]
        tfidf_sim     = _tfidf_similarity(user_answer, question)       # [0, 1]
        blended       = sem_sim * 0.7 + tfidf_sim * 0.3
        # Calibration matches nlp_evaluator: blended 0.40–0.80 → score 75–85 (strong)
        #                                    blended ~0.00       → score ~15 (irrelevant)
        calibrated    = min(1.0, blended * 1.5 + 0.15)
    return round(calibrated * 100, 2)


def analyze_sentiment(text: str):
    """Returns (label, score_0_100). Handles simple negation."""
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
    negation_window = r"\b(not|never|no|didn't|wasn't|isn't|aren't|couldn't)\b.{0,20}"

    def negated(word):
        return bool(re.search(negation_window + word, text_lower))

    pos_count = sum(1 for w in positive_words if w in text_lower and not negated(w))
    neg_count = sum(1 for w in negative_words if w in text_lower and not negated(w))

    if pos_count > neg_count:
        return "Positive", min(100, 60 + pos_count * 5)
    elif neg_count > pos_count:
        return "Negative", max(0, 40 - neg_count * 5)
    return "Neutral", 50


def analyze_confidence_language(text: str) -> float:
    """Confidence language score in [0, 100]. Handles negation on confident phrases."""
    text_lower = text.lower()
    negation_pattern = r"\b(not|never|no|don't|didn't|can't|won't|isn't)\b.{0,15}"

    def negated(phrase):
        idx = text_lower.find(phrase)
        if idx == -1:
            return False
        surrounding = text_lower[max(0, idx - 25):idx]
        return bool(re.search(r"\b(not|never|no|don't|didn't|can't)\b", surrounding))

    confident_count = sum(
        1 for p in CONFIDENT_PHRASES if p in text_lower and not negated(p)
    )
    weak_count = sum(1 for p in WEAK_PHRASES if p in text_lower)
    return min(100, max(0, 60 + confident_count * 8 - weak_count * 10))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NEW INTERVIEW QUALITY ANALYSIS LAYER
# ═════════════════════════════════════════════════════════════════════════════

# ── 3.1  Question-Type Detection ─────────────────────────────────────────────

def detect_question_type(question: str) -> str:
    """
    Classify question into: behavioral | technical | leadership | project | hr.
    Uses signal-bank keyword matching with a simple voting mechanism.
    Falls back to 'hr' (most generic) if nothing matches.
    """
    q = question.lower()
    scores = {qtype: 0 for qtype in QUESTION_TYPE_SIGNALS}
    for qtype, signals in QUESTION_TYPE_SIGNALS.items():
        for signal in signals:
            if signal in q:
                scores[qtype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "hr"


# ── 3.2  STAR Framework Analysis ─────────────────────────────────────────────

def analyze_star(text: str) -> dict:
    """
    Score each STAR component as present/partial/absent.
    Returns individual component scores and overall star_score in [0, 100].
    """
    text_lower = text.lower()
    component_scores = {}
    component_present = {}

    for component, signals in STAR_SIGNALS.items():
        hits = sum(1 for s in signals if s in text_lower)
        if hits >= 2:
            component_scores[component] = 1.0
            component_present[component] = "strong"
        elif hits == 1:
            component_scores[component] = 0.5
            component_present[component] = "weak"
        else:
            component_scores[component] = 0.0
            component_present[component] = "absent"

    raw = sum(component_scores.values()) / 4.0  # normalise over 4 components
    star_score = round(raw * 100, 2)
    return {
        "star_score": star_score,
        "star_components": component_present,
    }


# ── 3.3  Ownership Analysis ───────────────────────────────────────────────────

def analyze_ownership(text: str) -> float:
    """
    Score ownership clarity in [0, 100].
    Rewards first-person ownership language; penalises vague collective language.
    """
    text_lower = text.lower()
    strong_hits = sum(
        1 for p in OWNERSHIP_FIRST_PERSON if re.search(p, text_lower)
    )
    diffuse_hits = sum(
        1 for p in DIFFUSE_OWNERSHIP if re.search(p, text_lower)
    )
    base = 50
    score = base + strong_hits * 10 - diffuse_hits * 8
    return round(min(100, max(0, score)), 2)


# ── 3.4  Impact Analysis ──────────────────────────────────────────────────────

def analyze_impact(text: str) -> float:
    """
    Score impact articulation in [0, 100].
    Quantified metrics > qualitative claims.
    """
    text_lower = text.lower()
    quant_hits = sum(
        1 for p in QUANTIFIED_IMPACT if re.search(p, text_lower)
    )
    qual_hits = sum(1 for w in QUALITATIVE_IMPACT if w in text_lower)

    # Quantified evidence is worth 2× qualitative
    raw = quant_hits * 2 + qual_hits
    # Map: 0 hits → 20, 1 quant hit → ~55, 2+ quant hits → 80+
    score = 20 + math.log1p(raw) * 28
    return round(min(100, score), 2)


# ── 3.5  Technical Depth Analysis ────────────────────────────────────────────

def analyze_technical_depth(text: str) -> float:
    """
    Score technical specificity in [0, 100].
    Named tools help, but only when paired with implementation detail.
    """
    text_lower = text.lower()
    generic_terms = {term for term in TECH_TERMS if _has_phrase(text_lower, term)}
    specific_terms = {term for term in SPECIFIC_TECH_TERMS if _has_phrase(text_lower, term)}
    implementation_hits = sum(
        1 for signal in TECH_IMPLEMENTATION_SIGNALS if _has_phrase(text_lower, signal)
    )
    architecture_hits = sum(
        1 for signal in ARCHITECTURE_SIGNALS if _has_phrase(text_lower, signal)
    )
    outcome_hits = sum(1 for signal in TECH_OUTCOME_SIGNALS if _has_phrase(text_lower, signal))
    quantified_hits = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|ms|s|sec|seconds?|mins?|minutes?|hours?|x)\b",
            text_lower,
        )
    )

    specificity_score = len(generic_terms) * 2 + len(specific_terms) * 7
    implementation_score = min(20, implementation_hits * 6)
    architecture_score = min(12, architecture_hits * 4)
    evidence_bonus = 0
    if generic_terms or specific_terms:
        evidence_bonus += min(8, quantified_hits * 4)
    if (specific_terms or len(generic_terms) >= 2) and implementation_hits > 0:
        evidence_bonus += min(10, outcome_hits * 3)

    score = specificity_score + implementation_score + architecture_score + evidence_bonus

    if not specific_terms and implementation_hits == 0:
        score = min(score, 38)
    elif not specific_terms and architecture_hits == 0:
        score = min(score, 50)
    elif implementation_hits == 0 and architecture_hits == 0:
        score = min(score, 48)

    return round(min(100, max(0, score)), 2)


# ── 3.6  Evidence Analysis ────────────────────────────────────────────────────

def analyze_evidence(text: str) -> float:
    """
    Score use of concrete examples in [0, 100].
    Checks for explicit evidence markers AND presence of numbers/specifics.
    """
    text_lower = text.lower()
    marker_hits = sum(1 for p in EVIDENCE_PHRASES if p in text_lower)
    # Also reward presence of numbers (dates, percentages, counts) as implicit evidence
    number_count = len(re.findall(r"\b\d+\b", text))
    raw = marker_hits * 15 + min(number_count, 5) * 8
    return round(min(100, 20 + raw), 2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WEIGHTED INTERVIEW QUALITY SCORE
# ═════════════════════════════════════════════════════════════════════════════

def compute_interview_quality_score(
    question_type: str,
    star_score: float,
    ownership_score: float,
    impact_score: float,
    technical_depth_score: float,
    evidence_score: float,
    relevance_score: float,
    confidence_score: float,
) -> float:
    """
    Blend all sub-scores using question-type-aware weights.
    Returns interview_quality_score in [0, 100].
    """
    w = QUESTION_WEIGHTS.get(question_type, QUESTION_WEIGHTS["hr"])
    w_star, w_own, w_imp, w_tech, w_ev, w_rel, w_conf = w

    score = (
        star_score            * w_star +
        ownership_score       * w_own  +
        impact_score          * w_imp  +
        technical_depth_score * w_tech +
        evidence_score        * w_ev   +
        relevance_score       * w_rel  +
        confidence_score      * w_conf
    )
    return round(score, 2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FEEDBACK GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _score_band(score: float) -> str:
    if score >= 80:
        return "strong"
    elif score >= 60:
        return "adequate"
    elif score >= 40:
        return "weak"
    return "missing"


def generate_recruiter_feedback(
    question_type: str,
    star_data: dict,
    ownership_score: float,
    impact_score: float,
    technical_depth_score: float,
    evidence_score: float,
    confidence_score: float,
    relevance_score: float,
    sentiment: str,
    interview_quality_score: float,
    word_count: int,
) -> list[str]:
    """
    Generate recruiter-perspective feedback answering:
      - Would I interview this candidate further?
      - What are the strengths?
      - What are the risks?
    """
    feedback = []
    star_score = star_data["star_score"]
    components = star_data["star_components"]

    # ── Overall hiring signal ─────────────────────────────────────────────────
    if interview_quality_score >= 78:
        feedback.append(
            "✅ ADVANCE: Strong answer. Candidate demonstrates clear structure, "
            "ownership and impact. Worth progressing to next interview stage."
        )
    elif interview_quality_score >= 58:
        feedback.append(
            "⚠️ CONSIDER: Answer shows potential but has notable gaps. "
            "Probe deeper in a follow-up question before deciding."
        )
    else:
        feedback.append(
            "❌ HOLD: Answer lacks sufficient depth, structure, or evidence. "
            "Unlikely to pass a rigorous panel without significant improvement."
        )

    # ── STAR structure assessment ─────────────────────────────────────────────
    if star_score >= 75:
        feedback.append(
            f"STRENGTH (Structure): Answer follows a clear STAR structure. "
            f"All four components detected — easy to evaluate."
        )
    else:
        missing = [k for k, v in components.items() if v == "absent"]
        weak    = [k for k, v in components.items() if v == "weak"]
        if missing:
            feedback.append(
                f"RISK (Structure): Missing STAR components: {', '.join(missing).upper()}. "
                "Interviewer must probe manually — increases evaluation time."
            )
        if weak:
            feedback.append(
                f"RISK (Structure): Weak signals for: {', '.join(weak).upper()}. "
                "Candidate hinted at these but didn't develop them."
            )

    # ── Ownership ─────────────────────────────────────────────────────────────
    if ownership_score >= 70:
        feedback.append(
            "STRENGTH (Ownership): Candidate takes clear personal responsibility. "
            "First-person language indicates individual contributor mindset."
        )
    elif ownership_score < 50:
        feedback.append(
            "RISK (Ownership): Answer uses vague collective language ('we', 'the team'). "
            "Hard to assess individual contribution — classic red flag in senior roles."
        )

    # ── Impact ────────────────────────────────────────────────────────────────
    if impact_score >= 70:
        feedback.append(
            "STRENGTH (Impact): Answer includes quantified or clearly stated impact. "
            "Candidate connects their actions to business outcomes."
        )
    elif impact_score < 45:
        feedback.append(
            "RISK (Impact): No measurable outcomes mentioned. Candidate describes "
            "activity without demonstrating results — weak signal for high-performance culture."
        )

    # ── Technical depth (only flag for technical/project questions) ───────────
    if question_type in ("technical", "project"):
        if technical_depth_score >= 65:
            feedback.append(
                "STRENGTH (Technical): Strong technical vocabulary and specificity. "
                "Candidate appears comfortable discussing implementation details."
            )
        elif technical_depth_score < 40:
            feedback.append(
                "RISK (Technical): Low technical depth for a technical/project question. "
                "Consider whether candidate can articulate implementation decisions under pressure."
            )

    # ── Evidence ──────────────────────────────────────────────────────────────
    if evidence_score >= 65:
        feedback.append(
            "STRENGTH (Evidence): Answer is grounded in concrete examples. "
            "Candidate doesn't rely on generalities."
        )
    elif evidence_score < 40:
        feedback.append(
            "RISK (Evidence): No specific examples cited. Answer stays at an abstract "
            "level — difficult to verify claims in a reference check."
        )

    # ── Confidence & tone ─────────────────────────────────────────────────────
    if confidence_score < 50:
        feedback.append(
            "RISK (Confidence): Candidate uses hedging language ("
            "'I guess', 'maybe', 'I'm not sure'). May signal lack of self-belief "
            "or insufficient preparation."
        )
    if sentiment == "Negative":
        feedback.append(
            "RISK (Tone): Negative sentiment detected. Watch for signs of "
            "bitterness about previous roles — could indicate culture fit issues."
        )

    # ── Answer length ─────────────────────────────────────────────────────────
    if word_count < 30:
        feedback.append(
            "RISK (Length): Answer is too brief (<30 words). Candidate may struggle "
            "to communicate at the depth required for this role."
        )
    elif word_count > 300:
        feedback.append(
            "NOTE (Length): Answer is verbose (>300 words). Candidate may have "
            "difficulty being concise. Check for rambling in live interview."
        )

    return feedback


def generate_candidate_feedback(
    question_type: str,
    star_data: dict,
    ownership_score: float,
    impact_score: float,
    technical_depth_score: float,
    evidence_score: float,
    confidence_score: float,
    relevance_score: float,
    word_count: int,
) -> list[str]:
    """
    Generate candidate-perspective coaching answering:
      - What should I improve?
      - What information was missing?
      - How can I make this answer stronger?
    """
    feedback = []
    star_score = star_data["star_score"]
    components = star_data["star_components"]

    # ── STAR coaching ─────────────────────────────────────────────────────────
    absent_components = [k for k, v in components.items() if v == "absent"]
    weak_components   = [k for k, v in components.items() if v == "weak"]

    if absent_components:
        component_tips = {
            "situation": (
                "📍 Missing: SITUATION — Start by setting the scene. Tell the interviewer "
                "where you were, what your role was, and what the context looked like. "
                "Example opener: 'In my previous role at [Company], we were facing...'"
            ),
            "task": (
                "🎯 Missing: TASK — Clarify your specific responsibility. What were YOU "
                "personally responsible for? Use: 'My role was to...' or 'I was tasked with...'"
            ),
            "action": (
                "⚡ Missing: ACTION — This is the most important part. List 2-3 specific "
                "actions YOU took. Use 'I decided to...', 'I implemented...', 'I led...' "
                "Avoid passive voice and vague statements."
            ),
            "result": (
                "📈 Missing: RESULT — Always close with outcomes. What changed because of you? "
                "Numbers are gold: 'reduced load time by 40%', 'saved the team 5 hours/week'. "
                "If you have no numbers, describe qualitative impact: 'The team's morale improved significantly'."
            ),
        }
        for comp in absent_components:
            feedback.append(component_tips[comp])

    if weak_components and not absent_components:
        feedback.append(
            f"🔧 Strengthen these STAR components: {', '.join(weak_components).upper()}. "
            "You hinted at them but didn't develop them. Add 1-2 more specific sentences for each."
        )

    if star_score >= 75:
        feedback.append(
            "✅ STAR Structure: Strong. Your answer has a clear narrative. "
            "Keep using this framework consistently."
        )

    # ── Ownership coaching ────────────────────────────────────────────────────
    if ownership_score < 55:
        feedback.append(
            "👤 Ownership: Replace 'we did' and 'the team handled' with 'I did' and "
            "'I was responsible for'. Interviewers want to know YOUR contribution, "
            "not the team's. You can acknowledge teamwork briefly but stay first-person."
        )
    elif ownership_score >= 75:
        feedback.append(
            "✅ Ownership: You clearly articulate your individual contribution. Strong."
        )

    # ── Impact coaching ───────────────────────────────────────────────────────
    if impact_score < 50:
        feedback.append(
            "📊 Impact: Your answer doesn't mention any results. Before the interview, "
            "prepare 2-3 metrics from each significant experience: percentages, time saved, "
            "revenue generated, team size, project scale. If you don't have numbers, "
            "describe the qualitative difference your work made."
        )
    elif impact_score < 70:
        feedback.append(
            "📊 Impact: You mention some results but could be more specific. "
            "Convert vague claims ('significantly improved') into measurable ones "
            "('reduced processing time from 8s to 1.2s')."
        )

    # ── Technical coaching (only for technical/project) ───────────────────────
    if question_type in ("technical", "project") and technical_depth_score < 55:
        feedback.append(
            "🔬 Technical Depth: For technical/project questions, name the specific "
            "technologies, patterns, or approaches you used. Don't just say 'I used a "
            "database' — say 'I used PostgreSQL with a read-replica setup to handle "
            "the query load'. Specificity signals expertise."
        )

    # ── Evidence coaching ─────────────────────────────────────────────────────
    if evidence_score < 50:
        feedback.append(
            "🧾 Evidence: Use concrete examples. Phrases like 'for example', "
            "'specifically', or 'in one case' signal to the interviewer that your claims "
            "are real, not fabricated. Every claim needs at least one supporting data point."
        )

    # ── Confidence coaching ───────────────────────────────────────────────────
    if confidence_score < 55:
        feedback.append(
            "💪 Confidence Language: Reduce hedging words like 'I think maybe', "
            "'I guess', 'sort of'. Replace with: 'I believe', 'I know', 'I have done'. "
            "Confident language has a measurable impact on how interviewers perceive competence."
        )

    # ── Length coaching ───────────────────────────────────────────────────────
    if word_count < 40:
        feedback.append(
            "📝 Answer Length: Your answer is too short. Aim for 100-200 words for "
            "behavioural questions. Use the STAR framework to expand — each component "
            "should be 1-3 sentences."
        )
    elif word_count > 280:
        feedback.append(
            "✂️ Answer Length: Your answer is very long. Try to stay under 200 words. "
            "Practice giving the 'headline result' first, then the supporting context. "
            "Interviewers lose focus after 90 seconds."
        )

    # ── Relevance coaching ────────────────────────────────────────────────────
    if relevance_score < 45:
        feedback.append(
            "🎯 Relevance: Your answer may have drifted off-topic. Re-read the question "
            "and make sure every sentence directly answers what was asked. "
            "Start your answer by acknowledging the question: 'Great question — in my "
            "experience with [relevant topic]...'"
        )

    return feedback


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — UNIFIED EVALUATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_answer(question: str, user_answer: str) -> dict:
    """
    Full ARIES evaluation. Returns a single dict with:
    ── Original metrics (preserved) ────────────────────────────────────────
      relevance_score, sentiment, sentiment_score, confidence_score, word_count
    ── New interview quality layer ──────────────────────────────────────────
      interview_quality_score, question_type,
      star_score, ownership_score, impact_score,
      technical_depth_score, evidence_score,
      star_components,
      recruiter_feedback, candidate_feedback
    """

    # ── Guard: empty answer ───────────────────────────────────────────────────
    if not user_answer or len(user_answer.strip()) < 3:
        return {
            "relevance_score": 0.0,
            "sentiment": "Neutral",
            "sentiment_score": 50,
            "confidence_score": 50.0,
            "word_count": 0,
            "interview_quality_score": 0.0,
            "question_type": detect_question_type(question),
            "star_score": 0.0,
            "star_components": {k: "absent" for k in STAR_SIGNALS},
            "ownership_score": 0.0,
            "impact_score": 0.0,
            "technical_depth_score": 0.0,
            "evidence_score": 0.0,
            "recruiter_feedback": ["❌ HOLD: No answer provided."],
            "candidate_feedback": ["Please provide an answer to evaluate."],
        }

    # ── 1. Original metrics ───────────────────────────────────────────────────
    relevance_score              = evaluate_relevance(question, user_answer)
    sentiment, sentiment_score   = analyze_sentiment(user_answer)
    confidence_score             = analyze_confidence_language(user_answer)
    word_count                   = len(user_answer.split())

    # Light length penalty (preserved from original)
    length_penalty = 0.5 if word_count < 10 else (0.9 if word_count > 200 else 1.0)
    relevance_score = round(relevance_score * length_penalty, 2)

    # ── 2. Question-type detection ────────────────────────────────────────────
    question_type = detect_question_type(question)

    # ── 3. New analysis layer ─────────────────────────────────────────────────
    star_data             = analyze_star(user_answer)
    ownership_score       = analyze_ownership(user_answer)
    impact_score          = analyze_impact(user_answer)
    technical_depth_score = analyze_technical_depth(user_answer)
    evidence_score        = analyze_evidence(user_answer)

    # ── 4. Weighted interview quality score ───────────────────────────────────
    interview_quality_score = compute_interview_quality_score(
        question_type,
        star_data["star_score"],
        ownership_score,
        impact_score,
        technical_depth_score,
        evidence_score,
        relevance_score,
        confidence_score,
    )

    # ── 5. Feedback generation ────────────────────────────────────────────────
    recruiter_feedback = generate_recruiter_feedback(
        question_type, star_data, ownership_score, impact_score,
        technical_depth_score, evidence_score, confidence_score,
        relevance_score, sentiment, interview_quality_score, word_count,
    )
    candidate_feedback = generate_candidate_feedback(
        question_type, star_data, ownership_score, impact_score,
        technical_depth_score, evidence_score, confidence_score,
        relevance_score, word_count,
    )

    # ── 6. Assemble output ────────────────────────────────────────────────────
    return {
        # Original metrics (preserved)
        "relevance_score":        relevance_score,
        "sentiment":              sentiment,
        "sentiment_score":        sentiment_score,
        "confidence_score":       confidence_score,
        "word_count":             word_count,

        # New interview quality layer
        "interview_quality_score": interview_quality_score,
        "question_type":           question_type,
        "star_score":              star_data["star_score"],
        "star_components":         star_data["star_components"],
        "ownership_score":         ownership_score,
        "impact_score":            impact_score,
        "technical_depth_score":   technical_depth_score,
        "evidence_score":          evidence_score,
        "recruiter_feedback":      recruiter_feedback,
        "candidate_feedback":      candidate_feedback,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STANDALONE TEST
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json

    test_cases = [
        {
            "label": "Behavioral — Strong STAR",
            "question": "Tell me about a time you resolved a conflict within a team.",
            "answer": (
                "In my previous role at a fintech startup, I was working on the payments team "
                "when two senior engineers had a persistent disagreement about the database "
                "architecture for a new feature. My responsibility was to keep the sprint "
                "on track while also making sure the technical decision was sound. "
                "I decided to schedule a structured technical review session where each "
                "engineer presented their approach with concrete trade-offs. I created a "
                "scoring rubric covering performance, maintainability, and migration cost. "
                "As a result, we reached consensus within one session, shipped the feature "
                "two weeks ahead of schedule, and the solution reduced query latency by 35%."
            ),
        },
        {
            "label": "HR — Weak ownership",
            "question": "Tell me about yourself.",
            "answer": (
                "We worked on a lot of projects. The team built some good software. "
                "I think maybe I am good at coding. We sort of delivered most things on time."
            ),
        },
        {
            "label": "Technical — With depth",
            "question": "How would you design a rate limiting system for an API?",
            "answer": (
                "I would implement a token bucket algorithm backed by Redis. "
                "Each client key stores a counter and a TTL. For distributed systems "
                "the main challenge is atomic increments — I would use Redis INCR with "
                "a Lua script to ensure atomicity. I have built this before at scale: "
                "the system handled 50,000 requests per second with p99 latency under 2ms. "
                "For burst traffic, I would configure a sliding window counter to be "
                "more forgiving than a fixed window, which avoids the boundary-condition "
                "spike problem. I would expose the remaining quota in response headers "
                "so clients can self-throttle."
            ),
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*70}")
        print(f"  {tc['label']}")
        print(f"  Q: {tc['question']}")
        print(f"{'='*70}")
        result = evaluate_answer(tc["question"], tc["answer"])
        # Pretty-print scores
        print(f"  Interview Quality Score : {result['interview_quality_score']}/100")
        print(f"  Question Type Detected  : {result['question_type']}")
        print(f"  STAR Score              : {result['star_score']}/100  {result['star_components']}")
        print(f"  Ownership Score         : {result['ownership_score']}/100")
        print(f"  Impact Score            : {result['impact_score']}/100")
        print(f"  Technical Depth         : {result['technical_depth_score']}/100")
        print(f"  Evidence Score          : {result['evidence_score']}/100")
        print(f"  Relevance Score         : {result['relevance_score']}/100")
        print(f"  Confidence Score        : {result['confidence_score']}/100")
        print(f"  Sentiment               : {result['sentiment']} ({result['sentiment_score']}/100)")
        print(f"  Word Count              : {result['word_count']}")
        print(f"\n  ── RECRUITER FEEDBACK ──────────────────────────────────")
        for line in result["recruiter_feedback"]:
            print(f"  {line}")
        print(f"\n  ── CANDIDATE FEEDBACK ──────────────────────────────────")
        for line in result["candidate_feedback"]:
            print(f"  {line}")
