"""
ARIES — backend/services/followup_service.py
Orchestrates follow-up question generation:
  extract_keywords → build_prompt → call_llm → sanitise → return
"""

import logging
import re

from backend.schemas.followup import FollowUpRequest, FollowUpResponse
from backend.services.llm_provider import call_llm

logger = logging.getLogger("aries.followup_service")

# ---------------------------------------------------------------------------
# Stop-words (no external NLP dependency needed)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "and",
    "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was",
    "are", "were", "it", "this", "that", "so", "do", "did", "have", "had",
    "he", "she", "they", "be", "been", "by", "as", "from", "also", "very",
    "about", "into", "just", "its", "than", "then", "some", "there", "when",
    "which", "who", "what", "how", "can", "could", "would", "should", "will",
    "used", "using", "made", "make", "built", "build", "work", "worked",
}

# ---------------------------------------------------------------------------
# Category-specific interviewer hints injected into the prompt
# ---------------------------------------------------------------------------

_CATEGORY_HINTS: dict[str, str] = {
    "AI/ML": (
        "Focus on model architecture, data pipelines, evaluation metrics, "
        "overfitting/underfitting, training strategies, or deployment challenges."
    ),
    "Technical": (
        "Probe system design decisions, time/space complexity, code quality, "
        "debugging approach, or tool/library choices."
    ),
    "System Design": (
        "Explore scalability, fault tolerance, CAP theorem trade-offs, "
        "database choices, caching strategy, or API design."
    ),
    "DSA": (
        "Dig into algorithmic complexity, edge cases, alternative approaches, "
        "or space-time trade-offs."
    ),
    "HR": (
        "Explore motivation, teamwork, conflict resolution, adaptability, "
        "career goals, or cultural fit."
    ),
    "Behavioral": (
        "Use STAR framing; probe situation context, specific actions taken, "
        "measurable results, and lessons learned."
    ),
    "General": (
        "Ask a thoughtful clarifying question that naturally extends the conversation."
    ),
}


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 6) -> list[str]:
    """
    Lightweight keyword extraction — no spaCy or NLTK required.
    Prioritises proper nouns / acronyms, then frequency.
    Plugs directly into ARIES's existing NLP pipeline if needed later.
    """
    tokens = re.findall(r"[A-Za-z][\w\-]*", text)
    freq: dict[str, int] = {}

    for tok in tokens:
        if tok.lower() in _STOP_WORDS or len(tok) < 3:
            continue
        freq[tok] = freq.get(tok, 0) + 1

    def score(item: tuple[str, int]) -> tuple[int, int]:
        word, count = item
        is_proper = int(word[0].isupper() or word.isupper())
        return (is_proper, count)

    ranked = sorted(freq.items(), key=score, reverse=True)
    return [w for w, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(req: FollowUpRequest, keywords: list[str]) -> str:
    hint   = _CATEGORY_HINTS.get(req.category, _CATEGORY_HINTS["General"])
    kw_str = ", ".join(keywords) if keywords else "not identified"

    # Per-category few-shot examples — one good, one bad — to anchor quality and form
    _EXAMPLES: dict[str, dict[str, str]] = {
        "AI/ML": {
            "good": "How did you validate your emotion recognition model's accuracy beyond the training dataset?",
            "bad":  "Given Whisper's capabilities, how did you?",
        },
        "Technical": {
            "good": "What trade-offs did you consider when choosing FastAPI over Flask for the backend?",
            "bad":  "Given your use of FastAPI, how did it?",
        },
        "System Design": {
            "good": "How would your current architecture handle a tenfold increase in concurrent interview sessions?",
            "bad":  "Given scalability concerns, what about?",
        },
        "DSA": {
            "good": "What is the time complexity of your keyword extraction approach, and did you consider any alternatives?",
            "bad":  "Given the algorithm you used, how did?",
        },
        "HR": {
            "good": "What motivated you to take on the most technically complex part of the project yourself?",
            "bad":  "Given your motivation, how did you?",
        },
        "Behavioral": {
            "good": "What specific action did you take when the integration wasn't working, and what was the outcome?",
            "bad":  "Given that situation, what did?",
        },
        "General": {
            "good": "What would you build differently if you were starting this project from scratch today?",
            "bad":  "Given what you said, how did you?",
        },
    }

    ex = _EXAMPLES.get(req.category, _EXAMPLES["General"])

    return f"""You are a senior {req.category} interviewer. Your job is to ask the single best follow-up question based on what the candidate just said.

---
CANDIDATE CONTEXT
Original question: {req.question}
Candidate's answer: {req.answer}
Key concepts they mentioned: {kw_str}

---
INTERVIEWER FOCUS FOR {req.category.upper()}
{hint}

---
RULES FOR YOUR QUESTION
1. Start directly with a question word: What, How, Why, Which, When, Where, or Who.
   - NEVER start with "Given", "Considering", "Based on", or any dependent clause.
2. The question must be complete and self-contained — a standalone sentence ending in "?".
3. Probe the candidate's reasoning, decisions, or trade-offs — not surface facts.
4. One sentence only. No sub-questions. No "and also..."
5. Do not repeat or paraphrase the original question.

---
EXAMPLES FOR THIS CATEGORY

GOOD (complete, direct, probing):
{ex["good"]}

BAD (fragment, dependent clause, incomplete):
{ex["bad"]}

---
Output the follow-up question and nothing else."""


# ---------------------------------------------------------------------------
# Response sanitisation
# ---------------------------------------------------------------------------


# Question words a well-formed follow-up must start with
_VALID_STARTERS = re.compile(
    r"^(What|How|Why|Which|When|Where|Who|Did|Do|Does|Would|Could|Can|Should|Is|Are|Was|Were)\b",
    re.IGNORECASE,
)

# Dependent-clause openers that signal a fragment — raise, don't repair
_FRAGMENT_OPENERS = re.compile(
    r"^(Given|Considering|Based on|Since|As|Although|While|Despite|In light of)",
    re.IGNORECASE,
)

# A complete question must have at least one verb after the opening word
_HAS_PREDICATE = re.compile(
    r"\b(did|do|does|would|could|can|should|is|are|was|were|have|has|had"
    r"|use|build|choose|design|handle|approach|decide|implement|integrate"
    r"|face|measure|validate|test|deploy|scale|manage|work|make|find)\b",
    re.IGNORECASE,
)


def _sanitise(raw: str) -> str:
    """
    Validates and lightly cleans the model output.
    REJECTS fragments — does not repair them.
    Raises ValueError so generate_followup can surface the error cleanly.
    """
    q = raw.strip().strip('"').strip("'").strip()
    q = re.sub(r"^\d+[\.\)]\s*", "", q)  # strip accidental "1. " prefix

    # Reject dependent-clause openers (the original bug pattern)
    if _FRAGMENT_OPENERS.search(q):
        raise ValueError(
            f"Model returned a dependent-clause fragment: {q!r}. "
            "This indicates a token-budget or prompt issue."
        )

    # Must start with a recognised question word
    if not _VALID_STARTERS.match(q):
        raise ValueError(
            f"Model output does not start with a question word: {q!r}"
        )

    # Must contain at least one verb (basic predicate check)
    if not _HAS_PREDICATE.search(q):
        raise ValueError(
            f"Model output appears to lack a predicate: {q!r}"
        )

    # Ensure it ends with exactly one question mark
    q = q.rstrip(".!?") + "?"

    # Sanity-check final length
    if len(q) < 20:
        raise ValueError(f"Question is suspiciously short after sanitisation: {q!r}")

    return q


# ---------------------------------------------------------------------------
# Public service function
# ---------------------------------------------------------------------------

async def generate_followup(req: FollowUpRequest) -> FollowUpResponse:
    """
    Full pipeline:
      1. Extract keywords from the candidate's answer
      2. Build a category-aware prompt
      3. Call the configured LLM (Anthropic or Gemini via llm_provider)
      4. Sanitise and validate the output
      5. Return FollowUpResponse
    """
    logger.info(
        "generate_followup | category=%s | q_len=%d | a_len=%d",
        req.category, len(req.question), len(req.answer),
    )

    keywords         = extract_keywords(req.answer)
    prompt           = _build_prompt(req, keywords)
    raw              = await call_llm(prompt)
    followup_question = _sanitise(raw)

    if not followup_question or len(followup_question) < 10:
        raise RuntimeError("Generated follow-up question was too short or empty.")

    logger.info("Follow-up generated: %s", followup_question)

    return FollowUpResponse(
        followup_question=followup_question,
        category=req.category,
        based_on_keywords=keywords or None,
    )
