import re

# ─────────────────────────────────────────────
# FILLER WORDS LIST
# ─────────────────────────────────────────────
FILLER_WORDS = [
    'um', 'uh', 'umm', 'uhh', 'like', 'you know', 'basically', 
    'actually', 'literally', 'so', 'right', 'okay', 'well', 
    'hmm', 'er', 'err', 'kind of', 'sort of', 'i mean'
]

# ─────────────────────────────────────────────
# COUNT FILLER WORDS
# ─────────────────────────────────────────────
def count_fillers(text):
    text_lower = text.lower()
    filler_count = 0
    found_fillers = []

    for filler in FILLER_WORDS:
        # Count occurrences of each filler word
        count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        if count > 0:
            filler_count += count
            found_fillers.append(f"{filler} ({count}x)")

    return filler_count, found_fillers

# ─────────────────────────────────────────────
# CALCULATE SPEECH RATE
# Based on word count and estimated speaking time
# Assumes average speaking rate to estimate duration
# ─────────────────────────────────────────────
def calculate_speech_rate(text, duration_seconds=None):
    words = text.split()
    word_count = len(words)

    if duration_seconds and duration_seconds > 0:
        # If we have actual duration from recording
        words_per_minute = (word_count / duration_seconds) * 60
    else:
        # Estimate: average person speaks ~130 words per minute
        # We estimate duration from word count
        estimated_duration = word_count / 130 * 60
        words_per_minute = 130  # default assumption

    return round(words_per_minute, 1), word_count

# ─────────────────────────────────────────────
# CALCULATE FLUENCY SCORE
# Based on filler words and speech rate
# ─────────────────────────────────────────────
def calculate_fluency_score(word_count, filler_count, words_per_minute=130):
    # Start with perfect score
    score = 100

    # Penalty for filler words
    # Each filler word = -5 points, max penalty = -40
    filler_penalty = min(40, filler_count * 5)
    score -= filler_penalty

    # Penalty for too few words (too short answer)
    if word_count < 20:
        score -= 20
    elif word_count < 40:
        score -= 10

    # Penalty for speaking too fast or too slow
    if words_per_minute > 180:
        score -= 15  # too fast
    elif words_per_minute < 80:
        score -= 10  # too slow

    # Bonus for good length (50-150 words is ideal)
    if 50 <= word_count <= 150:
        score += 5

    return max(0, min(100, round(score, 2)))

# ─────────────────────────────────────────────
# GET FLUENCY FEEDBACK
# Human-readable feedback based on scores
# ─────────────────────────────────────────────
def get_fluency_feedback(fluency_score, filler_count, word_count, found_fillers):
    feedback = []

    if filler_count == 0:
        feedback.append("✅ Excellent! No filler words detected.")
    elif filler_count <= 2:
        feedback.append(f"⚠️ Minor: {filler_count} filler word(s) detected: {', '.join(found_fillers)}")
    else:
        feedback.append(f"❌ Too many filler words ({filler_count}): {', '.join(found_fillers)}")
        feedback.append("💡 Tip: Practice pausing silently instead of using filler words.")

    if word_count < 20:
        feedback.append("❌ Answer too short. Try to elaborate more.")
    elif word_count < 40:
        feedback.append("⚠️ Answer is brief. Consider adding more detail.")
    elif word_count > 200:
        feedback.append("⚠️ Answer is very long. Try to be more concise.")
    else:
        feedback.append("✅ Good answer length.")

    if fluency_score >= 80:
        feedback.append("🌟 Overall fluency is excellent!")
    elif fluency_score >= 60:
        feedback.append("👍 Fluency is good, minor improvements needed.")
    else:
        feedback.append("📚 Practice speaking more smoothly and confidently.")

    return feedback

# ─────────────────────────────────────────────
# FULL FLUENCY ANALYSIS
# Main function called by other modules
# ─────────────────────────────────────────────
def analyze_fluency(text, duration_seconds=None):
    # Count fillers
    filler_count, found_fillers = count_fillers(text)

    # Speech rate
    words_per_minute, word_count = calculate_speech_rate(text, duration_seconds)

    # Fluency score
    fluency_score = calculate_fluency_score(word_count, filler_count, words_per_minute)

    # Feedback
    feedback = get_fluency_feedback(fluency_score, filler_count, word_count, found_fillers)

    return {
        'fluency_score': fluency_score,
        'filler_count': filler_count,
        'found_fillers': found_fillers,
        'word_count': word_count,
        'words_per_minute': words_per_minute,
        'feedback': feedback
    }

# ─────────────────────────────────────────────
# TEST (when script is run directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test with a sample answer containing filler words
    test_answer = """Um, so basically I am a computer science student and like 
    I have worked on several projects. You know, I am really passionate about 
    AI and machine learning. I think I can contribute well to any team."""

    print("=== FLUENCY ANALYSIS TEST ===\n")
    print(f"Answer: {test_answer.strip()}\n")

    results = analyze_fluency(test_answer)

    print(f"Fluency Score:     {results['fluency_score']}/100")
    print(f"Filler Words:      {results['filler_count']} found")
    if results['found_fillers']:
        print(f"Fillers Detected:  {', '.join(results['found_fillers'])}")
    print(f"Word Count:        {results['word_count']} words")
    print(f"Speech Rate:       {results['words_per_minute']} WPM (estimated)")
    print(f"\nFeedback:")
    for fb in results['feedback']:
        print(f"  {fb}")
        