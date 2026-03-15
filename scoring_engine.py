import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# EMOTION TO SCORE MAPPING
# Converts emotion label to a numeric score
# ─────────────────────────────────────────────
EMOTION_SCORES = {
    'confident': 90,
    'neutral':   65,
    'nervous':   40,
    'stressed':  30,
    'unknown':   50
}

# ─────────────────────────────────────────────
# SCORING WEIGHTS
# Must add up to 1.0
# ─────────────────────────────────────────────
WEIGHTS = {
    'relevance':   0.35,
    'emotion':     0.25,
    'fluency':     0.20,
    'sentiment':   0.20
}

# ─────────────────────────────────────────────
# CALCULATE FINAL SCORE
# ─────────────────────────────────────────────
def calculate_final_score(relevance_score, emotion_label, fluency_score, sentiment_score):
    emotion_score = EMOTION_SCORES.get(emotion_label, 50)

    final_score = (
        WEIGHTS['relevance']  * relevance_score +
        WEIGHTS['emotion']    * emotion_score   +
        WEIGHTS['fluency']    * fluency_score   +
        WEIGHTS['sentiment']  * sentiment_score
    )

    return round(final_score, 2)

# ─────────────────────────────────────────────
# GET PERFORMANCE GRADE
# ─────────────────────────────────────────────
def get_grade(score):
    if score >= 85:
        return "A", "Excellent"
    elif score >= 70:
        return "B", "Good"
    elif score >= 55:
        return "C", "Average"
    elif score >= 40:
        return "D", "Needs Improvement"
    else:
        return "F", "Poor"

# ─────────────────────────────────────────────
# GENERATE FEEDBACK
# ─────────────────────────────────────────────
def generate_feedback(relevance_score, emotion_label, fluency_score, sentiment_score, final_score):
    feedback = {
        'strengths': [],
        'improvements': [],
        'tips': []
    }

    # Relevance feedback
    if relevance_score >= 70:
        feedback['strengths'].append("✅ Your answer was highly relevant to the question.")
    elif relevance_score >= 50:
        feedback['improvements'].append("⚠️ Your answer was partially relevant. Try to stay more focused on the question.")
    else:
        feedback['improvements'].append("❌ Your answer lacked relevance. Make sure to directly address what was asked.")
        feedback['tips'].append("💡 Tip: Use the STAR method (Situation, Task, Action, Result) to structure answers.")

    # Emotion feedback
    if emotion_label == 'confident':
        feedback['strengths'].append("✅ You sounded confident and composed.")
    elif emotion_label == 'neutral':
        feedback['strengths'].append("✅ You maintained a calm, neutral tone.")
    elif emotion_label == 'nervous':
        feedback['improvements'].append("⚠️ You sounded nervous. Practice deep breathing before answering.")
        feedback['tips'].append("💡 Tip: Pause for 2 seconds before answering to collect your thoughts.")
    elif emotion_label == 'stressed':
        feedback['improvements'].append("❌ You sounded stressed. Try to slow down and speak calmly.")
        feedback['tips'].append("💡 Tip: Practice mock interviews regularly to build confidence.")

    # Fluency feedback
    if fluency_score >= 80:
        feedback['strengths'].append("✅ Excellent fluency — minimal filler words.")
    elif fluency_score >= 60:
        feedback['improvements'].append("⚠️ Some filler words detected. Practice pausing instead of saying 'um' or 'uh'.")
    else:
        feedback['improvements'].append("❌ Too many filler words. Record yourself practicing and listen back.")

    # Sentiment feedback
    if sentiment_score >= 70:
        feedback['strengths'].append("✅ Positive and enthusiastic tone detected.")
    elif sentiment_score < 50:
        feedback['improvements'].append("⚠️ Try to use more positive and energetic language.")

    # Overall tip based on final score
    if final_score >= 85:
        feedback['tips'].append("🌟 Outstanding performance! You're interview-ready.")
    elif final_score >= 70:
        feedback['tips'].append("👍 Good performance. A little more practice and you'll be excellent.")
    elif final_score >= 55:
        feedback['tips'].append("📚 Average performance. Focus on relevance and reducing filler words.")
    else:
        feedback['tips'].append("🎯 Keep practicing! Record yourself and review your answers daily.")

    return feedback

# ─────────────────────────────────────────────
# EMOTION STABILITY INDEX (ESI)
# Measures emotional consistency across questions
# ESI = 1 - standard deviation of emotion scores
# Higher ESI = more consistent emotional state
# ─────────────────────────────────────────────
def calculate_esi(emotion_labels):
    import numpy as np
    scores = [EMOTION_SCORES.get(e, 50) for e in emotion_labels]
    if len(scores) < 2:
        return 1.0
    std = np.std(scores)
    esi = round(1 - (std / 100), 2)
    return max(0, esi)

# ─────────────────────────────────────────────
# SAVE SESSION RESULTS
# Stores results to a JSON file for dashboard
# ─────────────────────────────────────────────
def save_session(session_data, filepath='models/sessions.json'):
    # Load existing sessions
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_sessions = json.load(f)
    else:
        all_sessions = []

    # Add timestamp
    session_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Append new session
    all_sessions.append(session_data)

    # Save back
    with open(filepath, 'w') as f:
        json.dump(all_sessions, f, indent=2)

    print(f"Session saved! Total sessions: {len(all_sessions)}")

# ─────────────────────────────────────────────
# FULL EVALUATION REPORT
# Combines all scores into one report
# ─────────────────────────────────────────────
def generate_report(question, answer, emotion_label, relevance_score, 
                    fluency_score, sentiment_score):

    final_score = calculate_final_score(
        relevance_score, emotion_label, fluency_score, sentiment_score
    )
    grade, grade_label = get_grade(final_score)
    feedback = generate_feedback(
        relevance_score, emotion_label, fluency_score, sentiment_score, final_score
    )

    report = {
        'question': question,
        'answer_preview': answer[:100] + '...' if len(answer) > 100 else answer,
        'scores': {
            'relevance':  relevance_score,
            'emotion':    EMOTION_SCORES.get(emotion_label, 50),
            'fluency':    fluency_score,
            'sentiment':  sentiment_score,
            'final':      final_score
        },
        'emotion_label': emotion_label,
        'grade': grade,
        'grade_label': grade_label,
        'feedback': feedback
    }

    return report

# ─────────────────────────────────────────────
# PRINT REPORT (nice formatted output)
# ─────────────────────────────────────────────
def print_report(report):
    print("\n" + "="*50)
    print("       INTERVIEW EVALUATION REPORT")
    print("="*50)
    print(f"Question: {report['question']}")
    print(f"Answer:   {report['answer_preview']}")
    print("-"*50)
    print(f"Relevance Score:  {report['scores']['relevance']}/100")
    print(f"Emotion Score:    {report['scores']['emotion']}/100  ({report['emotion_label']})")
    print(f"Fluency Score:    {report['scores']['fluency']}/100")
    print(f"Sentiment Score:  {report['scores']['sentiment']}/100")
    print("-"*50)
    print(f"FINAL SCORE:      {report['scores']['final']}/100  |  Grade: {report['grade']} ({report['grade_label']})")
    print("="*50)

    if report['feedback']['strengths']:
        print("\nSTRENGTHS:")
        for s in report['feedback']['strengths']:
            print(f"  {s}")

    if report['feedback']['improvements']:
        print("\nAREAS TO IMPROVE:")
        for i in report['feedback']['improvements']:
            print(f"  {i}")

    if report['feedback']['tips']:
        print("\nTIPS:")
        for t in report['feedback']['tips']:
            print(f"  {t}")
    print("="*50)

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    report = generate_report(
        question="Tell me about yourself",
        answer="I am a computer science student with a passion for AI and machine learning. I have worked on several projects and I am confident in my ability to contribute to a team.",
        emotion_label="confident",
        relevance_score=60.94,
        fluency_score=75,
        sentiment_score=65
    )
    print_report(report)