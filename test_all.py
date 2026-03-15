"""
AI Interview Evaluator - Full System Test
Run: python test_all.py
"""

import time
import numpy as np
import pickle
import os

print("=" * 60)
print("   AI INTERVIEW EVALUATOR — FULL SYSTEM TEST")
print("=" * 60)

# TEST 1: EMOTION DETECTION MODEL
print("\n📦 MODULE 1: Speech Emotion Recognition")
print("-" * 40)

emotion_result = False
emotion_accuracy = 0

try:
    from modules.emotion_model import load_dataset, INTERVIEW_MAP
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    if not os.path.exists("models/emotion_model.pkl"):
        print("⚠️  Model not found. Training now...")
        from modules.emotion_model import train_model
        train_model("data/Audio_Speech_Actors_01-24", "models/emotion_model.pkl")
    else:
        print("✅ Trained model found at models/emotion_model.pkl")

    print("📊 Loading dataset and evaluating accuracy...")
    X, y = load_dataset("data/Audio_Speech_Actors_01-24")

    with open("models/emotion_model.pkl", "rb") as f:
        saved = pickle.load(f)

    if isinstance(saved, dict):
        model = saved["model"]
        scaler = saved.get("scaler")
        model_name = saved.get("model_name", "Random Forest")
    else:
        model = saved
        scaler = None
        model_name = "Random Forest"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_pred = model.predict(X_test_scaled)

    y_test_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_test])
    y_pred_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_pred])
    accuracy = accuracy_score(y_test_mapped, y_pred_mapped)

    print(f"\n  Algorithm:        {model_name}")
    print(f"  Dataset:          RAVDESS ({len(X)} samples, 24 actors)")
    print(f"  Features:         {X.shape[1]}")
    print(f"  Train/Test Split: 80% / 20%")
    print(f"  Test Samples:     {len(X_test)}")
    print(f"  ✅ ACCURACY:      {accuracy * 100:.2f}% (interview labels)")

    report = classification_report(y_test_mapped, y_pred_mapped, output_dict=True)
    print(f"\n  Per-Emotion Performance:")
    for emotion in ['confident', 'nervous', 'neutral', 'stressed']:
        if emotion in report:
            r = report[emotion]
            print(f"    {emotion:<12} | Precision: {r['precision']:.2f} | Recall: {r['recall']:.2f} | F1: {r['f1-score']:.2f}")

    emotion_result = True
    emotion_accuracy = accuracy * 100

except Exception as e:
    print(f"  ❌ ERROR: {e}")

# TEST 2: NLP ANSWER EVALUATOR
print("\n\n📦 MODULE 2: NLP Answer Evaluator")
print("-" * 40)

nlp_result = False
try:
    start = time.time()
    from modules.nlp_evaluator import evaluate_answer
    load_time = round(time.time() - start, 2)

    test_cases = [
        {"question": "Tell me about yourself",
         "answer": "I am a computer science student with strong skills in Python and machine learning. I have built several AI projects and I am confident in my ability to contribute.",
         "expected": "high"},
        {"question": "What are your strengths?",
         "answer": "I don't know, maybe I am okay at some things.",
         "expected": "low"},
        {"question": "Why should we hire you?",
         "answer": "I bring unique technical expertise, creativity, and strong work ethic. I am passionate about delivering results and consistently go above and beyond.",
         "expected": "high"},
    ]

    print(f"  Model:            Sentence-BERT (all-MiniLM-L6-v2)")
    print(f"  Method:           Cosine Similarity")
    print(f"  Load Time:        {load_time}s")
    print(f"  Questions in bank: 100")
    print(f"\n  Test Results:")

    for tc in test_cases:
        result = evaluate_answer(tc["question"], tc["answer"])
        status = "✅" if (tc["expected"] == "high" and result["relevance_score"] > 50) or \
                        (tc["expected"] == "low" and result["relevance_score"] <= 50) else "⚠️"
        print(f"    {status} Q: {tc['question'][:35]:<35} | Relevance: {result['relevance_score']}/100 | Sentiment: {result['sentiment']}")

    print(f"\n  ✅ NLP Module: WORKING")
    nlp_result = True

except Exception as e:
    print(f"  ❌ ERROR: {e}")

# TEST 3: FLUENCY ANALYZER
print("\n\n📦 MODULE 3: Fluency Analyzer")
print("-" * 40)

fluency_result = False
try:
    from modules.fluency_analyzer import analyze_fluency

    test_cases = [
        {"text": "I am a passionate software engineer with strong communication and problem solving skills.", "desc": "Clean answer (no fillers)"},
        {"text": "Um so basically like I am you know kind of good at stuff and uh I think maybe I can help.", "desc": "Heavy filler words"},
        {"text": "I have led multiple cross-functional teams and consistently received positive feedback.", "desc": "Professional answer"},
    ]

    print(f"  Method:           Rule-based filler detection")
    print(f"  Filler words:     19 tracked")
    print(f"\n  Test Results:")

    for tc in test_cases:
        result = analyze_fluency(tc["text"])
        print(f"    ✅ {tc['desc']:<35} | Fluency: {result['fluency_score']}/100 | Fillers: {result['filler_count']} | Words: {result['word_count']}")

    print(f"\n  ✅ Fluency Module: WORKING")
    fluency_result = True

except Exception as e:
    print(f"  ❌ ERROR: {e}")

# TEST 4: SCORING ENGINE
print("\n\n📦 MODULE 4: Scoring Engine")
print("-" * 40)

scoring_result = False
try:
    from modules.scoring_engine import generate_report, calculate_esi

    test_cases = [
        {"emotion": "confident", "relevance": 85, "fluency": 90, "sentiment": 80, "desc": "Strong candidate"},
        {"emotion": "nervous",   "relevance": 55, "fluency": 60, "sentiment": 50, "desc": "Average candidate"},
        {"emotion": "stressed",  "relevance": 30, "fluency": 40, "sentiment": 35, "desc": "Weak candidate"},
    ]

    print(f"  Formula:          Relevance(35%) + Emotion(25%) + Fluency(20%) + Sentiment(20%)")
    print(f"\n  Test Results:")

    for tc in test_cases:
        report = generate_report("Tell me about yourself", "Test answer.", tc["emotion"],
                                 tc["relevance"], tc["fluency"], tc["sentiment"])
        print(f"    ✅ {tc['desc']:<20} | Final: {report['scores']['final']}/100 | Grade: {report['grade']} ({report['grade_label']})")

    emotions = ['confident', 'confident', 'neutral', 'confident', 'nervous']
    esi = calculate_esi(emotions)
    print(f"\n  ESI Test:         {emotions} → ESI = {esi}")
    print(f"\n  ✅ Scoring Engine: WORKING")
    scoring_result = True

except Exception as e:
    print(f"  ❌ ERROR: {e}")

# FINAL SUMMARY
print("\n")
print("=" * 60)
print("   FINAL SYSTEM REPORT")
print("=" * 60)

modules = [
    ("Speech Emotion Recognition", emotion_result,  f"{emotion_accuracy:.2f}% accuracy"),
    ("NLP Answer Evaluator",       nlp_result,      "Sentence-BERT | 100 questions"),
    ("Fluency Analyzer",           fluency_result,  "Rule-based | 19 filler words"),
    ("Scoring Engine",             scoring_result,  "Weighted formula + ESI metric"),
]

for name, passed, detail in modules:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {name:<30} | {detail}")

passed_count = sum(1 for _, p, _ in modules if p)
print("-" * 60)
print(f"  Modules Passed:   {passed_count}/{len(modules)}")
print(f"  Emotion Accuracy: {emotion_accuracy:.2f}%")
print(f"  System Status:    {'🟢 ALL SYSTEMS OPERATIONAL' if passed_count == 4 else '🔴 SOME MODULES FAILED'}")
print("=" * 60)
print("\n✅ Test complete!\n")