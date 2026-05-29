# ARIES — Master Context File
## Automated Real-time Interview Evaluation & Integrity System

---

## 1. Project Goal

Build an AI-powered interview practice and evaluation system for students preparing for job and internship interviews. The system evaluates answers across four dimensions simultaneously: speech emotion, answer relevance, fluency, and communication quality. It includes a student interface for practice and a separate interviewer interface for assessment. An anti-cheat module monitors eye gaze during skill tests.

---

## 2. Project Details

- **Student:** Manan Sharma
- **Year:** 2nd Year Engineering
- **OS:** Windows
- **Editor:** VS Code
- **Python:** 3.13 (global install, NO venv)
- **Project Path:** `C:\Users\Manan Sharma\OneDrive\Documents\ai interview project\interview_evaluator\`
- **Run Rule:** Always cd into interview_evaluator before running anything. Never activate venv.

---

## 3. Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| UI Framework | Streamlit |
| ML Models | Random Forest, SVM, MLP, XGBoost |
| NLP | Sentence-BERT (all-MiniLM-L6-v2) |
| Audio Features | Librosa |
| Speech-to-Text | SpeechRecognition (Google API) |
| AI Feedback | Anthropic Claude API (claude-sonnet-4-20250514) |
| Visualization | Matplotlib, Seaborn |
| PDF Export | FPDF2 |
| Data | Pandas, NumPy |
| Audio Recording | SoundDevice, SoundFile |
| Anti-Cheat (planned) | MediaPipe, OpenCV |
| Face Emotion (planned) | DeepFace |
| Speech-to-Text v2 (planned) | Whisper AI |

---

## 4. Folder Structure

```
interview_evaluator/
├── app.py                        ← Main Streamlit app (Student + Interviewer UI)
├── test_all.py                   ← Full system test (4/4 modules)
├── run.bat                       ← Quick launch script
├── data/
│   ├── Audio_Speech_Actors_01-24/ ← RAVDESS dataset (1440 samples, 24 actors)
│   └── CREMA-D/
│       └── AudioWAV/             ← CREMA-D dataset (7442 samples)
├── models/
│   ├── emotion_model.pkl         ← Trained ML model (auto-selects best algorithm)
│   ├── sessions.json             ← All student session history
│   └── confusion_matrix.png      ← Model evaluation chart
└── modules/
    ├── emotion_model.py          ← Speech emotion recognition
    ├── nlp_evaluator.py          ← Answer relevance scoring
    ├── fluency_analyzer.py       ← Filler word detection
    ├── scoring_engine.py         ← Final score calculator
    ├── questions_bank.py         ← All 275 questions + ideal answers
    └── ai_evaluator.py           ← Claude API answer feedback
```

---

## 5. Datasets

| Dataset | Samples | Emotions | Size | Status |
|---|---|---|---|---|
| RAVDESS | 1440 | 8 (acted) | 215MB | Done |
| CREMA-D | 7442 | 6 (acted) | 900MB | Downloaded, retraining pending |
| Combined | 8882 | 8 mapped to 4 | 1.1GB | Training pending |

---

## 6. Emotion Mapping (KEY DECISION)

Train on 8 original emotions, map to 4 interview labels AFTER prediction:
```
happy + surprised  → confident
calm + neutral     → neutral
fearful            → nervous
sad + angry + disgust → stressed
```
This approach improved accuracy from 72.92% (old) to 96.18% (RAVDESS only).
Combined RAVDESS + CREMA-D training is pending.

---

## 7. Audio Features (101 total)

| Feature | Count | What it captures |
|---|---|---|
| MFCC Mean | 40 | Average tonal quality |
| MFCC Std | 40 | Tonal variation over time |
| Chroma | 12 | Harmonic content |
| Mel Spectrogram Mean+Std | 2 | Energy across frequencies |
| Zero Crossing Rate Mean+Std | 2 | Voice sharpness |
| RMS Energy Mean+Std | 2 | Loudness |
| Spectral Centroid | 1 | Voice brightness |
| Spectral Rolloff | 1 | High frequency content |
| Pitch | 1 | Fundamental frequency |
| **Total** | **101** | |

---

## 8. Scoring Formula

```
Final Score = Relevance(35%) + Emotion(25%) + Fluency(20%) + Sentiment(20%)
```

Emotion scores: confident=90, neutral=65, nervous=40, stressed=30

Grades: A(85+), B(70+), C(55+), D(40+), F(below 40)

Custom metric: Emotion Stability Index (ESI) = 1 - std(emotion_scores)/100

---

## 9. Question Bank

| Category | Questions |
|---|---|
| HR & General | 35 |
| Behavioral | 20 |
| Amazon Leadership Principles | 25 |
| Google | 20 |
| Microsoft | 20 |
| Meta (Facebook) | 20 |
| Apple | 20 |
| CS Fundamentals | 20 |
| System Design | 20 |
| Machine Learning | 20 |
| Data Science | 20 |
| Campus Placement | 20 |
| Situational | 15 |
| **Total** | **275** |

---

## 10. ML Model Status

| Algorithm | RAVDESS Only | RAVDESS+CREMA-D |
|---|---|---|
| Random Forest (300 trees) | 75.35% mapped | Pending |
| SVM (RBF kernel) | Not tested | 61.96% raw |
| MLP Neural Network | Fixed for Python 3.13 | Pending |
| XGBoost | Not tested | Pending |
| **Best so far** | **96.18% (RF, RAVDESS)** | **Pending** |

Note: 96.18% is on RAVDESS interview labels. Combined dataset result pending.

---

## 11. APIs Used

| API | Purpose | Status |
|---|---|---|
| Anthropic Claude API | AI-powered answer feedback | Key not set up yet |
| Google Speech Recognition | Voice to text transcription | Working |
| HuggingFace (Sentence-BERT) | NLP embeddings | Working |

### Claude API Setup
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```
Get key from: https://console.anthropic.com
Model used: claude-sonnet-4-20250514
Max tokens: 1000

---

## 12. Key Decisions Made

1. **No venv** — broken on this machine, using global Python 3.13
2. **Post-prediction mapping** — train on 8 emotions, map to 4 after prediction (improves accuracy)
3. **101 features** — upgraded from 42 features (old model)
4. **Multi-algorithm training** — train RF + SVM + MLP + XGBoost, auto-select best
5. **CREMA-D added** — 7442 more samples combined with RAVDESS
6. **Claude API for feedback** — replaces generic feedback with specific AI evaluation
7. **275 questions** — expanded from 100, includes real company questions
8. **ESI metric** — custom original contribution measuring emotional stability

---

## 13. App Features

### Student Interface
- Name-based login with session persistence
- Practice Mode: choose category + question, type or record, instant feedback
- Skill Test Mode: 5 random questions, final grade + bar chart
- Progress Dashboard: score over time, ESI progress bar, skill breakdown

### Interviewer Interface
- All Students Overview: sortable table + bar chart
- Student Detail: per-session expandable view
- Export PDF: generates downloadable report via FPDF2

---

## 14. Current Accuracy

| Module | Accuracy | Method |
|---|---|---|
| Speech Emotion | 96.18% (RAVDESS) | Random Forest, 101 features |
| NLP Evaluator | Cosine similarity | Sentence-BERT |
| Fluency Analyzer | Rule-based | 19 filler words |
| Scoring Engine | Formula-based | Weighted average |

---

## 15. Pending Work

### Immediate
- [ ] Retrain with CREMA-D + MLP fix (run python modules\emotion_model.py)
- [ ] Set up Anthropic API key
- [ ] Place all new files (questions_bank.py, ai_evaluator.py, updated app.py, updated nlp_evaluator.py)
- [ ] Test mic recording end to end
- [ ] Test PDF export

### Short Term (Phase 2)
- [ ] Facial expression analysis (DeepFace + OpenCV)
- [ ] Eye tracking anti-cheat (MediaPipe)
- [ ] CNN deep learning model for comparison
- [ ] Hybrid model (RF + SVM ensemble)
- [ ] Whisper AI for offline speech-to-text

### Medium Term (Phase 3)
- [ ] Mock interview simulation mode
- [ ] Leaderboard between students
- [ ] Resume upload and analysis
- [ ] More question categories

---

## 16. How to Run

```powershell
# Always start here
cd "C:\Users\Manan Sharma\OneDrive\Documents\ai interview project\interview_evaluator"

# Set API key (do this every new terminal session)
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Train emotion model
python modules\emotion_model.py

# Run full test
python test_all.py

# Launch app
streamlit run app.py
```

App runs at: http://localhost:8501

---

## 17. Files Delivered (download from Claude outputs)

| File | Location | Status |
|---|---|---|
| emotion_model.py | modules\ | Updated with CREMA-D + 4 algorithms |
| nlp_evaluator.py | modules\ | Updated, imports from questions_bank |
| questions_bank.py | modules\ | NEW FILE — 275 questions + ideal answers |
| ai_evaluator.py | modules\ | NEW FILE — Claude API feedback |
| app.py | interview_evaluator\ | Updated with all new features |
| test_all.py | interview_evaluator\ | Fixed for new model dict format |

---

*Last updated: March 2026*
*Project: ARIES — Automated Real-time Interview Evaluation & Integrity System*
