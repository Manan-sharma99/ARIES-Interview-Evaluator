# Aries Emotion Intelligence Module
Version: 2.x
Status: Active Development

---

# 1. Purpose

This module is NOT intended to be a generic Speech Emotion Recognition (SER) system.

Its purpose is to act as the emotional intelligence component of the Aries Interview Analysis Platform.

The final objective is not to predict:

- happy
- sad
- angry
- fearful

The final objective is to estimate interview-relevant behavioral states:

- confident
- neutral
- nervous
- stressed

Emotion recognition is only an intermediate signal used to support interview evaluation.

---

# 2. Project Philosophy

Many academic SER projects report high accuracy by:

- using acted datasets
- using random train/test splits
- allowing speaker leakage
- optimizing only benchmark metrics

These approaches often fail in real-world environments.

This project prioritizes:

1. Generalization over benchmark accuracy.
2. Real interview usefulness over research leaderboard performance.
3. Robust evaluation over inflated results.
4. Explainability over black-box predictions.

---

# 3. Current Problem Definition

Input: Audio response from interview candidate.

Output:

```json
{
  "raw_emotion": "fearful",
  "interview_category": "nervous",
  "confidence_score": 0.78
}
```

---

# 4. Current Datasets

## RAVDESS
Strengths:
- Balanced
- High quality recordings
- Widely used SER benchmark

Weaknesses:
- Acted emotions
- Studio environment

## CREMA-D
Strengths:
- More speakers
- Greater variability

Weaknesses:
- Still acted speech

### Important Limitation

Neither dataset contains real interview speech.

Current model should be viewed as:

> Emotion approximation

rather than

> Interview confidence detector

---

# 5. Architecture Evolution

## Version 1

Audio â†’ MFCC â†’ RF/SVM/XGB â†’ Emotion

Problems:
- Speaker leakage
- Weak features
- No augmentation
- No VAD
- Single train/test split

## Version 2

Audio â†’ VAD â†’ Augmentation â†’ Advanced Features â†’ Multiple Models â†’ Best Model â†’ Emotion â†’ Interview Mapping

Improvements:
- Speaker-aware splitting
- Delta MFCC
- Delta-Delta MFCC
- Spectral Contrast
- Optimized feature extraction with Tonnetz and harmonic/percussive features removed
- Pitch Statistics
- Macro F1 evaluation
- Grid Search
- Model metadata saving

---

# 6. Current Pipeline

Audio
â†“
Preprocessing
â†“
Silence Removal
â†“
Feature Extraction
â†“
Feature Scaling
â†“
Model Training
â†“
Emotion Prediction
â†“
Interview Mapping
â†“
Final Output

---

# 7. Feature Extraction Design

Current feature count = 207 dimensions.

## Core Features

### MFCC
Purpose: Capture vocal tract shape.

### Delta MFCC
Purpose: Capture temporal change.

### Delta-Delta MFCC
Purpose: Capture acceleration of change.

### Spectral Contrast
Purpose: Capture difference between peaks and valleys.

### Pitch Statistics
- Mean
- Std
- Min
- Max
- Median

### RMS + ZCR
Purpose: Energy and speech activity.

### Removed for Performance
- Tonnetz
- Harmonic/percussive energy statistics

Reason:
- Low value for interview speech
- High feature extraction cost due to harmonic/percussive decomposition
- Removed to reduce training time while preserving the existing model architecture

---

# 8. Current Known Engineering Issues

## Issue #1 â€” Scaler Leakage
Priority: CRITICAL

Fix:
- Use sklearn Pipeline
- Fit scaler inside CV folds

## Issue #2 â€” Metric Inconsistency
Priority: HIGH

Fix:
- Use scoring="f1_macro"

## Issue #3 â€” Class Imbalance
Priority: HIGH

Fix:
- class_weight="balanced"

## Issue #4 â€” Training Speed
Priority: MEDIUM

Fix:
- joblib.Parallel

Expected:
- 4â€“8x speedup

## Issue #5 â€” Reproducibility
Priority: MEDIUM

Fix:
- Global random seed

---

# 9. What Should NOT Be Changed

## Speaker-Aware Splitting
Never revert to train_test_split().

## Macro F1 Evaluation
Never use accuracy alone.

## Metadata Saving
Always save:
- model
- scaler
- encoder
- metadata

---

# 10. Highest ROI Future Improvements

## Priority 1 â€” Speaking Rate Analysis
Metrics:
- Words per minute
- Syllables per second

Expected gain:
- 5â€“10% interview scoring improvement

## Priority 2 â€” Pause Analysis
Metrics:
- Pause count
- Pause duration
- Pause frequency

## Priority 3 â€” Jitter and Shimmer
Library:
- Praat / Parselmouth

## Priority 4 â€” Harmonic-To-Noise Ratio (HNR)

## Priority 5 â€” Wav2Vec2 Embeddings

Architecture:

Audio
â†“
Wav2Vec2
â†“
Embeddings
â†“
XGBoost
â†“
Emotion

Expected gain:
- 10â€“20% Macro F1

Status:
- Phase 2 Research Upgrade

## Priority 6 â€” Additional Datasets
- TESS
- SAVEE
- IEMOCAP

---

# 11. Why Wav2Vec2 Was Deferred

Current focus:
- Reliable baseline
- Explainability
- Low latency

Wav2Vec2 introduces:
- Larger models
- Higher memory usage
- Slower inference

Add only after baseline validation.

---

# 12. Aries Integration Strategy

Recommended future weighting:

| Component | Weight |
|------------|---------|
| Emotion Analysis | 20% |
| Fluency Analysis | 25% |
| Content Analysis | 25% |
| Confidence Metrics | 20% |
| Pause Analysis | 10% |

---

# 13. Long-Term Vision

Audio
â†“
Speech-To-Text
â†“
Emotion Analysis
+
Speaking Rate
+
Pause Analysis
+
Fluency Analysis
+
Confidence Metrics
+
NLP Content Evaluation
â†“
Scoring Engine
â†“
Interview Report

Goal:

Transform Aries from an SER project into a full Interview Intelligence Platform.

---

# 14. Technical Debt Register

- [x] Fix scaler leakage
- [x] Convert GridSearch scoring to Macro F1
- [~] Add class balancing (RF/SVM/LightGBM done; XGBoost sample weights remaining)
- [x] Add deterministic augmentation
- [ ] Add parallel feature extraction
- [ ] Add speaking-rate module
- [ ] Add pause-analysis module
- [ ] Add jitter/shimmer module
- [ ] Add HNR module
- [ ] Prototype Wav2Vec2 pipeline
- [ ] Evaluate TESS
- [ ] Evaluate SAVEE
- [ ] Evaluate IEMOCAP

---

# 15. Success Criteria

The module is considered mature when:

- No evaluation leakage exists
- Macro F1 is stable across folds
- Results are reproducible
- Speaking-rate analysis is integrated
- Pause analysis is integrated
- Emotion prediction is combined with interview signals
- Real interview recordings outperform Version 1 baseline

At that point, Aries transitions from an SER project to a genuine Interview Intelligence Platform.

---

# 16. Implemented Fixes - Current Patch

Implemented in emotion_model.py:

- Fixed scaler leakage by moving StandardScaler into sklearn Pipeline for every model.
- Removed train_full_pipeline pre-CV scaling; raw train/test features are now passed to model pipelines.
- GridSearchCV now optimizes Macro F1 with scoring="f1_macro".
- CV reporting now uses Macro F1 on cloned unfitted pipelines.
- Best model saving now selects by held-out raw Macro F1 instead of accuracy/interview accuracy.
- Added class_weight="balanced" to Random Forest and LightGBM; SVM already used it.
- Made augmentation deterministic through RANDOM_STATE-backed RandomState.
- Updated inference to avoid double scaling when the saved model is a sklearn Pipeline while preserving legacy scaler support.

Feature extraction performance update:

- Removed Tonnetz features from emotion_model.py.
- Removed harmonic/percussive energy features from emotion_model.py.
- Feature vector changed from 217 actual dimensions to 207 dimensions.
- Saved model metadata now records feature_set_version, expected_feature_dim, and removed_features.

Remaining high-priority caveat:

- XGBoost does not yet receive per-class sample weights. Add sample_weight support in a future patch without changing model architecture.
