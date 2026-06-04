# Changelog

All notable changes to ARIES will be documented in this file.

The project follows a milestone-based development approach where each version represents a significant improvement to the Interview Intelligence Platform.

---

# v1.1.0 — Emotion Intelligence Pipeline Upgrade
Date: June 2026

## Added

### Dataset Expansion
- Integrated RAVDESS dataset
- Integrated CREMA-D dataset
- Expanded training corpus to 115 speakers and 8882 audio recordings

### Evaluation Improvements
- Implemented speaker-aware train/test splitting
- Added cross-validation support
- Added more reliable benchmarking workflow

### Feature Engineering
- MFCC Features
- Delta MFCC Features
- Delta-Delta MFCC Features
- Spectral Contrast Features
- Spectral Bandwidth Features
- Pitch Statistics
- RMS Energy
- Zero Crossing Rate

### Audio Processing
- Voice Activity Detection (VAD)
- Automatic silence trimming

### Training Pipeline
- Parallel feature extraction
- Feature caching system
- Progress logging
- CLI experiment controls

### Model Evaluation
- Random Forest benchmarking
- SVM benchmarking

---

## Improved

### Training Performance

Before:

```text
5–6 Hours
```

After:

```text
~22 Minutes
```

Approximate speed improvement:

```text
16× Faster
```

### Development Workflow

Improved:

- Experiment iteration speed
- Training visibility
- Feature reuse
- Reproducibility

---

## Results

### Best Model

```text
SVM
```

### Metrics

```text
Cross Validation Accuracy : 52.91%
Test Accuracy             : 48.89%
Macro F1                  : 47.75%
Interview Accuracy        : 57.66%
```

---

## Fixed

### Speaker Leakage

Resolved an evaluation flaw where speakers could appear in both training and testing datasets.

### Redundant Feature Extraction

Eliminated repeated feature computation through persistent caching.

### Slow Feature Processing

Resolved preprocessing bottlenecks through parallel execution.

---

## Known Limitations

- Class imbalance remains present
- Minority emotion classes require additional data
- XGBoost currently blocked by Windows memory limitations

---

# v1.0.0 — Initial ARIES Prototype

## Added

- Speech Emotion Recognition
- Whisper Speech-to-Text
- NLP Evaluation Pipeline
- FastAPI Backend
- Streamlit Frontend
- Gemini Follow-Up Question Generation
- Recruiter Dashboard Foundation
