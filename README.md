# ARIES – AI Recruitment & Interview Evaluation System

## Overview

ARIES (AI Recruitment & Interview Evaluation System) is an AI-powered Interview Intelligence Platform designed to evaluate candidates through speech analysis, emotion recognition, natural language processing, and behavioral assessment.

Most interview preparation tools focus only on whether an answer is correct. ARIES aims to go beyond content evaluation by analyzing how a candidate communicates, how confident they sound, how emotionally stable they remain under questioning, and how effectively they respond during an interview.

The long-term goal of ARIES is to transform traditional interview assessment into a data-driven process that combines Speech Processing, Natural Language Processing, Machine Learning, and Large Language Models.

---

## Why This Project Exists

During interviews, recruiters evaluate much more than technical knowledge.

Candidates are often judged on:

- Confidence
- Communication skills
- Emotional stability
- Clarity of speech
- Relevance of answers
- Ability to handle follow-up questions

However, these evaluations are often subjective and depend heavily on the interviewer.

ARIES was created to explore how Artificial Intelligence can provide a more objective and consistent interview assessment framework.

Rather than functioning as a simple Speech Emotion Recognition project, ARIES is being developed as a complete Interview Intelligence Platform capable of evaluating multiple dimensions of candidate performance.

---

## Current Capabilities

ARIES currently includes four major subsystems:

### 1. Speech Emotion Recognition

Analyzes candidate audio responses and predicts emotional states using machine learning.

The system currently evaluates emotions using acoustic features extracted from speech recordings.

### 2. NLP-Based Answer Evaluation

Uses Sentence-BERT embeddings to compare candidate responses against ideal answers.

This allows semantic similarity scoring instead of simple keyword matching.

### 3. Speech Processing Pipeline

Uses OpenAI Whisper for speech-to-text transcription.

Audio responses are automatically converted into text before evaluation.

### 4. AI Follow-Up Question Generation

Generates context-aware interview follow-up questions using Gemini.

Instead of asking generic questions, the system analyzes candidate responses and creates relevant probing questions similar to those asked by real interviewers.

---

## Project Evolution

ARIES originally started as a Speech Emotion Recognition project.

As development progressed, several limitations became apparent.

Emotion classification alone could not provide meaningful interview feedback.

A candidate may sound nervous but still provide excellent answers.

Similarly, a confident candidate may provide poor technical responses.

Because of this, the project evolved toward a larger vision:

```text
Speech Emotion Recognition
            ↓
Interview Intelligence Platform
```

Future versions will combine:

- Emotion Analysis
- Speaking Rate Analysis
- Pause Analysis
- Confidence Scoring
- NLP Evaluation
- Fluency Analysis
- AI Follow-Up Generation

to produce comprehensive interview assessments.

---

## Current Dataset

To improve robustness and speaker diversity, ARIES uses multiple public emotion datasets.

### RAVDESS

- 24 Speakers
- 1440 Audio Recordings

### CREMA-D

- 91 Speakers
- 7442 Audio Recordings

### Combined Dataset

- 115 Speakers
- 8882 Audio Samples

This scale provides significantly greater speaker diversity than many academic and student emotion-recognition projects.
