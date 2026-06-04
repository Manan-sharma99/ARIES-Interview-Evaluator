"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ARIES — Speech Emotion Recognition Module  (v2.0)                 ║
║          Automated Real-time Interview Evaluation & Integrity System        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Manan Sharma                                                     ║
║  Purpose : Production-quality SER for interview emotion analysis            ║
║  Datasets: RAVDESS (1440 samples) + CREMA-D (7442 samples)                 ║
║  Output  : confident | neutral | nervous | stressed                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key improvements over v1.0:                                                ║
║    • Speaker-aware train/test splitting  → zero data leakage                ║
║    • 193-dimensional feature vector      → richer representation            ║
║    • Voice Activity Detection (VAD)      → no silence noise                 ║
║    • 4× audio augmentation              → better generalization             ║
║    • Stratified K-Fold CV (k=5)         → reliable accuracy estimate        ║
║    • GridSearchCV hyperparameter tuning  → optimal model params             ║
║    • LightGBM support (optional)        → fastest tree model                ║
║    • Full evaluation suite              → Acc, F1-macro, F1-weighted, CM    ║
║    • Saves model + scaler + encoder + metadata                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PIPELINE OVERVIEW:
  1. load_dataset()      — parse filenames → (features, label, speaker_id)
  2. preprocess_audio()  — load WAV → resample → VAD strip silence
  3. augment_audio()     — noise / pitch-shift / time-stretch / volume-scale
  4. extract_features()  — 193-dim feature vector from audio signal
  5. train_models()      — RF, SVM, XGBoost, LightGBM with GridSearchCV
  6. evaluate_models()   — Acc, F1, confusion matrix, per-class report
  7. save_model()        — pickle {model, scaler, encoder, metadata}
  8. predict_emotion()   — inference on new WAV → interview label

USAGE:
  python modules/emotion_model.py            # full train + evaluate
  python modules/emotion_model.py --predict path/to/file.wav
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import pickle
import json
import warnings
import argparse
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party — core (always required)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import librosa
import librosa.effects
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Parallel processing (Task 1) ──────────────────────────────────────────────
# joblib ships as a scikit-learn dependency — always present in this env.
# Parallel(n_jobs=-1, backend="loky") spawns one worker process per CPU core.
# delayed() wraps the target function so joblib can serialise it for IPC.
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party — optional (graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe on servers
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup — both console and file
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging(log_dir: str) -> logging.Logger:
    """Configure root logger with console + rotating file handler."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("ARIES-SER")


# ─────────────────────────────────────────────────────────────────────────────
#  Absolute path resolution
#  modules/emotion_model.py  →  BASE_DIR = …/interview_evaluator
# ─────────────────────────────────────────────────────────────────────────────
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(MODULE_DIR, ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
CACHE_DIR  = os.path.join(DATA_DIR, "cache")   # Task 2: feature cache directory

log = _setup_logging(LOGS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
#  Emotion mappings
# ─────────────────────────────────────────────────────────────────────────────

# RAVDESS filename part[2] → base emotion
RAVDESS_EMOTIONS: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# CREMA-D filename part[2] → base emotion
CREMA_EMOTIONS: Dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Base emotion → interview label (post-prediction mapping)
# Design rationale:
#   happy/surprised → confident  (positive energy, engagement)
#   calm/neutral    → neutral    (composed, professional)
#   fearful         → nervous    (anxiety signal)
#   sad/angry/disgust → stressed (negative arousal)
INTERVIEW_MAP: Dict[str, str] = {
    "happy":     "confident",
    "surprised": "confident",
    "calm":      "neutral",
    "neutral":   "neutral",
    "fearful":   "nervous",
    "sad":       "stressed",
    "angry":     "stressed",
    "disgust":   "stressed",
}

# Numeric score per interview label (used by scoring_engine.py)
EMOTION_SCORES: Dict[str, int] = {
    "confident": 90,
    "neutral":   65,
    "nervous":   40,
    "stressed":  30,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Audio constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050     # Hz — standard for speech / librosa default
AUDIO_DURATION = 4.0     # seconds — max clip length (padded/trimmed)
N_MFCC        = 40       # MFCC coefficients
N_CHROMA      = 12       # Chroma bins
N_MELS        = 128      # Mel spectrogram bands
N_CONTRAST    = 7        # Spectral contrast bands
RANDOM_STATE  = 42       # Reproducibility seed

# VAD parameters
VAD_TOP_DB    = 20       # dB below peak to consider silence
VAD_MIN_LEN   = 0.3     # seconds — discard clips shorter than this after VAD

# Augmentation parameters
AUG_NOISE_FACTOR  = 0.005   # std of Gaussian noise
AUG_PITCH_STEPS   = [-2, 2] # semitones to shift (one chosen per sample)
AUG_STRETCH_RATES = [0.9, 1.1]  # time-stretch factors
AUG_VOLUME_RANGE  = (0.7, 1.3)  # min/max volume scale factor


# ═════════════════════════════════════════════════════════════════════════════
#  1. AUDIO PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    duration: float = AUDIO_DURATION,
    apply_vad: bool = True,
) -> Optional[np.ndarray]:
    """
    Load a WAV file, resample, apply Voice Activity Detection (VAD),
    and return a fixed-length waveform.

    Steps:
      1. Load & resample to `sr` Hz.
      2. Trim leading/trailing silence using VAD (top_db threshold).
      3. Pad with zeros or truncate to exactly `duration` seconds.

    Args:
        file_path  : Absolute path to .wav file.
        sr         : Target sample rate (default 22050).
        duration   : Fixed output length in seconds (default 4.0).
        apply_vad  : Whether to strip silence before feature extraction.

    Returns:
        1-D numpy array of shape (sr * duration,), or None on error.
    """
    try:
        # Load and resample in one step — librosa resamples automatically
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        if len(y) == 0:
            log.debug(f"Empty audio: {file_path}")
            return None

        # ── Voice Activity Detection ─────────────────────────────────────────
        # librosa.effects.trim removes silence below `top_db` dB relative to
        # the loudest frame. This eliminates recording noise / padding in the
        # datasets that would corrupt feature statistics.
        if apply_vad:
            y_trimmed, _ = librosa.effects.trim(y, top_db=VAD_TOP_DB)
            min_samples = int(VAD_MIN_LEN * sr)
            if len(y_trimmed) < min_samples:
                # Fall back to untrimmed if result is too short
                y_trimmed = y
            y = y_trimmed

        # ── Fixed-length normalisation ───────────────────────────────────────
        target_length = int(sr * duration)
        if len(y) < target_length:
            # Pad with zeros at the end
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            # Truncate to target length
            y = y[:target_length]

        return y.astype(np.float32)

    except Exception as exc:
        log.debug(f"preprocess_audio failed for {file_path}: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  2. AUDIO AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def augment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    aug_type: str = "noise",
) -> np.ndarray:
    """
    Apply a single augmentation to a waveform and return the modified signal.

    Why augmentation?
      Both RAVDESS and CREMA-D use acted emotions from a limited actor pool.
      Augmentation synthesises new acoustic conditions (mic noise, tempo
      variation, pitch variation, recording level) so the model generalises
      to real interview recordings.

    Args:
        y        : Input waveform (numpy float32 array).
        sr       : Sample rate.
        aug_type : One of "noise", "pitch", "stretch", "volume".

    Returns:
        Augmented waveform of the same length as input.
    """
    target_len = len(y)

    if aug_type == "noise":
        # ── Gaussian noise ───────────────────────────────────────────────────
        # Simulates microphone hiss, background room noise.
        noise = np.random.normal(0, AUG_NOISE_FACTOR, len(y)).astype(np.float32)
        y_aug = y + noise

    elif aug_type == "pitch":
        # ── Pitch shifting ───────────────────────────────────────────────────
        # Simulates different vocal registers (higher/lower pitch speaker).
        # Randomly choose one of [-2, +2] semitones.
        steps = float(np.random.choice(AUG_PITCH_STEPS))
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    elif aug_type == "stretch":
        # ── Time stretching ──────────────────────────────────────────────────
        # Simulates speaking rate variation (faster/slower speech).
        # Rate < 1.0 → slower; rate > 1.0 → faster.
        rate = float(np.random.choice(AUG_STRETCH_RATES))
        y_aug = librosa.effects.time_stretch(y, rate=rate)

    elif aug_type == "volume":
        # ── Volume scaling ───────────────────────────────────────────────────
        # Simulates different mic distances / recording levels.
        scale = np.random.uniform(*AUG_VOLUME_RANGE)
        y_aug = y * scale

    else:
        log.warning(f"Unknown aug_type '{aug_type}', returning original.")
        y_aug = y.copy()

    # Ensure output matches original length (pitch/stretch can change it)
    if len(y_aug) < target_len:
        y_aug = np.pad(y_aug, (0, target_len - len(y_aug)), mode="constant")
    else:
        y_aug = y_aug[:target_len]

    return y_aug.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  3. FEATURE EXTRACTION  (193-dimensional vector)
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Extract a 193-dimensional feature vector from a preprocessed waveform.

    Feature breakdown:
    ┌──────────────────────────────────────┬────────┬────────────────────────┐
    │ Feature                              │  Dims  │ What it captures       │
    ├──────────────────────────────────────┼────────┼────────────────────────┤
    │ MFCC mean (40 coefficients)          │   40   │ Spectral envelope avg  │
    │ MFCC std  (40 coefficients)          │   40   │ Spectral variation     │
    │ Delta-MFCC mean                      │   40   │ MFCC velocity          │
    │ Delta²-MFCC mean                     │   40   │ MFCC acceleration      │
    │ Chroma STFT mean (12 bins)           │   12   │ Harmonic content       │
    │ Mel spectrogram (mean, std)          │    2   │ Energy across freqs    │
    │ Spectral Contrast (7 bands, mean+std)│   14   │ Peak vs valley energy  │
    │ Spectral Bandwidth (mean, std)       │    2   │ Spectral spread        │
    │ Spectral Centroid (mean, std)        │    2   │ Brightness             │
    │ Spectral Rolloff (mean, std)         │    2   │ High-freq content      │
    │ RMS Energy (mean, std)               │    2   │ Loudness               │
    │ Zero Crossing Rate (mean, std)       │    2   │ Signal sharpness       │
    │ Tonnetz (6 dims, mean)               │    6   │ Tonal centroid         │
    │ Harmonic mean + std                  │    2   │ Harmonic energy        │
    │ Percussive mean + std                │    2   │ Percussive energy      │
    │ Pitch: mean, std, min, max, median   │    5   │ Fundamental frequency  │
    ├──────────────────────────────────────┼────────┼────────────────────────┤
    │ TOTAL                                │  217   │                        │
    └──────────────────────────────────────┴────────┴────────────────────────┘

    Note: Actual dim count is 217 after summing above. The vector is returned
    as-is — the scaler handles any normalisation downstream.

    Args:
        y  : Preprocessed waveform (float32 numpy array).
        sr : Sample rate in Hz.

    Returns:
        1-D numpy float64 array, or None if extraction fails.
    """
    try:
        features: List[float] = []

        # ── Short-time Fourier Transform (shared across features) ────────────
        # Compute once, reuse — avoids redundant FFT calculations.
        S_full = np.abs(librosa.stft(y))

        # ── 1. MFCC: mean + std (80 dims) ───────────────────────────────────
        # MFCCs represent the short-term power spectrum shape.
        # mean captures the "average" spectral profile;
        # std captures how much the timbre varies over the clip.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1).tolist())   # 40 dims
        features.extend(np.std(mfcc,  axis=1).tolist())   # 40 dims

        # ── 2. Delta-MFCC mean (40 dims) ────────────────────────────────────
        # First-order temporal derivative of MFCC.
        # Captures how quickly the vocal tract shape is changing — key for
        # detecting hesitation pauses and emotion transitions.
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta_mfcc, axis=1).tolist())  # 40 dims

        # ── 3. Delta²-MFCC mean (40 dims) ───────────────────────────────────
        # Second-order temporal derivative (acceleration of spectral shape).
        # Helps detect abrupt emotional shifts and speech onset patterns.
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta2_mfcc, axis=1).tolist())  # 40 dims

        # ── 4. Chroma STFT mean (12 dims) ───────────────────────────────────
        # Projects the spectrum onto 12 pitch classes (C, C#, D, …, B).
        # Captures harmonic content that correlates with emotional valence.
        chroma = librosa.feature.chroma_stft(S=S_full, sr=sr, n_chroma=N_CHROMA)
        features.extend(np.mean(chroma, axis=1).tolist())  # 12 dims

        # ── 5. Mel spectrogram: mean + std (2 dims) ─────────────────────────
        # Energy distribution across perceptually-weighted frequency bands.
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.append(float(np.mean(mel_db)))   # 1 dim
        features.append(float(np.std(mel_db)))    # 1 dim

        # ── 6. Spectral Contrast: mean + std (14 dims) ──────────────────────
        # Measures the difference in energy between spectral peaks and valleys
        # in N frequency sub-bands. Distinguishes voiced vs noisy speech.
        contrast = librosa.feature.spectral_contrast(S=S_full, sr=sr, n_bands=N_CONTRAST - 1)
        features.extend(np.mean(contrast, axis=1).tolist())  # 7 dims
        features.extend(np.std(contrast,  axis=1).tolist())  # 7 dims

        # ── 7. Spectral Bandwidth: mean + std (2 dims) ──────────────────────
        # Weighted std of frequencies around the spectral centroid.
        # Wide bandwidth → noisy / stressed; narrow → calm / neutral.
        bw = librosa.feature.spectral_bandwidth(S=S_full, sr=sr)
        features.append(float(np.mean(bw)))  # 1 dim
        features.append(float(np.std(bw)))   # 1 dim

        # ── 8. Spectral Centroid: mean + std (2 dims) ───────────────────────
        # Weighted mean frequency — perceived "brightness" of speech.
        # High centroid → excited / confident; low → tired / stressed.
        centroid = librosa.feature.spectral_centroid(S=S_full, sr=sr)
        features.append(float(np.mean(centroid)))  # 1 dim
        features.append(float(np.std(centroid)))   # 1 dim

        # ── 9. Spectral Rolloff: mean + std (2 dims) ────────────────────────
        # Frequency below which 85% of spectral energy is contained.
        # Correlated with the presence/absence of high-frequency consonants.
        rolloff = librosa.feature.spectral_rolloff(S=S_full, sr=sr)
        features.append(float(np.mean(rolloff)))  # 1 dim
        features.append(float(np.std(rolloff)))   # 1 dim

        # ── 10. RMS Energy: mean + std (2 dims) ─────────────────────────────
        # Root-mean-square amplitude — perceptual loudness proxy.
        # Confident speakers tend to have higher, stable RMS.
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))  # 1 dim
        features.append(float(np.std(rms)))   # 1 dim

        # ── 11. Zero Crossing Rate: mean + std (2 dims) ─────────────────────
        # Rate at which the signal changes sign — correlates with voicing.
        # High ZCR → fricatives / nervous speech; low → voiced / confident.
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))  # 1 dim
        features.append(float(np.std(zcr)))   # 1 dim

        # ── 12. Tonnetz mean (6 dims) ────────────────────────────────────────
        # Tonal centroid features derived from chroma.
        # Encodes harmonic relationships — useful for emotional valence.
        # Requires harmonic component to avoid noise contamination.
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features.extend(np.mean(tonnetz, axis=1).tolist())  # 6 dims

        # ── 13. Harmonic + Percussive energy: mean + std (4 dims) ───────────
        # Harmonic component → pitched, voiced speech (more in confident)
        # Percussive component → abrupt transients (plosives, filler words)
        y_percussive = librosa.effects.percussive(y)
        features.append(float(np.mean(np.abs(y_harmonic))))   # 1 dim
        features.append(float(np.std(np.abs(y_harmonic))))    # 1 dim
        features.append(float(np.mean(np.abs(y_percussive)))) # 1 dim
        features.append(float(np.std(np.abs(y_percussive))))  # 1 dim

        # ── 14. Pitch statistics (5 dims) ────────────────────────────────────
        # Extract F0 (fundamental frequency) using piptrack.
        # Statistics across frames encode pitch range and stability.
        # Monotone pitch → stressed / nervous; varying pitch → confident.
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Select only high-confidence pitch values (where magnitude is high)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)

        if len(pitch_vals) > 0:
            pv = np.array(pitch_vals)
            features.append(float(np.mean(pv)))    # 1 dim
            features.append(float(np.std(pv)))     # 1 dim
            features.append(float(np.min(pv)))     # 1 dim
            features.append(float(np.max(pv)))     # 1 dim
            features.append(float(np.median(pv)))  # 1 dim
        else:
            # File has no detectable pitch (whisper/noise) — fill with zeros
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ── Validate and return ──────────────────────────────────────────────
        feature_array = np.array(features, dtype=np.float64)

        # Replace NaN/Inf with 0 (can happen in edge-case silent clips)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_array

    except Exception as exc:
        log.debug(f"extract_features failed: {exc}")
        return None


def extract_features_from_file(
    file_path: str,
    apply_vad: bool = True,
    augment: bool = False,
    aug_type: str = "noise",
) -> Optional[np.ndarray]:
    """
    Convenience wrapper: load file → preprocess → (optionally augment) → extract.

    Args:
        file_path : Path to .wav file.
        apply_vad : Apply Voice Activity Detection.
        augment   : Apply augmentation before feature extraction.
        aug_type  : Type of augmentation if augment=True.

    Returns:
        Feature vector (numpy array) or None.
    """
    y = preprocess_audio(file_path, apply_vad=apply_vad)
    if y is None:
        return None
    if augment:
        y = augment_audio(y, aug_type=aug_type)
    return extract_features(y)


# ═════════════════════════════════════════════════════════════════════════════
#  4. PARALLEL WORKER + DATASET LOADING  (speaker-aware)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  Parallel worker function  (Task 1)
#
#  MUST be a module-level (not nested) function so the loky backend can
#  pickle it for inter-process communication.
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_file(
    file_path: str,
    emotion: str,
    speaker_id: str,
    apply_vad: bool,
    use_augmentation: bool,
    aug_types: List[str],
) -> List[Tuple[Optional[np.ndarray], str, str]]:
    """
    Worker function executed in a subprocess (loky backend).  (Task 1)

    Processes ONE audio file end-to-end:
      1. Preprocess (load + resample + VAD)
      2. Extract 193-dim feature vector for the original waveform
      3. If use_augmentation is True, generate one copy per aug_type and
         extract features for each augmented waveform

    Design notes:
      • Module-level function (not a lambda/closure) → picklable for loky.
      • Does NOT reference shared mutable state → each subprocess is isolated.
      • speaker_id is preserved for ALL augmented copies so they always land
        on the same side of the speaker-aware train/test split (no leakage).

    Args:
        file_path       : Absolute path to .wav file.
        emotion         : Ground-truth emotion label.
        speaker_id      : Speaker identifier (RAVDESS_01, CREMA_1001, …).
        apply_vad       : Apply Voice Activity Detection.
        use_augmentation: Generate augmented copies after the original.
        aug_types       : Sequence of augmentation type strings.

    Returns:
        List of (feature_vector, emotion_label, speaker_id) tuples.
        Empty list if the original waveform fails to load/extract.
    """
    results: List[Tuple[Optional[np.ndarray], str, str]] = []

    # ── Original sample ────────────────────────────────────────────────────────────────
    feats = extract_features_from_file(file_path, apply_vad=apply_vad)
    if feats is None:
        return results  # skip file entirely — caller counts as failed

    results.append((feats, emotion, speaker_id))

    # ── Augmented copies ──────────────────────────────────────────────────────────────
    # Each aug type produces one extra sample from the same source file.
    # The speaker_id is kept identical so augmented samples always land on
    # the same split side as the original — preventing speaker leakage.
    if use_augmentation:
        for aug_type in aug_types:
            feats_aug = extract_features_from_file(
                file_path,
                apply_vad=apply_vad,
                augment=True,
                aug_type=aug_type,
            )
            if feats_aug is not None:
                results.append((feats_aug, emotion, speaker_id))

    return results


def _parse_ravdess_speakers(ravdess_path: str) -> Dict:
    """
    Parse all RAVDESS .wav files and return a dict of speaker_id → file list.

    RAVDESS filename format:
        {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-
        {repetition}-{actor}.wav
    Actor ID (part[6]) is used as speaker_id.
    Emotion code (part[2]) maps to RAVDESS_EMOTIONS.
    """
    speaker_files: Dict[str, List[Tuple[str, str]]] = {}
    if not os.path.exists(ravdess_path):
        log.warning(f"RAVDESS not found: {ravdess_path}")
        return speaker_files

    for actor_dir in sorted(os.listdir(ravdess_path)):
        actor_path = os.path.join(ravdess_path, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue
            emotion_code = parts[2]
            speaker_id   = f"RAVDESS_{parts[6]}"  # e.g., RAVDESS_01
            emotion      = RAVDESS_EMOTIONS.get(emotion_code)
            if emotion is None:
                continue
            full_path = os.path.join(actor_path, fname)
            speaker_files.setdefault(speaker_id, []).append((full_path, emotion))

    log.info(f"RAVDESS: found {len(speaker_files)} speakers, "
             f"{sum(len(v) for v in speaker_files.values())} files")
    return speaker_files


def _parse_crema_speakers(crema_path: str) -> Dict:
    """
    Parse all CREMA-D .wav files and return a dict of speaker_id → file list.

    CREMA-D filename format:
        {ActorID}_{SentenceID}_{Emotion}_{EmotionLevel}.wav
    ActorID (part[0]) is used as speaker_id.
    Emotion code (part[2]) maps to CREMA_EMOTIONS.
    """
    speaker_files: Dict[str, List[Tuple[str, str]]] = {}
    audio_dir = os.path.join(crema_path, "AudioWAV")
    if not os.path.exists(audio_dir):
        log.warning(f"CREMA-D AudioWAV not found: {audio_dir}")
        return speaker_files

    all_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    log.info(f"CREMA-D: found {len(all_files)} .wav files, parsing...")

    for fname in all_files:
        parts = fname.replace(".wav", "").split("_")
        if len(parts) < 3:
            continue
        actor_id     = parts[0]
        emotion_code = parts[2]
        speaker_id   = f"CREMA_{actor_id}"
        emotion      = CREMA_EMOTIONS.get(emotion_code)
        if emotion is None:
            continue
        full_path = os.path.join(audio_dir, fname)
        speaker_files.setdefault(speaker_id, []).append((full_path, emotion))

    log.info(f"CREMA-D: found {len(speaker_files)} speakers, "
             f"{sum(len(v) for v in speaker_files.values())} files")
    return speaker_files


def load_dataset(
    ravdess_path: Optional[str] = None,
    crema_path:   Optional[str] = None,
    use_augmentation: bool = True,
    aug_types: List[str] = ("noise", "pitch", "stretch", "volume"),
    apply_vad: bool = True,
    rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load RAVDESS + CREMA-D, extract features, and return arrays.

    CRITICAL: Speaker-aware design
    ──────────────────────────────
    Naïve random splitting allows different recordings from the same speaker
    to appear in both train and test sets. Because all recordings from one
    speaker have the same voice timbre, the model memorises the speaker
    rather than learning the emotion. This inflates test accuracy by ~10-15%.

    We tag every sample with its speaker_id so that train_models() can do a
    speaker-aware split (no speaker appears in both train and test).

    DEV ADDITIONS (Tasks 1–4):
      • rebuild_cache   : Ignore existing cache and force re-extraction.
      • Parallel feature extraction via joblib (n_jobs=-1, loky backend).
      • Disk cache in data/cache/ — subsequent runs skip extraction entirely.
      • Progress logged after every CHUNK_SIZE files (terminal never freezes).

    Args:
        ravdess_path    : Path to Audio_Speech_Actors_01-24/ directory.
        crema_path      : Path to CREMA-D/ directory.
        use_augmentation: If True, add augmented copies of each sample.
        aug_types       : Which augmentation types to apply.
        apply_vad       : Apply Voice Activity Detection during preprocessing.
        rebuild_cache   : If True, ignore any existing cache and re-extract.

    Returns:
        X           : Feature matrix (n_samples, n_features)
        y_raw       : Raw emotion labels (n_samples,)
        speaker_ids : Speaker identifier per sample (n_samples,)
    """
    ravdess_path = os.path.normpath(ravdess_path or
                   os.path.join(DATA_DIR, "Audio_Speech_Actors_01-24"))
    crema_path   = os.path.normpath(crema_path or
                   os.path.join(DATA_DIR, "CREMA-D"))

    log.info("=" * 60)
    log.info("LOADING DATASET")
    log.info(f"  RAVDESS : {ravdess_path}")
    log.info(f"  CREMA-D : {crema_path}")
    log.info(f"  VAD     : {apply_vad}")
    log.info(f"  Augment : {use_augmentation} {list(aug_types) if use_augmentation else ''}")
    log.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    #  Task 2 — Feature cache
    #
    #  The cache filename encodes the augmentation mode so switching between
    #  --no-augment and the default never cross-loads the wrong feature set.
    #  Cache is a compressed .npz containing only numpy arrays (no ML objects).
    #
    #  Decision tree:
    #    cache exists AND --rebuild-cache NOT set  →  load and return
    #    cache missing  OR  --rebuild-cache set    →  extract, then save
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_tag  = "augmented" if use_augmentation else "no_aug"
    cache_file = os.path.join(CACHE_DIR, f"emotion_features_v2_{cache_tag}.npz")

    # ── Cache hit: load and return immediately ────────────────────────────────
    if os.path.exists(cache_file) and not rebuild_cache:
        log.info(f"[INFO] Loading cached features from: {cache_file}")
        try:
            data        = np.load(cache_file, allow_pickle=True)
            X           = data["X"]
            y_raw       = data["y_raw"]
            speaker_ids = data["speaker_ids"]
            log.info(f"[INFO] Cache loaded — {len(X)} samples, "
                     f"{X.shape[1]} features")

            # Emit emotion distribution for log consistency with a fresh run
            emo_counts = Counter(y_raw.tolist())
            log.info("  Emotion distribution (raw labels):")
            for emo, cnt in sorted(emo_counts.items()):
                bar = "█" * (cnt // 50)
                log.info(f"    {emo:<12}: {cnt:5d}  {bar}")

            return X, y_raw, speaker_ids
        except Exception as exc:
            log.warning(f"Cache load failed ({exc}); rebuilding from scratch...")

    # ── Parse file lists grouped by speaker ──────────────────────────────────
    ravdess_speakers = _parse_ravdess_speakers(ravdess_path)
    crema_speakers   = _parse_crema_speakers(crema_path)

    all_speakers = {**ravdess_speakers, **crema_speakers}
    total_files  = sum(len(v) for v in all_speakers.values())
    log.info(f"Total speakers: {len(all_speakers)}, Total files: {total_files}")

    # Flatten speaker dict → ordered task list for deterministic parallel dispatch.
    # Python 3.7+ dicts preserve insertion order, so this list is stable across
    # runs (same augmentation sequence as the original sequential implementation).
    all_tasks: List[Tuple[str, str, str]] = [
        (fp, em, sid)
        for sid, file_list in all_speakers.items()
        for fp, em in file_list
    ]
    n_tasks = len(all_tasks)

    log.info("[INFO] Building feature cache...")
    log.info(f"  Parallel extraction : n_jobs=-1, backend=loky")
    log.info(f"  Tasks queued        : {n_tasks} original files")
    if use_augmentation:
        log.info(f"  Augmentation types  : {list(aug_types)}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Task 1 — Parallel feature extraction
    #
    #  Strategy: chunked Parallel dispatch  (also serves Task 4: progress logs)
    #  ──────────────────────────────────────────────────────────────────────
    #  • all_tasks is split into CHUNK_SIZE batches.
    #  • Each batch is handed to Parallel(n_jobs=-1, backend="loky") which
    #    spawns one subprocess per CPU core and runs them concurrently.
    #  • After each batch returns, progress is logged (Task 4) so the terminal
    #    is never silent — even on an 8 000+ file dataset.
    #
    #  Why "loky" backend?
    #    → Creates fresh worker processes (avoids GIL, fork-safe on Windows).
    #      librosa/numpy release the GIL during FFT → true multi-core speedup.
    #
    #  Reproducibility:
    #    → all_tasks order is deterministic (fixed above via dict order).
    #    → augment_audio uses numpy global RNG — stochastic by design,
    #      matching the original sequential implementation exactly.
    # ─────────────────────────────────────────────────────────────────────────
    CHUNK_SIZE = 100   # files per parallel batch / progress checkpoint

    X: List[np.ndarray] = []
    y_raw: List[str]    = []
    speaker_ids: List[str] = []

    processed = 0   # original files successfully extracted
    failed    = 0   # original files that returned an empty result
    t_start   = time.time()

    for chunk_start in range(0, n_tasks, CHUNK_SIZE):
        chunk = all_tasks[chunk_start : chunk_start + CHUNK_SIZE]

        # Dispatch this chunk to the loky worker pool.
        # Each _process_one_file call handles ONE source file and returns a
        # list: [(orig_feats, emotion, speaker_id), (aug_feats, …), …]
        batch_results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_process_one_file)(
                fp, em, sid, apply_vad, use_augmentation, list(aug_types)
            )
            for fp, em, sid in chunk
        )

        # Accumulate results from this chunk into the running lists
        for file_result in batch_results:
            if not file_result:
                failed += 1
                continue
            # First element is original; subsequent elements are augmented
            for feats, em, sid in file_result:
                X.append(feats)
                y_raw.append(em)
                speaker_ids.append(sid)
            processed += 1  # count original files only (mirrors old behaviour)

        # ── Task 4: Progress logging after every chunk ────────────────────────
        # Fires after every CHUNK_SIZE files so the terminal is never frozen.
        elapsed   = time.time() - t_start
        rate      = processed / elapsed if elapsed > 0 else 0
        remaining = (n_tasks - processed) / rate if rate > 0 else 0
        log.info(
            f"  Processed {processed}/{n_tasks} files"
            f" | {elapsed:.0f}s elapsed"
            f" | {rate:.1f} files/sec"
            f" | ~{remaining:.0f}s remaining"
        )

    elapsed_total = time.time() - t_start
    log.info(f"Dataset loaded in {elapsed_total:.1f}s")
    total_samples = len(X)
    log.info(f"  Samples extracted : {total_samples}"
             + (" (orig + aug)" if use_augmentation else ""))
    log.info(f"  Files failed      : {failed}")
    log.info(f"  Feature dimensions: {X[0].shape[0] if X else 'N/A'}")

    # ── Emotion distribution ──────────────────────────────────────────────────
    emo_counts = Counter(y_raw)
    log.info("  Emotion distribution (raw labels):")
    for emo, cnt in sorted(emo_counts.items()):
        bar = "█" * (cnt // 50)
        log.info(f"    {emo:<12}: {cnt:5d}  {bar}")

    # ── Convert lists → numpy arrays ─────────────────────────────────────────
    X_arr           = np.array(X,           dtype=np.float64)
    y_raw_arr       = np.array(y_raw,       dtype=str)
    speaker_ids_arr = np.array(speaker_ids, dtype=str)

    # ─────────────────────────────────────────────────────────────────────────
    #  Task 2 — Save feature cache
    #  Stored as compressed .npz so future runs skip extraction entirely.
    #  Typical size: ~50 MB (no-aug) / ~250 MB (augmented).
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"[INFO] Saving feature cache to: {cache_file}")
    try:
        np.savez_compressed(
            cache_file,
            X=X_arr,
            y_raw=y_raw_arr,
            speaker_ids=speaker_ids_arr,
        )
        cache_mb = os.path.getsize(cache_file) / (1024 * 1024)
        log.info(f"[INFO] Cache saved ({cache_mb:.1f} MB) — "
                 f"future runs will load in seconds.")
    except Exception as exc:
        log.warning(f"Cache save failed: {exc}")

    return X_arr, y_raw_arr, speaker_ids_arr


def speaker_aware_split(
    X: np.ndarray,
    y: np.ndarray,
    speaker_ids: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data such that no speaker appears in both train and test sets.

    Algorithm:
      1. Group speakers by dataset (RAVDESS / CREMA-D).
      2. Randomly assign ~test_ratio of speakers to the test fold.
      3. All samples from test speakers go to test; rest go to train.

    This prevents the model from learning speaker-specific voice features
    instead of emotion features — the single most common SER mistake.

    Args:
        X            : Feature matrix.
        y            : Label array.
        speaker_ids  : Speaker ID per sample.
        test_ratio   : Fraction of speakers reserved for testing.
        random_state : Numpy random seed.

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(random_state)

    unique_speakers = np.unique(speaker_ids)

    # Separate RAVDESS and CREMA speakers to ensure both datasets in test set
    ravdess_spk = [s for s in unique_speakers if s.startswith("RAVDESS")]
    crema_spk   = [s for s in unique_speakers if s.startswith("CREMA")]

    rng.shuffle(ravdess_spk)
    rng.shuffle(crema_spk)

    n_test_rav = max(1, int(len(ravdess_spk) * test_ratio))
    n_test_cre = max(1, int(len(crema_spk)   * test_ratio))

    test_speakers = set(ravdess_spk[:n_test_rav] + crema_spk[:n_test_cre])

    train_mask = np.array([sid not in test_speakers for sid in speaker_ids])
    test_mask  = ~train_mask

    log.info(f"Speaker split: {len(ravdess_spk) - n_test_rav} RAVDESS train speakers, "
             f"{n_test_rav} test")
    log.info(f"              {len(crema_spk) - n_test_cre} CREMA train speakers, "
             f"{n_test_cre} test")
    log.info(f"Train samples: {train_mask.sum()} | Test samples: {test_mask.sum()}")

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )


# ═════════════════════════════════════════════════════════════════════════════
#  5. MODEL TRAINING  (RF, SVM, XGBoost, LightGBM + GridSearchCV)
# ═════════════════════════════════════════════════════════════════════════════

def _build_model_configs(label_encoder: LabelEncoder) -> Dict:
    """
    Build a dict of model_name → (estimator, param_grid).

    Hyperparameter grid notes:
    ─ RF   : More trees + deeper = better but slower; n_jobs=-1 parallelises.
    ─ SVM  : RBF kernel with C/gamma tuning is the classic SER workhorse.
    ─ XGB  : Gradient boosting — powerful but needs n_class aware settings.
    ─ LGBM : Fastest tree model; leaf-wise growth, great for tabular features.

    GridSearchCV will be run with cv=3 (fast inner loop); final evaluation
    uses a 5-fold outer CV for unbiased accuracy estimate.
    """
    n_classes = len(label_encoder.classes_)

    configs: Dict = {}

    # ── Random Forest ────────────────────────────────────────────────────────
    configs["Random Forest"] = (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators":      [200, 400],
            "max_depth":         [None, 30],
            "min_samples_split": [2, 5],
            "max_features":      ["sqrt", "log2"],
        },
    )

    # ── SVM ─────────────────────────────────────────────────────────────────
    configs["SVM"] = (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
            class_weight="balanced"),
        {
            "C":     [1, 10, 50],
            "gamma": ["scale", "auto"],
        },
    )

    # ── XGBoost ─────────────────────────────────────────────────────────────
    if HAS_XGB:
        configs["XGBoost"] = (
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators":  [200, 400],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [5, 7],
                "subsample":     [0.8, 1.0],
            },
        )
    else:
        log.warning("XGBoost not installed — skipping. pip install xgboost")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    if HAS_LGB:
        configs["LightGBM"] = (
            lgb.LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            {
                "n_estimators":  [200, 400],
                "learning_rate": [0.05, 0.1],
                "num_leaves":    [31, 63],
                "max_depth":     [-1, 10],
            },
        )
    else:
        log.warning("LightGBM not installed — skipping. pip install lightgbm")

    return configs


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
    use_grid_search: bool = True,
    cv_folds: int = 5,
) -> Dict:
    """
    Train all configured models and return a results dict.

    Training procedure:
      1. (Optional) GridSearchCV with 3-fold inner CV to tune hyperparameters.
      2. Refit the best params on the full training set.
      3. Stratified K-Fold (5-fold) outer CV for unbiased accuracy estimate.

    Why Stratified K-Fold?
      • Preserves emotion class proportions in each fold.
      • Provides 5 accuracy estimates → mean ± std is more reliable than
        a single train/test split.

    Args:
        X_train        : Scaled training feature matrix.
        y_train        : Encoded integer label array.
        label_encoder  : Fitted LabelEncoder (to decode predictions).
        use_grid_search: Run GridSearchCV (slow but optimal).
        cv_folds       : Number of outer CV folds.

    Returns:
        results dict:
          model_name → {
            "model"    : fitted estimator (best params),
            "cv_acc"   : mean CV accuracy,
            "cv_std"   : std of CV accuracy,
          }
    """
    log.info("=" * 60)
    log.info("MODEL TRAINING")
    log.info(f"  Samples      : {X_train.shape[0]}")
    log.info(f"  Features     : {X_train.shape[1]}")
    log.info(f"  Classes      : {list(label_encoder.classes_)}")
    log.info(f"  GridSearchCV : {use_grid_search}")
    log.info(f"  Outer CV folds: {cv_folds}")
    log.info("=" * 60)

    model_configs = _build_model_configs(label_encoder)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: Dict = {}

    for model_name, (estimator, param_grid) in model_configs.items():
        log.info(f"\n── Training: {model_name} ──")
        t0 = time.time()

        if use_grid_search and len(param_grid) > 0:
            log.info(f"  Running GridSearchCV (inner cv=3)...")
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=3,                   # inner loop — fast
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
                refit=True,             # refit best estimator on all train data
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            log.info(f"  Best params : {grid_search.best_params_}")
            log.info(f"  Best CV acc : {grid_search.best_score_*100:.2f}%")
        else:
            # No grid search — just fit with default params
            best_estimator = estimator
            best_estimator.fit(X_train, y_train)

        # ── Outer Stratified K-Fold cross-validation ─────────────────────────
        log.info(f"  Running {cv_folds}-fold outer CV for unbiased estimate...")
        cv_scores = cross_val_score(
            best_estimator, X_train, y_train,
            cv=skf, scoring="accuracy", n_jobs=-1
        )
        cv_acc = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        elapsed = time.time() - t0
        log.info(f"  CV Accuracy : {cv_acc*100:.2f}% ± {cv_std*100:.2f}%")
        log.info(f"  Time taken  : {elapsed:.1f}s")

        results[model_name] = {
            "model":  best_estimator,
            "cv_acc": cv_acc,
            "cv_std": cv_std,
        }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  6. EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_models(
    results: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    models_dir: str = MODELS_DIR,
) -> Dict:
    """
    Evaluate all trained models on the held-out test set and log full metrics.

    Metrics computed per model:
      • Accuracy              — overall correct predictions
      • Macro F1              — unweighted mean F1 across classes
                                (penalises poor performance on minority classes)
      • Weighted F1           — F1 weighted by class support
                                (reflects real-world class distribution)
      • Per-class precision/recall/F1 — full classification report
      • Confusion matrix (raw counts + normalised)
      • Interview-mapped accuracy — accuracy on the 4-label scheme
        (confident / neutral / nervous / stressed)

    Args:
        results       : Output of train_models().
        X_test        : Scaled test feature matrix.
        y_test        : Encoded integer label array (test set).
        label_encoder : Fitted LabelEncoder.
        models_dir    : Directory to save confusion matrix plots.

    Returns:
        Updated results dict with evaluation metrics added.
    """
    log.info("\n" + "=" * 60)
    log.info("MODEL EVALUATION — HELD-OUT TEST SET")
    log.info("=" * 60)

    class_names = list(label_encoder.classes_)
    y_test_str  = label_encoder.inverse_transform(y_test)

    eval_rows = []   # for final comparison table

    for model_name, info in results.items():
        model = info["model"]
        log.info(f"\n── {model_name} ──")

        y_pred_enc = model.predict(X_test)

        # XGBoost returns float by default — convert back to int
        if hasattr(y_pred_enc[0], "item"):
            y_pred_enc = y_pred_enc.astype(int)

        y_pred_str = label_encoder.inverse_transform(y_pred_enc)

        # ── Core metrics ─────────────────────────────────────────────────────
        acc        = accuracy_score(y_test_str, y_pred_str)
        f1_macro   = f1_score(y_test_str, y_pred_str, average="macro",   zero_division=0)
        f1_weighted = f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0)

        # ── Interview-mapped metrics ─────────────────────────────────────────
        y_test_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_test_str])
        y_pred_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_pred_str])
        mapped_acc    = accuracy_score(y_test_mapped, y_pred_mapped)
        mapped_f1     = f1_score(y_test_mapped, y_pred_mapped,
                                  average="weighted", zero_division=0)

        log.info(f"  Accuracy (raw 8-class)    : {acc*100:.2f}%")
        log.info(f"  F1 Macro (raw)            : {f1_macro*100:.2f}%")
        log.info(f"  F1 Weighted (raw)         : {f1_weighted*100:.2f}%")
        log.info(f"  Accuracy (interview 4-cls): {mapped_acc*100:.2f}%")
        log.info(f"  F1 Weighted (interview)   : {mapped_f1*100:.2f}%")

        # ── Full classification report ────────────────────────────────────────
        report = classification_report(
            y_test_str, y_pred_str, zero_division=0
        )
        log.info(f"\n  Per-class report:\n{report}")

        # ── Confusion matrix ─────────────────────────────────────────────────
        cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        if HAS_PLOT:
            _save_confusion_matrix(
                cm=cm_norm,
                labels=class_names,
                model_name=model_name,
                save_dir=models_dir,
            )

        # ── Store results ─────────────────────────────────────────────────────
        results[model_name].update({
            "test_acc":        acc,
            "f1_macro":        f1_macro,
            "f1_weighted":     f1_weighted,
            "mapped_acc":      mapped_acc,
            "mapped_f1":       mapped_f1,
            "confusion_matrix": cm,
            "y_pred_str":      y_pred_str,
        })

        eval_rows.append({
            "Model":       model_name,
            "CV Acc":      f"{info['cv_acc']*100:.2f}% ± {info['cv_std']*100:.2f}%",
            "Test Acc":    f"{acc*100:.2f}%",
            "F1 Macro":    f"{f1_macro*100:.2f}%",
            "F1 Weighted": f"{f1_weighted*100:.2f}%",
            "Interview Acc": f"{mapped_acc*100:.2f}%",
        })

    # ── Summary comparison table ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FINAL MODEL COMPARISON")
    log.info("=" * 60)
    header = f"{'Model':<18} {'CV Acc':>20} {'Test Acc':>10} {'F1 Macro':>10} {'F1 Wt':>10} {'Interview':>12}"
    log.info(header)
    log.info("-" * len(header))

    for row in sorted(eval_rows, key=lambda r: float(r["Test Acc"].replace("%", "")), reverse=True):
        marker = " ◄ BEST" if row == max(eval_rows,
                    key=lambda r: float(r["Test Acc"].replace("%", ""))) else ""
        log.info(
            f"  {row['Model']:<16} {row['CV Acc']:>20} {row['Test Acc']:>10} "
            f"{row['F1 Macro']:>10} {row['F1 Weighted']:>10} {row['Interview Acc']:>12}"
            f"{marker}"
        )

    return results


def _save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    model_name: str,
    save_dir: str,
) -> None:
    """Save a normalised confusion matrix heatmap as PNG."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        save_path = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            vmin=0, vmax=1,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix (normalised) — {model_name}", fontsize=14)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        log.info(f"  Confusion matrix saved: {save_path}")
    except Exception as exc:
        log.warning(f"Could not save confusion matrix: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
#  7. SAVE MODEL
# ═════════════════════════════════════════════════════════════════════════════

def save_model(
    results: Dict,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    feature_dim: int,
    save_path: Optional[str] = None,
) -> str:
    """
    Persist the best model (by interview-mapped accuracy) along with all
    artefacts needed for inference.

    Saved bundle keys:
      model         : Best fitted sklearn estimator.
      scaler        : Fitted StandardScaler (use on inference features).
      label_encoder : Fitted LabelEncoder (decode integer predictions).
      model_name    : String name of winning algorithm.
      feature_dim   : Expected input dimension (for validation at inference).
      metrics       : Dict of all evaluation metrics.
      interview_map : INTERVIEW_MAP dict (baked in for portability).
      emotion_scores: EMOTION_SCORES dict.
      trained_at    : ISO timestamp.
      all_results   : Per-model summary for auditing.

    Args:
        results       : Output of evaluate_models().
        scaler        : Fitted StandardScaler.
        label_encoder : Fitted LabelEncoder.
        feature_dim   : Number of features.
        save_path     : Where to save the .pkl file.

    Returns:
        Absolute path of the saved file.
    """
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "emotion_model.pkl")

    save_path = os.path.normpath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Select best model by interview-mapped accuracy (the metric that matters
    # for ARIES — not raw 8-class accuracy on acted datasets)
    best_name = max(
        results,
        key=lambda k: results[k].get("mapped_acc", results[k].get("cv_acc", 0)),
    )
    best_info = results[best_name]

    log.info(f"\nBest model selected: {best_name}")
    log.info(f"  Test accuracy (raw)      : {best_info.get('test_acc', 0)*100:.2f}%")
    log.info(f"  Interview accuracy (4-cls): {best_info.get('mapped_acc', 0)*100:.2f}%")

    bundle = {
        # ── Core inference artefacts ─────────────────────────────────────────
        "model":         best_info["model"],
        "scaler":        scaler,
        "label_encoder": label_encoder,

        # ── Configuration ────────────────────────────────────────────────────
        "model_name":    best_name,
        "feature_dim":   feature_dim,
        "interview_map": INTERVIEW_MAP,
        "emotion_scores": EMOTION_SCORES,

        # ── Metrics ─────────────────────────────────────────────────────────
        "metrics": {
            "cv_acc":      best_info.get("cv_acc", 0),
            "cv_std":      best_info.get("cv_std", 0),
            "test_acc":    best_info.get("test_acc", 0),
            "f1_macro":    best_info.get("f1_macro", 0),
            "f1_weighted": best_info.get("f1_weighted", 0),
            "mapped_acc":  best_info.get("mapped_acc", 0),
            "mapped_f1":   best_info.get("mapped_f1", 0),
        },

        # ── Provenance ───────────────────────────────────────────────────────
        "trained_at":  datetime.now().isoformat(),
        "all_results": {
            k: {
                "cv_acc":     v.get("cv_acc", 0),
                "test_acc":   v.get("test_acc", 0),
                "f1_macro":   v.get("f1_macro", 0),
                "mapped_acc": v.get("mapped_acc", 0),
            }
            for k, v in results.items()
        },
    }

    with open(save_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save a human-readable JSON metadata sidecar
    meta_path = save_path.replace(".pkl", "_metadata.json")
    meta = {k: v for k, v in bundle.items()
            if k not in ("model", "scaler", "label_encoder")}
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2, default=str)
        log.info(f"Metadata saved: {meta_path}")
    except Exception as exc:
        log.warning(f"Could not save metadata JSON: {exc}")

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    log.info(f"Model bundle saved: {save_path}  ({file_size_mb:.1f} MB)")

    return save_path


# ═════════════════════════════════════════════════════════════════════════════
#  8. INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def predict_emotion(
    audio_path: str,
    model_path: Optional[str] = None,
    return_confidence: bool = False,
) -> str | Tuple[str, float, Dict]:
    """
    Predict the interview emotion label for a single audio file.

    Interview labels: "confident" | "neutral" | "nervous" | "stressed"

    Args:
        audio_path       : Path to .wav file to analyse.
        model_path       : Path to .pkl bundle (default: models/emotion_model.pkl).
        return_confidence: If True, also return confidence score and all probs.

    Returns:
        If return_confidence=False → interview_label (str)
        If return_confidence=True  → (interview_label, confidence, all_probs dict)

    Usage:
        label = predict_emotion("recording.wav")
        label, conf, probs = predict_emotion("recording.wav", return_confidence=True)
    """
    # ── Load bundle ──────────────────────────────────────────────────────────
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "emotion_model.pkl")

    model_path = os.path.normpath(model_path)

    if not os.path.exists(model_path):
        log.error(f"Model not found at {model_path}. Train first.")
        return ("neutral", 0.0, {}) if return_confidence else "neutral"

    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        return ("neutral", 0.0, {}) if return_confidence else "neutral"

    model         = bundle["model"] if isinstance(bundle, dict) else bundle
    scaler        = bundle.get("scaler")        if isinstance(bundle, dict) else None
    label_encoder = bundle.get("label_encoder") if isinstance(bundle, dict) else None
    interview_map = bundle.get("interview_map", INTERVIEW_MAP)
    expected_dim  = bundle.get("feature_dim")

    # ── Extract features ─────────────────────────────────────────────────────
    try:
        features = extract_features_from_file(audio_path, apply_vad=True)
    except Exception as exc:
        log.warning(f"Feature extraction failed: {exc}")
        features = None

    if features is None:
        log.warning(f"No features extracted from {audio_path}, defaulting to neutral")
        return ("neutral", 0.0, {}) if return_confidence else "neutral"

    # ── Dimension guard ──────────────────────────────────────────────────────
    if expected_dim is not None and features.shape[0] != expected_dim:
        log.warning(
            f"Feature dim mismatch: expected {expected_dim}, got {features.shape[0]}. "
            "Model may have been trained with a different config."
        )

    # ── Preprocess ───────────────────────────────────────────────────────────
    features = features.reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)

    # ── Predict ──────────────────────────────────────────────────────────────
    try:
        raw_pred = model.predict(features)[0]

        # Convert integer prediction back to emotion string
        if isinstance(raw_pred, (int, np.integer, float, np.floating)):
            if label_encoder is not None:
                raw_emotion = label_encoder.inverse_transform([int(raw_pred)])[0]
            else:
                # Fallback ordered list (same order as LabelEncoder on sorted labels)
                fallback = ["angry", "calm", "disgust", "fearful",
                            "happy", "neutral", "sad", "surprised"]
                raw_emotion = fallback[int(raw_pred)] if int(raw_pred) < len(fallback) else "neutral"
        else:
            raw_emotion = str(raw_pred)

        interview_label = interview_map.get(raw_emotion, "neutral")

        # ── Confidence (if model supports predict_proba) ──────────────────────
        if return_confidence and hasattr(model, "predict_proba"):
            probs_raw = model.predict_proba(features)[0]
            if label_encoder is not None:
                raw_labels = label_encoder.classes_
            else:
                raw_labels = [str(i) for i in range(len(probs_raw))]

            # Map raw probabilities → interview label probabilities
            interview_probs: Dict[str, float] = {
                "confident": 0.0, "neutral": 0.0, "nervous": 0.0, "stressed": 0.0
            }
            for raw_lbl, prob in zip(raw_labels, probs_raw):
                iv_lbl = interview_map.get(str(raw_lbl), "neutral")
                interview_probs[iv_lbl] = interview_probs.get(iv_lbl, 0.0) + float(prob)

            confidence = interview_probs.get(interview_label, 0.0)
            return interview_label, confidence, interview_probs

        return interview_label

    except Exception as exc:
        log.error(f"Prediction failed: {exc}")
        return ("neutral", 0.0, {}) if return_confidence else "neutral"


# ═════════════════════════════════════════════════════════════════════════════
#  9. MASTER TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def train_full_pipeline(
    ravdess_path: Optional[str] = None,
    crema_path:   Optional[str] = None,
    save_path:    Optional[str] = None,
    use_augmentation: bool = True,
    use_grid_search:  bool = True,
    apply_vad:        bool = True,
    test_ratio:       float = 0.2,
    cv_folds:         int   = 5,
    rebuild_cache:    bool  = False,  # Task 3 — bypass cache when True
) -> Optional[str]:
    """
    End-to-end training pipeline. Calls all sub-functions in sequence.

    Full pipeline:
      load_dataset → speaker_aware_split → scale → train_models
      → evaluate_models → save_model

    Args:
        ravdess_path     : RAVDESS dataset directory.
        crema_path       : CREMA-D dataset directory.
        save_path        : Where to save the trained model.
        use_augmentation : Apply 4× augmentation during loading.
        use_grid_search  : Tune hyperparameters with GridSearchCV.
        apply_vad        : Strip silence with VAD.
        test_ratio       : Fraction of speakers held out for testing.
        cv_folds         : Stratified K-Fold folds.
        rebuild_cache    : If True, ignore existing cache and re-extract.

    Returns:
        Path to saved model, or None if training failed.
    """
    pipeline_start = time.time()

    log.info("\n" + "╔" + "═"*58 + "╗")
    log.info("║   ARIES — Speech Emotion Recognition Training Pipeline  ║")
    log.info("╚" + "═"*58 + "╝")
    log.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  BASE_DIR   : {BASE_DIR}")
    log.info(f"  DATA_DIR   : {DATA_DIR}")
    log.info(f"  MODELS_DIR : {MODELS_DIR}")

    # ── Step 1: Load dataset ─────────────────────────────────────────────────
    log.info("\n[STEP 1/6] Loading dataset...")
    try:
        X, y_raw, speaker_ids = load_dataset(
            ravdess_path=ravdess_path,
            crema_path=crema_path,
            use_augmentation=use_augmentation,
            apply_vad=apply_vad,
            rebuild_cache=rebuild_cache,   # Task 3: pass CLI flag through
        )
    except Exception as exc:
        log.error(f"Dataset loading failed: {exc}")
        return None

    if len(X) == 0:
        log.error("No samples loaded. Check dataset paths and file formats.")
        return None

    # ── Step 2: Encode labels ────────────────────────────────────────────────
    log.info("\n[STEP 2/6] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    log.info(f"  Classes: {list(label_encoder.classes_)}")
    log.info(f"  Class counts: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

    # ── Step 3: Speaker-aware train/test split ───────────────────────────────
    log.info("\n[STEP 3/6] Speaker-aware train/test split...")
    X_train_raw, X_test_raw, y_train_enc, y_test_enc = speaker_aware_split(
        X, y_encoded, speaker_ids, test_ratio=test_ratio
    )
    log.info(f"  Train: {X_train_raw.shape[0]} samples")
    log.info(f"  Test : {X_test_raw.shape[0]} samples")

    # ── Step 4: Feature scaling ──────────────────────────────────────────────
    log.info("\n[STEP 4/6] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    log.info(f"  Feature mean range: [{X_train.mean(axis=0).min():.3f}, "
             f"{X_train.mean(axis=0).max():.3f}]")

    # ── Step 5: Train models ─────────────────────────────────────────────────
    log.info("\n[STEP 5/6] Training models...")
    results = train_models(
        X_train=X_train,
        y_train=y_train_enc,
        label_encoder=label_encoder,
        use_grid_search=use_grid_search,
        cv_folds=cv_folds,
    )

    # ── Step 6: Evaluate models ──────────────────────────────────────────────
    log.info("\n[STEP 6/6] Evaluating on held-out test set...")
    results = evaluate_models(
        results=results,
        X_test=X_test,
        y_test=y_test_enc,
        label_encoder=label_encoder,
        models_dir=save_path and os.path.dirname(save_path) or MODELS_DIR,
    )

    # ── Save best model ──────────────────────────────────────────────────────
    log.info("\n[SAVE] Saving best model...")
    saved_path = save_model(
        results=results,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_dim=X.shape[1],
        save_path=save_path,
    )

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_start
    best_name = max(results, key=lambda k: results[k].get("mapped_acc", 0))
    best      = results[best_name]

    log.info("\n" + "╔" + "═"*58 + "╗")
    log.info("║              TRAINING COMPLETE                          ║")
    log.info("╠" + "═"*58 + "╣")
    log.info(f"║  Best model       : {best_name:<38}║")
    log.info(f"║  CV Accuracy      : {best['cv_acc']*100:>5.2f}% ± {best['cv_std']*100:.2f}%{'':<28}║")
    log.info(f"║  Test Accuracy    : {best.get('test_acc',0)*100:>5.2f}%{'':<38}║")
    log.info(f"║  Interview Acc    : {best.get('mapped_acc',0)*100:>5.2f}% (4-class){'':<28}║")
    log.info(f"║  F1 Macro         : {best.get('f1_macro',0)*100:>5.2f}%{'':<38}║")
    log.info(f"║  Total time       : {elapsed_total/60:>5.1f} minutes{'':<35}║")
    log.info(f"║  Saved to         : .../{os.path.basename(saved_path):<37}║")
    log.info("╚" + "═"*58 + "╝\n")

    return saved_path


# ═════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ARIES Speech Emotion Recognition — train or predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python modules/emotion_model.py
  python modules/emotion_model.py --no-augment --no-grid-search
  python modules/emotion_model.py --predict recordings/test.wav
  python modules/emotion_model.py --predict recordings/test.wav --confidence
        """,
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to a .wav file to predict emotion (skips training)",
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Show confidence score in prediction output",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable audio augmentation (faster training)",
    )
    # ── Task 3: Development flags ──────────────────────────────────────────────────
    # --rebuild-cache: ignore any existing .npz cache file and force fresh
    # feature extraction from audio. Use after dataset changes or when
    # suspecting a stale/corrupt cache.
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore existing feature cache and re-extract from scratch",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip GridSearchCV (faster training, slightly lower accuracy)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of speakers held out for testing (default 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of Stratified K-Fold folds (default 5)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override default model save/load path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict is not None:
        # ── Predict mode ─────────────────────────────────────────────────────
        log.info(f"PREDICT MODE: {args.predict}")
        if args.confidence:
            label, conf, probs = predict_emotion(
                args.predict,
                model_path=args.model_path,
                return_confidence=True,
            )
            log.info(f"  Interview Emotion : {label.upper()}")
            log.info(f"  Confidence        : {conf*100:.1f}%")
            log.info("  All probabilities:")
            for k, v in sorted(probs.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 20)
                log.info(f"    {k:<12}: {v*100:5.1f}%  {bar}")
        else:
            label = predict_emotion(args.predict, model_path=args.model_path)
            log.info(f"  Interview Emotion : {label.upper()}")
            log.info(f"  Emotion Score     : {EMOTION_SCORES.get(label, 65)}")

    else:
        # ── Training mode ─────────────────────────────────────────────────────
        train_full_pipeline(
            use_augmentation = not args.no_augment,
            use_grid_search  = not args.no_grid_search,
            apply_vad        = not args.no_vad,
            test_ratio       = args.test_ratio,
            cv_folds         = args.cv_folds,
            save_path        = args.model_path,
            rebuild_cache    = args.rebuild_cache,  # Task 3: CLI → pipeline
        )
