"""
Speech Emotion Recognition
Trains on RAVDESS + CREMA-D combined
Algorithms: Random Forest, SVM, XGBoost, MLP — auto-selects best
"""

import os
import numpy as np
import pickle
import librosa
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Resolve absolute paths regardless of where the script is called from.
# modules/emotion_model.py  →  MODULE_DIR = …/modules
#                           →  BASE_DIR   = …/interview_evaluator  (project root)
#                           →  DATA_DIR   = …/interview_evaluator/data
# ---------------------------------------------------------------------------
MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(MODULE_DIR, ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

CREMA_EMOTIONS = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

INTERVIEW_MAP = {
    'happy': 'confident',    'surprised': 'confident',
    'calm': 'neutral',       'neutral': 'neutral',
    'fearful': 'nervous',    'sad': 'stressed',
    'angry': 'stressed',     'disgust': 'stressed'
}

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        if len(y) == 0:
            return None
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.append(np.mean(mel))
        features.append(np.std(mel))
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        features.append(np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0)
        return np.array(features[:101])
    except Exception:
        return None

def load_ravdess(data_path):
    X, y = [], []
    if not os.path.exists(data_path):
        print(f"  RAVDESS not found at {data_path}")
        return X, y
    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith('.wav'):
                continue
            parts = fname.replace('.wav', '').split('-')
            if len(parts) < 3:
                continue
            emotion = RAVDESS_EMOTIONS.get(parts[2])
            if not emotion:
                continue
            features = extract_features(os.path.join(actor_path, fname))
            if features is not None:
                X.append(features)
                y.append(emotion)
    print(f"  RAVDESS: {len(X)} samples loaded")
    return X, y

def load_crema(data_path):
    X, y = [], []
    audio_path = os.path.join(data_path, 'AudioWAV')
    if not os.path.exists(audio_path):
        print(f"  CREMA-D not found at {audio_path}")
        return X, y
    files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print(f"  Loading {len(files)} CREMA-D files...")
    for i, fname in enumerate(files):
        if i % 1000 == 0 and i > 0:
            print(f"    Processed {i}/{len(files)}...")
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        emotion = CREMA_EMOTIONS.get(parts[2])
        if not emotion:
            continue
        features = extract_features(os.path.join(audio_path, fname))
        if features is not None:
            X.append(features)
            y.append(emotion)
    print(f"  CREMA-D: {len(X)} samples loaded")
    return X, y

def load_dataset(
    ravdess_path=None,
    crema_path=None,
):
    # Default to absolute paths derived from this file's location
    if ravdess_path is None:
        ravdess_path = os.path.join(DATA_DIR, "Audio_Speech_Actors_01-24")
    if crema_path is None:
        crema_path = os.path.join(DATA_DIR, "CREMA-D")

    # Normalise for Windows (forward-slash → backslash where needed)
    ravdess_path = os.path.normpath(ravdess_path)
    crema_path   = os.path.normpath(crema_path)

    # ── Debug: confirm resolved paths ──────────────────────────────────────
    print("\n  [PATH DEBUG]")
    print(f"    BASE_DIR     : {BASE_DIR}")
    print(f"    DATA_DIR     : {DATA_DIR}")
    print(f"    RAVDESS path : {ravdess_path}")
    print(f"      exists?    : {os.path.exists(ravdess_path)}")
    print(f"    CREMA-D path : {crema_path}")
    print(f"      exists?    : {os.path.exists(crema_path)}")
    print()

    print("Loading RAVDESS...")
    X_r, y_r = load_ravdess(ravdess_path)
    print("Loading CREMA-D...")
    X_c, y_c = load_crema(crema_path)
    X = X_r + X_c
    y = y_r + y_c
    print(f"  Combined: {len(X)} total samples")
    return np.array(X), np.array(y)

def train_model(
    ravdess_path=None,
    crema_path=None,
    save_path=None,
):
    # Resolve defaults using absolute paths
    if ravdess_path is None:
        ravdess_path = os.path.join(DATA_DIR, "Audio_Speech_Actors_01-24")
    if crema_path is None:
        crema_path = os.path.join(DATA_DIR, "CREMA-D")
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "emotion_model.pkl")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report

    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("  XGBoost not installed, skipping")

    print("\n" + "="*55)
    print("  TRAINING SPEECH EMOTION MODEL")
    print("  Dataset: RAVDESS + CREMA-D")
    print("="*55)

    X, y = load_dataset(ravdess_path, crema_path)

    if len(X) == 0:
        print("No data loaded. Check dataset paths.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\n  Total samples:    {len(X)}")
    print(f"  Features:         {X.shape[1]}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    results = {}

    # Random Forest
    print("\n  Training Random Forest (300 trees)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    results['Random Forest'] = (rf, accuracy_score(y_test, rf_pred), rf_pred)
    print(f"  RF Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}%")

    # SVM
    print("\n  Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train_s, y_train)
    svm_pred = svm.predict(X_test_s)
    results['SVM'] = (svm, accuracy_score(y_test, svm_pred), svm_pred)
    print(f"  SVM Accuracy: {accuracy_score(y_test, svm_pred)*100:.2f}%")

    # MLP — fixed for Python 3.13
    print("\n  Training MLP Neural Network...")
    try:
        le_mlp = LabelEncoder()
        y_train_enc = le_mlp.fit_transform(y_train).astype(np.float64)
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                            max_iter=500, random_state=42,
                            early_stopping=False)
        mlp.fit(X_train_s, y_train_enc)
        mlp_pred_enc = mlp.predict(X_test_s).astype(int)
        mlp_pred = le_mlp.inverse_transform(mlp_pred_enc)
        mlp_acc = accuracy_score(y_test, mlp_pred)
        results['MLP Neural Network'] = (mlp, mlp_acc, mlp_pred)
        print(f"  MLP Accuracy: {mlp_acc*100:.2f}%")
    except Exception as e:
        print(f"  MLP skipped: {e}")

    # XGBoost
    if HAS_XGB:
        print("\n  Training XGBoost...")
        le = LabelEncoder()
        y_train_xgb = le.fit_transform(y_train)
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.1,
                            random_state=42, eval_metric='mlogloss',
                            verbosity=0)
        xgb.fit(X_train_s, y_train_xgb)
        xgb_pred_enc = xgb.predict(X_test_s)
        xgb_pred = le.inverse_transform(xgb_pred_enc)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = (xgb, xgb_acc, xgb_pred)
        print(f"  XGB Accuracy: {xgb_acc*100:.2f}%")

    # Select best
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_pred = results[best_name]

    y_test_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_test])
    best_pred_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in best_pred])
    mapped_acc = accuracy_score(y_test_mapped, best_pred_mapped)

    print("\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)
    for name, (_, acc, pred) in results.items():
        pred_m = np.array([INTERVIEW_MAP.get(e, e) for e in pred])
        m_acc = accuracy_score(y_test_mapped, pred_m)
        marker = " BEST" if name == best_name else ""
        print(f"  {name:<22} Raw: {acc*100:.2f}%  Mapped: {m_acc*100:.2f}%{marker}")

    print(f"\n  Best Model:    {best_name}")
    print(f"  Raw Accuracy:  {best_acc*100:.2f}%")
    print(f"  Interview Acc: {mapped_acc*100:.2f}%")
    print(classification_report(y_test_mapped, best_pred_mapped))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'model_name': best_name,
            'raw_accuracy': best_acc,
            'mapped_accuracy': mapped_acc,
            'all_results': {k: v[1] for k, v in results.items()}
        }, f)
    print(f"\n  Model saved to {save_path}")
    print("="*55)

def predict_emotion(audio_path):
    try:
        model_path = os.path.join(MODELS_DIR, "emotion_model.pkl")
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model  = saved['model'] if isinstance(saved, dict) else saved
        scaler = saved.get('scaler') if isinstance(saved, dict) else None
        features = extract_features(audio_path)
        if features is None:
            return "neutral"
        features = features.reshape(1, -1)
        if scaler:
            features = scaler.transform(features)
        raw = model.predict(features)[0]
        if isinstance(raw, (int, np.integer)):
            elist = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
            raw = elist[raw] if raw < len(elist) else 'neutral'
        return INTERVIEW_MAP.get(raw, 'neutral')
    except Exception:
        return "neutral"

if __name__ == "__main__":
    train_model()