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

# ── Emotion mappings ──
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

CREMA_EMOTIONS = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

# Map raw emotions → interview labels (applied AFTER training)
INTERVIEW_MAP = {
    'happy': 'confident',    'surprised': 'confident',
    'calm': 'neutral',       'neutral': 'neutral',
    'fearful': 'nervous',    'sad': 'stressed',
    'angry': 'stressed',     'disgust': 'stressed'
}

def extract_features(file_path):
    """Extract 101 audio features from a wav file"""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        if len(y) == 0:
            return None

        features = []

        # MFCC mean + std (80 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Chroma (12 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # Mel spectrogram mean + std (2 features)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.append(np.mean(mel))
        features.append(np.std(mel))

        # Zero Crossing Rate mean + std (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # RMS Energy mean + std (2 features)
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # Spectral Centroid mean (1 feature)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))

        # Spectral Rolloff mean (1 feature)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))

        # Pitch mean (1 feature)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        features.append(np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0)

        return np.array(features[:101])
    except Exception:
        return None


def load_ravdess(data_path):
    """Load RAVDESS dataset"""
    X, y = [], []
    if not os.path.exists(data_path):
        print(f"  ⚠️  RAVDESS not found at {data_path}")
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
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTIONS.get(emotion_code)
            if not emotion:
                continue
            features = extract_features(os.path.join(actor_path, fname))
            if features is not None:
                X.append(features)
                y.append(emotion)

    print(f"  ✅ RAVDESS: {len(X)} samples loaded")
    return X, y


def load_crema(data_path):
    """Load CREMA-D dataset"""
    X, y = [], []
    audio_path = os.path.join(data_path, 'AudioWAV')
    if not os.path.exists(audio_path):
        print(f"  ⚠️  CREMA-D AudioWAV not found at {audio_path}")
        return X, y

    files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print(f"  Loading {len(files)} CREMA-D files...")

    for i, fname in enumerate(files):
        if i % 1000 == 0 and i > 0:
            print(f"    Processed {i}/{len(files)}...")
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion = CREMA_EMOTIONS.get(emotion_code)
        if not emotion:
            continue
        features = extract_features(os.path.join(audio_path, fname))
        if features is not None:
            X.append(features)
            y.append(emotion)

    print(f"  ✅ CREMA-D: {len(X)} samples loaded")
    return X, y


def load_dataset(ravdess_path="data/Audio_Speech_Actors_01-24",
                 crema_path="data/CREMA-D"):
    """Load and combine both datasets"""
    print("Loading RAVDESS...")
    X_r, y_r = load_ravdess(ravdess_path)
    print("Loading CREMA-D...")
    X_c, y_c = load_crema(crema_path)

    X = X_r + X_c
    y = y_r + y_c
    print(f"\n  Combined: {len(X)} total samples")
    return np.array(X), np.array(y)


def train_model(ravdess_path="data/Audio_Speech_Actors_01-24",
                crema_path="data/CREMA-D",
                save_path="models/emotion_model.pkl"):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report

    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("  ⚠️  XGBoost not installed, skipping")

    print("\n" + "="*55)
    print("  TRAINING SPEECH EMOTION MODEL")
    print("  Dataset: RAVDESS + CREMA-D")
    print("="*55)

    X, y = load_dataset(ravdess_path, crema_path)

    if len(X) == 0:
        print("❌ No data loaded. Check dataset paths.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n  Total samples:    {len(X)}")
    print(f"  Features:         {X.shape[1]}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Emotion classes:  {sorted(set(y))}")

    results = {}

    # ── Random Forest ──
    print("\n  Training Random Forest (300 trees)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = (rf, rf_acc, rf_pred)
    print(f"  RF Raw Accuracy:  {rf_acc*100:.2f}%")

    # ── SVM ──
    print("\n  Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train_s, y_train)
    svm_pred = svm.predict(X_test_s)
    svm_acc = accuracy_score(y_test, svm_pred)
    results['SVM'] = (svm, svm_acc, svm_pred)
    print(f"  SVM Raw Accuracy: {svm_acc*100:.2f}%")

    # ── MLP Neural Network ──
    print("\n  Training MLP Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                        max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train_s, y_train)
    mlp_pred = mlp.predict(X_test_s)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    results['MLP Neural Network'] = (mlp, mlp_acc, mlp_pred)
    print(f"  MLP Raw Accuracy: {mlp_acc*100:.2f}%")

    # ── XGBoost ──
    if HAS_XGB:
        print("\n  Training XGBoost...")
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.1,
                            random_state=42, eval_metric='mlogloss',
                            verbosity=0)
        xgb.fit(X_train_s, y_train_enc)
        xgb_pred_enc = xgb.predict(X_test_s)
        xgb_pred = le.inverse_transform(xgb_pred_enc)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = (xgb, xgb_acc, xgb_pred)
        print(f"  XGB Raw Accuracy: {xgb_acc*100:.2f}%")

    # ── Select best model ──
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_pred = results[best_name]

    # Map to interview labels
    y_test_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in y_test])
    best_pred_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in best_pred])
    mapped_acc = accuracy_score(y_test_mapped, best_pred_mapped)

    print("\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)
    for name, (_, acc, pred) in results.items():
        pred_mapped = np.array([INTERVIEW_MAP.get(e, e) for e in pred])
        m_acc = accuracy_score(y_test_mapped, pred_mapped)
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<22} Raw: {acc*100:.2f}%  Mapped: {m_acc*100:.2f}%{marker}")

    print(f"\n  ✅ Best Model:     {best_name}")
    print(f"  ✅ Raw Accuracy:   {best_acc*100:.2f}%")
    print(f"  ✅ Interview Acc:  {mapped_acc*100:.2f}%")

    print(f"\n  Classification Report (Interview Labels):")
    print(classification_report(y_test_mapped, best_pred_mapped))

    # Save
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

    print(f"\n  ✅ Model saved to {save_path}")
    print("="*55)


def predict_emotion(audio_path):
    """Predict interview emotion from audio file"""
    try:
        with open("models/emotion_model.pkl", "rb") as f:
            saved = pickle.load(f)

        model = saved['model'] if isinstance(saved, dict) else saved
        scaler = saved.get('scaler') if isinstance(saved, dict) else None

        features = extract_features(audio_path)
        if features is None:
            return "neutral"

        features = features.reshape(1, -1)
        if scaler:
            features = scaler.transform(features)

        raw_emotion = model.predict(features)[0]

        # Handle XGBoost label encoding
        if isinstance(raw_emotion, (int, np.integer)):
            emotions_list = ['angry', 'calm', 'disgust', 'fearful',
                           'happy', 'neutral', 'sad', 'surprised']
            raw_emotion = emotions_list[raw_emotion] if raw_emotion < len(emotions_list) else 'neutral'

        return INTERVIEW_MAP.get(raw_emotion, 'neutral')
    except Exception as e:
        return "neutral"


if __name__ == "__main__":
    train_model()
