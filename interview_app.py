import json
import os
import random
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageStat
import cv2

# ── New architecture modules ────────────────────────────────────────────────
from modules.session_manager import (
    load_sessions, save_sessions, save_session_block,
    save_abandoned_session, build_student_summary, normalize_sessions,
)
from modules.ui_live_session import init_live_session_state, render_custom_question_input

SESSIONS_FILE = "models/sessions.json"
TEMP_AUDIO_FILE = "temp_recording.wav"
THEME = {
    "bg": "#07111f",
    "panel": "#101c30",
    "panel_alt": "#16243d",
    "line": "#263553",
    "text": "#e8eeff",
    "muted": "#9aa9c7",
    "primary": "#64d2ff",
    "primary_2": "#8b7bff",
    "success": "#4ade80",
    "warning": "#f59e0b",
    "danger": "#fb7185",
}
INTERVIEW_PRESETS = {
    "HR Screening": ["HR & General", "Behavioral", "Situational"],
    "Campus Placement": ["Campus Placement", "HR & General", "Behavioral"],
    "Software Engineer": ["CS Fundamentals", "System Design", "Behavioral"],
    "ML Engineer": ["Machine Learning", "Data Science", "Behavioral"],
    "Company Mix": ["Amazon (Leadership Principles)", "Google", "Microsoft", "Meta (Facebook)", "Apple"],
}


def configure_page():
    st.set_page_config(page_title="ARIES Interview Studio", page_icon="ARIES", layout="wide")


def install_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
            background:
                radial-gradient(circle at top left, rgba(100, 210, 255, 0.10), transparent 22%),
                radial-gradient(circle at top right, rgba(139, 123, 255, 0.14), transparent 26%),
                linear-gradient(180deg, #050b16 0%, {THEME["bg"]} 100%);
            color: {THEME["text"]};
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(5, 11, 22, 0.9);
            border-right: 1px solid rgba(38, 53, 83, 0.8);
        }}
        [data-testid="stSidebar"] * {{
            color: {THEME["text"]};
        }}
        .block-container {{
            max-width: 1380px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Manrope', sans-serif;
            letter-spacing: -0.03em;
        }}
        .hero {{
            padding: 1.6rem 1.8rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(22, 36, 61, 0.95), rgba(16, 28, 48, 0.92));
            border: 1px solid rgba(38, 53, 83, 0.8);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }}
        .hero-kicker {{
            color: {THEME["primary"]};
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.74rem;
            font-weight: 700;
        }}
        .hero-title {{
            font-size: 3rem;
            font-weight: 800;
            margin: 0.35rem 0 0;
        }}
        .hero-copy {{
            color: {THEME["muted"]};
            font-size: 1.05rem;
            max-width: 64rem;
            line-height: 1.7;
            margin-top: 0.85rem;
        }}
        .panel {{
            background: rgba(16, 28, 48, 0.92);
            border: 1px solid rgba(38, 53, 83, 0.8);
            border-radius: 26px;
            padding: 1.25rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.20);
        }}
        .question-panel {{
            position: relative;
            min-height: 280px;
            overflow: hidden;
        }}
        .question-panel::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 6px;
            height: 100%;
            background: linear-gradient(180deg, {THEME["primary"]}, {THEME["primary_2"]});
        }}
        .label {{
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            font-weight: 700;
            color: {THEME["muted"]};
        }}
        .question-title {{
            font-size: 1.8rem;
            line-height: 1.4;
            margin: 0.7rem 0 0.95rem;
            font-weight: 800;
        }}
        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }}
        .pill {{
            border-radius: 999px;
            padding: 0.5rem 0.82rem;
            background: rgba(22, 36, 61, 0.95);
            color: {THEME["text"]};
            font-size: 0.78rem;
        }}
        .stage-grid {{
            display: grid;
            grid-template-columns: 1fr 1.08fr;
            gap: 1rem;
            align-items: start;
        }}
        .voice-box {{
            min-height: 240px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(8, 15, 26, 0.95), rgba(16, 28, 48, 0.95));
            border: 1px solid rgba(38, 53, 83, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        .bars {{
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            opacity: 0.85;
        }}
        .bars span {{
            width: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, {THEME["primary"]}, {THEME["primary_2"]});
        }}
        .voice-center {{
            position: relative;
            z-index: 1;
            text-align: center;
        }}
        .voice-time {{
            font-size: 2.8rem;
            font-weight: 800;
            font-family: 'Manrope', sans-serif;
        }}
        .metric {{
            background: rgba(22, 36, 61, 0.92);
            border: 1px solid rgba(38, 53, 83, 0.8);
            border-radius: 22px;
            padding: 1rem;
            min-height: 145px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.15rem;
            font-weight: 800;
            font-family: 'Manrope', sans-serif;
            margin-top: 0.3rem;
        }}
        .score-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
        }}
        .feedback-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1px;
            background: rgba(38, 53, 83, 0.8);
            border-radius: 26px;
            overflow: hidden;
            border: 1px solid rgba(38, 53, 83, 0.8);
        }}
        .feedback-col {{
            background: rgba(16, 28, 48, 0.95);
            padding: 1.2rem 1.35rem;
        }}
        .timeline-card {{
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(22, 36, 61, 0.8);
            border: 1px solid rgba(38, 53, 83, 0.75);
            margin-bottom: 0.8rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            border: none;
            border-radius: 999px;
            padding: 0.78rem 1.15rem;
            font-weight: 700;
            background: linear-gradient(135deg, {THEME["primary_2"]}, #b794f6);
            color: #0b1020;
            box-shadow: 0 10px 30px rgba(139, 123, 255, 0.25);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            filter: brightness(1.05);
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stNumberInput input {{
            background: rgba(11, 18, 31, 0.95) !important;
            color: {THEME["text"]} !important;
            border-radius: 16px !important;
            border: 1px solid rgba(38, 53, 83, 0.85) !important;
        }}
        .stAlert {{
            background: rgba(16, 28, 48, 0.95);
            border: 1px solid rgba(38, 53, 83, 0.85);
            color: {THEME["text"]};
        }}
        /* ── Live Companion Styles ── */
        .companion-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.7rem 1.1rem;
            background: rgba(8, 14, 26, 0.97);
            border: 1px solid rgba(38, 53, 83, 0.9);
            border-radius: 18px;
            margin-bottom: 1rem;
        }}
        .companion-brand {{
            font-family: 'Manrope', sans-serif;
            font-size: 1.15rem;
            font-weight: 800;
            color: {THEME["primary"]};
        }}
        .live-dot {{
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #ff4a4a;
            box-shadow: 0 0 10px rgba(255,74,74,0.7);
            animation: blink 1.4s infinite;
            margin-right: 6px;
        }}
        @keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
        .companion-live-badge {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #ff4a4a;
        }}
        .score-pill-row {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
        }}
        .score-pill {{
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: rgba(22, 36, 61, 0.9);
            border: 1px solid rgba(38, 53, 83, 0.8);
        }}
        .companion-log-item {{
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            background: rgba(22, 36, 61, 0.75);
            border: 1px solid rgba(38, 53, 83, 0.7);
            margin-bottom: 0.55rem;
        }}
        .progress-row {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 0.3rem 0;
        }}
        .progress-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: {THEME["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            min-width: 4.5rem;
        }}
        .progress-track {{
            flex: 1;
            height: 6px;
            border-radius: 999px;
            background: rgba(38, 53, 83, 0.6);
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .progress-val {{
            font-size: 0.78rem;
            font-weight: 700;
            min-width: 2.5rem;
            text-align: right;
        }}
        .tip-banner {{
            padding: 0.65rem 0.9rem;
            border-radius: 14px;
            background: rgba(100, 210, 255, 0.07);
            border-left: 3px solid {THEME["primary"]};
            font-size: 0.84rem;
            line-height: 1.6;
            margin-top: 0.7rem;
        }}
        .stat-badge {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            padding: 0.6rem 0.9rem;
            border-radius: 14px;
            background: rgba(22, 36, 61, 0.9);
            border: 1px solid rgba(38, 53, 83, 0.8);
            min-width: 5rem;
        }}
        .stat-val {{ font-family: 'Manrope', sans-serif; font-size: 1.55rem; font-weight: 800; }}
        .stat-lbl {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.12em; color: {THEME["muted"]}; margin-top: 0.2rem; }}
        @media (max-width: 980px) {{
            .stage-grid, .feedback-grid, .score-grid {{
                grid-template-columns: 1fr;
            }}
            .hero-title {{
                font-size: 2.2rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_questions():
    from modules.questions_bank import QUESTIONS

    return QUESTIONS


@st.cache_resource
def load_models():
    from modules.nlp_evaluator import evaluate_answer
    from modules.fluency_analyzer import analyze_fluency
    from modules.scoring_engine import calculate_esi, generate_report
    from interview_evaluator.modules.emotion_model_legacy import predict_emotion

    return evaluate_answer, analyze_fluency, calculate_esi, generate_report, predict_emotion


# normalize_sessions, load_sessions, save_sessions are imported from session_manager.
# The local stubs below are kept so any in-file call sites continue to work
# without modification — they simply delegate to the module.


def ensure_state():
    defaults = {
        "workspace": "Student",
        "screen": "Interview",
        "candidate_name": "",
        "session_started": False,
        "interview_mode": "HR Screening",
        "selected_categories": [],
        "question_pool": [],
        "current_question_index": 0,
        "current_transcript": "",
        "current_duration": 30,
        "current_recording_seconds": 0,
        "question_results": [],
        "final_report_ready": False,
        "camera_enabled": False,
        "camera_show_preview": True,
        "camera_capture_mode": "Pre-interview check",
        "camera_last_check_label": "Not captured",
        "camera_visibility_status": "No camera frame captured yet",
        "camera_visibility_score": 0,
        # Live Companion state
        "companion_active": False,
        "companion_log": [],
        "companion_transcript": "",
        "companion_question": "",
        "companion_category": "HR & General",
        "companion_duration": 30,
        "companion_saved": False,
        "companion_ai_result": None,
        "companion_question_val": "",   # shared question text for analysis (both modes)
        # Practice mode question picker state
        "practice_generated_pool": [],   # auto-generated questions shown in picker
        "practice_selected_qs": [],      # questions user ticked from generated pool
        "practice_extra_qs": [],         # user-typed extra questions
        "practice_pool_built": False,    # True after "Generate" clicked
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "candidate_name_setup" not in st.session_state:
        st.session_state.candidate_name_setup = st.session_state.candidate_name
    if "screen_sidebar" not in st.session_state:
        st.session_state.screen_sidebar = st.session_state.screen
    if "workspace_sidebar" not in st.session_state:
        st.session_state.workspace_sidebar = st.session_state.workspace
    # ── Live session state keys (Task 1 & 2) ──────────────────────────────
    init_live_session_state()


def hero(title, copy, kicker="ARIES Interview Studio"):
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-copy">{copy}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title):
    st.markdown(f"<section class='panel'><div class='label'>{title}</div>", unsafe_allow_html=True)


def panel_end():
    st.markdown("</section>", unsafe_allow_html=True)


def metric_card(label, value, caption, color):
    st.markdown(
        f"""
        <div class="metric">
            <div class="label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div style="margin-top:.35rem; color:{THEME["muted"]}; font-size:.82rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyze_camera_frame(uploaded_file):
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("L")
        width, height = image.size
        brightness = ImageStat.Stat(image).mean[0]
        blur_metric = ImageStat.Stat(image.filter(ImageFilter.FIND_EDGES)).mean[0]
        image_np = np.array(image)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        score = 100
        notes = []

        if width < 320 or height < 240:
            score -= 25
            notes.append("low resolution")
        if brightness < 55:
            score -= 20
            notes.append("too dark")
        elif brightness > 210:
            score -= 15
            notes.append("too bright")
        if blur_metric < 12:
            score -= 20
            notes.append("blurry frame")
        if len(faces) == 0:
            score -= 35
            notes.append("no face detected")
        elif len(faces) > 1:
            score -= 10
            notes.append("multiple faces detected")

        status = "Face detected clearly"
        if score < 45:
            status = "Visibility poor"
        elif score < 70:
            status = "Visibility needs improvement"
        elif len(faces) == 0:
            status = "No face detected"
        elif len(faces) > 1:
            status = "Multiple faces detected"

        detail = ", ".join(notes) if notes else "lighting and frame quality look usable"
        return {
            "status": status,
            "score": max(0, min(100, int(score))),
            "detail": detail,
            "resolution": f"{width}x{height}",
            "faces": int(len(faces)),
        }
    except Exception:
        return {
            "status": "Camera frame captured",
            "score": 50,
            "detail": "quality analysis unavailable",
            "resolution": "unknown",
            "faces": 0,
        }


def render_camera_settings():
    panel_start("Camera Settings")
    st.toggle(
        "Enable camera checks",
        key="camera_enabled",
        help="Use the browser camera for pre-interview framing and visibility checks.",
    )
    if st.session_state.camera_enabled:
        st.radio(
            "Camera use",
            ["Pre-interview check", "During interview check-ins"],
            key="camera_capture_mode",
            horizontal=True,
        )
        st.checkbox("Show camera panel during interview", key="camera_show_preview")
        camera_file = st.camera_input("Camera check", key="camera_input_setup")
        if camera_file is not None:
            st.session_state.camera_last_check_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis = analyze_camera_frame(camera_file)
            st.session_state.camera_visibility_status = analysis["status"]
            st.session_state.camera_visibility_score = analysis["score"]
            st.success("Camera frame captured successfully.")
            st.caption(f"Visibility: {analysis['status']} • Faces: {analysis['faces']} • Score: {analysis['score']}/100 • {analysis['detail']} • {analysis['resolution']}")
        st.caption(f"Last camera check: {st.session_state.camera_last_check_label}")
        st.caption(f"Current visibility status: {st.session_state.camera_visibility_status}")
        st.info("Browser permission is required. In Streamlit, camera access is best used as a check-in capture rather than a continuous live stream.")
    else:
        st.caption("Camera checks are off. You can still run the interview with voice and transcript evaluation.")
    panel_end()


def render_voice_visual(duration_seconds):
    heights = [28, 54, 88, 62, 104, 138, 90, 120, 68, 112, 78, 146, 86, 56, 32]
    bars = "".join(f"<span style='height:{height}px'></span>" for height in heights)
    mins = duration_seconds // 60
    secs = duration_seconds % 60
    st.markdown(
        f"""
        <div class="voice-box">
            <div class="bars">{bars}</div>
            <div class="voice-center">
                <div class="voice-time">{mins:02d}:{secs:02d}</div>
                <div class="label" style="margin-top:.45rem;">Interview answer capture</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def choose_question_pool(questions):
    """Build the question pool from auto-generated picks + any user-selected/extra questions."""
    mode = st.session_state.interview_mode
    categories = INTERVIEW_PRESETS.get(mode, ["HR & General"])
    st.session_state.selected_categories = categories

    # If user has built a custom pool via the picker, use that instead
    selected = st.session_state.get("practice_selected_qs", [])
    extra = st.session_state.get("practice_extra_qs", [])
    if selected or extra:
        pool = list(selected) + list(extra)
        random.shuffle(pool)
        st.session_state.question_pool = pool
        return

    # Fallback: auto-generate as before
    pool = []
    for category in categories:
        available = questions.get(category, [])
        count = 2 if len(categories) > 1 else 5
        if available:
            picks = random.sample(available, min(count, len(available)))
            for question in picks:
                pool.append({"category": category, "question": question})
    random.shuffle(pool)
    st.session_state.question_pool = pool[:5]


def _generate_practice_pool(questions):
    """Generate a fresh set of random candidate questions for the picker panel."""
    mode = st.session_state.interview_mode
    categories = INTERVIEW_PRESETS.get(mode, ["HR & General"])
    pool = []
    for category in categories:
        available = questions.get(category, [])
        count = 3 if len(categories) > 1 else 8
        if available:
            picks = random.sample(available, min(count, len(available)))
            for q in picks:
                pool.append({"category": category, "question": q})
    random.shuffle(pool)
    st.session_state.practice_generated_pool = pool
    st.session_state.practice_selected_qs = []
    st.session_state.practice_extra_qs = []
    st.session_state.practice_pool_built = True


def start_interview(questions):
    candidate = st.session_state.candidate_name.strip()
    if not candidate:
        st.warning("Enter the candidate name first.")
        return
    choose_question_pool(questions)
    if not st.session_state.question_pool:
        st.error("No questions were generated for this interview mode.")
        return
    st.session_state.session_started = True
    st.session_state.current_question_index = 0
    st.session_state.current_transcript = ""
    st.session_state.question_results = []
    st.session_state.final_report_ready = False
    # Reset picker state for next session
    st.session_state.practice_pool_built = False
    st.session_state.practice_generated_pool = []
    st.session_state.practice_selected_qs = []
    st.session_state.practice_extra_qs = []


def transcribe_with_whisper(audio_path):
    """Run faster-whisper STT and return a clean transcript for the UI."""
    from modules.whisper_stt import transcribe_audio

    with st.spinner("Transcribing audio with Whisper..."):
        return transcribe_audio(audio_path)


def capture_answer(duration):
    try:
        import sounddevice as sd
        import soundfile as sf

        with st.spinner(f"Recording answer for {duration} seconds..."):
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            sf.write(TEMP_AUDIO_FILE, audio, 16000)

        transcript = transcribe_with_whisper(TEMP_AUDIO_FILE)
        st.session_state.current_transcript = transcript
        st.session_state.current_recording_seconds = duration
        st.success("Answer recorded and transcribed with Whisper.")
    except Exception as exc:
        st.error(f"Voice recording failed: {exc}")


def transcribe_browser_audio(audio_file):
    try:
        with open(TEMP_AUDIO_FILE, "wb") as file:
            file.write(audio_file.getvalue())

        transcript = transcribe_with_whisper(TEMP_AUDIO_FILE)
        st.session_state.current_transcript = transcript
        st.session_state.current_recording_seconds = 0
        st.success("Audio captured from browser microphone and transcribed with Whisper.")
    except Exception as exc:
        st.error(f"Browser audio transcription failed: {exc}")


def evaluate_current_question():
    """Evaluate current question using the full evaluation pipeline (Tasks 3, 4, 7)."""
    transcript = st.session_state.current_transcript.strip()
    # Edge case: empty answer (Task 7)
    if not transcript:
        st.warning("Record an answer or paste the transcript before evaluation.")
        return

    from modules.evaluation_pipeline import run_evaluation

    # Support both question_pool questions AND custom questions (Task 2)
    custom_q = st.session_state.get("current_question", "").strip()
    if custom_q:
        question = custom_q
        category = st.session_state.get("live_question_category", "General")
    else:
        question_data = st.session_state.question_pool[st.session_state.current_question_index]
        question = question_data["question"]
        category = question_data["category"]

    audio_path = TEMP_AUDIO_FILE if os.path.exists(TEMP_AUDIO_FILE) else None
    duration   = st.session_state.current_recording_seconds or None

    with st.spinner("Running NLP, AI, fluency and emotion analysis…"):
        result = run_evaluation(question, transcript, audio_path, duration, category)

    if result is None:
        st.warning("Evaluation returned no result. Check transcript content.")
        return

    # Warn if AI evaluator fell back to NLP only
    if result.get("ai_failed"):
        st.info("⚠️ AI evaluator unavailable — using NLP relevance only.")

    # Upsert into question_results
    existing = next(
        (i for i, r in enumerate(st.session_state.question_results) if r["question"] == question),
        None,
    )
    if existing is None:
        st.session_state.question_results.append(result)
    else:
        st.session_state.question_results[existing] = result
    st.success("Question evaluated. Review scorecards below or move to the next question.")


def save_current_interview():
    """Save the current interview using grouped session format (Task 5)."""
    candidate = st.session_state.candidate_name.strip()
    results   = st.session_state.question_results
    if not candidate:
        return
    if not results:
        # No answers — save as abandoned so it’s still tracked (Task 1)
        save_abandoned_session(candidate, session_type="interview")
        return
    save_session_block(
        candidate,
        session_type=st.session_state.get("live_session_type", "interview"),
        records=results,
        status="completed",
    )



def move_question(step):
    max_index = len(st.session_state.question_pool) - 1
    st.session_state.current_question_index = min(max(st.session_state.current_question_index + step, 0), max_index)
    current_question = st.session_state.question_pool[st.session_state.current_question_index]["question"]
    previous = next((item for item in st.session_state.question_results if item["question"] == current_question), None)
    st.session_state.current_transcript = previous["answer"] if previous else ""
    st.session_state.current_recording_seconds = 0


def finish_interview():
    save_current_interview()
    st.session_state.final_report_ready = True
    st.session_state.screen = "Report"


def render_question_picker(questions):
    """Practice Mode Question Picker — lets user choose from random auto-generated questions + add extras."""
    mode = st.session_state.interview_mode
    all_cats = list(questions.keys())

    st.markdown(
        "<div style='font-size:.72rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:.18em; color:#9aa9c7; margin:.8rem 0 .45rem;'>"
        "🎲 Practice Question Picker</div>",
        unsafe_allow_html=True,
    )
    panel_start("Choose Your Questions")

    gen_col, info_col = st.columns([1, 2])
    with gen_col:
        if st.button("🔀 Generate Random Questions", use_container_width=True, key="btn_gen_pool"):
            _generate_practice_pool(questions)
            st.rerun()
    with info_col:
        cats = INTERVIEW_PRESETS.get(mode, [])
        pills_html = "".join(f'<span class="pill" style="margin-right:.3rem;">{c}</span>' for c in cats)
        st.markdown(
            f"<div style='padding:.45rem 0; font-size:.84rem; color:#9aa9c7;'>"
            f"Pulling from: {pills_html}</div>",
            unsafe_allow_html=True,
        )

    generated = st.session_state.get("practice_generated_pool", [])

    if generated:
        st.markdown(
            "<div style='font-size:.8rem; color:#9aa9c7; margin:.5rem 0 .3rem;'>"
            "<b>Tick the questions you want to include in your session:</b></div>",
            unsafe_allow_html=True,
        )
        selected = []
        for idx, item in enumerate(generated):
            checked = st.checkbox(
                f"[{item['category']}] {item['question']}",
                key=f"picker_q_{idx}",
                value=True,  # default: all selected
            )
            if checked:
                selected.append(item)
        st.session_state.practice_selected_qs = selected

        sel_count = len(selected)
        color = THEME["success"] if sel_count > 0 else THEME["danger"]
        st.markdown(
            f"<div style='font-size:.82rem; margin-top:.4rem; color:{color};'>"
            f"✔ {sel_count} question(s) selected from generated pool.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Click **Generate Random Questions** to get a fresh set based on your interview mode.")

    # ── Extra custom questions ────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.8rem; font-weight:700; color:#9aa9c7; margin:.75rem 0 .3rem;'>"
        "➕ Add Your Own Questions (one per line):</div>",
        unsafe_allow_html=True,
    )
    extra_col, cat_col = st.columns([3, 1])
    with extra_col:
        extra_raw = st.text_area(
            "Custom questions",
            key="picker_extra_raw",
            height=110,
            placeholder="e.g. Explain your final year project\nHow do you handle ambiguous requirements?\nDescribe your biggest technical challenge.",
            label_visibility="collapsed",
        )
    with cat_col:
        extra_cat = st.selectbox(
            "Category for extras",
            all_cats,
            key="picker_extra_cat",
            label_visibility="collapsed",
        )

    # Parse extra questions
    extra_list = []
    if extra_raw.strip():
        for line in extra_raw.strip().splitlines():
            line = line.strip()
            if line:
                extra_list.append({"category": extra_cat, "question": line})
    st.session_state.practice_extra_qs = extra_list

    if extra_list:
        st.markdown(
            f"<div style='font-size:.82rem; color:{THEME['primary']}; margin-top:.3rem;'>"
            f"📝 {len(extra_list)} custom question(s) will be added.</div>",
            unsafe_allow_html=True,
        )

    total = len(st.session_state.get("practice_selected_qs", [])) + len(extra_list)
    if total > 0:
        st.success(f"Ready: **{total}** question(s) will be used in your session.")

    panel_end()


def render_setup(questions):
    hero(
        "Interview your candidate, don't just fill a text form",
        "Choose your interview type, pick or generate practice questions, add your own, then start the session.",
        "Interview Control Room",
    )

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        panel_start("Candidate Setup")
        st.text_input("Candidate name", key="candidate_name_setup", placeholder="e.g. Manan Sharma")
        st.session_state.candidate_name = st.session_state.candidate_name_setup.strip()
        st.selectbox("Interview mode", list(INTERVIEW_PRESETS.keys()), key="interview_mode")
        st.number_input("Answer time per question (seconds)", min_value=15, max_value=120, value=30, step=5, key="current_duration")
        categories = INTERVIEW_PRESETS.get(st.session_state.interview_mode, [])
        pills_html = "".join(f'<div class="pill">{c}</div>' for c in categories)
        st.markdown(f"<div class='pill-row'>{pills_html}</div>", unsafe_allow_html=True)
        panel_end()
        if st.button("Start Interview Session", use_container_width=True, type="primary"):
            start_interview(questions)
            if st.session_state.session_started:
                st.rerun()

    with right:
        panel_start("What This Session Uses")
        st.markdown(
            """
            - `NLP evaluator` for relevance and sentiment
            - `Fluency analyzer` for fillers, pace, and answer length
            - `Emotion model` for voice-based confidence signal
            - `Scoring engine` for final grade and feedback
            """,
        )
        st.markdown("<div class='pill-row'><div class='pill'>Voice-first flow</div><div class='pill'>Sequential interview</div><div class='pill'>Saved session history</div></div>", unsafe_allow_html=True)
        panel_end()
        render_camera_settings()

    # ── Question Picker (full width below setup columns) ──────────────────
    render_question_picker(questions)


def render_live_interview():
    question_data = st.session_state.question_pool[st.session_state.current_question_index]
    question_number = st.session_state.current_question_index + 1
    total_questions = len(st.session_state.question_pool)

    hero(
        f"Live Interview: Question {question_number} of {total_questions}",
        "Use the interviewer panel on the left and the response workspace on the right. Record the answer, inspect the transcript, run evaluation, and move through the interview one question at a time.",
        "Interview In Progress",
    )

    # ── Custom question input + session controls (Tasks 1 & 2) ───────────
    questions_data = load_questions()
    all_categories = list(questions_data.keys())
    custom_q, _ = render_custom_question_input(all_categories)

    # End Session button row — works even with 0 answers (Task 1)
    end_cols = st.columns([3, 1])
    with end_cols[1]:
        if st.button("⏹ End & Save Session", use_container_width=True, key="live_end_session_btn"):
            candidate = st.session_state.candidate_name.strip()
            results   = st.session_state.question_results
            if candidate:
                if results:
                    save_session_block(candidate, "interview", results, status="completed")
                    st.success(f"Session saved — {len(results)} answer(s) recorded.")
                else:
                    save_abandoned_session(candidate, session_type="interview")
                    st.info("Session ended with no answers — saved as abandoned.")
            st.session_state.session_started = False
            st.session_state.question_results = []
            st.session_state.screen = "Dashboard"
            st.rerun()

    # Determine active question (custom takes priority over pool)
    active_question = custom_q if custom_q else question_data["question"]
    active_category = st.session_state.get("live_question_category", question_data["category"])

    left, right = st.columns([1, 1.08], gap="large")
    with left:
        st.markdown(
            f"""
            <section class="panel question-panel">
                <div class="label">Current Question</div>
                <div class="question-title">{active_question}</div>
                <div style="color:{THEME["muted"]}; line-height:1.7;">Ask this question naturally, then let the candidate answer.</div>
                <div class="pill-row">
                    <div class="pill">Category: {active_category}</div>
                    <div class="pill">Mode: {st.session_state.interview_mode}</div>
                    <div class="pill">Time limit: {st.session_state.current_duration}s</div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        panel_start("Interview Timeline")
        if not st.session_state.question_results:
            st.info("No questions evaluated yet. Record the first answer to begin building the report.")
        for idx, result in enumerate(st.session_state.question_results, start=1):
            st.markdown(
                f"""
                <div class="timeline-card">
                    <div class="label">Q{idx} • {result["category"]}</div>
                    <div style="font-family:Manrope,sans-serif; font-weight:700; margin-top:.35rem;">{result["question"]}</div>
                    <div style="margin-top:.55rem; color:{THEME["muted"]};">Score {result["scores"]["final"]:.0f}/100 • Grade {result["grade"]} • Emotion {result["emotion"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        panel_end()
        if st.session_state.camera_enabled and st.session_state.camera_show_preview:
            panel_start("Camera Check")
            st.caption(f"Mode: {st.session_state.camera_capture_mode}")
            camera_file = st.camera_input("Interview camera", key="camera_input_live")
            if camera_file is not None:
                st.session_state.camera_last_check_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                analysis = analyze_camera_frame(camera_file)
                st.session_state.camera_visibility_status = analysis["status"]
                st.session_state.camera_visibility_score = analysis["score"]
                st.success("Interview camera frame updated.")
                st.caption(f"Visibility: {analysis['status']} • Faces: {analysis['faces']} • Score: {analysis['score']}/100 • {analysis['detail']}")
            st.caption(f"Latest camera check: {st.session_state.camera_last_check_label}")
            st.caption(f"Current visibility status: {st.session_state.camera_visibility_status}")
            panel_end()

    with right:
        render_voice_visual(st.session_state.current_duration)
        browser_audio = st.audio_input("Browser microphone", key="browser_audio_input")
        if browser_audio is not None:
            transcribe_browser_audio(browser_audio)
        st.text_area(
            "Transcript",
            key="current_transcript",
            height=220,
            placeholder="The transcribed answer will appear here after recording. You can also paste or fix the transcript manually before evaluation.",
        )
        st.caption("This is transcript review, not the interview itself. The actual flow is voice-first.")
        actions = st.columns([1.05, 1.05, 1.1])
        with actions[0]:
            if st.button("Record Answer (desktop mic)", use_container_width=True):
                capture_answer(st.session_state.current_duration)
        with actions[1]:
            if st.button("Evaluate Answer", use_container_width=True):
                evaluate_current_question()
        with actions[2]:
            if st.button("Finish Interview", use_container_width=True):
                finish_interview()
                st.rerun()

        nav = st.columns([1, 1, 1.2])
        with nav[0]:
            if st.button("Previous", use_container_width=True, disabled=st.session_state.current_question_index == 0):
                move_question(-1)
                st.rerun()
        with nav[1]:
            if st.button("Next Question", use_container_width=True, disabled=st.session_state.current_question_index >= total_questions - 1):
                move_question(1)
                st.rerun()
        with nav[2]:
            if st.button("Save Progress", use_container_width=True):
                save_current_interview()
                st.success("Interview progress saved.")

    st.markdown("</div>", unsafe_allow_html=True)

    current_question = active_question
    current_result = next((item for item in st.session_state.question_results if item["question"] == current_question), None)
    if current_result:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        cols = st.columns(5)
        with cols[0]:
            metric_card("Relevance", f"{current_result['scores']['relevance']:.0f}", "Question alignment", THEME["primary"])
        with cols[1]:
            metric_card("Confidence", f"{current_result['scores'].get('confidence', 70):.0f}", "From emotion", THEME["primary_2"])
        with cols[2]:
            metric_card("Fluency", f"{current_result['scores']['fluency']:.0f}", "Pace and fillers", THEME["success"])
        with cols[3]:
            metric_card("Emotion", current_result["emotion"].title(), "Detected voice state", THEME["warning"])
        with cols[4]:
            metric_card("Final Score", f"{current_result['scores']['final']:.0f}", current_result["grade_label"], THEME["danger"])

        feedback_cols = st.columns(2)
        with feedback_cols[0]:
            panel_start("Strengths")
            for item in current_result["feedback"]["strengths"] or ["No major strengths detected yet."]:
                st.markdown(f"- {item}")
            panel_end()
        with feedback_cols[1]:
            panel_start("Improve Next")
            items = current_result["feedback"]["improvements"] + current_result["fluency_feedback"][:2]
            for item in items or ["No issues flagged for this answer."]:
                st.markdown(f"- {item}")
            panel_end()


def render_final_report():
    results = st.session_state.question_results
    hero(
        "Interview Report",
        "This final report summarizes the full interview, not a single typed answer. It combines all evaluated responses into one end-of-session view.",
        "Session Summary",
    )
    if not results:
        st.info("No evaluated questions yet.")
        return

    average_score = sum(item["scores"]["final"] for item in results) / len(results)
    emotions = [item["emotion"] for item in results]
    _, _, calculate_esi, _, _ = load_models()
    esi = calculate_esi(emotions)
    categories = ", ".join(sorted({item["category"] for item in results}))

    top = st.columns(4)
    with top[0]:
        metric_card("Candidate", st.session_state.candidate_name, "Interviewed profile", THEME["primary"])
    with top[1]:
        metric_card("Avg Score", f"{average_score:.1f}", "Across all responses", THEME["success"])
    with top[2]:
        metric_card("Emotion Stability", f"{esi:.2f}", "Consistency index", THEME["primary_2"])
    with top[3]:
        metric_card("Question Count", str(len(results)), categories, THEME["warning"])

    panel_start("Question-by-Question Breakdown")
    for idx, result in enumerate(results, start=1):
        st.markdown(
            f"""
            <div class="timeline-card">
                <div class="label">Question {idx} • {result["category"]}</div>
                <div style="font-family:Manrope,sans-serif; font-size:1.1rem; font-weight:700; margin-top:.35rem;">{result["question"]}</div>
                <div style="margin-top:.6rem; color:{THEME["muted"]}; line-height:1.7;">
                    Final {result["scores"]["final"]:.0f}/100 • Relevance {result["scores"]["relevance"]:.0f} • Fluency {result["scores"]["fluency"]:.0f} • Sentiment {result["scores"]["sentiment"]:.0f} • Emotion {result["emotion"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    panel_end()

    actions = st.columns([1, 1, 1.2])
    with actions[0]:
        if st.button("Back to Interview", use_container_width=True):
            st.session_state.screen = "Interview"
            st.rerun()
    with actions[1]:
        if st.button("Open Dashboard", use_container_width=True):
            save_current_interview()
            st.session_state.screen = "Dashboard"
            st.rerun()
    with actions[2]:
        if st.button("Start New Interview", use_container_width=True):
            st.session_state.session_started = False
            st.session_state.question_pool = []
            st.session_state.question_results = []
            st.session_state.current_transcript = ""
            st.session_state.current_question_index = 0
            st.session_state.final_report_ready = False
            st.session_state.screen = "Interview"
            st.rerun()


def render_dashboard():
    hero(
        "Practice Score Tracker",
        "Track your interview growth — score trends, grade distribution, emotion patterns, category strengths, and session comparisons all in one place.",
        "Analytics Dashboard",
    )
    sessions = load_sessions()
    candidate = st.session_state.candidate_name.strip()
    if not candidate or candidate not in sessions:
        st.info("Enter a candidate name with saved sessions, or complete an interview first.")
        return

    records = sessions[candidate]
    if not records:
        st.info("No sessions recorded yet. Complete a practice session or interview to populate the dashboard.")
        return

    scores = [item["scores"]["final"] for item in records]
    emotions = [item.get("emotion", "neutral") for item in records if item.get("emotion")]
    grades = [item.get("grade", "F") for item in records]
    _, _, calculate_esi, _, _ = load_models()
    esi = calculate_esi(emotions) if emotions else 0.0

    # ── Streak: consecutive sessions with grade B or above ──
    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    streak = 0
    for g in reversed(grades):
        if grade_order.get(g, 0) >= 4:
            streak += 1
        else:
            break

    # ── Improvement rate: first 3 vs last 3 sessions ──
    if len(scores) >= 6:
        first3_avg = sum(scores[:3]) / 3
        last3_avg = sum(scores[-3:]) / 3
        improvement = round(((last3_avg - first3_avg) / max(first3_avg, 1)) * 100, 1)
        improv_label = f"+{improvement}%" if improvement >= 0 else f"{improvement}%"
        improv_color = THEME["success"] if improvement >= 0 else THEME["danger"]
    else:
        improv_label = "Need 6+ sessions"
        improv_color = THEME["muted"]

    # ── Top row KPI cards ──
    top = st.columns(5)
    with top[0]:
        metric_card("Sessions", str(len(records)), "Total evaluated answers", THEME["primary"])
    with top[1]:
        metric_card("Avg Score", f"{sum(scores)/len(scores):.1f}", "Overall average", THEME["success"])
    with top[2]:
        metric_card("Best Score", f"{max(scores):.0f}", "Your personal best", THEME["warning"])
    with top[3]:
        metric_card("B+ Streak", str(streak), "Consecutive B or above", THEME["primary_2"])
    with top[4]:
        metric_card("Improvement", improv_label, "First 3 vs last 3 sessions", improv_color)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Row 1: Score trend + Grade donut ──
    trend_col, donut_col = st.columns([1.7, 1], gap="large")
    with trend_col:
        panel_start("Score Timeline")
        fig, ax = plt.subplots(figsize=(9, 3.4))
        fig.patch.set_facecolor(THEME["panel"])
        ax.set_facecolor(THEME["panel"])
        x = range(1, len(scores) + 1)
        ax.plot(x, scores, color=THEME["primary"], linewidth=2.6, marker="o", markersize=5)
        ax.fill_between(x, scores, color=THEME["primary"], alpha=0.13)
        # Grade threshold lines
        for threshold, label, color in [(85, "A", THEME["success"]), (70, "B", THEME["primary"]), (55, "C", THEME["warning"])]:
            ax.axhline(threshold, color=color, linewidth=0.8, linestyle="--", alpha=0.45)
            ax.text(len(scores) + 0.1, threshold, label, color=color, fontsize=8, va="center")
        ax.set_ylim(0, 108)
        ax.set_xlim(0.5, len(scores) + 1.5)
        ax.set_title("Score progression across sessions", color=THEME["text"], pad=8)
        ax.tick_params(colors=THEME["text"])
        ax.set_ylabel("Score", color=THEME["text"])
        ax.set_xlabel("Session #", color=THEME["text"])
        for spine in ax.spines.values():
            spine.set_color(THEME["line"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        panel_end()

    with donut_col:
        panel_start("Grade Distribution")
        grade_counts = {g: grades.count(g) for g in ["A", "B", "C", "D", "F"] if grades.count(g) > 0}
        if grade_counts:
            grade_colors_map = {
                "A": THEME["success"], "B": THEME["primary"],
                "C": THEME["warning"], "D": "#fb923c", "F": THEME["danger"]
            }
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.4))
            fig2.patch.set_facecolor(THEME["panel"])
            wedges, texts, autotexts = ax2.pie(
                list(grade_counts.values()),
                labels=list(grade_counts.keys()),
                autopct="%1.0f%%",
                colors=[grade_colors_map[g] for g in grade_counts],
                startangle=90,
                wedgeprops={"width": 0.55, "edgecolor": THEME["panel"], "linewidth": 2},
                textprops={"color": THEME["text"], "fontsize": 11, "fontweight": "bold"},
            )
            for autotext in autotexts:
                autotext.set_color(THEME["text"])
                autotext.set_fontsize(9)
            ax2.set_title("Grades earned", color=THEME["text"], pad=8)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        panel_end()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Category radar + Emotion trend ──
    radar_col, emotion_col = st.columns([1, 1], gap="large")

    with radar_col:
        panel_start("Category Performance Radar")
        categories_seen = list({r.get("category", "Unknown") for r in records})
        if len(categories_seen) >= 3:
            cat_avgs = []
            for cat in categories_seen:
                cat_scores = [r["scores"]["final"] for r in records if r.get("category") == cat]
                cat_avgs.append(round(sum(cat_scores) / len(cat_scores), 1))
            N = len(categories_seen)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            cat_avgs_plot = cat_avgs + cat_avgs[:1]
            fig3, ax3 = plt.subplots(figsize=(4.8, 3.8), subplot_kw=dict(polar=True))
            fig3.patch.set_facecolor(THEME["panel"])
            ax3.set_facecolor(THEME["panel"])
            ax3.plot(angles, cat_avgs_plot, color=THEME["primary_2"], linewidth=2)
            ax3.fill(angles, cat_avgs_plot, color=THEME["primary_2"], alpha=0.22)
            ax3.set_xticks(angles[:-1])
            short_labels = [c[:12] if len(c) > 12 else c for c in categories_seen]
            ax3.set_xticklabels(short_labels, color=THEME["text"], fontsize=8)
            ax3.set_ylim(0, 100)
            ax3.set_yticks([25, 50, 75, 100])
            ax3.set_yticklabels(["25", "50", "75", "100"], color=THEME["muted"], fontsize=7)
            ax3.tick_params(colors=THEME["text"])
            ax3.spines["polar"].set_color(THEME["line"])
            ax3.grid(color=THEME["line"], alpha=0.4)
            ax3.set_title("Avg score by category", color=THEME["text"], pad=14)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
        else:
            st.info("Answer questions from 3+ different categories to unlock the radar chart.")
        panel_end()

    with emotion_col:
        panel_start("Emotion Trend")
        emotion_order = ["confident", "neutral", "nervous", "stressed"]
        emotion_colors_map = {
            "confident": THEME["success"],
            "neutral": THEME["primary"],
            "nervous": THEME["warning"],
            "stressed": THEME["danger"],
        }
        emotion_counts = {e: emotions.count(e) for e in emotion_order if emotions.count(e) > 0}
        if emotion_counts:
            fig4, ax4 = plt.subplots(figsize=(4.8, 3.8))
            fig4.patch.set_facecolor(THEME["panel"])
            ax4.set_facecolor(THEME["panel"])
            bars = ax4.bar(
                list(emotion_counts.keys()),
                list(emotion_counts.values()),
                color=[emotion_colors_map.get(e, THEME["muted"]) for e in emotion_counts],
                width=0.55,
            )
            ax4.bar_label(bars, color=THEME["text"], fontsize=9, fontweight="bold")
            ax4.set_title("Emotion detection count", color=THEME["text"], pad=8)
            ax4.tick_params(colors=THEME["text"])
            ax4.set_ylabel("Count", color=THEME["text"])
            for spine in ax4.spines.values():
                spine.set_color(THEME["line"])
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)
        else:
            st.info("Complete voice-recorded sessions to see the emotion detection breakdown.")
        panel_end()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Row 3: Session comparison (early vs recent) ──
    if len(records) >= 4:
        panel_start("Session Comparison — Early vs Recent")
        half = min(3, len(records) // 2)
        early = records[:half]
        recent = records[-half:]
        comp_rows = []
        for idx in range(half):
            comp_rows.append({
                "#": idx + 1,
                "Early Date": early[idx].get("date", "-"),
                "Early Score": early[idx]["scores"]["final"],
                "Early Grade": early[idx].get("grade", "-"),
                "Recent Date": recent[idx].get("date", "-"),
                "Recent Score": recent[idx]["scores"]["final"],
                "Recent Grade": recent[idx].get("grade", "-"),
                "Delta": round(recent[idx]["scores"]["final"] - early[idx]["scores"]["final"], 1),
            })
        st.dataframe(comp_rows, use_container_width=True, hide_index=True)
        panel_end()

    # ── Full log table ──
    panel_start("Saved Answer Log")
    table_rows = [
        {
            "Date": record.get("date", "-"),
            "Mode": record.get("mode", "practice"),
            "Category": record.get("category", "-"),
            "Question": (record.get("question", "-") or "-")[:60] + ("…" if len(record.get("question", "") or "") > 60 else ""),
            "Score": record["scores"]["final"],
            "Grade": record.get("grade", "-"),
            "Emotion": record.get("emotion", "-"),
        }
        for record in records[::-1]
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
    panel_end()


# ══════════════════════════════════════════════════════════════
# LIVE INTERVIEW COMPANION
# A compact, distraction-free screen designed to run alongside
# Zoom or Google Meet in a snapped/half-screen window.
# ══════════════════════════════════════════════════════════════

def _companion_score_bar(label, value, color):
    """Render a single labelled progress bar (pure HTML, no Streamlit progress widget)."""
    pct = max(0, min(100, float(value)))
    st.markdown(
        f"""
        <div class="progress-row">
            <span class="progress-label">{label}</span>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="progress-val" style="color:{color};">{pct:.0f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _companion_emotion_tip(emotion):
    tips = {
        "confident": "🟢 Great energy — keep this composure for the next question.",
        "neutral":   "🔵 Steady tone. Try adding slightly more enthusiasm to stand out.",
        "nervous":   "🟡 You sounded nervous. Take a slow breath before the next answer.",
        "stressed":  "🔴 High stress detected. Pause, exhale, and slow your pace.",
    }
    return tips.get(emotion, "Keep going — each answer is practice.")


def _companion_capture_voice(duration):
    try:
        import sounddevice as sd
        import soundfile as sf
        with st.spinner(f"Recording for {duration}s…"):
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            sf.write(TEMP_AUDIO_FILE, audio, 16000)
        transcript = transcribe_with_whisper(TEMP_AUDIO_FILE)
        # Write to shadow buffer — never to the widget-owned key
        st.session_state["_companion_transcript_buf"] = transcript
        st.success("Transcribed ✓")
    except Exception as exc:
        st.error(f"Recording error: {exc}")


def _companion_analyze():
    """Evaluate one companion answer through the full pipeline."""
    # Read from shadow buffers, not the widget-owned keys
    transcript = st.session_state.get("_companion_transcript_buf", "").strip()
    question = (
        st.session_state.get("companion_question_val", "").strip()
        or st.session_state.get("_companion_question_buf", "").strip()
    )
    # Edge cases
    if not transcript:
        st.warning("Paste or record your answer first.")
        return
    if not question:
        st.warning("Enter the interview question first.")
        return

    from modules.evaluation_pipeline import run_evaluation
    category = st.session_state.get("companion_category", "General")
    audio_path = TEMP_AUDIO_FILE if os.path.exists(TEMP_AUDIO_FILE) else None

    with st.spinner("Analyzing with NLP, AI, fluency and emotion models…"):
        result = run_evaluation(question, transcript, audio_path, category=category)

    if result is None:
        st.warning("Could not evaluate — is the answer empty?")
        return

    if result.get("ai_failed"):
        st.info("⚠️ AI evaluator unavailable — using NLP relevance only.")

    # Store AI result for the panel
    st.session_state.companion_ai_result = {
        "overall_score": result["scores"].get("relevance", 0),
        "ai_clarity":    result.get("ai_clarity", 0),
        "ai_structure":  result.get("ai_structure", 0),
        "ai_depth":      result.get("ai_depth", 0),
    }

    entry = {
        **result,
        "category": category,
        "mode":     "live_companion",
        "tip":      _companion_emotion_tip(result["emotion"]),
    }
    st.session_state.companion_log.append(entry)
    # Clear via shadow buffers — widget keys are NOT touched
    st.session_state["_companion_transcript_buf"] = ""
    st.session_state["_companion_question_buf"]   = ""
    st.session_state["companion_question_val"]    = ""


def _companion_end_session():
    """Save companion log to disk via session_manager (Task 1: works with 0 answers)."""
    candidate = st.session_state.candidate_name.strip()
    log       = st.session_state.companion_log
    if not candidate:
        st.warning("No candidate name set. Enter your name in the sidebar first.")
        return False
    # 0-answer session — save as abandoned instead of blocking (Task 1 fix)
    if not log:
        save_abandoned_session(candidate, session_type="live_companion")
    else:
        save_session_block(candidate, "live_companion", log, status="completed")
    # Reset companion state
    st.session_state.companion_log        = []
    st.session_state.companion_ai_result  = None
    st.session_state.companion_active     = False
    st.session_state.companion_saved      = True
    return True


def render_live_companion():
    """Compact interview companion — designed to run alongside Zoom / Google Meet."""
    questions_data = load_questions()
    all_categories = list(questions_data.keys())

    # ── Companion header bar ──
    st.markdown(
        """
        <div class="companion-header">
            <div class="companion-brand">ARIES <span style="color:#9aa9c7; font-size:.85rem; font-weight:500;">Live Companion</span></div>
            <div class="companion-live-badge"><span class="live-dot"></span>LIVE INTERVIEW MODE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate = st.session_state.candidate_name.strip()
    if not candidate:
        st.warning("Set your candidate name in the sidebar first.")
        return

    # ── Show post-save confirmation banner if session was just saved ──
    if st.session_state.get("companion_saved"):
        st.success("Session saved! All answers are now in your Dashboard.")
        goto_cols = st.columns([1, 1, 2])
        with goto_cols[0]:
            if st.button("Open Dashboard", use_container_width=True, key="companion_goto_dash"):
                st.session_state.companion_saved = False
                st.session_state.screen = "Dashboard"
                st.session_state.screen_sidebar = "Dashboard"
                st.rerun()
        with goto_cols[1]:
            if st.button("New Session", use_container_width=True, key="companion_new_session"):
                st.session_state.companion_saved = False
                st.rerun()
        return  # don't render rest of companion UI after save

    # ── Running session stats bar ──
    log = st.session_state.companion_log
    q_count = len(log)
    avg_so_far = round(sum(e["scores"]["final"] for e in log) / q_count, 1) if log else 0
    last_grade = log[-1]["grade"] if log else "--"

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(f"<div class='stat-badge'><span class='stat-val' style='color:{THEME['primary']};'>{q_count}</span><span class='stat-lbl'>Questions</span></div>", unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"<div class='stat-badge'><span class='stat-val' style='color:{THEME['success']};'>{avg_so_far}</span><span class='stat-lbl'>Avg Score</span></div>", unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(f"<div class='stat-badge'><span class='stat-val' style='color:{THEME['warning']};'>{last_grade}</span><span class='stat-lbl'>Last Grade</span></div>", unsafe_allow_html=True)
    with stat_cols[3]:
        if st.button("End & Save Session", use_container_width=True, key="companion_end", type="primary"):
            saved = _companion_end_session()
            if saved:
                st.rerun()

    st.markdown("<hr style='border-color:rgba(38,53,83,0.6); margin:.6rem 0;'>", unsafe_allow_html=True)

    # ── Question entry ──
    left_col, right_col = st.columns([1.1, 1], gap="medium")

    with left_col:
        st.markdown("<div class='label' style='margin-bottom:.35rem;'>Interview Question</div>", unsafe_allow_html=True)
        # Quick-pick from bank OR custom type
        input_mode = st.radio("Question source", ["Type / Paste question", "Pick from question bank"], horizontal=True, key="companion_qmode", label_visibility="collapsed")
        if input_mode == "Pick from question bank":
            # selectbox owns companion_category — do NOT also set it via session_state
            st.selectbox("Category", all_categories, key="companion_category")
            active_cat = st.session_state.companion_category
            picked = st.selectbox("Question", questions_data[active_cat], key="companion_picked_q")
            # Write the picked question into companion_question only if it differs
            # (avoid the same widget-key conflict by using a separate non-widget key)
            st.session_state["companion_question_val"] = picked
            st.markdown(f"<div class='companion-log-item' style='margin-top:.4rem;'><span style='font-size:.9rem; font-weight:600;'>{picked}</span></div>", unsafe_allow_html=True)
        else:
            st.text_input(
                "Paste or type the question the interviewer just asked",
                key="companion_question",
                placeholder="e.g. Tell me about a time you handled conflict…",
                label_visibility="collapsed",
            )
            st.session_state["companion_question_val"] = st.session_state.companion_question
            st.selectbox("Category", all_categories, key="companion_category")

        st.markdown("<div class='label' style='margin:.6rem 0 .35rem;'>Your Answer</div>", unsafe_allow_html=True)
        browser_audio = st.audio_input("Browser mic", key="companion_browser_audio")
        if browser_audio is not None:
            try:
                with open(TEMP_AUDIO_FILE, "wb") as f:
                    f.write(browser_audio.getvalue())
                # Write to shadow buffer only
                st.session_state["_companion_transcript_buf"] = transcribe_with_whisper(TEMP_AUDIO_FILE)
                st.success("Transcribed ✓")
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

        # text_area uses its OWN key; we sync to shadow buffer on every render
        transcript_display = st.text_area(
            "Transcript / typed answer",
            value=st.session_state.get("_companion_transcript_buf", ""),
            height=130,
            placeholder="Your spoken answer will appear here after recording, or type it directly…",
            label_visibility="collapsed",
            key="companion_transcript_widget",
        )
        # Keep shadow buffer in sync with whatever the user typed
        st.session_state["_companion_transcript_buf"] = transcript_display
        act_cols = st.columns([1, 1])
        with act_cols[0]:
            dur = st.slider("Rec (sec)", 10, 90, 30, key="companion_duration", label_visibility="collapsed")
            if st.button("🎙 Record (desktop)", use_container_width=True, key="companion_record"):
                _companion_capture_voice(dur)
        with act_cols[1]:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            if st.button("⚡ Analyze Answer", use_container_width=True, key="companion_analyze", type="primary"):
                _companion_analyze()
                st.rerun()

    with right_col:
        # ── Latest result ──
        if log:
            latest = log[-1]
            st.markdown("<div class='label' style='margin-bottom:.4rem;'>Latest Result</div>", unsafe_allow_html=True)
            s = latest["scores"]
            emotion_color_map = {
                "confident": THEME["success"], "neutral": THEME["primary"],
                "nervous": THEME["warning"], "stressed": THEME["danger"]
            }
            ec = emotion_color_map.get(latest["emotion"], THEME["muted"])
            grade_color_map = {"A": THEME["success"], "B": THEME["primary"], "C": THEME["warning"], "D": "#fb923c", "F": THEME["danger"]}
            gc = grade_color_map.get(latest["grade"], THEME["muted"])
            st.markdown(
                f"""
                <div class="companion-log-item">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-family:Manrope,sans-serif; font-size:1.5rem; font-weight:800; color:{gc};">{latest['grade']}</span>
                        <span style="font-size:.82rem; color:{ec}; font-weight:700; text-transform:capitalize;">&#9679; {latest['emotion']}</span>
                    </div>
                    <div style="font-size:.8rem; color:{THEME['muted']}; margin-top:.3rem;">{latest['grade_label']} &middot; {latest['date']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _companion_score_bar("Relevance", s.get("relevance", 0), THEME["primary"])
            _companion_score_bar("Fluency", s.get("fluency", 0), THEME["primary_2"])
            _companion_score_bar("Final", s.get("final", 0), THEME["success"])
            st.markdown(f"<div class='tip-banner'>{latest['tip']}</div>", unsafe_allow_html=True)

            # ── AI Relevance Panel ──
            ai = st.session_state.get("companion_ai_result")
            if ai:
                api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
                if "error" in ai:
                    st.warning(f"AI check failed: {ai['error']}")
                else:
                    verdict = ai.get("verdict", "Acceptable")
                    ai_score = ai.get("overall_score", 50)
                    verdict_colors = {
                        "Strong": THEME["success"], "Acceptable": THEME["primary"],
                        "Needs Work": THEME["warning"], "Poor": THEME["danger"]
                    }
                    vc = verdict_colors.get(verdict, THEME["primary"])
                    star_icon = "Yes" if ai.get("star_method_used") else "No"
                    st.markdown(
                        f"""
                        <div class="companion-log-item" style="margin-top:.6rem; border-left:3px solid {vc};">
                            <div class="label" style="color:{vc};">Claude AI Relevance Check</div>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:.35rem;">
                                <span style="font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:800; color:{vc};">{verdict}</span>
                                <span style="font-size:1.1rem; font-weight:800; color:{vc};">{ai_score}/100</span>
                            </div>
                            <div style="font-size:.78rem; color:{THEME['muted']}; margin-top:.3rem;">STAR method: {star_icon} &nbsp;&middot;&nbsp; Confidence: {ai.get('confidence_level','Medium')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if ai.get("what_was_good"):
                        with st.expander("What was good / missing", expanded=False):
                            st.markdown("**Good:**")
                            for p in ai.get("what_was_good", []):
                                st.markdown(f"- {p}")
                            st.markdown("**Missing:**")
                            for p in ai.get("what_was_missing", []):
                                st.markdown(f"- {p}")
                            if ai.get("improved_answer_hint"):
                                st.info(ai["improved_answer_hint"])
            elif not os.environ.get("ANTHROPIC_API_KEY", ""):
                st.markdown(
                    "<div class='tip-banner' style='border-left-color:#8b7bff; background:rgba(139,123,255,0.07);'>"
                    "<b>AI Relevance Check</b> is available but requires your Anthropic API key. "
                    "Set <code>ANTHROPIC_API_KEY</code> in your terminal before launching the app."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""
                <div class="companion-log-item" style="text-align:center; padding:1.8rem 1rem;">
                    <div style="font-size:1.5rem; margin-bottom:.5rem;">🎙</div>
                    <div style="font-weight:700;">No answers analyzed yet</div>
                    <div style="color:{THEME['muted']}; font-size:.84rem; margin-top:.3rem;">Ask the interviewer to start, then type the question and record or paste your answer.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Session log (all answered questions this session) ──
    if log:
        st.markdown("<hr style='border-color:rgba(38,53,83,0.5); margin:.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='label' style='margin-bottom:.5rem;'>This Session Log</div>", unsafe_allow_html=True)
        for idx, entry in enumerate(reversed(log), start=1):
            grade_color_map = {"A": THEME["success"], "B": THEME["primary"], "C": THEME["warning"], "D": "#fb923c", "F": THEME["danger"]}
            gc = grade_color_map.get(entry["grade"], THEME["muted"])
            q_short = (entry["question"][:70] + "…") if len(entry["question"]) > 70 else entry["question"]
            st.markdown(
                f"""
                <div class="companion-log-item">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <span class="label">Q{len(log) - idx + 1} · {entry['category']}</span>
                            <div style="font-size:.9rem; font-weight:600; margin-top:.2rem;">{q_short}</div>
                        </div>
                        <div style="text-align:right; min-width:4rem;">
                            <div style="font-family:Manrope,sans-serif; font-size:1.3rem; font-weight:800; color:{gc};">{entry['grade']}</div>
                            <div style="font-size:.72rem; color:{THEME['muted']}">{entry['scores']['final']:.0f}/100</div>
                        </div>
                    </div>
                    <div>
                    """,
                unsafe_allow_html=True,
            )
            _companion_score_bar("Rel", entry["scores"].get("relevance", 0), THEME["primary"])
            _companion_score_bar("Flu", entry["scores"].get("fluency", 0), THEME["primary_2"])
            st.markdown("</div></div>", unsafe_allow_html=True)


# build_student_summary is imported from session_manager at the top of this file.
# The function includes Last Score and Trend columns for Task 6.


def render_interviewer_workspace():
    hero(
        "Interviewer Workplace",
        "Review all students in one place, compare average performance, inspect emotional stability, and drill into each student's interview history.",
        "Recruiter Analytics",
    )
    sessions = load_sessions()
    if not sessions:
        st.info("No student interview data has been saved yet.")
        return



    # ── Sort controls (Task 6) ─────────────────────────────────────────────────
    ctrl_cols = st.columns([1.5, 1.5, 3])
    with ctrl_cols[0]:
        sort_by = st.selectbox(
            "Sort students by",
            ["Avg Score", "Last Score", "Sessions", "ESI", "Trend"],
            key="interviewer_sort",
        )
    with ctrl_cols[1]:
        sort_asc = st.checkbox("Ascending", value=False, key="interviewer_sort_asc")

    # All categories across all students for filter
    all_cats = sorted({
        r.get("category", "-")
        for records in sessions.values()
        for r in records
        if r.get("category")
    })
    with ctrl_cols[2]:
        cat_filter = st.multiselect(
            "Filter by category (leave blank = show all)",
            all_cats,
            key="interviewer_cat_filter",
        )

    # Apply category filter to records before building summaries
    def _filter_records(recs):
        if not cat_filter:
            return recs
        return [r for r in recs if r.get("category") in cat_filter]

    filtered_sessions = {s: _filter_records(r) for s, r in sessions.items()}
    _, _, calculate_esi, _, _ = load_models()
    summaries = [
        build_student_summary(s, r, calculate_esi)
        for s, r in filtered_sessions.items() if r
    ]
    if not summaries:
        st.info("No records match the selected category filter.")
        return

    # Sort
    sort_key = sort_by
    summaries.sort(key=lambda x: x.get(sort_key, 0) if isinstance(x.get(sort_key, 0), (int, float)) else 0,
                   reverse=not sort_asc)

    all_scores = [row["Avg Score"] for row in summaries]
    top = st.columns(4)
    with top[0]:
        metric_card("Students", str(len(summaries)), "Profiles tracked", THEME["primary"])
    with top[1]:
        metric_card("Total Sessions", str(sum(row["Sessions"] for row in summaries)), "Evaluated answers saved", THEME["success"])
    with top[2]:
        metric_card("Overall Avg", f"{sum(all_scores)/len(all_scores):.1f}", "Average student score", THEME["primary_2"])
    with top[3]:
        metric_card("Top Avg", f"{max(all_scores):.1f}", "Best student average", THEME["warning"])

    table_col, chart_col = st.columns([1.25, 1], gap="large")
    with table_col:
        panel_start("All Students")
        st.dataframe(summaries, use_container_width=True, hide_index=True)
        panel_end()

    with chart_col:
        panel_start("Average Score Comparison")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(THEME["panel"])
        ax.set_facecolor(THEME["panel"])
        ax.bar([row["Student"] for row in summaries], [row["Avg Score"] for row in summaries], color=THEME["primary"])
        ax.set_ylim(0, 100)
        ax.set_title("Average score by student", color=THEME["text"])
        ax.tick_params(colors=THEME["text"], rotation=20)
        for spine in ax.spines.values():
            spine.set_color(THEME["line"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        panel_end()

    student_options = [row["Student"] for row in summaries]
    selected_student = st.selectbox("Inspect student", student_options, key="interviewer_student")
    records = sessions[selected_student]

    score_line, info_col = st.columns([1.35, 1], gap="large")
    with score_line:
        panel_start(f"{selected_student} Score Trend")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        fig.patch.set_facecolor(THEME["panel"])
        ax.set_facecolor(THEME["panel"])
        student_scores = [item["scores"]["final"] for item in records]
        ax.plot(range(1, len(student_scores) + 1), student_scores, color=THEME["primary_2"], linewidth=2.8, marker="o")
        ax.fill_between(range(1, len(student_scores) + 1), student_scores, color=THEME["primary_2"], alpha=0.15)
        ax.set_ylim(0, 100)
        ax.set_title("Session-by-session performance", color=THEME["text"])
        ax.tick_params(colors=THEME["text"])
        for spine in ax.spines.values():
            spine.set_color(THEME["line"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        panel_end()

    with info_col:
        panel_start("Student Snapshot")
        summary = next(row for row in summaries if row["Student"] == selected_student)
        st.markdown(f"- Average score: `{summary['Avg Score']}`")
        st.markdown(f"- Best score: `{summary['Best Score']}`")
        st.markdown(f"- Emotional stability index: `{summary['ESI']}`")
        st.markdown(f"- Answers evaluated: `{summary['Sessions']}`")
        st.markdown(f"- Last seen: `{summary['Last Seen']}`")
        panel_end()

    panel_start("Detailed Student Log")
    detail_rows = [
        {
            "Date": record.get("date", "-"),
            "Category": record.get("category", "-"),
            "Question": record.get("question", "-"),
            "Score": record["scores"]["final"],
            "Relevance": record["scores"].get("relevance", 0),
            "Fluency": record["scores"].get("fluency", 0),
            "Sentiment": record["scores"].get("sentiment", 0),
            "Emotion": record.get("emotion", "-"),
            "Grade": record.get("grade", "-"),
        }
        for record in records[::-1]
    ]
    st.dataframe(detail_rows, use_container_width=True, hide_index=True)
    panel_end()


def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='font-family:Manrope,sans-serif; margin-bottom:.2rem;'>ARIES</h2>", unsafe_allow_html=True)
        st.caption("Voice-first interview evaluator")
        st.radio("Workspace", ["Student", "Interviewer"], key="workspace_sidebar", horizontal=True)
        st.session_state.workspace = st.session_state.workspace_sidebar
        if st.session_state.workspace == "Student":
            st.markdown(f"**Candidate:** `{st.session_state.candidate_name or 'Not set'}`")
            st.radio(
                "Screen",
                ["Interview", "Report", "Dashboard", "Live Companion"],
                key="screen_sidebar",
                help="Switch between the live interview, report, analytics dashboard, or the live companion for use alongside Zoom/Meet.",
            )
            st.session_state.screen = st.session_state.screen_sidebar
            if st.session_state.screen == "Live Companion":
                st.markdown(
                    "<div style='font-size:.78rem; color:#9aa9c7; margin-top:.4rem; padding:.5rem .6rem; background:rgba(100,210,255,0.06); border-radius:10px; border-left:2px solid #64d2ff;'>"
                    "💡 <b>Snap tip:</b> Press <kbd>Win + ←</kbd> to snap this window to half-screen, then open Zoom/Meet on the other half."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**Interviewer View:** all student analytics")
            st.session_state.screen = "Dashboard"
        st.markdown("---")
        st.markdown("Use `Interviewer` to compare students and review the shared performance index.")


def render_main():
    questions = load_questions()
    render_sidebar()

    if st.session_state.workspace == "Interviewer":
        render_interviewer_workspace()
        return

    if st.session_state.screen == "Interview":
        if not st.session_state.session_started:
            render_setup(questions)
        else:
            render_live_interview()
    elif st.session_state.screen == "Report":
        render_final_report()
    elif st.session_state.screen == "Live Companion":
        render_live_companion()
    else:
        render_dashboard()


def main():
    configure_page()
    install_styles()
    ensure_state()
    render_main()


if __name__ == "__main__":
    main()
