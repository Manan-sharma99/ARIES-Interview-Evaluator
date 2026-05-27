from interview_app import main

main()
raise SystemExit

import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="ARIES Interview Evaluator", layout="wide", page_icon="ARIES")

SESSIONS_FILE = "models/sessions.json"
THEME = {
    "bg": "#060e20",
    "surface_low": "#091328",
    "surface": "#0f1930",
    "surface_high": "#141f38",
    "surface_bright": "#1f2b49",
    "outline": "#40485d",
    "text": "#dee5ff",
    "muted": "#a3aac4",
    "primary": "#bd9dff",
    "primary_dim": "#8a4cfc",
    "secondary": "#34b5fa",
    "tertiary": "#919bff",
    "success": "#4caf50",
    "warning": "#ff9800",
    "error": "#ff6e84",
}
GRADE_COLORS = {
    "A": THEME["success"],
    "B": THEME["secondary"],
    "C": THEME["warning"],
    "D": "#ff7043",
    "F": THEME["error"],
}
EMOTION_COLORS = {
    "confident": THEME["success"],
    "neutral": THEME["secondary"],
    "nervous": THEME["warning"],
    "stressed": THEME["error"],
}


def install_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
        :root {{
            --bg: {THEME['bg']}; --surface-low: {THEME['surface_low']}; --surface: {THEME['surface']};
            --surface-high: {THEME['surface_high']}; --surface-bright: {THEME['surface_bright']};
            --outline: {THEME['outline']}; --text: {THEME['text']}; --muted: {THEME['muted']};
            --primary: {THEME['primary']}; --primary-dim: {THEME['primary_dim']};
            --secondary: {THEME['secondary']}; --tertiary: {THEME['tertiary']};
            --success: {THEME['success']}; --warning: {THEME['warning']}; --error: {THEME['error']};
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
            background:
                radial-gradient(circle at top left, rgba(138, 76, 252, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(52, 181, 250, 0.10), transparent 24%),
                linear-gradient(180deg, #071122 0%, var(--bg) 100%);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stSidebar"] {{ background: rgba(0, 0, 0, 0.45); border-right: 1px solid rgba(64, 72, 93, 0.2); }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        h1, h2, h3, .headline {{ font-family: 'Manrope', sans-serif; letter-spacing: -0.02em; }}
        .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}
        .top-shell {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:2rem; padding:1rem 1.4rem; background:rgba(15,25,48,.62); backdrop-filter:blur(24px); border:1px solid rgba(64,72,93,.15); border-radius:1.75rem; box-shadow:0 20px 40px rgba(0,0,0,.28); }}
        .brand {{ font-family:'Manrope',sans-serif; font-size:1.7rem; font-weight:800; color:var(--primary); }}
        .status-pill {{ display:inline-flex; align-items:center; gap:.55rem; padding:.55rem .9rem; border-radius:999px; background:rgba(9,19,40,.95); color:var(--secondary); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.16em; }}
        .status-dot {{ width:.55rem; height:.55rem; border-radius:50%; background:var(--secondary); box-shadow:0 0 14px rgba(52,181,250,.7); }}
        .hero-card, .glass-card {{ background:rgba(15,25,48,.70); backdrop-filter:blur(30px); border:1px solid rgba(64,72,93,.15); border-radius:2rem; box-shadow:0 24px 48px rgba(0,0,0,.30); }}
        .hero-card {{ padding:1.75rem 1.9rem; margin-bottom:1.5rem; }}
        .eyebrow {{ color:var(--secondary); font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.22em; }}
        .hero-title {{ margin:.35rem 0 0; font-size:3rem; font-weight:800; line-height:1; }}
        .hero-copy {{ margin-top:.7rem; color:var(--muted); max-width:52rem; line-height:1.65; }}
        .panel-card {{ background:rgba(20,31,56,.78); border-radius:2rem; padding:1.4rem; border:1px solid rgba(64,72,93,.15); margin-bottom:1rem; }}
        .question-card {{ position:relative; overflow:hidden; min-height:240px; }}
        .question-card::before {{ content:''; position:absolute; left:0; top:0; width:6px; height:100%; background:linear-gradient(180deg,var(--primary),var(--primary-dim)); border-radius:999px; }}
        .muted {{ color:var(--muted); }}
        .small-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); font-weight:700; }}
        .question-text {{ font-family:'Manrope',sans-serif; font-size:1.65rem; line-height:1.45; font-weight:700; margin:.75rem 0 1.15rem; }}
        .pill-row {{ display:flex; flex-wrap:wrap; gap:.65rem; }}
        .soft-pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.45rem .85rem; border-radius:999px; background:rgba(31,43,73,.55); color:var(--text); font-size:.82rem; }}
        .tip-card {{ background:rgba(9,19,40,.90); border-radius:1.75rem; padding:1.1rem 1.2rem; border-left:4px solid rgba(52,181,250,.45); }}
        .wave-card {{ background:rgba(0,0,0,.55); border-radius:1.7rem; min-height:240px; display:flex; align-items:center; justify-content:center; margin-bottom:1rem; overflow:hidden; position:relative; }}
        .wave-grid {{ position:absolute; inset:0; display:flex; justify-content:center; align-items:center; gap:7px; opacity:.72; }}
        .wave-grid span {{ display:block; width:7px; border-radius:999px; background:linear-gradient(180deg,var(--primary-dim),var(--primary)); }}
        .wave-center {{ position:relative; text-align:center; z-index:1; }}
        .wave-time {{ font-family:'Manrope',sans-serif; font-size:2.7rem; font-weight:800; }}
        .metric-card {{ background:rgba(20,31,56,.76); border-radius:1.8rem; padding:1.2rem; border-top:1px solid rgba(255,255,255,.06); text-align:center; min-height:150px; }}
        .metric-value {{ font-family:'Manrope',sans-serif; font-size:2.2rem; font-weight:800; margin-top:.4rem; }}
        .grade-card {{ padding:1.5rem; border-radius:1.8rem; text-align:center; background:linear-gradient(135deg, rgba(189,157,255,.10), rgba(52,181,250,.08)); border:1px solid rgba(64,72,93,.16); }}
        .grade-value {{ font-family:'Manrope',sans-serif; font-size:4rem; font-weight:800; line-height:1; }}
        .insight-wrap {{ background:rgba(9,19,40,.85); border-radius:2rem; overflow:hidden; border:1px solid rgba(64,72,93,.16); }}
        .insight-header {{ padding:1.25rem 1.4rem; border-bottom:1px solid rgba(64,72,93,.16); }}
        .insight-body {{ padding:1.3rem 1.4rem; }}
        .table-shell, .chart-shell {{ background:rgba(15,25,48,.70); border-radius:2rem; padding:1rem; border:1px solid rgba(64,72,93,.15); }}
        .launch-shell {{ display:grid; grid-template-columns: minmax(280px, 1fr) minmax(340px, 1.25fr); gap:1.5rem; align-items:stretch; }}
        .launch-panel {{ background:rgba(15,25,48,.72); backdrop-filter:blur(32px); border:1px solid rgba(64,72,93,.15); border-radius:2rem; padding:1.6rem; box-shadow:0 24px 48px rgba(0,0,0,.28); }}
        .launch-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }}
        .launch-card {{ min-height:190px; display:flex; flex-direction:column; justify-content:space-between; border-radius:1.7rem; padding:1.25rem; background:rgba(20,31,56,.82); border-top:1px solid rgba(189,157,255,.12); }}
        .launch-icon {{ width:3.25rem; height:3.25rem; border-radius:999px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; font-weight:800; }}
        .badge-row {{ display:flex; flex-wrap:wrap; gap:.65rem; margin-top:1rem; }}
        .tiny-badge {{ padding:.45rem .8rem; border-radius:999px; background:rgba(25,37,64,.82); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
        .ring-shell {{ width:132px; height:132px; border-radius:50%; display:grid; place-items:center; margin:0 auto .8rem; background:conic-gradient(var(--ring-color) calc(var(--score) * 1%), rgba(64,72,93,.35) 0); }}
        .ring-core {{ width:102px; height:102px; border-radius:50%; background:rgba(9,19,40,.95); display:flex; align-items:center; justify-content:center; flex-direction:column; }}
        .insight-split {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; background:rgba(64,72,93,.14); }}
        .insight-column {{ background:rgba(9,19,40,.92); padding:1.2rem 1.3rem; }}
        .overview-shell {{ display:grid; grid-template-columns:1.15fr .85fr; gap:1.2rem; }}
        .side-stack {{ display:grid; gap:1rem; }}
        .nav-pills [data-baseweb="tab-list"] {{ justify-content:flex-start; flex-wrap:wrap; }}
        @media (max-width: 900px) {{
            .launch-shell, .overview-shell, .insight-split {{ grid-template-columns:1fr; }}
            .launch-grid {{ grid-template-columns:1fr; }}
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap:.6rem; background:transparent; }}
        .stTabs [data-baseweb="tab"] {{ background:rgba(9,19,40,.85); border-radius:999px; color:var(--muted); border:1px solid rgba(64,72,93,.15); padding:.65rem 1rem; font-weight:700; }}
        .stTabs [aria-selected="true"] {{ background:linear-gradient(135deg, rgba(138,76,252,.35), rgba(189,157,255,.18)) !important; color:var(--text) !important; border-color:rgba(189,157,255,.35) !important; }}
        .stButton > button, .stDownloadButton > button {{ border:none; border-radius:999px; padding:.75rem 1.25rem; font-weight:800; letter-spacing:.03em; background:linear-gradient(135deg,var(--primary-dim),var(--primary)); color:#14051f; box-shadow:0 0 24px rgba(189,157,255,.22); }}
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {{ background:rgba(9,19,40,.92) !important; color:var(--text) !important; border:1px solid rgba(64,72,93,.24) !important; border-radius:1rem !important; }}
        .stAlert {{ background:rgba(15,25,48,.82); border:1px solid rgba(64,72,93,.18); color:var(--text); border-radius:1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models():
    from modules.nlp_evaluator import evaluate_answer
    from modules.fluency_analyzer import analyze_fluency
    from modules.scoring_engine import generate_report, calculate_esi
    from modules.emotion_model import predict_emotion
    return evaluate_answer, analyze_fluency, generate_report, calculate_esi, predict_emotion


@st.cache_resource
def load_questions():
    from modules.questions_bank import QUESTIONS
    return QUESTIONS


def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            normalized = {}
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                student_name = entry.get("student") or entry.get("name") or "Unknown Student"
                normalized.setdefault(student_name, [])
                if entry.get("questions"):
                    for question_entry in entry["questions"]:
                        if not isinstance(question_entry, dict):
                            continue
                        normalized[student_name].append(
                            {
                                "date": entry.get("timestamp", "-"),
                                "question": question_entry.get("question", ""),
                                "category": entry.get("category", "Imported Session"),
                                "answer": question_entry.get("answer", ""),
                                "emotion": question_entry.get("emotion", "neutral"),
                                "scores": {
                                    "final": question_entry.get("final_score", entry.get("average_score", 0)),
                                    "relevance": question_entry.get("relevance", 0),
                                    "fluency": question_entry.get("fluency", 0),
                                    "sentiment": question_entry.get("sentiment", 0),
                                },
                                "grade": question_entry.get("grade", entry.get("grade", "N/A")),
                                "grade_label": entry.get("grade_label", ""),
                                "mode": entry.get("mode", "practice"),
                            }
                        )
                else:
                    normalized[student_name].append(
                        {
                            "date": entry.get("timestamp", "-"),
                            "category": entry.get("category", "Imported Session"),
                            "emotion": entry.get("emotion", "neutral"),
                            "scores": {"final": entry.get("average_score", 0)},
                            "grade": entry.get("grade", "N/A"),
                            "grade_label": entry.get("grade_label", ""),
                            "mode": entry.get("mode", "practice"),
                        }
                    )
            return normalized
    return {}


def save_sessions(sessions):
    os.makedirs("models", exist_ok=True)
    with open(SESSIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(sessions, file, indent=2)


def render_shell_header(interface_label, session_label):
    st.markdown(f"<div class='top-shell'><div><div class='brand'>ARIES</div><div class='muted' style='font-size:0.9rem;'>{interface_label}</div></div><div class='status-pill'><span class='status-dot'></span>{session_label}</div></div>", unsafe_allow_html=True)


def render_hero(title, eyebrow, copy):
    st.markdown(f"<section class='hero-card'><div class='eyebrow'>{eyebrow}</div><h1 class='hero-title'>{title}</h1><div class='hero-copy'>{copy}</div></section>", unsafe_allow_html=True)


def set_student_view(view_name):
    st.session_state.student_view = view_name


def render_launchpad(name):
    candidate_label = name if name else "Candidate"
    st.markdown(
        f"""
        <section class='hero-card'>
            <div class='eyebrow'>AI-Powered Training Active</div>
            <h1 class='hero-title'>Master Your Next Interview with ARIES</h1>
            <div class='hero-copy'>This launchpad maps to your first mockup screen. Use it as the polished entry into practice, timed testing, and progress review for {candidate_label}.</div>
        </section>
        <div class='launch-shell'>
            <div class='launch-panel'>
                <div class='small-label'>Candidate Name</div>
                <div style='font-family:Manrope,sans-serif; font-size:2rem; font-weight:800; margin-top:.45rem;'>{candidate_label}</div>
                <div class='muted' style='margin-top:.55rem; line-height:1.7;'>Launch a guided interview flow, jump straight into the practice studio, or review past sessions in the performance deck.</div>
                <div class='badge-row'>
                    <div class='tiny-badge' style='color:var(--primary);'>Biometric Analysis</div>
                    <div class='tiny-badge' style='color:var(--secondary);'>Real-time STT</div>
                    <div class='tiny-badge' style='color:var(--tertiary);'>Sentiment Mapping</div>
                </div>
            </div>
            <div class='launch-grid'>
                <div class='launch-card'>
                    <div class='launch-icon' style='background:rgba(52,181,250,.15); color:var(--secondary);'>HR</div>
                    <div>
                        <div style='font-family:Manrope,sans-serif; font-size:1.35rem; font-weight:800;'>HR Behavioral</div>
                        <div class='muted' style='margin-top:.35rem;'>Practice STAR-based leadership and communication answers.</div>
                    </div>
                </div>
                <div class='launch-card'>
                    <div class='launch-icon' style='background:rgba(189,157,255,.15); color:var(--primary);'>DSA</div>
                    <div>
                        <div style='font-family:Manrope,sans-serif; font-size:1.35rem; font-weight:800;'>DSA Patterns</div>
                        <div class='muted' style='margin-top:.35rem;'>Train algorithmic reasoning and structured explanation.</div>
                    </div>
                </div>
                <div class='launch-card'>
                    <div class='launch-icon' style='background:rgba(145,155,255,.16); color:var(--tertiary);'>SD</div>
                    <div>
                        <div style='font-family:Manrope,sans-serif; font-size:1.35rem; font-weight:800;'>System Design</div>
                        <div class='muted' style='margin-top:.35rem;'>Rehearse scalable thinking, tradeoffs, and architecture calls.</div>
                    </div>
                </div>
                <div class='launch-card'>
                    <div class='launch-icon' style='background:rgba(23,168,236,.14); color:#81ccff;'>ML</div>
                    <div>
                        <div style='font-family:Manrope,sans-serif; font-size:1.35rem; font-weight:800;'>ML & Data</div>
                        <div class='muted' style='margin-top:.35rem;'>Cover model evaluation, pipelines, and practical ML communication.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1.2, 1.2, 1])
    with action_cols[0]:
        if st.button("Start Practice", use_container_width=True):
            set_student_view("Practice Studio")
            st.rerun()
    with action_cols[1]:
        if st.button("Open Skill Test", use_container_width=True):
            set_student_view("Skill Test Arena")
            st.rerun()
    with action_cols[2]:
        if st.button("View Dashboard", use_container_width=True):
            set_student_view("Performance Deck")
            st.rerun()


def render_question_panel(question, category):
    st.markdown(f"<div class='panel-card question-card'><div class='small-label'>Active Question</div><div class='question-text'>{question}</div><div class='pill-row'><div class='soft-pill'>Category: {category}</div><div class='soft-pill'>Focus: Conceptual depth</div><div class='soft-pill'>Format: Practice session</div></div></div>", unsafe_allow_html=True)


def render_tip_panel(category):
    tip_map = {
        "Behavioral": "Answer in a STAR structure so the evaluator can pick up action, impact, and ownership clearly.",
        "System Design": "State tradeoffs early. Strong answers compare scale, latency, reliability, and cost in plain language.",
        "Machine Learning": "Pair the definition with one concrete example so your answer sounds grounded, not memorized.",
        "Amazon Leadership Principles": "Tie your answer to one principle with a measurable result so it feels interviewer-ready.",
    }
    tip = tip_map.get(category, "Use one concrete example and one clear takeaway. It makes your answer sound confident and easier to score highly.")
    st.markdown(f"<div class='tip-card'><div class='small-label'>Interview Tip</div><div style='margin-top:0.35rem; line-height:1.7;'>{tip}</div></div>", unsafe_allow_html=True)


def render_wave_panel(answer_text):
    heights = [28, 52, 84, 66, 102, 132, 88, 118, 64, 108, 74, 138, 92, 54, 34]
    bars = "".join(f"<span style='height:{height}px'></span>" for height in heights)
    words = len(answer_text.split()) if answer_text else 0
    time_label = f"{min(words, 99):02d}:{(words * 3) % 60:02d}"
    st.markdown(f"<div class='wave-card'><div class='wave-grid'>{bars}</div><div class='wave-center'><div class='wave-time'>{time_label}</div><div class='small-label' style='margin-top:0.45rem;'>Response in progress</div></div></div>", unsafe_allow_html=True)


def render_metric_card(label, value, caption, color):
    st.markdown(f"<div class='metric-card'><div class='small-label'>{label}</div><div class='metric-value' style='color:{color};'>{value}</div><div class='muted' style='font-size:0.8rem; margin-top:0.25rem;'>{caption}</div></div>", unsafe_allow_html=True)


def render_score_ring(label, value, caption, color):
    safe_value = max(0, min(100, float(value)))
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='ring-shell' style='--score:{safe_value}; --ring-color:{color};'>
                <div class='ring-core'>
                    <div style='font-family:Manrope,sans-serif; font-size:1.45rem; font-weight:800; color:var(--text);'>{safe_value:.0f}%</div>
                </div>
            </div>
            <div class='small-label'>{label}</div>
            <div class='muted' style='font-size:0.8rem; margin-top:0.25rem;'>{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grade_card(grade, label):
    color = GRADE_COLORS.get(grade, THEME["muted"])
    st.markdown(f"<div class='grade-card'><div class='small-label'>Final grade</div><div class='grade-value' style='color:{color};'>{grade}</div><div class='muted' style='margin-top:0.25rem;'>{label}</div></div>", unsafe_allow_html=True)


def render_emotion_card(emotion):
    color = EMOTION_COLORS.get(emotion, THEME["muted"])
    st.markdown(f"<div class='grade-card'><div class='small-label'>Detected emotion</div><div class='metric-value' style='color:{color}; text-transform:capitalize;'>{emotion}</div><div class='muted' style='margin-top:0.25rem;'>Emotional signal from your latest attempt</div></div>", unsafe_allow_html=True)

def render_feedback_panel(report, nlp, fluency, emotion):
    strengths = [
        f"Your answer stayed relevant to the prompt with a relevance score of {nlp['relevance_score']:.0f}/100.",
        f"Fluency measured at {fluency['fluency_score']:.0f}/100, showing how easy the response was to follow.",
        f"The overall evaluation ended at {report['scores']['final']:.0f}/100 with grade {report['grade']}.",
    ]
    growth = [
        "Add one sharper example or measurable outcome to make the response feel more interview-ready.",
        "Aim for concise structure: opening point, example, takeaway. That usually improves both relevance and fluency.",
    ]
    if fluency["filler_count"] > 0:
        growth.append(f"Filler words detected: {', '.join(fluency['found_fillers'])}.")
    if emotion in {"nervous", "stressed"}:
        growth.append("Try a slower first sentence. A calmer opening often improves both delivery and confidence signals.")
    left = "".join(f"<li style='margin-bottom:0.8rem;'>{item}</li>" for item in strengths)
    right = "".join(f"<li style='margin-bottom:0.8rem;'>{item}</li>" for item in growth)
    st.markdown(
        f"<div class='insight-wrap'><div class='insight-header'><div class='eyebrow' style='color:var(--primary);'>Evaluation Complete</div><div style='font-family:Manrope,sans-serif; font-size:1.6rem; font-weight:800; margin-top:0.35rem;'>Candidate Analysis</div><div class='muted' style='margin-top:0.35rem;'>Your latest answer has been translated into ARIES-style feedback.</div></div><div class='insight-split'><div class='insight-column'><div class='small-label' style='color:var(--secondary); margin-bottom:0.9rem;'>Key strengths</div><ul style='padding-left:1.1rem; line-height:1.7; margin:0;'>{left}</ul></div><div class='insight-column'><div class='small-label' style='color:var(--tertiary); margin-bottom:0.9rem;'>Growth opportunities</div><ul style='padding-left:1.1rem; line-height:1.7; margin:0;'>{right}</ul></div></div></div>",
        unsafe_allow_html=True,
    )
    


def get_ai_feedback(question, answer, category):
    try:
        from modules.ai_evaluator import evaluate_with_ai, display_ai_feedback
        with st.spinner("ARIES AI is analyzing your answer..."):
            result = evaluate_with_ai(question, answer, category)
        display_ai_feedback(result, st)
    except Exception as exc:
        st.warning(f"AI feedback unavailable: {exc}")


def capture_voice_answer(duration):
    try:
        import sounddevice as sd
        import soundfile as sf
        import speech_recognition as sr
        with st.spinner(f"Recording for {duration} seconds..."):
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            sf.write("temp_recording.wav", audio, 16000)
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_recording.wav") as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
        st.session_state.practice_answer = transcript
        st.success("Voice answer captured and transcribed.")
    except Exception as exc:
        st.error(f"Recording error: {exc}")


def evaluate_current_answer(name, category, question):
    answer = st.session_state.practice_answer.strip()
    if not answer:
        st.warning("Please enter or record an answer before evaluation.")
        return
    evaluate_answer, analyze_fluency, generate_report, _, predict_emotion = load_models()
    with st.spinner("Analyzing your answer..."):
        nlp = evaluate_answer(question, answer)
        fluency = analyze_fluency(answer)
        emotion = "neutral"
        if os.path.exists("temp_recording.wav"):
            try:
                emotion = predict_emotion("temp_recording.wav")
            except Exception:
                emotion = "neutral"
        report = generate_report(question, answer, emotion, nlp["relevance_score"], fluency["fluency_score"], nlp["sentiment_score"])
    sessions = load_sessions()
    sessions.setdefault(name, [])
    sessions[name].append(
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "question": question,
            "category": category,
            "answer": answer,
            "emotion": emotion,
            "scores": report["scores"],
            "grade": report["grade"],
            "grade_label": report.get("grade_label", ""),
            "mode": "practice",
        }
    )
    save_sessions(sessions)
    st.session_state.last_result = {"question": question, "category": category, "answer": answer, "emotion": emotion, "nlp": nlp, "fluency": fluency, "report": report}
    st.success("Session saved.")


def render_student_practice(name, questions, all_categories):
    render_hero("Practice Interview Session", "Module: Adaptive Interview Lab", "Use the mockup-inspired ARIES workspace below to practice, record, and review answers without changing your scoring pipeline.")
    controls = st.columns([1.1, 1.1, 1])
    with controls[0]:
        category = st.selectbox("Select category", all_categories, key="practice_category")
    with controls[1]:
        question = st.selectbox("Select question", questions[category], key="practice_question")
    with controls[2]:
        input_method = st.radio("Answer method", ["Type answer", "Voice recording"], horizontal=True)
    left, right = st.columns([1.02, 1.2], gap="large")
    with left:
        render_question_panel(question, category)
        render_tip_panel(category)
    with right:
        render_wave_panel(st.session_state.practice_answer)
        st.text_area("Response draft / notes", key="practice_answer", height=220, placeholder="Type your answer here. A strong answer usually explains the idea, gives one example, and ends with the key takeaway.")
        word_count = len(st.session_state.practice_answer.split()) if st.session_state.practice_answer else 0
        st.caption(f"Word count: {word_count} | Recommended range: 50-150 words")
        action_cols = st.columns([1, 1, 1.2])
        with action_cols[0]:
            if input_method == "Voice recording":
                duration = st.slider("Recording duration (sec)", 15, 60, 30, key="record_duration")
                if st.button("Capture voice", use_container_width=True):
                    capture_voice_answer(duration)
            else:
                st.markdown("<div class='small-label' style='padding-top:0.7rem;'>Typing mode active</div>", unsafe_allow_html=True)
        with action_cols[1]:
            if st.button("Clear response", use_container_width=True):
                st.session_state.practice_answer = ""
                st.session_state.last_result = None
                st.rerun()
        with action_cols[2]:
            if st.button("Evaluate response", type="primary", use_container_width=True):
                evaluate_current_answer(name, category, question)
    result = st.session_state.get("last_result")
    if result:
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        score_cols = st.columns(4)
        with score_cols[0]:
            render_score_ring("Relevance", result["nlp"]["relevance_score"], "Technical alignment", THEME["secondary"])
        with score_cols[1]:
            render_score_ring("Emotion", result["report"]["scores"].get("emotion", 65), "EQ and confidence", THEME["primary"])
        with score_cols[2]:
            render_score_ring("Fluency", result["fluency"]["fluency_score"], "Delivery clarity", THEME["tertiary"])
        with score_cols[3]:
            render_score_ring("Final Score", result["report"]["scores"]["final"], "Overall evaluation", "#81ccff")
        spotlight = st.columns(2)
        with spotlight[0]:
            render_grade_card(result["report"]["grade"], result["report"].get("grade_label", "Evaluation summary"))
        with spotlight[1]:
            render_emotion_card(result["emotion"])
        render_feedback_panel(result["report"], result["nlp"], result["fluency"], result["emotion"])
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        get_ai_feedback(result["question"], result["answer"], result["category"])
        quick_actions = st.columns([1.1, 1.1, 1.2])
        with quick_actions[0]:
            if st.button("Open Performance Deck", use_container_width=True, key="goto_dashboard_after_eval"):
                set_student_view("Performance Deck")
                st.rerun()
        with quick_actions[1]:
            if st.button("Start Another Session", use_container_width=True, key="goto_launch_after_eval"):
                set_student_view("Launchpad")
                st.rerun()
        with quick_actions[2]:
            st.markdown("<div class='small-label' style='padding-top:.85rem;'>Latest result screen matches mockup 2</div>", unsafe_allow_html=True)


def render_skill_test(name, questions, all_categories):
    render_hero("Skill Test Arena", "Timed multi-question run", "Launch a five-question challenge and get a single performance summary at the end, using the same scoring engine behind practice mode.")
    if "test_questions" not in st.session_state:
        st.session_state.test_questions = []
        st.session_state.test_category = all_categories[0]
        st.session_state.test_answers = {}
        st.session_state.test_result = None
    top = st.columns([1, 3])
    with top[0]:
        if st.button("New test", use_container_width=True):
            chosen_category = random.choice(all_categories)
            pool = questions[chosen_category]
            st.session_state.test_questions = random.sample(pool, min(5, len(pool)))
            st.session_state.test_category = chosen_category
            st.session_state.test_answers = {question: "" for question in st.session_state.test_questions}
            st.session_state.test_result = None
    with top[1]:
        st.markdown(f"<div class='panel-card'><div class='small-label'>Current category</div><div style='font-family:Manrope,sans-serif; font-size:1.6rem; font-weight:800; margin-top:0.35rem;'>{st.session_state.test_category}</div></div>", unsafe_allow_html=True)
    if not st.session_state.test_questions:
        st.info("Start a new test to generate five interview questions.")
        return
    for index, question in enumerate(st.session_state.test_questions, start=1):
        st.markdown(f"<div class='panel-card'><div class='small-label'>Question {index}</div><div style='font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:700; margin-top:0.4rem;'>{question}</div></div>", unsafe_allow_html=True)
        answer = st.text_area(f"Answer {index}", value=st.session_state.test_answers.get(question, ""), key=f"test_answer_{index}", height=120, label_visibility="collapsed", placeholder="Type your answer here...")
        st.session_state.test_answers[question] = answer
    if st.button("Submit test", type="primary"):
        submitted_answers = {q: a.strip() for q, a in st.session_state.test_answers.items() if a.strip()}
        if not submitted_answers:
            st.warning("Please answer at least one question before submitting.")
        else:
            evaluate_answer, analyze_fluency, generate_report, _, _ = load_models()
            results = []
            for question, answer in submitted_answers.items():
                nlp = evaluate_answer(question, answer)
                fluency = analyze_fluency(answer)
                report = generate_report(question, answer, "neutral", nlp["relevance_score"], fluency["fluency_score"], nlp["sentiment_score"])
                results.append(report["scores"]["final"])
            average = round(sum(results) / len(results), 2)
            grade = "A" if average >= 85 else "B" if average >= 70 else "C" if average >= 55 else "D" if average >= 40 else "F"
            labels = {"A": "Excellent", "B": "Good", "C": "Average", "D": "Needs Improvement", "F": "Poor"}
            st.session_state.test_result = {"average": average, "grade": grade, "label": labels[grade], "scores": results}
            sessions = load_sessions()
            sessions.setdefault(name, [])
            sessions[name].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "category": st.session_state.test_category, "scores": {"final": average}, "grade": grade, "grade_label": labels[grade], "mode": "skill_test", "num_questions": len(results)})
            save_sessions(sessions)
            st.success("Skill test saved.")
    if st.session_state.test_result:
        result = st.session_state.test_result
        cols = st.columns([1, 1.5])
        with cols[0]:
            render_grade_card(result["grade"], result["label"])
        with cols[1]:
            st.markdown("<div class='chart-shell'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            fig.patch.set_facecolor(THEME["surface"])
            ax.set_facecolor(THEME["surface_low"])
            values = result["scores"]
            labels = [f"Q{i}" for i in range(1, len(values) + 1)]
            colors = [THEME["success"] if score >= 70 else THEME["warning"] if score >= 50 else THEME["error"] for score in values]
            ax.bar(labels, values, color=colors)
            ax.set_ylim(0, 100)
            ax.tick_params(colors=THEME["text"])
            ax.set_ylabel("Score", color=THEME["text"])
            ax.set_title("Question-wise Performance", color=THEME["text"])
            for spine in ax.spines.values():
                spine.set_color(THEME["outline"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard(name):
    render_hero("ARIES Performance Dashboard", "Student analytics terminal", "Your saved sessions now render in a more editorial dashboard, with progress, emotion stability, and a cleaner attempt history.")
    sessions = load_sessions()
    user_sessions = sessions.get(name, [])
    if not user_sessions:
        st.info("No sessions yet. Complete a practice round or skill test to populate the dashboard.")
        return
    scores = [session["scores"]["final"] for session in user_sessions]
    practice_sessions = [session for session in user_sessions if session.get("mode") == "practice"]
    categories = [session.get("category", "Unknown") for session in user_sessions]
    best_category = max(categories, key=lambda cat: sum(s["scores"]["final"] for s in user_sessions if s.get("category") == cat))
    avg_score = round(sum(scores) / len(scores), 1)
    metrics = st.columns(3)
    with metrics[0]:
        render_metric_card("Total Attempts", str(len(scores)), "+ live session history", THEME["primary"])
    with metrics[1]:
        render_metric_card("Average Score", f"{avg_score}", "Across all saved sessions", THEME["secondary"])
    with metrics[2]:
        render_metric_card("Top Category", best_category, "Best performing theme", THEME["tertiary"])
    st.markdown("<div class='overview-shell'>", unsafe_allow_html=True)
    charts = st.columns([1.6, 1], gap="large")
    with charts[0]:
        st.markdown("<div class='chart-shell'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(THEME["surface"])
        ax.set_facecolor(THEME["surface_low"])
        ax.plot(range(len(scores)), scores, color=THEME["primary"], linewidth=2.8, marker="o")
        ax.fill_between(range(len(scores)), scores, alpha=0.18, color=THEME["primary"])
        ax.set_ylim(0, 100)
        ax.set_title("Score Trend", color=THEME["text"])
        ax.set_ylabel("Score", color=THEME["text"])
        ax.tick_params(colors=THEME["text"])
        for spine in ax.spines.values():
            spine.set_color(THEME["outline"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with charts[1]:
        st.markdown("<div class='side-stack'>", unsafe_allow_html=True)
        emotions = [session.get("emotion", "neutral") for session in practice_sessions if session.get("emotion")]
        _, _, _, calculate_esi, _ = load_models()
        esi = calculate_esi(emotions) if emotions else 0.0
        render_metric_card("Emotion Stability", f"{esi:.2f}", "Consistency over practice sessions", THEME["secondary"])
        st.progress(float(esi) if esi <= 1 else float(esi) / 100)
        render_metric_card("Best Score", f"{max(scores):.0f}", "Peak recorded performance", THEME["success"])
        if practice_sessions:
            dominant_emotion = max(
                set(emotions or ["neutral"]),
                key=lambda emotion_name: (emotions or ["neutral"]).count(emotion_name),
            )
            render_metric_card("Primary Emotion", dominant_emotion.title(), "Most common detected state", EMOTION_COLORS.get(dominant_emotion, THEME["muted"]))
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='table-shell'>", unsafe_allow_html=True)
    table_rows = []
    for index, session in enumerate(reversed(user_sessions[-10:]), start=1):
        table_rows.append({"Session": index, "Date": session.get("date", "-"), "Mode": session.get("mode", "practice"), "Category": session.get("category", "-"), "Score": session["scores"]["final"], "Grade": session.get("grade", "-"), "Emotion": session.get("emotion", "-")})
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_interviewer_interface():
    render_hero("Interviewer Intelligence Desk", "Recruiter mode", "Browse student performance, drill into attempts, and export a PDF report using the same session data already saved by the student workflow.")
    sessions = load_sessions()
    if not sessions:
        st.info("No student data yet.")
        return
    all_scores = [entry["scores"]["final"] for student_sessions in sessions.values() for entry in student_sessions]
    if all_scores:
        metrics = st.columns(4)
        with metrics[0]:
            render_metric_card("Students", str(len(sessions)), "Tracked profiles", THEME["primary"])
        with metrics[1]:
            render_metric_card("Sessions", str(sum(len(s) for s in sessions.values())), "Saved evaluations", THEME["secondary"])
        with metrics[2]:
            render_metric_card("Average Score", f"{sum(all_scores)/len(all_scores):.1f}", "Across all students", THEME["tertiary"])
        with metrics[3]:
            render_metric_card("Best Session", f"{max(all_scores):.0f}", "Highest recorded score", THEME["success"])
    tab1, tab2, tab3 = st.tabs(["All Students", "Student Detail", "Export PDF"])
    with tab1:
        overview = []
        for student_name, student_sessions in sessions.items():
            if student_sessions:
                scores = [entry["scores"]["final"] for entry in student_sessions]
                overview.append({"Student": student_name, "Sessions": len(student_sessions), "Avg Score": round(sum(scores) / len(scores), 1), "Best Score": max(scores), "Last Grade": student_sessions[-1].get("grade", "N/A")})
        if overview:
            left, right = st.columns([1.15, 0.85], gap="large")
            with left:
                st.markdown("<div class='table-shell'>", unsafe_allow_html=True)
                st.dataframe(overview, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.markdown("<div class='chart-shell'>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor(THEME["surface"])
                ax.set_facecolor(THEME["surface_low"])
                ax.bar([row["Student"] for row in overview], [row["Avg Score"] for row in overview], color=THEME["primary"])
                ax.set_ylim(0, 100)
                ax.tick_params(colors=THEME["text"], rotation=20)
                ax.set_title("Average Scores by Student", color=THEME["text"])
                for spine in ax.spines.values():
                    spine.set_color(THEME["outline"])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown("</div>", unsafe_allow_html=True)
    with tab2:
        student = st.selectbox("Select student", list(sessions.keys()), key="student_detail")
        if student:
            total = len(sessions[student])
            for index, session in enumerate(reversed(sessions[student]), start=1):
                title = f"Session {total - index + 1} | {session.get('date', '-')} | Score: {session['scores']['final']} | Grade: {session.get('grade', '-')}"
                with st.expander(title, expanded=index == 1):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Category: {session.get('category', 'N/A')}")
                        st.write(f"Mode: {session.get('mode', 'practice')}")
                        if "question" in session:
                            st.write(f"Question: {session['question']}")
                    with col2:
                        st.write(f"Final Score: {session['scores']['final']}/100")
                        st.write(f"Grade: {session.get('grade', '-')}")
                        if "emotion" in session:
                            st.write(f"Emotion: {session['emotion']}")
    with tab3:
        student_pdf = st.selectbox("Select student for PDF", list(sessions.keys()), key="pdf_student")
        if st.button("Generate PDF"):
            try:
                from fpdf import FPDF
                student_sessions = sessions[student_pdf]
                scores = [entry["scores"]["final"] for entry in student_sessions]
                average = round(sum(scores) / len(scores), 1) if scores else 0
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.set_text_color(100, 126, 234)
                pdf.cell(0, 15, "ARIES Interview Evaluator Report", ln=True, align="C")
                pdf.set_font("Arial", "B", 14)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 10, f"Student: {student_pdf}", ln=True)
                pdf.cell(0, 10, f"Total Sessions: {len(student_sessions)}", ln=True)
                pdf.cell(0, 10, f"Average Score: {average}/100", ln=True)
                pdf.cell(0, 10, f"Best Score: {max(scores) if scores else 0}/100", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.ln(5)
                for index, session in enumerate(student_sessions[-10:], start=1):
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, f"Session {index} - {session.get('date', '-')}", ln=True)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 6, f"Score: {session['scores']['final']}/100 | Grade: {session.get('grade', '-')} | Category: {session.get('category', 'N/A')}", ln=True)
                    pdf.ln(2)
                pdf_path = f"models/{student_pdf}_report.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as file:
                    st.download_button("Download PDF", file, file_name=f"{student_pdf}_report.pdf", mime="application/pdf")
                st.success("PDF generated.")
            except Exception as exc:
                st.error(f"PDF error: {exc}")


def main():
    install_styles()
    if "practice_answer" not in st.session_state:
        st.session_state.practice_answer = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "student_view" not in st.session_state:
        st.session_state.student_view = "Launchpad"
    questions = load_questions()
    all_categories = list(questions.keys())
    with st.sidebar:
        st.markdown("<div class='brand' style='font-size:1.5rem;'>ARIES</div>", unsafe_allow_html=True)
        st.markdown("<div class='muted' style='margin-bottom:1rem;'>Cinematic interview evaluation workspace</div>", unsafe_allow_html=True)
        interface = st.radio("Workspace", ["Student", "Interviewer"], horizontal=True)
        name = ""
        if interface == "Student":
            name = st.text_input("Candidate name", placeholder="e.g. Manan Sharma")
            st.markdown("<div class='small-label' style='margin-top:0.8rem;'>Screens</div>", unsafe_allow_html=True)
            st.session_state.student_view = st.radio(
                "Student screen",
                ["Launchpad", "Practice Studio", "Skill Test Arena", "Performance Deck"],
                key="student_view_picker",
                label_visibility="collapsed",
                index=["Launchpad", "Practice Studio", "Skill Test Arena", "Performance Deck"].index(st.session_state.student_view),
            )
            st.caption("All four student mockup screens are mapped here.")
        else:
            st.caption("Review all saved student sessions and export reports.")
    render_shell_header(f"{interface} Workspace", "Session Active")
    if interface == "Student":
        if not name:
            render_hero("ARIES Interview Evaluator", "Student workspace", "Enter your name in the sidebar to unlock the practice studio, skill test arena, and personal performance dashboard.")
            return
        sessions = load_sessions()
        sessions.setdefault(name, [])
        save_sessions(sessions)
        if st.session_state.student_view == "Launchpad":
            render_launchpad(name)
        elif st.session_state.student_view == "Practice Studio":
            render_student_practice(name, questions, all_categories)
        elif st.session_state.student_view == "Skill Test Arena":
            render_skill_test(name, questions, all_categories)
        else:
            render_dashboard(name)
    else:
        render_interviewer_interface()


if __name__ == "__main__":
    main()
