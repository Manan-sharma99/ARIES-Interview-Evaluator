import streamlit as st
import json
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI Interview Evaluator", layout="wide", page_icon="🎤")

# ── Dark theme ──
st.markdown("""
<style>
body, .stApp { background-color: #0f1117; color: #e0e0e0; }
.stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 12px;
    padding: 10px 24px; font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }
.stSelectbox>div>div, .stTextArea>div>div {
    background-color: #1e2433 !important; color: #e0e0e0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
div[data-testid="metric-container"] {
    background: #1e2433; border: 1px solid #2d3748;
    border-radius: 12px; padding: 16px;
}
.stTabs [data-baseweb="tab"] { background: #1e2433; border-radius: 8px 8px 0 0; color: #90a4ae; }
.stTabs [aria-selected="true"] { background: #667eea !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Load modules ──
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

# ── Sessions ──
SESSIONS_FILE = "models/sessions.json"
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    os.makedirs("models", exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

# ── Score card ──
def score_card(label, score, max_score=100):
    score = float(score)
    pct = score / max_score
    color = "#4caf50" if pct >= 0.75 else "#ff9800" if pct >= 0.5 else "#f44336"
    st.markdown(f"""
    <div style="background:#1e2433; border:1px solid {color}; border-radius:12px;
         padding:16px; text-align:center; margin:4px;">
        <div style="color:#90a4ae; font-size:0.8rem;">{label}</div>
        <div style="color:{color}; font-size:2rem; font-weight:800;">{score:.0f}</div>
        <div style="color:#555; font-size:0.75rem;">/ {max_score}</div>
    </div>""", unsafe_allow_html=True)

def grade_badge(grade, label):
    colors = {"A":"#4caf50","B":"#2196f3","C":"#ff9800","D":"#ff5722","F":"#f44336"}
    c = colors.get(grade, "#90a4ae")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{c}22,{c}44); border:2px solid {c};
         border-radius:16px; padding:20px; text-align:center; margin:8px 0;">
        <div style="font-size:3rem; font-weight:900; color:{c};">{grade}</div>
        <div style="color:{c}; font-size:1rem;">{label}</div>
    </div>""", unsafe_allow_html=True)

# ── AI Feedback ──
def get_ai_feedback(question, answer, category):
    try:
        from modules.ai_evaluator import evaluate_with_ai, display_ai_feedback
        with st.spinner("🤖 AI is analyzing your answer..."):
            result = evaluate_with_ai(question, answer, category)
        display_ai_feedback(result, st)
    except Exception as e:
        st.warning(f"AI feedback unavailable: {e}")

# ── MAIN ──
QUESTIONS = load_questions()
ALL_CATEGORIES = list(QUESTIONS.keys())

st.markdown("""
<div style="text-align:center; padding:24px 0 8px;">
    <h1 style="background:linear-gradient(135deg,#667eea,#764ba2);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
       font-size:2.8rem; font-weight:900;">🎤 AI Interview Evaluator</h1>
    <p style="color:#90a4ae;">Practice with Google, Amazon, Microsoft, Meta & Apple questions</p>
</div>""", unsafe_allow_html=True)

# ── Interface selector ──
interface = st.radio("", ["🎓 Student", "👔 Interviewer"], horizontal=True)

# ══════════════════════════════════════════
# STUDENT INTERFACE
# ══════════════════════════════════════════
if interface == "🎓 Student":
    name = st.text_input("👤 Enter your name to start", placeholder="e.g. Manan Sharma")
    if not name:
        st.info("Enter your name above to begin.")
        st.stop()

    sessions = load_sessions()
    if name not in sessions:
        sessions[name] = []

    tab1, tab2, tab3 = st.tabs(["💬 Practice Mode", "📝 Skill Test", "📊 My Progress"])

    # ── PRACTICE MODE ──
    with tab1:
        st.subheader("Practice Mode")
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Select Category", ALL_CATEGORIES)
        with col2:
            question = st.selectbox("Select Question", QUESTIONS[category])

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e2433,#252d3d);
             border-left:4px solid #667eea; border-radius:12px; padding:20px; margin:16px 0;">
            <div style="color:#90a4ae; font-size:0.85rem;">Interview Question</div>
            <div style="color:#fff; font-size:1.2rem; font-weight:600; margin-top:8px;">{question}</div>
        </div>""", unsafe_allow_html=True)

        input_method = st.radio("Answer Method", ["⌨️ Type Answer", "🎤 Voice Recording"], horizontal=True)

        user_answer = ""
        if input_method == "⌨️ Type Answer":
            user_answer = st.text_area("Your Answer", height=150,
                placeholder="Type your answer here... (aim for 3-5 sentences)")
            word_count = len(user_answer.split()) if user_answer else 0
            st.caption(f"Word count: {word_count} | Recommended: 50-150 words")

        else:
            duration = st.slider("Recording Duration (seconds)", 15, 60, 30)
            if st.button("🎤 Start Recording"):
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    import speech_recognition as sr
                    with st.spinner(f"Recording for {duration} seconds..."):
                        audio = sd.rec(int(duration * 16000), samplerate=16000,
                                      channels=1, dtype='int16')
                        sd.wait()
                        sf.write("temp_recording.wav", audio, 16000)
                    st.success("Recording complete!")
                    r = sr.Recognizer()
                    with sr.AudioFile("temp_recording.wav") as source:
                        audio_data = r.record(source)
                    user_answer = r.recognize_google(audio_data)
                    st.text_area("Transcribed Answer", user_answer, height=100)
                except Exception as e:
                    st.error(f"Recording error: {e}")

        if st.button("🚀 Evaluate My Answer", type="primary") and user_answer.strip():
            try:
                evaluate_answer, analyze_fluency, generate_report, calculate_esi, predict_emotion = load_models()

                with st.spinner("Analyzing your answer..."):
                    nlp = evaluate_answer(question, user_answer)
                    fluency = analyze_fluency(user_answer)

                    emotion = "neutral"
                    if os.path.exists("temp_recording.wav"):
                        try:
                            emotion = predict_emotion("temp_recording.wav")
                        except:
                            pass

                    report = generate_report(question, user_answer, emotion,
                                           nlp['relevance_score'],
                                           fluency['fluency_score'],
                                           nlp['sentiment_score'])

                st.markdown("---")
                st.subheader("📊 Your Results")

                cols = st.columns(4)
                with cols[0]: score_card("Relevance", nlp['relevance_score'])
                with cols[1]: score_card("Fluency", fluency['fluency_score'])
                with cols[2]: score_card("Sentiment", nlp['sentiment_score'])
                with cols[3]: score_card("Final Score", report['scores']['final'])

                col_g, col_e = st.columns(2)
                with col_g:
                    grade_badge(report['grade'], report['grade_label'])
                with col_e:
                    emotion_colors = {"confident":"#4caf50","neutral":"#2196f3",
                                     "nervous":"#ff9800","stressed":"#f44336"}
                    ec = emotion_colors.get(emotion, "#90a4ae")
                    st.markdown(f"""
                    <div style="background:#1e2433; border:2px solid {ec};
                         border-radius:16px; padding:20px; text-align:center; margin:8px 0;">
                        <div style="color:#90a4ae;">Detected Emotion</div>
                        <div style="color:{ec}; font-size:2rem; font-weight:800; text-transform:capitalize;">{emotion}</div>
                    </div>""", unsafe_allow_html=True)

                if fluency['filler_count'] > 0:
                    st.warning(f"⚠️ Filler words detected: {', '.join(fluency['found_fillers'])}")

                # AI Feedback
                st.markdown("---")
                get_ai_feedback(question, user_answer, category)

                # Save session
                sessions[name].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "question": question,
                    "category": category,
                    "answer": user_answer,
                    "emotion": emotion,
                    "scores": report['scores'],
                    "grade": report['grade'],
                    "mode": "practice"
                })
                save_sessions(sessions)
                st.success("✅ Session saved!")

            except Exception as e:
                st.error(f"Evaluation error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ── SKILL TEST ──
    with tab2:
        st.subheader("📝 Skill Test — 5 Random Questions")

        if 'test_questions' not in st.session_state or st.button("🔄 New Test"):
            test_cat = random.choice(ALL_CATEGORIES)
            pool = QUESTIONS[test_cat]
            st.session_state.test_questions = random.sample(pool, min(5, len(pool)))
            st.session_state.test_category = test_cat
            st.session_state.test_answers = {}
            st.session_state.test_done = False

        st.info(f"Category: **{st.session_state.test_category}**")

        for i, q in enumerate(st.session_state.test_questions):
            st.markdown(f"**Q{i+1}: {q}**")
            ans = st.text_area(f"Answer {i+1}", key=f"test_ans_{i}", height=100,
                              placeholder="Type your answer...")
            if ans:
                st.session_state.test_answers[q] = ans

        if st.button("✅ Submit Test", type="primary"):
            if len(st.session_state.test_answers) == 0:
                st.warning("Please answer at least one question.")
            else:
                try:
                    evaluate_answer, analyze_fluency, generate_report, calculate_esi, _ = load_models()
                    results = []
                    for q, a in st.session_state.test_answers.items():
                        nlp = evaluate_answer(q, a)
                        fluency = analyze_fluency(a)
                        report = generate_report(q, a, "neutral",
                                               nlp['relevance_score'],
                                               fluency['fluency_score'],
                                               nlp['sentiment_score'])
                        results.append(report['scores']['final'])

                    avg = round(sum(results) / len(results), 2)
                    grade = "A" if avg >= 85 else "B" if avg >= 70 else "C" if avg >= 55 else "D" if avg >= 40 else "F"
                    labels = {"A":"Excellent","B":"Good","C":"Average","D":"Needs Improvement","F":"Poor"}

                    st.markdown("---")
                    st.subheader("🏆 Test Results")
                    grade_badge(grade, labels[grade])
                    st.metric("Average Score", f"{avg}/100")

                    # Bar chart
                    fig, ax = plt.subplots(figsize=(8, 3))
                    fig.patch.set_facecolor('#0f1117')
                    ax.set_facecolor('#1e2433')
                    qs = [f"Q{i+1}" for i in range(len(results))]
                    bars = ax.bar(qs, results, color=['#4caf50' if r >= 70 else '#ff9800' if r >= 50 else '#f44336' for r in results])
                    ax.set_ylim(0, 100)
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values(): spine.set_color('#2d3748')
                    st.pyplot(fig)
                    plt.close()

                    sessions[name].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "category": st.session_state.test_category,
                        "scores": {"final": avg},
                        "grade": grade,
                        "mode": "skill_test",
                        "num_questions": len(results)
                    })
                    save_sessions(sessions)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ── PROGRESS ──
    with tab3:
        st.subheader(f"📊 Progress for {name}")
        user_sessions = sessions.get(name, [])
        if not user_sessions:
            st.info("No sessions yet. Start practicing!")
        else:
            scores = [s['scores']['final'] for s in user_sessions]
            dates = [s['date'] for s in user_sessions]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1e2433')
            ax.plot(range(len(scores)), scores, color='#667eea', linewidth=2, marker='o')
            ax.fill_between(range(len(scores)), scores, alpha=0.2, color='#667eea')
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score", color='white')
            ax.tick_params(colors='white')
            ax.set_title("Score Over Time", color='white')
            for spine in ax.spines.values(): spine.set_color('#2d3748')
            st.pyplot(fig)
            plt.close()

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Sessions", len(scores))
            with col2: st.metric("Average Score", f"{round(sum(scores)/len(scores), 1)}/100")
            with col3: st.metric("Best Score", f"{max(scores)}/100")

            _, _, _, calculate_esi, _ = load_models()
            emotions = [s.get('emotion', 'neutral') for s in user_sessions if 'emotion' in s]
            if emotions:
                esi = calculate_esi(emotions)
                st.markdown(f"**Emotion Stability Index (ESI): {esi}**")
                st.progress(esi)

# ══════════════════════════════════════════
# INTERVIEWER INTERFACE
# ══════════════════════════════════════════
else:
    st.subheader("👔 Interviewer Dashboard")
    sessions = load_sessions()

    if not sessions:
        st.info("No student data yet.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📋 All Students", "🔍 Student Detail", "📄 Export PDF"])

    with tab1:
        st.subheader("All Students Overview")
        data = []
        for sname, slist in sessions.items():
            if slist:
                scores = [s['scores']['final'] for s in slist]
                data.append({
                    "Student": sname,
                    "Sessions": len(slist),
                    "Avg Score": round(sum(scores)/len(scores), 1),
                    "Best Score": max(scores),
                    "Last Grade": slist[-1].get('grade', 'N/A')
                })
        if data:
            import pandas as pd
            df = pd.DataFrame(data).sort_values("Avg Score", ascending=False)
            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1e2433')
            ax.bar(df['Student'], df['Avg Score'], color='#667eea')
            ax.set_ylim(0, 100)
            ax.tick_params(colors='white')
            ax.set_title("Average Scores by Student", color='white')
            for spine in ax.spines.values(): spine.set_color('#2d3748')
            st.pyplot(fig)
            plt.close()

    with tab2:
        student = st.selectbox("Select Student", list(sessions.keys()))
        if student:
            sdata = sessions[student]
            st.subheader(f"Sessions for {student}")
            for i, s in enumerate(reversed(sdata)):
                with st.expander(f"Session {len(sdata)-i} — {s['date']} | Score: {s['scores']['final']} | Grade: {s['grade']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category:** {s.get('category','N/A')}")
                        st.write(f"**Mode:** {s.get('mode','practice')}")
                        if 'question' in s:
                            st.write(f"**Question:** {s['question']}")
                    with col2:
                        st.write(f"**Final Score:** {s['scores']['final']}/100")
                        st.write(f"**Grade:** {s['grade']}")
                        if 'emotion' in s:
                            st.write(f"**Emotion:** {s['emotion']}")

    with tab3:
        st.subheader("Export PDF Report")
        student_pdf = st.selectbox("Select Student for PDF", list(sessions.keys()), key="pdf_sel")
        if st.button("📄 Generate PDF"):
            try:
                from fpdf import FPDF
                sdata = sessions[student_pdf]
                scores = [s['scores']['final'] for s in sdata]
                avg = round(sum(scores)/len(scores), 1) if scores else 0

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.set_text_color(100, 126, 234)
                pdf.cell(0, 15, "AI Interview Evaluator Report", ln=True, align='C')
                pdf.set_font("Arial", "B", 14)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 10, f"Student: {student_pdf}", ln=True)
                pdf.cell(0, 10, f"Total Sessions: {len(sdata)}", ln=True)
                pdf.cell(0, 10, f"Average Score: {avg}/100", ln=True)
                pdf.cell(0, 10, f"Best Score: {max(scores) if scores else 0}/100", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.ln(5)
                for i, s in enumerate(sdata[-10:]):
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, f"Session {i+1} — {s['date']}", ln=True)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 6, f"  Score: {s['scores']['final']}/100  |  Grade: {s['grade']}  |  Category: {s.get('category','N/A')}", ln=True)
                    pdf.ln(2)

                pdf_path = f"models/{student_pdf}_report.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=f"{student_pdf}_report.pdf", mime="application/pdf")
                st.success("PDF generated!")
            except Exception as e:
                st.error(f"PDF error: {e}")
