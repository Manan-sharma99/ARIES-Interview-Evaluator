"""
ui_live_session.py
Live session UI components for ARIES Interview System.

Provides:
  - init_live_session_state()   — register new state keys (call from ensure_state)
  - render_live_session_controls() — Start / End session buttons (Task 1)
  - render_custom_question_input() — interviewer custom question input (Task 2)
"""

import streamlit as st


def init_live_session_state():
    """
    Ensure all live-session state keys exist.
    Call once inside interview_app.ensure_state().
    """
    defaults = {
        "is_live_session":      False,   # True while a live session is running
        "live_session_type":    "live",  # "live" | "practice" | "skill_test"
        "live_session_answers": [],      # list of result dicts for this session
        "current_question":     "",      # active question set by interviewer
        "live_question_category": "HR & General",
        "live_session_ended":   False,   # True right after End Session pressed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_live_session_controls(candidate: str, on_start=None, on_end=None):
    """
    Render Start / End session buttons.

    Args:
        candidate : str       — candidate name (must be non-empty)
        on_start  : callable  — called with no args when Start is clicked
        on_end    : callable(answers: list, status: str) — called on End
                    status is "completed" (≥1 answer) or "abandoned" (0 answers)
    """
    if not candidate:
        st.warning("Enter a candidate name in the sidebar before starting a session.")
        return

    col_start, col_end, col_type = st.columns([1, 1, 1.4])

    with col_type:
        st.selectbox(
            "Session type",
            ["live", "practice", "skill_test"],
            key="live_session_type",
            label_visibility="collapsed",
        )

    if not st.session_state.get("is_live_session", False):
        with col_start:
            if st.button("▶ Start Live Session", use_container_width=True,
                         type="primary", key="btn_start_live"):
                st.session_state.is_live_session      = True
                st.session_state.live_session_answers  = []
                st.session_state.live_session_ended   = False
                st.session_state.current_question     = ""
                if on_start:
                    on_start()
                st.rerun()
    else:
        with col_start:
            # Status indicator — not a button
            st.markdown(
                "<div style='padding:.52rem .8rem; border-radius:12px; "
                "background:rgba(255,74,74,0.1); border:1px solid rgba(255,74,74,0.35); "
                "font-size:.82rem; font-weight:700; color:#ff4a4a; text-align:center;'>"
                "🔴 Live session active</div>",
                unsafe_allow_html=True,
            )
        with col_end:
            # End Session works even with 0 answers (Task 1 fix)
            if st.button("⏹ End Session", use_container_width=True, key="btn_end_live"):
                answers = st.session_state.get("live_session_answers", [])
                status  = "completed" if answers else "abandoned"
                st.session_state.is_live_session    = False
                st.session_state.live_session_ended = True
                if on_end:
                    on_end(answers, status)
                st.rerun()


def render_custom_question_input(all_categories: list = None) -> tuple[str, str]:
    """
    Render the interviewer's custom question input box.

    Does NOT depend on question_bank — any string question is valid (Task 2).

    Args:
        all_categories : list of str — for the category picker beside the input

    Returns:
        (question: str, category: str) — current widget values
    """
    categories = all_categories or ["General", "HR & General", "Behavioral",
                                     "Technical", "CS Fundamentals", "Machine Learning"]
    st.markdown(
        "<div style='font-size:.72rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:.18em; color:#9aa9c7; margin-bottom:.4rem;'>"
        "Interviewer — Ask a Question</div>",
        unsafe_allow_html=True,
    )
    col_q, col_cat = st.columns([3, 1])
    with col_q:
        question = st.text_input(
            "Custom question",
            key="current_question",
            placeholder="Type the question you want to ask the candidate…",
            label_visibility="collapsed",
        )
    with col_cat:
        category = st.selectbox(
            "Category",
            categories,
            key="live_question_category",
            label_visibility="collapsed",
        )
    return question, category
