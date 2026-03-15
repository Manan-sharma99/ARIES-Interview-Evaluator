"""
AI-Powered Answer Evaluator using Claude API
Gives detailed, specific feedback on interview answers
"""

import anthropic

client = anthropic.Anthropic()

def evaluate_with_ai(question, user_answer, category="General"):
    """
    Uses Claude AI to evaluate the interview answer in detail.
    Returns specific feedback on what's good, what's missing, and how to improve.
    """
    try:
        prompt = f"""You are an expert interview coach evaluating a candidate's answer.

Question Category: {category}
Interview Question: {question}
Candidate's Answer: {user_answer}

Evaluate this answer and respond in exactly this JSON format (no extra text, just JSON):
{{
    "overall_score": <number 0-100>,
    "what_was_good": ["point 1", "point 2", "point 3"],
    "what_was_missing": ["missing point 1", "missing point 2"],
    "key_concepts_mentioned": ["concept 1", "concept 2"],
    "key_concepts_missing": ["concept 1", "concept 2"],
    "improved_answer_hint": "A 2-3 sentence hint on how to improve the answer",
    "star_method_used": true or false,
    "confidence_level": "High/Medium/Low",
    "verdict": "Strong/Acceptable/Needs Work/Poor"
}}

Be specific and actionable. For technical questions check if key concepts are mentioned.
For behavioral questions check if STAR method (Situation Task Action Result) is used."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        raw = response.content[0].text.strip()
        # Clean JSON if wrapped in backticks
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result

    except Exception as e:
        return {
            "overall_score": 50,
            "what_was_good": ["Answer was provided"],
            "what_was_missing": ["Could not analyze in detail"],
            "key_concepts_mentioned": [],
            "key_concepts_missing": [],
            "improved_answer_hint": "Practice structuring your answer clearly.",
            "star_method_used": False,
            "confidence_level": "Medium",
            "verdict": "Acceptable",
            "error": str(e)
        }


def display_ai_feedback(result, st):
    """Display AI feedback in Streamlit"""
    if not result:
        return

    verdict = result.get("verdict", "Acceptable")
    score = result.get("overall_score", 50)

    verdict_colors = {
        "Strong": "#4caf50",
        "Acceptable": "#2196f3",
        "Needs Work": "#ff9800",
        "Poor": "#f44336"
    }
    color = verdict_colors.get(verdict, "#2196f3")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e2433, #252d3d);
         border: 1px solid {color}; border-radius: 16px; padding: 24px; margin: 16px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:1.1rem; color:#90a4ae;">🤖 AI Evaluation</div>
                <div style="font-size:2rem; font-weight:800; color:{color};">{verdict}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:2.5rem; font-weight:900; color:{color};">{score}</div>
                <div style="color:#90a4ae; font-size:0.85rem;">AI Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ What Was Good:**")
        for point in result.get("what_was_good", []):
            st.success(f"• {point}")

        if result.get("key_concepts_mentioned"):
            st.markdown("**🎯 Key Concepts Mentioned:**")
            for c in result.get("key_concepts_mentioned", []):
                st.markdown(f"✅ `{c}`")

    with col2:
        st.markdown("**❌ What Was Missing:**")
        for point in result.get("what_was_missing", []):
            st.error(f"• {point}")

        if result.get("key_concepts_missing"):
            st.markdown("**⚠️ Key Concepts Missing:**")
            for c in result.get("key_concepts_missing", []):
                st.markdown(f"❌ `{c}`")

    st.markdown("**💡 How to Improve:**")
    st.info(result.get("improved_answer_hint", ""))

    col3, col4 = st.columns(2)
    with col3:
        star = result.get("star_method_used", False)
        st.markdown(f"**STAR Method Used:** {'✅ Yes' if star else '❌ No'}")
    with col4:
        st.markdown(f"**Confidence Level:** {result.get('confidence_level', 'Medium')}")
