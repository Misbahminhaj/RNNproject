import streamlit as st
import numpy as np
import pandas as pd
from predict import evaluate_student

st.set_page_config(page_title="RNN Student Evaluator", page_icon="🔄", layout="centered")

st.title("🔄 RNN Student Performance Evaluator")
st.markdown("Enter student details below. The RNN will predict **Pass** or **Fail**.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    attendance  = st.slider("📅 Attendance (%)",     0, 100, 75)
    assignment  = st.slider("📝 Assignment Score",   0, 100, 70)
    quiz        = st.slider("🧩 Quiz Score",         0, 100, 65)
with col2:
    mid         = st.slider("📖 Mid-term Score",     0, 100, 60)
    study_hours = st.slider("⏱️ Study Hours / Week", 0,  20,  6)

st.divider()

if st.button("🔮 Predict Result", use_container_width=True, type="primary"):
    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    if result["prediction"] == 1:
        st.success(f"### {result['label']}  —  {result['performance']}")
    else:
        st.error(f"### {result['label']}  —  {result['performance']}")

    c1, c2 = st.columns(2)
    c1.metric("✅ Pass Probability", f"{result['prob_pass']}%")
    c2.metric("❌ Fail Probability", f"{result['prob_fail']}%")

    st.markdown("**Pass Probability**")
    st.progress(int(result["prob_pass"]))
    st.markdown("**Fail Probability**")
    st.progress(int(result["prob_fail"]))

    st.divider()
    st.markdown("#### 📊 Student Profile")
    chart_data = pd.DataFrame({
        "Score": [attendance, assignment, quiz, mid, study_hours * 5]
    }, index=["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hrs x5"])
    st.bar_chart(chart_data)

    st.divider()
    st.markdown("#### 💡 Recommendation")
    if "High" in result["performance"]:
        st.info("Strong performance across all features. Keep it up!")
    elif "Medium" in result["performance"]:
        st.warning("Borderline result. Improve quiz scores and study hours.")
    else:
        st.error("At risk of failing. Improve attendance and study habits immediately.")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Model:** RNN (NumPy scratch)  
**Accuracy:** 85.83%  
**Input:** 5 features as time steps  
**Output:** Pass / Fail  

**Performance:**  
🌟 High → Pass ≥ 80%  
⚠️ Medium → Pass 50–80%  
🔴 Low → Pass < 50%  
    """)
