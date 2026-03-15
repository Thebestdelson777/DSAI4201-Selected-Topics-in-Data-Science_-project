import streamlit as st
import os
import pandas as pd
import io
from extractor import extract_text
from scorer import score_candidate, generate_interview_questions
import ollama

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background-color: #0f0f1a;
        color: #e8e8f0;
    }
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #7c6bff, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #6b6b80;
        font-size: 14px;
        margin-bottom: 30px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .score-green { color: #6bffd8; font-size: 28px; font-weight: 800; }
    .score-yellow { color: #ffd96b; font-size: 28px; font-weight: 800; }
    .score-red { color: #ff6b6b; font-size: 28px; font-weight: 800; }
    .skill-match {
        display: inline-block;
        background: rgba(107,255,216,0.1);
        color: #6bffd8;
        border: 1px solid rgba(107,255,216,0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 2px;
    }
    .skill-miss {
        display: inline-block;
        background: rgba(255,107,107,0.1);
        color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 2px;
    }
    .section-header {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #6b6b80;
        margin-bottom: 8px;
    }
    hr { border-color: #2a2a3a; }
    .streamlit-expanderHeader {
        background: #1a1a2e !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: render structured explanation ──
def render_explanation(explanation, compact=False):
    lines = explanation.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("SUMMARY:"):
            text = line.replace("SUMMARY:", "").strip()
            st.markdown(f"📋 **Summary:** {text}")
        elif line.startswith("STRENGTHS:"):
            text = line.replace("STRENGTHS:", "").strip()
            st.success(f"💪 **Strengths:** {text}")
        elif line.startswith("WEAKNESSES:"):
            text = line.replace("WEAKNESSES:", "").strip()
            st.error(f"⚠️ **Weaknesses:** {text}")
        elif line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
            if "Not Recommended" in verdict:
                label = f"❌ Verdict: {verdict}" if compact else f"### ❌ Verdict: {verdict}"
            elif "Recommended" in verdict:
                label = f"✅ Verdict: {verdict}" if compact else f"### ✅ Verdict: {verdict}"
            else:
                label = f"🔶 Verdict: {verdict}" if compact else f"### 🔶 Verdict: {verdict}"
            st.markdown(f"**{label}**" if compact else label)
        else:
            st.caption(line)


# ── Helper: render interview questions ──
def render_interview_questions(questions_raw, compact=False):
    if not isinstance(questions_raw, str):
        return
    lines = questions_raw.split("\n")
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("TECHNICAL:"):
            st.markdown("#### 🔧 Technical Questions")
            current_section = "technical"
        elif line.startswith("BEHAVIORAL:"):
            st.markdown("#### 🤝 Behavioral Questions")
            current_section = "behavioral"
        elif line.startswith("GAP:"):
            st.markdown("#### 🔍 Gap Verification Questions")
            current_section = "gap"
        elif line and line[0].isdigit():
            if current_section == "technical":
                st.info(f"💡 {line}")
            elif current_section == "behavioral":
                st.success(f"🤝 {line}")
            elif current_section == "gap":
                st.warning(f"⚠️ {line}")
            else:
                st.write(line)
        else:
            if not compact:
                st.caption(line)


def employer_chatbot(question, results, jd_text):
    if not question.strip():
        return "Please enter a question."

    candidate_summary = []
    for c in results:
        candidate_summary.append(
            {
                "name": c.get("name", c.get("filename", "Unknown")),
                "score": c.get("score", 0),
                "similarity": c.get("similarity", 0),
                "matched_skills": c.get("matched_skills", []),
                "missing_skills": c.get("missing_skills", []),
                "experience_score": c.get("scores", {}).get("experience", 0),
                "skills_score": c.get("scores", {}).get("skills", 0),
                "education_score": c.get("scores", {}).get("education", 0),
                "projects_score": c.get("scores", {}).get("projects", 0),
                "explanation": c.get("explanation", "")
            }
        )

    prompt = f"""
You are an HR assistant helping an employer understand candidate ranking results.

Job Description:
{jd_text}

Candidate Results:
{candidate_summary}

Employer Question:
{question}

Rules:
- Answer clearly and professionally
- Base the answer only on the provided candidate results
- If comparing candidates, explain using score, matched skills, missing skills, and evaluation
- If the answer is not available, say so honestly
- Keep the answer concise but helpful
"""

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error generating chatbot response: {str(e)}"
    
# ── Header ──
st.markdown('<div class="main-title">🧠 SmartHire AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent CV Screening · Powered by Local AI · No Data Leaves Your Machine</div>', unsafe_allow_html=True)
st.divider()


# ── Input Section ──
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📋 Step 1 — Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=220,
        placeholder="We are looking for a Data Analyst with 2 years experience...\nRequired skills: Python, SQL, Power BI...",
        label_visibility="collapsed"
    )

with col_right:
    st.markdown("### 📁 Step 2 — Upload CVs")
    uploaded_files = st.file_uploader(
        "Upload CVs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} CV(s) ready to screen")

st.divider()

# ── Run Button ──
st.markdown("### ⚡ Step 3 — Run Screening")
run = st.button("⚡ Run AI Screening", use_container_width=True, type="primary")

if run:

    if not jd_text.strip():
        st.error("⚠️ Please paste a job description first!")
        st.stop()

    if not uploaded_files:
        st.error("⚠️ Please upload at least one CV!")
        st.stop()

    # clear previous results and questions when re-running
    for key in list(st.session_state.keys()):
        if key.startswith("questions_") or key.startswith("interview_") or key == "results":
            del st.session_state[key]

    results = []
    errors = []

    progress = st.progress(0, text="Starting analysis...")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(
            int((i / total) * 100),
            text=f"Analyzing {uploaded_file.name}... ({i+1}/{total})"
        )

        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            cv_text = extract_text(temp_path)

            if not cv_text.strip():
                errors.append(f"Could not read text from {uploaded_file.name}")
                continue

            result = score_candidate(uploaded_file.name, cv_text, jd_text)
            result["filename"] = uploaded_file.name
            results.append(result)

        except Exception as e:
            errors.append(f"Error processing {uploaded_file.name}: {str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    progress.progress(100, text="✅ Analysis complete!")

    for err in errors:
        st.warning(f"⚠️ {err}")

    if not results:
        st.error("No CVs could be processed. Please check your files.")
        st.stop()

    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # ── SAVE TO SESSION STATE so buttons don't wipe the page ──
    st.session_state["results"] = results


# ── Display Results — OUTSIDE if run block so it survives reruns ──
if "results" in st.session_state:
    results = st.session_state["results"]

    top_score = results[0].get("score", 0)
    if top_score < 30:
        st.warning(
            "⚠️ No strong matches found. All candidates scored below 30/100. "
            "Consider reviewing the job requirements or uploading different CVs."
        )

    st.divider()

    # ── Results Header + CSV Export ──
    res_col1, res_col2 = st.columns([3, 1])
    with res_col1:
        st.markdown(f"### 🏆 Ranking Results — {len(results)} Candidates Screened")
    with res_col2:
        export_data = []
        for i, c in enumerate(results):
            export_data.append({
                "Rank": i + 1,
                "Name": c.get("name", ""),
                "Score": c.get("score", 0),
                "Similarity %": c.get("similarity", 0),
                "Matched Skills": ", ".join(c.get("matched_skills", [])),
                "Missing Skills": ", ".join(c.get("missing_skills", [])),
                "Summary": c.get("summary", ""),
                "Explanation": c.get("explanation", "")
            })
        df_export = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv_buffer.getvalue(),
            file_name="smarthire_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Candidate Cards ──
    for i, candidate in enumerate(results):
        rank = i + 1
        score = candidate.get("score", 0)
        name = candidate.get("name", candidate["filename"])
        similarity = candidate.get("similarity", 0)
        matched = candidate.get("matched_skills", [])
        missing = candidate.get("missing_skills", [])
        scores = candidate.get("scores", {})

        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

        if score >= 70:
            score_class = "score-green"
            score_emoji = "🟢"
        elif score >= 50:
            score_class = "score-yellow"
            score_emoji = "🟡"
        else:
            score_class = "score-red"
            score_emoji = "🔴"

        with st.expander(f"{medal}  {name}  ·  {score_emoji} {score}/100  ·  {similarity}% similarity"):

            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                st.markdown('<div class="section-header">Overall Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{score_class}">{score}/100</div>', unsafe_allow_html=True)
                st.caption(f"TF-IDF similarity: {similarity}%")

            with c2:
                st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
                st.progress(scores.get("skills", 0) / 100,
                    text=f"Skills: {scores.get('skills', 0)}/100")
                st.progress(scores.get("experience", 0) / 100,
                    text=f"Experience: {scores.get('experience', 0)}/100")
                st.progress(scores.get("education", 0) / 100,
                    text=f"Education: {scores.get('education', 0)}/100")
                st.progress(scores.get("projects", 0) / 100,
                    text=f"Projects: {scores.get('projects', 0)}/100")

            with c3:
                st.markdown('<div class="section-header">Matched Skills</div>', unsafe_allow_html=True)
                if matched:
                    tags = " ".join([f'<span class="skill-match">✓ {s}</span>' for s in matched])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("No skills matched")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Missing Skills</div>', unsafe_allow_html=True)
                if missing:
                    tags = " ".join([f'<span class="skill-miss">✗ {s}</span>' for s in missing])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("None — perfect skill match! 🎉")

            st.divider()

            if candidate.get("is_suspicious"):
                st.warning(f"⚠️ Fraud Alert: Skills not backed by work experience. Trust score: {candidate.get('trust_score')}/1.0")

            st.markdown('<div class="section-header">🤖 AI Evaluation</div>', unsafe_allow_html=True)
            render_explanation(candidate.get("explanation", ""))

            # ── Interview Question Generator ──
            st.divider()
            st.markdown('<div class="section-header">🎯 Interview Preparation</div>', unsafe_allow_html=True)

            btn_key = f"interview_{i}_{name}"
            if st.button(f"🎤 Generate Interview Questions for {name}", key=btn_key):
                with st.spinner(f"Generating tailored interview questions for {name}..."):
                    questions_raw = generate_interview_questions(
                        candidate.get("cv_profile", {}),
                        candidate.get("jd_profile", {}),
                        candidate.get("result_raw", {})
                    )
                    st.session_state[f"questions_{i}"] = questions_raw

            if f"questions_{i}" in st.session_state and isinstance(st.session_state[f"questions_{i}"], str):
                render_interview_questions(st.session_state[f"questions_{i}"])


    # ── Candidate Comparison ──
    if len(results) >= 2:
        st.divider()
        st.markdown("### ⚖️ Compare Two Candidates")

        candidate_names = [c.get("name", c["filename"]) for c in results]

        col_a, col_b = st.columns(2)
        with col_a:
            candidate_a = st.selectbox("Select Candidate A", candidate_names, index=0, key="compare_a")
        with col_b:
            candidate_b = st.selectbox("Select Candidate B", candidate_names, index=1, key="compare_b")

        if candidate_a != candidate_b:
            data_a = next(c for c in results if c.get("name", c["filename"]) == candidate_a)
            data_b = next(c for c in results if c.get("name", c["filename"]) == candidate_b)

            st.markdown("---")

            def show_compare_card(col, data, card_id):
                with col:
                    score = data.get("score", 0)
                    score_class = "score-green" if score >= 70 else "score-yellow" if score >= 50 else "score-red"

                    st.markdown(f"#### {data.get('name', data['filename'])}")
                    st.markdown(f'<div class="{score_class}">{score}/100</div>', unsafe_allow_html=True)
                    st.caption(f"TF-IDF Similarity: {data.get('similarity', 0)}%")

                    st.markdown("---")
                    st.markdown("**Score Breakdown**")
                    scores = data.get("scores", {})
                    st.progress(scores.get("skills", 0) / 100, text=f"Skills: {scores.get('skills', 0)}/100")
                    st.progress(scores.get("experience", 0) / 100, text=f"Experience: {scores.get('experience', 0)}/100")
                    st.progress(scores.get("education", 0) / 100, text=f"Education: {scores.get('education', 0)}/100")
                    st.progress(scores.get("projects", 0) / 100, text=f"Projects: {scores.get('projects', 0)}/100")

                    st.markdown("---")
                    st.markdown("**✅ Matched Skills**")
                    matched = data.get("matched_skills", [])
                    if matched:
                        tags = " ".join([f'<span class="skill-match">✓ {s}</span>' for s in matched])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None matched")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**❌ Missing Skills**")
                    missing = data.get("missing_skills", [])
                    if missing:
                        tags = " ".join([f'<span class="skill-miss">✗ {s}</span>' for s in missing])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None missing 🎉")

                    if data.get("is_suspicious"):
                        st.warning(f"⚠️ Fraud Alert: Trust score: {data.get('trust_score')}/1.0")

                    st.markdown("---")
                    st.markdown("**🤖 AI Evaluation**")
                    render_explanation(data.get("explanation", ""), compact=True)

                    st.markdown("---")
                    compare_btn_key = f"interview_compare_btn_{card_id}"
                    questions_key = f"interview_compare_questions_{card_id}"

                    if st.button("🎤 Generate Interview Questions", key=compare_btn_key):
                        with st.spinner("Generating tailored questions..."):
                            questions_raw = generate_interview_questions(
                                data.get("cv_profile", {}),
                                data.get("jd_profile", {}),
                                data.get("result_raw", {})
                            )
                            st.session_state[questions_key] = questions_raw

                    if questions_key in st.session_state and isinstance(st.session_state[questions_key], str):
                        render_interview_questions(st.session_state[questions_key], compact=True)

            col1, col2 = st.columns(2)
            show_compare_card(col1, data_a, "a")
            show_compare_card(col2, data_b, "b")

            st.markdown("---")
            score_a = data_a.get("score", 0)
            score_b = data_b.get("score", 0)

            if score_a > score_b:
                diff = score_a - score_b
                st.success(f"🏆 **{candidate_a}** is the stronger candidate by **{diff} points**")
            elif score_b > score_a:
                diff = score_b - score_a
                st.success(f"🏆 **{candidate_b}** is the stronger candidate by **{diff} points**")
            else:
                st.info("🤝 Both candidates scored equally!")

        else:
            st.warning("Please select two different candidates to compare.")
    # ── Employer Chatbot ──
    st.divider()
    st.markdown("### 💬 Employer Chat Assistant")
    st.caption("Ask questions about the ranked candidates, strengths, weaknesses, and shortlist decisions.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.text_input(
        "Ask something about the candidates",
        placeholder="Who is the best candidate and why?"
    )

    if st.button("Ask Chatbot", key="ask_chatbot_btn"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                answer = employer_chatbot(user_question, results, jd_text)
                st.session_state["chat_history"].append(("You", user_question))
                st.session_state["chat_history"].append(("Assistant", answer))

    for role, message in st.session_state["chat_history"]:
        if role == "You":
            st.markdown(f"**🧑 Employer:** {message}")
        else:
            st.markdown(f"**🤖 Assistant:** {message}")