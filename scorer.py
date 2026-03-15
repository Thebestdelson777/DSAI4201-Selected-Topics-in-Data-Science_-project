import re
import json
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Step 1: TF-IDF cosine similarity ──
def tfidf_similarity(jd_text, cv_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, cv_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)


# ── Step 2: AI extracts requirements from JD ──
def extract_requirements(jd_text):
    prompt = f"""Read this job description carefully.
Extract SPECIFIC skills, not broad terms.
Example: write "python" not "programming", write "sql" not "databases".

Return ONLY this JSON, nothing else, no extra text:
{{
    "required_skills": ["python", "sql", "excel"],
    "job_role": "exact job title here",
    "years_required": 2,
    "education_required": "bachelor"
}}

Job Description:
{jd_text}"""

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    return {
        "required_skills": [],
        "job_role": "",
        "years_required": 0,
        "education_required": "none"
    }


# ── Step 3: AI extracts profile from CV (explicit + inferred from projects) ──
def extract_cv_profile(cv_text):
    prompt = f"""Read this CV carefully. Extract ONLY skills that are EXPLICITLY written in the CV.
DO NOT guess or infer skills. If the word is not literally there, do not include it.
DO NOT write "python" just because someone works with data.
DO NOT write "sql" just because someone cataloged information.
DO NOT write "excel" just because someone managed inventory or reports.
ONLY include a skill if it is literally written as a technology, tool, or software name in the CV.

Return ONLY this JSON, nothing else, no extra text:
{{
    "candidate_name": "full name or Unknown",
    "skills": ["only skills literally mentioned in the CV"],
    "job_roles_held": ["exact job titles from the CV"],
    "years_of_experience": 3,
    "education_level": "bachelor",
    "has_projects": true,
    "project_descriptions": ["exact project descriptions from CV"]
}}

CV:
{cv_text[:3000]}"""

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            profile = json.loads(match.group())

            # ── Second AI call: infer skills from projects ──
            projects = profile.get("project_descriptions", [])
            if projects:
                infer_prompt = f"""A candidate has done these projects:
{projects}

What specific technical tools and skills were MOST LIKELY used to build these?
Only infer skills that are STRONGLY implied by the project type.
DO NOT infer programming skills from non-technical projects.
DO NOT infer python or sql from biology, wildlife, or fieldwork projects.
DO NOT infer python or sql from marketing or social media projects.
DO NOT infer python or sql from plumbing, construction, or medical projects.

Examples of correct inference:
- "built sales dashboard" → excel, power bi
- "developed REST API" → python, sql
- "created web scraping script" → python
- "inventory management system" → sql, excel
- "e-commerce platform with react and flask" → python, react, sql
- "machine learning model for fraud detection" → python, machine learning, sql
- "wildlife habitat report" → report writing (NO python, NO sql)
- "social media campaign analysis" → excel, google ads (NO python)
- "plumbing installation project" → NO technical software skills
- "patient diagnosis report" → NO python, NO sql

Return ONLY a JSON list of inferred skills, nothing else:
["skill1", "skill2"]"""

                infer_response = ollama.chat(
                    model="gemma3:1b",
                    messages=[{"role": "user", "content": infer_prompt}]
                )
                infer_raw = infer_response["message"]["content"]
                infer_match = re.search(r'\[.*\]', infer_raw, re.DOTALL)
                if infer_match:
                    inferred = json.loads(infer_match.group())
                    existing = [s.lower() for s in profile.get("skills", [])]
                    for skill in inferred:
                        if skill.lower() not in existing:
                            profile["skills"].append(skill.lower())

            return profile

    except:
        pass

    return {
        "candidate_name": "Unknown",
        "skills": [],
        "job_roles_held": [],
        "years_of_experience": 0,
        "education_level": "none",
        "has_projects": False,
        "project_descriptions": []
    }


# ── Step 4: Fraud detection — cross validate skills ──
def cross_validate_skills(matched_skills, cv_text):
    cv_lower = cv_text.lower()

    skills_section = ""
    evidence_section = ""

    lines = cv_lower.split('\n')
    current_section = "other"

    for line in lines:
        if any(word in line for word in ["skill", "technical", "competenc"]):
            current_section = "skills"
        elif any(word in line for word in ["experience", "work", "employment",
                                            "project", "responsibilities"]):
            current_section = "evidence"

        if current_section == "skills":
            skills_section += line + " "
        elif current_section == "evidence":
            evidence_section += line + " "

    if not evidence_section.strip():
        evidence_section = cv_lower

    total_trust = 0
    skill_trust_details = {}

    for skill in matched_skills:
        in_evidence = skill in evidence_section
        in_skills_section = skill in skills_section

        if in_evidence:
            trust = 1.0
        elif in_skills_section:
            trust = 0.5
        else:
            trust = 0.3

        skill_trust_details[skill] = trust
        total_trust += trust

    if not matched_skills:
        return 0, skill_trust_details

    avg_trust = total_trust / len(matched_skills)
    return round(avg_trust, 2), skill_trust_details


# ── Step 5: Flexible skill matching helper ──
def skills_match(required_skill, candidate_skills, cv_text_lower=""):
    # direct match from AI extracted skills — most reliable
    if any(required_skill in c or c in required_skill for c in candidate_skills):
        return True

    # word level match on AI extracted skills only
    req_words = [w for w in required_skill.split() if len(w) > 4]
    for c in candidate_skills:
        c_words = [w for w in c.split() if len(w) > 4]
        if any(rw in c for rw in req_words):
            return True
        if any(cw in required_skill for cw in c_words):
            return True

    # raw CV text fallback — ONLY for multi-word technical skills
    # single words like "python", "sql", "excel" must come from AI extraction
    if cv_text_lower and len(required_skill.split()) >= 2:
        req_words = [w for w in required_skill.split() if len(w) > 4]
        if req_words and all(w in cv_text_lower for w in req_words):
            return True

    return False


# ── Step 6: Python scoring logic ──
def compare_and_score(jd_profile, cv_profile, similarity_score, cv_text="", jd_text=""):

    cv_text_lower = cv_text.lower()

    # --- Skills Score (40 pts) ---
    skills_score = round((similarity_score / 100) * 40)

    required = [s.lower().strip() for s in jd_profile.get("required_skills", [])]
    candidate = [s.lower().strip() for s in cv_profile.get("skills", [])]

    matched_skills = [s for s in required if skills_match(s, candidate, cv_text_lower)]
    missing_skills = [s for s in required if not skills_match(s, candidate, cv_text_lower)]

    trust_score, skill_trust_details = cross_validate_skills(matched_skills, cv_text)

    if required:
        direct_match_ratio = len(matched_skills) / len(required)
        direct_bonus = round(direct_match_ratio * 20 * trust_score)
        skills_score = min(skills_score + direct_bonus, 40)

    is_suspicious = trust_score < 0.6 and len(matched_skills) >= 3

    # --- Experience Score (30 pts) ---
    required_role = jd_profile.get("job_role", "").lower().strip()
    held_roles = [r.lower().strip() for r in cv_profile.get("job_roles_held", [])]
    years_required = int(jd_profile.get("years_required", 0))
    cv_years = int(cv_profile.get("years_of_experience", 0))

    role_relevant = False

    # primary check — direct role title word match
    if required_role and held_roles:
        required_words = [w for w in required_role.split() if len(w) > 3]
        for held in held_roles:
            if any(word in held for word in required_words):
                role_relevant = True
                break
            held_words = [w for w in held.split() if len(w) > 3]
            if any(word in required_role for word in held_words):
                role_relevant = True
                break

    # stricter fallback — only count as relevant if candidate also has
    # at least 2 matched skills, preventing "software engineer" from
    # matching "data scientist" just because both are technical roles
    if not role_relevant and len(matched_skills) >= 2:
        if required_role and held_roles:
            jd_lower = jd_text.lower()
            for held in held_roles:
                for word in held.split():
                    if len(word) > 4 and word in jd_lower:
                        role_relevant = True
                        break
                if role_relevant:
                    break

    if role_relevant:
        exp_score = 20
    else:
        exp_score = 4

    if years_required == 0:
        exp_score += 10
    elif cv_years >= years_required:
        exp_score += 10
    elif cv_years >= years_required - 1:
        exp_score += 6
    elif cv_years > 0:
        exp_score += 3

    exp_score = min(exp_score, 30)

    # --- Education Score (15 pts) ---
    edu_required = jd_profile.get("education_required", "none").lower()
    edu_candidate = cv_profile.get("education_level", "none").lower()

    edu_levels = {
        "phd": 4, "doctorate": 4,
        "master": 3, "msc": 3, "mba": 3,
        "bachelor": 2, "bsc": 2, "degree": 2,
        "diploma": 1, "certificate": 1,
        "none": 0
    }

    required_level = edu_levels.get(edu_required, 0)
    candidate_level = edu_levels.get(edu_candidate, 0)

    if candidate_level == 0 and cv_text_lower:
        for edu_keyword, level in edu_levels.items():
            if edu_keyword in cv_text_lower and level > candidate_level:
                candidate_level = level

    if required_level == 0:
        edu_score = 10
    elif candidate_level >= required_level:
        edu_score = 15
    elif candidate_level == required_level - 1:
        edu_score = 8
    else:
        edu_score = 3

    # --- Projects Score (15 pts) ---
    has_projects = cv_profile.get("has_projects", False)
    project_text = " ".join(cv_profile.get("project_descriptions", [])).lower()

    if not has_projects and any(w in cv_text_lower for w in ["project", "built", "developed", "designed", "constructed"]):
        has_projects = True

    proj_score = 0
    if has_projects:
        proj_score += 7
        relevant_count = sum(1 for s in matched_skills if s in project_text or s in cv_text_lower)
        if relevant_count >= 2:
            proj_score += 8
        elif relevant_count >= 1:
            proj_score += 4

    proj_score = min(proj_score, 15)

    total = skills_score + exp_score + edu_score + proj_score

    return {
        "total": total,
        "skills_score": skills_score,
        "exp_score": exp_score,
        "edu_score": edu_score,
        "proj_score": proj_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "role_relevant": role_relevant,
        "cv_years": cv_years,
        "similarity_score": similarity_score,
        "trust_score": trust_score,
        "is_suspicious": is_suspicious,
        "skill_trust_details": skill_trust_details
    }


# ── Step 7: Structured AI explanation ──
# ── Step 7: Structured AI explanation ──
def get_explanation(cv_profile, jd_profile, result):

    matched_count = len(result.get("matched_skills", []))
    similarity = result.get("similarity_score", 0)
    role_relevant = result.get("role_relevant", False)
    total_score = result.get("total", 0)

    # Hard rejection rule for clearly irrelevant candidates
    if matched_count == 0 and similarity < 15 and not role_relevant:
        return (
            "SUMMARY: The candidate's background does not align with the requirements of this role.\n"
            "STRENGTHS: The candidate may have experience in another field, but it is not relevant to this position.\n"
            "WEAKNESSES: The CV lacks the required technical skills and domain experience needed for this job.\n"
            "VERDICT: Not Recommended"
        )

    prompt = f"""You are a strict recruiter writing a structured candidate evaluation.

IMPORTANT RULES:
- Evaluate the candidate ONLY for the target job role.
- Do NOT praise unrelated experience.
- Do NOT describe irrelevant background as a strength.
- If the candidate lacks the required skills, say so clearly.
- If the candidate is from a different field, state that their experience is not relevant.
- Be strict and realistic.

Job role needed: {jd_profile.get('job_role')}
Required skills: {jd_profile.get('required_skills')}

Candidate: {cv_profile.get('candidate_name')}
Their skills: {cv_profile.get('skills')}
Roles they held: {cv_profile.get('job_roles_held')}
Years experience: {cv_profile.get('years_of_experience')}
Matched skills: {result.get('matched_skills')}
Missing skills: {result.get('missing_skills')}
Relevant experience: {result.get('role_relevant')}
Similarity: {result.get('similarity_score')}%
Trust score: {result.get('trust_score')} out of 1.0
Total score: {result.get('total')}/100

Write EXACTLY in this format, no extra text:

SUMMARY: One sentence overall verdict about this candidate for THIS role only.
STRENGTHS: Mention only role-relevant strengths. If none, say there are no major role-relevant strengths.
WEAKNESSES: Clearly mention missing technical skills or irrelevant experience.
VERDICT: Recommended / Potential / Not Recommended
"""

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response["message"]["content"].strip()

        has_summary = "SUMMARY:" in raw
        has_strengths = "STRENGTHS:" in raw
        has_weaknesses = "WEAKNESSES:" in raw
        has_verdict = "VERDICT:" in raw

        if not has_summary:
            raw += f"\nSUMMARY: The candidate shows limited alignment with the role, matching {matched_count} required skills with {similarity}% similarity."

        if not has_strengths:
            if matched_count > 0:
                raw += f"\nSTRENGTHS: Shows some alignment in {', '.join(result.get('matched_skills', [])[:3])}."
            else:
                raw += "\nSTRENGTHS: There are no major role-relevant strengths for this job."

        if not has_weaknesses:
            raw += f"\nWEAKNESSES: Missing {', '.join(result.get('missing_skills', [])[:3]) if result.get('missing_skills') else 'important required skills'}."

        if not has_verdict:
            verdict = "Recommended" if total_score >= 70 else "Potential" if total_score >= 50 else "Not Recommended"
            raw += f"\nVERDICT: {verdict}"

        return raw

    except Exception:

        verdict = "Recommended" if total_score >= 70 else "Potential" if total_score >= 50 else "Not Recommended"

        return (
            f"SUMMARY: The candidate shows limited alignment with the requirements of this role, matching {matched_count} required skills with {similarity}% similarity.\n"
            f"STRENGTHS: {'Shows some alignment in relevant areas.' if matched_count > 0 else 'There are no major role-relevant strengths for this job.'}\n"
            f"WEAKNESSES: Missing {', '.join(result.get('missing_skills', [])[:3]) if result.get('missing_skills') else 'important required skills'}.\n"
            f"VERDICT: {verdict}"
        )

# ── Step 8: AI Interview Question Generator ──
def generate_interview_questions(cv_profile, jd_profile, result):

    projects = cv_profile.get('project_descriptions', [])
    project_text = projects[0] if projects else "no specific project mentioned"

    matched = result.get('matched_skills', [])
    missing = result.get('missing_skills', [])

    prompt = f"""You are an expert HR interviewer. Generate specific interview questions for this exact candidate.

Job Role: {jd_profile.get('job_role')}
Required Skills: {jd_profile.get('required_skills')}

Candidate: {cv_profile.get('candidate_name')}
Their Skills: {cv_profile.get('skills')}
Roles They Held: {cv_profile.get('job_roles_held')}
Years Experience: {cv_profile.get('years_of_experience')}
Their Projects: {cv_profile.get('project_descriptions')}
Matched Skills: {matched}
Missing Skills: {missing}

RULES:
- Every question must be SPECIFIC to this candidate
- Technical questions must reference their actual skills like {matched[:2] if matched else 'their skills'}
- Project question must reference their actual project: "{project_text}"
- Gap questions must ask about their missing skills: {missing[:2] if missing else 'skill gaps'}
- Do NOT write generic questions
- Do NOT ask about skills they do not have in the technical section

Write EXACTLY in this format, no extra text, no intro:

TECHNICAL:
1. [specific technical question about {matched[0] if len(matched) > 0 else 'their top skill'}]
2. [specific technical question about {matched[1] if len(matched) > 1 else matched[0] if len(matched) > 0 else 'their experience'}]
3. [deep dive question about their project: {project_text[:80] if project_text else 'their work'}]

BEHAVIORAL:
1. [behavioral question specific to {jd_profile.get('job_role')} role]
2. [behavioral question about a challenge relevant to their background]

GAP:
1. [question probing why they lack {missing[0] if len(missing) > 0 else 'a required skill'} and how they would learn it]
2. [question testing their awareness of {missing[1] if len(missing) > 1 else missing[0] if len(missing) > 0 else 'another gap'} and their plan to improve]"""

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()

        if "TECHNICAL:" in raw or "1." in raw:
            return raw

        retry_prompt = f"""Generate 7 interview questions for a {jd_profile.get('job_role')} candidate named {cv_profile.get('candidate_name')}.
They have these skills: {cv_profile.get('skills')}
They are missing: {missing}
Their project: {project_text}

Write exactly like this, no intro:
TECHNICAL:
1. question
2. question
3. question
BEHAVIORAL:
1. question
2. question
GAP:
1. question
2. question"""

        retry = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": retry_prompt}]
        )
        return retry["message"]["content"].strip()

    except Exception as e:
        return f"TECHNICAL:\n1. Error generating questions: {str(e)}\nBEHAVIORAL:\n1. Please try again.\nGAP:\n1. Please try again."


# ── MAIN FUNCTION called by app.py ──
def score_candidate(candidate_name, cv_text, jd_text):

    similarity = tfidf_similarity(jd_text, cv_text)
    jd_profile = extract_requirements(jd_text)
    cv_profile = extract_cv_profile(cv_text)
    result = compare_and_score(jd_profile, cv_profile, similarity, cv_text, jd_text)
    explanation = get_explanation(cv_profile, jd_profile, result)

    name = cv_profile.get("candidate_name", candidate_name)
    if not name or name.lower() in ["unknown", "none", ""]:
        name = candidate_name.replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace("_", " ").strip()

    return {
        "name": name,
        "filename": candidate_name,
        "score": result["total"],
        "scores": {
            "skills":     round((result["skills_score"] / 40) * 100),
            "experience": round((result["exp_score"]    / 30) * 100),
            "education":  round((result["edu_score"]    / 15) * 100),
            "projects":   round((result["proj_score"]   / 15) * 100),
        },
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "similarity": result["similarity_score"],
        "trust_score": result["trust_score"],
        "is_suspicious": result["is_suspicious"],
        "summary": f"Similarity: {similarity}% | {result['cv_years']} yrs exp | Relevant role: {result['role_relevant']}",
        "explanation": explanation,
        "cv_profile": cv_profile,
        "jd_profile": jd_profile,
        "result_raw": result
    }