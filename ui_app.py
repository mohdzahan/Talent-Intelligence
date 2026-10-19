import os
import streamlit as st
import httpx
from pypdf import PdfReader
from dotenv import load_dotenv
from agent.orchestrator import evaluate_candidates_with_llm

load_dotenv()

TALENT_API_URL = os.getenv("TALENT_API_URL", "http://localhost:8000/search")
INGEST_API_URL = TALENT_API_URL.replace("/search", "/ingest")

st.set_page_config(
    page_title="Talent Intelligence | Autonomous AI Recruiter",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"    
)

# Custom Styling for LinkedIn Screenshots
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .badge-hire {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
    .badge-borderline {
        background-color: #78350f;
        color: #fcd34d;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
    .badge-reject {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Enterprise Talent Intelligence Engine")
st.caption("Asynchronous Two-Stage Hybrid Retrieval & Autonomous Candidate Evaluation")

with st.sidebar:
    st.header("System Controls")
    api_status = "🟢 Live on AWS EC2" if "http" in TALENT_API_URL else "🟡 Localhost"
    st.info(f"**Backend Status:** {api_status}\n\n**Endpoint:** `{TALENT_API_URL}`")
    limit = st.slider("Max Candidates to Retrieve", min_value=1, max_value=10, value=3)
    use_reranker = st.toggle("Enable Cross-Encoder Reranker", value=True)

tab_search, tab_ingest = st.tabs(["🔍 Candidate Search & Agent Evaluation", "📤 Ingest New Resumes"])

# ----------------- TAB 1: SEARCH & EVALUATION -----------------
with tab_search:
    st.subheader("1. Define Hiring Requirement")
    job_brief = st.text_area(
        "Job Description / Technical Brief",
        value="Need an experienced Data Engineer with PySpark and AWS Glue to build big data pipelines.",
        height=100
    )

    if st.button("🚀 Run Agentic Screening", type="primary"):
        with st.spinner("Step 1: Hybrid Retrieval (BGE-M3 Dense + BM25 Sparse)..."):
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(
                        TALENT_API_URL,
                        json={"query": job_brief, "limit": limit, "rerank": use_reranker}
                    )
                    resp.raise_for_status()
                    candidates = resp.json().get("results", [])
            except Exception as e:
                st.error(f"Retrieval Service Error: {e}")
                candidates = []

        if candidates:
            with st.spinner("Step 2: Autonomous Agent Generating Structured Scorecards..."):
                try:
                    eval_report = evaluate_candidates_with_llm(job_brief, candidates)
                    st.success(f"Evaluated {len(eval_report.evaluations)} candidate profiles against requirement.")
                    
                    st.markdown("### 2. Candidate Evaluation Reports")
                    for evaluation in eval_report.evaluations:
                        rec = evaluation.recommendation.value
                        badge_class = "badge-hire" if rec == "Strong Hire" else ("badge-borderline" if rec == "Borderline" else "badge-reject")
                        
                        with st.container():
                            col1, col2, col3 = st.columns([1, 1, 2])
                            col1.metric("Match Score", f"{evaluation.match_score:.0f}/100")
                            col2.markdown(f"**Recommendation**<br><span class='{badge_class}'>{rec}</span>", unsafe_allow_html=True)
                            col3.markdown(f"**Primary Evidence Quote:**<br>*\"{evaluation.text_evidence}\"*", unsafe_allow_html=True)
                            
                            st.write(f"**Executive Rationale:** {evaluation.summary_rationale}")
                            
                            with st.expander("Detailed Skill Breakdown"):
                                col_a, col_b = st.columns(2)
                                col_a.write("**Matched Skills:**")
                                for s in evaluation.matched_skills:
                                    col_a.markdown(f"- ✅ `{s}`")
                                col_b.write("**Missing / Gap Skills:**")
                                for s in evaluation.missing_skills:
                                    col_b.markdown(f"- ❌ `{s}`")
                            st.divider()
                except Exception as e:
                    st.error(f"Agent Evaluation Failed: {e}")
        else:
            st.warning("No candidate records found matching this criteria.")

# ----------------- TAB 2: INGEST RESUMES -----------------
with tab_ingest:
    st.subheader("Add Candidate CV to Database")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    
    cv_text = ""
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            cv_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            cv_text = uploaded_file.read().decode("utf-8")

    manual_text = st.text_area("Or Paste CV Text Directly", value=cv_text, height=180)
    skills_raw = st.text_input("Explicit Skills (Comma-separated)", placeholder="PySpark, Databricks, SQL, AWS")

    if st.button("📥 Index into Qdrant", type="secondary"):
        if not manual_text.strip():
            st.error("Please provide candidate CV text.")
        else:
            skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
            with st.spinner("Generating BGE-M3 Dense + BM25 Sparse Vectors & Upserting to AWS..."):
                try:
                    with httpx.Client(timeout=25.0) as client:
                        resp = client.post(
                            INGEST_API_URL,
                            json={"text": manual_text, "document_type": "CV", "skills": skills_list}
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        st.success(f"Candidate indexed successfully! Assigned Point ID: `{result['point_id']}`")
                except Exception as e:
                    st.error(f"Failed to ingest resume: {e}")