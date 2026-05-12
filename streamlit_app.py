"""
Assignment 5 -- Option B: Job Fit Analyzer
BSAN 6200 | Spring 2026

Run with: python -m streamlit run streamlit_app.py

YOUR TASKS (search for TODO):
1. Implement your chunking strategy
2. Write 3 analysis prompts (Skill Gap, Keyword Alignment, Fit Summary)
3. These must be developed and iterated in your notebook first
"""

import streamlit as st
import chromadb
from huggingface_hub import InferenceClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Job Fit Analyzer", page_icon="🎯", layout="wide")

# ── Config ──
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"


# ══════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════

def load_text_file(filepath):
    """Load a .txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(filepath):
    """Load a .pdf file."""
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(filepath):
    """Load a .txt or .pdf file."""
    if filepath.endswith(".pdf"):
        return load_pdf_file(filepath)
    else:
        return load_text_file(filepath)


def load_all_jds(jd_dir="data/job_descriptions"):
    """Load all JD files from the directory."""
    docs = []
    if not os.path.exists(jd_dir):
        return docs
    for filename in sorted(os.listdir(jd_dir)):
        filepath = os.path.join(jd_dir, filename)
        if filename.endswith((".txt", ".pdf")):
            text = load_file(filepath)
            if text.strip():
                docs.append({"text": text, "source": filename})
    return docs


def load_resume(resume_dir="data/resume"):
    """Load the resume file (first .txt or .pdf found)."""
    if not os.path.exists(resume_dir):
        return ""
    for filename in os.listdir(resume_dir):
        if filename.endswith((".txt", ".pdf")):
            return load_file(os.path.join(resume_dir, filename))
    return ""


def ask_llm(hf_client, prompt):
    """Send a prompt to the LLM and return the response."""
    try:
        response = hf_client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
            stop=["\n\n\n"],
            provider="hf-inference",
        )
    except TypeError:
        response = hf_client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
            stop=["\n\n\n"],
        )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════
# TODO 1: Chunking strategy
# ══════════════════════════════════════════

def chunk_documents(documents):
    """
    TODO: Implement your chunking strategy here.

    Input: list of dicts [{"text": "...", "source": "filename"}, ...]
    Output: list of dicts [{"text": "chunk text", "source": "filename"}, ...]
    """
    # PLACEHOLDER: no chunking
    return documents


# ══════════════════════════════════════════
# TODO 2: Analysis prompts
# Write your 3 analysis prompts below.
# Must be developed in your notebook first
# with 3+ iterations documented.
# ══════════════════════════════════════════

SKILL_GAP_PROMPT = """TODO: Write your Skill Gap analysis prompt here.

It should:
- Compare the candidate's resume against the job description
- List matching skills, missing skills, and recommended actions
- Use ONLY the provided context

Delete this placeholder and write your own."""


KEYWORD_PROMPT = """TODO: Write your Keyword Alignment prompt here.

It should:
- Extract key terms from the JD
- Check which appear (or have equivalents) in the resume
- Report a match percentage

Delete this placeholder and write your own."""


FIT_SUMMARY_PROMPT = """TODO: Write your Fit Summary prompt here.

It should:
- Produce a 3-4 sentence narrative assessment
- Cite specific evidence from both documents
- Be balanced (strengths and gaps)

Delete this placeholder and write your own."""


ANALYSIS_TYPES = {
    "Skill Gap Report": SKILL_GAP_PROMPT,
    "Keyword Alignment": KEYWORD_PROMPT,
    "Fit Summary": FIT_SUMMARY_PROMPT,
}


# ══════════════════════════════════════════
# Load resources (cached)
# ══════════════════════════════════════════

@st.cache_resource
def load_vectorstore():
    jd_docs = load_all_jds()
    if not jd_docs:
        return None, []
    chunks = chunk_documents(jd_docs)
    client = chromadb.Client()
    collection = client.create_collection("job_fit")
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection, jd_docs


@st.cache_data
def load_metadata():
    path = "data/jd_metadata.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_llm():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return None
    return InferenceClient(token=token)


# ══════════════════════════════════════════
# Analysis logic
# ══════════════════════════════════════════

def run_analysis(hf_client, analysis_prompt, jd_text, resume_text):
    """Run a single analysis: format prompt with JD + resume, call LLM."""
    full_prompt = f"""{analysis_prompt}

Job Description:
{jd_text}

Candidate Resume:
{resume_text}"""

    return ask_llm(hf_client, full_prompt)


# ══════════════════════════════════════════
# UI
# ══════════════════════════════════════════

collection, jd_docs = load_vectorstore()
metadata = load_metadata()
hf_client = load_llm()
resume_text = load_resume()

st.title("🎯 Job Fit Analyzer")
st.caption("Compare your resume against job descriptions.")

# ── Error checks ──
if not hf_client:
    st.error("HF_TOKEN not found. Add it to your .env file.")
    st.stop()

if collection is None:
    st.error("No JDs found in data/job_descriptions/. Add your job description files there.")
    st.stop()

if not resume_text:
    st.error("No resume found in data/resume/. Add your resume file there.")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write("This tool compares your resume against job descriptions using RAG.")
    st.write(f"**JDs loaded:** {len(jd_docs)}")
    st.write(f"**Resume loaded:** Yes")
    st.write(f"**Model:** {MODEL_ID}")
    st.divider()
    st.caption("BSAN 6200 | Assignment 5 | Option B")

# ── JD selector ──
col_select, col_analysis = st.columns([1, 1])

with col_select:
    st.subheader("1. Select a Job Description")

    if not metadata.empty:
        jd_options = {
            f"{row['company']} -- {row['title']}": row["filename"]
            for _, row in metadata.iterrows()
        }
    else:
        jd_options = {doc["source"]: doc["source"] for doc in jd_docs}

    selected_label = st.selectbox("Choose a JD:", list(jd_options.keys()))
    selected_filename = jd_options[selected_label]

    # Find the JD text
    jd_text = ""
    for doc in jd_docs:
        if doc["source"] == selected_filename:
            jd_text = doc["text"]
            break

    with st.expander("Preview job description"):
        st.text(jd_text[:1000] + ("..." if len(jd_text) > 1000 else ""))

with col_analysis:
    st.subheader("2. Choose Analysis Type")
    analysis_type = st.radio(
        "Select analysis:",
        list(ANALYSIS_TYPES.keys()),
    )

# ── Run analysis ──
st.divider()

if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
    with st.spinner(f"Running {analysis_type}..."):
        try:
            prompt = ANALYSIS_TYPES[analysis_type]
            result = run_analysis(hf_client, prompt, jd_text, resume_text)
            st.subheader(f"Results: {analysis_type}")
            st.markdown(result)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

st.divider()

if st.button("📊 Run All 3 Analyses", use_container_width=True):
    for name, prompt in ANALYSIS_TYPES.items():
        with st.spinner(f"Running {name}..."):
            try:
                result = run_analysis(hf_client, prompt, jd_text, resume_text)
                st.subheader(name)
                st.markdown(result)
                st.divider()
            except Exception as e:
                st.error(f"{name} failed: {str(e)}")
