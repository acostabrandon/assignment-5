"""
Assignment 5 -- Option B: Job Fit Analyzer
BSAN 6200 | Spring 2026

Run with:
python -m streamlit run streamlit_app.py
"""

# ══════════════════════════════════════════
# Imports
# ══════════════════════════════════════════

import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


# ══════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════

st.set_page_config(
    page_title="Job Fit Analyzer",
    page_icon="🎯",
    layout="wide"
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path("data")
JD_DIR = DATA_DIR / "job_descriptions"
RESUME_DIR = DATA_DIR / "resume"
METADATA_PATH = DATA_DIR / "jd_metadata.csv"
CHROMA_DIR = "chroma_db_streamlit"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════
# File loading helpers
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
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".pdf":
        return load_pdf_file(filepath)

    return load_text_file(filepath)


@st.cache_data
def load_metadata():
    """Load JD metadata CSV."""
    if not METADATA_PATH.exists():
        return pd.DataFrame()

    metadata = pd.read_csv(METADATA_PATH)

    # Clean possible spacing issues
    for col in metadata.columns:
        if metadata[col].dtype == "object":
            metadata[col] = metadata[col].astype(str).str.strip()

    return metadata


@st.cache_data
def load_all_jds():
    """Load all job descriptions from the job_descriptions folder."""
    metadata = load_metadata()
    jd_documents = []

    if metadata.empty:
        return jd_documents

    for _, row in metadata.iterrows():
        filename = row["filename"]
        filepath = JD_DIR / filename

        if filepath.exists() and filepath.suffix.lower() in [".txt", ".pdf"]:
            text = load_file(filepath)

            if text.strip():
                jd_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "doc_type": "job_description",
                            "filename": filename,
                            "company": row.get("company", ""),
                            "title": row.get("title", ""),
                            "source_url": row.get("source_url", ""),
                            "date_collected": row.get("date_collected", "")
                        }
                    )
                )

    return jd_documents


@st.cache_data
def load_resume():
    """Load the first resume file found in data/resume."""
    if not RESUME_DIR.exists():
        return ""

    resume_files = sorted(
        list(RESUME_DIR.glob("*.txt")) + list(RESUME_DIR.glob("*.pdf"))
    )

    if not resume_files:
        return ""

    return load_file(resume_files[0])


# ══════════════════════════════════════════
# Chunking strategy
# ══════════════════════════════════════════

def chunk_documents(documents):
    """
    Final chunking strategy from the notebook:
    smaller, section-aware recursive chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=[
            "\n\nResponsibilities",
            "\n\nQualifications",
            "\n\nRequirements",
            "\n\nPreferred Qualifications",
            "\n\nSkills",
            "\n\nAbout",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_documents(documents)


# ══════════════════════════════════════════
# Cached model/vector resources
# ══════════════════════════════════════════

@st.cache_resource
def load_embeddings():
    """Load OpenAI embedding model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )


@st.cache_resource
def load_llm():
    """Load OpenAI chat model."""
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )


@st.cache_resource
def load_vector_store():
    """
    Build Chroma vector store from JD chunks.
    The app retrieves from selected JD files only.
    """

    jd_documents = load_all_jds()

    if not jd_documents:
        return None, []

    final_chunks = chunk_documents(jd_documents)
    embeddings = load_embeddings()

    # Rebuild local app vector store each session cache build
    # This avoids stale chunks after changing JD files.
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    vector_store = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store, jd_documents


# ══════════════════════════════════════════
# Prompt templates
# IMPORTANT: Replace these with your final notebook prompts
# ══════════════════════════════════════════

SKILL_GAP_PROMPT_TEXT = """
Pretend you are a recruiter reviewing job applications using a resume and job description. You are being provided the job description text file and resume text file. Please compare the job description's requirements with the resume's skill section. 

Your output must include (1) a list of matching skills, (2) a list of missing skills, and (3) recommended actions to close the skill gap. 

You must only use the provided inputs. Do not hallucinate experience, make assumptions, or make changes to the input files.

Job Description: {job_description}
Resume: {resume}
"""


KEYWORD_ALIGNMENT_PROMPT_TEXT = """
Pretend you are a recruiter reviewing resumes to match them to a job description. You are being provided with the job description and resume contexts for review. 

Job Description: {job_description}
Resume: {resume}

Your output will be a table. Column1 of the table will be a list of keywords or phrases directly from the job description. These should be justified, specific, and not repeated. Keep them brief, big-picture keywords from the job description. 

Column 2 will indicate whether each keyword matches the provided resume. If each keyword directly matches the resume, it is a match. If not, it is missing, or requires expansion. This is a Match/No Match result.

Column 3 should serve as your justification from the resume for the match, to ensure that it is proper. For any slight matches, you can include your justification for decision in this column as well. Keep your reasoning concise. 
Below the table, provide a final match rate for the keywords between the job description and resume. 

Do not hallucinate any keywords or skills from either the job description or resume; you are only limited to those. Do not make assumptions. 
"""


FIT_SUMMARY_PROMPT_TEXT = """
Pretend you are a recruiter reviewing resumes and matching them against the job description for a job posting. You are being provided the resume context and job description context. 

Job Description: {job_description} 
Resume: {resume} 

Do a comprehensive analysis of the keywords, experience requirements, and desired skill sets in the job description. Match this against the provided resume. 

Write a short 3-4 sentence narrative assessment of the job fit, make sure it is no longer than 4 sentences. 

Make sure to cite evidence from both the job description and resume. Do not make hallucinations, assumptions from the resume, or exaggerate the fit. Make sure there is no bias. 
"""


ANALYSIS_PROMPTS = {
    "Skill Gap Report": SKILL_GAP_PROMPT_TEXT,
    "Keyword Alignment": KEYWORD_ALIGNMENT_PROMPT_TEXT,
    "Fit Summary": FIT_SUMMARY_PROMPT_TEXT,
}


# ══════════════════════════════════════════
# Retrieval and analysis functions
# ══════════════════════════════════════════

def get_jd_context(vector_store, filename, analysis_type, k=6):
    """Retrieve relevant chunks from the selected JD only."""

    query_map = {
        "Skill Gap Report": "required skills qualifications responsibilities experience tools education",
        "Keyword Alignment": "keywords tools technologies qualifications required skills preferred skills education",
        "Fit Summary": "role responsibilities qualifications required experience preferred qualifications"
    }

    query = query_map.get(
        analysis_type,
        "required skills qualifications responsibilities"
    )

    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter={"filename": filename}
    )

    context = "\n\n".join([doc.page_content for doc in results])

    return context, results


def run_analysis(analysis_type, jd_context, resume_context):
    """Run selected analysis chain."""

    llm = load_llm()
    prompt_text = ANALYSIS_PROMPTS[analysis_type]

    prompt = PromptTemplate(
        input_variables=["job_description", "resume"],
        template=prompt_text
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "job_description": jd_context,
        "resume": resume_context
    })


def get_selected_jd_text(jd_documents, selected_filename):
    """Return full JD text for preview."""
    for doc in jd_documents:
        if doc.metadata.get("filename") == selected_filename:
            return doc.page_content
    return ""


# ══════════════════════════════════════════
# App UI
# ══════════════════════════════════════════

st.title("🎯 Job Fit Analyzer")
st.caption("RAG-powered resume-to-job description comparison tool")

# Error checks before loading resources
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add your API key to a local .env file.")
    st.stop()

metadata = load_metadata()
resume_text = load_resume()
vector_store, jd_documents = load_vector_store()

if metadata.empty:
    st.error("No jd_metadata.csv found or metadata file is empty.")
    st.stop()

if not jd_documents:
    st.error("No job descriptions loaded. Check data/job_descriptions/ and jd_metadata.csv.")
    st.stop()

if not resume_text:
    st.error("No resume found. Add a .txt or .pdf resume file to data/resume/.")
    st.stop()


# Sidebar
with st.sidebar:
    st.header("About this app")
    st.write(
        "This Job Fit Analyzer compares a resume against real job descriptions "
        "using retrieval-augmented generation."
    )

    st.divider()

    st.write("**Instructions**")
    st.write("1. Select a job description.")
    st.write("2. Preview the JD and resume context if needed.")
    st.write("3. Run one analysis type or all three.")
    st.write("4. Review the formatted output.")

    st.divider()

    st.write(f"**JDs loaded:** {len(jd_documents)}")
    st.write("**Resume loaded:** Yes")
    st.write(f"**Embedding model:** {EMBEDDING_MODEL}")
    st.write(f"**Chat model:** {CHAT_MODEL}")

    st.divider()
    st.caption("BSAN 6200 | Assignment 5 | Option B")


# JD selector
st.subheader("1. Select a Job Description")

jd_options = {}

for _, row in metadata.iterrows():
    label = f"{row['company']} — {row['title']}"
    jd_options[label] = row["filename"]

selected_label = st.selectbox(
    "Choose a job description:",
    list(jd_options.keys())
)

selected_filename = jd_options[selected_label]
selected_jd_text = get_selected_jd_text(jd_documents, selected_filename)

selected_row = metadata[metadata["filename"] == selected_filename].iloc[0]

col_info_1, col_info_2, col_info_3 = st.columns(3)

with col_info_1:
    st.metric("Company", selected_row["company"])

with col_info_2:
    st.metric("Role", selected_row["title"])

with col_info_3:
    st.metric("Filename", selected_filename)


# Analysis controls
st.subheader("2. Choose Analysis")

analysis_type = st.radio(
    "Select one analysis type:",
    list(ANALYSIS_PROMPTS.keys()),
    horizontal=True
)

run_one = st.button("🔍 Run Selected Analysis", type="primary", use_container_width=True)
run_all = st.button("📊 Run All 3 Analyses", use_container_width=True)


# Preview section
with st.expander("Preview selected job description and resume"):
    preview_col_1, preview_col_2 = st.columns(2)

    with preview_col_1:
        st.markdown("### Job Description Preview")
        st.text_area(
            label="Selected JD",
            value=selected_jd_text[:4000],
            height=350,
            label_visibility="collapsed"
        )

    with preview_col_2:
        st.markdown("### Resume Preview")
        st.text_area(
            label="Resume",
            value=resume_text[:4000],
            height=350,
            label_visibility="collapsed"
        )


# Run selected analysis
if run_one:
    try:
        with st.spinner(f"Running {analysis_type}..."):
            jd_context, retrieved_chunks = get_jd_context(
                vector_store=vector_store,
                filename=selected_filename,
                analysis_type=analysis_type,
                k=6
            )

            result = run_analysis(
                analysis_type=analysis_type,
                jd_context=jd_context,
                resume_context=resume_text
            )

        st.divider()
        st.subheader(f"Results: {analysis_type}")
        st.markdown(result)

        with st.expander("View retrieved JD context used for this analysis"):
            st.markdown("The app retrieved the following JD chunks from the selected job description.")
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"#### Retrieved Chunk {i}")
                st.write(chunk.page_content)
                st.caption(chunk.metadata)

        with st.expander("Side-by-side context used by the model"):
            side_col_1, side_col_2 = st.columns(2)

            with side_col_1:
                st.markdown("### Retrieved JD Context")
                st.text_area(
                    label="JD Context",
                    value=jd_context,
                    height=400,
                    label_visibility="collapsed"
                )

            with side_col_2:
                st.markdown("### Full Resume Context")
                st.text_area(
                    label="Resume Context",
                    value=resume_text,
                    height=400,
                    label_visibility="collapsed"
                )

    except Exception as e:
        st.error("The analysis could not be completed.")
        st.exception(e)


# Run all analyses
if run_all:
    st.divider()
    st.subheader("Results: All 3 Analyses")

    for current_analysis_type in ANALYSIS_PROMPTS.keys():
        try:
            with st.spinner(f"Running {current_analysis_type}..."):
                jd_context, retrieved_chunks = get_jd_context(
                    vector_store=vector_store,
                    filename=selected_filename,
                    analysis_type=current_analysis_type,
                    k=6
                )

                result = run_analysis(
                    analysis_type=current_analysis_type,
                    jd_context=jd_context,
                    resume_context=resume_text
                )

            with st.container(border=True):
                st.markdown(f"## {current_analysis_type}")
                st.markdown(result)

                with st.expander(f"Retrieved JD context for {current_analysis_type}"):
                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"#### Retrieved Chunk {i}")
                        st.write(chunk.page_content)

        except Exception as e:
            st.error(f"{current_analysis_type} failed.")
            st.exception(e)