"""
Assignment 5 -- Option B: Job Fit Analyzer
BSAN 6200 | Spring 2026

Run locally with:
python -m streamlit run streamlit_app.py

Deployment note:
This Streamlit version uses an in-memory vector store instead of ChromaDB
to avoid Chroma/protobuf deployment issues on Streamlit Community Cloud.
"""

# ══════════════════════════════════════════
# Imports
# ══════════════════════════════════════════

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ══════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════

st.set_page_config(
    page_title="Job Fit Analyzer",
    page_icon="🎯",
    layout="wide"
)

load_dotenv()

# API key loading:
# - Locally: .env file
# - Streamlit Cloud: Manage app > Settings > Secrets
OPENAI_API_KEY = None

try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ══════════════════════════════════════════
# Paths and constants
# ══════════════════════════════════════════

DATA_DIR = Path("data")
JD_DIR = DATA_DIR / "job_descriptions"
RESUME_DIR = DATA_DIR / "resume"
METADATA_PATH = DATA_DIR / "jd_metadata.csv"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════
# File loading helpers
# ══════════════════════════════════════════

def load_text_file(filepath: Path) -> str:
    """Load a .txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(filepath: Path) -> str:
    """Load a .pdf file."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n".join(text)


def load_file(filepath: Path) -> str:
    """Load a .txt or .pdf file."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".pdf":
        return load_pdf_file(filepath)

    return load_text_file(filepath)


@st.cache_data
def load_metadata() -> pd.DataFrame:
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
def load_all_jds() -> list:
    """Load all job descriptions from data/job_descriptions/."""
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
def load_resume() -> str:
    """Load the first resume file found in data/resume/."""
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

def chunk_documents(documents: list) -> list:
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
# Lightweight in-memory vector store
# ══════════════════════════════════════════

class InMemoryVectorStore:
    """
    Lightweight vector store for Streamlit deployment.

    This avoids ChromaDB deployment issues on Streamlit Cloud while keeping
    the same retrieval idea: embed chunks, embed query, return most similar chunks.
    """

    def __init__(self, documents: list, embeddings_model: OpenAIEmbeddings):
        self.documents = documents
        self.embeddings_model = embeddings_model

        texts = [doc.page_content for doc in documents]
        vectors = embeddings_model.embed_documents(texts)

        self.vectors = np.array(vectors, dtype=np.float32)

        # Normalize document vectors for cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = self.vectors / norms

    def similarity_search(self, query: str, k: int = 5, filter: dict | None = None) -> list:
        """Return top-k most similar documents, optionally filtered by metadata."""
        query_vector = np.array(
            self.embeddings_model.embed_query(query),
            dtype=np.float32
        )

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1

        query_vector = query_vector / query_norm

        # Candidate indices based on metadata filter
        candidate_indices = []

        for i, doc in enumerate(self.documents):
            include_doc = True

            if filter:
                for key, value in filter.items():
                    if doc.metadata.get(key) != value:
                        include_doc = False
                        break

            if include_doc:
                candidate_indices.append(i)

        if not candidate_indices:
            return []

        candidate_vectors = self.vectors[candidate_indices]
        similarities = candidate_vectors @ query_vector

        top_local_indices = np.argsort(similarities)[::-1][:k]
        top_doc_indices = [candidate_indices[i] for i in top_local_indices]

        return [self.documents[i] for i in top_doc_indices]


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
    Build an in-memory vector store from JD chunks.
    """

    jd_documents = load_all_jds()

    if not jd_documents:
        return None, []

    final_chunks = chunk_documents(jd_documents)
    embeddings = load_embeddings()

    vector_store = InMemoryVectorStore(
        documents=final_chunks,
        embeddings_model=embeddings
    )

    return vector_store, jd_documents


# ══════════════════════════════════════════
# Prompt templates
# Replace these with the final prompts from your notebook if needed.
# Keep {job_description} and {resume}.
# ══════════════════════════════════════════

SKILL_GAP_PROMPT_TEXT = """
You are reviewing a candidate's resume against a specific job description.

Use only the information provided in the job description context and resume context below.
Do not ask for more files.
Do not invent skills or experience that are not supported by the provided text.

Job Description Context:
{job_description}

Resume Context:
{resume}

Task:
Compare the resume against the job description and produce a skill gap report.

Your report should include:
1. Matching skills or experience found in both the job description and resume
2. Skills, tools, or qualifications from the job description that are missing or weak in the resume
3. Recommended actions the candidate could take to close each gap

Format your answer with clear section headings.
"""


KEYWORD_ALIGNMENT_PROMPT_TEXT = """
You are reviewing keyword alignment between a job description and a candidate resume.

Use only the information provided in the job description context and resume context below.
Do not ask for more files.
Do not invent skills or experience that are not supported by the provided text.

Job Description Context:
{job_description}

Resume Context:
{resume}

Task:
Extract important keywords and phrases from the job description, then compare them against the resume.

Your output should include:
1. A table of important keywords or phrases from the job description
2. Whether each keyword is an exact match, semantic match, or no match in the resume
3. A brief justification for each match decision
4. A final match rate

Important:
Ignore legal disclaimers, equal opportunity statements, application instructions, and general recruitment process language.
Focus on skills, tools, platforms, responsibilities, qualifications, and role requirements.
"""


FIT_SUMMARY_PROMPT_TEXT = """
You are comparing a candidate's resume to a specific job description.

Use only the job description context and resume context provided below.
Do not ask for additional files.
Do not make claims that are not supported by the provided context.

Job Description Context:
{job_description}

Resume Context:
{resume}

Task:
Write a concise 3-4 sentence fit summary for this role.

Your summary should:
- State the candidate's overall fit for the role
- Mention the strongest areas of alignment
- Mention one important gap or limitation if present
- Cite specific evidence from both the job description and the resume context
"""


ANALYSIS_PROMPTS = {
    "Skill Gap Report": SKILL_GAP_PROMPT_TEXT,
    "Keyword Alignment": KEYWORD_ALIGNMENT_PROMPT_TEXT,
    "Fit Summary": FIT_SUMMARY_PROMPT_TEXT,
}


# ══════════════════════════════════════════
# Retrieval and analysis functions
# ══════════════════════════════════════════

def get_jd_context(vector_store, filename: str, analysis_type: str, k: int = 6):
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


def run_analysis(analysis_type: str, jd_context: str, resume_context: str) -> str:
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


def get_selected_jd_text(jd_documents: list, selected_filename: str) -> str:
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

# API key check
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not found. For local use, add it to a .env file. "
        "For Streamlit Cloud, add it under Manage app > Settings > Secrets."
    )
    st.stop()

# Load resources
try:
    metadata = load_metadata()
    resume_text = load_resume()
    vector_store, jd_documents = load_vector_store()
except Exception as e:
    st.error("The app could not load required resources.")
    st.exception(e)
    st.stop()

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
    st.write("**Vector store:** In-memory cosine similarity")

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

run_col_1, run_col_2 = st.columns(2)

with run_col_1:
    run_one = st.button(
        "🔍 Run Selected Analysis",
        type="primary",
        use_container_width=True
    )

with run_col_2:
    run_all = st.button(
        "📊 Run All 3 Analyses",
        use_container_width=True
    )


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

            if not jd_context.strip():
                st.warning("No JD context was retrieved for this selection.")
                st.stop()

            result = run_analysis(
                analysis_type=analysis_type,
                jd_context=jd_context,
                resume_context=resume_text
            )

        st.divider()
        st.subheader(f"Results: {analysis_type}")
        st.markdown(result)

        with st.expander("View retrieved JD context used for this analysis"):
            st.markdown(
                "The app retrieved the following JD chunks from the selected job description."
            )

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

                if not jd_context.strip():
                    st.warning(
                        f"No JD context was retrieved for {current_analysis_type}."
                    )
                    continue

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
                        st.caption(chunk.metadata)

        except Exception as e:
            st.error(f"{current_analysis_type} failed.")
            st.exception(e)