# Assignment 5 — Option B: Job Fit Analyzer

## 1. Project Title and Option

**Project Title:** Job Fit Analyzer: Resume-to-Job Description Matching with RAG  
**Assignment Option:** Option B — Job Fit Analyzer  
**Course:** BSAN 6200: Text Mining and Social Media Analytics  
**Date:** May 2026  

---

## 2. Student Name

**Name:** Brandon Acosta

---

## 3. Project Description

This project builds a RAG-powered Job Fit Analyzer that compares a resume against real job descriptions. The system allows a user to select a job description from a corpus of 10 real postings and run three structured analyses:

1. **Skill Gap Report**  
   Compares the selected job description against the resume and identifies matching skills, missing or weak skills, and recommended actions.

2. **Keyword Alignment**  
   Extracts important keywords and phrases from the job description, compares them against the resume, and reports a keyword match rate.

3. **Fit Summary**  
   Produces a short narrative assessment of the candidate’s fit for the selected role, using evidence from the job description and resume.

The project includes a Jupyter notebook for developing and testing the RAG pipeline, a Streamlit app for user interaction, and an evaluation folder documenting the analyzer’s performance across three target job descriptions.

---

## 4. Setup Instructions

### Step 1: Clone or download the repository

```bash
git clone <your-repository-link>
cd bsan6200-assignment5
```

### Step 2: Create and activate a virtual environment

Recommended for Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install required packages

```bash
pip install -r requirements.txt
```

### Step 4: Create a local `.env` file

Create a file named `.env` in the main project folder.

Add the following line:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is intentionally excluded from GitHub using `.gitignore` so the API key is not committed to the repository.

### Step 5: Confirm data files are in place

The project expects the following files and folders:

```text
data/
├── job_descriptions/
│   ├── jd_01_sales_operations_analyst.txt
│   ├── jd_02_revenue_operations_analyst.txt
│   ├── jd_03_crm_operations_analyst.txt
│   ├── jd_04_gtm_operations_analyst.txt
│   ├── jd_05_business_analyst.txt
│   ├── jd_06_data_analyst.txt
│   ├── jd_07_business_intelligence_analyst.txt
│   ├── jd_08_operations_analyst.txt
│   ├── jd_09_strategy_analyst.txt
│   └── jd_10_associate_consultant_operations.txt
├── resume/
│   └── resume.txt
└── jd_metadata.csv
```

### Step 6: Run the Jupyter notebook

Open and run:

```text
notebooks/rag_pipeline.ipynb
```

The notebook loads the data, compares chunking strategies, creates embeddings, builds the vector store, tests the analysis prompts, runs the zero-shot vs. few-shot comparison, and supports the evaluation workflow.

### Step 7: Run the Streamlit app

From the main project folder, run:

```bash
python -m streamlit run streamlit_app.py
```

The app allows the user to select a job description and run the Skill Gap Report, Keyword Alignment, and Fit Summary analyses.

---

## 5. Models and Tools Used

### Models

- **Embedding Model:** OpenAI `text-embedding-3-small`
- **Chat Model:** OpenAI `gpt-4o-mini`

### Tools and Libraries

- Python
- Jupyter Notebook
- Streamlit
- LangChain
- ChromaDB
- OpenAI API
- Pandas
- python-dotenv
- pypdf
- GitHub

### RAG Pipeline Components

- Job descriptions and resume loaded from local files
- Text chunking using `RecursiveCharacterTextSplitter`
- Embeddings generated with OpenAI `text-embedding-3-small`
- Vector storage and retrieval using ChromaDB
- Prompt templates built with LangChain
- Streamlit interface for selecting job descriptions and running analyses

---

## 6. Paid vs. Free Path

This project uses the **paid OpenAI API path**.

### Paid Path Used

- OpenAI embeddings: `text-embedding-3-small`
- OpenAI chat model: `gpt-4o-mini`
- API key stored locally in `.env`

### Reason for Choosing Paid Path

The paid OpenAI path was selected because it provided a reliable embedding model, straightforward API integration, and consistent results for the RAG pipeline. The project required semantic matching between job description requirements and resume content, so using OpenAI embeddings made the retrieval step more accurate and easier to implement.

### Free Path Not Used

The assignment also allowed free options such as HuggingFace, sentence-transformers, or local models. However, this project used OpenAI because the paid path was more stable for the available development time and aligned with the assignment’s recommended paid implementation option.

---

## 7. Key Findings

The analyzer was evaluated on three target job descriptions:

1. JD_01 — Sales Operations Analyst  
2. JD_02 — Revenue Operations Associate  
3. JD_03 — CRM Operations Associate  

Each job description was tested across three analysis types:

- Skill Gap Report
- Keyword Alignment
- Fit Summary

This produced 9 total analyses.

### Main Findings

The **Fit Summary** analysis performed best overall. It consistently produced clear, faithful, and actionable summaries that connected the resume to the selected job descriptions. It was especially useful because it summarized both strengths and gaps without overwhelming the user with too much detail.

The **Skill Gap Report** performed well for the Revenue Operations and CRM Operations job descriptions. It was strongest when the job description contained clear hard skills, tools, or role requirements. However, it was less effective when the job description included broad soft skills because the model sometimes treated those as missing even when related experience was present in the resume.

The **Keyword Alignment** analysis was the most inconsistent. It was useful for identifying overlap between resume language and job description language, but it sometimes pulled irrelevant terms from job posting sections such as recruitment notices or general applicant instructions. This made some keyword match rates less reliable.

### Best Performing Job Description

The CRM Operations Associate job description produced the strongest results overall. The model was able to identify relevant hard skills, CRM experience, automation experience, and industry-related experience. The outputs were organized, faithful to the documents, and actionable.

### Weakest Performing Job Description

The Sales Operations Analyst job description produced weaker results in some analysis types. In particular, the system initially missed education-related resume content and sometimes treated soft skills as missing rather than making reasonable connections from existing experience.

### Important Pipeline Improvement

One important improvement was passing the **full resume text** into the analysis prompts instead of relying only on retrieved resume chunks. Since the resume was short enough to fit in the prompt, this reduced the risk of missing important sections such as education, skills, or projects. The system still used retrieval for the selected job description, but the full resume context improved completeness and reduced inaccurate “no match” results.

---

## 8. File Descriptions

```text
bsan6200-assignment5/
├── README.md
├── memo.md
├── requirements.txt
├── ai_log.md
├── .gitignore
├── streamlit_app.py
├── data/
│   ├── job_descriptions/
│   ├── resume/
│   └── jd_metadata.csv
├── notebooks/
│   └── rag_pipeline.ipynb
└── evaluation/
```

### Main Files

#### `README.md`

Provides the project overview, setup instructions, tools used, paid/free path explanation, key findings, and file descriptions.

#### `memo.md`

Business-facing memo summarizing the project, methodology, findings, limitations, and recommendations.

#### `requirements.txt`

Lists the Python packages required to run the notebook and Streamlit app.

#### `ai_log.md`

Documents allowed AI usage for the project, following the assignment’s Tier 2 AI usage guidelines.

#### `.gitignore`

Prevents sensitive or unnecessary files from being committed to GitHub, including `.env`, cache folders, Python cache files, and local vector database folders.

#### `streamlit_app.py`

Main Streamlit application. It allows users to select a job description, run the three analysis types, view formatted results, and inspect the context used by the model.

---

### Data Folder

#### `data/job_descriptions/`

Contains 10 real job descriptions saved as individual text files.

#### `data/resume/`

Contains the resume file used for comparison. Sensitive contact information was removed before use.

#### `data/jd_metadata.csv`

Contains metadata for each job description, including filename, company, title, source URL, and collection date.

---

### Notebook Folder

#### `notebooks/rag_pipeline.ipynb`

Main development notebook. It includes:

- Setup and imports
- API key loading
- Job description and resume loading
- Chunking strategy comparison
- Embedding and vector store creation
- Analysis chain testing
- Zero-shot vs. few-shot comparison
- Evaluation workflow

---

### Evaluation Folder

#### `evaluation/`

Contains evaluation materials and outputs for the analyzer.

Possible files include:

- `analysis_outputs.csv`
- `manual_evaluation_summary.csv`
- `analysis_type_summary.csv`
- `jd_summary.csv`
- `test_results.md`
- Evaluation PDF/table used to summarize performance

---

## Running the Project

To run the notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/rag_pipeline.ipynb
```

To run the Streamlit app:

```bash
python -m streamlit run streamlit_app.py
```

---

## Notes on API Key Safety

The OpenAI API key is stored in a local `.env` file and loaded with `python-dotenv`. The `.env` file is excluded from GitHub and should not be uploaded or shared publicly.

Required `.gitignore` entries include:

```text
.env
chroma_db/
chroma_db_streamlit/
__pycache__/
.ipynb_checkpoints/
.DS_Store
```