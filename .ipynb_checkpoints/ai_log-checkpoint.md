# AI Usage Log

This log documents allowed AI assistance used for Assignment 5, Option B: Job Fit Analyzer. Per the Tier 2 AI usage guidelines, entries are limited to debugging, syntax/API help, boilerplate, learning concepts, and code refactoring. Evaluation interpretation, prompt design decisions, memo drafting, and ground-truth labeling are not included.

---

## Entry 1

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help understanding the overall workflow for Option B and how to organize the project using the provided assignment requirements.  
**What I used:** I used the high-level explanation of the expected folder structure, required files, and major implementation steps.  
**What I modified:** I adapted the suggested structure to match the professor’s starter code instead of building the project from scratch.

---

## Entry 2

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked whether my resume should be saved as a PDF or text file for the project.  
**What I used:** I used the recommendation to save my resume as a `.txt` file to reduce parsing issues and make the app easier to test.  
**What I modified:** I removed personal contact information from the resume file before placing it in the `data/resume/` folder.

---

## Entry 3

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help setting up the OpenAI API path and safely loading the API key in Jupyter Notebook.  
**What I used:** I used the guidance to create a local `.env` file, load it with `python-dotenv`, and access the key using `os.getenv("OPENAI_API_KEY")`.  
**What I modified:** I added `.env` to `.gitignore` so that the API key would not be committed to GitHub.

---

## Entry 4

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help completing the first notebook setup/import block for the paid OpenAI path.  
**What I used:** I used the suggested import structure for `pandas`, `pathlib`, `dotenv`, `OpenAIEmbeddings`, `ChatOpenAI`, `Chroma`, `PromptTemplate`, and `StrOutputParser`.  
**What I modified:** I kept only the imports needed for my final OpenAI-based workflow and removed unnecessary starter-code pieces related to the free HuggingFace path.

---

## Entry 5

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I received an OpenAI `insufficient_quota` / 429 error and asked what it meant.  
**What I used:** I used the explanation that the API key was valid but the account did not have available credits.  
**What I modified:** I added credits to the OpenAI API account and reran the connection test successfully.

---

## Entry 6

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help loading `jd_metadata.csv`, job description text files, and the resume without using LangChain document loaders.  
**What I used:** I used the boilerplate code based on `Path`, `pandas`, and `Document` objects to load the 10 job descriptions and resume manually.  
**What I modified:** I adjusted the file paths and metadata fields to match my actual project folder and `jd_metadata.csv`.

---

## Entry 7

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help building two chunking strategy code blocks and a comparison table.  
**What I used:** I used the code structure for testing two `RecursiveCharacterTextSplitter` strategies and comparing total chunks, average chunk length, minimum length, maximum length, JD chunks, and resume chunks.  
**What I modified:** I selected the chunking strategy based on my own review of the comparison results and the structure of my job descriptions.

---

## Entry 8

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help creating the embeddings and vector store section using the paid OpenAI path.  
**What I used:** I used the boilerplate code for `OpenAIEmbeddings(model="text-embedding-3-small")`, ChromaDB vector store creation, and a test similarity search.  
**What I modified:** I used my final selected chunks from the notebook and confirmed that the similarity search returned relevant job description and resume sections.

---

## Entry 9

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for technical guidance on setting up the analysis chain structure without having AI make my prompt design decisions.  
**What I used:** I used the provided chain structure with `PromptTemplate`, `ChatOpenAI`, and `StrOutputParser`, including placeholders for `{job_description}` and `{resume}`.  
**What I modified:** I wrote and revised my own prompt language inside the template placeholders to comply with the assignment’s Tier 2 restrictions.

---

## Entry 10

**Date:** May 12, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked why my analysis output was asking me to provide a resume and job description even though I had already loaded the context.  
**What I used:** I used the debugging explanation that my prompt template did not include the `{job_description}` and `{resume}` placeholders, so the model was not receiving the retrieved context.  
**What I modified:** I updated the prompt templates to include the placeholders and reran the chains successfully.

---

## Entry 11

**Date:** May 13, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help setting up the evaluation code so I could run all 9 required analyses and manually evaluate them myself.  
**What I used:** I used the boilerplate loop structure for running 3 analysis types across 3 selected job descriptions and organizing the outputs in a dataframe.  
**What I modified:** I completed the actual evaluation scoring and written interpretation myself based on the assignment’s required criteria.

---

## Entry 12

**Date:** May 13, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for debugging help when the evaluation loop failed with an `IndexError` related to job description metadata.  
**What I used:** I used the diagnostic code to compare filenames in `jd_metadata.csv`, loaded documents, and the vector store.  
**What I modified:** I adjusted the evaluation workflow so that filenames were selected directly from the metadata dataframe instead of relying on manually typed filenames.

---

## Entry 13

**Date:** May 13, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help understanding why the system incorrectly stated that my resume did not show a bachelor’s degree.  
**What I used:** I used the debugging explanation that the retrieved resume chunks may not have included the education section.  
**What I modified:** This is the case where my approach improved on the AI suggestion. Instead of relying only on retrieved resume chunks, I decided to pass the full resume text into the analysis while still retrieving relevant chunks from the selected job description. This worked better because the resume is short enough to fit in the prompt and prevents important sections like education from being omitted.

---

## Entry 14

**Date:** May 13, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help finalizing the `streamlit_app.py` file using the professor’s starter code and the notebook pipeline.  
**What I used:** I used boilerplate and refactoring guidance for Streamlit layout, `@st.cache_data`, `@st.cache_resource`, dropdown selection, analysis buttons, formatted output, sidebar instructions, and error handling.  
**What I modified:** I inserted my own final prompt templates from the notebook and adjusted the app to use the OpenAI paid path instead of the starter HuggingFace path.

---

## Entry 15

**Date:** May 13, 2026  
**Tool:** ChatGPT  
**What I asked:** I asked for help creating a final AI usage log that only includes allowed AI assistance under the Tier 2 guidelines.  
**What I used:** I used the structured markdown format with the required fields: date, tool, what I asked, what I used, and what I modified.  
**What I modified:** I reviewed the entries to ensure they only documented allowed use cases such as debugging, syntax/API help, boilerplate, learning concepts, and code refactoring. I excluded prohibited uses such as evaluation interpretation, prompt design decisions, memo drafting, and ground-truth labeling.