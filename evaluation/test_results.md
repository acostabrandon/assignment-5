# Assignment 5 -- Option B: Job Fit Analyzer

## Quick Start

```
python setup.py
copy .env.example .env        # then add your HF token
```

Add your data:
- Place 10+ JD files in `data/job_descriptions/`
- Place your resume in `data/resume/`
- Update `data/jd_metadata.csv` with your JD info

```
python -m streamlit run streamlit_app.py
```

## What YOU Need to Do

Search for `TODO` in `streamlit_app.py`:

1. **TODO 1 -- Chunking:** Implement your chosen chunking strategy in `chunk_documents()`
2. **TODO 2 -- Analysis prompts:** Write your 3 prompts: Skill Gap, Keyword Alignment, Fit Summary

All must be developed and iterated in your **notebook** first, then copied here.

## Project Structure

```
a5_option_b/
├── streamlit_app.py     # Main app (edit TODOs)
├── data/
│   ├── job_descriptions/    # YOUR JD files (10+ .txt or .pdf)
│   ├── resume/              # YOUR resume (.txt or .pdf)
│   └── jd_metadata.csv     # JD info (company, title, URL)
├── requirements.txt
├── setup.py
├── .env.example         # Copy to .env, add token
├── .gitignore
└── README.md
```
