# Zepto Data \& AI Platform — Capstone Project

This repository contains the three connected modules required by the capstone brief:

* `/data\\\_pipeline` — scraping, cleaning, INR conversion, SQLite schema, SQL queries and pandas validation.
* `/analytics` — Titanic profiling, EDA, predictive modeling, imbalance handling, tuning, regression and saved pipeline.
* `/support\\\_assistant` — offline-first RAG support assistant using Sentence Transformers, ChromaDB, LangGraph and FastAPI.

## Repository structure

```text
zepto\\\_capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── data\\\_pipeline/
│   ├── README.md
│   ├── run\\\_pipeline.py
│   ├── queries.sql
│   ├── sql\\\_outputs.txt
│   ├── data/
│   └── database/
├── analytics/
│   ├── README.md
│   ├── 01\\\_eda.py
│   ├── 02\\\_modeling.py
│   ├── titanic.csv
│   ├── outputs/
│   └── models/
└── support\\\_assistant/
    ├── README.md
    ├── main.py
    ├── rag.py
    ├── prompt.py
    ├── schemas.py
    ├── ingest.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .env.example
    ├── docs/
    │   ├── doc\\\_01.txt
    │   ├── doc\\\_02.txt
    │   ├── doc\\\_03.txt
    │   ├── doc\\\_04.txt
    │   ├── doc\\\_05.txt
    │   ├── doc\\\_06.txt
    │   ├── doc\\\_07.txt
    │   └── doc\\\_08.txt
    └── chroma\\\_db/
```

## Requirements choice

A single consolidated `requirements.txt` is used at the repository root. The support assistant also contains a module-specific requirements file because it has heavier GenAI dependencies.

Install from the root:

```powershell
python -m venv .venv
.venv\\\\Scripts\\\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Windows PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\\\\Scripts\\\\activate
```

## Module 1 — Data Pipeline

The scraper uses `requests` and `BeautifulSoup` against `books.toscrape.com`, using the first five paginated catalogue pages. This produces at least 60 books without login or an API key.

The required project-defined currency rate is:

**1 GBP = 105.50 INR**

This is a fixed assignment constant, not a live market rate.

Run:

```powershell
cd data\\\_pipeline
python run\\\_pipeline.py
```

The script:

1. Scrapes five pages.
2. Cleans price, rating and availability.
3. Applies median imputation to numeric parsing failures.
4. Creates a normalized SQLite database with `categories` and `books`.
5. Runs the required SQL queries.
6. Saves query output in `sql\\\_outputs.txt`.
7. Reads SQL results into pandas and reproduces the JOIN using `pd.merge`.

## Module 2 — Analytics

The analytics module deliberately loads the Titanic dataset once in `01\\\_eda.py` using:

```python
sns.load\\\_dataset("titanic")
```

Immediately after loading, it saves:

```text
analytics/titanic.csv
```

The modeling module then reads that committed CSV and does not call `sns.load\\\_dataset()` again.

Run:

```powershell
cd analytics
python 01\\\_eda.py
python 02\\\_modeling.py
```

The EDA script creates the required profiling output, missing-value analysis, outlier analysis, survival-rate breakdowns, correlation heatmap, four or more multivariate charts, and the age/fare standardization check.

The modeling script:

* performs a stratified train/test split before preprocessing;
* uses a train-only `ColumnTransformer` and `Pipeline`;
* trains Logistic Regression, Decision Tree and Random Forest;
* reports confusion matrices, accuracy, precision, recall, F1 and ROC/AUC;
* compares baseline, `class\\\_weight="balanced"` and SMOTE;
* runs `GridSearchCV` on Random Forest;
* reports OOB score using `RandomForestClassifier(oob\\\_score=True, ...)`;
* performs the fare regression side-task;
* saves the complete preprocessing + estimator pipeline with `joblib.dump`.

## Module 3 — Support Assistant

The graded baseline is fully offline for LLM calls. The default is:

```text
MOCK\\\_LLM=1
```

No LLM API key is required for the graded path.

Embeddings use `all-MiniLM-L6-v2` locally and vectors are stored in ChromaDB.

Run:

```powershell
cd support\\\_assistant
python ingest.py
uvicorn main:app --reload
```

Then POST to:

```text
http://127.0.0.1:8000/ask
```

Example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"query":"How long does Zepto delivery take?"}'
```

General-question example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of India?"}'
```

### Support-assistant architecture

```text
8 policy documents
       |
       v
chunking -> all-MiniLM-L6-v2 embeddings -> ChromaDB
                                           |
query -> classify\\\_intent ------------------+
            |                              |
   policy\\\_question                   general\\\_question
            |                              |
            v                              v
 retrieve\\\_and\\\_answer                 direct\\\_answer
            |
            +------------+
                         v
              Pydantic response schema
                         |
                         v
                    FastAPI /ask
```

Only generation/classification behavior changes with `MOCK\\\_LLM`. Retrieval itself always uses the local embedding model and ChromaDB. In mock mode there is no LLM network call.

## Docker — Support Assistant

From `/support\\\_assistant`:

```powershell
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

Then use:

```text
http://127.0.0.1:7860/ask
```

## Git workflow required by the brief

The capstone requires at least one feature branch that is:

* created from `main`;
* committed at least twice;
* merged back into `main`.

Use the following sequence from the repository root:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial capstone structure"

git checkout -b feature/capstone-modules
git add .
git commit -m "Add capstone modules"
git add .
git commit -m "Complete analysis and support assistant"

git checkout main
git merge --no-ff feature/capstone-modules -m "Merge capstone feature branch"

git log --graph --oneline --decorate --all
```

Create one public GitHub repository, for example:

```text
zepto-capstone
```

Then:

```powershell
git remote add origin https://github.com/YOUR\\\_USERNAME/zepto-capstone.git
git push -u origin main
git push origin feature/capstone-modules
```

The final submission is the single public GitHub repository link.

## Academic-integrity note

The capstone brief explicitly requires that the code, analysis and written interpretations be authored by the student. This package is a working implementation scaffold/reference. Before submission, read every file, run it yourself, understand it, and adapt the written interpretations to your own observed outputs.

## Git Workflow



This project was developed using a feature branch and merged into the main branch.

