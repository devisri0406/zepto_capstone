# Zepto Data & AI Platform — Capstone Project

This repository contains the three connected modules required by the capstone brief:

- `/data_pipeline` — scraping, cleaning, INR conversion, SQLite schema, SQL queries and pandas validation.
- `/analytics` — Titanic profiling, EDA, predictive modeling, imbalance handling, tuning, regression and saved pipeline.
- `/support_assistant` — offline-first RAG support assistant using Sentence Transformers, ChromaDB, LangGraph and FastAPI.

---

## Repository Structure

```text
zepto_capstone/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_pipeline/
│   ├── README.md
│   ├── run_pipeline.py
│   ├── queries.sql
│   ├── sql_outputs.txt
│   ├── data/
│   └── database/
│
├── analytics/
│   ├── README.md
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── outputs/
│   └── models/
│
└── support_assistant/
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
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    └── chroma_db/
```

---

## Requirements and Setup

A single consolidated `requirements.txt` is used at the repository root.

The Support Assistant also contains a module-specific `requirements.txt` for its additional dependencies.

### Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

### Install Dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

---

# Module 1 — Data Pipeline

## Overview

The data pipeline collects book data from `books.toscrape.com`, cleans the scraped values, converts prices from GBP to INR using the fixed project-defined exchange rate, stores the data in SQLite, executes SQL queries and validates the SQL results using pandas.

The project-defined conversion rate is:

**1 GBP = 105.50 INR**

This is a fixed assignment constant and not a live market exchange rate.

## Run

From the repository root:

```powershell
cd data_pipeline
python run_pipeline.py
```

## Pipeline Steps

The script:

1. Scrapes five pages from `books.toscrape.com`.
2. Collects at least 60 books.
3. Cleans price, rating and availability fields.
4. Applies median imputation to numeric parsing failures.
5. Creates a normalized SQLite database.
6. Stores categories and books using relational tables.
7. Executes the required SQL queries.
8. Saves SQL results in `sql_outputs.txt`.
9. Reads SQL results into pandas.
10. Reproduces the required JOIN using `pd.merge`.

## Generated Outputs

The pipeline generates:

```text
data_pipeline/
├── database/
│   └── zepto_books.db
└── sql_outputs.txt
```

---

# Module 2 — Analytics

## Overview

The analytics module uses the Titanic dataset to perform data profiling, exploratory data analysis, preprocessing, classification, imbalance handling, hyperparameter tuning and regression.

The Titanic dataset is loaded once in `01_eda.py` using:

```python
sns.load_dataset("titanic")
```

Immediately after loading, it is saved as:

```text
analytics/titanic.csv
```

The modeling module reads the committed CSV instead of loading the dataset again.

## Run

From the repository root:

```powershell
cd analytics
python 01_eda.py
python 02_modeling.py
```

## EDA

The EDA script performs:

- Dataset profiling
- Missing-value analysis
- Cleaning decisions
- Univariate analysis
- Multivariate analysis
- Survival-rate analysis
- Correlation analysis
- Outlier analysis
- Age and fare standardization
- Required charts and visualizations

## Modeling

The modeling script:

- Performs a stratified train/test split.
- Uses preprocessing pipelines.
- Trains Logistic Regression.
- Trains Decision Tree.
- Trains Random Forest.
- Reports confusion matrices.
- Reports accuracy, precision, recall, F1 and ROC/AUC.
- Compares baseline and balanced class weighting.
- Compares SMOTE-based training.
- Applies SMOTE only to the training data.
- Runs `GridSearchCV` on Random Forest.
- Reports the Random Forest OOB score.
- Performs the fare regression task.
- Reports regression metrics.
- Saves the complete preprocessing and model pipeline using `joblib`.
- Reloads the saved pipeline and verifies predictions.

## Generated Outputs

The analytics module produces:

```text
analytics/
├── titanic.csv
├── outputs/
└── models/
```

---

# Module 3 — Support Assistant

## Overview

The Support Assistant is an offline-first Retrieval-Augmented Generation system for answering Zepto policy questions.

The graded baseline uses deterministic mock behavior and does not require an external LLM API key.

The default mode is:

```text
MOCK_LLM=1
```

No LLM API key is required for the graded path.

The system uses:

- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic
- FastAPI

## Run

From the repository root:

```powershell
cd support_assistant
python ingest.py
uvicorn main:app --reload
```

The local API runs at:

```text
http://127.0.0.1:7860
```

Swagger documentation is available at:

```text
http://127.0.0.1:7860/docs
```

## Policy Query Example

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"How long does Zepto delivery take?"}'
```

Expected response structure:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01_chunk_0"],
  "confidence": 1.0
}
```

## General Query Example

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of India?"}'
```

Expected response structure:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

# Support Assistant Architecture

```text
8 policy documents
       |
       v
Document chunking
       |
       v
all-MiniLM-L6-v2 embeddings
       |
       v
ChromaDB
       |
       v
Query
       |
       v
classify_intent
       |
       +-------------------------+
       |                         |
       v                         v
policy_question          general_question
       |                         |
       v                         v
retrieve_and_answer       direct_answer
       |                         |
       +------------+------------+
                    |
                    v
          Pydantic validation
                    |
                    v
               FastAPI /ask
```

The LangGraph workflow contains three main nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The conditional routing sends policy questions to retrieval and general questions to the direct-answer path.

Retrieval uses the local embedding model and ChromaDB.

In `MOCK_LLM` mode, no external LLM network call is made.

---

# MOCK_LLM Mode

The Support Assistant supports deterministic offline execution.

When `MOCK_LLM` is unset or set to `1`:

- Intent classification uses the required keyword-based logic.
- Policy questions are routed to retrieval.
- Relevant document chunks are retrieved from ChromaDB.
- A deterministic context-based answer is generated.
- General questions receive the fixed response.
- No external LLM API call is made.

The optional real-LLM path can be enabled separately using `MOCK_LLM=0`.

The real-LLM path is not required for the graded baseline.

---

# Docker — Support Assistant

A Dockerfile is included for the Support Assistant.

From the `support_assistant` directory:

```powershell
docker build -t zepto-support .
```

Run:

```powershell
docker run --rm -p 7860:7860 zepto-support
```

The Docker container starts Uvicorn on port `7860`.

The Docker API can then be accessed at:

```text
http://127.0.0.1:7860
```

Swagger documentation:

```text
http://127.0.0.1:7860/docs
```

Docker is an optional local execution method for the Support Assistant. The graded baseline can be executed locally using `uvicorn`.

---

# Design Decisions

## Data Pipeline

SQLite was selected as the relational database because it provides lightweight relational storage without requiring a separate database server.

A fixed exchange rate of:

```text
1 GBP = 105.50 INR
```

is used so that the pipeline produces reproducible INR values.

The database is normalized into related tables for categories and books.

Pandas `merge` is used to reproduce the required SQL JOIN and validate the relational result outside SQL.

## Analytics

The Titanic dataset is loaded once and saved locally as `titanic.csv` so that the modeling stage can use an offline dataset.

A stratified train/test split is used to preserve the observed target-class proportions.

Preprocessing is performed inside pipelines to prevent data leakage.

SMOTE is applied only to the training data.

Multiple classification models are evaluated instead of relying on a single classifier.

The complete preprocessing and estimator pipeline is saved so that it can be reloaded and used on raw data.

## Support Assistant

Local Sentence Transformer embeddings and ChromaDB were selected to provide document retrieval without requiring an external vector database.

LangGraph is used to represent the intent-routing and retrieval workflow.

`MOCK_LLM` provides a deterministic offline execution path so that the graded system does not depend on an external LLM provider.

Pydantic is used to validate the structured response.

FastAPI provides the REST API interface.

Docker provides a reproducible runtime configuration for the Support Assistant.

---

# Project Outputs

## Data Pipeline

- Scraped and cleaned book data
- SQLite database
- SQL query outputs
- Pandas JOIN validation

## Analytics

- Offline Titanic dataset
- EDA results
- Visualizations
- Classification metrics
- Imbalance comparison
- GridSearchCV results
- OOB score
- Regression metrics
- Saved and reloadable ML pipeline

## Support Assistant

- Policy document corpus
- Document chunks
- ChromaDB vector index
- Retrieved document sources
- Structured responses
- FastAPI `/ask` endpoint
- Swagger documentation
- Dockerfile

---

# Git Workflow

The project follows a feature-branch workflow as required by the capstone.

The repository history contains:

- Initial project commit
- `feature/capstone` feature branch
- Multiple commits on the feature branch
- Merge of `feature/capstone` into `main`

The Git history can be viewed using:

```powershell
git log --graph --oneline --decorate --all
```

The final project is maintained as one public GitHub repository containing:

```text
/data_pipeline
/analytics
/support_assistant
README.md
```

---

# Academic Integrity

The code, analysis and written interpretations in this repository are intended to represent the student's own project work.

All three modules have been executed and tested locally before submission.

---

# Final Project

The three modules together demonstrate an end-to-end AI/ML workflow:

```text
Data Collection
       |
       v
Data Cleaning
       |
       v
Database Storage
       |
       v
Data Analysis
       |
       v
Machine Learning
       |
       v
Model Evaluation
       |
       v
Document Retrieval
       |
       v
Support Assistant
       |
       v
FastAPI Service
```

The complete project is submitted as a single public GitHub repository.