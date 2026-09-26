# Module 3 — Support Assistant

## Graded baseline

The service runs with `MOCK_LLM` unset or set to `1`.

No LLM account or API key is required for the graded baseline.

## Setup

From the repository root:

```powershell
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --host 127.0.0.1 --port 7860
```

## Example calls

Policy query:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"How long does delivery take?"}'
```

Expected shape:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01_chunk_0"],
  "confidence": 1.0
}
```

General query:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of India?"}'
```

Expected shape:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Architecture

```text
Document ingestion
    -> per-document chunking
    -> all-MiniLM-L6-v2 embeddings
    -> ChromaDB collection "zepto_policies"
    -> query embedding
    -> LangGraph classify_intent
        -> policy_question
            -> retrieve_and_answer
        -> general_question
            -> direct_answer
    -> Pydantic validation
    -> FastAPI /ask
```

### Components

- `ingest.py`: loads the eight policy documents, chunks each document and stores local embeddings in ChromaDB.
- `rag.py`: creates the ChromaDB client, embedding function and LangGraph graph.
- `prompt.py`: contains the role-context-task-format-length prompt skeleton, negative constraint and few-shot example for the optional real-LLM path.
- `schemas.py`: defines the Pydantic request/response schemas and graph state.
- `main.py`: FastAPI application and `/ask` endpoint.

### MOCK_LLM behavior

- `MOCK_LLM=1` or unset:
  - `classify_intent` uses the required keyword heuristic.
  - Retrieval still happens for policy questions.
  - `retrieve_and_answer` returns a deterministic context-based answer.
  - `direct_answer` returns the fixed general-question response.
  - No LLM API call occurs.
- `MOCK_LLM=0`:
  - Optional real-LLM classification and answer generation are available through Groq.
  - The final answer is validated with Pydantic and retried up to two additional times if validation fails.

## Docker

```powershell
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support

