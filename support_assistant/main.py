from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from rag import app_graph

app = FastAPI(title="Zepto Policy Support Assistant")


@app.get("/")
def root():
    return {"service": "Zepto Policy Support Assistant", "status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = app_graph.invoke({
        "query": request.query
    })
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )
