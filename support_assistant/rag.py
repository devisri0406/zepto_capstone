import os
from typing import List
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from schemas import GraphState, AskResponse
from prompt import PROMPT_TEMPLATE

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "chroma_db")

KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]


def get_collection():
    client = chromadb.PersistentClient(path=DB)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="zepto_policies",
        embedding_function=ef
    )


def real_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY")
    )


def classify_intent(state: GraphState):
    query = state["query"]
    mock = os.getenv("MOCK_LLM", "1") != "0"

    if mock:
        lowered = query.lower()
        intent = (
            "policy_question"
            if any(keyword in lowered for keyword in KEYWORDS)
            else "general_question"
        )
        return {"intent": intent}

    response = real_llm().invoke(
        f"Classify this query as exactly policy_question or general_question: {query}"
    )
    intent = response.content.strip()
    if intent not in {"policy_question", "general_question"}:
        intent = "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: GraphState):
    query = state["query"]
    collection = get_collection()

    result = collection.query(
        query_texts=[query],
        n_results=3
    )

    docs: List[str] = result.get("documents", [[]])[0]
    ids: List[str] = result.get("ids", [[]])[0]

    if not docs:
        return {
            "answer": "No relevant policy context was found.",
            "sources": [],
            "confidence": 0.0
        }

    mock = os.getenv("MOCK_LLM", "1") != "0"

    if mock:
        snippet = docs[0][:200]
        return {
            "answer": f"Based on the retrieved context: {snippet}",
            "sources": ids,
            "confidence": 1.0
        }

    context = "\n\n".join(
        f"[{chunk_id}] {doc}"
        for chunk_id, doc in zip(ids, docs)
    )
    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=context
    )

    llm = real_llm()

    for attempt in range(3):
        raw = llm.invoke(prompt).content
        try:
            validated = AskResponse.model_validate_json(raw)
            return validated.model_dump()
        except Exception:
            prompt += (
                "\nYour previous response failed JSON validation. "
                "Return only valid JSON matching the requested schema."
            )

    return {
        "answer": "ERROR: the real LLM response failed schema validation.",
        "sources": ids,
        "confidence": 0.0
    }


def direct_answer(state: GraphState):
    mock = os.getenv("MOCK_LLM", "1") != "0"

    if mock:
        return {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }

    raw = real_llm().invoke(
        "Answer this general question briefly and return JSON with "
        "answer, sources, confidence. sources must be empty: "
        + state["query"]
    ).content

    for _ in range(3):
        try:
            validated = AskResponse.model_validate_json(raw)
            return validated.model_dump()
        except Exception:
            raw = real_llm().invoke(
                "Return only valid JSON with fields answer, sources, confidence. "
                "sources must be []. Question: " + state["query"]
            ).content

    return {
        "answer": "ERROR: the real LLM response failed schema validation.",
        "sources": [],
        "confidence": 0.0
    }


def route(state: GraphState):
    return state["intent"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer"
        }
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


app_graph = build_graph()
