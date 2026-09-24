from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
COLLECTION = "zepto_policies"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef
    )


def chunk_text(text, max_chars=1200):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars
    return chunks


def ingest():
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            ids.append(f"{path.stem}_chunk_{i}")
            documents.append(chunk)
            metadatas.append({
                "document_id": path.stem,
                "chunk_index": i
            })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Indexed {len(ids)} chunks from {len(list(DOCS.glob('doc_*.txt')))} documents.")


if __name__ == "__main__":
    ingest()
