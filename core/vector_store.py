import chromadb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database"
DOCUMENTS_PATH = PROJECT_ROOT / "documents"

_client = chromadb.PersistentClient(path=str(DATABASE_PATH))


def get_collection():
    return _client.get_or_create_collection(name=DOCUMENTS_PATH.name)

def add(doc_id: str, embedding: list[float], text: str, metadata: dict | None = None):
    get_collection().add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata] if metadata else None,
    )

def query(embedding: list[float], n: int = 5) -> list[dict]:
    """Return top-n chunks with source + chunk_index metadata."""
    result = get_collection().query(
        query_embeddings=[embedding],
        n_results=n,
    )
    hits = []
    docs  = result["documents"][0]
    metas = result["metadatas"][0]
    for i in range(len(docs)):
        hits.append({
            "text":   docs[i],
            "source": metas[i].get("source", "unknown"),
            "chunk":  metas[i].get("chunk_index", i),
        })
    return hits

def delete_source(source: str):
    """Remove all chunks belonging to one file."""
    get_collection().delete(where={"source": source})

def upsert_chunks(ids: list[str], embeddings: list[list[float]],
                  documents: list[str], metadatas: list[dict]):
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

def all_sources() -> dict[str, int]:
    """Return {filename: chunk_count} for everything ingested."""
    result = get_collection().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts

