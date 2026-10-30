from mcp.server import MCPServer
from core import embedder, rag, vector_store
from core.vector_store import delete_source

mcp = MCPServer("OmniVerse RAG Server")


@mcp.tool()
def ask_documents(question: str) -> str:
    """
    Answer a question using the ingested documents (PDF, DOCX, TXT, CSV, Excel).
    Runs the full RAG pipeline and returns a finished answer with citations
    in the [filename § chunk] format.
    Use this for any question about the content of the documents.
    """
    return rag.answer(question)


@mcp.tool()
def search_documents(question: str, n_results: int = 5) -> str:
    """
    Return the raw document chunks most similar to the question, without
    generating an answer. Each chunk is tagged [filename § chunk].
    Use this when you need to inspect the source text itself.
    """
    q_emb = embedder.embed(question, input_type="query")
    hits = vector_store.query(q_emb, n=n_results)
    if not hits:
        return "No matching chunks found (no documents ingested yet)."
    return "\n\n---\n\n".join(
        f"[{h['source']} § {h['chunk']}]\n{h['text']}" for h in hits
    )


@mcp.tool()
def show_files() -> str:
    """
    List all ingested documents with their chunk counts.
    Use this when asked which files are available or already ingested.
    """
    sources = vector_store.all_sources()
    if not sources:
        return "No files have been ingested yet."
    return "\n".join(f"- {name}: {count} chunks" for name, count in sorted(sources.items()))


@mcp.tool()
def delete_document(source: str) -> str:
    """Delete all embedded chunks for one exact source filename."""
    if source not in vector_store.all_sources():
        return f"No embedded document named '{source}' was found."

    delete_source(source)
    return f"Deleted all embedded chunks for '{source}'."


if __name__ == "__main__":
    mcp.run()
