from core import embedder, llm, vector_store

def answer(question: str, n_results: int = 5) -> str:
    q_emb = embedder.embed(question)
    hits = vector_store.query(q_emb, n=n_results)

    if not hits:
        return "I don't know (no documents ingested yet)."

    # Build context with citation tags the LLM can reuse
    context = "\n\n---\n\n".join(
        f"[{h['source']} § {h['chunk']}]\n{h['text']}"
        for h in hits
    )

    return llm.ask(context, question)