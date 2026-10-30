import os
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loaders import load_file
from core import embedder, vector_store

DAILY_BUDGET = 900       # 1K requests/day quota
MAX_CHUNK = 6000         # Characters, roughly 1500 tokens for typical English text
CHUNK_OVERLAP = 600
TPM_LIMIT = 30_000


def build_chunks(text: str):
    """Use the largest configured chunks to minimize embedding requests."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def ingest(path: str):
    print(f"Loading {path} ...")
    text = load_file(path)
    if not text or not text.strip():
        raise ValueError("No text extracted from the file.")

    print(f"   Extracted {len(text):,} characters.")
    chunks = build_chunks(text)
    print(
        f"   Split into {len(chunks):,} chunks "
        f"(chunk_size={MAX_CHUNK}, overlap={CHUNK_OVERLAP})."
    )

    if len(chunks) > DAILY_BUDGET:
        raise ValueError(
            f"This file needs {len(chunks):,} embedding requests, exceeding the "
            f"configured daily budget of {DAILY_BUDGET:,}. No embeddings were requested "
            "and the vector store was not changed. Split the file or check your actual "
            "quota before changing DAILY_BUDGET."
        )

    print("   Embedding chunks ...")
    embeddings = []
    for i, chunk in enumerate(chunks, 1):
        embeddings.append(embedder.embed(chunk, input_type="passage"))

        # Approximate pacing only: character count is not an exact token count.
        estimated_tokens = len(chunk) / 4
        time.sleep(estimated_tokens / (TPM_LIMIT * 0.9) * 60)

        if i % 50 == 0 or i == len(chunks):
            print(f"   {i}/{len(chunks)}")

    source = os.path.basename(path)
    ids = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    # Replace existing chunks only after all embedding requests succeed.
    vector_store.delete_source(source)
    vector_store.upsert_chunks(ids, embeddings, chunks, metadatas)

    print(f"✅ Stored {len(chunks)} chunks from '{source}' in database.")


if __name__ == "__main__":
    while True:
        path = input("\nEnter file path (or 'exit' to quit): ").strip()
        if path.lower() == "exit":
            print("Bye 👋")
            break
        if not path:
            print("❌ No path given.")
            continue
        try:
            ingest(path)
        except Exception as exc:
            print(f"❌ Failed: {exc}")


# documents/data.txt
# documents/mission_Titan.pdf
# documents/retail_sales.xlsx
# documents/space_missions.csv
# documents/urban_gardening.docx
