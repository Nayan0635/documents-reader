"""List every file embedded in the local ChromaDB collection."""
from pathlib import Path
import chromadb

DATABASE_PATH = Path(__file__).resolve().parent.parent / "database"
COLLECTION_NAME = "documents"

def main():
    collection = chromadb.PersistentClient(path=str(DATABASE_PATH)).get_collection(COLLECTION_NAME)
    result = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for metadata in result["metadatas"]:
        source = metadata.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    if not sources:
        print("📭 No files ingested yet. Run 'python _ingest.py' first.")
        return

    print(f"\n📁 ChromaDB collection '{collection.name}'")
    print("-" * 55)
    total = 0
    for name, count in sorted(sources.items()):
        print(f"   {name:<35} → {count:>3} chunks")
        total += count
    print("-" * 55)
    print(f"   {len(sources)} files, {total} total chunks\n")


if __name__ == "__main__":
    main()
