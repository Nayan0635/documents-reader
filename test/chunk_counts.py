import math
import statistics
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_file

# --- keep these the same as in ingest.py ---
DAILY_BUDGET = 900      # embedding requests you allow per day (limit is 1K)
MIN_CHUNK = 1000
MAX_CHUNK = 6000        # ~1500 tokens, under the ~2048 token input limit
TPM_LIMIT = 30_000      # embedding tokens per minute
# -------------------------------------------


def build_chunks(text: str):
    size = max(MIN_CHUNK, round(len(text) / (DAILY_BUDGET * 0.65), -2))
    size = int(min(size, MAX_CHUNK))

    while True:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 10,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        if len(chunks) <= DAILY_BUDGET or size >= MAX_CHUNK:
            return chunks, size
        size = min(int(size * 1.25), MAX_CHUNK)


def check(path: str):
    print(f"Loading {path} ...")
    text = load_file(path)

    if not text or not text.strip():
        print("❌ No text extracted (scanned PDF? it needs OCR first).")
        return

    chunks, size = build_chunks(text)
    lengths = [len(c) for c in chunks]
    est_tokens = sum(lengths) / 4
    est_minutes = est_tokens / (TPM_LIMIT * 0.9)
    days = math.ceil(len(chunks) / DAILY_BUDGET)

    print(f"\nCharacters        : {len(text):,}")
    print(f"Chunk size/overlap: {size:,} / {size // 10:,}")
    print(f"Chunks (requests) : {len(chunks):,}  (daily budget {DAILY_BUDGET:,})")
    print(f"Chunk min/avg/max : {min(lengths):,} / {int(statistics.mean(lengths)):,} / {max(lengths):,}")
    print(f"Tiny chunks <100  : {sum(1 for n in lengths if n < 100)}")
    print(f"Est. tokens       : {int(est_tokens):,}")
    print(f"Est. embed time   : ~{est_minutes:.0f} min")

    if len(chunks) <= DAILY_BUDGET:
        print("\n✅ POSSIBLE: this file fits in one day's quota.")
    else:
        cap = DAILY_BUDGET * int(MAX_CHUNK * 0.65)
        print(f"\n❌ NOT POSSIBLE in one day ({len(chunks):,} requests needed).")
        print(f"   Would take ~{days} days, or split the file / clean the text.")
        print(f"   One-day capacity at max chunk size is about {cap:,} characters.")


if __name__ == "__main__":
    while True:
        path = input("\nEnter file path (or 'exit' to quit): ").strip()
        if path.lower() == "exit":
            print("Bye 👋")
            break
        if not path:
            print("No path given.")
            continue
        try:
            check(path)
        except Exception as e:
            print(f"❌ Failed: {e}")