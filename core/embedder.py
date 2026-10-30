from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def embed(text: str, input_type: str = "document") -> list[float]:
    """Embed a document passage or a retrieval query."""
    if input_type == "query":
        return _embeddings.embed_query(text)
    if input_type == "document":
        return _embeddings.embed_documents([text])[0]
    raise ValueError("input_type must be 'document' or 'query'")
