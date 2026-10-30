from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

def ask(context: str, question: str) -> str:
    prompt = f"""You are answering questions using ONLY the context below.

    Context (each block is tagged with its source, e.g. [file.pdf § 3]): {context}
    Question: {question}

    Rules:
    - Answer using only the context above.
    - After each fact, cite the block(s) it came from using the same tag format: [file.pdf § 3].
    - If two files both contribute, cite both.
    - If the answer is not present in the context, say "I don't know".
    """

    response = _llm.invoke(prompt)
    return response.text
