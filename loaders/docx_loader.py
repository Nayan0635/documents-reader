from docx import Document

def load(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)