# Documents Reader

A Python-based document Q&A application using **RAG (Retrieval-Augmented Generation), Gemini, ChromaDB, LangChain, and MCP**.

It extracts text from different document formats, generates embeddings, stores them in ChromaDB, and answers questions using an LLM.

## Supported Formats

- PDF
- DOCX
- TXT
- CSV
- Excel (XLSX, XLS)

## Tech Stack

- **Python**: Application development
- **LangChain**: Document processing and LLM integration
- **Gemini**: Embeddings and chat generation
- **ChromaDB**: Vector database
- **MCP**: Exposing document operations as tools
- **uv**: Dependency and environment management

## Setup

Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

Clone the repository:

```bash
git clone https://github.com/Syris695/documents-reader.git
cd documents-reader
```

Install dependencies:

```bash
uv sync
uv pip install langchain-google-genai
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key
```

## Usage

**1. Ingest documents** Place your documents inside the `documents/` directory.

```bash
uv run python _ingest.py
```

Enter the document path when prompted.

**2. Check chunk counts** :

```bash
uv run python test/chunk_counts.py
```

**3. List ingested files** :

```bash
uv run python test/show_files.py
```

**4. Start the AI agent** :

```bash
uv run python agent.py
```

Ask questions about your ingested documents through the interactive agent.

## MCP Tools

The MCP server exposes four tools:

| Tool | Description |
|---|---|
| `ask_documents` | Answer questions using RAG |
| `search_documents` | Retrieve relevant document chunks |
| `show_files` | List ingested documents |
| `delete_document` | Delete an embedded document |

## Project Structure

```text
documents-reader/
├── core/
│   ├── embedder.py
│   ├── llm.py
│   ├── rag.py
│   └── vector_store.py
├── loaders/
│   ├── txt_loader.py
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── csv_loader.py
│   └── excel_loader.py
├── test/
├── documents/
├── database/
├── _ingest.py
├── mcp_server.py
├── agent.py
├── pyproject.toml
└── uv.lock
```

**Note:** Documents and ChromaDB data are stored locally. Keep your API key private and never commit `.env`.
