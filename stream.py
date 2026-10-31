import asyncio
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import Client

import math
import statistics


from test.chunk_counts import DAILY_BUDGET, MAX_CHUNK, TPM_LIMIT, build_chunks
from _ingest import ingest
from agent import MAX_STEPS, SYSTEM_PROMPT, llm, server
from loaders import load_file
from core import vector_store


st.set_page_config(
    page_title="Documents Reader",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "csv", "xlsx", "xls"]


def initialize_state():
    defaults = {
        "chat_history": [],
        "agent_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inspect_upload(uploaded_file):
    """Extract and chunk a file without making embedding API requests."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {f".{ext}" for ext in SUPPORTED_TYPES}:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / Path(uploaded_file.name).name
        file_path.write_bytes(uploaded_file.getvalue())
        text = load_file(str(file_path))

    if not text or not text.strip():
        raise ValueError("No text could be extracted from this file.")

    chunks, chunk_size = build_chunks(text)
    if not chunks:
        raise ValueError("The file did not produce any text chunks.")

    lengths = [len(c) for c in chunks]
    est_tokens = sum(lengths) / 4
    est_minutes = est_tokens / (TPM_LIMIT * 0.9)
    days = math.ceil(len(chunks) / DAILY_BUDGET)

    return {
        "characters": len(text),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_size // 10,
        "requests": len(chunks),
        "min_chunk": min(lengths),
        "avg_chunk": int(statistics.mean(lengths)),
        "max_chunk": max(lengths),
        "tiny_chunks": sum(1 for n in lengths if n < 100),
        "est_tokens": int(est_tokens),
        "est_minutes": est_minutes,
        "days": days,
        "feasible": len(chunks) <= DAILY_BUDGET,
    }


def embed_upload(uploaded_file):
    """Pass an uploaded file to the existing ingestion pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / Path(uploaded_file.name).name
        file_path.write_bytes(uploaded_file.getvalue())
        ingest(str(file_path))


async def ask_agent(question):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *st.session_state.agent_messages,
        HumanMessage(content=question),
    ]
    tool_logs = []
    final_answer = None

    async with Client(server) as session:
        available_tools = await session.list_tools()
        tool_list = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in available_tools.tools
        ]
        agent = llm.bind_tools(tool_list)

        for _ in range(MAX_STEPS):
            response = agent.invoke(messages)

            if not response.tool_calls:
                final_answer = response.text
                break

            messages.append(response)

            for call in response.tool_calls:
                tool_logs.append(f"{call['name']}({call['args']})")
                result = await session.call_tool(call["name"], call["args"])

                tool_output = "\n".join(
                    block.text
                    for block in result.content
                    if hasattr(block, "text")
                )
                if not tool_output:
                    tool_output = "The tool returned no text."

                messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=call["id"],
                    )
                )

    return (
        final_answer or "I couldn't finish that request. Please try again.",
        messages[1:],
        tool_logs,
    )


initialize_state()

with st.sidebar:
    st.title("📚 Documents Reader")
    st.caption("Upload, embed and manage your documents.")

    st.subheader("File upload")
    uploaded_files = st.file_uploader(
        "Choose documents from your computer",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, CSV, XLSX and XLS.",
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")

        # Duplicate basenames would map to the same Chroma source and IDs.
        names = [Path(file.name).name for file in uploaded_files]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})

        if duplicate_names:
            st.error(
                "Two or more selected files have the same filename: "
                + ", ".join(duplicate_names)
                + ". Rename one before uploading."
            )
            st.session_state.pop("upload_check", None)
        else:
            if st.button("Check embedding capacity", use_container_width=True):
                checks = {}
                total_chunks = 0

                with st.spinner("Checking files and estimating chunks..."):
                    for file in uploaded_files:
                        try:
                            stats = inspect_upload(file)
                            checks[file.name] = {
                                "stats": stats,
                                "error": None,
                            }
                            total_chunks += stats["requests"]
                        except Exception as exc:
                            checks[file.name] = {
                                "stats": None,
                                "error": str(exc),
                            }

                st.session_state.upload_check = {
                    "checks": checks,
                    "total_chunks": total_chunks,
                    "file_names": names,
                }

            check = st.session_state.get("upload_check")
            if check and check["file_names"] == names:
                st.markdown("**Preflight results**")
                all_valid = True

                for filename, result in check["checks"].items():
                    if result["error"]:
                        all_valid = False
                        st.error(f"{filename}: {result['error']}")
                    else:
                        stats = result["stats"]
                        st.markdown(f"**{filename}**")
                        status_text = (
                            "✅ POSSIBLE: this file fits in one day's quota."
                            if stats["feasible"]
                            else f"❌ NOT POSSIBLE in one day ({stats['requests']:,} requests needed)."
                        )
                        st.text(
                            f"Characters        : {stats['characters']:,}\n"
                            f"Chunks (requests) : {stats['requests']:,}  (daily budget {DAILY_BUDGET:,})\n"
                            f"{status_text}"
                        )

                total = check["total_chunks"]
                st.caption(
                    f"Estimated embedding requests for this batch: "
                    f"{total:,} / {DAILY_BUDGET:,}"
                )

                within_configured_limit = total <= DAILY_BUDGET
                if all_valid and within_configured_limit:
                    st.success(
                        "Files passed preflight and fit within the configured "
                        "batch budget."
                    )
                    if st.button(
                        "▶ Start Embedding",
                        type="primary",
                        use_container_width=True,
                    ):
                        successful = []
                        failures = []

                        progress = st.progress(0)
                        status = st.status(
                            "Embedding selected files...",
                            expanded=True,
                        )

                        for index, file in enumerate(uploaded_files, start=1):
                            status.write(f"Processing {file.name}...")
                            try:
                                embed_upload(file)
                                successful.append(file.name)
                                status.write(f"Completed: {file.name}")
                            except Exception as exc:
                                failures.append((file.name, str(exc)))
                                status.write(f"Failed: {file.name}: {exc}")

                            progress.progress(index / len(uploaded_files))

                        status.update(
                            label="Embedding process finished",
                            state="complete" if not failures else "error",
                        )

                        if successful:
                            st.toast(
                                f"Embedded {len(successful)} file(s) successfully.",
                                icon="✅",
                            )
                        if failures:
                            for filename, error in failures:
                                st.error(f"{filename}: {error}")
                        else:
                            st.success("All selected files were embedded.")

                        st.session_state.pop("upload_check", None)
                        st.rerun()
                elif all_valid:
                    st.error(
                        f"This batch needs {total:,} chunks, exceeding the "
                        f"configured batch budget of {DAILY_BUDGET:,}."
                    )
                else:
                    st.warning("Fix the file errors before starting embedding.")

    st.divider()
    st.subheader("Embedded files")

    sources = vector_store.all_sources()
    if sources:
        st.caption(f"{len(sources)} file(s) in the vector database")

        for filename, chunk_count in sorted(sources.items()):
            file_col, delete_col = st.columns([5, 1])
            with file_col:
                st.write(f"📄 {filename}")
                st.caption(f"{chunk_count:,} chunks")
            with delete_col:
                if st.button(
                    "✕",
                    key=f"delete_{filename}",
                    help=f"Delete embeddings for {filename}",
                ):
                    vector_store.delete_source(filename)
                    st.toast(f"Deleted embeddings for {filename}.", icon="🗑️")
                    st.rerun()
    else:
        st.info("No files have been embedded yet.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_messages = []
        st.rerun()

    st.caption("Powered by Gemini · MCP · ChromaDB")


st.title("Document Analysis Agent")
st.caption("Ask questions grounded in the documents you have embedded.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                answer, agent_messages, tool_logs = asyncio.run(
                    ask_agent(question)
                )

            if tool_logs:
                with st.expander("MCP tools used"):
                    for log in tool_logs:
                        st.code(log)

            st.markdown(answer)
            st.session_state.agent_messages = agent_messages
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            st.error(f"Agent error: {exc}")
