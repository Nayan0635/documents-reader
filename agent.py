import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import Client, StdioServerParameters
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

server = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_ROOT / "mcp_server.py")],
    cwd=PROJECT_ROOT,
)

SYSTEM_PROMPT = (
    "You answer questions about the user's ingested documents using the tools provided. "
    "Prefer ask_documents for questions about document content, show_files for "
    "'what files do I have', and search_documents when you need raw source text. "
    "Base answers only on tool results and keep the [file § chunk] citations. "
    "If the tools return nothing relevant, say \"I don't know\"."
)

MAX_STEPS = 4  # safety cap on tool-call rounds per question


async def run():
    async with Client(server) as session:
        tools = await session.list_tools()
        print("Connected to MCP server. Available tools:")

        tool_list = []
        for tool in tools.tools:
            print(f"   - {tool.name}")
            tool_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            })

        print("\nType 'exit' to quit.")
        while True:
            user_input = input("\nAsk? ")
            if user_input.strip().lower() == "exit":
                print("Agent: Cya!")
                return

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]

            final = None
            for _ in range(MAX_STEPS):
                msg = llm.bind_tools(tool_list).invoke(messages)

                if not msg.tool_calls:
                    final = msg.text
                    break

                messages.append(msg)
                for call in msg.tool_calls:
                    print(f"   tool: {call['name']}({call['args']})")
                    result = await session.call_tool(call["name"], call["args"])
                    messages.append(ToolMessage(
                        content=result.content[0].text,
                        tool_call_id=call["id"],
                    ))

            print("\nAgent:")
            print(final or "I couldn't finish that request. Please try rephrasing.")


if __name__ == "__main__":
    asyncio.run(run())
