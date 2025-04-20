from __future__ import annotations
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent, RunContext
from mcp_use import MCPClient  # <-- Added for direct MCP tool calls
import markdown  # <-- For Markdown to HTML conversion
import re  # <-- For stripping headings

def strip_leading_heading(text: str) -> str:
    """
    Remove leading Markdown or HTML headings from the blog content.
    """
    # Remove leading Markdown headings (e.g. # Title, ## Title)
    text = re.sub(r'^(#|##|###|####|#####|######)\s+.*\n+', '', text, count=1)
    # Remove leading HTML headings (e.g. <h1>Title</h1>)
    text = re.sub(r'^<h[1-6]>.*?</h[1-6]>\s*', '', text, count=1, flags=re.DOTALL)
    return text.lstrip()

load_dotenv()

def get_model() -> OpenAIModel:
    """
    Returns an OpenAIModel instance configured from environment variables.
    """
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# ========== Agent Army Context ==========
class AgentArmyContext:
    """
    Context object to hold shared agent/server state for the Agent Army system.
    """
    def __init__(self) -> None:
        self.brave_server: MCPServerHTTP | None = None
        self.ghost_server: MCPServerHTTP | None = None
        self.brave_agent: Agent | None = None
        self.ghost_agent: Agent | None = None
        self.primary_agent: Agent | None = None

# ========== Set up MCP servers for each service (SSE transport) ==========
# MCPServerHTTP instances will be created in main() and shared

# ========== MCP-USE integration ==========
brave_mcp_client = MCPClient.from_dict({
    "mcpServers": {
        "http": {"url": os.getenv('BRAVE_SSE_URL')}
    }
})
ghost_mcp_client = MCPClient.from_dict({
    "mcpServers": {
        "http": {"url": os.getenv('GHOST_SSE_URL')}
    }
})

# Async initialization of MCP sessions
def get_session(client: MCPClient, name: str = "http"):
    try:
        return client.get_session(name)
    except Exception:
        return None

async def initialize_mcp_sessions() -> None:
    """
    Asynchronously initialize sessions for Brave and Ghost MCP clients.
    """
    print("[DEBUG] Initializing Brave MCP session...")
    await brave_mcp_client.create_session("http", auto_initialize=True)
    print("[DEBUG] Brave MCP session initialized.")
    print("[DEBUG] Initializing Ghost MCP session...")
    await ghost_mcp_client.create_session("http", auto_initialize=True)
    print("[DEBUG] Ghost MCP session initialized.")

# List tools for a given client, using its session
def list_tools(client: MCPClient, name: str) -> None:
    """
    List available tools for a given MCP client and print their descriptions.
    """
    session = get_session(client, "http")
    if not session:
        print(f"No session found for {name} MCP client.")
        return
    print(f"\nAvailable tools on {name} MCP server:")
    for tool in session.tools:
        print(f"- {tool.name}: {getattr(tool, 'description', '')}")

# Call a tool on a given client, using its session
async def call_tool(client: MCPClient, name: str) -> None:
    """
    Prompt user for tool name and parameters, call the tool, and print the result.
    Retries parameter input if JSON is invalid.
    """
    session = get_session(client, "http")
    if not session:
        print(f"No session found for {name} MCP client.")
        return
    tool = input(f"Enter the tool name to call on {name}: ")
    import json
    while True:
        params = input("Enter parameters as JSON (e.g., {'query': 'Walmart'}): ")
        try:
            params_dict = json.loads(params)
            break
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}. Please try again.")
    print(f"Calling {tool} on {name} with params {params_dict}...")
    try:
        result = await session.call_tool(tool, params_dict)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error calling tool '{tool}' on {name}: {e}")

# ========== Main execution function ==========
import traceback
import http.client
from urllib.parse import urlparse

def check_mcp_server(url: str) -> None:
    """
    Check if the MCP server at the given URL is reachable and print the status.
    """
    try:
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
        conn.request("GET", parsed.path or "/")
        resp = conn.getresponse()
        print(f"[DEBUG] MCP server {url} responded with status {resp.status}")
        conn.close()
    except Exception as e:
        print(f"[DEBUG] MCP server {url} is not reachable: {e}")

async def main():
    print("MCP Agent Army - Multi-agent system using Model Context Protocol + SSE")
    print("Brave MCP server URL:", os.getenv('BRAVE_SSE_URL', 'http://192.168.31.135:8054/sse'))
    print("Ghost MCP server URL:", os.getenv('GHOST_SSE_URL', 'http://192.168.31.135:8053/sse'))

    context = AgentArmyContext()
    context.brave_server = MCPServerHTTP(url=os.getenv('BRAVE_SSE_URL'))
    context.ghost_server = MCPServerHTTP(url=os.getenv('GHOST_SSE_URL'))

    async with context.brave_server, context.ghost_server:
        check_mcp_server(os.getenv('BRAVE_SSE_URL', 'http://192.168.31.135:8054/sse'))
        check_mcp_server(os.getenv('GHOST_SSE_URL', 'http://192.168.31.135:8053/sse'))
        print("Enter 'exit' to quit the program.")

        await initialize_mcp_sessions()

        context.brave_agent = Agent(
            get_model(),
            system_prompt="You are a web search specialist using Brave Search. Use the Brave MCP server to find relevant information on the web.",
            mcp_servers=[context.brave_server]
        )
        context.ghost_agent = Agent(
            get_model(),
            system_prompt="You are a Ghost blog system specialist. Use the Ghost MCP server to create, edit, or manage blog posts.",
            mcp_servers=[context.ghost_server]
        )
        print("[DEBUG] Initializing primary orchestration agent...")

        from datetime import datetime, timedelta, timezone
        # Get current time in UTC+8
        utc8 = timezone(timedelta(hours=8))
        current_datetime = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
        system_prompt_template = (
            f"""
            Current date and time: {current_datetime}

            First, Search use brave agent to search {{topic_query}}

            Write a blog post that reads like a casual news article. Make it sound natural and friendly, like you're talking to a friend. Use simple English, and don’t make it feel too formal or robotic. Think of it like something someone would post on Blogspot or a personal blog.

            Please format the article nicely with:

            IMPORTANT: Do NOT include a title or heading in the article body. The blog title will be handled separately—start with the intro paragraph.

            A short intro paragraph to hook the reader

            Clear body paragraphs that explain the news (who, what, when, where, why, how)

            Use subheadings if it helps organize the info better

            Add quotes or opinions if relevant, to make it feel more human

            A conclusion paragraph to wrap things up, maybe share what might happen next or a final thought

            Make sure there are proper line breaks between paragraphs to keep it easy on the eyes. Keep it around 400–600 words max.

            And then second step call ghost agent to post public to Ghost blogspot and setup tags accordingly. Attach relevant picture link from the web search.
            """
        )
        context.primary_agent = Agent(
            get_model(),
            system_prompt=system_prompt_template.format(topic_query=""),
        )
        print("[DEBUG] Primary agent initialized.")

        # Helper to sanitize user input for prompt injection and curly braces
        def sanitize_user_input(user_input: str) -> str:
            """
            Escape curly braces and other problematic characters in user input to prevent prompt injection.
            """
            return user_input.replace('{', '{{').replace('}', '}}')

        # Tool functions now use context
        async def use_brave_search_agent(query: str) -> str:
            print(f"Calling Brave agent with query: {query}")
            result = await context.brave_agent.run(query)
            return result.text if hasattr(result, "text") else str(result)

        async def use_ghost_blog_agent(title: str, content: str, status: str = "published", tags: list = None, **kwargs) -> str:
            """
            Calls the Ghost MCP create_post tool directly with all required parameters.
            Enhanced with debug logging and exception handling.
            Converts Markdown content to HTML before publishing.
            """
            print(f"[DEBUG] Calling Ghost MCP create_post with title: {title}, status: {status}, tags: {tags}")
            # Strip any leading heading from the content
            cleaned_content = strip_leading_heading(content)
            # Convert Markdown to HTML
            html_content = markdown.markdown(cleaned_content)
            params = {
                "title": title,
                "content": html_content,
                "status": status,
                "tags": tags or []
            }
            session = get_session(ghost_mcp_client, "http")
            if not session:
                print("[ERROR] No Ghost MCP session available.")
                return "[ERROR] No Ghost MCP session available."
            try:
                result = await session.call_tool("create_post", params)
                print(f"[DEBUG] Ghost MCP create_post result: {result}")
                return str(result)
            except Exception as e:
                print(f"[ERROR] Exception in use_ghost_blog_agent: {e}")
                import traceback
                traceback.print_exc()
                return f"[ERROR] Exception in use_ghost_blog_agent: {e}"



        # Register tools with primary_agent after creation
        context.primary_agent.tool_plain(use_brave_search_agent)
        context.primary_agent.tool_plain(use_ghost_blog_agent)
        messages = []
        async with context.primary_agent.run_mcp_servers():
            while True:
                user_input = input("\n[You] ")
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("Goodbye!")
                    break
                if user_input.lower() == "list brave tools":
                    list_tools(brave_mcp_client, "Brave")
                    continue
                if user_input.lower() == "list ghost tools":
                    list_tools(ghost_mcp_client, "Ghost")
                    continue
                if user_input.lower() == "call brave tool":
                    await call_tool(brave_mcp_client, "Brave")
                    continue
                if user_input.lower() == "call ghost tool":
                    await call_tool(ghost_mcp_client, "Ghost")
                    continue
                try:
                    print("\n[Assistant]")
                    print("[DEBUG] Setting system prompt for this topic...")
                    safe_input = sanitize_user_input(user_input)
                    context.primary_agent.system_prompt = system_prompt_template.format(topic_query=safe_input)
                    print("[DEBUG] Calling primary_agent.run()...")
                    result = await context.primary_agent.run(user_input, message_history=messages)
                    print(f"[DEBUG] AgentRunResult: {result}")
                    if hasattr(result, "text"):
                        print(result.text)
                        article_content = result.text
                        # Use the user input as the blog title
                        article_title = user_input.strip().capitalize()
                        # Simple tag extraction: capitalize words longer than 3 chars
                        tags = [word.capitalize() for word in user_input.split() if len(word) > 3][:5]
                        print("[DEBUG] Calling Ghost agent to publish...")
                        ghost_result = await use_ghost_blog_agent(article_title, article_content, "published", tags)
                        print("[DEBUG] Ghost agent result:", ghost_result)
                    elif hasattr(result, "output"):
                        print(result.output)
                    elif hasattr(result, "data"):
                        print(result.data)
                    else:
                        print(result)
                    messages.extend(result.all_messages())
                except Exception as e:
                    print(f"\n[Error] An error occurred: {str(e)}")
                    traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Attempted to exit a cancel scope" in str(e):
            pass  # Suppress known anyio shutdown error
        else:
            raise
