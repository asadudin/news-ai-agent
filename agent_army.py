from __future__ import annotations
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent
from mcp_use import MCPClient
import markdown
import re
import uvicorn
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_army")

# --- Centralized Configuration ---
def load_config() -> Dict[str, str]:
    """
    Load and return configuration from environment variables.
    """
    load_dotenv()
    return {
        'MODEL_CHOICE': os.getenv('MODEL_CHOICE'),
        'BASE_URL': os.getenv('BASE_URL'),
        'LLM_API_KEY': os.getenv('LLM_API_KEY'),
        'BRAVE_SSE_URL': os.getenv('BRAVE_SSE_URL'),
        'GHOST_SSE_URL': os.getenv('GHOST_SSE_URL'),
    }

CONFIG = load_config()

def strip_leading_heading(text: str) -> str:
    """
    Remove leading Markdown or HTML headings from the blog content.
    """
    text = re.sub(r'^(#|##|###|####|#####|######)\s+.*\n+', '', text, count=1)
    text = re.sub(r'^<h[1-6]>.*?</h[1-6]>\s*', '', text, count=1, flags=re.DOTALL)
    return text.lstrip()

def get_model() -> OpenAIModel:
    """
    Returns an OpenAIModel instance configured from CONFIG.
    """
    return OpenAIModel(
        CONFIG['MODEL_CHOICE'],
        provider=OpenAIProvider(base_url=CONFIG['BASE_URL'], api_key=CONFIG['LLM_API_KEY'])
    )

def get_session(client: MCPClient, name: str = "http") -> Optional[Any]:
    """
    Safely get a session from the MCP client.
    """
    try:
        return client.get_session(name)
    except Exception as e:
        logger.error(f"Failed to get session for {name}: {e}")
        return None

# ========== Agent Army Context ==========
class AgentArmyContext:
    """
    Context object to hold shared agent/server state for the Agent Army system.
    """
    def __init__(self) -> None:
        self.brave_server: Optional[MCPServerHTTP] = None
        self.ghost_server: Optional[MCPServerHTTP] = None
        self.brave_agent: Optional[Agent] = None
        self.ghost_agent: Optional[Agent] = None
        self.primary_agent: Optional[Agent] = None

# ========== FastAPI App ==========
app = FastAPI()

class UserInputRequest(BaseModel):
    user_input: str

async def agent_army_respond(user_input: str, context=None, messages=None):
    """
    Process user input and generate a response using the primary agent.
    Used by the FastAPI endpoint.
    """
    if context is None:
        context = AgentArmyContext()
    if messages is None:
        messages = []
        
    # Sanitize input to prevent prompt injection
    def sanitize_user_input(user_input: str) -> str:
        return user_input.replace('{', '{{').replace('}', '}}')
    
    safe_input = sanitize_user_input(user_input)
    
    # Get current date/time for system prompt
    utc8 = timezone(timedelta(hours=8))
    current_datetime = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    
    # Use the template from app.state and update with current time and user query
    template = app.state.system_prompt_template.replace("{current_datetime}", current_datetime)
    
    # Update the system prompt with the user's query
    context.primary_agent.system_prompt = template.format(topic_query=safe_input)
    
    # Run the agent with the user input
    result = await context.primary_agent.run(user_input, message_history=messages)
    
    # Return the result text or string representation
    return result.text if hasattr(result, "text") else str(result)

@app.on_event("startup")
async def startup_event():
    """
    Initialize MCP clients, sessions, and agents for FastAPI.
    Store them in app.state for safe reuse across requests.
    """
    logger.info("[FastAPI] Initializing MCP clients and sessions on startup...")
    # MCP Clients
    app.state.brave_mcp_client = MCPClient.from_dict({
        "mcpServers": {"http": {"url": CONFIG['BRAVE_SSE_URL']}}
    })
    app.state.ghost_mcp_client = MCPClient.from_dict({
        "mcpServers": {"http": {"url": CONFIG['GHOST_SSE_URL']}}
    })
    # Sessions
    await app.state.brave_mcp_client.create_session("http", auto_initialize=True)
    await app.state.ghost_mcp_client.create_session("http", auto_initialize=True)
    logger.info("[FastAPI] MCP sessions initialized.")
    # Create and store MCP servers
    app.state.brave_server = MCPServerHTTP(url=CONFIG['BRAVE_SSE_URL'])
    app.state.ghost_server = MCPServerHTTP(url=CONFIG['GHOST_SSE_URL'])
    # Create agents with the MCP servers
    brave_agent = Agent(
        get_model(),
        system_prompt="You are a web search specialist using Brave Search. Use the Brave MCP server to find relevant information on the web.",
        mcp_servers=[app.state.brave_server]
    )
    ghost_agent = Agent(
        get_model(),
        system_prompt="You are a Ghost blog system specialist. Use the Ghost MCP server to create, edit, or manage blog posts.",
        mcp_servers=[app.state.ghost_server]
    )
    # Orchestration agent
    # Store the system prompt template in app.state for use by agent_army_respond
    # Use {current_datetime} as a placeholder to be replaced at runtime
    app.state.system_prompt_template = """
        Current date and time: {current_datetime}
        First, Search use brave agent to search {topic_query}
        Write a blog post that reads like a casual news article. Make it sound natural and friendly, like you're talking to a friend. Use simple English, and don't make it feel too formal or robotic. Think of it like something someone would post on Blogspot or a personal blog.
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
    
    # Get current time for initial agent setup
    utc8 = timezone(timedelta(hours=8))
    current_datetime = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    
    # Initialize the primary agent with the current time
    initial_prompt = app.state.system_prompt_template.replace("{current_datetime}", current_datetime).format(topic_query="")
    primary_agent = Agent(
        get_model(),
        system_prompt=initial_prompt,
    )
    # Register tools
    async def brave_search_tool(query: str):
        return await brave_agent.run(query)
    
    async def ghost_blog_tool(title: str, content: str, status: str = "published", tags: list = None, **kwargs):
        """Direct MCP tool call implementation for Ghost blog posting."""
        logger.info(f"[API] Calling Ghost MCP create_post with title: {title}, status: {status}, tags: {tags}")
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
        session = get_session(app.state.ghost_mcp_client, "http")
        if not session:
            logger.error("[API] No Ghost MCP session available.")
            return "[ERROR] No Ghost MCP session available."
        try:
            result = await session.call_tool("create_post", params)
            logger.info(f"[API] Ghost MCP create_post result: {result}")
            return str(result)
        except Exception as e:
            logger.error(f"[API] Exception in ghost_blog_tool: {e}", exc_info=True)
            return f"[ERROR] Exception in ghost_blog_tool: {e}"
            
    primary_agent.tool_plain(brave_search_tool)
    primary_agent.tool_plain(ghost_blog_tool)
    # Store in app.state
    app.state.brave_agent = brave_agent
    app.state.ghost_agent = ghost_agent
    app.state.primary_agent = primary_agent
    logger.info("[FastAPI] Agents initialized and registered.")

@app.post("/user_input")
async def user_input_endpoint(request: UserInputRequest):
    """
    FastAPI endpoint for user input. Returns structured error on failure.
    Uses app.state for MCP/agent context.
    """
    try:
        # Use agents and servers from app.state
        context = AgentArmyContext()
        context.brave_server = app.state.brave_server
        context.ghost_server = app.state.ghost_server
        context.brave_agent = app.state.brave_agent
        context.ghost_agent = app.state.ghost_agent
        context.primary_agent = app.state.primary_agent
        
        # Critical: Use BOTH context managers with proper error handling
        try:
            # First activate the raw MCP servers
            async with context.brave_server, context.ghost_server:
                # Then run the agent's MCP servers within that context
                async with context.primary_agent.run_mcp_servers():
                    response = await agent_army_respond(request.user_input, context=context)
            
            return JSONResponse(content={"response": response})
        except Exception as e:
            logger.error(f"MCP server error: {e}", exc_info=True)
            # Attempt to reconnect MCP sessions on failure
            try:
                logger.info("Attempting to reconnect MCP sessions...")
                await app.state.brave_mcp_client.create_session("http", auto_initialize=True)
                await app.state.ghost_mcp_client.create_session("http", auto_initialize=True)
                logger.info("MCP sessions reconnected successfully.")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect MCP sessions: {reconnect_error}")
            
            return JSONResponse(
                content={
                    "error": {
                        "type": type(e).__name__, 
                        "message": f"MCP server error: {str(e)}"
                    }
                }, 
                status_code=500
            )
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return JSONResponse(content={"error": {"type": type(e).__name__, "message": str(e)}}, status_code=500)

def run_app():
    """
    Entrypoint for running the FastAPI app.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("agent_army:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    run_app()