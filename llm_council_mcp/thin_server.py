"""Thin MCP server that proxies to HTTP backend.

This is a stable MCP wrapper that:
1. Receives tool calls from Claude Code
2. Makes HTTP requests to the backend (fast, non-blocking)
3. Returns responses

The actual heavy lifting (CLI provider calls) happens in the HTTP backend,
keeping the MCP stdio connection stable.

Usage:
1. Start HTTP backend: cd llm-council && python -m uvicorn backend.main:app --port 8001
2. Configure MCP to use this server (thin_server.py)
"""

import asyncio
import httpx
from typing import Dict, Any, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# HTTP backend URL (main.py runs on port 8001)
BACKEND_URL = "http://127.0.0.1:8001"

# Initialize MCP server
server = Server("llm-council")

# HTTP client with reasonable timeout for proxying
http_client = httpx.AsyncClient(timeout=300.0)  # 5 min max for long operations


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available council tools."""
    return [
        Tool(
            name="council_deliberate",
            description="Run full 3-stage council deliberation on a question",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to deliberate"},
                    "save": {"type": "boolean", "description": "Save conversation", "default": True}
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="council_quick_query",
            description="Query a single provider for a fast response",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["claude", "ollama", "gemini", "llama_server"]},
                    "prompt": {"type": "string", "description": "Prompt to send"},
                    "timeout": {"type": "number", "description": "Timeout in seconds", "default": 60}
                },
                "required": ["provider", "prompt"]
            }
        ),
        Tool(
            name="council_get_providers",
            description="List available LLM providers",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="council_list_patterns",
            description="List all available deliberation patterns",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="council_run_pattern",
            description="Run a specific deliberation pattern (debate, socratic, red_team, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["deliberation", "debate", "devils_advocate", "socratic",
                                "red_team", "tree_of_thought", "self_consistency",
                                "round_robin", "expert_panel"]
                    },
                    "question": {"type": "string"},
                    "rounds": {"type": "integer", "default": 2},
                    "branches": {"type": "integer", "default": 3}
                },
                "required": ["pattern", "question"]
            }
        ),
        Tool(
            name="council_compare_providers",
            description="Compare all providers on the same prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                },
                "required": ["prompt"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls by proxying to HTTP backend."""
    try:
        if name == "council_get_providers":
            resp = await http_client.get(f"{BACKEND_URL}/api/mcp/providers")
            data = resp.json()
            return [TextContent(type="text", text=format_providers(data))]

        elif name == "council_quick_query":
            resp = await http_client.post(f"{BACKEND_URL}/api/mcp/query", json=arguments)
            data = resp.json()
            if data.get("success"):
                return [TextContent(type="text", text=data["content"])]
            else:
                return [TextContent(type="text", text=f"Error: {data.get('error', 'Unknown error')}")]

        elif name == "council_deliberate":
            resp = await http_client.post(f"{BACKEND_URL}/api/mcp/deliberate", json=arguments)
            data = resp.json()
            if data.get("success"):
                return [TextContent(type="text", text=format_deliberation(data["result"]))]
            else:
                return [TextContent(type="text", text=f"Error: {data.get('error', 'Unknown error')}")]

        elif name == "council_list_patterns":
            resp = await http_client.get(f"{BACKEND_URL}/api/mcp/patterns")
            data = resp.json()
            return [TextContent(type="text", text=format_patterns(data["patterns"]))]

        elif name == "council_run_pattern":
            resp = await http_client.post(f"{BACKEND_URL}/api/mcp/patterns/run", json=arguments)
            data = resp.json()
            if data.get("success"):
                return [TextContent(type="text", text=format_pattern_result(arguments["pattern"], data["result"]))]
            else:
                return [TextContent(type="text", text=f"Error: {data.get('error', 'Unknown error')}")]

        elif name == "council_compare_providers":
            resp = await http_client.post(f"{BACKEND_URL}/api/mcp/compare", json=arguments)
            data = resp.json()
            return [TextContent(type="text", text=format_comparison(data["responses"]))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except httpx.ConnectError:
        return [TextContent(type="text", text="Error: Council HTTP backend not running. Start it with: cd llm-council && python -m uvicorn backend.main:app --port 8001")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def format_providers(data: Dict) -> str:
    """Format providers response."""
    lines = [
        f"**LLM Council Providers**\n",
        f"Mode: {data['mode']}",
        f"Council Members: {data['council_members']}",
        f"Chairman: {data['chairman']}\n",
        "**Available Providers:**"
    ]
    for p in data["providers"]:
        chairman = " Chairman" if p["is_chairman"] else ""
        lines.append(f"  - {p['display_name']} [Council Member{chairman}]")
    return "\n".join(lines)


def format_patterns(patterns: List[Dict]) -> str:
    """Format patterns list."""
    lines = ["**Available Deliberation Patterns**\n"]
    for p in patterns:
        lines.append(f"### {p['name']}")
        lines.append(f"**ID:** `{p['id']}`")
        lines.append(f"**Description:** {p['description']}")
        lines.append(f"**Best For:** {', '.join(p['best_for'])}")
        lines.append(f"**Flow:** {p['flow']}")
        lines.append("")
    return "\n".join(lines)


def format_deliberation(result: Dict) -> str:
    """Format deliberation result."""
    lines = ["=" * 60, "COUNCIL DELIBERATION", "=" * 60, ""]

    # Stage 1: Initial responses
    if "initial_responses" in result:
        lines.append("**Stage 1: Initial Responses**")
        lines.append("-" * 40)
        for resp in result["initial_responses"]:
            lines.append(f"\n### {resp.get('model', 'Unknown')}")
            lines.append(resp.get("content", "No response"))
        lines.append("")

    # Stage 2: Rankings
    if "rankings" in result:
        lines.append("**Stage 2: Rankings**")
        lines.append("-" * 40)
        for rank in result["rankings"]:
            lines.append(f"\n### {rank.get('ranker', 'Unknown')}")
            lines.append(rank.get("ranking", "No ranking"))
        lines.append("")

    # Stage 3: Synthesis
    if "synthesis" in result:
        lines.append("**Stage 3: Chairman's Synthesis**")
        lines.append("-" * 40)
        lines.append(result["synthesis"])

    return "\n".join(lines)


def format_pattern_result(pattern: str, result: Dict) -> str:
    """Format pattern execution result."""
    lines = ["=" * 60, f"PATTERN: {pattern.upper()}", "=" * 60, ""]

    for stage in result.get("stages", []):
        lines.append("-" * 40)
        lines.append(f"**{stage.get('name', 'Stage')}**")
        lines.append("-" * 40)

        if "responses" in stage:
            for resp in stage["responses"]:
                role = resp.get("role", resp.get("model", "Unknown"))
                content = resp.get("response", resp.get("content", "No content"))
                lines.append(f"\n### {role}")
                lines.append(content)
        elif "response" in stage:
            lines.append(stage["response"])
        elif "content" in stage:
            lines.append(stage["content"])
        lines.append("")

    if result.get("final_answer"):
        lines.append("-" * 40)
        lines.append("**FINAL ANSWER**")
        lines.append("-" * 40)
        lines.append(result["final_answer"])

    return "\n".join(lines)


def format_comparison(responses: List[Dict]) -> str:
    """Format provider comparison."""
    lines = ["**Provider Comparison**\n"]
    for resp in responses:
        status = "✓" if resp["success"] else "✗"
        lines.append(f"### {status} {resp['display_name']}")
        if resp["success"]:
            lines.append(resp["content"])
        else:
            lines.append("*No response*")
        lines.append("")
    return "\n".join(lines)


async def main():
    """Run the thin MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
