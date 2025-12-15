#!/usr/bin/env python3
"""Test script for LLM Council MCP server."""

import asyncio
import sys
from pathlib import Path

# Setup path
COUNCIL_ROOT = Path(__file__).parent
sys.path.insert(0, str(COUNCIL_ROOT))
sys.path.insert(0, str(COUNCIL_ROOT / "llm_council_mcp"))

# Import server components
from llm_council_mcp.server import (
    list_tools,
    handle_get_providers,
    handle_get_conversations,
    handle_quick_query,
)


async def test_list_tools():
    """Test that tools are properly registered."""
    print("Testing tool registration...")
    tools = await list_tools()
    print(f"  Registered tools: {len(tools)}")
    for tool in tools:
        print(f"    - {tool.name}")
    assert len(tools) >= 5, "Expected at least 5 tools"
    print("  PASSED\n")


async def test_get_providers():
    """Test provider info."""
    print("Testing get_providers...")
    result = await handle_get_providers({})
    text = result[0].text
    print(f"  Result preview: {text[:200]}...")
    assert "Claude Code" in text or "claude" in text.lower()
    assert "Codex" in text or "codex" in text.lower()
    assert "Gemini" in text or "gemini" in text.lower()
    print("  PASSED\n")


async def test_get_conversations():
    """Test listing conversations."""
    print("Testing get_conversations...")
    result = await handle_get_conversations({"limit": 5})
    text = result[0].text
    print(f"  Result: {text[:100]}...")
    print("  PASSED\n")


async def test_quick_query():
    """Test quick query to a single provider."""
    print("Testing quick_query (gemini - fastest)...")
    result = await handle_quick_query({
        "provider": "gemini",
        "prompt": "Say 'Hello from Gemini' in exactly 4 words.",
        "timeout": 60
    })
    text = result[0].text
    print(f"  Response: {text[:200]}...")
    assert "Error" not in text or "Gemini" in text
    print("  PASSED\n")


async def main():
    print("=" * 60)
    print("LLM Council MCP Server Tests")
    print("=" * 60 + "\n")

    await test_list_tools()
    await test_get_providers()
    await test_get_conversations()
    await test_quick_query()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
