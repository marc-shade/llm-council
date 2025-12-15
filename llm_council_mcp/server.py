#!/usr/bin/env python3
"""
LLM Council MCP Server

Provides MCP interface to the LLM Council deliberation system.
Enables Claude Code to invoke multi-LLM council deliberations directly from CLI.

Tools:
- council_deliberate: Run full 3-stage council deliberation
- council_run_pattern: Run any deliberation pattern (debate, socratic, red_team, etc.)
- council_list_patterns: List available deliberation patterns
- council_quick_query: Query a single provider for fast responses
- council_get_providers: List available LLM providers
- council_get_conversations: List past deliberations
- council_get_conversation: Get specific conversation details
- council_compare_providers: Compare all providers on same prompt
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
COUNCIL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(COUNCIL_ROOT))

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Council imports
from backend.council import (
    run_full_council,
    stage1_collect_responses,
    generate_conversation_title,
)
from backend.cli_providers import (
    query_cli_provider,
    get_available_providers,
    get_provider_info,
    PROVIDERS,
)
from backend.storage import (
    list_conversations,
    get_conversation,
    create_conversation,
    add_user_message,
    add_assistant_message,
    update_conversation_title,
)
from backend.config import (
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    PROVIDER_MODE,
    PROVIDER_DISPLAY_NAMES,
)
from backend.patterns import (
    Pattern,
    PATTERN_INFO,
    list_patterns,
    run_pattern,
)


# Initialize MCP server
server = Server("llm-council")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available LLM Council tools."""
    return [
        Tool(
            name="council_deliberate",
            description="""Run a full 3-stage LLM Council deliberation on a question.

Stage 1: All council members (Claude, Codex, Gemini) provide individual responses
Stage 2: Each member ranks the anonymized responses from other members
Stage 3: Chairman synthesizes a final answer based on all inputs

Returns comprehensive results including individual responses, rankings, and final synthesis.
Use this for important questions that benefit from multiple perspectives.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to deliberate on"
                    },
                    "save_conversation": {
                        "type": "boolean",
                        "description": "Whether to save this deliberation for future reference (default: true)",
                        "default": True
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="council_quick_query",
            description="""Query a single LLM provider for a fast response.

Use this for quick questions that don't need full council deliberation.
Available providers: claude, codex, gemini""",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to query (claude, codex, or gemini)",
                        "enum": ["claude", "codex", "gemini"]
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 120)",
                        "default": 120
                    }
                },
                "required": ["provider", "prompt"]
            }
        ),
        Tool(
            name="council_get_providers",
            description="""Get information about available LLM providers in the council.

Returns details about each provider including name, display name, and current status.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="council_get_conversations",
            description="""List past council deliberations.

Returns metadata for all saved conversations including ID, title, creation date, and message count.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of conversations to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="council_get_conversation",
            description="""Get details of a specific council deliberation.

Returns the full conversation including all stages of each deliberation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "ID of the conversation to retrieve"
                    }
                },
                "required": ["conversation_id"]
            }
        ),
        Tool(
            name="council_compare_providers",
            description="""Query all providers with the same prompt and compare responses.

Useful for understanding how different models approach the same question.
Does NOT include ranking or synthesis - just raw parallel responses.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to all providers"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="council_list_patterns",
            description="""List all available deliberation patterns.

Each pattern represents a different multi-mind collaboration strategy:
- deliberation: Default 3-stage (respond → rank → synthesize)
- debate: Pro vs Con with judge
- devils_advocate: Answer challenged by critic, then refined
- socratic: Question-driven dialogue for deeper understanding
- red_team: Blue proposes, red attacks, iterate to robust solution
- tree_of_thought: Explore branches, prune, find best path
- self_consistency: Multiple samples with majority voting
- round_robin: Sequential building on previous responses
- expert_panel: Route to domain specialists""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="council_run_pattern",
            description="""Run a specific deliberation pattern on a question.

Different patterns suit different use cases:
- debate: Best for controversial topics, pros/cons analysis
- devils_advocate: Best for stress-testing ideas, finding weaknesses
- socratic: Best for complex reasoning, philosophical questions
- red_team: Best for security analysis, robustness testing
- tree_of_thought: Best for problem solving, strategic planning
- self_consistency: Best for high-confidence factual answers
- round_robin: Best for creative tasks, brainstorming
- expert_panel: Best for domain-specific technical questions

Use council_list_patterns to see all options with details.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to use",
                        "enum": ["deliberation", "debate", "devils_advocate", "socratic",
                                "red_team", "tree_of_thought", "self_consistency",
                                "round_robin", "expert_panel"]
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to deliberate on"
                    },
                    "rounds": {
                        "type": "integer",
                        "description": "Number of rounds for patterns that support it (socratic: 2, etc.)",
                        "default": 2
                    },
                    "branches": {
                        "type": "integer",
                        "description": "Number of branches for tree_of_thought pattern",
                        "default": 3
                    }
                },
                "required": ["pattern", "question"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""

    if name == "council_deliberate":
        return await handle_deliberate(arguments)
    elif name == "council_quick_query":
        return await handle_quick_query(arguments)
    elif name == "council_get_providers":
        return await handle_get_providers(arguments)
    elif name == "council_get_conversations":
        return await handle_get_conversations(arguments)
    elif name == "council_get_conversation":
        return await handle_get_conversation(arguments)
    elif name == "council_compare_providers":
        return await handle_compare_providers(arguments)
    elif name == "council_list_patterns":
        return await handle_list_patterns(arguments)
    elif name == "council_run_pattern":
        return await handle_run_pattern(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_deliberate(args: Dict[str, Any]) -> List[TextContent]:
    """Run full council deliberation."""
    question = args.get("question", "")
    save = args.get("save_conversation", True)

    if not question:
        return [TextContent(type="text", text="Error: Question is required")]

    try:
        # Run full council
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(question)

        # Format output
        output_parts = []

        # Header
        output_parts.append("=" * 60)
        output_parts.append("LLM COUNCIL DELIBERATION")
        output_parts.append("=" * 60)
        output_parts.append(f"\nQuestion: {question}\n")
        output_parts.append(f"Council Members: {', '.join(PROVIDER_DISPLAY_NAMES.get(m, m) for m in COUNCIL_MODELS)}")
        output_parts.append(f"Chairman: {PROVIDER_DISPLAY_NAMES.get(CHAIRMAN_MODEL, CHAIRMAN_MODEL)}\n")

        # Stage 1: Individual Responses
        output_parts.append("-" * 60)
        output_parts.append("STAGE 1: Individual Responses")
        output_parts.append("-" * 60)
        for result in stage1_results:
            output_parts.append(f"\n### {result.get('display_name', result['model'])}")
            output_parts.append(result.get('response', 'No response'))
            output_parts.append("")

        # Stage 2: Rankings
        output_parts.append("-" * 60)
        output_parts.append("STAGE 2: Peer Rankings")
        output_parts.append("-" * 60)

        # Aggregate rankings
        if metadata.get('aggregate_rankings'):
            output_parts.append("\n**Aggregate Rankings (by average position):**")
            for i, rank in enumerate(metadata['aggregate_rankings'], 1):
                output_parts.append(f"  {i}. {rank.get('display_name', rank['model'])} (avg: {rank['average_rank']:.2f})")
            output_parts.append("")

        # Individual evaluations (abbreviated)
        for result in stage2_results:
            output_parts.append(f"\n### {result.get('display_name', result['model'])}'s Evaluation")
            parsed = result.get('parsed_ranking', [])
            if parsed:
                output_parts.append(f"Ranking: {' > '.join(parsed)}")
            output_parts.append("")

        # Stage 3: Final Synthesis
        output_parts.append("-" * 60)
        output_parts.append("STAGE 3: Chairman's Synthesis")
        output_parts.append("-" * 60)
        output_parts.append(f"\n### {stage3_result.get('display_name', stage3_result['model'])}")
        output_parts.append(stage3_result.get('response', 'No synthesis available'))

        # Save conversation if requested
        if save:
            import uuid
            conv_id = str(uuid.uuid4())
            create_conversation(conv_id)
            add_user_message(conv_id, question)
            add_assistant_message(conv_id, stage1_results, stage2_results, stage3_result)

            # Generate and save title
            title = await generate_conversation_title(question)
            update_conversation_title(conv_id, title)

            output_parts.append(f"\n\n[Conversation saved: {conv_id}]")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error during deliberation: {str(e)}")]


async def handle_quick_query(args: Dict[str, Any]) -> List[TextContent]:
    """Query a single provider."""
    provider = args.get("provider", "")
    prompt = args.get("prompt", "")
    timeout = args.get("timeout", 120)

    if not provider or not prompt:
        return [TextContent(type="text", text="Error: Provider and prompt are required")]

    if provider not in PROVIDERS:
        return [TextContent(type="text", text=f"Error: Unknown provider '{provider}'. Available: {', '.join(PROVIDERS.keys())}")]

    try:
        result = await query_cli_provider(provider, prompt, timeout)

        if result is None:
            return [TextContent(type="text", text=f"Error: {provider} failed to respond")]

        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider)
        output = f"**{display_name}**\n\n{result.get('content', 'No content')}"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error querying {provider}: {str(e)}")]


async def handle_get_providers(args: Dict[str, Any]) -> List[TextContent]:
    """Get provider information."""
    output_parts = []
    output_parts.append("**LLM Council Providers**\n")
    output_parts.append(f"Mode: {PROVIDER_MODE}")
    output_parts.append(f"Council Members: {len(COUNCIL_MODELS)}")
    output_parts.append(f"Chairman: {PROVIDER_DISPLAY_NAMES.get(CHAIRMAN_MODEL, CHAIRMAN_MODEL)}\n")

    output_parts.append("**Available Providers:**")
    for name in get_available_providers():
        info = get_provider_info(name)
        if info:
            is_council = "Council Member" if name in COUNCIL_MODELS else ""
            is_chairman = " Chairman" if name == CHAIRMAN_MODEL else ""
            role = f" [{is_council}{is_chairman}]" if is_council or is_chairman else ""
            output_parts.append(f"  - {info['display_name']}{role}")
            output_parts.append(f"    Timeout: {info['timeout']}s")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_get_conversations(args: Dict[str, Any]) -> List[TextContent]:
    """List past conversations."""
    limit = args.get("limit", 20)

    try:
        conversations = list_conversations()[:limit]

        if not conversations:
            return [TextContent(type="text", text="No conversations found.")]

        output_parts = []
        output_parts.append(f"**Past Deliberations** ({len(conversations)} shown)\n")

        for conv in conversations:
            output_parts.append(f"- **{conv.get('title', 'Untitled')}**")
            output_parts.append(f"  ID: {conv['id']}")
            output_parts.append(f"  Created: {conv['created_at']}")
            output_parts.append(f"  Messages: {conv['message_count']}")
            output_parts.append("")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing conversations: {str(e)}")]


async def handle_get_conversation(args: Dict[str, Any]) -> List[TextContent]:
    """Get specific conversation details."""
    conv_id = args.get("conversation_id", "")

    if not conv_id:
        return [TextContent(type="text", text="Error: conversation_id is required")]

    try:
        conv = get_conversation(conv_id)

        if conv is None:
            return [TextContent(type="text", text=f"Conversation not found: {conv_id}")]

        output_parts = []
        output_parts.append(f"**{conv.get('title', 'Untitled')}**")
        output_parts.append(f"ID: {conv['id']}")
        output_parts.append(f"Created: {conv['created_at']}")
        output_parts.append(f"Messages: {len(conv['messages'])}\n")

        for i, msg in enumerate(conv['messages']):
            if msg['role'] == 'user':
                output_parts.append(f"**User:** {msg['content']}\n")
            else:
                output_parts.append("**Council Response:**")
                if msg.get('stage3'):
                    stage3 = msg['stage3']
                    output_parts.append(f"Final Answer ({stage3.get('display_name', stage3.get('model', 'Unknown'))}):")
                    output_parts.append(stage3.get('response', 'No response'))
                output_parts.append("")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting conversation: {str(e)}")]


async def handle_compare_providers(args: Dict[str, Any]) -> List[TextContent]:
    """Compare all providers on the same prompt."""
    prompt = args.get("prompt", "")

    if not prompt:
        return [TextContent(type="text", text="Error: Prompt is required")]

    try:
        # Use stage1 which queries all providers in parallel
        results = await stage1_collect_responses(prompt)

        output_parts = []
        output_parts.append("**Provider Comparison**\n")
        output_parts.append(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")

        for result in results:
            output_parts.append(f"### {result.get('display_name', result['model'])}")
            output_parts.append(result.get('response', 'No response'))
            output_parts.append("")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error comparing providers: {str(e)}")]


async def handle_list_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """List all available deliberation patterns."""
    output_parts = []
    output_parts.append("**Available Deliberation Patterns**\n")
    output_parts.append("Each pattern represents a different multi-mind collaboration strategy:\n")

    patterns = list_patterns()  # Returns list of dicts with pattern info
    for info in patterns:
        output_parts.append(f"### {info['name']}")
        output_parts.append(f"**Pattern ID:** `{info['id']}`")
        output_parts.append(f"**Description:** {info['description']}")
        output_parts.append(f"**Best For:** {', '.join(info['best_for'])}")
        output_parts.append(f"**Flow:** {info['flow']}")
        output_parts.append("")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_run_pattern(args: Dict[str, Any]) -> List[TextContent]:
    """Run a specific deliberation pattern."""
    pattern_name = args.get("pattern", "")
    question = args.get("question", "")
    rounds = args.get("rounds", 2)
    branches = args.get("branches", 3)

    if not pattern_name or not question:
        return [TextContent(type="text", text="Error: Pattern and question are required")]

    try:
        # Convert pattern name to enum
        try:
            pattern = Pattern(pattern_name)
        except ValueError:
            available = ", ".join([p.value for p in Pattern])
            return [TextContent(type="text", text=f"Error: Unknown pattern '{pattern_name}'. Available: {available}")]

        # Get pattern info from the list
        all_patterns = list_patterns()
        info = next((p for p in all_patterns if p['id'] == pattern_name), None)
        pattern_display = info['name'] if info else pattern_name

        # Run the pattern
        result = await run_pattern(pattern, question, rounds=rounds, branches=branches)

        # Format output
        output_parts = []
        output_parts.append("=" * 60)
        output_parts.append(f"PATTERN: {pattern_display}")
        output_parts.append("=" * 60)
        output_parts.append(f"\n**Question:** {question}\n")

        if info:
            output_parts.append(f"**Strategy:** {info['description']}\n")

        # Display stages
        stages = result.get("stages", [])
        for i, stage in enumerate(stages, 1):
            output_parts.append("-" * 60)
            stage_name = stage.get("name", f"Stage {i}")
            output_parts.append(f"**{stage_name}**")
            output_parts.append("-" * 60)

            # Handle different stage content types
            if "responses" in stage:
                for resp in stage["responses"]:
                    role = resp.get("role", resp.get("model", "Unknown"))
                    content = resp.get("response", resp.get("content", "No response"))
                    output_parts.append(f"\n### {role}")
                    output_parts.append(content)
                    output_parts.append("")
            elif "response" in stage:
                output_parts.append(stage["response"])
                output_parts.append("")
            elif "branches" in stage:
                for j, branch in enumerate(stage["branches"], 1):
                    output_parts.append(f"\n**Branch {j}:**")
                    output_parts.append(branch.get("response", branch.get("content", "No content")))
                    output_parts.append("")
            elif "content" in stage:
                output_parts.append(stage["content"])
                output_parts.append("")

        # Final answer
        if result.get("final_answer"):
            output_parts.append("-" * 60)
            output_parts.append("**FINAL ANSWER**")
            output_parts.append("-" * 60)
            output_parts.append(result["final_answer"])

        # Metadata
        if result.get("metadata"):
            meta = result["metadata"]
            output_parts.append("\n---")
            if meta.get("consensus_rate"):
                output_parts.append(f"*Consensus Rate: {meta['consensus_rate']:.0%}*")
            if meta.get("winning_branch"):
                output_parts.append(f"*Winning Branch: {meta['winning_branch']}*")
            if meta.get("rounds_completed"):
                output_parts.append(f"*Rounds Completed: {meta['rounds_completed']}*")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error running pattern: {str(e)}\n{traceback.format_exc()}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
