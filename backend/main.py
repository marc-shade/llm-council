"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, PROVIDER_DISPLAY_NAMES
from .cli_providers import query_cli_provider, query_providers_parallel, get_available_providers, get_provider_info
from .patterns import Pattern, list_patterns, run_pattern

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# MCP-Friendly Endpoints (used by thin MCP server)
# ============================================================================

class QuickQueryRequest(BaseModel):
    """Request for quick single-provider query."""
    provider: str
    prompt: str
    timeout: float = 60.0


class RunPatternRequest(BaseModel):
    """Request to run a deliberation pattern."""
    pattern: str
    question: str
    rounds: int = 2
    branches: int = 3


class CompareRequest(BaseModel):
    """Request to compare all providers."""
    prompt: str


@app.get("/api/mcp/health")
async def mcp_health():
    """Health check for MCP integration."""
    return {"status": "healthy", "service": "llm-council", "mcp_ready": True}


@app.get("/api/mcp/providers")
async def get_providers():
    """Get available LLM providers for MCP."""
    providers = []
    for name in get_available_providers():
        info = get_provider_info(name)
        if info:
            providers.append({
                "name": name,
                "display_name": info["display_name"],
                "is_chairman": name == CHAIRMAN_MODEL,
                "timeout": info["timeout"]
            })

    return {
        "mode": "cli",
        "council_members": len(COUNCIL_MODELS),
        "chairman": PROVIDER_DISPLAY_NAMES.get(CHAIRMAN_MODEL, CHAIRMAN_MODEL),
        "providers": providers
    }


@app.post("/api/mcp/query")
async def quick_query(request: QuickQueryRequest):
    """Query a single provider (for MCP)."""
    result = await query_cli_provider(
        request.provider,
        request.prompt,
        timeout=request.timeout
    )

    if result and result.get("content"):
        return {"success": True, "content": result["content"]}
    else:
        return {"success": False, "error": f"{request.provider} failed to respond"}


@app.get("/api/mcp/patterns")
async def get_patterns():
    """List available deliberation patterns."""
    return {"patterns": list_patterns()}


@app.post("/api/mcp/patterns/run")
async def run_pattern_endpoint(request: RunPatternRequest):
    """Run a specific deliberation pattern."""
    try:
        pattern = Pattern(request.pattern)
    except ValueError:
        available = [p.value for p in Pattern]
        raise HTTPException(400, f"Unknown pattern '{request.pattern}'. Available: {available}")

    try:
        result = await run_pattern(
            pattern,
            request.question,
            rounds=request.rounds,
            branches=request.branches
        )
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/mcp/compare")
async def compare_providers(request: CompareRequest):
    """Compare all providers on the same prompt."""
    results = await query_providers_parallel(COUNCIL_MODELS, request.prompt)

    responses = []
    for model, response in results.items():
        display_name = PROVIDER_DISPLAY_NAMES.get(model, model)
        if response and response.get("content"):
            responses.append({
                "provider": model,
                "display_name": display_name,
                "content": response["content"],
                "success": True
            })
        else:
            responses.append({
                "provider": model,
                "display_name": display_name,
                "content": None,
                "success": False
            })

    return {"responses": responses}


@app.post("/api/mcp/deliberate")
async def mcp_deliberate(request: SendMessageRequest):
    """Run full council deliberation (for MCP, no conversation storage)."""
    try:
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content
        )
        return {
            "success": True,
            "result": {
                "stage1": stage1_results,
                "stage2": stage2_results,
                "stage3": stage3_result,
                "metadata": metadata
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
