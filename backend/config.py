"""Configuration for the LLM Council - CLI Provider Edition.

This version uses local CLI tools (Claude Code, Gemini) and Ollama cloud models
instead of OpenRouter.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Provider mode: "cli" for local CLIs, "openrouter" for API
PROVIDER_MODE = os.getenv("PROVIDER_MODE", "cli")

# OpenRouter API key (only used if PROVIDER_MODE == "openrouter")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# CLI Council Configuration
# Claude Code + Gemini CLI + Ollama cloud (gpt-oss via ollama.com)
CLI_COUNCIL_MODELS = [
    "claude",        # Claude Code CLI (Anthropic)
    "ollama",        # Ollama Cloud - gpt-oss:120b-cloud (OpenAI-compatible)
    "gemini",        # Gemini CLI (Google)
    "llama_server",  # llama-server (local llama.cpp, qwen2.5-coder-14b)
]

# CLI Chairman model - synthesizes final response
CLI_CHAIRMAN_MODEL = "gemini"  # Gemini as chairman

# OpenRouter Council Configuration (legacy/alternative)
OPENROUTER_COUNCIL_MODELS = [
    "openai/gpt-4.1",
    "google/gemini-2.5-pro-preview",
    "anthropic/claude-sonnet-4",
    "x-ai/grok-3",
]

OPENROUTER_CHAIRMAN_MODEL = "anthropic/claude-sonnet-4"

# Active configuration based on mode
if PROVIDER_MODE == "cli":
    COUNCIL_MODELS = CLI_COUNCIL_MODELS
    CHAIRMAN_MODEL = CLI_CHAIRMAN_MODEL
else:
    COUNCIL_MODELS = OPENROUTER_COUNCIL_MODELS
    CHAIRMAN_MODEL = OPENROUTER_CHAIRMAN_MODEL

# OpenRouter API endpoint (only used if PROVIDER_MODE == "openrouter")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Display names for CLI providers
PROVIDER_DISPLAY_NAMES = {
    "claude": "Claude Code (Anthropic)",
    "ollama": "GPT-OSS 120B (Ollama Cloud)",
    "gemini": "Gemini CLI (Google)",
    "llama_server": "llama-server (Local)",
}

# Timeouts for each provider (CLI tools can be slower)
PROVIDER_TIMEOUTS = {
    "claude": 180.0,
    "ollama": 120.0,
    "gemini": 180.0,
    "llama_server": 300.0,
}
